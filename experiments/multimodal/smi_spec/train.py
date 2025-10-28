import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import json
import optuna
import time
from torch.utils.data import ConcatDataset
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models import SMILESEncoder, SpectrumEncoder, MultiModalSMISpec
from utils.dataset import SMILESDataset, SpectrumDataset, SMILESSpectrumDataset

# Add the root for CReSS models to sys.path
CRESS_ROOT = os.path.join(ROOT, "experiments", "spectrum")
if CRESS_ROOT not in sys.path:
    sys.path.insert(0, CRESS_ROOT)
from experiments.spectrum.CReSS.infer import ModelInference

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Load data & split ─────────────────────────────────────────────────────
train_csv = os.path.join(ROOT, "data", "train_spectra.csv")
valid_csv = os.path.join(ROOT, "data", "valid_spectra.csv")
spectra_dir = os.path.join(ROOT, "data", "spectra")
labels = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
          "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
          "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

s_tr = SMILESDataset(
    csv_path=train_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
s_va = SMILESDataset(
    csv_path=valid_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
sp_tr = SpectrumDataset(train_csv, labels, spectra_dir)
sp_va = SpectrumDataset(valid_csv, labels, spectra_dir)

s_full = ConcatDataset([s_tr, s_va])
sp_full = ConcatDataset([sp_tr, sp_va])

def collate(batch):
    """
    Custom collate function to handle batching of SMILES and spectrum data.
    Args:
        batch: List of tuples (transformer IDs, transformer mask, ppm_list, labels).
    Returns:
        Tuple of batched transformer IDs, transformer mask, list of ppm values, and labels.
    """
    ids, mask, ppm_lists, y = zip(*batch)
    t_ids = torch.stack(ids)
    t_mask = torch.stack(mask)
    ppm_batch = list(ppm_lists)
    y_batch = torch.stack(y)
    return t_ids, t_mask, ppm_batch, y_batch

# ─── 3) encoder load & freeze ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "smiles_best_params.json")) as f:
    smiles_best_params = json.load(f)
smiles_encoder = SMILESEncoder(
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    emb_dim=smiles_best_params["emb_dim"],
    max_length=202
); smiles_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "smiles_encoder.pth"), map_location=torch.device('cpu')))

# load CReSS model for 1D CNN
config_path     = os.path.join(ROOT, "experiments", "spectrum", "8.json")
pretrain_path   = os.path.join(ROOT, "experiments", "spectrum", "8.pth")
model_inference = ModelInference(
    config_path=config_path,
    pretrain_model_path=pretrain_path,
    device=device
)

with open(os.path.join(ROOT, "checkpoints", "parameters", "spectrum_best_params.json")) as f:
    spectrum_best_params = json.load(f)
spectrum_encoder = SpectrumEncoder(
    model_inference,
    hidden_dim=spectrum_best_params["hidden_dim"],
    emb_dim=spectrum_best_params["emb_dim"]
); spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "spectrum_encoder.pth"), map_location=torch.device('cpu')))

for net in (smiles_encoder, spectrum_encoder):
    for p in net.parameters():
        p.requires_grad = False # Freeze parameters

# ─── 4) Optuna objective ────────────────────────────────────────────────
start_search_time = time.time() # Record the start time for hyperparameter search
def objective(trial):
    # Hyperparameters to tune
    emb_dim = trial.suggest_categorical("emb_dim", [128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [512, 768])
    num_layers = trial.suggest_categorical("num_layers", [1])
    lr    =    trial.suggest_categorical("lr", [1e-4, 3e-4])
    wd    =    trial.suggest_categorical("weight_decay", [1e-5, 1e-4])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5])

    # create the model
    model = MultiModalSMISpec(
        smiles_encoder,
        spectrum_encoder,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_tasks=len(labels)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=wd)

    train_loader = DataLoader(
        SMILESSpectrumDataset(s_tr, sp_tr),
        batch_size=32,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate,
        drop_last=True
    )
    valid_loader   = DataLoader(
        SMILESSpectrumDataset(s_va, sp_va),
        batch_size=64,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate,
        drop_last=True
    )
    best_val_auc, best_epoch = 0.0, 0
    patience = 5

    for epoch in range(1,31):
        # --- train ---
        model.train()
        for t_ids, t_mask, ppm_list, y in train_loader:
            t_ids, t_mask, y = t_ids.to(device), t_mask.to(device), y.to(device)
            logits = model(t_ids, t_mask, ppm_list)

            mask_loss = (y >= 0).float()
            targets   = y.clamp(min=0)
            per_ent   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (per_ent * mask_loss).sum() / mask_loss.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()
        # print(f"Epoch {epoch: 2d}")

        # --- validate ---
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for t_ids, t_mask, ppm_list, y in valid_loader:
                t_ids, t_mask = t_ids.to(device), t_mask.to(device)
                probs = torch.sigmoid(model(t_ids, t_mask, ppm_list))
                all_p.append(probs.cpu())
                all_y.append(y)

        y_true = torch.cat(all_y).numpy()
        y_pred = torch.cat(all_p).numpy()

        aucs = []
        for i in range(y_true.shape[1]):
            valid = (y_true[:, i] >= 0)
            vals  = y_true[valid, i]
            preds = y_pred[valid, i]

            # Only compute AUC if both classes are present
            if len(np.unique(vals)) == 2:
                aucs.append(roc_auc_score(vals, preds))

        # Average AUC
        val_auc = float(np.mean(aucs)) if aucs else 0.0

        # --- early stopping ---
        if val_auc > best_val_auc:
            best_val_auc, best_epoch = val_auc, epoch
            trial.set_user_attr("best_model_state", model.state_dict())
            trial.set_user_attr("best_epoch", epoch)
        elif epoch - best_epoch >= patience:
            break

    return best_val_auc

# ─── 5) Run Optuna ───────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=10)

end_search_time = time.time() # Record the end time for hyperparameter search
print(f"Total Searching completed in {end_search_time - start_search_time:.2f} seconds")

# ─── 6) Save hyperparameters ────────────────────────────────────────────────
best_trial = study.best_trial
best_params = best_trial.params
best_params["best_epoch"] = best_trial.user_attrs["best_epoch"]

with open(os.path.join(ROOT, "checkpoints", "parameters", "smi_spec_best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("Best AUC: ", study.best_value, "at epoch", best_trial.user_attrs["best_epoch"])

# ─── 7) Retrain on full dataset ──────────────────────────────────────────────────────────────
train_loader = DataLoader(
    SMILESSpectrumDataset(s_full, sp_full),
    batch_size=32,
    shuffle=True,
    pin_memory=False,
    collate_fn=collate,
    drop_last=True
)

# load encoders trained with train and valid datasets
smiles_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "smiles_encoder.pth"), map_location=torch.device('cpu')))
spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "spectrum_encoder.pth"), map_location=torch.device('cpu')))

# Freeze parameters of the encoders
for net in (smiles_encoder, spectrum_encoder):
    for p in net.parameters():
        p.requires_grad = False

# load the best hyperparameters and create the model
emb_dim = best_params["emb_dim"]
hidden_dim = best_params["hidden_dim"]
num_layers = best_params["num_layers"]
dropout = best_params["dropout"]
lr = best_params["lr"]
wd = best_params["weight_decay"]
epoch = best_params["best_epoch"]

model = MultiModalSMISpec(
    smiles_encoder,
    spectrum_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wd
)

# train the model on the full dataset (train + valid)
start_train_time = time.time()  # Record the start time for train performance
for epoch in range(1, epoch + 1):
    model.train()
    for t_ids, t_mask, ppm_list, y in train_loader:
        t_ids, t_mask, y = t_ids.to(device), t_mask.to(device), y.to(device)
        logits = model(t_ids, t_mask, ppm_list)

        mask = (y >= 0).float()
        targets = y.clamp(min=0)

        per_ent = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (per_ent * mask).sum() / mask.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch: 2d}")

end_train_time = time.time() # Record the end time for train performance
print(f"Total training time: {end_train_time - start_train_time:.2f} seconds")

torch.save(
    model.state_dict(),
    os.path.join(ROOT, "checkpoints", "model", "smi_spec.pth")
)  # Save the final model state
