import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
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

from models.models        import GraphEncoder, SpectrumEncoder, MultiModalGphSpec
from utils.dataset     import GraphDataset, SpectrumDataset, GraphSpectrumDataset

# Add the root for CReSS models to sys.path
CRESS_ROOT = os.path.join(ROOT, "experiments", "spectrum")
if CRESS_ROOT not in sys.path:
    sys.path.insert(0, CRESS_ROOT)
from experiments.spectrum.CReSS.infer import ModelInference

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Load data & split ─────────────────────────────────────────────────────
train_csv = os.path.join(ROOT, "data", "train.csv")
valid_csv = os.path.join(ROOT, "data", "valid.csv") 
spectra_dir = os.path.join(ROOT, "data", "spectra")
labels = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
          "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
          "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

g_tr  = GraphDataset(train_csv, labels)
g_va  = GraphDataset(valid_csv, labels)
sp_tr = SpectrumDataset(train_csv, labels, spectra_dir, allow_missing=True)
sp_va = SpectrumDataset(valid_csv, labels, spectra_dir, allow_missing=True)
g_full = ConcatDataset([g_tr, g_va])
sp_full = ConcatDataset([sp_tr, sp_va])
    
def collate(batch):
    """
    Custom collate function to handle batching of graph and spectrum data.
    Args:
        batch: List of tuples (graph, ppm_list, labels).
    Returns:
        Tuple of batched graph data, list of ppm values, and labels.
    """
    graphs, ppm_lists, y = zip(*batch)
    g_batch = Batch.from_data_list(graphs)
    ppm_batch = list(ppm_lists)
    y_batch = torch.stack(y)
    return g_batch, ppm_batch, y_batch

# ─── 3) encoder load & freeze ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "graph_best_params.json")) as f:
    graph_best_params = json.load(f)
graph_encoder = GraphEncoder(
    in_dim = 78,
    hidden_dim = graph_best_params["hidden_dim"],
    num_layers = graph_best_params["num_layers"],
    emb_dim = graph_best_params["emb_dim"]
); graph_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "graph_encoder.pth"), map_location=torch.device('cpu')))

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
    emb_dim = spectrum_best_params["emb_dim"],
); spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "spectrum_encoder.pth"), map_location=torch.device('cpu')), strict=False)

for net in (graph_encoder, spectrum_encoder):
    for p in net.parameters():
        p.requires_grad = False # Freeze parameters

spectrum_encoder.missing_token.requires_grad = True  # Unfreeze missing token embedding for spectrum encoder

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
    model = MultiModalGphSpec(
        graph_encoder,
        spectrum_encoder,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_tasks=len(labels)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    train_loader = DataLoader(
        GraphSpectrumDataset(g_tr, sp_tr),
        batch_size=32,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate,
        drop_last=True
    )
    valid_loader = DataLoader(
        GraphSpectrumDataset(g_va, sp_va),
        batch_size=64,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate,
        drop_last=True
    )
    best_val_auc, best_epoch = 0.0, 0
    patience = 5
    for epoch in range(1, 31):
        # --- train ---
        model.train()
        for g, ppm, y in train_loader:
            g, y = g.to(device), y.to(device) 
            logits = model(g, ppm) 

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
            for g, ppm, y in valid_loader:
                g, y = g.to(device), y.to(device)
                probs = torch.sigmoid(model(g, ppm))
                all_p.append(probs.cpu())
                all_y.append(y.cpu())

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

# ─── Run Optuna ───────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=20)

end_search_time = time.time() # Record the end time for hyperparameter search
print(f"Total Searching completed in {end_search_time - start_search_time:.2f} seconds")

# ─── 6) Save hyperparameters ────────────────────────────────────────────────
best_trial = study.best_trial
best_params = best_trial.params
best_params["best_epoch"] = best_trial.user_attrs["best_epoch"]

with open(os.path.join(ROOT, "checkpoints", "parameters", "gph_spec_best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("Best AUC: ", study.best_value, "at epoch", best_trial.user_attrs["best_epoch"])

# ─── 7) Retrain on full dataset ──────────────────────────────────────────────────────────────
train_loader = DataLoader(
    GraphSpectrumDataset(g_full, sp_full),
    batch_size=32,
    shuffle=True,
    pin_memory=False,
    collate_fn=collate,
    drop_last=True
)

# load encoders trained with train and valid datasets
graph_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "graph_encoder.pth"), map_location=torch.device('cpu')))
spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "spectrum_encoder.pth"), map_location=torch.device('cpu')), strict=False)

# Freeze parameters of the encoders
for net in (graph_encoder, spectrum_encoder):
    for p in net.parameters():
        p.requires_grad = False

spectrum_encoder.missing_token.requires_grad = True  # Unfreeze missing token embedding for spectrum encoder

# load the best hyperparameters and create the model
emb_dim = best_params["emb_dim"]
hidden_dim = best_params["hidden_dim"]
num_layers = best_params["num_layers"]
dropout = best_params["dropout"]
lr = best_params["lr"]
weight_decay = best_params["weight_decay"]
epoch      = best_params["best_epoch"]

model = MultiModalGphSpec(
    graph_encoder,
    spectrum_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
opt = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

# train the model on the full dataset (train + valid)
start_train_time = time.time()  # Record the start time for train performance
for epoch in range(1, epoch + 1):
    model.train()
    for g, ppm, y in train_loader:
        g, y = g.to(device), y.to(device)

        logits = model(g, ppm) 
        mask_loss = (y >= 0).float()              
        targets   = y.clamp(min=0)            
        per_ent   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")          
        loss = (per_ent * mask_loss).sum() / mask_loss.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch: 2d}")

end_train_time = time.time() # Record the end time for train performance
print(f"Total training time: {end_train_time - start_train_time:.2f} seconds")

torch.save(
    model.state_dict(),
    os.path.join(ROOT, "checkpoints", "model", "gph_spec.pth")
)