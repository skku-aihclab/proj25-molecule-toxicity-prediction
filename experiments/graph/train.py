import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import optuna
import time

from sklearn.metrics       import roc_auc_score
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import ConcatDataset
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.dataset import GraphDataset 
from models.models     import GraphEncoder, GraphClassifier

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2) Load data & split ─────────────────────────────────────────────────────
train_csv = os.path.join(ROOT, "data", "train.csv")
valid_csv = os.path.join(ROOT, "data", "valid.csv")
label_list = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

train_ds = GraphDataset(train_csv, label_list)
valid_ds = GraphDataset(valid_csv, label_list)
full_ds = ConcatDataset([train_ds, valid_ds])

# ─── 3) Optuna objective ─────────────────────────────────────────────────────
start_search_time = time.time() # Record the start time for hyperparameter search
def objective(trial):
    # Hyperparameters to tune
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 756])
    num_layers = trial.suggest_int("num_layers", 3, 5)
    emb_dim    = trial.suggest_categorical("emb_dim", [256, 512])
    lr         = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3])
    wd         = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])

    # create the model
    encoder = GraphEncoder(
        in_dim=78,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        emb_dim=emb_dim
    )
    model = GraphClassifier(encoder, num_tasks=len(label_list)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = GeoDataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = GeoDataLoader(valid_ds, batch_size=64, shuffle=False, drop_last=True)

    best_val_auc, best_epoch = 0.0, 0
    patience = 10

    for epoch in range(1, 51):
        # --- train ---
        model.train()
        for data in train_loader:
            data   = data.to(device)
            logits = model(data)  # forward pass
            labels = data.y.to(device)     

            mask    = (labels >= 0).float()  # create mask for valid entries
            targets = labels.clamp(min=0)  

            per_ent = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ) # compute loss
            loss = (per_ent * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Epoch {epoch: 2d}")

        # --- validate ---
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for data in valid_loader:
                data   = data.to(device)
                probs  = torch.sigmoid(model(data))  
                all_p.append(probs.cpu())
                all_y.append(data.y.cpu())   

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
            trial.set_user_attr("best_encoder_state", encoder.state_dict())
            trial.set_user_attr("best_epoch", epoch)
        elif epoch - best_epoch >= patience:
            break

    return best_val_auc

# ─── 4) Run Optuna ───────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50)

end_search_time = time.time() # Record the end time for hyperparameter search
print(f"Total Searching completed in {end_search_time - start_search_time:.2f} seconds")

# ─── 5) Save encoder ─────────────────────────────────────────────────────────────
best_trial  = study.best_trial
best_params = best_trial.params
best_params["best_epoch"] = best_trial.user_attrs["best_epoch"]

with open(os.path.join(ROOT, "checkpoints", "parameters", "graph_best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

best_encoder_state = best_trial.user_attrs["best_encoder_state"]
torch.save(best_encoder_state, os.path.join(ROOT, "checkpoints", "encoder", "train_only", "graph_encoder.pth"))

print("Best AUC:", study.best_value, "at epoch", best_trial.user_attrs["best_epoch"])

# ─── 6) Retrain on full dataset ──────────────────────────────────────────────────────────────
train_loader = GeoDataLoader(full_ds, batch_size=32, shuffle=True, drop_last=True)

# load the best parameters and create the model
emb_dim     = best_params["emb_dim"]
lr          = best_params["lr"]
wd          = best_params["weight_decay"]
num_layers  = best_params["num_layers"]
hidden_dim  = best_params["hidden_dim"]
epoch      = best_params["best_epoch"]

encoder = GraphEncoder(
    in_dim     = 78,
    hidden_dim = hidden_dim,
    num_layers = num_layers,
    emb_dim    = emb_dim,
).to(device)
model = GraphClassifier(encoder, num_tasks=len(label_list)).to(device)
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wd
)

# train the model on the full dataset (train + valid)
start_train_time = time.time()  # Record the start time for train performance 
for epoch in range(1, epoch + 1): 
    model.train()
    for data in train_loader:
        data = data.to(device)
        logits = model(data)         
        labels = data.y.to(device)     

        mask    = (labels >= 0).float()  # create mask for valid entries
        targets = labels.clamp(min=0)  

        per_ent = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) # compute loss
        loss = (per_ent * mask).sum() / mask.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch: 2d}")

end_train_time = time.time() # Record the end time for train performance 
print(f"Total training time: {end_train_time - start_train_time:.2f} seconds")

torch.save(
    encoder.state_dict(), 
    os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "graph_encoder.pth")
) # Save the final encoder state
torch.save(
    model.state_dict(), 
    os.path.join(ROOT, "checkpoints", "model", "graph.pth")
) # Save the final model state