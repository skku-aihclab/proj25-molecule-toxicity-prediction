import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import time

from sklearn.metrics       import roc_auc_score
from torch.utils.data      import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from CReSS.infer import ModelInference
from utils.dataset import SpectrumDataset 
from models.models     import SpectrumEncoder, SpectrumClassifier

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2) Load & split your data ────────────────────────────────────────────────
test_csv   = os.path.join(ROOT, "data", "test_spectra.csv")
label_list = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]
spectra_dir = os.path.join(ROOT, "data", "spectra")

test_ds = SpectrumDataset(test_csv, label_list, spectra_dir)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# ─── 3) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "spectrum_best_params.json")) as f:
    best_params = json.load(f)
    
hidden_dim = best_params["hidden_dim"]
emb_dim = best_params["emb_dim"]

config_path     = os.path.join(ROOT, "experiments", "spectrum", "8.json")
pretrain_path   = os.path.join(ROOT, "experiments", "spectrum", "8.pth")
model_inference = ModelInference(
    config_path=config_path,
    pretrain_model_path=pretrain_path,
    device=device
)

# ─── 4) Create the model ──────────────────────────────
encoder = SpectrumEncoder(
    model_inference,
    hidden_dim=hidden_dim,
    emb_dim=emb_dim
).to(device)
encoder.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "spectrum_encoder.pth"), map_location=device))
model = SpectrumClassifier(encoder, num_tasks=len(label_list)).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "spectrum.pth"), map_location=device))

# ─── 5) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for spec, y_batch in test_loader:
        y_batch = y_batch.to(device)
        probs   = torch.sigmoid(model(spec))  
        all_p.append(probs.cpu())
        all_y.append(y_batch.cpu())
        
y_true = torch.cat(all_y).numpy() 
y_pred = torch.cat(all_p).numpy()

# calculate AUC for each label
aucs = {}
for i, lab in enumerate(label_list):
    mask = (y_true[:, i] >= 0)
    if mask.sum() > 0 and len(np.unique(y_true[mask, i])) == 2:
        aucs[lab] = roc_auc_score(y_true[mask, i], y_pred[mask, i])
    else:
        aucs[lab] = float("nan")

end_test_time = time.time() # Record the end time for test performance
print(f"Total testing time: {end_test_time - start_test_time:.2f} seconds")

# Print AUC results
for lab, auc in aucs.items():
    print(f"{lab:15s}: {auc:.4f}")
print("-" * 30)
print(f"Mean AUC        : {np.nanmean(list(aucs.values())):.4f}")
