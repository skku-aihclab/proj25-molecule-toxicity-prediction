import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.dataset import SMILESDataset
from models.models import SMILESEncoder, SMILESClassifier

# ─── 1) Define your device ────────────────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2) Load data & split ───────────────────────────────────────────────────── 
test_csv   = os.path.join(ROOT, "data", "test_spectra.csv")
label_list = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

test_ds = SMILESDataset(
    csv_path=test_csv,
    label_name=label_list,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
test_loader = DataLoader(
    test_ds,
    batch_size= 64,
    shuffle=True,
    pin_memory=True
)

# ─── 3) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "smiles_best_params.json")) as f:
    best_params = json.load(f)

emb_dim = best_params["emb_dim"]

# ─── 4) Create the model ──────────────────────────────
encoder = SMILESEncoder(
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    emb_dim=emb_dim,
    dropout=0.1,
    max_length=202
).to(device)
encoder.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "smiles_encoder.pth"), map_location=device))
model = SMILESClassifier(
    encoder=encoder,
    num_tasks=len(label_list)
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "smiles.pth"), map_location=device))

# ─── 5) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for input_ids, attn_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        probs     = torch.sigmoid(model(input_ids, attn_mask))
        all_p.append(probs.cpu())
        all_y.append(labels)
        
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