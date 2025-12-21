import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
import json
import time
import torch.nn.functional as F
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models import GraphEncoder, SpectrumEncoder, MultiModalGphSpec
from utils.dataset import GraphDataset, SpectrumDataset, GraphSpectrumDataset

# Add the root for CReSS models to sys.path
CRESS_ROOT = os.path.join(ROOT, "experiments", "spectrum")
if CRESS_ROOT not in sys.path:
    sys.path.insert(0, CRESS_ROOT)
from experiments.spectrum.CReSS.infer import ModelInference

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Load data & split ─────────────────────────────────────────────────────
test_csv = os.path.join(ROOT, "data", "test.csv")
spectra_dir = os.path.join(ROOT, "data", "spectra")
labels = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
          "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
          "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

g_test = GraphDataset(test_csv, labels)
sp_test = SpectrumDataset(test_csv, labels, spectra_dir, allow_missing=True)
    
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

test_loader = DataLoader(
    GraphSpectrumDataset(g_test, sp_test),
    batch_size=64,
    shuffle=False,
    collate_fn=collate
)

# ─── 3) encoder load ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "graph_best_params.json")) as f:
    graph_best_params = json.load(f)
graph_encoder = GraphEncoder(
    in_dim = 78,
    hidden_dim = graph_best_params["hidden_dim"],
    num_layers = graph_best_params["num_layers"],
    emb_dim = graph_best_params["emb_dim"]
); graph_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "graph_encoder.pth"), map_location=torch.device('cpu')))

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
    emb_dim=spectrum_best_params["emb_dim"],
); spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "spectrum_encoder.pth"), map_location=torch.device('cpu')), strict=False)

# ─── 4) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "gph_spec_best_params.json")) as f:
    gph_spec_best_params = json.load(f)

emb_dim = gph_spec_best_params["emb_dim"]
hidden_dim = gph_spec_best_params["hidden_dim"]
num_layers = gph_spec_best_params["num_layers"]
dropout = gph_spec_best_params["dropout"]
epoch      = gph_spec_best_params["best_epoch"]

# ─── 5) Create the model ────────────────────────────────────────
model = MultiModalGphSpec(
    graph_encoder,
    spectrum_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "gph_spec.pth"), map_location=device))

# ─── 6) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for g, ppm, y in test_loader:
        g, y = g.to(device), y.to(device)
        logits = model(g, ppm)
        probs = torch.sigmoid(logits)
        all_y.append(y.cpu())
        all_p.append(probs.cpu())

y_true = torch.cat(all_y).numpy() 
y_pred = torch.cat(all_p).numpy()

# calculate AUC for each label
aucs = {}
for i, lab in enumerate(labels):
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
