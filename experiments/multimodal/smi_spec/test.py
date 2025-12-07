import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import json
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models import SMILESEncoder, SpectrumEncoder, MultiModalSMISpec
from utils.dataset import SMILESDataset, SpectrumDataset, SMILESSpectrumDataset
from utils.attention_analysis import (
    compute_modality_contributions,
    compute_sample_wise_contributions,
    save_attention_analysis,
    print_attention_summary,
    compute_cross_modal_attention_matrix,
    plot_cross_modal_attention_heatmap,
    print_cross_modal_attention_matrix,
    save_cross_modal_attention_analysis
)

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

s_test = SMILESDataset(
    csv_path=test_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
sp_test = SpectrumDataset(test_csv, labels, spectra_dir, allow_missing=True)

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

test_loader = DataLoader(
    SMILESSpectrumDataset(s_test, sp_test),
    batch_size=64,
    shuffle=False,
    collate_fn=collate
)

# ─── 3) encoder load & freeze ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "smiles_best_params.json")) as f:
    smiles_best_params = json.load(f)
smiles_encoder = SMILESEncoder(
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    emb_dim=smiles_best_params["emb_dim"],
    max_length=202
); smiles_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "smiles_encoder.pth"), map_location=torch.device('cpu')))

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
); spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "spectrum_encoder.pth"), map_location=torch.device('cpu')), strict=False)

# Freeze parameters of the encoders
for net in (smiles_encoder, spectrum_encoder):
    for p in net.parameters():
        p.requires_grad = False

# ─── 4) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "smi_spec_best_params.json")) as f:
    smi_spec_best_params = json.load(f)

emb_dim = smi_spec_best_params["emb_dim"]
hidden_dim = smi_spec_best_params["hidden_dim"]
num_layers = smi_spec_best_params["num_layers"]
dropout = smi_spec_best_params["dropout"]
epoch = smi_spec_best_params["best_epoch"]

# ─── 5) Create the model ──────────────────────────────
model = MultiModalSMISpec(
    smiles_encoder,
    spectrum_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "smi_spec.pth"), map_location=device))

# ─── 6) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p, all_attn = [], [], []
with torch.no_grad():
    for t_ids, t_mask, ppm_list, y in test_loader:
        t_ids, t_mask = t_ids.to(device), t_mask.to(device)
        logits, attn_weights = model(t_ids, t_mask, ppm_list, return_attention=True)
        probs = torch.sigmoid(logits)
        all_p.append(probs.cpu())
        all_attn.append(attn_weights.cpu())
        all_y.append(y)

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

# ─── 7) Attention weight analysis ─────────────────────────────────────
modality_names = ['SMILES', 'Spectrum']

# Compute overall modality contributions
all_attn_stacked = torch.cat(all_attn, dim=0)
overall_contributions = compute_modality_contributions(all_attn_stacked, modality_names)

# Compute sample-wise statistics
mean_contributions, std_contributions = compute_sample_wise_contributions(all_attn, modality_names)
contribution_stats = {
    name: {'mean': float(mean_contributions[i]), 'std': float(std_contributions[i])}
    for i, name in enumerate(modality_names)
}

# Print summary
print_attention_summary(overall_contributions, contribution_stats, modality_names)

# Save results
attention_save_dir = os.path.join(ROOT, "checkpoints", "attention_analysis")

# ─── 8) Cross-modal attention matrix analysis ─────────────────────────────────────
# Compute cross-modal attention matrix
mean_matrix, std_matrix = compute_cross_modal_attention_matrix(
    all_attn_stacked,
    modality_names
)

# Print cross-modal attention matrix to console
print_cross_modal_attention_matrix(mean_matrix, std_matrix, modality_names)

# Create and save heatmap (saves to attention_save_dir/image/)
plot_cross_modal_attention_heatmap(
    mean_matrix=mean_matrix,
    std_matrix=std_matrix,
    modality_names=modality_names,
    save_dir=attention_save_dir,
    model_name="smi_spec",
    title='SMILES-Spectrum Cross-Modal Attention',
    figsize=(10, 8)
)

# Save cross-modal attention analysis to JSON (saves to attention_save_dir/json/)
save_cross_modal_attention_analysis(
    mean_matrix=mean_matrix,
    std_matrix=std_matrix,
    modality_names=modality_names,
    save_dir=attention_save_dir,
    model_name="smi_spec"
)
