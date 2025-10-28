import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torchvision import transforms
import json
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models import GraphEncoder, SMILESEncoder, ImageEncoder, SpectrumEncoder, MoltiTox
from utils.dataset import GraphDataset, SMILESDataset, ImageDataset, SpectrumDataset, MoltiToxDataset
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
test_csv = os.path.join(ROOT, "data", "test_spectra.csv")
ckpt_dir = os.path.join(ROOT, "experiments", "image", "ImageMol.pth")
img_dir = os.path.join(ROOT, "data", "images")
spectra_dir = os.path.join(ROOT, "data", "spectra")
labels = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
          "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
          "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

# Define transforms for image data
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

g_test = GraphDataset(test_csv, labels)
s_test = SMILESDataset(
    csv_path=test_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
i_test = ImageDataset(test_csv, labels, img_dir, transform=test_transform)
sp_test = SpectrumDataset(test_csv, labels, spectra_dir)

def collate(batch):
    """
    Custom collate function to handle batching of graph, smiles, image, and spectrum data.
    Args:
        batch: List of tuples (graph, transformer IDs, transformer mask, image, spectra, labels).
    Returns:
        Tuple of batched graph data, transformer IDs, transformer masks, image tensors, spectra data, and labels.
    """
    g, ids, mask, img, ppm_list, y = zip(*batch)
    g_batch = Batch.from_data_list(g)
    t_ids = torch.stack(ids)
    t_mask = torch.stack(mask)
    img_batch = torch.stack(img)
    ppm_batch = list(ppm_list)
    y_batch = torch.stack(y)
    return g_batch, t_ids, t_mask, img_batch, ppm_batch, y_batch

test_loader = DataLoader(
    MoltiToxDataset(g_test, s_test, i_test, sp_test),
    batch_size=64,
    shuffle=False,
    collate_fn=collate
)

# ─── 3) encoder load & freeze ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "graph_best_params.json")) as f:
    graph_best_params = json.load(f)
graph_encoder = GraphEncoder(
    in_dim = 78,
    hidden_dim = graph_best_params["hidden_dim"],
    num_layers = graph_best_params["num_layers"],
    emb_dim = graph_best_params["emb_dim"]
); graph_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "graph_encoder.pth"), map_location=torch.device('cpu')))

with open(os.path.join(ROOT, "checkpoints", "parameters", "smiles_best_params.json")) as f:
    smiles_best_params = json.load(f)
smiles_encoder = SMILESEncoder(
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    emb_dim=smiles_best_params["emb_dim"],
    max_length=202
); smiles_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "smiles_encoder.pth"), map_location=torch.device('cpu')))

with open(os.path.join(ROOT, "checkpoints", "parameters", "image_best_params.json")) as f:
    image_best_params = json.load(f)
image_encoder = ImageEncoder(
    ckpt_path=ckpt_dir,
    emb_dim = image_best_params["emb_dim"],
    hidden_size= image_best_params["hidden_size"]
); image_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth"), map_location=torch.device('cpu')))

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
); spectrum_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "spectrum_encoder.pth"), map_location=torch.device('cpu')))

# Freeze parameters of the backbone models
for net in (graph_encoder, image_encoder, smiles_encoder, spectrum_encoder):
    for p in net.parameters():
        p.requires_grad = False

# ─── 4) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "moltitox_best_params.json")) as f:
    moltitox_best_params = json.load(f)

emb_dim = moltitox_best_params["emb_dim"]
hidden_dim = moltitox_best_params["hidden_dim"]
num_layers = moltitox_best_params["num_layers"]
dropout = moltitox_best_params["dropout"]
epoch      = moltitox_best_params["best_epoch"]

# ─── 5) Create the model ──────────────────────────────
model = MoltiTox(
    graph_encoder,
    smiles_encoder,
    image_encoder,
    spectrum_encoder,
    emb_dim = emb_dim,
    hidden_dim= hidden_dim,
    num_layers= num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "moltitox.pth"), map_location=device))

# ─── 6) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p, all_attn = [], [], []
with torch.no_grad():
    for g, t_ids, t_mask, img, ppm, y in test_loader:
        g   = g.to(device)
        t_ids = t_ids.to(device)
        t_mask = t_mask.to(device)
        img = img.to(device)
        logits, attn_weights = model(g, t_ids, t_mask, img, ppm, return_attention=True)

        probs = torch.sigmoid(logits)
        all_p.append(probs.cpu())
        all_y.append(y)
        all_attn.append(attn_weights.cpu())

y_true = torch.cat(all_y).numpy()
y_pred = torch.cat(all_p).numpy()

# Calculate AUC for each label
aucs = {}
for i, lab in enumerate(labels):
    mask = (y_true[:, i] >= 0)
    if mask.sum() > 0 and len(np.unique(y_true[mask, i])) == 2:
        aucs[lab] = roc_auc_score(y_true[mask, i], y_pred[mask, i])
    else:
        aucs[lab] = float("nan")

end_test_time = time.time()  # Record the end time for test performance
print(f"Total testing time: {end_test_time - start_test_time:.2f} seconds")

# Print AUC results
for lab, auc in aucs.items():
    print(f"{lab:15s}: {auc:.4f}")
print("-" * 30)
print(f"Mean AUC        : {np.nanmean(list(aucs.values())):.4f}")

# ─── 7) Attention weight analysis ─────────────────────────────────────
modality_names = ['Graph', 'SMILES', 'Image', 'Spectrum']

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
    model_name="moltitox",
    title='MoltiTox Cross-Modal Attention',
    figsize=(10, 8)
)

# Save cross-modal attention analysis to JSON (saves to attention_save_dir/json/)
save_cross_modal_attention_analysis(
    mean_matrix=mean_matrix,
    std_matrix=std_matrix,
    modality_names=modality_names,
    save_dir=attention_save_dir,
    model_name="moltitox"
)