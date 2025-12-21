import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import json
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models import SpectrumEncoder, ImageEncoder, MultiModalSpecImg
from utils.dataset import SpectrumDataset, ImageDataset, SpectrumImageDataset

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
ckpt_dir = os.path.join(ROOT, "experiments", "image", "ImageMol.pth") 
img_dir = os.path.join(ROOT, "data", "images")
labels = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
          "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
          "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

sp_test = SpectrumDataset(test_csv, labels, spectra_dir, allow_missing=True)
i_test = ImageDataset(test_csv, labels, img_dir, transform=test_transform)

def collate(batch):
    """
    Custom collate function to handle batching of spectrum and image data.
    Args:
        batch: List of tuples (ppm_list, image, labels).
    Returns:
        Tuple of list of ppm values, batched images, and labels.
    """
    ppm_lists, img, y = zip(*batch)
    ppm_batch = list(ppm_lists)
    img_batch = torch.stack(img)
    y_batch = torch.stack(y)
    return ppm_batch, img_batch, y_batch

test_loader = DataLoader(
    SpectrumImageDataset(sp_test, i_test),
    batch_size=64,
    shuffle=False,
    collate_fn=collate
)

# ─── 3) encoder load ──────────────────────────────────────
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

with open(os.path.join(ROOT, "checkpoints", "parameters", "image_best_params.json")) as f:
    image_best_params = json.load(f)
image_encoder = ImageEncoder(
    ckpt_path= ckpt_dir,
    hidden_size=image_best_params["hidden_size"],
    emb_dim=image_best_params["emb_dim"]
); image_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth"), map_location=torch.device('cpu')))

# ─── 4) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "spec_img_best_params.json")) as f:
    spec_img_best_params = json.load(f)

emb_dim = spec_img_best_params["emb_dim"]
hidden_dim = spec_img_best_params["hidden_dim"]
num_layers = spec_img_best_params["num_layers"]
dropout = spec_img_best_params["dropout"]
epoch = spec_img_best_params["best_epoch"]

# ─── 5) Create the model ──────────────────────────────
model = MultiModalSpecImg(
    spectrum_encoder,
    image_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "spec_img.pth"), map_location=device))

# ─── 6) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for ppm_list, img, y in test_loader:
        img = img.to(device)
        logits = model(ppm_list, img)
        probs = torch.sigmoid(logits)
        all_p.append(probs.cpu())
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
