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

from models.models import SMILESEncoder, ImageEncoder, MultiModalSMIImg
from utils.dataset import SMILESDataset, ImageDataset, SMILESImageDataset

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Load data & split ─────────────────────────────────────────────────────
test_csv = os.path.join(ROOT, "data", "test.csv")
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

s_test = SMILESDataset(
    csv_path=test_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
i_test = ImageDataset(test_csv, labels, img_dir, transform=test_transform)

def collate(batch):
    """
    Custom collate function to handle batching of SMILES and image data.
    Args:
        batch: List of tuples (transformer IDs, transformer mask, image, labels).
    Returns:
        Tuple of batched transformer IDs, transformer mask, images, and labels.
    """
    ids, mask, img, y = zip(*batch)
    t_ids = torch.stack(ids)
    t_mask = torch.stack(mask)
    img_batch = torch.stack(img)
    y_batch = torch.stack(y)
    return t_ids, t_mask, img_batch, y_batch

test_loader = DataLoader(
    SMILESImageDataset(s_test, i_test),
    batch_size=64,
    shuffle=False,
    collate_fn=collate
)

# ─── 3) encoder load  ──────────────────────────────────────
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
    hidden_size=image_best_params["hidden_size"],
    emb_dim=image_best_params["emb_dim"]
); image_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth"), map_location=torch.device('cpu')))

# ─── 4) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "smi_img_best_params.json")) as f:
    smi_img_best_params = json.load(f)

emb_dim = smi_img_best_params["emb_dim"]
hidden_dim = smi_img_best_params["hidden_dim"]
num_layers = smi_img_best_params["num_layers"]
dropout = smi_img_best_params["dropout"]
epoch = smi_img_best_params["best_epoch"]

# ─── 5) Create the model ──────────────────────────────
model = MultiModalSMIImg(
    smiles_encoder,
    image_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "smi_img.pth"), map_location=device))

# ─── 6) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for t_ids, t_mask, img, y in test_loader:
        t_ids, t_mask, img = t_ids.to(device), t_mask.to(device), img.to(device)
        logits = model(t_ids, t_mask, img)
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
