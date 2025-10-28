import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.dataset import ImageDataset
from models.models import ImageEncoder, ImageClassifier

# ─── 1) Define your device ────────────────────────────────────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2) Load data & split ─────────────────────────────────────────────────────
test_csv   = os.path.join(ROOT, "data", "test.csv")
img_dir  = os.path.join(ROOT, "data", "images")
ckpt_dir = "ImageMol.pth"  # Path to the pre-trained CNN backbone
label_list = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# Define transforms for image data (Imagenet-style preprocessing)
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_ds     = ImageDataset(test_csv, label_list, img_dir, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, pin_memory=True)

# ─── 3) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "image_best_params.json")) as f:
    best_params = json.load(f)

emb_dim       = best_params["emb_dim"]
hidden_size   = best_params["hidden_size"]

# ─── 4) Create the model ──────────────────────────────
encoder = ImageEncoder(
    ckpt_path=ckpt_dir,
    hidden_size=hidden_size,
    emb_dim=emb_dim
).to(device)
encoder.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth"), map_location=device))
model = ImageClassifier(encoder, num_tasks=len(label_list)).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "image.pth"), map_location=device))

# ─── 5) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        probs   = torch.sigmoid(model(images))
        all_p.append(probs.cpu())
        all_y.append(labels.cpu())

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