import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import json
import torch.nn.functional as F
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models import GraphEncoder, ImageEncoder, MultiModalGphImg
from utils.dataset import GraphDataset, ImageDataset, GraphImageDataset

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Load data & split ─────────────────────────────────────────────────────
test_csv = os.path.join(ROOT, "data", "test.csv")
ckpt_dir = os.path.join(ROOT, "experiments", "image", "ImageMol.pth") 
img_dir= os.path.join(ROOT, "data", "images")
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
i_test = ImageDataset(test_csv, labels, img_dir, transform=test_transform)

def collate(batch):
    """
    Custom collate function to handle batching of graph and image data.
    Args:
        batch (list): List of tuples (graph, image, labels).
    Returns:
        Tuple of batched graph data, image tensors, and labels.
    """
    graph, img, y = zip(*batch)
    g_batch = Batch.from_data_list(graph)
    img_batch = torch.stack(img)
    y_batch = torch.stack(y)
    return g_batch, img_batch, y_batch

test_loader = DataLoader(
    GraphImageDataset(g_test, i_test),
    batch_size=64,
    shuffle=False,
    pin_memory=False,
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

with open(os.path.join(ROOT, "checkpoints", "parameters", "image_best_params.json")) as f:
    image_best_params = json.load(f)
image_encoder = ImageEncoder(
    ckpt_path=ckpt_dir,
    emb_dim = image_best_params["emb_dim"],
    hidden_size= image_best_params["hidden_size"]
); image_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth"), map_location=torch.device('cpu')))

# ─── 4) hyperparameters ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "gph_img_best_params.json")) as f:
    gph_img_best_params = json.load(f)

emb_dim = gph_img_best_params["emb_dim"]
hidden_dim = gph_img_best_params["hidden_dim"]
num_layers = gph_img_best_params["num_layers"]
dropout = gph_img_best_params["dropout"]
epoch      = gph_img_best_params["best_epoch"]

# ─── 5) Create the model ───────────────────────────────
model = MultiModalGphImg(
    graph_encoder,
    image_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_tasks=len(labels),
    dropout=dropout
).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT, "checkpoints", "model", "gph_img.pth"), map_location=device))

# ─── 6) Test model ─────────────────────────────────────
start_test_time = time.time()  # Record the start time for test performance
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for g, img, y in test_loader:
        g, img = g.to(device), img.to(device)
        logits = model(g, img)
        probs = torch.sigmoid(logits)
        all_y.append(y)
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
