import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torchvision import transforms
import json
import optuna
import time
from torch.utils.data import ConcatDataset
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.models        import GraphEncoder, SMILESEncoder, ImageEncoder, MultiModalGphSMIImg
from utils.dataset     import GraphDataset, SMILESDataset, ImageDataset, GraphSMILESImageDataset

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Load data & split ─────────────────────────────────────────────────────
train_csv = os.path.join(ROOT, "data", "train.csv")
valid_csv = os.path.join(ROOT, "data", "valid.csv")
ckpt_dir = os.path.join(ROOT, "experiments", "image", "ImageMol.pth") 
img_dir= os.path.join(ROOT, "data", "images")
labels = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
          "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
          "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

# Define transforms for image data
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
valid_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

g_tr  = GraphDataset(train_csv, labels)
g_va  = GraphDataset(valid_csv, labels)
s_tr = SMILESDataset(                     
    csv_path=train_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
s_va = SMILESDataset(
    csv_path=valid_csv,
    label_name=labels,
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    max_length=202
)
i_tr  = ImageDataset(train_csv, labels, img_dir, transform=train_transform)
i_va  = ImageDataset(valid_csv, labels, img_dir, transform=valid_transform)
g_full = ConcatDataset([g_tr, g_va])
s_full = ConcatDataset([s_tr, s_va])
i_full = ConcatDataset([i_tr, i_va])

def collate(batch):
    """
    Custom collate function to handle batching of graph, smiles, and image data.
    Args:
        batch: List of tuples (graph, transformer IDs, transformer mask, image, labels).
    Returns:
        Tuple of batched graph data, transformer IDs, transformer masks, image tensors, and labels.
    """
    g,ids,mask,img,y = zip(*batch)
    g_batch = Batch.from_data_list(g)
    t_ids = torch.stack(ids)
    t_mask = torch.stack(mask)
    img_batch = torch.stack(img)
    y_batch = torch.stack(y)
    return g_batch, t_ids, t_mask, img_batch, y_batch

# ─── 3) encoder load & freeze ──────────────────────────────────────
with open(os.path.join(ROOT, "checkpoints", "parameters", "graph_best_params.json")) as f:
    graph_best_params = json.load(f)
graph_encoder = GraphEncoder(
    in_dim = 78,
    hidden_dim = graph_best_params["hidden_dim"],
    num_layers = graph_best_params["num_layers"],
    emb_dim = graph_best_params["emb_dim"]
); graph_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "graph_encoder.pth"), map_location=torch.device('cpu')))

with open(os.path.join(ROOT, "checkpoints", "parameters", "smiles_best_params.json")) as f:
    smiles_best_params = json.load(f)
smiles_encoder = SMILESEncoder(
    pretrained_model_name="ibm-research/MoLFormer-XL-both-10pct",
    emb_dim=smiles_best_params["emb_dim"],
    max_length=202
); smiles_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "smiles_encoder.pth"), map_location=torch.device('cpu')))

with open(os.path.join(ROOT, "checkpoints", "parameters", "image_best_params.json")) as f:
    image_best_params = json.load(f)
image_encoder = ImageEncoder(
    ckpt_path=ckpt_dir,
    emb_dim = image_best_params["emb_dim"],
    hidden_size= image_best_params["hidden_size"]
); image_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_only", "image_encoder.pth"), map_location=torch.device('cpu')))

for net in (graph_encoder, smiles_encoder, image_encoder):
    for p in net.parameters():
        p.requires_grad = False # Freeze parameters

# ─── 4) Optuna objective ────────────────────────────────────────────────
start_search_time = time.time() # Record the start time for hyperparameter search
def objective(trial):
    # Hyperparameters to tune
    emb_dim = trial.suggest_categorical("emb_dim", [128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [512, 768])
    num_layers = trial.suggest_categorical("num_layers", [1])
    lr    =    trial.suggest_categorical("lr", [1e-4, 3e-4])
    wd    =    trial.suggest_categorical("weight_decay", [1e-5, 1e-4])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5])

    # create the model
    model = MultiModalGphSMIImg(
        graph_encoder,
        smiles_encoder,
        image_encoder,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_tasks=len(labels)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=wd)

    train_loader = DataLoader(
        GraphSMILESImageDataset(g_tr,s_tr,i_tr), 
        batch_size = 32, 
        shuffle= True, 
        pin_memory=False, 
        collate_fn=collate,
        drop_last=True
    )
    valid_loader   = DataLoader(
        GraphSMILESImageDataset(g_va,s_va,i_va), 
        batch_size= 64, 
        shuffle= False, 
        pin_memory=False, 
        collate_fn=collate,
        drop_last=True
    )

    best_val_auc, best_epoch = 0.0, 0
    patience = 5
    for epoch in range(1, 31):
        # --- train ---
        model.train()
        for g, t_ids, t_mask, img, y in train_loader:
            g, t_ids, t_mask, img, y = (
                g.to(device),
                t_ids.to(device),
                t_mask.to(device),
                img.to(device),
                y.to(device)
            )
            logits = model(g, t_ids, t_mask, img)  

            mask_loss = (y >= 0).float()              
            targets   = y.clamp(min=0)            
            per_ent   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")          
            loss = (per_ent * mask_loss).sum() / mask_loss.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()
        # print(f"Epoch {epoch: 2d}")

        # --- validate ---
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for g, t_ids, t_mask, img, y in valid_loader:
                g, t_ids, t_mask, img = (
                    g.to(device),
                    t_ids.to(device),
                    t_mask.to(device),
                    img.to(device)
                )
                probs = torch.sigmoid(model(g, t_ids, t_mask, img))
                all_p.append(probs.cpu())
                all_y.append(y)

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
            trial.set_user_attr("best_model_state", model.state_dict())
            trial.set_user_attr("best_epoch", epoch)
        elif epoch - best_epoch >= patience:
            break

    return best_val_auc

# ─── 5) Run Optuna ────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=15)

end_search_time = time.time() # Record the end time for hyperparameter search
print(f"Total Searching completed in {end_search_time - start_search_time:.2f} seconds")

# ─── 6) Save hyperparameters ────────────────────────────────────────────────
best_trial = study.best_trial
best_params = best_trial.params
best_params["best_epoch"] = best_trial.user_attrs["best_epoch"]

with open(os.path.join(ROOT, "checkpoints", "parameters", "gph_smi_img_best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("Best AUC: ", study.best_value, "at epoch", best_trial.user_attrs["best_epoch"])

# ─── 7) Retrain on full dataset ──────────────────────────────────────────────────────────────
train_loader = DataLoader(
    GraphSMILESImageDataset(g_full, s_full, i_full), 
    batch_size=32, 
    shuffle=True, 
    pin_memory=False, 
    collate_fn=collate,
    drop_last=True
)

# load encoders trained with train and valid datasets
graph_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "graph_encoder.pth"), map_location=torch.device('cpu')))
smiles_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "smiles_encoder.pth"), map_location=torch.device('cpu')))
image_encoder.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth"), map_location=torch.device('cpu')))

# Freeze parameters of the encoders
for net in (graph_encoder, smiles_encoder, image_encoder):
    for p in net.parameters():
        p.requires_grad = False

# load the best hyperparameters and create the model
emb_dim = best_params["emb_dim"]
hidden_dim = best_params["hidden_dim"]
num_layers = best_params["num_layers"]
dropout = best_params["dropout"]
lr = best_params["lr"]
weight_decay = best_params["weight_decay"]
epoch      = best_params["best_epoch"]

model = MultiModalGphSMIImg(
    graph_encoder,
    smiles_encoder,
    image_encoder,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    num_tasks=len(labels)
).to(device)
opt = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

# train the model on the full dataset (train + valid)
start_train_time = time.time()  # Record the start time for train performance
for epoch in range(1, epoch + 1):
    model.train()
    for g, t_ids, t_mask, img, y in train_loader:
        g, t_ids, t_mask, img, y = (
            g.to(device),
            t_ids.to(device),
            t_mask.to(device),
            img.to(device),
            y.to(device)
        )

        logits = model(g, t_ids, t_mask, img)   
        mask_loss = (y >= 0).float()              
        targets   = y.clamp(min=0)            
        per_ent   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")          
        loss = (per_ent * mask_loss).sum() / mask_loss.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch: 2d}")
    
end_train_time = time.time() # Record the end time for train performance
print(f"Total training time: {end_train_time - start_train_time:.2f} seconds")

torch.save(
    model.state_dict(),
    os.path.join(ROOT, "checkpoints", "model", "gph_smi_img.pth")
) # Save the final model state