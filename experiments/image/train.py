import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import optuna
import time

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.dataset import ImageDataset
from models.models import ImageEncoder, ImageClassifier

# ─── 1) Define your device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2) Load & split your data ─────────────────────────────────────────────────
train_csv = os.path.join(ROOT, "data", "train.csv")
valid_csv = os.path.join(ROOT, "data", "valid.csv")
img_dir  = os.path.join(ROOT, "data", "images")
ckpt_dir = "ImageMol.pth"  # Path to the pre-trained 2D CNN backbone
label_list = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# Define transforms for image data (Imagenet-style preprocessing)
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

train_ds = ImageDataset(train_csv, label_list, img_dir, transform=train_transform)
valid_ds = ImageDataset(valid_csv, label_list, img_dir, transform=valid_transform)
full_ds = ConcatDataset([train_ds, valid_ds])

# ─── 3) Optuna objective ─────────────────────────────────────────────────────
start_search_time = time.time() # Record the start time for hyperparameter search
def objective(trial):
    # Hyperparameters to tune
    hidden_size   = trial.suggest_categorical("hidden_size", [256, 512, 768])
    emb_dim       = trial.suggest_categorical("emb_dim", [128, 256])
    lr    =    trial.suggest_categorical("lr", [1e-4, 2e-4, 3e-4])
    wd         = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])

    # create the model
    encoder = ImageEncoder(
        ckpt_path= ckpt_dir,
        hidden_size= hidden_size,        
        emb_dim = emb_dim,
    )
    model = ImageClassifier(encoder, num_tasks=len(label_list)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, drop_last=True)

    best_val_auc, best_epoch = 0.0, 0
    patience = 10

    for epoch in range(1, 51):
        # --- train ---
        model.train()
        for images, labels in train_loader:
            images = images.to(device)     
            labels = labels.to(device)    

            logits = model(images)       

            mask    = (labels >= 0).float()  # create mask for valid entries
            targets = labels.clamp(min=0)      

            per_ent = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )  # compute loss                                     
            loss = (per_ent * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Epoch {epoch: 2d}")

        # --- validate ---
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                probs   = torch.sigmoid(model(images))  
                all_p.append(probs.cpu())
                all_y.append(labels.cpu())

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
            trial.set_user_attr("best_encoder_state", encoder.state_dict())
            trial.set_user_attr("best_epoch", epoch)
        elif epoch - best_epoch >= patience:
            break

    return best_val_auc


# ─── 4) Run Optuna ───────────────────────────────────────────────────────────────
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50)

end_search_time = time.time() # Record the end time for hyperparameter search
print(f"Total Searching completed in {end_search_time - start_search_time:.2f} seconds")

# ─── 5) Save encoder ──────────────────────────────────────────────────────────────
best_trial   = study.best_trial
best_params  = best_trial.params
best_params["best_epoch"] = best_trial.user_attrs["best_epoch"]

with open(os.path.join(ROOT, "checkpoints", "parameters", "image_best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

best_encoder_state = best_trial.user_attrs["best_encoder_state"]
torch.save(best_encoder_state, os.path.join(ROOT, "checkpoints", "encoder", "train_only", "image_encoder.pth"))

print("Best AUC:", study.best_value, "at epoch", best_trial.user_attrs["best_epoch"])

# ─── 6) Retrain on full dataset ──────────────────────────────────────────────────────────────
train_loader = DataLoader(full_ds, batch_size=32, shuffle=True, drop_last=True)

emb_dim       = best_params["emb_dim"]
hidden_size   = best_params["hidden_size"]
lr            = best_params["lr"]
wd            = best_params["weight_decay"]
epoch         = best_params["best_epoch"]

encoder = ImageEncoder(
    ckpt_path=ckpt_dir,
    hidden_size=hidden_size,
    emb_dim=emb_dim
).to(device)
model = ImageClassifier(encoder, num_tasks=len(label_list)).to(device)
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wd
)

# train the model on the full dataset (train + valid)
start_train_time = time.time()  # Record the start time for train performance
for epoch in range(1, epoch + 1):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)     
        labels = labels.to(device)   
        logits = model(images)    

        mask    = (labels >= 0).float()  # create mask for valid entries
        targets = labels.clamp(min=0)   

        per_ent = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # compute loss
        loss = (per_ent * mask).sum() / mask.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch: 2d}")

end_train_time = time.time() # Record the end time for train performance
print(f"Total training time: {end_train_time - start_train_time:.2f} seconds")

torch.save(
    encoder.state_dict(),
    os.path.join(ROOT, "checkpoints", "encoder", "train_and_valid", "image_encoder.pth")
) # Save the final encoder state
torch.save(
    model.state_dict(),
    os.path.join(ROOT, "checkpoints", "model", "image.pth")
) # Save the final model state