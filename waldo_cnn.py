"""
Waldo? — CNN From Scratch Classifier
=============================================
Task    : Classify 256×256 images as Waldo or Not Waldo
Loss    : BCEWithLogitsLoss
Metrics : Accuracy, Precision, Recall, F1

Run:
    python waldo_cnn.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# =============================================
# TRANSFORMS
# =============================================

# Diversifies the training data by tweaking its appearance
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08), #randomly tweaks the color of the image
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),  #slight rotation, translation, and zoom
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),    #randomly blacks out a rectangular region
])

# Normalize the image size, tensor, and pixel values
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =============================================
# DATASET
# =============================================

class WaldoDataset(Dataset):
    """
    Loads 256×256 images from waldo/ and notwaldo/ folders.

    Args:
        file_list : list of (image_path, label) tuples
                    label = 1 for waldo, 0 for notwaldo
        train     : if True applies augmentation transforms
    """
    def __init__(self, file_list, train=True):
        self.samples   = file_list
        self.transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor([label], dtype=torch.float32)


# Splits data into train (70%), validate (15%), and test(15%)
def build_splits(data_root, val_size=0.15, test_size=0.15, seed=42):
    """
    Builds stratified train / val / test splits.
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    samples = []
    for label, folder in [(1, "waldo"), (0, "notwaldo")]:
        folder_path = os.path.join(data_root, folder)
        for fname in os.listdir(folder_path):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                samples.append((os.path.join(folder_path, fname), label))

    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    p_tv, p_test, l_tv, l_test = train_test_split(paths, labels, test_size=test_size, stratify=labels, random_state=seed)

    val_ratio = val_size / (1 - test_size)
    p_train, p_val, l_train, l_val = train_test_split(p_tv, l_tv, test_size=val_ratio, stratify=l_tv, random_state=seed)

    train_list = list(zip(p_train, l_train))
    val_list   = list(zip(p_val,   l_val))
    test_list  = list(zip(p_test,  l_test))

    def count_waldo(lst): return sum(1 for _, l in lst if l == 1)
    print(f"\n[Splits]")
    print(f"  Train : {len(train_list):4d} samples  ({count_waldo(train_list)} waldo)")
    print(f"  Val   : {len(val_list):4d} samples  ({count_waldo(val_list)} waldo)")
    print(f"  Test  : {len(test_list):4d} samples  ({count_waldo(test_list)} waldo)")

    return train_list, val_list, test_list


def make_weighted_sampler(file_list):
    """
    Oversampler to correctly handle any imbalanced data set
    """
    labels     = [l for _, l in file_list]
    n_waldo    = max(sum(labels), 1)
    n_notwaldo = len(labels) - n_waldo
    weight_map = {1: n_notwaldo / n_waldo, 0: 1.0}
    weights    = [weight_map[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# =============================================
# CNN MODEL
# =============================================

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),     # normalizes output of filter
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class WaldoCNN(nn.Module):
    """
    Custom CNN trained from scratch - 5 convolutional blocks

    Spatial reduction:  256 -> 128 -> 64 -> 32 -> 16 -> 8
    Channel expansion:    3 -> 32 -> 64 -> 128 -> 256 -> 256

    Input  : (B, 3, 256, 256)
    Output : (B, 1)  raw logit
    """
    def __init__(self, dropout=0.5):
        super().__init__()

        # ----- Feature extractor -----
        # Block 1: 256×256 -> 128×128  |  3 -> 32 channels
        self.block1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2)
        )
        # Block 2: 128×128 -> 64×64   |  32 -> 64 channels
        self.block2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2)
        )
        # Block 3: 64×64 -> 32×32     |  64 -> 128 channels
        self.block3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2)
        )
        # Block 4: 32×32 -> 16×16     |  128 -> 256 channels
        self.block4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2)
        )
        # Block 5: 16×16 -> 8×8       |  256 -> 256 channels
        self.block5 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2)
        )

        # ----- Classification head -----
        # Global Average Pooling: (B, 256, 8, 8) -> (B, 256)
        # Much fewer params than Flatten (256×8×8=16384 -> 256)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)       # raw logit
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        return self.head(x)         # (B, 1)

    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            probs = torch.sigmoid(self.forward(x))
            return (probs >= threshold).long().squeeze(1)


# =============================================
# METRICS
# =============================================

def compute_metrics(logits, labels, threshold=0.5):
    """Returns accuracy, precision, recall, F1 from raw logits."""
    probs     = torch.sigmoid(logits).squeeze(1)
    predicted = (probs >= threshold).long()
    labels    = labels.squeeze(1).long()

    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()

    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall  = tp / (tp + fn + 1e-6)
    f1   = 2 * precision * recall / (precision + recall + 1e-6)
    return accuracy, precision, recall, f1


# =============================================
# TRAINING
# =============================================

def train(
    data_root  = "data_roboflow/",
    epochs     = 50,         
    batch_size = 16,
    lr         = 1e-3,
    device     = None,
    save_path  = "waldo_cnn_best.pth"
):

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    pin = (device == "cuda")
    print(f"\n{'='*60}")
    print(f"  Waldo CNN — Classifier Training")
    print(f"  Epochs: {epochs}")
    print(f"{'='*60}")

    # ===== Data ===== 
    train_list, val_list, test_list = build_splits(data_root)

    train_ds = WaldoDataset(train_list, train=True)
    val_ds   = WaldoDataset(val_list,   train=False)
    test_ds  = WaldoDataset(test_list,  train=False)

    sampler      = make_weighted_sampler(train_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # ===== Model ===== 
    model     = WaldoCNN(dropout=0.5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "f1": []}

    for epoch in range(1, epochs + 1):

        # ===== Train ===== 
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{epochs} [Train]", leave=False)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # ===== Validate ===== 
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        val_bar = tqdm(val_loader, desc=f"Ep {epoch:03d}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                logits    = model(imgs)
                val_loss += criterion(logits, labels).item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                val_bar.set_postfix(loss=f"{criterion(logits, labels).item():.4f}")

        val_loss   /= len(val_loader)
        logits_cat  = torch.cat(all_logits)
        labels_cat  = torch.cat(all_labels)
        acc, prec, rec, f1 = compute_metrics(logits_cat, labels_cat)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["f1"].append(f1)

        print(f"Ep {epoch:03d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (F1={best_f1:.4f})")

    # ===== Test evaluation ===== 
    print(f"\n{'='*60}")
    print(f"  Final Test Set Evaluation")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            all_logits.append(model(imgs.to(device)).cpu())
            all_labels.append(labels)

    acc, prec, rec, f1 = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    
    print(f"  Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

    return model, history


# =============================================
# INFERENCE
# =============================================

def predict(image_path, model_path="waldo_cnn_best.pth",
            threshold=0.5, device=None):
    """Run classifier on a single 256×256 image."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    pin = (device == "cuda")
    model  = WaldoCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    img    = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    return {
        "is_waldo" : prob >= threshold,
        "prob"     : prob,
        "label"    : "Waldo" if prob >= threshold else "Not Waldo"
    }


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    
    model = WaldoCNN()
    total = sum(p.numel() for p in model.parameters())
    print("Running Waldo CNN evaluation.....")
    print(f"  Total params : {total:,}")

    # Train
    model, history = train(
        data_root  = "data_roboflow/",
        epochs     = 50,
        batch_size = 16,
        lr         = 1e-3,
    )