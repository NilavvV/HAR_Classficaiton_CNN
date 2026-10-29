
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(os.environ.get("HAR_DATA_DIR", "data/UCI HAR Dataset")).expanduser()
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2  # fraction of training subjects held out for validation
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]




def find_file(split, filename):
    """Locate a dataset file in the standard layout or a flattened one."""
    candidates = [
        DATA_DIR / split / "Inertial Signals" / filename,
        DATA_DIR / split / filename,
        DATA_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in: {[str(c) for c in candidates]}. "
        f"Set HAR_DATA_DIR or edit DATA_DIR (currently {DATA_DIR})."
    )


def read_table(path):
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


def load_split(split):
    X = np.stack(
        [read_table(find_file(split, f"{name}_{split}.txt")) for name in SIGNAL_NAMES],
        axis=1,
    )
    y = read_table(find_file(split, f"y_{split}.txt")).ravel() - 1  # labels 1..6 -> 0..5
    subjects = read_table(find_file(split, f"subject_{split}.txt")).ravel()
    return X, y, subjects


X_full, y_full, subj_full = load_split("train")
X_test, y_test, _ = load_split("test")
print("train:", X_full.shape, "test:", X_test.shape)  # (7352, 9, 128) (2947, 9, 128)
print("train class counts:", np.bincount(y_full))


splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
train_idx, val_idx = next(splitter.split(X_full, y_full, groups=subj_full))

X_train, y_train = X_full[train_idx], y_full[train_idx]
X_val, y_val = X_full[val_idx], y_full[val_idx]

print(f"train windows: {len(X_train)} ({len(np.unique(subj_full[train_idx]))} subjects)")
print(f"val windows:   {len(X_val)} ({len(np.unique(subj_full[val_idx]))} subjects)")



# %%
mean = X_train.mean(axis=(0, 2), keepdims=True)  # shape (1, 9, 1)
std = X_train.std(axis=(0, 2), keepdims=True)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Sanity check: training channels should have mean ~0 and std ~1.
print("train mean per channel:", X_train.mean(axis=(0, 2)).round(2))
print("train std per channel: ", X_train.std(axis=(0, 2)).round(2))


# %%
def to_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = to_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
val_loader = to_loader(X_val, y_val, 256, shuffle=False)
test_loader = to_loader(X_test, y_test, 256, shuffle=False)




class HARCNN(nn.Module):
    def __init__(self, n_channels=9, n_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # time: 128 -> 64

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # time: 64 -> 32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 32, n_classes),  # 64 channels x 32 time steps = 2048
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = HARCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # built from THIS model

# Sanity check: one forward pass should give (batch, 6).
xb, _ = next(iter(train_loader))
print("output shape:", tuple(model(xb.to(device)).shape))
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict(loader):
    """Return (predictions, labels) for every batch in a loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu())
            labels.append(yb)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


best_val_acc, best_epoch, best_state = 0.0, 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    val_pred, val_true = predict(val_loader)
    val_acc = (val_pred == val_true).mean()

    if val_acc > best_val_acc:
        best_val_acc, best_epoch = val_acc, epoch
        best_state = copy.deepcopy(model.state_dict())

    print(
        f"epoch {epoch:2d}: train loss = {running_loss / len(X_train):.4f}, "
        f"val acc = {val_acc:.4f}"
    )

print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")


model.load_state_dict(best_state)
y_pred, y_true = predict(test_loader)

test_acc = (y_pred == y_true).mean()
print(f"Test accuracy: {test_acc:.4f}\n")

print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, display_labels=CLASS_NAMES, xticks_rotation=45, ax=ax, colorbar=False
)
ax.set_title(f"UCI HAR test set (accuracy {test_acc:.3f})")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.show()


torch.save(best_state, RESULTS_DIR / "har_cnn_pytorch.pt")
print("Saved weights to", RESULTS_DIR / "har_cnn_pytorch.pt")
