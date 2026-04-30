"""
Malware family classification: TF-IDF + LR baseline, then PyTorch MLP.
Input:  data/processed_data.pkl  (raw numpy feature vectors, integer labels)
Output: models/mlp_best.pt, results/mlp_results.pkl, results/per_family_results.csv,
        results/confusion_matrix.png, results/model_comparison.png
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH          = "data/processed_data.pkl"
MODEL_PATH         = "models/mlp_best.pt"
RESULTS_DIR        = "results"
NUM_CLASSES        = 44
INPUT_DIM          = 2381
BATCH_SIZE         = 256
EPOCHS             = 100
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4
EARLY_STOP_PATIENCE = 15
SCHEDULER_PATIENCE  = 5
SEED               = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

X_train = d["X_train"].astype(np.float32)
X_val   = d["X_val"].astype(np.float32)
X_test  = d["X_test"].astype(np.float32)
y_train = d["y_train"]
y_val   = d["y_val"]
y_test  = d["y_test"]
le      = d["label_encoder"]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

from sklearn.preprocessing import StandardScaler

print("Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

import pickle as _pickle
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    _pickle.dump(scaler, f)
print("Scaler saved to models/scaler.pkl")

# ── TF-IDF + LR Baseline ─────────────────────────────────────────────────────
print("\n=== TF-IDF + Logistic Regression Baseline ===")

def vectors_to_sequences(X):
    """Convert raw feature vectors to feat_i token sequences for TF-IDF."""
    seqs = []
    for vec in X:
        indices = np.nonzero(vec)[0]
        seqs.append(" ".join(f"feat_{i}" for i in indices))
    return seqs

print("Converting feature vectors to token sequences for TF-IDF...")
train_seqs = vectors_to_sequences(X_train)
test_seqs  = vectors_to_sequences(X_test)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_seqs)
X_test_tfidf  = tfidf.transform(test_seqs)

lr_model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED)
lr_model.fit(X_train_tfidf, y_train)
bl_preds = lr_model.predict(X_test_tfidf)

bl_acc = accuracy_score(y_test, bl_preds)
bl_f1  = f1_score(y_test, bl_preds, average="macro")
print(f"Baseline  Accuracy: {bl_acc:.4f}  Macro-F1: {bl_f1:.4f}")

baseline_results = {
    "predictions": bl_preds,
    "y_test":      y_test,
    "accuracy":    bl_acc,
    "macro_f1":    bl_f1,
}
with open(f"{RESULTS_DIR}/baseline_results.pkl", "wb") as f:
    pickle.dump(baseline_results, f)
print("Saved results/baseline_results.pkl")

# ── MLP Dataset ───────────────────────────────────────────────────────────────
class MalwareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MalwareDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(MalwareDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(MalwareDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── MLP Model ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

model     = MLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE, verbose=True)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n=== MLP Training ===")
best_val_f1 = 0.0
no_improve  = 0
history     = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(DEVICE)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1  = f1_score(all_labels, all_preds, average="macro")
    scheduler.step(val_f1)

    history.append({"epoch": epoch, "loss": avg_loss,
                    "val_acc": val_acc, "val_macro_f1": val_f1})
    print(f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} "
          f"| val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_PATH)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\n=== Test Evaluation ===")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch.to(DEVICE)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())

mlp_acc = accuracy_score(all_labels, all_preds)
mlp_f1  = f1_score(all_labels, all_preds, average="macro")
report  = classification_report(all_labels, all_preds,
                                target_names=le.classes_, output_dict=True)
print(f"Test Accuracy : {mlp_acc:.4f}")
print(f"Test Macro-F1 : {mlp_f1:.4f}")

mlp_results = {
    "predictions":           all_preds,
    "y_test":                all_labels,
    "accuracy":              mlp_acc,
    "macro_f1":              mlp_f1,
    "classification_report": report,
    "history":               history,
}
with open(f"{RESULTS_DIR}/mlp_results.pkl", "wb") as f:
    pickle.dump(mlp_results, f)
print("Saved results/mlp_results.pkl")

# ── Per-family CSV ─────────────────────────────────────────────────────────────
per_family = [
    {"family": cls,
     "f1":        report[cls]["f1-score"],
     "precision": report[cls]["precision"],
     "recall":    report[cls]["recall"],
     "support":   int(report[cls]["support"])}
    for cls in le.classes_
]
pd.DataFrame(per_family).sort_values("f1", ascending=False).to_csv(
    f"{RESULTS_DIR}/per_family_results.csv", index=False
)
print("Saved results/per_family_results.csv")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel("Predicted Family")
ax.set_ylabel("True Family")
ax.set_title("MLP Confusion Matrix — Malware Family Classification")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
print("Saved results/confusion_matrix.png")

# ── Model Comparison Chart ────────────────────────────────────────────────────
models_list = ["TF-IDF + LR", "MLP"]
accuracy    = [bl_acc,  mlp_acc]
macro_f1    = [bl_f1,   mlp_f1]

x     = np.arange(len(models_list))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="steelblue")
ax.bar(x + width / 2, macro_f1, width, label="Macro F1", color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Malware Family Classification")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
print("Saved results/model_comparison.png")

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n=== Final Summary ===")
print(f"TF-IDF + LR  Acc={bl_acc:.4f}  Macro-F1={bl_f1:.4f}")
print(f"MLP          Acc={mlp_acc:.4f}  Macro-F1={mlp_f1:.4f}")
print(f"Delta        Acc={mlp_acc - bl_acc:+.4f}  Macro-F1={mlp_f1 - bl_f1:+.4f}")
