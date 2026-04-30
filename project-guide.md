# Complete Project Guide: Malware Family Classification with MLP

> **Task:** Classify Windows malware samples by family using 2,381-dimensional behavioral feature vectors, comparing a TF-IDF + Logistic Regression baseline against a PyTorch MLP neural network.
> **Fill in:** Replace all `[X.XXXX]` placeholders with your actual metric values after running training.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Upload to Cluster](#3-data-upload-to-cluster)
4. [HPC Job Scripts](#4-hpc-job-scripts)
5. [Jupyter Access via SSH Tunnel](#5-jupyter-access-via-ssh-tunnel)
6. [Preprocessing](#6-preprocessing)
7. [Baseline Model](#7-baseline-model)
8. [MLP Training](#8-mlp-training)
9. [Evaluation and Analysis](#9-evaluation-and-analysis)
10. [Report Compilation](#10-report-compilation)
11. [Submission Checklist](#11-submission-checklist)

---

## 1. Project Structure

```
project/
├── data/
│   ├── bodmas.npz              # raw BODMAS feature matrix
│   ├── bodmas_metadata.csv     # sample metadata (family labels)
│   └── processed_data.pkl      # preprocessed splits and encoder (raw numpy)
├── notebooks/
│   └── sprint1.ipynb           # EDA and preprocessing notebook
├── src/
│   └── train_mlp.py            # PyTorch MLP training script
├── models/
│   └── mlp_best.pt             # best MLP checkpoint (by val macro F1)
├── results/
│   ├── baseline_results.pkl    # TF-IDF + LR predictions and metrics
│   ├── mlp_results.pkl         # MLP predictions, metrics, and history
│   ├── per_family_results.csv  # per-family F1/precision/recall
│   ├── class_distribution.png # family distribution plot
│   ├── confusion_matrix.png    # MLP confusion matrix
│   └── model_comparison.png    # baseline vs MLP bar chart
├── report/
│   ├── main.tex
│   └── references.bib
├── run_jupyter.sh              # SLURM script for Jupyter notebook
├── run_train.sh                # SLURM script for MLP training
└── project-guide.md
```

---

## 2. Environment Setup

All work runs on the iTiger HPC cluster. Log in first:

```bash
ssh jtbass1@itiger.memphis.edu
```

### 2.1 Create Conda Environment

```bash
conda create -n malware python=3.9 -y
conda activate malware
```

### 2.2 Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter \
            torch ydata-profiling
```

### 2.3 Verify GPU Access

```python
import torch
print(torch.cuda.is_available())       # should print True on a GPU node
print(torch.cuda.get_device_name(0))
```

---

## 3. Data Upload to Cluster

From your local machine, upload the BODMAS dataset files via `scp`:

```bash
scp bodmas.npz       jtbass1@itiger.memphis.edu:~/project/data/
scp bodmas_metadata.csv jtbass1@itiger.memphis.edu:~/project/data/
```

Confirm the files arrived on the cluster:

```bash
ls -lh ~/project/data/
```

---

## 4. HPC Job Scripts

### 4.1 Jupyter Notebook — `run_jupyter.sh`

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=bigTiger
#SBATCH --job-name=jupyter

source ~/.bashrc
conda activate malware

echo "*** Starting Jupyter on: "$(hostname)
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

Submit the job:

```bash
sbatch run_jupyter.sh
```

### 4.2 MLP Training — `run_train.sh`

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=bigTiger
#SBATCH --job-name=mlp_train
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate malware

cd ~/project
python src/train_mlp.py
```

Submit the training job:

```bash
sbatch run_train.sh
```

Monitor job status:

```bash
squeue -u jtbass1
```

---

## 5. Jupyter Access via SSH Tunnel

After the Jupyter job starts, find the node name from the SLURM output:

```bash
cat slurm-<JOBID>.out   # look for "Starting Jupyter on: NODENAME"
```

On your **local machine**, open a two-hop SSH tunnel:

```bash
ssh -L 9999:localhost:9999 jtbass1@itiger.memphis.edu \
    -t ssh -L 9999:localhost:8888 NODENAME
```

Then open your browser to `http://localhost:9999` and enter the token printed in the SLURM output file.

---

## 6. Preprocessing

Run the following code in the Jupyter notebook or as a standalone script on the cluster. This loads the BODMAS dataset, filters to families with sufficient samples, converts feature vectors to token sequences, performs a stratified split, and saves processed data.

### 6.1 Load Dataset and Filter Families

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load raw data
data = np.load("data/bodmas.npz")
X = data["X"]                              # shape: (134435, 2381)
meta = pd.read_csv("data/bodmas_metadata.csv")
y_raw = meta["family"].values              # string family labels

# Filter: keep only families with >= 200 samples
family_counts = pd.Series(y_raw).value_counts()
valid_families = family_counts[family_counts >= 200].index.tolist()
mask = pd.Series(y_raw).isin(valid_families).values

X_filt = X[mask]
y_filt = y_raw[mask]

print(f"Families kept : {len(valid_families)}")   # 44
print(f"Samples kept  : {X_filt.shape[0]}")       # 49,970
```

### 6.2 Feature-to-Token Conversion

Each feature vector is converted to a sequence of `feat_i` tokens, one token per non-zero feature dimension:

```python
def vector_to_sequence(vec):
    """Convert a sparse feature vector to a space-separated feat_i token string."""
    nonzero_indices = np.nonzero(vec)[0]
    tokens = [f"feat_{i}" for i in nonzero_indices]
    return " ".join(tokens)

# Convert all filtered samples (this may take a few minutes)
sequences = [vector_to_sequence(X_filt[i]) for i in range(X_filt.shape[0])]
print("Example sequence (first 80 chars):", sequences[0][:80])
```

### 6.3 Label Encoding

```python
le = LabelEncoder()
y_encoded = le.fit_transform(y_filt)   # integer labels 0–43

print("Number of classes:", len(le.classes_))   # 44
print("Classes:", le.classes_[:5], "...")
```

### 6.4 Stratified Train / Validation / Test Split

```python
# 70% train, 15% val, 15% test — stratified by family
X_train_seq, X_temp, y_train, y_temp = train_test_split(
    sequences, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42
)
X_val_seq, X_test_seq, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train : {len(X_train_seq)}")   # 34,979
print(f"Val   : {len(X_val_seq)}")     # 7,495
print(f"Test  : {len(X_test_seq)}")    # 7,496
```

### 6.5 Save Processed Data

```python
processed = {
    "X_train": X_train_seq,
    "X_val":   X_val_seq,
    "X_test":  X_test_seq,
    "y_train": y_train,
    "y_val":   y_val,
    "y_test":  y_test,
    "label_encoder": le,
}
with open("data/processed_data.pkl", "wb") as f:
    pickle.dump(processed, f)
print("Saved processed_data.pkl")
```

---

## 7. Baseline Model

### 7.1 TF-IDF + Logistic Regression

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load processed data
with open("data/processed_data.pkl", "rb") as f:
    d = pickle.load(f)

X_train, y_train = d["X_train"], d["y_train"]
X_test,  y_test  = d["X_test"],  d["y_test"]

# TF-IDF over feat_i token sequences
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
lr.fit(X_train_tfidf, y_train)

preds = lr.predict(X_test_tfidf)
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds, average="macro")

print(f"Baseline  Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
print(classification_report(y_test, preds))
```

### 7.2 Save Baseline Results

```python
baseline_results = {
    "predictions": preds,
    "y_test":      y_test,
    "accuracy":    acc,
    "macro_f1":    f1,
    "tfidf":       tfidf,
    "model":       lr,
}
with open("results/baseline_results.pkl", "wb") as f:
    pickle.dump(baseline_results, f)
print("Saved results/baseline_results.pkl")
```

---

## 8. MLP Training

> **Why MLP instead of CodeBERT?**
> CodeBERT failed to converge on this data because the `feat_i` token representation
> (`feat_0`, `feat_14`, `feat_203`, …) provides no semantically meaningful input to a
> transformer pre-trained on real code. The MLP takes the raw 2,381-dimensional feature
> vectors directly — no tokenization needed — and trains natively on GPU via PyTorch.
> The high-dimensional behavioral feature space is well-suited to a feedforward network
> with batch normalization and dropout.

Place the following script at `src/train_mlp.py` on the cluster and submit via `run_train.sh`.

### 8.1 MLP Architecture

| Layer          | Size | Activation | Regularization |
|----------------|------|------------|----------------|
| Input          | 2,381 | —         | —              |
| Hidden 1       | 1,024 | ReLU      | BatchNorm, Dropout(0.3) |
| Hidden 2       | 512   | ReLU      | BatchNorm, Dropout(0.3) |
| Hidden 3       | 256   | ReLU      | BatchNorm, Dropout(0.2) |
| Output         | 44    | (softmax via CrossEntropyLoss) | — |

**Training settings:** AdamW optimizer (lr=1e-3, weight_decay=1e-4), batch size 256,
up to 50 epochs, early stopping on val macro F1 (patience=10),
ReduceLROnPlateau scheduler (patience=3).

### 8.2 `src/train_mlp.py`

```python
"""
Malware family classification: TF-IDF + LR baseline, then PyTorch MLP.
Input:  data/processed_data.pkl  (raw numpy feature vectors, integer labels)
Output: models/mlp_best.pt, results/mlp_results.pkl, results/per_family_results.csv,
        results/confusion_matrix.png, results/model_comparison.png
"""

import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

DATA_PATH = "data/processed_data.pkl"
MODEL_PATH = "models/mlp_best.pt"
RESULTS_DIR = "results"
NUM_CLASSES, INPUT_DIM = 44, 2381
BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY = 256, 50, 1e-3, 1e-4
EARLY_STOP_PATIENCE, SCHEDULER_PATIENCE, SEED = 10, 3, 42

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")
os.makedirs("models", exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)
X_train = d["X_train"].astype(np.float32); X_val = d["X_val"].astype(np.float32)
X_test  = d["X_test"].astype(np.float32)
y_train, y_val, y_test, le = d["y_train"], d["y_val"], d["y_test"], d["label_encoder"]

# ── TF-IDF + LR Baseline ──────────────────────────────────────────────────────
print("\n=== TF-IDF + Logistic Regression Baseline ===")
def vectors_to_sequences(X):
    return [" ".join(f"feat_{i}" for i in np.nonzero(vec)[0]) for vec in X]

train_seqs, test_seqs = vectors_to_sequences(X_train), vectors_to_sequences(X_test)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_seqs)
X_test_tfidf  = tfidf.transform(test_seqs)
lr_model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED)
lr_model.fit(X_train_tfidf, y_train)
bl_preds = lr_model.predict(X_test_tfidf)
bl_acc = accuracy_score(y_test, bl_preds)
bl_f1  = f1_score(y_test, bl_preds, average="macro")
print(f"Baseline  Accuracy: {bl_acc:.4f}  Macro-F1: {bl_f1:.4f}")
with open(f"{RESULTS_DIR}/baseline_results.pkl", "wb") as f:
    pickle.dump({"predictions": bl_preds, "y_test": y_test,
                 "accuracy": bl_acc, "macro_f1": bl_f1}, f)

# ── MLP ───────────────────────────────────────────────────────────────────────
class MalwareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(MalwareDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(MalwareDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(MalwareDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.net(x)

model     = MLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=SCHEDULER_PATIENCE)

print("\n=== MLP Training ===")
best_val_f1, no_improve, history = 0.0, 0, []
for epoch in range(1, EPOCHS + 1):
    model.train(); total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad(); loss = criterion(model(X_b), y_b)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            all_preds.extend(model(X_b.to(DEVICE)).argmax(1).cpu().tolist())
            all_labels.extend(y_b.tolist())
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1  = f1_score(all_labels, all_preds, average="macro")
    scheduler.step(val_f1)
    history.append({"epoch": epoch, "loss": avg_loss,
                    "val_acc": val_acc, "val_macro_f1": val_f1})
    print(f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} "
          f"| val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1; torch.save(model.state_dict(), MODEL_PATH); no_improve = 0
    else:
        no_improve += 1
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}"); break

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\n=== Test Evaluation ===")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)); model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        all_preds.extend(model(X_b.to(DEVICE)).argmax(1).cpu().tolist())
        all_labels.extend(y_b.tolist())
mlp_acc = accuracy_score(all_labels, all_preds)
mlp_f1  = f1_score(all_labels, all_preds, average="macro")
report  = classification_report(all_labels, all_preds,
                                target_names=le.classes_, output_dict=True)
print(f"Test Accuracy : {mlp_acc:.4f}\nTest Macro-F1 : {mlp_f1:.4f}")
with open(f"{RESULTS_DIR}/mlp_results.pkl", "wb") as f:
    pickle.dump({"predictions": all_preds, "y_test": all_labels,
                 "accuracy": mlp_acc, "macro_f1": mlp_f1,
                 "classification_report": report, "history": history}, f)
print("Saved results/mlp_results.pkl")

# ── Outputs ───────────────────────────────────────────────────────────────────
per_family = [{"family": cls, "f1": report[cls]["f1-score"],
               "precision": report[cls]["precision"],
               "recall": report[cls]["recall"],
               "support": int(report[cls]["support"])} for cls in le.classes_]
pd.DataFrame(per_family).sort_values("f1", ascending=False).to_csv(
    f"{RESULTS_DIR}/per_family_results.csv", index=False)

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel("Predicted Family"); ax.set_ylabel("True Family")
ax.set_title("MLP Confusion Matrix — Malware Family Classification")
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)

x = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - w/2, [bl_acc, mlp_acc], w, label="Accuracy",  color="steelblue")
ax.bar(x + w/2, [bl_f1,  mlp_f1],  w, label="Macro F1",  color="tomato")
ax.set_xticks(x); ax.set_xticklabels(["TF-IDF + LR", "MLP"])
ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
ax.set_title("Model Comparison: Malware Family Classification"); ax.legend()
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)

print("\n=== Final Summary ===")
print(f"TF-IDF + LR  Acc={bl_acc:.4f}  Macro-F1={bl_f1:.4f}")
print(f"MLP          Acc={mlp_acc:.4f}  Macro-F1={mlp_f1:.4f}")
print(f"Delta        Acc={mlp_acc - bl_acc:+.4f}  Macro-F1={mlp_f1 - bl_f1:+.4f}")
```

---

## 9. Evaluation and Analysis

Run the following analysis code in the Jupyter notebook after both models have been trained.

All outputs (confusion matrix, per-family CSV, model comparison chart) are generated
automatically by `src/train_mlp.py` at the end of training. To load and inspect results
manually in the Jupyter notebook:

### 9.1 Load Results

```python
import pickle

with open("data/processed_data.pkl", "rb") as f:
    d = pickle.load(f)
le = d["label_encoder"]

with open("results/baseline_results.pkl", "rb") as f:
    bl = pickle.load(f)

with open("results/mlp_results.pkl", "rb") as f:
    mlp = pickle.load(f)

print(f"Baseline  Acc={bl['accuracy']:.4f}  Macro-F1={bl['macro_f1']:.4f}")
print(f"MLP       Acc={mlp['accuracy']:.4f}  Macro-F1={mlp['macro_f1']:.4f}")
```

### 9.2 Per-Family F1 Scores

```python
import pandas as pd

df = pd.read_csv("results/per_family_results.csv")
print("Bottom 5 families by F1:")
print(df.tail(5)[["family", "f1"]].to_string(index=False))
print("\nTop 5 families by F1:")
print(df.head(5)[["family", "f1"]].to_string(index=False))
```

### 9.3 View Saved Figures

```python
from IPython.display import Image
Image("results/confusion_matrix.png")
```

```python
Image("results/model_comparison.png")
```

---

## 10. Report Compilation

### 10.1 Download ACL Style Files (on cluster or local machine)

```bash
cd report/
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl.sty
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl_natbib.sty
```

### 10.2 Compile LaTeX Report

```bash
cd report/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The final PDF should be at `report/main.pdf`. Verify it is at least 4 pages and all placeholder values have been replaced.

---

## 11. Submission Checklist

```
Deliverables
 [ ] report/main.tex compiled to PDF without errors (>= 4 pages)
 [ ] All [X.XXXX] placeholders replaced with actual metric values
 [ ] All [FILL IN] / [Your Full Name] / [UM ID: XXXXXXX] placeholders filled in
 [ ] Confusion matrix figure saved to results/confusion_matrix.png
 [ ] Model comparison chart saved to results/model_comparison.png
 [ ] Per-family results saved to results/per_family_results.csv

Code
 [ ] src/train_mlp.py runs end-to-end via sbatch run_train.sh
 [ ] Baseline results saved to results/baseline_results.pkl
 [ ] MLP results saved to results/mlp_results.pkl
 [ ] Best MLP checkpoint saved to models/mlp_best.pt
 [ ] processed_data.pkl present in data/

Repository
 [ ] No BigVul, CWE, or vulnerability classification content present
 [ ] project-guide.md, report/main.tex, report/references.bib up to date
 [ ] Random seed fixed at 42 throughout for reproducibility
```
