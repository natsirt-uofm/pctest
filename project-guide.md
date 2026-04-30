# Project Guide: Malware Family Classification via NLP on API Call Sequences Using CodeBERT

> **Dataset:** BODMAS (134,435 samples, 582 malware families, filtered to 44 families)
> **Cluster:** iTiger HPC — bigTiger partition
> **Fill in:** Replace all `[X.XXXX]` placeholders with real metric values after training.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Conda Environment Setup](#2-conda-environment-setup)
3. [Upload Data to Cluster](#3-upload-data-to-cluster)
4. [iTiger HPC Batch Scripts](#4-itiger-hpc-batch-scripts)
5. [Jupyter Access via SSH Tunnel](#5-jupyter-access-via-ssh-tunnel)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Baseline Model](#7-baseline-model)
8. [CodeBERT Training Script](#8-codebert-training-script)
9. [Analysis and Visualization](#9-analysis-and-visualization)
10. [Report Compilation](#10-report-compilation)
11. [Submission Checklist](#11-submission-checklist)

---

## 1. Project Structure

```
project/
├── data/
│   ├── bodmas.npz                # raw feature matrix + labels
│   ├── bodmas_metadata.csv       # sample metadata (family names, timestamps)
│   └── processed/
│       └── splits.pkl            # train/val/test splits after preprocessing
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_analysis.ipynb
├── src/
│   └── train_codebert.py         # full fine-tuning script
├── models/
│   └── codebert_best/            # saved best checkpoint (created at runtime)
├── results/
│   ├── baseline_results.pkl
│   └── codebert_results.pkl
├── report/
│   ├── main.tex
│   └── references.bib
├── run_jupyter.sh                # sbatch script — interactive Jupyter session
├── run_train.sh                  # sbatch script — CodeBERT training job
└── project-guide.md
```

---

## 2. Conda Environment Setup

Run these commands on the iTiger login node:

```bash
# Create environment with Python 3.9
conda create -n malware python=3.9 -y
conda activate malware

# Install all required packages
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter \
            transformers torch ydata-profiling

# Verify GPU-capable PyTorch (run after allocating a GPU node)
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 3. Upload Data to Cluster

From your **local machine**, copy the BODMAS files to your iTiger home directory:

```bash
# Upload raw data files
scp bodmas.npz        jtbass1@itiger.memphis.edu:~/project/data/
scp bodmas_metadata.csv jtbass1@itiger.memphis.edu:~/project/data/

# Verify upload
ssh jtbass1@itiger.memphis.edu "ls -lh ~/project/data/"
```

---

## 4. iTiger HPC Batch Scripts

### 4.1 `run_jupyter.sh` — Interactive Jupyter Session

Save as `run_jupyter.sh` in your project root:

```bash
#!/bin/bash
#SBATCH --job-name=jupyter_malware
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=jupyter_%j.log

module load anaconda3
conda activate malware

# Print the node name so you can build the SSH tunnel
echo "Jupyter node: $(hostname)"

jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

Submit the job:

```bash
sbatch run_jupyter.sh
# Check the log to get NODENAME
tail -f jupyter_<JOB_ID>.log
```

### 4.2 `run_train.sh` — CodeBERT Training Job

Save as `run_train.sh` in your project root:

```bash
#!/bin/bash
#SBATCH --job-name=codebert_malware
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.log

module load anaconda3
conda activate malware

cd ~/project
python src/train_codebert.py
```

Submit the training job:

```bash
sbatch run_train.sh
tail -f train_<JOB_ID>.log
```

---

## 5. Jupyter Access via SSH Tunnel

After the Jupyter job starts and you have `NODENAME` from the log:

```bash
# Two-hop SSH tunnel — run this on your LOCAL machine
ssh -L 9999:localhost:9999 jtbass1@itiger.memphis.edu \
    -t ssh -L 9999:localhost:8888 NODENAME
```

Then open `http://localhost:9999` in your browser.
Copy the token from the log file: `tail jupyter_<JOB_ID>.log`.

---

## 6. Data Preprocessing

Run in `notebooks/02_preprocessing.ipynb` or as a standalone script.

### 6.1 Load BODMAS Data

```python
import numpy as np
import pandas as pd

data = np.load("data/bodmas.npz", allow_pickle=True)
X    = data["X"]          # shape: (134435, 2381)
y    = data["y"]          # malware family labels (strings)

meta = pd.read_csv("data/bodmas_metadata.csv")
print(f"Total samples : {X.shape[0]}")
print(f"Feature dims  : {X.shape[1]}")
print(f"Unique families: {len(np.unique(y))}")
```

### 6.2 Filter to Families with >= 200 Samples

```python
from collections import Counter

family_counts = Counter(y)
valid_families = {fam for fam, cnt in family_counts.items() if cnt >= 200}
print(f"Families with >= 200 samples: {len(valid_families)}")  # 44

mask = np.array([label in valid_families for label in y])
X_filtered = X[mask]
y_filtered = y[mask]
print(f"Working samples after filtering: {X_filtered.shape[0]}")  # 49,970
```

### 6.3 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
import pickle, os

le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)

print(f"Number of classes: {len(le.classes_)}")  # 44
print("Sample family mapping:")
for i, name in enumerate(le.classes_[:5]):
    print(f"  {i}: {name}")

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
```

### 6.4 Feature Vector to Token Sequence

Each non-zero feature dimension `i` becomes the token `"feat_i"`.
This is the NLP framing that lets transformer models process feature vectors as text.

```python
def vector_to_sequence(feature_vector):
    """Convert a feature vector to a space-separated token sequence.

    Non-zero dimension i becomes token 'feat_i'.
    Zero-valued dimensions are omitted (sparse representation).
    """
    tokens = [f"feat_{i}" for i, val in enumerate(feature_vector) if val != 0]
    return " ".join(tokens)

# Convert all samples
print("Converting feature vectors to token sequences...")
sequences = [vector_to_sequence(row) for row in X_filtered]
print(f"Example sequence (first 80 chars): {sequences[0][:80]}")
print(f"Average tokens per sequence: {sum(len(s.split()) for s in sequences) / len(sequences):.1f}")
```

### 6.5 Stratified Train / Val / Test Split

```python
from sklearn.model_selection import train_test_split

X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    sequences, y_encoded,
    test_size=0.15,
    stratify=y_encoded,
    random_state=42,
)
X_train_seq, X_val_seq, y_train, y_val = train_test_split(
    X_train_seq, y_train,
    test_size=0.15 / 0.85,   # 15% of original total
    stratify=y_train,
    random_state=42,
)

print(f"Train : {len(X_train_seq):,}")   # 34,979
print(f"Val   : {len(X_val_seq):,}")     #  7,495
print(f"Test  : {len(X_test_seq):,}")    #  7,496

splits = {
    "X_train": X_train_seq, "y_train": y_train,
    "X_val":   X_val_seq,   "y_val":   y_val,
    "X_test":  X_test_seq,  "y_test":  y_test,
}
with open("data/processed/splits.pkl", "wb") as f:
    pickle.dump(splits, f)
print("Saved data/processed/splits.pkl")
```

---

## 7. Baseline Model

Run in `notebooks/03_baseline.ipynb`.

```python
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

with open("data/processed/splits.pkl", "rb") as f:
    splits = pickle.load(f)

X_train = splits["X_train"]; y_train = splits["y_train"]
X_val   = splits["X_val"];   y_val   = splits["y_val"]
X_test  = splits["X_test"];  y_test  = splits["y_test"]

# Build and fit pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf",   LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
])
pipeline.fit(X_train, y_train)

# Evaluate on test set
preds    = pipeline.predict(X_test)
test_acc = accuracy_score(y_test, preds)
test_f1  = f1_score(y_test, preds, average="macro")

print(f"Baseline  Accuracy : {test_acc:.4f}")
print(f"Baseline  Macro-F1 : {test_f1:.4f}")
print()
print(classification_report(y_test, preds))

# Save results
os.makedirs("results", exist_ok=True)
with open("results/baseline_results.pkl", "wb") as f:
    pickle.dump({
        "test_acc": test_acc,
        "test_f1":  test_f1,
        "preds":    preds,
        "targets":  y_test,
        "pipeline": pipeline,
    }, f)
print("Saved results/baseline_results.pkl")
```

---

## 8. CodeBERT Training Script

Save the full script below as `src/train_codebert.py`, then submit with `sbatch run_train.sh`.

```python
#!/usr/bin/env python3
"""
src/train_codebert.py
Fine-tune microsoft/codebert-base for malware family classification.
Run on iTiger HPC via: sbatch run_train.sh  (bigTiger partition, 1 GPU)
"""
import os, pickle, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (RobertaTokenizer,
                          RobertaForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE    = "data/processed/splits.pkl"
MODEL_DIR    = "models/codebert_best"
RESULTS_FILE = "results/codebert_results.pkl"
BASE_MODEL   = "microsoft/codebert-base"
MAX_LEN      = 512
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Load preprocessed splits ─────────────────────────────────────────────────
with open(DATA_FILE, "rb") as f:
    splits = pickle.load(f)

X_train, y_train = splits["X_train"], splits["y_train"]
X_val,   y_val   = splits["X_val"],   splits["y_val"]
X_test,  y_test  = splits["X_test"],  splits["y_test"]
NUM_CLASSES = int(max(y_train)) + 1
print(f"Classes: {NUM_CLASSES}  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)

class MalwareDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.sequences[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_loader = DataLoader(MalwareDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(MalwareDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(MalwareDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
model = RobertaForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_CLASSES,
).to(device)

optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0
history      = []

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    t0 = time.time()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)
        outputs   = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        train_loss += outputs.loss.item()
    avg_loss = train_loss / len(train_loader)

    # --- Validate ---
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            targets.extend(batch["label"].tolist())
    val_acc = accuracy_score(targets, preds)
    val_f1  = f1_score(targets, preds, average="macro")
    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  "
          f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({elapsed:.0f}s)")
    history.append({"epoch": epoch, "loss": avg_loss,
                    "val_acc": val_acc, "val_f1": val_f1})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"  -> Saved best checkpoint  (val_acc={best_val_acc:.4f})")

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nLoading best checkpoint for test evaluation...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
preds, targets = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        targets.extend(batch["label"].tolist())

test_acc = accuracy_score(targets, preds)
test_f1  = f1_score(targets, preds, average="macro")
print(f"\n=== Test Results ===")
print(f"Accuracy : {test_acc:.4f}")
print(f"Macro-F1 : {test_f1:.4f}")

os.makedirs("results", exist_ok=True)
with open(RESULTS_FILE, "wb") as f:
    pickle.dump({
        "test_acc": test_acc,
        "test_f1":  test_f1,
        "preds":    preds,
        "targets":  targets,
        "history":  history,
    }, f)
print(f"Results saved -> {RESULTS_FILE}")
```

---

## 9. Analysis and Visualization

Run in `notebooks/04_analysis.ipynb`.

### 9.1 Load Both Results

```python
import pickle
import numpy as np

with open("results/baseline_results.pkl", "rb") as f:
    bl = pickle.load(f)
with open("results/codebert_results.pkl", "rb") as f:
    cb = pickle.load(f)
with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print(f"Baseline  acc={bl['test_acc']:.4f}  f1={bl['test_f1']:.4f}")
print(f"CodeBERT  acc={cb['test_acc']:.4f}  f1={cb['test_f1']:.4f}")
```

### 9.2 Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# CodeBERT confusion matrix (44x44 — use abbreviated family names)
cm = confusion_matrix(cb["targets"], cb["preds"])
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel("Predicted Family"); ax.set_ylabel("True Family")
ax.set_title("CodeBERT — Confusion Matrix (44 Malware Families)")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.show()
```

### 9.3 Per-Family F1 Score

```python
from sklearn.metrics import f1_score

per_family_f1_bl = f1_score(bl["targets"], bl["preds"], average=None)
per_family_f1_cb = f1_score(cb["targets"], cb["preds"], average=None)

# Sort by CodeBERT F1 descending
order = np.argsort(per_family_f1_cb)[::-1]
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(le.classes_))
ax.bar(x - 0.2, per_family_f1_bl[order], width=0.4, label="Baseline", alpha=0.8, color="steelblue")
ax.bar(x + 0.2, per_family_f1_cb[order], width=0.4, label="CodeBERT", alpha=0.8, color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(le.classes_[order], rotation=90, fontsize=7)
ax.set_ylabel("F1 Score"); ax.set_title("Per-Family F1 — Baseline vs CodeBERT")
ax.legend()
plt.tight_layout()
plt.savefig("results/per_family_f1.png", dpi=150)
plt.show()
```

### 9.4 Model Comparison Bar Chart

```python
metrics   = ["Accuracy", "Macro F1"]
bl_scores = [bl["test_acc"], bl["test_f1"]]
cb_scores = [cb["test_acc"], cb["test_f1"]]

fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(metrics))
ax.bar(x - 0.2, bl_scores, width=0.35, label="TF-IDF + LR", color="steelblue")
ax.bar(x + 0.2, cb_scores, width=0.35, label="CodeBERT",    color="tomato")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.legend()
for i, (b, c) in enumerate(zip(bl_scores, cb_scores)):
    ax.text(i - 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
    ax.text(i + 0.2, c + 0.01, f"{c:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.show()
```

---

## 10. Report Compilation

```bash
cd ~/project/report/

# Download ACL 2023 style files (required once)
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl.sty
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl_natbib.sty

# Compile (run sequence twice for cross-references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# View output
ls -lh main.pdf
```

---

## 11. Submission Checklist

```
Data & Preprocessing
 [ ] data/bodmas.npz and data/bodmas_metadata.csv uploaded to cluster
 [ ] data/processed/splits.pkl exists with 34,979 / 7,495 / 7,496 samples
 [ ] data/processed/label_encoder.pkl saved (44 classes)

Training
 [ ] sbatch run_train.sh submitted on bigTiger partition
 [ ] results/codebert_results.pkl saved after training completes
 [ ] results/baseline_results.pkl saved after baseline run

Results
 [ ] All [X.XXXX] placeholders in report/main.tex replaced with real values
 [ ] results/confusion_matrix.png generated
 [ ] results/per_family_f1.png generated

Report
 [ ] report/main.tex compiles to PDF without errors (pdflatex + bibtex)
 [ ] Minimum 4 pages in compiled PDF
 [ ] Contributions section contains your full name and UM ID
 [ ] Exactly 4 BibTeX entries in references.bib

Repository
 [ ] run_jupyter.sh and run_train.sh both specify partition=bigTiger
```
