# Complete Project Guide: Malware Family Classification with CodeBERT

> **Task:** Classify Windows malware samples by family using API call feature sequences as NLP token sequences, comparing a TF-IDF + Logistic Regression baseline against a fine-tuned CodeBERT model.
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
8. [CodeBERT Fine-Tuning](#8-codebert-fine-tuning)
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
│   └── processed_data.pkl      # preprocessed splits and encoder
├── notebooks/
│   └── sprint1.ipynb           # EDA and preprocessing notebook
├── src/
│   └── train_codebert.py       # CodeBERT fine-tuning script
├── models/                     # saved model checkpoints
├── results/
│   ├── baseline_results.pkl    # TF-IDF + LR predictions and metrics
│   └── class_distribution.png # family distribution plot
├── report/
│   ├── main.tex
│   └── references.bib
├── run_jupyter.sh              # SLURM script for Jupyter notebook
├── run_train.sh                # SLURM script for CodeBERT training
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
            transformers torch ydata-profiling
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

### 4.2 CodeBERT Training — `run_train.sh`

```bash
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=bigTiger
#SBATCH --job-name=codebert_train
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate malware

cd ~/project
python src/train_codebert.py
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

## 8. CodeBERT Fine-Tuning

Place the following script at `src/train_codebert.py` on the cluster and submit via `run_train.sh`.

### 8.1 `src/train_codebert.py`

```python
"""
Fine-tune microsoft/codebert-base for malware family classification.
Input: feat_i token sequences derived from BODMAS feature vectors.
Output: best model checkpoint saved to models/best_codebert/
"""

import os, pickle, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import (RobertaTokenizer,
                          RobertaForSequenceClassification,
                          AdamW,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, f1_score

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/processed_data.pkl"
MODEL_NAME  = "microsoft/codebert-base"
SAVE_DIR    = "models/best_codebert"
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-5
MAX_LEN     = 512
NUM_CLASSES = 44
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Dataset ─────────────────────────────────────────────────────────────────
class MalwareDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = sequences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Load data ────────────────────────────────────────────────────────────────
with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

train_ds = MalwareDataset(d["X_train"], d["y_train"], tokenizer, MAX_LEN)
val_ds   = MalwareDataset(d["X_val"],   d["y_val"],   tokenizer, MAX_LEN)
test_ds  = MalwareDataset(d["X_test"],  d["y_test"],  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── Model ────────────────────────────────────────────────────────────────────
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ── Training loop ─────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1  = f1_score(all_labels, all_preds, average="macro")
    print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ** Saved best checkpoint (val_acc={best_val_acc:.4f})")

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\n=== Test Evaluation ===")
best_model = RobertaForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
best_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = best_model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        ).logits
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

test_acc = accuracy_score(all_labels, all_preds)
test_f1  = f1_score(all_labels, all_preds, average="macro")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")

import pickle
codebert_results = {
    "predictions": all_preds,
    "y_test":      all_labels,
    "accuracy":    test_acc,
    "macro_f1":    test_f1,
}
with open("results/codebert_results.pkl", "wb") as f:
    pickle.dump(codebert_results, f)
print("Saved results/codebert_results.pkl")
```

---

## 9. Evaluation and Analysis

Run the following analysis code in the Jupyter notebook after both models have been trained.

### 9.1 Load Results

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

with open("data/processed_data.pkl", "rb") as f:
    d = pickle.load(f)
le = d["label_encoder"]

with open("results/baseline_results.pkl", "rb") as f:
    bl = pickle.load(f)

with open("results/codebert_results.pkl", "rb") as f:
    cb = pickle.load(f)

print(f"Baseline  Acc={bl['accuracy']:.4f}  Macro-F1={bl['macro_f1']:.4f}")
print(f"CodeBERT  Acc={cb['accuracy']:.4f}  Macro-F1={cb['macro_f1']:.4f}")
```

### 9.2 Confusion Matrix

```python
cm = confusion_matrix(cb["y_test"], cb["predictions"])
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel("Predicted Family")
ax.set_ylabel("True Family")
ax.set_title("CodeBERT Confusion Matrix — Malware Family Classification")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
print("Saved results/confusion_matrix.png")
```

### 9.3 Per-Family F1 Scores

```python
from sklearn.metrics import classification_report

report = classification_report(
    cb["y_test"], cb["predictions"],
    target_names=le.classes_, output_dict=True
)
per_family_f1 = {cls: report[cls]["f1-score"] for cls in le.classes_}
sorted_f1 = sorted(per_family_f1.items(), key=lambda x: x[1])

print("Bottom 5 families by F1:")
for fam, score in sorted_f1[:5]:
    print(f"  {fam}: {score:.4f}")

print("\nTop 5 families by F1:")
for fam, score in sorted_f1[-5:]:
    print(f"  {fam}: {score:.4f}")
```

### 9.4 Model Comparison Bar Chart

```python
models    = ["TF-IDF + LR", "CodeBERT"]
accuracy  = [bl["accuracy"],  cb["accuracy"]]
macro_f1  = [bl["macro_f1"],  cb["macro_f1"]]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, accuracy, width, label="Accuracy",  color="steelblue")
ax.bar(x + width/2, macro_f1, width, label="Macro F1",  color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Malware Family Classification")
ax.legend()
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
print("Saved results/model_comparison.png")
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

Code
 [ ] src/train_codebert.py runs end-to-end via sbatch run_train.sh
 [ ] Baseline results saved to results/baseline_results.pkl
 [ ] CodeBERT results saved to results/codebert_results.pkl
 [ ] processed_data.pkl present in data/

Repository
 [ ] No BigVul, CWE, or vulnerability classification content present
 [ ] project-guide.md, report/main.tex, report/references.bib up to date
 [ ] Random seed fixed at 42 throughout for reproducibility
```
