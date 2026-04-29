# Complete Project Guide: Vulnerability Classification with CodeBERT

> **Task:** Classify source code snippets by vulnerability type (CWE category) using a baseline TF-IDF + Logistic Regression model and a fine-tuned CodeBERT model.
> **Fill in:** Replace all `[X.XXXX]` placeholders with your actual metric values after running training.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Acquisition](#3-dataset-acquisition)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Preprocessing](#5-preprocessing)
6. [Baseline Model](#6-baseline-model)
7. [CodeBERT Fine-Tuning](#7-codebert-fine-tuning)
8. [Evaluation](#8-evaluation)
9. [Error Analysis](#9-error-analysis)
10. [Report & Slides Compilation](#10-report--slides-compilation)
11. [Submission Checklist](#11-submission-checklist)

---

## 1. Project Structure

```
project/
├── data/
│   ├── raw/                  # original downloaded dataset
│   └── processed/            # tokenized, split CSVs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_codebert.ipynb
├── src/
│   ├── dataset.py            # PyTorch Dataset class
│   ├── baseline.py           # TF-IDF + LR pipeline
│   ├── model.py              # CodeBERT fine-tune wrapper
│   ├── train.py              # training loop
│   └── evaluate.py           # metrics computation
├── report/
│   ├── main.tex
│   └── references.bib
├── slides/
│   └── slides.html
├── requirements.txt
└── project-guide.md
```

---

## 2. Environment Setup

### 2.1 Python Virtual Environment

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2.2 Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas numpy matplotlib seaborn
pip install jupyter ipykernel accelerate evaluate
pip install tqdm tokenizers sentencepiece

# Save exact versions
pip freeze > requirements.txt
```

### 2.3 Verify GPU Access

```python
import torch
print(torch.cuda.is_available())          # should print True
print(torch.cuda.get_device_name(0))      # e.g. NVIDIA A100
```

### 2.4 Hugging Face Login (for gated models)

```bash
huggingface-cli login
# paste your HF token when prompted
```

---

## 3. Dataset Acquisition

### 3.1 BigVul Dataset (recommended)

BigVul contains 3,754 C/C++ vulnerability-fixing commits labeled with CWE IDs.

```bash
# Download BigVul
wget https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view
# or clone the MSR2020 repo
git clone https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git data/raw/bigvul
```

```python
import pandas as pd

df = pd.read_csv("data/raw/bigvul/MSR_20_Code_vulnerability_CSV_Dataset.csv")
print(df.shape)             # (186,878, 19)
print(df.columns.tolist())
print(df["CWE ID"].value_counts().head(10))
```

### 3.2 Alternative: Devign Dataset

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF -O data/raw/devign.json
```

```python
import json, pandas as pd

with open("data/raw/devign.json") as f:
    raw = json.load(f)
df = pd.DataFrame(raw)
print(df.head())
```

### 3.3 Train / Validation / Test Split

```python
from sklearn.model_selection import train_test_split

# Keep top-N CWE classes to avoid extreme class imbalance
TOP_N = 10
top_cwes = df["CWE ID"].value_counts().head(TOP_N).index
df_filtered = df[df["CWE ID"].isin(top_cwes)].copy()

train_df, test_df = train_test_split(df_filtered, test_size=0.15, stratify=df_filtered["CWE ID"], random_state=42)
train_df, val_df  = train_test_split(train_df,    test_size=0.15, stratify=train_df["CWE ID"],  random_state=42)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv",     index=False)
test_df.to_csv("data/processed/test.csv",   index=False)
```

---

## 4. Exploratory Data Analysis

Run `notebooks/01_eda.ipynb` with the cells below.

```python
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

train_df = pd.read_csv("data/processed/train.csv")

# --- Class distribution ---
fig, ax = plt.subplots(figsize=(10, 4))
train_df["CWE ID"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("CWE Class Distribution (Train)")
ax.set_xlabel("CWE ID"); ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig("data/processed/class_dist.png", dpi=150)

# --- Code length distribution ---
train_df["code_len"] = train_df["func_before"].str.split().str.len()
print(train_df["code_len"].describe())

fig, ax = plt.subplots(figsize=(8, 3))
train_df["code_len"].clip(upper=1000).hist(bins=50, ax=ax, color="tomato")
ax.set_title("Token Count Distribution (clipped at 1000)"); ax.set_xlabel("Tokens")
plt.tight_layout(); plt.savefig("data/processed/len_dist.png", dpi=150)
```

---

## 5. Preprocessing

### 5.1 Label Encoding

```python
# src/dataset.py  (relevant excerpt)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

train_df = pd.read_csv("data/processed/train.csv")
le = LabelEncoder()
le.fit(train_df["CWE ID"])

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/processed/{split}.csv")
    df["label"] = le.transform(df["CWE ID"])
    df.to_csv(f"data/processed/{split}.csv", index=False)

import pickle
with open("data/processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Classes:", le.classes_)
NUM_LABELS = len(le.classes_)
```

### 5.2 PyTorch Dataset Class

```python
# src/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

TOKENIZER = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
MAX_LEN   = 512

class CodeDataset(Dataset):
    def __init__(self, df, text_col="func_before"):
        self.texts  = df[text_col].fillna("").tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = TOKENIZER(
            self.texts[idx],
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
```

### 5.3 Preprocessing for Baseline

```python
# Baseline uses raw token counts; minimal cleaning only
import re

def clean_code(text: str) -> str:
    text = re.sub(r"//.*?\n",  " ", text)   # strip line comments
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)  # block comments
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["clean"] = train_df["func_before"].fillna("").apply(clean_code)
```

---

## 6. Baseline Model

### 6.1 TF-IDF + Logistic Regression

```python
# src/baseline.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd, pickle

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",   # character n-grams capture code syntax
        ngram_range=(3, 6),
        max_features=100_000,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    )),
])

pipeline.fit(train_df["clean"], train_df["label"])

# --- Evaluate on test set ---
preds = pipeline.predict(test_df["clean"])
acc   = accuracy_score(test_df["label"], preds)
f1    = f1_score(test_df["label"], preds, average="macro")

print(f"Baseline  Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
print(classification_report(test_df["label"], preds))

with open("data/processed/baseline_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
```

### 6.2 Additional Baselines (optional)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Swap the clf step to compare
rf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)),
    ("clf",   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
])
rf_pipeline.fit(train_df["clean"], train_df["label"])
```

---

## 7. CodeBERT Fine-Tuning

### 7.1 Model Definition

```python
# src/model.py
from transformers import RobertaForSequenceClassification

def get_codebert_model(num_labels: int):
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=num_labels,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    return model
```

### 7.2 Training Loop

```python
# src/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

from src.dataset import CodeDataset
from src.model   import get_codebert_model

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5
WARMUP     = 0.1   # fraction of total steps

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
NUM_LABELS = train_df["label"].nunique()

train_loader = DataLoader(CodeDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(CodeDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False)

model     = get_codebert_model(NUM_LABELS).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP)
scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

best_val_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # --- Validate ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            logits    = model(input_ids=input_ids, attention_mask=attn_mask).logits
            preds     = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    from sklearn.metrics import f1_score
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"  Loss: {avg_loss:.4f}  Val Macro-F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained("data/processed/best_codebert")
        print(f"  ** Saved best model (F1={best_val_f1:.4f})")
```

### 7.3 Run Training

```bash
python -m src.train \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --warmup 0.1 \
  --seed 42
```

---

## 8. Evaluation

### 8.1 Test Set Evaluation

```python
# src/evaluate.py
import torch, pickle, pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)
import seaborn as sns, matplotlib.pyplot as plt

from src.dataset import CodeDataset

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df  = pd.read_csv("data/processed/test.csv")
loader   = DataLoader(CodeDataset(test_df), batch_size=32, shuffle=False)

model = RobertaForSequenceClassification.from_pretrained("data/processed/best_codebert").to(DEVICE)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        ).logits
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

print("=== CodeBERT Test Results ===")
print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
print(f"Macro-F1 : {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"Macro-P  : {precision_score(all_labels, all_preds, average='macro'):.4f}")
print(f"Macro-R  : {recall_score(all_labels, all_preds, average='macro'):.4f}")
print()
print(classification_report(all_labels, all_preds))

# --- Confusion Matrix ---
with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_,
            cmap="Blues", ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("CodeBERT Confusion Matrix")
plt.tight_layout(); plt.savefig("data/processed/confusion_matrix.png", dpi=150)
```

### 8.2 Results Table

| Model                     | Accuracy | Macro-F1 | Macro-P | Macro-R |
|---------------------------|----------|----------|---------|---------|
| TF-IDF + LR (baseline)    | [X.XXXX] | [X.XXXX] | [X.XXXX]| [X.XXXX]|
| CodeBERT (fine-tuned)     | [X.XXXX] | [X.XXXX] | [X.XXXX]| [X.XXXX]|
| Δ (CodeBERT − Baseline)   | [+X.XX]  | [+X.XX]  | [+X.XX] | [+X.XX] |

---

## 9. Error Analysis

```python
# Identify misclassified samples
test_df["pred"] = all_preds
errors = test_df[test_df["label"] != test_df["pred"]].copy()

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

errors["true_cwe"] = le.inverse_transform(errors["label"])
errors["pred_cwe"] = le.inverse_transform(errors["pred"])

# Most frequent error pairs
print(errors.groupby(["true_cwe", "pred_cwe"]).size().sort_values(ascending=False).head(10))

# Show a few examples
for _, row in errors.head(3).iterrows():
    print(f"\nTrue: {row['true_cwe']}  Predicted: {row['pred_cwe']}")
    print(row["func_before"][:300])
    print("---")
```

---

## 10. Report & Slides Compilation

### 10.1 Compile LaTeX Report

```bash
cd report/

# Download ACL 2023 style files first
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl.sty
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl_natbib.sty

# Compile (run twice for references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 10.2 View Slides

```bash
# Open in browser
open slides/slides.html          # macOS
xdg-open slides/slides.html      # Linux
start slides/slides.html         # Windows
```

### 10.3 Export Slides to PDF (optional)

```bash
# Using Chrome headless
google-chrome --headless --print-to-pdf=slides/slides.pdf \
  --print-to-pdf-no-header slides/slides.html
```

---

## 11. Submission Checklist

```
Deliverables
 [ ] report/main.tex compiled to PDF without errors
 [ ] All [X.XXXX] placeholders replaced with real numbers
 [ ] All [FILL IN] sections completed
 [ ] slides/slides.html opens in Chrome/Firefox without errors
 [ ] Slides use actual result values (not placeholders)
 [ ] Code is reproducible (fixed random seeds, requirements.txt present)
 [ ] data/processed/ contains train.csv, val.csv, test.csv
 [ ] Confusion matrix figure saved and referenced in report

Code Quality
 [ ] src/ modules importable (no circular imports)
 [ ] Training completes end-to-end with python -m src.train
 [ ] Evaluation script prints all four metrics

Repository
 [ ] .gitignore excludes .venv/, __pycache__/, *.pyc, data/raw/
 [ ] requirements.txt committed
 [ ] README references this guide
```
