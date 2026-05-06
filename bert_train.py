"""
BERT ファインチューニング（学習用）
- モデル: neuralmind/bert-base-portuguese-cased
- Kaggle Notebook（GPU ON, インターネットON）で実行
- 学習済みモデルを保存し、提出用Notebookで読み込む
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import mlflow

# ============================================================
# 設定
# ============================================================
EXP_NAME = "exp001"  # ★実験番号をここで指定
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-5
N_SPLITS = 5
SEED = 42
NUM_CLASSES = 7  # target: 0~6
OUTPUT_DIR = f"./results/{EXP_NAME}/bert_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# データ読み込み・前処理
# ============================================================
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

def clean_text(text):
    text = str(text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train['report'] = train['report'].fillna('').apply(clean_text)
test['report'] = test['report'].fillna('').apply(clean_text)

# ============================================================
# クラス重み計算（少数クラスを重視）
# ============================================================
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train['target'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

# ============================================================
# Dataset
# ============================================================
class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# ============================================================
# Focal Loss（クラス重み併用）
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # 正解クラスの予測確率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================
# 学習関数
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
    model.train()
    total_loss = 0
    loss_fn = FocalLoss(weight=class_weights, gamma=2.0)
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    preds = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return np.array(preds), np.array(labels_all)

def predict(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())
    return np.array(preds)

# ============================================================
# Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# MLflow 設定
# ============================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("SPR2026")
mlflow.start_run(run_name=EXP_NAME)
mlflow.log_params({
    "exp_name": EXP_NAME,
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "n_splits": N_SPLITS,
    "seed": SEED,
    "num_classes": NUM_CLASSES,
})

# ============================================================
# StratifiedKFold + 学習
# ============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds_all = np.zeros((N_SPLITS, len(test)), dtype=int)
scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train['target'])):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_SPLITS}")
    print(f"{'='*50}")

    # MLflow: Fold子Run開始
    with mlflow.start_run(run_name=f"{EXP_NAME}_fold{fold}", nested=True) as fold_run:
        mlflow.log_param("fold", fold)

        tr_texts = train['report'].iloc[tr_idx].values.tolist()
        va_texts = train['report'].iloc[va_idx].values.tolist()
        tr_labels = train['target'].iloc[tr_idx].values
        va_labels = train['target'].iloc[va_idx].values

        tr_dataset = ReportDataset(tr_texts, tr_labels, tokenizer, MAX_LEN)
        va_dataset = ReportDataset(va_texts, va_labels, tokenizer, MAX_LEN)

        tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        va_loader = DataLoader(va_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # モデル初期化
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(tr_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_f1 = 0
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, tr_loader, optimizer, scheduler, device, class_weights)
            va_preds, va_true = eval_epoch(model, va_loader, device)
            f1 = f1_score(va_true, va_preds, average='macro')
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val F1-macro: {f1:.4f}")

            # MLflow: Epoch メトリクス（Foldごとにstep=epochで記録）
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_f1_macro": f1,
            }, step=epoch)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"bert_fold{fold}.pth"))
                print(f"  -> Best model saved (F1: {best_f1:.4f})")

        # ベストモデルで検証・テスト予測
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"bert_fold{fold}.pth")))
        va_preds, va_true = eval_epoch(model, va_loader, device)
        oof_preds[va_idx] = va_preds
        f1 = f1_score(va_true, va_preds, average='macro')
        print(f"\nFold {fold+1} Best F1-macro: {f1:.4f}")
        print(classification_report(va_true, va_preds, digits=4))
        scores.append(f1)

        # MLflow: Fold Best F1
        mlflow.log_metric("best_f1_macro", f1)

    # テスト予測
    test_dataset = TestDataset(test['report'].values.tolist(), tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_preds_all[fold] = predict(model, test_loader, device)

    # メモリ解放
    del model
    torch.cuda.empty_cache()

# ============================================================
# CV結果
# ============================================================
cv_mean = np.mean(scores)
print(f"\nCV mean F1-macro: {cv_mean:.4f}")

# MLflow: CV mean F1
mlflow.log_metric("cv_mean_f1_macro", cv_mean)

# ============================================================
# テスト予測（多数決）
# ============================================================
from scipy.stats import mode
test_pred_final = mode(test_preds_all, axis=0)[0].flatten().astype(int)

# ============================================================
# submission作成
# ============================================================
submission = pd.DataFrame({'ID': test['ID'], 'target': test_pred_final})
submission.to_csv(os.path.join(f'./results/{EXP_NAME}', 'submission_bert.csv'), index=False)
print(f"submission_bert.csv saved to results/{EXP_NAME}/!")

# ============================================================
# Tokenizer・Config保存（提出用Notebookで読み込むため）
# ============================================================
tokenizer.save_pretrained(OUTPUT_DIR)
# config.jsonも保存（推論時にfrom_configで使う）
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
config.save_pretrained(OUTPUT_DIR)
print(f"Tokenizer & Config saved to {OUTPUT_DIR}")

# ============================================================
# MLflow: アーティファクト保存 & run終了
# ============================================================
submission_path = os.path.join(f'./results/{EXP_NAME}', 'submission_bert.csv')
mlflow.log_artifact(submission_path)
mlflow.end_run()

print("Done!")