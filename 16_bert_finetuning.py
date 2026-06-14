"""
German BERT/RoBERTa fine-tuning — one model per task, run for each embedding model.

Tasks:
  interaction  — 5-class (CrossEntropyLoss)   — pair input [CLS] parent [SEP] child
  stance       — 3-class (CrossEntropyLoss)   — child-only input
  techniques   — 14-label multi-label (BCEWithLogitsLoss, threshold 0.5) — child-only

Embedding models (both are fine-tuned independently):
  gbert    — deepset/gbert-base        (German BERT)
  xlmr     — FacebookAI/xlm-roberta-base (XLM-RoBERTa)

Training: AdamW + linear warmup/decay, early stopping (patience=2 on val macro-F1).
Checkpoints: 16_bert_results/checkpoints/<model_key>/<task>/best_model/
Predictions: 16_bert_results/predictions/<model_key>_<task>.jsonl
Metrics:     16_bert_results/metrics.json  
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    XLMRobertaForSequenceClassification,
    BertTokenizer,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Explicit tokenizer/model classes 
TOKENIZER_CLASSES = {
    "gbert": BertTokenizer,
    "xlmr":  XLMRobertaTokenizer,
}
SEQ_CLS_CLASSES = {
    "gbert": BertForSequenceClassification,
    "xlmr":  XLMRobertaForSequenceClassification,
}
from torch.optim import AdamW
from sklearn.metrics import (f1_score, cohen_kappa_score, hamming_loss,
                             accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import pearsonr

from utils.jsonl_io import save_jsonl

load_dotenv()

BERT_BATCH    = int(os.getenv("BERT_BATCH_SIZE", 16))

# Both models are fine-tuned independently — one checkpoint set per model per task
BERT_MODELS = {
    "gbert": "deepset/gbert-base",
    "xlmr":  "FacebookAI/xlm-roberta-base",
}
BERT_EPOCHS   = int(os.getenv("BERT_EPOCHS", 5))
BERT_LR       = float(os.getenv("BERT_LR", 2e-5))
BERT_MAX_LEN  = int(os.getenv("BERT_MAX_LEN", 256))
RANDOM_SEED   = int(os.getenv("RANDOM_SEED", 42))

FEAT_DIR = Path("14_features")
OUT_DIR  = Path("16_bert_results")
PRED_DIR = OUT_DIR / "predictions"
CKPT_DIR = OUT_DIR / "checkpoints"
for d in (OUT_DIR, PRED_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)
# Per-model sub-dirs are created dynamically in finetune_task()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

VALID_TECHNIQUES = [
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice',
    'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy',
    'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
    'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling',
    'Repetition', 'Slogans/Thought-terminating_Cliches',
    'Whataboutism,Straw_Men', 'Appeal_to_Time',
]

SCORE_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
CLASS_TO_SCORE = {v: k for k, v in SCORE_TO_CLASS.items()}


# Dataset

class PairDataset(Dataset):
    """For interaction task: tokenize parent + child as sentence pair."""
    def __init__(self, records, tokenizer, max_len):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["ParentText"], r["ChildText"],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(r["label"], dtype=torch.long),
            "pair_id":        r["PairID"],
        }
        # token_type_ids: present for BERT (distinguishes segment A/B); absent for XLM-R
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


class ChildDataset(Dataset):
    """For stance / techniques: tokenize child text only."""
    def __init__(self, records, tokenizer, max_len, task):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.task      = task

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["ChildText"],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        if self.task == "techniques":
            label = torch.tensor(r["label"], dtype=torch.float)
        else:
            label = torch.tensor(r["label"], dtype=torch.long)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          label,
            "pair_id":        r["PairID"],
        }


# Data loading
def prepare_records(df, raw_pairs_map, split, task, mlb=None):
    """Build labelled record list for a split."""
    sub = df[df["split"] == split]
    records = []
    for _, row in sub.iterrows():
        pid = row["PairID"]
        raw = raw_pairs_map.get(pid, {})
        if task == "interaction":
            if pd.isna(row["InteractionScore"]):
                continue
            s = float(row["InteractionScore"])
            label = SCORE_TO_CLASS[min(SCORE_TO_CLASS, key=lambda k: abs(k - s))]
        elif task == "stance":
            if pd.isna(row["StanceLabel"]):
                continue
            label = int(row["StanceLabel"])
        else:  # techniques
            tech_list = json.loads(row["Techniques"]) if isinstance(row["Techniques"], str) else []
            label = mlb.transform([tech_list])[0].tolist()
        records.append({
            "PairID":      pid,
            "ParentText":  raw.get("ParentText", ""),
            "ChildText":   raw.get("ChildText", ""),
            "label":       label,
        })
    return records


# Training utilities
def eval_epoch(model, loader, task, device):
    model.eval()
    all_preds, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            logits = model(**kwargs).logits
            if task == "techniques":
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                labels = batch["label"].numpy().astype(int)
            else:
                preds  = logits.argmax(dim=-1).cpu().numpy()
                labels = batch["label"].numpy()
            all_preds.append(preds)
            all_labels.append(labels)
            all_pids.extend(batch["pair_id"])

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if task == "techniques":
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return macro_f1, y_pred, y_true, all_pids


def compute_metrics(task, y_true, y_pred):
    if task == "interaction":
        scores_true = np.array([CLASS_TO_SCORE[c] for c in y_true])
        scores_pred = np.array([CLASS_TO_SCORE[c] for c in y_pred])
        return {
            "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro",
                                                    zero_division=0)), 4),
            "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(precision_score(y_true, y_pred, average="macro",
                                                           zero_division=0)), 4),
            "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro",
                                                        zero_division=0)), 4),
            "kappa_quadratic": round(float(cohen_kappa_score(y_true, y_pred,
                                                              weights="quadratic")), 4),
            "mae":             round(float(np.mean(np.abs(scores_true - scores_pred))), 4),
            "pearson_r":       round(float(pearsonr(scores_true, scores_pred)[0])
                               if len(set(scores_pred.tolist())) > 1 else 0.0, 4),
            "per_class_f1":    [round(v, 4) for v in
                                f1_score(y_true, y_pred, average=None,
                                         zero_division=0).tolist()],
        }
    elif task == "stance":
        per = f1_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
        return {
            "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro",
                                                    zero_division=0)), 4),
            "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(precision_score(y_true, y_pred, average="macro",
                                                           zero_division=0)), 4),
            "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro",
                                                        zero_division=0)), 4),
            "kappa":           round(float(cohen_kappa_score(y_true, y_pred)), 4),
            "per_class_f1":    {"Against": round(float(per[0]),4),
                                "Neutral":  round(float(per[1]),4),
                                "Support":  round(float(per[2]),4)},
        }
    else:  # techniques
        return {
            "macro_f1":             round(float(f1_score(y_true, y_pred, average="macro",
                                                         zero_division=0)), 4),
            "exact_match_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision":      round(float(precision_score(y_true, y_pred, average="macro",
                                                                zero_division=0)), 4),
            "macro_recall":         round(float(recall_score(y_true, y_pred, average="macro",
                                                             zero_division=0)), 4),
            "hamming_loss":         round(float(hamming_loss(y_true, y_pred)), 4),
            "per_label_f1":         {t: round(float(f), 4) for t, f in
                                     zip(VALID_TECHNIQUES,
                                         f1_score(y_true, y_pred, average=None,
                                                  zero_division=0))},
        }


# Fine-tuning loop
def finetune_task(task, df, raw_pairs_map, tokenizer, all_metrics,
                  model_key: str = "gbert",
                  model_name: str = "deepset/gbert-base"):
    """
    Fine-tune one task for one embedding model (model_key).
    Checkpoints: CKPT_DIR / model_key / task / best_model
    Predictions: PRED_DIR / f"{model_key}_{task}.jsonl"
    Metrics stored under all_metrics[model_key][task].
    """
    print(f"\n{'='*50}\nFine-tuning: {task}  [{model_key}]\n{'='*50}")
    ckpt_path = CKPT_DIR / model_key / task / "best_model"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    pred_file = PRED_DIR / f"{model_key}_{task}.jsonl"

    # Label count
    num_labels = {"interaction": 5, "stance": 3, "techniques": 14}[task]

    # MLB for techniques
    mlb = None
    if task == "techniques":
        mlb = MultiLabelBinarizer(classes=VALID_TECHNIQUES)
        mlb.fit([[]])  # initialize with empty fit; we use known classes

    # Build datasets
    train_recs = prepare_records(df, raw_pairs_map, "train", task, mlb)
    val_recs   = prepare_records(df, raw_pairs_map, "val",   task, mlb)
    test_recs  = prepare_records(df, raw_pairs_map, "test",  task, mlb)

    if not train_recs:
        print(f"  No training data for {task}, skipping.")
        return

    DatasetCls = PairDataset if task == "interaction" else ChildDataset
    if task == "interaction":
        train_ds = DatasetCls(train_recs, tokenizer, BERT_MAX_LEN)
        val_ds   = DatasetCls(val_recs,   tokenizer, BERT_MAX_LEN)
        test_ds  = DatasetCls(test_recs,  tokenizer, BERT_MAX_LEN)
    else:
        train_ds = DatasetCls(train_recs, tokenizer, BERT_MAX_LEN, task)
        val_ds   = DatasetCls(val_recs,   tokenizer, BERT_MAX_LEN, task)
        test_ds  = DatasetCls(test_recs,  tokenizer, BERT_MAX_LEN, task)

    train_loader = DataLoader(train_ds, batch_size=BERT_BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BERT_BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BERT_BATCH, shuffle=False, num_workers=0)

    # Load model (from checkpoint if exists, else fresh from HF hub)
    # Checkpoints saved by save_pretrained() have model_type in config.json.
    # Initial HF hub load uses explicit class
    # missing model_type.
    if ckpt_path.exists():
        print(f"  Loading existing checkpoint: {ckpt_path}")
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
    else:
        cls = SEQ_CLS_CLASSES[model_key]
        model = cls.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
        )
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
    total_steps = len(train_loader) * BERT_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if task == "techniques":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_f1 = 0.0
    patience_counter = 0
    PATIENCE = 2

    if not ckpt_path.exists():
        for epoch in range(1, BERT_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{BERT_EPOCHS}",
                        unit="batch", leave=False)
            for batch in pbar:
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                kwargs = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
                logits = model(**kwargs).logits
                loss   = loss_fn(logits, label)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.close()

            val_f1, _, _, _ = eval_epoch(model, val_loader, task, DEVICE)
            print(f"  Epoch {epoch}/{BERT_EPOCHS}  loss={total_loss/len(train_loader):.4f}"
                  f"  val_macro_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                print(f"    -> Best checkpoint saved (val_f1={val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Reload best checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path)).to(DEVICE)

    # Evaluate on test set
    _, y_pred, y_true, pair_ids = eval_epoch(model, test_loader, task, DEVICE)
    metrics = compute_metrics(task, y_true, y_pred)
    all_metrics.setdefault(model_key, {})[task] = metrics

    # Print full metrics
    print(f"\n  -- Test results [{model_key} / {task}] --")
    if task == "interaction":
        print(f"    macro_f1        : {metrics['macro_f1']}")
        print(f"    accuracy        : {metrics['accuracy']}")
        print(f"    macro_precision : {metrics['macro_precision']}")
        print(f"    macro_recall    : {metrics['macro_recall']}")
        print(f"    kappa_quadratic : {metrics['kappa_quadratic']}")
        print(f"    mae             : {metrics['mae']}")
        print(f"    pearson_r       : {metrics['pearson_r']}")
        labels = ["Strong-Neg", "Weak-Neg", "Neutral", "Weak-Pos", "Strong-Pos"]
        for lbl, f1 in zip(labels, metrics["per_class_f1"]):
            print(f"      {lbl:<12}: F1={f1}")
    elif task == "stance":
        print(f"    macro_f1        : {metrics['macro_f1']}")
        print(f"    accuracy        : {metrics['accuracy']}")
        print(f"    macro_precision : {metrics['macro_precision']}")
        print(f"    macro_recall    : {metrics['macro_recall']}")
        print(f"    kappa           : {metrics['kappa']}")
        for cls, f1 in metrics["per_class_f1"].items():
            print(f"      {cls:<8}: F1={f1}")
    else:  # techniques
        print(f"    macro_f1             : {metrics['macro_f1']}")
        print(f"    exact_match_accuracy : {metrics['exact_match_accuracy']}")
        print(f"    macro_precision      : {metrics['macro_precision']}")
        print(f"    macro_recall         : {metrics['macro_recall']}")
        print(f"    hamming_loss         : {metrics['hamming_loss']}")
        for lbl, f1 in metrics["per_label_f1"].items():
            print(f"      {lbl:<40}: F1={f1}")

    # Save predictions
    if not pred_file.exists():
        preds = [{"PairID": pid, "y_true": yt, "y_pred": yp}
                 for pid, yt, yp in zip(pair_ids,
                                        y_true.tolist() if hasattr(y_true, "tolist") else list(y_true),
                                        y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred))]
        save_jsonl(preds, str(pred_file))


# Main
def main():
    df = pd.read_parquet(FEAT_DIR / "features.parquet")

    # Restrict to oracle-valid pairs — same rows as steps 15, 17, 19
    df = df[df["StanceLabel"].notna()].copy()
    print(f"Oracle-filtered dataset: {len(df)} pairs "
          f"(train={len(df[df['split']=='train'])}, "
          f"val={len(df[df['split']=='val'])}, "
          f"test={len(df[df['split']=='test'])})")

    from utils.jsonl_io import load_jsonl
    # Load LLM annotations first, then gold overwrites shared PairIDs
    raw_pairs_map = {}
    for path in [
        Path("10_llm_annotations/llm_annotated_subset.jsonl"),
        Path("11_llm_annotations/llm_annotated_rest.jsonl"),
        Path("09_manual_annotations/gold_standard.jsonl"),   
    ]:
        if path.exists():
            for r in load_jsonl(str(path)):
                raw_pairs_map[r["PairID"]] = r
        else:
            print(f"  Warning: {path} not found, skipping.")

    print(f"Loaded {len(raw_pairs_map)} raw pair records.")

    # Load existing metrics so the run is resumable
    metrics_file = OUT_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    for model_key, model_name in BERT_MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Embedding model: {model_key}  ({model_name})")
        print(f"{'#'*60}")
        tokenizer = TOKENIZER_CLASSES[model_key].from_pretrained(model_name)

        for task in ["interaction", "stance", "techniques"]:
            # Skip if already computed and saved
            if model_key in all_metrics and task in all_metrics[model_key]:
                print(f"\n[{model_key}/{task}] Already in metrics.json, skipping.")
                continue
            finetune_task(task, df, raw_pairs_map, tokenizer, all_metrics,
                          model_key=model_key, model_name=model_name)
            # Save incrementally after each task
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        # Free memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nMetrics saved -> {metrics_file}")
    print("\n" + "="*60)
    print("Fine-tuning complete — full results summary")
    print("="*60)
    for mk, tasks in all_metrics.items():
        print(f"\n  [{mk}]")
        for task, m in tasks.items():
            print(f"    {task}:")
            skip = {"per_class_f1", "per_label_f1"}
            for k, v in m.items():
                if k not in skip:
                    print(f"      {k:<22}: {v}")


if __name__ == "__main__":
    main()
