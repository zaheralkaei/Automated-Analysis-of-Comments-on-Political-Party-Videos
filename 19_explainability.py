"""
Explainability and error analysis

A. SHAP for shallow learners (TreeExplainer for RF/XGBoost, LinearExplainer for LR/SVM)
   Scope: one best feature set per task, chosen by macro-F1 across all models/feature
   sets in step 15's metrics.json.
   Output: mean |SHAP| per feature (JSON) + beeswarm plot (PNG).

B. Permutation importance for shallow learners
   sklearn.inspection.permutation_importance on test set, n_repeats=10.
   Output: ranked feature importance with std (JSON).

C. Token attributions for fine-tuned BERT (Captum LayerIntegratedGradients)
   Attributions computed for the full test set per model/task.
   Output: token attributions JSON (all test pairs) + HTML visualisation for the
   first 10 examples.

D. Error analysis
   Per task: misclassified examples, confusion matrix, interaction-quality-specific
   agreement/disagreement confusion analysis, stance Against/Support confusion.
   Output: error_analysis.json.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

BERT_MAX_LEN  = int(os.getenv("BERT_MAX_LEN", 256))
RANDOM_SEED   = int(os.getenv("RANDOM_SEED", 42))

BERT_MODELS = {
    "gbert": "deepset/gbert-base",
    "xlmr":  "FacebookAI/xlm-roberta-base",
}

# tokenizer and model classes 
from transformers import BertTokenizer, XLMRobertaTokenizer, BertModel, XLMRobertaModel
TOKENIZER_CLASSES = {
    "gbert": BertTokenizer,
    "xlmr":  XLMRobertaTokenizer,
}

# Map internal model key to npz file suffix 
FILE_KEY_MAP = {"gbert": "gbert", "xlmr": "xlmr"}

FEAT_DIR      = Path("14_features")
SHALLOW_DIR   = Path("15_shallow_results")
BERT_DIR      = Path("16_bert_results")
OUT_DIR       = Path("19_explainability")
SHAP_SHALLOW  = OUT_DIR / "shap_shallow"
SHAP_BERT_DIR = OUT_DIR / "shap_bert"
for d in (OUT_DIR, SHAP_SHALLOW, SHAP_BERT_DIR):
    d.mkdir(parents=True, exist_ok=True)

THEORY_COLS = [
    
    "child_len_words", "parent_len_words", "len_ratio", "child_ttr",
    "vocab_overlap", "child_excl_count", "child_quest_count", "child_caps_ratio",
    "child_neg_count", "child_intensifier_count", "mention_count",
    "child_political_count", "has_url", "child_first_person_ratio", "child_avg_word_len",
    "sent_pos", "sent_neg", "sent_neu", "sent_compound",
    # structural
    "word_count_diff", "parent_caps_ratio",
]  # 21 features — B block

# Stance block (S) for SHAP on B+S feature sets
STANCE_COLS = ["child_stance", "parent_stance", "same_stance", "abs_stance_diff"]

# Technique aggregate for SHAP on B+S+T feature sets
TECHNIQUE_AGGREGATE_COL = ["technique_count"]

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
SCORE_ENUM     = [-1.0, -0.5, 0.0, 0.5, 1.0]

random.seed(RANDOM_SEED)


# Data loaders
def load_features():
    df = pd.read_parquet(FEAT_DIR / "features.parquet")

    # Restrict to oracle-valid pairs
    df = df[df["StanceLabel"].notna()].copy()

    emb_maps = {}
    for model_key in ("gbert", "xlmr"):
        file_key = FILE_KEY_MAP[model_key]
        for split_type in ("pair", "child"):
            fname = FEAT_DIR / f"bert_embeddings_{split_type}_{file_key}.npz"
            key   = f"{split_type}_{model_key}"
            if fname.exists():
                npz = np.load(fname, allow_pickle=True, mmap_mode='r')
                arr = np.array(npz["embeddings"])
                emb_maps[key] = {pid: arr[i]
                                 for i, pid in enumerate(npz["pair_ids"])}
            else:
                emb_maps[key] = {}
    return df, emb_maps


def get_test_X(df, task, feat_set, emb_maps):
    """
    Reconstruct the test feature matrix for a given feat_set — mirrors step 15's get_Xy().
    Accepts emb_maps dict with keys pair_gbert, child_gbert, pair_xlmr, child_xlmr.

    Named-feature sets (fully supported for SHAP + permutation importance):
      B         — 21 linguistic features
      B+S       — 25 features (B + 4 stance)
      B+S+T     — 40 features (B+S + 15 technique)
      bert      — 768-dim embedding
      B+bert    — 789-dim (B + embedding)

    Unsupported (return None):
      _pca sets       — PCA pipeline cannot be unwrapped by SHAP
      B+S+T+E1/E2     — 808-dim embeddings not individually nameable
    """
    from sklearn.preprocessing import MultiLabelBinarizer as _MLB
    sub = df[df["split"] == "test"].copy()
    if task == "interaction":
        sub = sub[sub["InteractionScore"].notna()]
    elif task == "stance":
        sub = sub[sub["StanceLabel"].notna()]

    pids     = sub["PairID"].tolist()
    theory_X = sub[THEORY_COLS].fillna(0).values.astype(float)

    # Determine embedding model from feature set name
    base_fs   = feat_set.replace("_pca", "")
    emb_model = "xlmr" if ("xlmr" in base_fs or "groberta" in base_fs) else "gbert"
    emb_key   = f"pair_{emb_model}" if task == "interaction" else f"child_{emb_model}"
    bert_X    = np.vstack([emb_maps[emb_key].get(pid, np.zeros(768)) for pid in pids])

    norm_fs = base_fs.replace("xlmr", "bert").replace("groberta", "bert")

    if norm_fs == "B":
        X = theory_X
        feat_names = list(THEORY_COLS)

    elif norm_fs == "B+S":
        stance_X = sub[STANCE_COLS].fillna(0).values.astype(float)
        X = np.hstack([theory_X, stance_X])
        feat_names = list(THEORY_COLS) + list(STANCE_COLS)

    elif norm_fs == "B+S+T":
        stance_X = sub[STANCE_COLS].fillna(0).values.astype(float)
        counts   = sub[TECHNIQUE_AGGREGATE_COL].fillna(0).values.astype(float)
        mlb      = _MLB(classes=VALID_TECHNIQUES)
        tech_bin = mlb.fit_transform(
            sub["Techniques"].apply(json.loads).tolist()
        ).astype(float)
        X = np.hstack([theory_X, stance_X, counts, tech_bin])
        feat_names = (list(THEORY_COLS) + list(STANCE_COLS)
                      + list(TECHNIQUE_AGGREGATE_COL) + list(VALID_TECHNIQUES))

    elif norm_fs == "bert":
        X = bert_X
        feat_names = [f"{emb_model}_{i}" for i in range(768)]

    elif norm_fs == "B+bert":
        X = np.hstack([theory_X, bert_X])
        feat_names = list(THEORY_COLS) + [f"{emb_model}_{i}" for i in range(768)]

    else:
        # Not suitable for named-feature SHAP:
        #   _pca sets      — PCA pipeline cannot be unwrapped by SHAP
        #   B+S+T+E1/E2    — 808-dim embeddings are not individually nameable
        return None, None, pids, []

    if task == "interaction":
        y = sub["InteractionScore"].apply(
            lambda s: SCORE_TO_CLASS[min(SCORE_TO_CLASS, key=lambda k: abs(k - float(s)))]
        ).values
    elif task == "stance":
        y = sub["StanceLabel"].astype(int).values
    else:
        mlb = _MLB(classes=VALID_TECHNIQUES)
        y = mlb.fit_transform(sub["Techniques"].apply(json.loads).tolist())

    return X, y, pids, feat_names


def find_best_feat_set(task: str) -> str:
    """Return best feature set suitable for named-feature SHAP.
    Eligible sets: B, B+S, B+S+T, bert, B+bert — all have named features.
    Skipped sets:
      _pca       — saved as sklearn Pipeline; SHAP cannot unwrap PCA transform
      B+S+T+E1/E2 — 808-dim embeddings are not individually nameable
    Falls back to 'B+bert' if no metrics file exists, or 'B' if nothing qualifies.
    """
    metrics_file = SHALLOW_DIR / "metrics.json"
    if not metrics_file.exists():
        return "B+bert"
    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)
    SKIP = ("_pca", "B+S+T+E")  # skip PCA (unwrappable) and E1/E2 (768-dim, not nameable)
    best_f1, best_feat = 0.0, "B"
    for model_name, feat_dict in metrics.get(task, {}).items():
        for feat_set, m in feat_dict.items():
            if any(s in feat_set for s in SKIP):
                continue
            if m.get("macro_f1", 0) > best_f1:
                best_f1, best_feat = m["macro_f1"], feat_set
    return best_feat


# A. SHAP for shallow learners + B. Permutation importance for shallow learners
def shap_shallow(df, emb_maps):
    import shap
    print("\n=== A. SHAP for shallow learners ===")

    tasks  = ["interaction", "stance", "techniques"]
    models = ["LR", "RF", "SVM", "XGB"]
    perm_results = {}

    # Human-readable class labels for per-class SHAP JSON output
    CLASS_LABELS = {
        "interaction": {0: "DD(-1.0)", 1: "DA(-0.5)", 2: "N(0.0)",
                        3: "CA(+0.5)", 4: "CD(+1.0)"},
        "stance":      {0: "Against", 1: "Neutral", 2: "Support"},
    }

    for task in tasks:
        feat_set = find_best_feat_set(task)
        print(f"\n  Task: {task}  best feature set: {feat_set}")
        perm_results[task] = {}
        X_test, y_test, pids_test, feat_names = get_test_X(df, task, feat_set, emb_maps)

        if X_test is None:
            print(f"    Skipping {task} — feature set not suitable for SHAP.")
            continue

        # X_test is kept raw here; each model applies its own paired scaler below.
        # limit to theory features (interpretable by name).
        has_theory = feat_set.startswith("B")   # "B", "B+bert", "B+S", etc.
        n_display  = len(THEORY_COLS) if has_theory else min(16, len(feat_names))
        shap_feat_names = feat_names[:n_display]

        for model_name in models:
            model_file  = SHALLOW_DIR / "models" / f"{task}_{model_name}_{feat_set}.joblib"
            scaler_file = SHALLOW_DIR / "models" / f"{task}_{model_name}_{feat_set}_scaler.joblib"
            if not model_file.exists():
                print(f"    Model not found: {model_file.name}")
                continue

            clf = joblib.load(model_file)

            # Apply the model's paired scaler so SHAP and permutation importance
            # receive the same scaled input that the model was trained on.
            if scaler_file.exists():
                _scaler = joblib.load(scaler_file)
                X_test_scaled = _scaler.transform(X_test)
            else:
                X_test_scaled = X_test

            print(f"    {model_name}...", end=" ", flush=True)

            shap_out_file    = SHAP_SHALLOW / f"{task}_{model_name}_mean_shap.json"
            signed_out_file  = SHAP_SHALLOW / f"{task}_{model_name}_signed_shap.json"
            beeswarm_file    = SHAP_SHALLOW / f"{task}_{model_name}_beeswarm.png"

            try:
                if task == "techniques" and hasattr(clf, "estimators_"):
                    # OVR wrapper: only linear models (LR/SVM) support per-estimator
                    # SHAP reliably. Tree-based OVR triggers numpy bool ambiguity
                    # inside TreeExplainer — use permutation importance instead.
                    if model_name in ("RF", "XGB"):
                        raise ValueError(
                            f"OVR TreeExplainer not supported for {model_name} "
                            f"— see permutation_importance.json for feature ranking"
                        )
                    sv_list = []
                    for est in clf.estimators_:
                        exp  = shap.LinearExplainer(est, X_test_scaled)
                        sv_i = exp.shap_values(X_test_scaled)
                        if isinstance(sv_i, list):
                            sv_i = sv_i[1]  # positive class for binary OVR
                        sv_list.append(sv_i)
                    sv_3d = np.stack(sv_list, axis=-1)  # (n, feats, n_labels)
                else:
                    if model_name in ("RF", "XGB"):
                        explainer = shap.TreeExplainer(clf)
                        sv = explainer.shap_values(X_test_scaled)
                    else:  # LR, SVM
                        explainer = shap.LinearExplainer(clf, X_test_scaled)
                        sv = explainer.shap_values(X_test_scaled)

                    # Normalise SHAP output to consistent 3D array:
                    # (n_samples, n_features, n_classes)
                    if isinstance(sv, list):
                        sv_3d = np.stack(sv, axis=-1)       # list of 2D >>> 3D
                    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                        sv_3d = sv                           # already 3D
                    else:
                        sv_3d = sv[:, :, np.newaxis]        # binary/single >>> 3D

                # Limit display columns to interpretable features
                sv_3d_disp = sv_3d[:, :n_display, :]
                X_disp     = X_test_scaled[:, :n_display]

                # Magnitude (mean |SHAP|, ranked, backward-compatible) 
                mean_abs = np.abs(sv_3d_disp).mean(axis=(0, 2))
                importance_dict = {
                    fname: round(float(val), 6)
                    for fname, val in sorted(
                        zip(shap_feat_names, mean_abs),
                        key=lambda x: -x[1]
                    )
                }
                with open(shap_out_file, "w", encoding="utf-8") as fout:
                    json.dump(importance_dict, fout, indent=2)

                # Directional: mean signed SHAP (collapsed and per-class) 
                n_classes = sv_3d_disp.shape[2]
                labels    = CLASS_LABELS.get(task, {c: str(c) for c in range(n_classes)})

                # Collapsed: mean across samples AND classes — overall direction
                mean_signed = sv_3d_disp.mean(axis=(0, 2))

                # Per-class: mean across samples only -> shape (n_features, n_classes)
                per_class   = sv_3d_disp.mean(axis=0)

                signed_dict = {
                    "mean_signed_shap": {
                        fname: round(float(val), 6)
                        for fname, val in sorted(
                            zip(shap_feat_names, mean_signed),
                            key=lambda x: -abs(x[1])
                        )
                    },
                    "per_class_mean_shap": {
                        labels.get(c, str(c)): {
                            fname: round(float(per_class[fi, c]), 6)
                            for fi, fname in enumerate(shap_feat_names)
                        }
                        for c in range(n_classes)
                    },
                }
                with open(signed_out_file, "w", encoding="utf-8") as fout:
                    json.dump(signed_dict, fout, indent=2)

                # Beeswarm: signed SHAP, top-15 features by mean |SHAP| 
                # Collapse multi-class to single 2D matrix by averaging across
                # classes — each dot is one test instance, coloured by feature value.
                sv_collapsed = sv_3d_disp.mean(axis=2)   # (n_samples, n_display)
                top_n  = min(15, len(shap_feat_names))
                order  = np.argsort(-mean_abs)[:top_n]
                shap.summary_plot(
                    sv_collapsed[:, order],
                    X_disp[:, order],
                    feature_names=[shap_feat_names[i] for i in order],
                    plot_type="dot",
                    show=False,
                    max_display=top_n,
                )
                plt.title(f"SHAP beeswarm — {task} / {model_name} / {feat_set}")
                plt.tight_layout()
                plt.savefig(str(beeswarm_file), dpi=120, bbox_inches="tight")
                plt.close("all")
                print("done")

            except Exception as e:
                print(f"SHAP error: {e}")

            # Permutation importance
            try:
                if task == "techniques":
                    # OVR multi-label: use a callable scorer that handles 2D targets
                    from sklearn.metrics import f1_score as _f1
                    def _ml_f1(est, X, y):
                        return float(_f1(y, est.predict(X), average="macro",
                                        zero_division=0))
                    perm_scoring = _ml_f1
                else:
                    perm_scoring = "f1_macro"
                # n_jobs=1 for techniques (14-label OVR models exhaust Windows resources
                # with full parallelism); n_jobs=-1 is fine for single-output tasks.
                _n_jobs = 1 if task == "techniques" else -1
                result = permutation_importance(
                    clf, X_test_scaled, y_test,
                    n_repeats=10, random_state=RANDOM_SEED, scoring=perm_scoring,
                    n_jobs=_n_jobs,
                )
                perm_results[task][model_name] = [
                    {"feature": fname, "importance": round(float(imp), 6),
                     "std": round(float(std), 6)}
                    for fname, imp, std in sorted(
                        zip(shap_feat_names, result.importances_mean, result.importances_std),
                        key=lambda x: -x[1]
                    )
                ]
            except Exception as e:
                print(f"    Permutation error ({model_name}): {e}")

    # Save permutation importance
    perm_file = OUT_DIR / "permutation_importance.json"
    with open(perm_file, "w", encoding="utf-8") as f:
        json.dump(perm_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Permutation importance -> {perm_file}")


# C. BERT token attributions via captum LayerIntegratedGradients
def _make_html(tokens: list, scores: list, pid: str, task: str, pred_class: int) -> str:
    """Generate a simple HTML visualization of token attributions."""
    max_abs = max(abs(s) for s in scores) or 1.0
    spans = []
    for tok, score in zip(tokens, scores):
        norm  = score / max_abs
        r = int(255 * max(0, -norm))   # red for negative
        g = int(255 * max(0,  norm))   # green for positive
        b = 0
        alpha = abs(norm) * 0.8
        style = (f"background:rgba({r},{g},{b},{alpha:.2f});"
                 f"padding:1px 2px;border-radius:2px;margin:1px;display:inline-block;")
        tok_html = tok.replace("&", "&amp;").replace("<", "&lt;")
        spans.append(f'<span style="{style}" title="{score:.4f}">{tok_html}</span>')
    body = " ".join(spans)
    return (f"<html><head><meta charset='utf-8'></head><body>"
            f"<h3>PairID: {pid} | Task: {task} | Predicted class: {pred_class}</h3>"
            f"<p>{body}</p></body></html>")


def shap_bert(df, raw_pairs_map):
    print("\nC. BERT token attributions (captum LayerIntegratedGradients)")

    try:
        from captum.attr import LayerIntegratedGradients
        from transformers import AutoModelForSequenceClassification
        import torch
    except ImportError as e:
        print(f"  captum or torch not available: {e}. Skipping BERT attributions.")
        return

    N_HTML    = 10    # HTML visualisations saved per task/model
    N_STEPS   = 50   # IG integration steps
    MAX_LEN   = BERT_MAX_LEN

    for model_key in BERT_MODELS:
        print(f"\n  Model: {model_key}")
        for task in ["interaction", "stance", "techniques"]:
            ckpt = BERT_DIR / "checkpoints" / model_key / task / "best_model"
            if not ckpt.exists():
                print(f"    No checkpoint for {model_key}/{task}. Skipping.")
                continue

            # Resumability: skip if attribution file already complete
            attr_file = SHAP_BERT_DIR / f"{model_key}_{task}_attributions.json"
            if attr_file.exists():
                try:
                    existing = json.load(open(attr_file, encoding="utf-8"))
                    # Count expected test pairs (without loading model)
                    _sub = df[df["split"] == "test"].copy()
                    if task == "interaction":
                        _sub = _sub[_sub["InteractionScore"].notna()]
                    elif task == "stance":
                        _sub = _sub[_sub["StanceLabel"].notna()]
                    if len(existing) >= len(_sub):
                        print(f"    Task: {task}  already complete ({len(existing)} pairs) — skipping.")
                        continue
                    print(f"    Task: {task}  partial ({len(existing)}/{len(_sub)}) — recomputing.")
                except Exception:
                    pass  # corrupted file then recompute


            print(f"    Task: {task}")
            tok_cls   = TOKENIZER_CLASSES[model_key]
            tokenizer = tok_cls.from_pretrained(str(ckpt))
            model     = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
            model.eval()

            # Resolve embedding layer (works for both BERT and XLM-RoBERTa)
            base_model = (getattr(model, "bert", None)
                          or getattr(model, "roberta", None)
                          or model.base_model)
            emb_layer = base_model.embeddings

            def forward_func(input_ids, attention_mask, token_type_ids=None):
                kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids
                return model(**kwargs).logits

            lig = LayerIntegratedGradients(forward_func, emb_layer)

            # Full test set
            test_df = df[df["split"] == "test"].copy()
            if task == "interaction":
                test_df = test_df[test_df["InteractionScore"].notna()]
            elif task == "stance":
                test_df = test_df[test_df["StanceLabel"].notna()]
            sample_ids = test_df["PairID"].tolist()

            attr_results  = {}
            html_examples = []

            n_total = len(sample_ids)
            print(f"      Running LIG on {n_total} test pairs ...", flush=True)
            for i, pid in enumerate(sample_ids):
                if i > 0 and i % 100 == 0:
                    print(f"      {i}/{n_total} pairs processed "
                          f"({len(attr_results)} attributed so far)", flush=True)
                raw         = raw_pairs_map.get(pid, {})
                child_text  = raw.get("ChildText", "")  or ""
                parent_text = raw.get("ParentText", "") or ""
                if not child_text:
                    continue

                try:
                    if task == "interaction":
                        enc = tokenizer(parent_text, child_text,
                                        return_tensors="pt", max_length=MAX_LEN,
                                        truncation=True, padding=True)
                    else:
                        enc = tokenizer(child_text,
                                        return_tensors="pt", max_length=MAX_LEN,
                                        truncation=True, padding=True)

                    input_ids      = enc["input_ids"]
                    attention_mask = enc["attention_mask"]
                    token_type_ids = enc.get("token_type_ids", None)

                    # Baseline: all-PAD token IDs
                    pad_id   = tokenizer.pad_token_id or 0
                    baseline = torch.full_like(input_ids, pad_id)

                    with torch.no_grad():
                        logits     = forward_func(input_ids, attention_mask, token_type_ids)
                        pred_class = int(logits.argmax(-1).item())

                    add_args = (attention_mask, token_type_ids)
                    attrs, _ = lig.attribute(
                        inputs=input_ids,
                        baselines=baseline,
                        target=pred_class,
                        additional_forward_args=add_args,
                        n_steps=N_STEPS,
                        return_convergence_delta=True,
                    )

                    # Sum over embedding dim to one score per token, normalise
                    attr_scores = attrs.sum(dim=-1).squeeze(0)
                    max_abs = attr_scores.abs().max().item() or 1.0
                    attr_norm = (attr_scores / max_abs).tolist()

                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

                    attr_results[pid] = {
                        "tokens":          tokens,
                        "attributions":    [round(float(a), 6) for a in attr_norm],
                        "predicted_class": pred_class,
                    }

                    if i < N_HTML:
                        html_examples.append(
                            (pid, _make_html(tokens, attr_norm, pid, task, pred_class))
                        )

                except Exception as e:
                    print(f"      Attribution error for {pid}: {e}")

            with open(attr_file, "w", encoding="utf-8") as f:
                json.dump(attr_results, f, indent=2, ensure_ascii=False)
            print(f"      Attributions ({len(attr_results)}/{len(sample_ids)}) -> {attr_file}")

            if html_examples:
                html_file = SHAP_BERT_DIR / f"{model_key}_{task}_examples.html"
                combined  = "\n<hr>\n".join(html for _, html in html_examples)
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(combined)
                print(f"      HTML ({len(html_examples)} examples) -> {html_file}")

            del model, lig
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# D. Error analysis
def error_analysis_all(df, raw_pairs_map):
    print("\nD. Error analysis")
    analysis = {}

    tasks = ["interaction", "stance", "techniques"]
    for task in tasks:
        pred_file = SHALLOW_DIR / "predictions"
        # Find best prediction file
        best_f1, best_preds = 0.0, None
        metrics_file = SHALLOW_DIR / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                metrics = json.load(f)
            for model_name, feat_dict in metrics.get(task, {}).items():
                for feat_set, m in feat_dict.items():
                    if m.get("macro_f1", 0) > best_f1:
                        best_f1  = m["macro_f1"]
                        best_preds = pred_file / f"{task}_{model_name}_{feat_set}.jsonl"

        if best_preds is None or not best_preds.exists():
            continue

        from utils.jsonl_io import load_jsonl
        preds = load_jsonl(str(best_preds))

        errors = []
        all_true, all_pred = [], []
        for r in preds:
            yt = r["y_true"]
            yp = r["y_pred"]
            if task == "techniques":
                # Multi-label: compare as sets
                all_true.append(yt); all_pred.append(yp)
            else:
                yt, yp = int(yt), int(yp)
                all_true.append(yt); all_pred.append(yp)
                if yt != yp:
                    pid = r["PairID"]
                    raw = raw_pairs_map.get(pid, {})
                    errors.append({
                        "PairID":       pid,
                        "y_true":       yt,
                        "y_pred":       yp,
                        "child_text":   raw.get("ChildText", "")[:300],
                        "parent_text":  raw.get("ParentText", "")[:300],
                    })

        # Confusion matrix
        if task != "techniques":
            from sklearn.metrics import confusion_matrix as cm_fn
            labels = sorted(list(set(all_true + all_pred)))
            cm = cm_fn(all_true, all_pred, labels=labels)
            cm_dict = {str(labels[i]): {str(labels[j]): int(cm[i][j])
                       for j in range(len(labels))}
                       for i in range(len(labels))}

            # Interaction-specific: agreement/disagreement confusion
            special_confusion = {}
            if task == "interaction":
                # Classes: 0=-1(HD), 1=-0.5(MD), 2=0(SN), 3=+0.5(CA), 4=+1(CD)
                # Constructive Agreement (3) vs Constructive Disagreement (4)
                agree_confuse = sum(1 for yt, yp in zip(all_true, all_pred)
                                    if {yt, yp} == {3, 4})
                dest_confuse  = sum(1 for yt, yp in zip(all_true, all_pred)
                                    if {yt, yp} == {0, 1})
                special_confusion = {
                    "constructive_agree_vs_disagree": agree_confuse,
                    "destructive_agree_vs_disagree":  dest_confuse,
                }

            # Stance-specific: Against (0) vs Support (2) confusion
            if task == "stance":
                against_support = sum(1 for yt, yp in zip(all_true, all_pred)
                                      if {yt, yp} == {0, 2})
                special_confusion = {"against_support_confusion": against_support}

            total = len(all_true)
            correct = sum(1 for yt, yp in zip(all_true, all_pred) if yt == yp)
            analysis[task] = {
                "n_test": total,
                "n_correct": correct,
                "pct_correct": round(100 * correct / total, 1) if total else 0,
                "n_errors": total - correct,
                "confusion_matrix": cm_dict,
                "special_confusion": special_confusion,
                "top_errors": errors[:10],
            }
        else:
            analysis[task] = {"note": "Multi-label task — see metrics.json for per-label F1."}

        print(f"  {task}: {total-correct if task != 'techniques' else '?'} errors")

    err_file = OUT_DIR / "error_analysis.json"
    with open(err_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n  Error analysis -> {err_file}")
    return analysis


# Main
def main():
    df, emb_maps = load_features()

    # Load raw pair texts for BERT SHAP + error analysis
    # Load LLM annotations first, then gold standard overwrites 
    from utils.jsonl_io import load_jsonl
    raw_pairs_map = {}
    for path in [
        Path("10_llm_annotations") / "llm_annotated_subset.jsonl",
        Path("11_llm_annotations") / "llm_annotated_rest.jsonl",
        Path("09_manual_annotations") / "gold_standard.jsonl",   
    ]:
        if path.exists():
            for r in load_jsonl(str(path)):
                raw_pairs_map[r["PairID"]] = r

    shap_shallow(df, emb_maps)
    shap_bert(df, raw_pairs_map)
    error_analysis_all(df, raw_pairs_map)

    print(f"\nExplainability complete. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()