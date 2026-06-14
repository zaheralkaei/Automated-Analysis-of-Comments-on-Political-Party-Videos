"""
Shallow learner training and evaluation — GridSearchCV hyperparameter tuning.

Tasks:
  interaction  — 5-class ordinal (-1/-0.5/0/+0.5/+1)  — pair-level features
  stance       — 3-class (0=Against/1=Neutral/2=Support) — child-only features
  techniques   — 14-label multi-label                   — child-only features

Models: LR, RF, SVM (LinearSVC), SVM_RBF, XGBoost

Embedding models:
  gbert    — deepset/gbert-base          (German BERT,   768-dim [CLS])
  groberta — FacebookAI/xlm-roberta-base (XLM-RoBERTa,  768-dim [CLS])

Feature sets — After Bassi's et al. (2025) ablation blocks
  B           — 21 pure linguistic features (THEORY_COLS)
  B+S         — B + stance block: child_stance, parent_stance, same_stance, abs_stance_diff (25-dim)
  B+S+T       — B+S + technique block: technique_count + 14 binary indicators (40-dim)
  B+S+T+E1    — B+S+T + gbert [CLS] embeddings (808-dim)   [German BERT]
  B+S+T+E2    — B+S+T + groberta [CLS] embeddings (808-dim) [XLM-RoBERTa]

Feature sets — embedding / PCA variants (all tasks)
  bert / groberta       — 768-dim [CLS] embedding, raw
  B+bert                — B + gbert (789-dim)
  B+groberta            — B + groberta (789-dim)
  bert_pca              — PCA on gbert; n_components grid-searched [20,50,100]
  groberta_pca          — PCA on groberta; n_components grid-searched
  B+bert_pca            — B (passthrough) + PCA on gbert part
  B+groberta_pca        — B (passthrough) + PCA on groberta part

Hyperparameter tuning:
  Stage 1 — GridSearchCV(StratifiedKFold/KFold cv=3) on X_train.
             For _pca feature sets, n_components is also tuned within the pipeline.
  Stage 2 — Fresh model with best params refitted on X_train ∪ X_val.
  X_test  — held out entirely; never seen during tuning or Stage-2 fit.

No oversampling — class_weight='balanced' handles class imbalance.

Resumable: existing .joblib model files are reloaded; metrics.json is updated
  incrementally after every configuration (fault-tolerant).

Outputs:
  15_shallow_results/metrics.json
  15_shallow_results/predictions/<task>_<model>_<feat_set>.jsonl
  15_shallow_results/models/<task>_<model>_<feat_set>.joblib
"""
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (f1_score, cohen_kappa_score, hamming_loss, make_scorer,
                             accuracy_score, precision_score, recall_score)
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr
from xgboost import XGBClassifier

# Suppress LR convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from utils.jsonl_io import save_jsonl

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

FEAT_DIR  = Path("14_features")
OUT_DIR   = Path("15_shallow_results")
PRED_DIR  = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"
for d in (OUT_DIR, PRED_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

THEORY_COLS = [
    # original 15 (child_sent_polarity removed)
    "child_len_words", "parent_len_words", "len_ratio", "child_ttr",
    "vocab_overlap", "child_excl_count", "child_quest_count", "child_caps_ratio",
    "child_neg_count", "child_intensifier_count", "mention_count",
    "child_political_count", "has_url", "child_first_person_ratio", "child_avg_word_len",
    # extended sentiment: 4 scores from oliverguhr/german-sentiment-bert
    "sent_pos", "sent_neg", "sent_neu", "sent_compound",
    "word_count_diff", "parent_caps_ratio",
]
# B block = THEORY_COLS (21 features)

# Stance block (S) — requires ParentStanceLabel annotation
STANCE_COLS = ["child_stance", "parent_stance", "same_stance", "abs_stance_diff"]

# Technique block (T) — technique_count + 14 binary indicators
TECHNIQUE_AGGREGATE_COL = ["technique_count"]
# VALID_TECHNIQUES defines the 14 binary indicators 
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

N_THEORY  = len(THEORY_COLS)           # 21
N_STANCE  = len(STANCE_COLS)           # 4
N_TECH    = 1 + len(VALID_TECHNIQUES)  # 15  (technique_count + 14 binary)
N_BERT    = 768                        # same for gbert and groberta

PCA_N_GRID = [20, 50, 100]

# Feature sets per task
FEAT_SETS_BY_TASK = {
    "interaction": [
        # Ablation blocks — but with two differnt embeddings E1=gbert, E2=groberta
        "B", "B+S", "B+S+T", "B+S+T+E1", "B+S+T+E2",
        # gbert (German BERT)  — "B" is the canonical alias for theory-only (21-dim)
        "bert", "B+bert",
        "bert_pca", "B+bert_pca",
        # xlmr / groberta (XLM-RoBERTa)
        "groberta", "B+groberta",
        "groberta_pca", "B+groberta_pca",
    ],
    "stance": [
        # B block only 
        "B",
        # embedding / PCA variants
        "bert", "B+bert",
        "bert_pca", "B+bert_pca",
        "groberta", "B+groberta",
        "groberta_pca", "B+groberta_pca",
    ],
    "techniques": [
        # B block only
        "B",
        # embedding / PCA variants
        "bert", "B+bert",
        "bert_pca", "B+bert_pca",
        "groberta", "B+groberta",
        "groberta_pca", "B+groberta_pca",
    ],
}

# RBF-SVM is only run on low-dimensional feature sets (tractable training time).
# B-block sets are low-dim; B+S+T+E1/E2 (808-dim) are excluded.
SVM_RBF_FEAT_SETS = {
    "interaction": ["B", "B+S", "B+S+T",
                    "bert_pca", "B+bert_pca",
                    "groberta_pca", "B+groberta_pca"],
    "stance":      ["B", "bert_pca", "B+bert_pca",
                    "groberta_pca", "B+groberta_pca"],
    "techniques":  ["B", "bert_pca", "B+bert_pca",
                    "groberta_pca", "B+groberta_pca"],
}

# Hyperparameter grids
PARAM_GRIDS = {
    "LR":      {"C": [0.01, 0.1, 1.0, 10.0]},
    "RF":      {"n_estimators": [200, 400], "max_depth": [None, 10, 20]},
    "SVM":     {"C": [0.01, 0.1, 1.0, 10.0]},
    "SVM_RBF": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.001]},
    "XGB": {
        "n_estimators":  [100, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [4, 6],
    },
}

MACRO_F1_SCORER = make_scorer(f1_score, average="macro", zero_division=0)


# Estimator factory
def _make_base(model_name: str, n_jobs: int = 1):
    if model_name == "LR":
        return LogisticRegression(
            max_iter=3000, solver="saga",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    elif model_name == "RF":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=n_jobs, random_state=RANDOM_SEED,
        )
    elif model_name == "SVM":
        return LinearSVC(
            class_weight="balanced", max_iter=2000,
            random_state=RANDOM_SEED,
        )
    elif model_name == "SVM_RBF":
        # RBF kernel SVM — only used on PCA-reduced (low-dim) feature sets.
        # probability=False for speed; class_weight handles imbalance.
        return SVC(
            kernel="rbf", class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    elif model_name == "XGB":
        return XGBClassifier(
            tree_method="hist", random_state=RANDOM_SEED, n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _wrap(base, task: str, n_jobs: int = 1):
    if task == "techniques":
        return OneVsRestClassifier(base, n_jobs=n_jobs)
    return base


def _canonical_pca(feat_set: str) -> str:
    """
    Normalise a _pca feature set name to its canonical gbert form.
    Used so that groberta PCA variants reuse the same pipeline logic.
      "bert_pca"      → "bert"
      "groberta_pca"  → "bert"    (same 768-dim PCA pipeline)
      "B+bert_pca"    → "B+bert"
      "B+groberta_pca"→ "B+bert"
    """
    return feat_set.replace("_pca", "").replace("groberta", "bert")


# Grid search construction
def build_grid_search(model_name: str, task: str, feat_set: str) -> GridSearchCV:
    """
    Build GridSearchCV for a model × task × feature_set combination.

    For _pca feature sets, both gbert and groberta use the same Pipeline
    structure — they produce identical-dimension (768-dim) embeddings so
    the same PCA logic applies to both.

    Param prefixes:
      Non-pipeline, non-OVR :  "C"
      Non-pipeline, OVR     :  "estimator__C"
      Pipeline, non-OVR     :  "clf__C"
      Pipeline, OVR         :  "clf__estimator__C"
    """
    base     = _make_base(model_name, n_jobs=1)
    clf      = _wrap(base, task, n_jobs=1)
    raw_grid = PARAM_GRIDS[model_name]
    is_ovr   = (task == "techniques")
    is_pca   = feat_set.endswith("_pca")

    if is_pca:
        clf_pfx  = "clf__estimator__" if is_ovr else "clf__"
        canonical = _canonical_pca(feat_set)  

        if canonical == "bert":
            # 768-dim input (embedding only) — PCA on the full input
            estimator = Pipeline([
                ("pca", PCA(random_state=RANDOM_SEED)),
                ("clf", clf),
            ])
            param_grid = {
                "pca__n_components": PCA_N_GRID,
                **{f"{clf_pfx}{k}": v for k, v in raw_grid.items()},
            }

        elif canonical == "B+bert":
            # (N_THEORY + N_BERT)-dim input — passthrough B block, PCA on BERT part
            ct = ColumnTransformer([
                ("B", "passthrough", list(range(N_THEORY))),
                ("pca",    PCA(random_state=RANDOM_SEED),
                           list(range(N_THEORY, N_THEORY + N_BERT))),
            ])
            estimator = Pipeline([
                ("preprocessor", ct),
                ("clf", clf),
            ])
            param_grid = {
                "preprocessor__pca__n_components": PCA_N_GRID,
                **{f"{clf_pfx}{k}": v for k, v in raw_grid.items()},
            }

        else:
            raise ValueError(f"Unrecognised PCA feature set: {feat_set}")

    else:
        estimator  = clf
        param_pfx  = "estimator__" if is_ovr else ""
        param_grid = {f"{param_pfx}{k}": v for k, v in raw_grid.items()}

    cv = (KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
          if is_ovr
          else StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED))

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"    Grid: {n_combos} combinations × 3 folds = {n_combos * 3} fits")

    return GridSearchCV(
        estimator, param_grid,
        scoring=MACRO_F1_SCORER,
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=0,
    )


def tune_and_fit(model_name: str, task: str, feat_set: str,
                 X_train, y_train, X_val, y_val):
    """
    Stage 1: GridSearchCV on X_train resulting in best_params, best_cv_f1.
    Stage 2: Fresh model with best_params, refitted on X_train ∪ X_val.
    Returns (fitted_clf, best_params_dict, best_cv_f1).
    """
    gs = build_grid_search(model_name, task, feat_set)
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_cv_f1  = float(gs.best_score_)
    print(f"    CV best macro-F1 = {best_cv_f1:.4f}  | params = {best_params}")

    X_tv = np.vstack([X_train, X_val])
    y_tv = (np.vstack([y_train, y_val])
            if task == "techniques"
            else np.concatenate([y_train, y_val]))

    if feat_set.endswith("_pca"):
        final_clf = clone(gs.best_estimator_)
        final_clf.fit(X_tv, y_tv)
    else:
        clean_params = {k.replace("estimator__", ""): v for k, v in best_params.items()}
        base_final   = _make_base(model_name, n_jobs=-1)
        base_final.set_params(**clean_params)
        final_clf = _wrap(base_final, task, n_jobs=1)
        final_clf.fit(X_tv, y_tv)

    return final_clf, best_params, best_cv_f1


# Data loading
def load_data():
    """
    Returns (df, emb_maps) where emb_maps is a dict:
      {
        "pair_gbert":     {PairID: embedding_vector, ...},
        "child_gbert":    {PairID: embedding_vector, ...},
        "pair_groberta":  {PairID: embedding_vector, ...},
        "child_groberta": {PairID: embedding_vector, ...},
      }
    If an embedding file is missing, the corresponding dict is empty
    and get_Xy() will substitute zero vectors.
    """
    df = pd.read_parquet(FEAT_DIR / "features.parquet")

    # Restrict to pairs with StanceLabel so all feature sets are evaluated on the
    # same train / val / test rows — pairs without StanceLabel lack parent-stance
    # annotation and cannot contribute to B+S / B+S+T / B+S+T+E1/E2 sets.
    df = df[df["StanceLabel"].notna()].copy()
    print(f"Oracle-filtered dataset: {len(df)} pairs "
          f"(train={len(df[df['split']=='train'])}, "
          f"val={len(df[df['split']=='val'])}, "
          f"test={len(df[df['split']=='test'])})")

    # Map feature-set key to actual filename key produced by step 14.
    # "groberta" feature sets use the xlm-roberta-base embeddings 
    FILE_KEY_MAP = {"gbert": "gbert", "groberta": "xlmr"}

    emb_maps = {}
    for model_key, file_key in FILE_KEY_MAP.items():
        for split_type in ("pair", "child"):
            fname = FEAT_DIR / f"bert_embeddings_{split_type}_{file_key}.npz"
            key   = f"{split_type}_{model_key}"
            if fname.exists():
                # mmap_mode='r' avoids loading the full array into RAM at once;
                # rows are read on demand — safer for large files on network/VFS paths.
                npz = np.load(fname, allow_pickle=True, mmap_mode='r')
                embs = np.array(npz["embeddings"])   
                pids = np.array(npz["pair_ids"])
                emb_maps[key] = {pid: embs[i] for i, pid in enumerate(pids)}
            else:
                print(f"  Warning: {fname.name} not found — will use zero vectors.")
                emb_maps[key] = {}

    return df, emb_maps


def get_stance_features(sub: pd.DataFrame) -> np.ndarray:
    """
    4-dim stance block (S): child_stance, parent_stance, same_stance, abs_stance_diff.
    Uses predicted stance labels from the LLM annotation pipeline.
    Missing values imputed as 0 (Against / no difference).
    """
    return sub[STANCE_COLS].fillna(0).values.astype(float)


def get_technique_block(sub: pd.DataFrame) -> np.ndarray:
    """
    15-dim technique block (T): technique_count (1) + 14 binary technique indicators.
    technique_count from features.parquet; binary indicators computed from Techniques field.
    """
    counts = sub[TECHNIQUE_AGGREGATE_COL].fillna(0).values.astype(float)  # (n, 1)
    mlb    = MultiLabelBinarizer(classes=VALID_TECHNIQUES)
    binary = mlb.fit_transform(
        sub["Techniques"].apply(json.loads).tolist()
    ).astype(float)                                                        # (n, 14)
    return np.hstack([counts, binary])                                     # (n, 15)



def get_Xy(df, split: str, task: str, feat_set: str, emb_maps: dict):
    """
    Return (X, y, pair_ids) for a given split / task / feature_set.
    Embedding model is inferred from feat_set name
    For _pca sets the same raw X is returned as the non-PCA counterpart;
    PCA is applied inside the sklearn Pipeline.

    For oracle sets (interaction task only), rows with missing StanceLabel
    are dropped so all oracle features are valid.
    """
    sub = df[df["split"] == split].copy()

    # Labels
    if task == "interaction":
        sub = sub[sub["InteractionScore"].notna()]
        # oracle and non-oracle sets now share the same rows.
        y_raw = sub["InteractionScore"].apply(
            lambda s: SCORE_TO_CLASS[min(SCORE_TO_CLASS.keys(),
                                         key=lambda k: abs(k - float(s)))]
        ).values

    elif task == "stance":
        sub   = sub[sub["StanceLabel"].notna()]
        y_raw = sub["StanceLabel"].astype(int).values

    else:  # techniques
        mlb   = MultiLabelBinarizer(classes=VALID_TECHNIQUES)
        y_raw = mlb.fit_transform(sub["Techniques"].apply(json.loads).tolist())

    pair_ids = sub["PairID"].tolist()
    theory_X = sub[THEORY_COLS].fillna(0).values.astype(float)

    # Embedding selection
    # Strip _pca suffix first; it does not affect which model is used.
    # B+S+T+E2 explicitly uses groberta; everything else defaults to gbert
    # unless "groberta" appears in the name.
    base_fs   = feat_set.replace("_pca", "")
    emb_model = "groberta" if ("groberta" in base_fs or base_fs == "B+S+T+E2") else "gbert"
    emb_key   = f"pair_{emb_model}" if task == "interaction" else f"child_{emb_model}"
    bert_X    = np.vstack([emb_maps[emb_key].get(pid, np.zeros(N_BERT))
                           for pid in pair_ids])

    # Feature matrix
    # Normalise groberta→bert so both share the same switch cases below.
    norm_fs = base_fs.replace("groberta", "bert")

    if norm_fs == "B":
        X = theory_X
    elif norm_fs == "B+S":
        X = np.hstack([theory_X, get_stance_features(sub)])
    elif norm_fs == "B+S+T":
        X = np.hstack([theory_X, get_stance_features(sub), get_technique_block(sub)])
    elif norm_fs in ("B+S+T+E1", "B+S+T+E2"):
        # E1 = gbert, E2 = groberta — embedding already resolved above via emb_model
        X = np.hstack([theory_X, get_stance_features(sub), get_technique_block(sub), bert_X])
    elif norm_fs == "bert":
        X = bert_X
    elif norm_fs == "B+bert":
        X = np.hstack([theory_X, bert_X])
    else:
        raise ValueError(f"Unknown feature set: {feat_set!r}")

    return X, y_raw, pair_ids


# Metrics
def eval_interaction(y_true, y_pred) -> dict:
    scores_true = np.array([CLASS_TO_SCORE[c] for c in y_true])
    scores_pred = np.array([CLASS_TO_SCORE[c] for c in y_pred])
    try:
        r, _ = pearsonr(scores_true, scores_pred)
    except Exception:
        r = 0.0
    return {
        "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro",
                                                zero_division=0)), 4),
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro",
                                                       zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro",
                                                    zero_division=0)), 4),
        "per_class_f1":    [round(v, 4) for v in
                            f1_score(y_true, y_pred, average=None,
                                     zero_division=0).tolist()],
        "kappa_quadratic": round(float(cohen_kappa_score(y_true, y_pred,
                                                          weights="quadratic")), 4),
        "mae":             round(float(np.mean(np.abs(scores_true - scores_pred))), 4),
        "pearson_r":       round(float(r), 4),
    }


def eval_stance(y_true, y_pred) -> dict:
    per = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None,
                   zero_division=0).tolist()
    return {
        "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro",
                                                zero_division=0)), 4),
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro",
                                                       zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro",
                                                    zero_division=0)), 4),
        "per_class_f1":    {"Against": round(per[0], 4),
                            "Neutral":  round(per[1], 4),
                            "Support":  round(per[2], 4)},
        "kappa":           round(float(cohen_kappa_score(y_true, y_pred)), 4),
    }


def eval_techniques(y_true, y_pred) -> dict:
    per_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1":             round(float(f1_score(y_true, y_pred, average="macro",
                                                     zero_division=0)), 4),
        "exact_match_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision":      round(float(precision_score(y_true, y_pred, average="macro",
                                                            zero_division=0)), 4),
        "macro_recall":         round(float(recall_score(y_true, y_pred, average="macro",
                                                         zero_division=0)), 4),
        "per_label_f1":         {t: round(float(f), 4)
                                 for t, f in zip(VALID_TECHNIQUES, per_arr)},
        "hamming_loss":         round(float(hamming_loss(y_true, y_pred)), 4),
    }


def compute_metrics(task: str, y_true, y_pred) -> dict:
    if task == "interaction":
        return eval_interaction(y_true, y_pred)
    elif task == "stance":
        return eval_stance(y_true, y_pred)
    else:
        return eval_techniques(y_true, y_pred)


# Main training loop
def run():
    df, emb_maps = load_data()

    TASKS  = ["interaction", "stance", "techniques"]
    MODELS = ["LR", "RF", "SVM", "SVM_RBF", "XGB"]

    metrics_file = OUT_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    for task in TASKS:
        all_metrics.setdefault(task, {})

        for model_name in MODELS:
            # SVM_RBF only runs on its restricted (low-dim) feature sets
            feat_sets = (SVM_RBF_FEAT_SETS[task]
                         if model_name == "SVM_RBF"
                         else FEAT_SETS_BY_TASK[task])
            all_metrics[task].setdefault(model_name, {})

            for feat_set in feat_sets:
                key        = f"{task}_{model_name}_{feat_set}"
                pred_file  = PRED_DIR  / f"{key}.jsonl"
                model_file = MODEL_DIR / f"{key}.joblib"

                print(f"\n{'─' * 62}")
                print(f"[{key}]")

                X_train, y_train, _ = get_Xy(df, "train", task, feat_set, emb_maps)
                X_val,   y_val,   _ = get_Xy(df, "val",   task, feat_set, emb_maps)
                X_test, y_test, pids_test = get_Xy(df, "test", task, feat_set, emb_maps)

                # Standardise features: scaler is always paired with its model.
                # If a saved model exists, load its original scaler (do NOT refit).
                # If training fresh, fit scaler on train only, then save both together.
                scaler_file = MODEL_DIR / f"{key}_scaler.joblib"

                if model_file.exists() and scaler_file.exists():
                    print(f"  Loading saved model and scaler → {model_file.name}")
                    clf    = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    X_train = scaler.transform(X_train)
                    X_val   = scaler.transform(X_val)
                    X_test  = scaler.transform(X_test)
                else:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val   = scaler.transform(X_val)
                    X_test  = scaler.transform(X_test)
                    joblib.dump(scaler, scaler_file)   # save scaler before model

                    clf, best_params, best_cv_f1 = tune_and_fit(
                        model_name, task, feat_set,
                        X_train, y_train, X_val, y_val,
                    )
                    joblib.dump(clf, model_file)
                    all_metrics[task][model_name].setdefault(feat_set, {})
                    all_metrics[task][model_name][feat_set]["best_params"] = best_params
                    all_metrics[task][model_name][feat_set]["best_cv_f1"]  = round(best_cv_f1, 4)

                print(f"  Samples — train: {len(X_train)}  val: {len(X_val)}"
                      f"  test: {len(X_test)}  features: {X_train.shape[1]}")

                y_pred  = clf.predict(X_test)
                metrics = compute_metrics(task, y_test, y_pred)

                existing = all_metrics[task][model_name].get(feat_set, {})
                for mk in ("best_params", "best_cv_f1"):
                    if mk in existing:
                        metrics[mk] = existing[mk]

                all_metrics[task][model_name][feat_set] = metrics
                cv_note = (f"  (CV: {metrics['best_cv_f1']})"
                           if "best_cv_f1" in metrics else "")
                print(f"  Test macro-F1: {metrics['macro_f1']:.4f}{cv_note}")

                if not pred_file.exists():
                    if task == "techniques":
                        preds = [{"PairID": pid, "y_true": yt, "y_pred": yp}
                                 for pid, yt, yp in
                                 zip(pids_test, y_test.tolist(), y_pred.tolist())]
                    else:
                        preds = [{"PairID": pid, "y_true": int(yt), "y_pred": int(yp)}
                                 for pid, yt, yp in
                                 zip(pids_test, y_test.tolist(), y_pred.tolist())]
                    save_jsonl(preds, str(pred_file))

                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    _print_summary(all_metrics, TASKS, MODELS)
    print(f"\nMetrics saved → {metrics_file}")


# Summary printer
def _get_pca_n(params: dict) -> str:
    for k, v in params.items():
        if "n_components" in k:
            return str(v)
    return "?"


def _print_summary(all_metrics: dict, tasks: list, models: list):
    bassi_blocks = ["B", "B+S", "B+S+T", "B+S+T+E1", "B+S+T+E2"]
    gbert_raw    = ["B", "bert",     "B+bert"]
    groberta_raw = ["B", "groberta", "B+groberta"]
    pca_sets     = [("bert_pca", "groberta_pca"),
                    ("B+bert_pca", "B+groberta_pca")]

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY  (test set macro-F1)")
    print("=" * 72)

    for task in tasks:
        tm = all_metrics.get(task, {})
        print(f"\n{'─' * 72}")
        print(f"TASK: {task.upper()}")

        # After Bassi's et al. (2025) ablation blocks
        # interaction: B / B+S / B+S+T / B+S+T+E1 / B+S+T+E2
        # stance / techniques: B only
        avail_blocks = [b for b in bassi_blocks
                        if any(b in tm.get(mn, {}) for mn in models)]
        block_label = " / ".join(avail_blocks) if avail_blocks else "—"
        print(f"\n  After Bassi's et al. (2025) ablation — {block_label}:")
        if avail_blocks:
            print(f"  {'Model':<8} " + "  ".join(f"{s:>10}" for s in avail_blocks))
            for mn in models:
                vals = [tm.get(mn, {}).get(s, {}).get("macro_f1", "—")
                        for s in avail_blocks]
                row = "  ".join(f"{v:>10.4f}" if isinstance(v, float) else f"{'—':>10}"
                                for v in vals)
                print("  " + mn.ljust(8) + row)
        else:
            print("  (no B-block results yet)")

        # Raw gbert (SVM_RBF not run on raw 768-dim)
        raw_models = [mn for mn in models if mn != "SVM_RBF"]
        gbert_raw  = ["B", "bert", "B+bert"]
        print(f"\n  gbert (German BERT):")
        print(f"  {'Model':<8} " + "  ".join(f"{s:>10}" for s in gbert_raw))
        for mn in raw_models:
            vals = [tm.get(mn, {}).get(s, {}).get("macro_f1", 0.0) for s in gbert_raw]
            print("  " + mn.ljust(8) + "  ".join(f"{v:>10.4f}" for v in vals))

        # Raw groberta/xlmr (SVM_RBF not run on raw 768-dim)
        print(f"\n  groberta/xlmr (XLM-RoBERTa):")
        print(f"  {'Model':<8} " + "  ".join(f"{s:>12}" for s in groberta_raw))
        for mn in raw_models:
            vals = [tm.get(mn, {}).get(s, {}).get("macro_f1", 0.0) for s in groberta_raw]
            print("  " + mn.ljust(8) + "  ".join(f"{v:>12.4f}" for v in vals))

        # PCA comparison: gbert_pca vs groberta_pca
        print(f"\n  PCA-reduced (best n_components in parentheses):")
        for gb_s, gr_s in pca_sets:
            print(f"  {'Model':<6}  {gb_s:>20}  {gr_s:>22}")
            for mn in models:
                gb_e = tm.get(mn, {}).get(gb_s, {})
                gr_e = tm.get(mn, {}).get(gr_s, {})
                gb_f1 = gb_e.get("macro_f1", 0.0)
                gr_f1 = gr_e.get("macro_f1", 0.0)
                gb_n  = _get_pca_n(gb_e.get("best_params", {}))
                gr_n  = _get_pca_n(gr_e.get("best_params", {}))
                print(f"  {mn:<6}  {f'{gb_f1:.4f}(n={gb_n})':>20}  "
                      f"{f'{gr_f1:.4f}(n={gr_n})':>22}")

        # E1 vs E2 embedding comparison (interaction only)
        if task == "interaction":
            e_models = [mn for mn in models if mn != "SVM_RBF"]
            print(f"\n  Embedding comparison (E1=gbert vs E2=groberta) at B+S+T level:")
            print(f"  {'Model':<8} {'B+S+T+E1':>12} {'B+S+T+E2':>12}  {'Δ(E2−E1)':>10}")
            for mn in e_models:
                e1 = tm.get(mn, {}).get("B+S+T+E1", {}).get("macro_f1", 0.0)
                e2 = tm.get(mn, {}).get("B+S+T+E2", {}).get("macro_f1", 0.0)
                print(f"  {mn:<8} {e1:>12.4f} {e2:>12.4f}  {e2-e1:>+10.4f}")


if __name__ == "__main__":
    run()
