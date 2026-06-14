"""
Feature engineering

Produces:
  data/14_features/split_index.json                   {PairID to "train"|"val"|"test"}
  data/14_features/features.parquet                   21 B-block features + stance/technique
                                                       cols + metadata
  data/14_features/bert_embeddings_pair_gbert.npz     pair [CLS] embeddings — deepset/gbert-base
  data/14_features/bert_embeddings_child_gbert.npz    child [CLS] embeddings — deepset/gbert-base
  data/14_features/bert_embeddings_pair_xlmr.npz      pair [CLS] embeddings — xlm-roberta-base
  data/14_features/bert_embeddings_child_xlmr.npz     child [CLS] embeddings — xlm-roberta-base

Theory-driven features (B block — 21 features)
  child_len_words, parent_len_words, len_ratio, child_ttr, vocab_overlap,
  child_excl_count, child_quest_count, child_caps_ratio, child_neg_count,
  child_intensifier_count, mention_count, child_political_count, has_url,
  child_first_person_ratio, child_avg_word_len,
  sent_pos, sent_neg, sent_neu, sent_compound,   >>> 4-score sentiment 
  word_count_diff, parent_caps_ratio             

Stance-derived features (S block — requires ParentStanceLabel in annotations)
  child_stance, parent_stance, same_stance, abs_stance_diff

Technique-aggregate feature (T block)
  technique_count   >>> count of techniques annotated for child comment

BERT embeddings: extracted for both German BERT and XLM-RoBERTa.
  Both models are 768-dim [CLS] token representations.
  Pair encoding  (interaction task): [CLS] parent [SEP] child [SEP]
  Child encoding (stance/techniques): [CLS] child [SEP]

Sentiment: oliverguhr/german-sentiment-bert called with top_k=None -> 3 class probabilities.
  sent_compound = sent_pos − sent_neg  (analogous to VADER compound score).
  Cache file: sentiment_cache.json  (format: dict per text).

"""
import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import torch
from transformers import pipeline, BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel

# Explicit tokenizer + model classes
TOKENIZER_CLASSES = {
    "gbert":    BertTokenizer,        # deepset/gbert-base, WordPiece BERT tokenizer
    "xlmr":     XLMRobertaTokenizer,  # xlm-roberta-base, SentencePiece tokenizer
}
MODEL_CLASSES = {
    "gbert":    BertModel,            # deepset/gbert-base
    "xlmr":     XLMRobertaModel,      # xlm-roberta-base
}

from utils.jsonl_io import load_jsonl

load_dotenv()

BERT_BATCH    = int(os.getenv("BERT_BATCH_SIZE", 16))
BERT_MAX_LEN  = int(os.getenv("BERT_MAX_LEN", 256))
TEST_SPLIT    = float(os.getenv("TEST_SPLIT", 0.15))
VAL_SPLIT     = float(os.getenv("VAL_SPLIT", 0.15))
RANDOM_SEED   = int(os.getenv("RANDOM_SEED", 42))

# Both embedding models 
# xlm-roberta-base 
BERT_MODELS = {
    "gbert": "deepset/gbert-base",
    "xlmr":  "FacebookAI/xlm-roberta-base",
}

# Input files
GOLD_FILE   = Path("09_manual_annotations/gold_standard.jsonl")
SUBSET_FILE = Path("10_llm_annotations/llm_annotated_subset.jsonl")
REST_FILE   = Path("11_llm_annotations/llm_annotated_rest.jsonl")

OUT_DIR    = Path("14_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE      = OUT_DIR / "split_index.json"
FEATURES_FILE   = OUT_DIR / "features.parquet"
SENT_CACHE_FILE = OUT_DIR / "sentiment_cache.json"

SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# German lexicons

NEG_WORDS = {
    "nicht", "nein", "kein", "keine", "keiner", "keinem", "keinen", "keines",
    "niemals", "nie", "nirgends", "nirgendwo", "weder", "ohne", "kaum",
    "nichts", "niemand", "nimmer",
}

INTENSIFIERS = {
    "sehr", "extrem", "absolut", "total", "komplett", "vollkommen", "gar",
    "besonders", "ausgesprochen", "unglaublich", "enorm", "riesig", "wahnsinnig",
    "unbedingt", "tiefgreifend", "zutiefst",
}

POLITICAL_KEYWORDS = {
    "migration", "migrant", "migranten", "flüchtling", "flüchtlinge", "asyl",
    "asylbewerber", "ausländer", "einwanderung", "abschiebung", "grenze",
    "grenzen", "klima", "klimawandel", "klimaschutz", "co2", "emission",
    "emissionen", "erneuerbar", "windkraft", "solar", "kohle", "energiewende",
    "diesel", "verbrenner", "nachhaltigkeit", "wähler", "partei", "regierung",
    "bundesregierung", "kanzler", "bundestag", "demokratie", "wahl",
}

FIRST_PERSON = {"ich", "mir", "mich", "wir", "uns", "unser", "unsere", "mein", "meine"}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


# Theory-driven features

def ttr(text: str) -> float:
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def jaccard(text_a: str, text_b: str) -> float:
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def caps_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    return sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0.0


def first_person_ratio(text: str) -> float:
    tokens = text.lower().split()
    return sum(1 for t in tokens if t in FIRST_PERSON) / len(tokens) if tokens else 0.0


def avg_word_len(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def extract_theory_features(pair: dict) -> dict:
    child  = pair.get("ChildText", "")
    parent = pair.get("ParentText", "")
    child_lower  = child.lower()
    child_tokens = child_lower.split()
    child_wc     = len(child_tokens)
    parent_wc    = len(parent.split())

    return {
        "child_len_words":          child_wc,
        "parent_len_words":         parent_wc,
        "len_ratio":                child_wc / (parent_wc + 1),
        "child_ttr":                ttr(child),
        "vocab_overlap":            jaccard(parent, child),
        "child_excl_count":         child.count("!"),
        "child_quest_count":        child.count("?"),
        "child_caps_ratio":         caps_ratio(child),
        "child_neg_count":          sum(1 for t in child_tokens if t in NEG_WORDS),
        "child_intensifier_count":  sum(1 for t in child_tokens if t in INTENSIFIERS),
        "mention_count":            len(re.findall(r"@\w+", child)),
        "child_political_count":    sum(1 for t in child_tokens if t in POLITICAL_KEYWORDS),
        "has_url":                  1 if URL_PATTERN.search(child) else 0,
        "child_first_person_ratio": first_person_ratio(child),
        "child_avg_word_len":       avg_word_len(child),
        "word_count_diff":          abs(child_wc - parent_wc),
        "parent_caps_ratio":        caps_ratio(parent),
    }


# Sentiment
def load_sentiment_cache() -> dict:
    if SENT_CACHE_FILE.exists():
        with open(SENT_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_sentiment_cache(cache: dict):
    with open(SENT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def compute_sentiment_scores(texts: list, cache: dict) -> list:
    """
    Returns list of sentiment score dicts using oliverguhr/german-sentiment-bert.

    Each dict has 4 keys: sent_pos, sent_neg, sent_neu, sent_compound.
    sent_compound = sent_pos - sent_neg).

    Cache format: {text: {"sent_pos": float, "sent_neg": float,
                                "sent_neu": float, "sent_compound": float}}
    """
    pending = [t for t in texts if t not in cache]
    if pending:
        print(f"  Computing sentiment for {len(pending)} texts (top_k=None)...")
        sent_pipe = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            truncation=True,
            max_length=512,
            top_k=None,           # returns all class probabilities
        )
        batch_size = 64
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            results = sent_pipe(batch, batch_size=batch_size)
            for text, res_list in zip(batch, results):
                # res_list = [{"label": "positive", "score": 0.9}, ...]
                scores = {r["label"].lower(): r["score"] for r in res_list}
                pos  = float(scores.get("positive", 0.0))
                neg  = float(scores.get("negative", 0.0))
                neu  = float(scores.get("neutral",  0.0))
                cache[text] = {
                    "sent_pos":      pos,
                    "sent_neg":      neg,
                    "sent_neu":      neu,
                    "sent_compound": pos - neg,
                }
            if (i // batch_size) % 5 == 0:
                print(f"    {i + len(batch)}/{len(pending)} texts processed...", end="\r")
        print()
        save_sentiment_cache(cache)
    return [cache[t] for t in texts]


# BERT embeddings
def bert_embed(texts: list, tokenizer, model) -> np.ndarray:
    """Return [CLS] embeddings for a list of texts. Shape: (N, 768)."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BERT_BATCH):
            batch = texts[i : i + BERT_BATCH]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            if (i // BERT_BATCH) % 10 == 0:
                print(f"    Embedded {i + len(batch)}/{len(texts)}", end="\r")
    print()
    return np.vstack(embeddings)


def bert_embed_pairs(pairs: list, tokenizer, model) -> np.ndarray:
    """Embed parent+child as sentence pair: [CLS] parent [SEP] child [SEP]."""
    texts_a = [p.get("ParentText", "") for p in pairs]
    texts_b = [p.get("ChildText",  "") for p in pairs]
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts_a), BERT_BATCH):
            ba = texts_a[i : i + BERT_BATCH]
            bb = texts_b[i : i + BERT_BATCH]
            enc = tokenizer(
                ba, bb,
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            if (i // BERT_BATCH) % 10 == 0:
                print(f"    Embedded {i + len(ba)}/{len(texts_a)}", end="\r")
    print()
    return np.vstack(embeddings)


# Labels and split
def score_to_bin(score) -> str:
    s = float(score) if score is not None else 0.0
    if s >= 0.25:  return "Constructive"
    if s >= -0.25: return "Slight/Neutral"
    if s >= -0.75: return "Moderately Destructive"
    return "Highly Destructive"


def load_all_pairs() -> list:
    """Merge LLM annotations (subset + rest) with manual gold labels.

    Merge strategy:
      - Start with LLM record as the base.
      - Apply gold fields on top via dict.update(), so gold values always win
        for shared keys (InteractionScore, StanceLabel, Techniques, etc.).
      - This preserves Topic (set by steps 06/07 and carried through the LLM
        annotator) for gold pairs that otherwise lose it through a hard overwrite.
    """
    records = {}
    for path in (SUBSET_FILE, REST_FILE):
        if path.exists():
            for r in load_jsonl(str(path)):
                records[r["PairID"]] = r
        else:
            print(f"  Warning: {path} not found, skipping.")
    if GOLD_FILE.exists():
        for r in load_jsonl(str(GOLD_FILE)):
            pid = r["PairID"]
            if pid in records:
                # keep LLM base, overwrite with gold annotation fields
                merged = dict(records[pid])
                merged.update(r)
                records[pid] = merged
            else:
                records[pid] = r   # gold-only pair (not in LLM set)
    else:
        print(f"  Warning: {GOLD_FILE} not found.")
    print(f"  Loaded: {len(records)} total pairs (gold merged over LLM, LLM Topic preserved)")
    return list(records.values())


def make_split(pairs: list) -> dict:
    """Stratified 70/15/15 split on InteractionBin. Returns {PairID: split_name}."""
    annotated = [p for p in pairs if p.get("InteractionScore") is not None]
    ids    = [p["PairID"] for p in annotated]
    strata = [score_to_bin(p["InteractionScore"]) for p in annotated]

    val_frac = VAL_SPLIT / (1.0 - TEST_SPLIT)
    ids_trainval, ids_test, strata_trainval, _ = train_test_split(
        ids, strata, test_size=TEST_SPLIT, stratify=strata, random_state=RANDOM_SEED
    )
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=val_frac, stratify=strata_trainval, random_state=RANDOM_SEED
    )
    split = {pid: "train" for pid in ids_train}
    split.update({pid: "val"  for pid in ids_val})
    split.update({pid: "test" for pid in ids_test})
    return split


# Main
def main():
    pairs = load_all_pairs()
    print(f"Total annotated pairs: {len(pairs)}")

    # Split
    if SPLIT_FILE.exists():
        print("Loading existing split...")
        with open(SPLIT_FILE, encoding="utf-8") as f:
            split = json.load(f)
    else:
        split = make_split(pairs)
        with open(SPLIT_FILE, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        counts = {s: sum(1 for v in split.values() if v == s) for s in ("train", "val", "test")}
        print(f"Split created: {counts}")

    annotated = [p for p in pairs if p["PairID"] in split]

    # Theory features + sentiment
    _REQUIRED_COLS = {"sent_pos", "word_count_diff", "parent_caps_ratio",
                      "same_stance", "technique_count"}
    _stale = (FEATURES_FILE.exists() and
              not _REQUIRED_COLS.issubset(set(pd.read_parquet(FEATURES_FILE).columns)))
    if _stale:
        print("features.parquet is stale (missing new columns) — regenerating...")
        FEATURES_FILE.unlink()

    if FEATURES_FILE.exists():
        print("features.parquet already exists, skipping feature extraction.")
        df = pd.read_parquet(FEATURES_FILE)
    else:
        print("Extracting theory-driven features...")
        child_texts = [p.get("ChildText", "")[:512] for p in annotated]
        sent_cache  = load_sentiment_cache()
        sent_scores = compute_sentiment_scores(child_texts, sent_cache)

        rows = []
        for pair, sent in zip(annotated, sent_scores):
            feats = extract_theory_features(pair)
            # Extended sentiment scores (B block — replaces child_sent_polarity)
            feats["sent_pos"]      = sent["sent_pos"]
            feats["sent_neg"]      = sent["sent_neg"]
            feats["sent_neu"]      = sent["sent_neu"]
            feats["sent_compound"] = sent["sent_compound"]
            # Technique aggregate (T block)
            techniques = pair.get("Techniques") or []
            feats["technique_count"] = len(techniques) if isinstance(techniques, list) \
                                        else len(json.loads(techniques))
            # Stance-derived features (S block) — requires ParentStanceLabel
            # Default to 1 (Neutral) when label is missing
            _cs = pair.get("StanceLabel")
            _ps = pair.get("ParentStanceLabel")
            child_stance  = int(_cs) if _cs is not None else 1
            parent_stance = int(_ps) if _ps is not None else 1
            feats["child_stance"]    = child_stance
            feats["parent_stance"]   = parent_stance
            feats["same_stance"]     = int(child_stance == parent_stance)
            feats["abs_stance_diff"] = abs(child_stance - parent_stance)
            # Metadata
            feats["PairID"]           = pair["PairID"]
            feats["VideoID"]          = pair.get("VideoID", "")
            feats["Topic"]            = pair.get("Topic", "")
            feats["ThreadID"]         = pair.get("ThreadID", "")
            feats["InteractionScore"] = pair.get("InteractionScore")
            feats["InteractionBin"]   = score_to_bin(pair.get("InteractionScore"))
            feats["StanceLabel"]      = pair.get("StanceLabel")
            feats["Techniques"]       = json.dumps(pair.get("Techniques") or [])
            feats["split"]            = split.get(pair["PairID"], "train")
            rows.append(feats)

        df = pd.DataFrame(rows)
        df.to_parquet(FEATURES_FILE, index=False)
        print(f"features.parquet saved: {len(df)} rows, {len(df.columns)} columns")
        # Verify no NaN in key columns
        key_cols = ["sent_pos", "sent_neg", "sent_neu", "sent_compound",
                    "word_count_diff", "parent_caps_ratio", "technique_count",
                    "same_stance", "abs_stance_diff"]
        nan_counts = df[key_cols].isna().sum()
        if nan_counts.any():
            print(f"  WARNING: NaN values detected:\n{nan_counts[nan_counts > 0]}")
        else:
            print(f"  All {len(key_cols)} new columns: 0 NaN values")

    # BERT embeddings — loop over all models
    child_texts_full = [p.get("ChildText", "") for p in annotated]
    pair_ids_arr     = np.array([p["PairID"] for p in annotated])

    for model_key, model_name in BERT_MODELS.items():
        pair_file  = OUT_DIR / f"bert_embeddings_pair_{model_key}.npz"
        child_file = OUT_DIR / f"bert_embeddings_child_{model_key}.npz"

        if pair_file.exists() and child_file.exists():
            print(f"\n[{model_key}] Embedding files already exist, skipping.")
            continue

        print(f"\n[{model_key}] Loading model: {model_name}")
        tokenizer  = TOKENIZER_CLASSES[model_key].from_pretrained(model_name)
        bert_model = MODEL_CLASSES[model_key].from_pretrained(model_name).to(DEVICE)

        if not pair_file.exists():
            print(f"[{model_key}] Computing pair embeddings (interaction task)...")
            pair_embs = bert_embed_pairs(annotated, tokenizer, bert_model)
            np.savez(pair_file, embeddings=pair_embs, pair_ids=pair_ids_arr)
            print(f"[{model_key}] Saved pair embeddings: {pair_embs.shape} at {pair_file.name}")

        if not child_file.exists():
            print(f"[{model_key}] Computing child embeddings (stance/techniques tasks)...")
            child_embs = bert_embed(child_texts_full, tokenizer, bert_model)
            np.savez(child_file, embeddings=child_embs, pair_ids=pair_ids_arr)
            print(f"[{model_key}] Saved child embeddings: {child_embs.shape} at {child_file.name}")

        # Free GPU memory before loading the next model
        del bert_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\nSplit distribution:")
    for s in ("train", "val", "test"):
        n = (df["split"] == s).sum()
        print(f"  {s}: {n} ({100*n/len(df):.1f}%)")
    print("\nClass distribution (InteractionBin):")
    print(df.groupby(["split", "InteractionBin"]).size().unstack(fill_value=0))
    print("\nFeature columns in parquet:")
    b_cols = ["child_len_words","parent_len_words","len_ratio","child_ttr","vocab_overlap",
              "child_excl_count","child_quest_count","child_caps_ratio","child_neg_count",
              "child_intensifier_count","mention_count","child_political_count","has_url",
              "child_first_person_ratio","child_avg_word_len",
              "sent_pos","sent_neg","sent_neu","sent_compound",
              "word_count_diff","parent_caps_ratio"]
    s_cols = ["child_stance","parent_stance","same_stance","abs_stance_diff"]
    t_cols = ["technique_count"]
    for block, cols in [("B (21)", b_cols), ("S (4)", s_cols), ("T (1)", t_cols)]:
        present = [c for c in cols if c in df.columns]
        print(f"  Block {block}: {len(present)}/{len(cols)} present "
              f"{'done' if len(present)==len(cols) else ' MISSING: ' + str(set(cols)-set(df.columns))}")
    print("\nEmbedding files produced:")
    for model_key in BERT_MODELS:
        for kind in ("pair", "child"):
            p = OUT_DIR / f"bert_embeddings_{kind}_{model_key}.npz"
            status = f"{np.load(p)['embeddings'].shape}" if p.exists() else "MISSING"
            print(f"  {p.name}: {status}")
    print("\nStep 14 complete.")


if __name__ == "__main__":
    main()
