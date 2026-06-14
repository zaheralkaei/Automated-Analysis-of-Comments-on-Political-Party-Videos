"""
Open-weights LLM evaluation via OpenRouter

Models:
  Gemma 4 family:  gemma4_31b — google/gemma-4-31b-it
  Qwen 2.5 family: qwen25_7b  — qwen/qwen-2.5-7b-instruct
                   qwen25_72b — qwen/qwen-2.5-72b-instruct

Interaction task — conditions × shot mode:
  full_instructions            (B:     text only)
  full_instructions+stance     (B+S:   + child & parent predicted stance labels)
  full_instructions+stance+techniques  (B+S+T: + stance + technique list)
  × zero_shot / few_shot  >>>  6 interaction run types per model

Stance & techniques tasks:
  full_instructions × zero_shot / few_shot  >>>  2 run types per model each

Total: 3 models × (6 interaction + 2 stance + 2 techniques) = 30 runs.

Outputs:
  17_llm_results/metrics.json
  17_llm_results/predictions/<task>_<model_label>_<condition>_<shot_mode>.jsonl
"""
import os
import json
import time
import re
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import (f1_score, cohen_kappa_score, hamming_loss,
                             accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import pearsonr

from utils.jsonl_io import load_jsonl, append_jsonl

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
RANDOM_SEED        = int(os.getenv("RANDOM_SEED", 42))
random.seed(RANDOM_SEED)

# Dedicated seed for few-shot example selection 
# To use manually-specified examples instead, set FEWSHOT_PAIRIDS in .env as a
# comma-separated list of PairIDs (one per few-shot slot, in class order):
#   interaction : 5 PairIDs  (-1.0, -0.5, 0.0, +0.5, +1.0)
#   stance      : 3 PairIDs  (Against, Neutral, Support)
#   techniques  : 3 PairIDs  (with_tech, with_tech, without_tech)
FEWSHOT_SEED    = int(os.getenv("FEWSHOT_SEED", 42))
_fewshot_rng    = random.Random(FEWSHOT_SEED)

# Manually curated few-shot examples (hand-picked for clarity and annotation quality).
# Set a task entry to None to fall back to random sampling via FEWSHOT_SEED instead.
# Order within each list must follow the class order documented above.

MANUAL_FEWSHOT_IDS: dict[str, list[str] | None] = {
    "interaction": [
        # Destructive Disagreement  (-1.0)  — snarky dismissal re migration, no techniques
        "mNnX_k9Blfw_UgxcEWFF1N7swmunfK54AaABAg.AHtaIy-6h0iAI1L0cqcHDT",
        # Destructive Agreement     (-0.5)  — anti-immigration echo-chamber agreement, no techniques
        "mNnX_k9Blfw_UgzKT7F6NpH5Bmt_x3d4AaABAg.AHsMAmfl_KIAHsMqVu16oo",
        # Neutral / Rephrase        ( 0.0)  — off-topic meta-comment about video title, no techniques
        "4oWNKOs5LiM_Ugy62dRy3UsF8_cmuSB4AaABAg.AUApTs-yYINAUB707N5fI5",
        # Constructive Agreement    (+0.5)  — extends parent's housing/communes policy argument, no techniques
        "Qyds5R3ORmc_Ugz6-7dvrwL9s8XOdkB4AaABAg.AEHmg4tE-W-AEJyTaBfqq_",
        # Constructive Disagreement (+1.0)  — factual rebuttal of housing/refugee sarcasm, no techniques
        "repy4Y7GKM4_UgyFQH-8r2BIG-NmcS54AaABAg.A4flY5WiUwUA4qMYFfWiQE",
    ],
    # stance is split by topic so each prompt gets on-topic examples
    "stance_migration": [
        # Against (0) — anti-immigration echo-chamber, lamenting German disappearance, no techniques
        "mNnX_k9Blfw_UgzKT7F6NpH5Bmt_x3d4AaABAg.AHsMAmfl_KIAHsMqVu16oo",
        # Neutral (1) — rhetorical non-answer ("Wer soll das machen"), no techniques
        "8vwbPCpeV8M_UgylwSVjJYAbcCmRAf94AaABAg.9mnnN_psC8_9n3dvVZlWJG",
        # Support (2) — openly pro-refugee ("Ich nehme gerne Flüchtlinge auf!"), no techniques
        "2E4s7MQUIFI_Ugy8Zhf1uq01WLy4Jud4AaABAg.A9eBFyvM4rzA9eX-OKFAvx",
    ],
    "stance_climate": [
        # Against (0) — defends diesel range vs EVs, dismisses CO2 tax, no techniques
        "yQ1o56b6U1I_Ugx_Oi8j8JI4mjW_MqB4AaABAg.8vo0d-mfU_u9AVZZiC0MH2",
        # Neutral (1) — asks who Luisa Neubauer is, climate video, no stance taken, no techniques
        "1eUG3aYpHWY_Ugx3Pc6w93ltyLRQKK14AaABAg.9gUJQEr2PRX9gURGqSz13v",
        # Support (2) — pro-climate factual correction ("unser Planet"), no techniques
        "2fSv493XzrY_UgyftbOh2zsm0HhniOd4AaABAg.A3eLGR83tvLA3eYU7JOIYl",
    ],
    "techniques": [
        # Loaded_Language + Name_Calling — "Scheiß auf Integration" (migration video)
        "Qyds5R3ORmc_Ugwhtu6-DD9xgLO6N2p4AaABAg.AEICRLTwvt_AJUXxJ6Dxhi",
        # Flag-Waving + Whataboutism — "Merkel hat keine deutsche Abstammung"
        "61XL3yHFuPg_UgwJLcGXFCBH9OrhDcR4AaABAg.9E-EnFn5bUj9E3mhSD27an",
        # (none) — neutral meta-comment about video title, zero propaganda techniques
        "4oWNKOs5LiM_Ugy62dRy3UsF8_cmuSB4AaABAg.AUApTs-yYINAUB707N5fI5",
    ],
}

# Model registry
MODELS = {
    "gemma4_31b":  "google/gemma-4-31b-it",
    "qwen25_7b":   "qwen/qwen-2.5-7b-instruct",
    "qwen25_72b":  "qwen/qwen-2.5-72b-instruct",
}

# Paths
FEAT_DIR = Path("14_features")
OUT_DIR  = Path("17_llm_results")
PRED_DIR = OUT_DIR / "predictions"
for d in (OUT_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

CALL_DELAY = 0.5   # seconds between API calls

# Label vocabularies
VALID_TECHNIQUES = [
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice',
    'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy',
    'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
    'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling',
    'Repetition', 'Slogans/Thought-terminating_Cliches',
    'Whataboutism,Straw_Men', 'Appeal_to_Time',
]
VALID_TECHNIQUES_SET = set(VALID_TECHNIQUES)

SCORE_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
CLASS_TO_SCORE = {v: k for k, v in SCORE_TO_CLASS.items()}
SCORE_ENUM     = [-1.0, -0.5, 0.0, 0.5, 1.0]
SCORE_TO_LABEL = {
    -1.0: "Destructive Disagreement",
    -0.5: "Destructive Agreement",
     0.0: "Neutral/Rephrase",
     0.5: "Constructive Agreement",
     1.0: "Constructive Disagreement",
}
STANCE_LABELS = {"0": 0, "1": 1, "2": 2, "against": 0, "neutral": 1, "support": 2}
STANCE_NAMES  = {0: "Against", 1: "Neutral", 2: "Support"}

# Conditions
TASK_CONDITIONS = {
    "interaction": [
        "full_instructions",                       # B:     text-only, full prompt
        "full_instructions+stance",                # B+S:   + predicted child & parent stance
        "full_instructions+stance+techniques",     # B+S+T: + stance + technique list
    ],
    "stance":      ["full_instructions"],          # no enrichment applicable
    "techniques":  ["full_instructions"],          # no enrichment applicable
}
SHOT_MODES = ["zero_shot", "few_shot"]             # applied to ALL conditions above

PROMPTS_DIR = Path("prompts")


# System prompts
def _load_prompt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""

_JSON_SUFFIX_INTERACTION = (
    "\n\nRespond ONLY with a JSON object on a single line: "
    "{\"score\": <float>, \"label\": <string>} "
    "where score is one of [-1.0, -0.5, 0.0, 0.5, 1.0] and label is one of: "
    "\"Constructive Disagreement\", \"Constructive Agreement\", "
    "\"Neutral/Rephrase\", \"Destructive Agreement\", \"Destructive Disagreement\"."
)
_JSON_SUFFIX_STANCE = (
    "\n\nRespond ONLY with a JSON object: {\"label\": <int>} "
    "where 0=Against, 1=Neutral, 2=Support."
)
_JSON_SUFFIX_TECHNIQUES = (
    "\n\nFor the given text, identify which propaganda techniques are present. "
    "Respond ONLY with a JSON object: {\"techniques\": [<list of strings>]} "
    "using only the technique names listed above, or {\"techniques\": []} if none apply."
)

def _build_interaction_full() -> str:
    text = _load_prompt_file(PROMPTS_DIR / "interaction.txt")
    text = text.replace("Use the classify_interaction tool to provide your answer.", "").strip()
    return text + _JSON_SUFFIX_INTERACTION

def _build_stance_full(topic: str = "migration") -> str:
    fname = "stance_climate.txt" if topic == "climate" else "stance_migration.txt"
    text  = _load_prompt_file(PROMPTS_DIR / fname)
    # Remove annotation-specific output instruction
    for cutoff in ["CRUCIAL: Answer ONLY with the number",
                   "The following is the content to analyze:"]:
        if cutoff in text:
            text = text[:text.index(cutoff)].strip()
    return text + _JSON_SUFFIX_STANCE

def _build_techniques_full() -> str:
    text = _load_prompt_file(PROMPTS_DIR / "techniques.txt")
    # Remove annotation-specific output example
    for cutoff in ["For the given text please state", "An example output"]:
        if cutoff in text:
            text = text[:text.index(cutoff)].strip()
    return text + _JSON_SUFFIX_TECHNIQUES

# Minimal system prompts
INTERACTION_SYSTEM = (
    "You classify the interaction quality of a parent-child YouTube comment pair. "
    "Respond ONLY with a JSON object on a single line: "
    "{\"score\": <float>, \"label\": <string>} "
    "where score is one of [-1.0, -0.5, 0.0, 0.5, 1.0] and label is one of: "
    "\"Constructive Disagreement\", \"Constructive Agreement\", "
    "\"Neutral/Rephrase\", \"Destructive Agreement\", \"Destructive Disagreement\"."
)
STANCE_SYSTEM = (
    "You classify the stance of a YouTube comment toward the video's political topic. "
    "Respond ONLY with a JSON object: {\"label\": <int>} "
    "where 0=Against, 1=Neutral, 2=Support."
)
TECHNIQUES_SYSTEM = (
    "You detect propaganda techniques in a YouTube comment. "
    "Respond ONLY with a JSON object: {\"techniques\": [<list of strings>]} "
    "where each string is one of the following (or an empty list if none apply): "
    + ", ".join(VALID_TECHNIQUES) + "."
)

# Full annotation-grade system prompts (built at import time from prompt files)
INTERACTION_SYSTEM_FULL   = _build_interaction_full()
TECHNIQUES_SYSTEM_FULL    = _build_techniques_full()
# Stance full prompts are topic-dependent — built per-pair in run_task()


# Few-shot example selection
def build_few_shot_pool(gold_path: Path, split_index: dict) -> list:
    """Load gold standard pairs that belong to the train split only."""
    if not gold_path.exists():
        return []
    pool = []
    for r in load_jsonl(str(gold_path)):
        if split_index.get(r["PairID"]) == "train":
            pool.append(r)
    return pool


def select_few_shot_examples(pool: list, task: str,
                             manual_ids: list[str] | None = None,
                             topic: str | None = None) -> list:
    """Select balanced few-shot examples from the gold train pool.

    Selection strategy (automatic)
    --------------------------------
      interaction : 1 example per score class  (-1.0, -0.5, 0.0, +0.5, +1.0)
      stance      : 1 example per stance class (Against=0, Neutral=1, Support=2)
      techniques  : 3 examples — 2 with at least one technique, 1 without

    Uses _fewshot_rng (seeded by FEWSHOT_SEED, default 42) so selection is
    fully reproducible and independent of every other random call in the script.

    Manual override
    ---------------
    Pass manual_ids as a list of PairIDs to pin specific examples.
    The pool is indexed by PairID; missing IDs are silently skipped.
    Set FEWSHOT_SEED to any integer in .env to get a different automatic draw.
    """
    pool_by_id = {r["PairID"]: r for r in pool}

    # Manual override: use exactly the given PairIDs (in order)
    if manual_ids:
        return [pool_by_id[pid] for pid in manual_ids if pid in pool_by_id]

    if task == "interaction":
        by_class = {s: [] for s in SCORE_ENUM}
        for r in pool:
            s = r.get("InteractionScore")
            if s is not None:
                snapped = min(SCORE_ENUM, key=lambda k: abs(k - float(s)))
                by_class[snapped].append(r)
        examples = []
        for score in SCORE_ENUM:
            bucket = by_class[score]
            if bucket:
                examples.append(_fewshot_rng.choice(bucket))
        return examples

    elif task == "stance":
        filtered_pool = pool
        if topic:
            filtered_pool = [r for r in pool
                             if str(r.get("category") or r.get("Category") or "").lower()
                             == topic.lower()]
        by_class = {0: [], 1: [], 2: []}
        for r in filtered_pool:
            s = r.get("StanceLabel")
            if s is not None:
                by_class[int(s)].append(r)
        examples = []
        for cls in [0, 1, 2]:
            bucket = by_class[cls]
            if bucket:
                examples.append(_fewshot_rng.choice(bucket))
        return examples

    else:  # techniques
        with_tech    = [r for r in pool if (
                            json.loads(r["Techniques"]) if isinstance(r.get("Techniques"), str)
                            else r.get("Techniques")
                        )]
        without_tech = [r for r in pool if r not in with_tech]
        chosen = _fewshot_rng.sample(with_tech, min(2, len(with_tech)))
        if without_tech:
            chosen.append(_fewshot_rng.choice(without_tech))
        return chosen[:3]


def format_few_shot_block(examples: list, task: str) -> str:
    """Format labelled examples as in-context demonstrations."""
    lines = ["Here are examples of the task:\n"]
    for i, ex in enumerate(examples, 1):
        parent = ex.get("ParentText", "")
        child  = ex.get("ChildText", "")
        lines.append(f"--- Example {i} ---")

        if task == "interaction":
            score   = float(ex.get("InteractionScore", 0.0))
            snapped = min(SCORE_ENUM, key=lambda k: abs(k - score))
            label   = SCORE_TO_LABEL[snapped]
            lines.append(f"PARENT COMMENT:\n{parent}")
            lines.append(f"CHILD COMMENT:\n{child}")
            lines.append(f"Answer: {{\"score\": {snapped}, \"label\": \"{label}\"}}\n")

        elif task == "stance":
            stance = int(ex.get("StanceLabel", 1))
            sname  = STANCE_NAMES[stance]
            lines.append(f"PARENT COMMENT:\n{parent}")
            lines.append(f"CHILD COMMENT:\n{child}")
            lines.append(f"Answer: {{\"label\": {stance}}}  # {sname}\n")

        else:  # techniques
            raw = ex.get("Techniques", [])
            techs = json.loads(raw) if isinstance(raw, str) else (raw or [])
            lines.append(f"COMMENT:\n{child}")
            lines.append(f"Answer: {{\"techniques\": {json.dumps(techs)}}}\n")

    lines.append("Now classify the following:\n")
    return "\n".join(lines)


# Prompt builders
def build_interaction_prompt(pair, shot_mode, few_shot_block,
                             condition: str = "full_instructions", row=None):
    """Build interaction user prompt. Optionally appends stance/technique context.

    condition:
      "full_instructions"                  — text only (B)
      "full_instructions+stance"           — + child & parent predicted stance (B+S)
      "full_instructions+stance+techniques"— + stance + technique list (B+S+T)

    row: features.parquet row (pandas Series) — needed for +stance/+techniques context.
    """
    lines = []
    if shot_mode == "few_shot" and few_shot_block:
        lines.append(few_shot_block)
    lines.append(f"PARENT COMMENT:\n{pair.get('ParentText', '')}")
    lines.append(f"\nCHILD COMMENT:\n{pair.get('ChildText', '')}")

    # Append predicted context for enriched conditions
    if condition in ("full_instructions+stance", "full_instructions+stance+techniques") \
            and row is not None:
        _cs_val = row.get("child_stance", row.get("StanceLabel", None))
        child_stance  = int(_cs_val) if _cs_val is not None else 1
        parent_stance = int(row.get("parent_stance", 1))
        lines.append(f"\nChild stance toward topic: {STANCE_NAMES.get(child_stance, 'Neutral')}")
        lines.append(f"Parent stance toward topic: {STANCE_NAMES.get(parent_stance, 'Neutral')}")

    if condition == "full_instructions+stance+techniques" and row is not None:
        raw_techs = row.get("Techniques", "[]")
        techs = json.loads(raw_techs) if isinstance(raw_techs, str) else (raw_techs or [])
        tech_str = ", ".join(techs) if techs else "none"
        lines.append(f"Rhetorical techniques in child: {tech_str}")

    lines.append("\nClassify the interaction quality.")
    return "\n".join(lines)


def build_stance_prompt(pair, shot_mode, few_shot_block):
    lines = []
    if shot_mode == "few_shot" and few_shot_block:
        lines.append(few_shot_block)
    lines.append(f"PARENT COMMENT:\n{pair.get('ParentText', '')}")
    lines.append(f"\nCHILD COMMENT:\n{pair.get('ChildText', '')}")
    lines.append("\nClassify the stance of the child comment toward the video topic.")
    return "\n".join(lines)


def build_techniques_prompt(pair, shot_mode, few_shot_block):
    lines = []
    if shot_mode == "few_shot" and few_shot_block:
        lines.append(few_shot_block)
    lines.append(f"COMMENT:\n{pair.get('ChildText', '')}")
    lines.append("\nDetect propaganda techniques.")
    return "\n".join(lines)


# API call + parsing
def call_llm(client, model, system, user_msg, max_tokens=200) -> str:
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=30,
            )
            if not resp.choices:
                raise ValueError("Empty choices in API response")
            content = resp.choices[0].message.content
            return content.strip() if content is not None else ""
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt, 60)  # cap at 60s
            print(f"    API error (attempt {attempt}): {e} — retrying in {wait}s")
            time.sleep(wait)


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find outermost { ... } allowing nested brackets
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    start = -1
    return None


def parse_interaction(text: str) -> dict:
    obj = _extract_json(text)
    if obj:
        try:
            raw_score = float(obj.get("score", 0.0))
            score = min(SCORE_ENUM, key=lambda s: abs(s - raw_score))
            label = str(obj.get("label", "Neutral/Rephrase"))
            return {"score": score, "label": label}
        except Exception:
            pass
    return {"score": 0.0, "label": "Neutral/Rephrase"}


def parse_stance(text: str) -> int:
    obj = _extract_json(text)
    if obj:
        try:
            raw = obj.get("label", 1)
            return STANCE_LABELS.get(str(raw).lower(), int(raw))
        except Exception:
            pass
    digit = re.search(r"\b[012]\b", text)
    return int(digit.group()) if digit else 1


def parse_techniques(text: str) -> list:
    obj = _extract_json(text)
    if obj:
        techs = obj.get("techniques", [])
        if isinstance(techs, list):
            return [t for t in techs if t in VALID_TECHNIQUES_SET]
    return []


# Evaluation
def eval_predictions(task, records):
    if task == "interaction":
        y_true = [SCORE_TO_CLASS[min(SCORE_ENUM, key=lambda k: abs(k - float(r["y_true"])))]
                  for r in records]
        y_pred = [SCORE_TO_CLASS[min(SCORE_ENUM, key=lambda k: abs(k - float(r["y_pred_score"])))]
                  for r in records]
        scores_true = np.array([CLASS_TO_SCORE[c] for c in y_true])
        scores_pred = np.array([CLASS_TO_SCORE[c] for c in y_pred])
        mae = float(np.mean(np.abs(scores_true - scores_pred)))
        pr  = float(pearsonr(scores_true, scores_pred)[0]) \
              if len(set(scores_pred.tolist())) > 1 else 0.0
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
            "mae":             round(mae, 4),
            "pearson_r":       round(pr, 4),
            "per_class_f1":    [round(v, 4) for v in
                                f1_score(y_true, y_pred, average=None,
                                         zero_division=0).tolist()],
        }
    elif task == "stance":
        y_true = [int(r["y_true"]) for r in records]
        y_pred = [int(r["y_pred"]) for r in records]
        per = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        return {
            "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro",
                                                    zero_division=0)), 4),
            "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(precision_score(y_true, y_pred, average="macro",
                                                           zero_division=0)), 4),
            "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro",
                                                        zero_division=0)), 4),
            "kappa":           round(float(cohen_kappa_score(y_true, y_pred)), 4),
            "per_class_f1":    {"Against": round(float(per[0]), 4),
                                "Neutral":  round(float(per[1]), 4),
                                "Support":  round(float(per[2]), 4)},
        }
    else:  # techniques
        mlb   = MultiLabelBinarizer(classes=VALID_TECHNIQUES)
        g_bin = mlb.fit_transform([r["y_true"] for r in records])
        p_bin = mlb.transform([r["y_pred"]  for r in records])
        return {
            "macro_f1":             round(float(f1_score(g_bin, p_bin, average="macro",
                                                         zero_division=0)), 4),
            "exact_match_accuracy": round(float(accuracy_score(g_bin, p_bin)), 4),
            "macro_precision":      round(float(precision_score(g_bin, p_bin, average="macro",
                                                                zero_division=0)), 4),
            "macro_recall":         round(float(recall_score(g_bin, p_bin, average="macro",
                                                             zero_division=0)), 4),
            "hamming_loss":         round(float(hamming_loss(g_bin, p_bin)), 4),
            "per_label_f1":         {t: round(float(f), 4) for t, f in
                                     zip(VALID_TECHNIQUES,
                                         f1_score(g_bin, p_bin, average=None,
                                                  zero_division=0))},
        }


# Main per-task runner
def run_task(client, model_name, model_label, df, raw_pairs_map,
             task, condition, shot_mode, few_shot_examples, all_metrics):
    key       = f"{task}_{model_label}_{condition}_{shot_mode}"
    pred_file = PRED_DIR / f"{key}.jsonl"
    print(f"\n  [{key}]")

    # Pre-build the few-shot block(s).
    # Stance uses topic-keyed blocks; interaction/techniques use a single block.
    few_shot_block = ""
    stance_few_shot_blocks: dict[str, str] = {}
    if shot_mode == "few_shot":
        if task == "stance" and isinstance(few_shot_examples, dict):
            for t, exs in few_shot_examples.items():
                if exs:
                    stance_few_shot_blocks[t] = format_few_shot_block(exs, "stance")
        elif few_shot_examples:
            few_shot_block = format_few_shot_block(few_shot_examples, task)

    # All conditions use full_instructions prompt; +stance/+techniques add context to user_msg
    if task == "interaction":
        system_prompt = INTERACTION_SYSTEM_FULL
    elif task == "techniques":
        system_prompt = TECHNIQUES_SYSTEM_FULL
    else:
        system_prompt = None  # resolved per-pair for stance (topic-dependent)

    # Already-done PairIDs
    done_ids = set()
    if pred_file.exists():
        for r in load_jsonl(str(pred_file)):
            done_ids.add(r["PairID"])

    test_df = df[df["split"] == "test"].copy()
    pending = [(row, raw_pairs_map[row["PairID"]])
               for _, row in test_df.iterrows()
               if row["PairID"] not in done_ids
               and row["PairID"] in raw_pairs_map]

    print(f"    Pending: {len(pending)} / {len(test_df)} (done: {len(done_ids)})")

    for row, raw in tqdm(pending, desc=f"    {key}", unit="pair", leave=False):
        pid = row["PairID"]

        if task == "interaction":
            user_msg = build_interaction_prompt(raw, shot_mode, few_shot_block,
                                               condition=condition, row=row)
            text   = call_llm(client, model_name, system_prompt, user_msg)
            parsed = parse_interaction(text)
            true_score = float(row["InteractionScore"]) if not pd.isna(row.get("InteractionScore")) else 0.0
            record = {"PairID": pid, "y_true": true_score,
                      "y_pred_score": parsed["score"], "y_pred_label": parsed["label"],
                      "condition": condition, "shot_mode": shot_mode}

        elif task == "stance":
            # Pick topic-specific full prompt per pair
            topic = raw.get("Category") or raw.get("Topic") or raw.get("category") or "migration"
            topic = str(topic).lower()
            sp = _build_stance_full(topic) if condition == "full_instructions" else STANCE_SYSTEM
            # Use topic-matched few-shot block if available, else fall back to generic
            pair_block = stance_few_shot_blocks.get(topic, few_shot_block)
            user_msg = build_stance_prompt(raw, shot_mode, pair_block)
            text   = call_llm(client, model_name, sp, user_msg)
            parsed = parse_stance(text)
            true_label = int(row["StanceLabel"]) if not pd.isna(row.get("StanceLabel")) else 1
            record = {"PairID": pid, "y_true": true_label, "y_pred": parsed,
                      "condition": condition, "shot_mode": shot_mode}

        else:  # techniques
            user_msg = build_techniques_prompt(raw, shot_mode, few_shot_block)
            text   = call_llm(client, model_name, system_prompt, user_msg)
            parsed = parse_techniques(text)
            true_techs = json.loads(row["Techniques"]) if isinstance(row.get("Techniques"), str) else []
            record = {"PairID": pid, "y_true": true_techs, "y_pred": parsed,
                      "condition": condition, "shot_mode": shot_mode}

        append_jsonl(record, str(pred_file))
        done_ids.add(pid)
        time.sleep(CALL_DELAY)

    # Evaluate full pred file
    all_records = list(load_jsonl(str(pred_file))) if pred_file.exists() else []
    if all_records:
        metrics = eval_predictions(task, all_records)
        (all_metrics
         .setdefault(task, {})
         .setdefault(model_label, {})
         .setdefault(condition, {})[shot_mode]) = metrics

        # Print full metrics to console
        print(f"    -- Results [{model_label} / {task} / {condition} / {shot_mode}] --")
        if task == "interaction":
            print(f"      macro_f1        : {metrics['macro_f1']}")
            print(f"      accuracy        : {metrics['accuracy']}")
            print(f"      macro_precision : {metrics['macro_precision']}")
            print(f"      macro_recall    : {metrics['macro_recall']}")
            print(f"      kappa_quadratic : {metrics['kappa_quadratic']}")
            print(f"      mae             : {metrics['mae']}")
            print(f"      pearson_r       : {metrics['pearson_r']}")
            labels = ["Strong-Neg", "Weak-Neg", "Neutral", "Weak-Pos", "Strong-Pos"]
            for lbl, f1 in zip(labels, metrics["per_class_f1"]):
                print(f"        {lbl:<12}: F1={f1}")
        elif task == "stance":
            print(f"      macro_f1        : {metrics['macro_f1']}")
            print(f"      accuracy        : {metrics['accuracy']}")
            print(f"      macro_precision : {metrics['macro_precision']}")
            print(f"      macro_recall    : {metrics['macro_recall']}")
            print(f"      kappa           : {metrics['kappa']}")
            for cls, f1 in metrics["per_class_f1"].items():
                print(f"        {cls:<8}: F1={f1}")
        else:  # techniques
            print(f"      macro_f1             : {metrics['macro_f1']}")
            print(f"      exact_match_accuracy : {metrics['exact_match_accuracy']}")
            print(f"      macro_precision      : {metrics['macro_precision']}")
            print(f"      macro_recall         : {metrics['macro_recall']}")
            print(f"      hamming_loss         : {metrics['hamming_loss']}")
            for lbl, f1 in metrics["per_label_f1"].items():
                print(f"        {lbl:<40}: F1={f1}")


# Prompt & few-shot persistence
def save_prompts_and_fewshots(few_shot_examples: dict) -> None:
    """Persist all system prompts and few-shot material to 17_llm_results/.

    Saved artefacts
    ---------------
    prompts_and_fewshots/
      system_prompts/
        interaction_system_prompt.txt          full annotation-grade system prompt
        stance_migration_system_prompt.txt
        stance_climate_system_prompt.txt
        techniques_system_prompt.txt
      few_shot_examples/
        <task>_fewshot_examples.json           raw gold records (PairID, texts, labels)
      few_shot_blocks/
        <task>_fewshot_block.txt               formatted in-context demonstration text
      full_prompt_examples/
        <task>_zero_shot_example.txt           complete request as sent to the LLM
        <task>_few_shot_example.txt            (system prompt + user message shown together)
    """
    base = OUT_DIR / "prompts_and_fewshots"
    for sub in ["system_prompts", "few_shot_examples",
                "few_shot_blocks", "full_prompt_examples"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

    # 1. System prompts
    system_prompts = {
        "interaction_system_prompt":     INTERACTION_SYSTEM_FULL,
        "stance_migration_system_prompt": _build_stance_full("migration"),
        "stance_climate_system_prompt":  _build_stance_full("climate"),
        "techniques_system_prompt":      TECHNIQUES_SYSTEM_FULL,
    }
    for name, text in system_prompts.items():
        path = base / "system_prompts" / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"    Saved -> {path}")

    # 2. Few-shot examples (raw JSON) — stance is topic-keyed dict
    def _serialise_examples(examples, task):
        clean = []
        for ex in examples:
            entry = {
                "PairID":           ex.get("PairID", ""),
                "VideoID":          ex.get("VideoID", ""),
                "Topic":            ex.get("Topic", "") or ex.get("category", ""),
                "ParentText":       ex.get("ParentText", ""),
                "ChildText":        ex.get("ChildText", ""),
                "InteractionScore": ex.get("InteractionScore"),
                "InteractionLabel": ex.get("InteractionLabel", ""),
                "StanceLabel":      ex.get("StanceLabel"),
                "Techniques":       ex.get("Techniques", "[]"),
            }
            if task == "interaction" and entry["InteractionScore"] is not None:
                snapped = min(SCORE_ENUM, key=lambda k: abs(k - float(entry["InteractionScore"])))
                entry["InteractionScore_snapped"] = snapped
                entry["InteractionLabel_resolved"] = SCORE_TO_LABEL[snapped]
            if task == "stance" and entry["StanceLabel"] is not None:
                entry["StanceName"] = STANCE_NAMES.get(int(entry["StanceLabel"]), "?")
            clean.append(entry)
        return clean

    for task, examples_or_dict in few_shot_examples.items():
        if isinstance(examples_or_dict, dict):
            # topic-keyed (stance)
            for subtopic, examples in examples_or_dict.items():
                clean = _serialise_examples(examples, task)
                path = base / "few_shot_examples" / f"{task}_{subtopic}_fewshot_examples.json"
                path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"    Saved -> {path}")
        else:
            clean = _serialise_examples(examples_or_dict, task)
            path = base / "few_shot_examples" / f"{task}_fewshot_examples.json"
            path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    Saved -> {path}")

    # 3. Formatted few-shot blocks (plain text)
    for task, examples_or_dict in few_shot_examples.items():
        if isinstance(examples_or_dict, dict):
            for subtopic, examples in examples_or_dict.items():
                block = format_few_shot_block(examples, task)
                path  = base / "few_shot_blocks" / f"{task}_{subtopic}_fewshot_block.txt"
                path.write_text(block, encoding="utf-8")
                print(f"    Saved -> {path}")
        else:
            block = format_few_shot_block(examples_or_dict, task)
            path  = base / "few_shot_blocks" / f"{task}_fewshot_block.txt"
            path.write_text(block, encoding="utf-8")
            print(f"    Saved -> {path}")

    # 4. Full prompt examples (system + user as sent to the LLM)
    for task, examples_or_dict in few_shot_examples.items():
        # For stance use migration examples as the demo; for others use the flat list
        if isinstance(examples_or_dict, dict):
            examples_for_demo = {t: exs for t, exs in examples_or_dict.items()}
        else:
            examples_for_demo = {None: examples_or_dict}

        for subtopic, examples in examples_for_demo.items():
            if not examples:
                continue
            demo_pair = examples[0]

            dummy_row = {
                "child_stance":  demo_pair.get("StanceLabel", 1) if demo_pair.get("StanceLabel") is not None else 1,
                "parent_stance": 1,
                "Techniques":    demo_pair.get("Techniques", "[]"),
            }

            fsblock = format_few_shot_block(examples, task)

            for shot_mode in ["zero_shot", "few_shot"]:
                fsb = fsblock if shot_mode == "few_shot" else ""

                if task == "interaction":
                    for condition in [
                        "full_instructions",
                        "full_instructions+stance",
                        "full_instructions+stance+techniques",
                    ]:
                        user_msg = build_interaction_prompt(
                            demo_pair, shot_mode, fsb,
                            condition=condition, row=dummy_row
                        )
                        cond_short = condition.replace("full_instructions", "B") \
                                              .replace("+stance+techniques", "+S+T") \
                                              .replace("+stance", "+S")
                        rendered = (
                            f"{'='*72}\n"
                            f"TASK: {task}  |  CONDITION: {cond_short}  |  SHOT: {shot_mode}\n"
                            f"{'='*72}\n\n"
                            f"--- SYSTEM PROMPT ---\n{INTERACTION_SYSTEM_FULL}\n\n"
                            f"--- USER MESSAGE ---\n{user_msg}\n"
                        )
                        fname = f"{task}_{cond_short}_{shot_mode}_example.txt"
                        path  = base / "full_prompt_examples" / fname
                        path.write_text(rendered, encoding="utf-8")
                        print(f"    Saved -> {path}")

                elif task == "stance":
                    topic = subtopic or "migration"
                    sp       = _build_stance_full(topic)
                    user_msg = build_stance_prompt(demo_pair, shot_mode, fsb)
                    rendered = (
                        f"{'='*72}\n"
                        f"TASK: {task}  |  TOPIC: {topic}  |  SHOT: {shot_mode}\n"
                        f"{'='*72}\n\n"
                        f"--- SYSTEM PROMPT ---\n{sp}\n\n"
                        f"--- USER MESSAGE ---\n{user_msg}\n"
                    )
                    fname = f"{task}_{topic}_{shot_mode}_example.txt"
                    path  = base / "full_prompt_examples" / fname
                    path.write_text(rendered, encoding="utf-8")
                    print(f"    Saved -> {path}")

            else:  # techniques
                user_msg = build_techniques_prompt(demo_pair, shot_mode, fsb)
                rendered = (
                    f"{'='*72}\n"
                    f"TASK: {task}  |  SHOT: {shot_mode}\n"
                    f"{'='*72}\n\n"
                    f"--- SYSTEM PROMPT ---\n{TECHNIQUES_SYSTEM_FULL}\n\n"
                    f"--- USER MESSAGE ---\n{user_msg}\n"
                )
                fname = f"{task}_{shot_mode}_example.txt"
                path  = base / "full_prompt_examples" / fname
                path.write_text(rendered, encoding="utf-8")
                print(f"    Saved -> {path}")

    print(f"  Prompts and few-shot material saved to {base}/")


# Main
def main():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        print("OPENROUTER_API_KEY not set in .env — aborting.")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/zaheralkaei/german-yt-pipeline",
            "X-Title": "German YT Comment Analysis",
        },
    )

    df = pd.read_parquet(FEAT_DIR / "features.parquet")

    # Restrict to oracle-valid pairs — same test set as steps 15, 16, 19
    df = df[df["StanceLabel"].notna()].copy()
    print(f"Oracle-filtered dataset: {len(df)} pairs "
          f"(test={len(df[df['split']=='test'])})")

    # Load raw pair texts — LLM first, gold overwrites.
    # start with the LLM record as base (carries fields like Topic),
    # then apply gold fields on top via dict.update() so gold values always win
    # but LLM-only fields (e.g. Topic) are preserved for gold pairs.
    raw_pairs_map = {}
    for path in [
        Path("10_llm_annotations/llm_annotated_subset.jsonl"),
        Path("11_llm_annotations/llm_annotated_rest.jsonl"),
        Path("09_manual_annotations/gold_standard.jsonl"),
    ]:
        if path.exists():
            for r in load_jsonl(str(path)):
                pid = r["PairID"]
                if pid in raw_pairs_map:
                    merged = dict(raw_pairs_map[pid])
                    merged.update(r)
                    raw_pairs_map[pid] = merged
                else:
                    raw_pairs_map[pid] = r
        else:
            print(f"  Warning: {path} not found, skipping.")
    print(f"Loaded {len(raw_pairs_map)} raw pair records.")

    # Load split index for few-shot pool filtering
    split_index_path = FEAT_DIR / "split_index.json"
    with open(split_index_path, encoding="utf-8") as f:
        split_index = json.load(f)

    # Build few-shot pool from gold train pairs only
    gold_path = Path("09_manual_annotations/gold_standard.jsonl")
    few_shot_pool = build_few_shot_pool(gold_path, split_index)
    print(f"Few-shot pool (gold train): {len(few_shot_pool)} pairs")

    # Pre-select few-shot examples per task.
    # MANUAL_FEWSHOT_IDS takes precedence; falls back to FEWSHOT_SEED sampling.
    # Stance is topic-split: few_shot_examples["stance"] = {"migration": [...], "climate": [...]}
    few_shot_examples = {}
    for task in TASK_CONDITIONS:
        if task == "stance":
            few_shot_examples[task] = {
                t: select_few_shot_examples(
                    few_shot_pool, "stance",
                    manual_ids=MANUAL_FEWSHOT_IDS.get(f"stance_{t}"),
                    topic=t,
                )
                for t in ["migration", "climate"]
            }
        else:
            few_shot_examples[task] = select_few_shot_examples(
                few_shot_pool, task,
                manual_ids=MANUAL_FEWSHOT_IDS.get(task),
            )
    for task, exs in few_shot_examples.items():
        if isinstance(exs, dict):
            for t, te in exs.items():
                print(f"  {task}/{t}: {len(te)} few-shot examples selected")
        else:
            print(f"  {task}: {len(exs)} few-shot examples selected")

    # Persist all prompts and few-shot material
    print("\nSaving prompts and few-shot material ...")
    save_prompts_and_fewshots(few_shot_examples)

    # Load or init metrics
    metrics_file = OUT_DIR / "metrics.json"
    all_metrics  = {}
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            all_metrics = json.load(f)

    for model_label, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  [{model_label}]")
        print(f"{'='*60}")
        for task, conditions in TASK_CONDITIONS.items():
            for condition in conditions:
                for shot_mode in SHOT_MODES:
                    run_task(client, model_name, model_label, df,
                             raw_pairs_map, task, condition, shot_mode,
                             few_shot_examples[task], all_metrics)
                    with open(metrics_file, "w", encoding="utf-8") as f:
                        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Summary table
    print("\n" + "=" * 72)
    print("LLM RESULTS SUMMARY  (macro-F1, test set)")
    print("=" * 72)
    for task, conditions in TASK_CONDITIONS.items():
        print(f"\n{task.upper()}")
        header_parts = [f"{'Condition':<28}", f"{'Shot':<10}"]
        for ml in MODELS:
            header_parts.append(f"{ml:>12}")
        print("  " + "  ".join(header_parts))
        for condition in conditions:
            for shot_mode in SHOT_MODES:
                row_parts = [f"{condition:<28}", f"{shot_mode:<10}"]
                for ml in MODELS:
                    v = (all_metrics
                         .get(task, {}).get(ml, {})
                         .get(condition, {}).get(shot_mode, {})
                         .get("macro_f1", "-"))
                    row_parts.append(f"{v:>12.4f}" if isinstance(v, float) else f"{'—':>12}")
                print("  " + "  ".join(row_parts))

    print(f"\nMetrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
