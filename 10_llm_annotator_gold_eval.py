"""
Phase 1: Annotate the manual gold standard with the LLM, then evaluate
against the manual labels to judge prompt/model quality.

Outputs:
  10_llm_annotations/llm_annotated_subset.jsonl  — LLM annotations for gold pairs
  10_llm_annotations/evaluation_report.json      — metrics report

Metrics:
  - Interaction (5-class ordinal): quadratic-weighted and unweighted kappa, MAE, Pearson r
  - Stance (3-class): unweighted kappa, per-class F1, macro F1
  - Techniques (14-label multi-label): per-label F1, macro F1, Hamming loss
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import anthropic
from sklearn.metrics import (
    cohen_kappa_score, f1_score, classification_report, hamming_loss,
)
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import pearsonr
from utils.jsonl_io import load_jsonl, append_jsonl

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

PAIRS_FILE  = Path("./08_pairs/sampled_pairs.jsonl")
GOLD_FILE   = Path("./09_manual_annotations/gold_standard.jsonl")
OUT_DIR     = Path("./10_llm_annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBSET_OUT  = OUT_DIR / "llm_annotated_subset.jsonl"
REPORT_FILE = OUT_DIR / "evaluation_report.json"

PROMPT_VERSION = "v1.0"
CALL_DELAY = 1  # seconds between API calls

VALID_TECHNIQUES = {
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice',
    'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy',
    'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
    'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling',
    'Repetition', 'Slogans/Thought-terminating_Cliches',
    'Whataboutism,Straw_Men', 'Appeal_to_Time',
}

SCORE_ORDINAL = [-1.0, -0.5, 0.0, 0.5, 1.0]

INTERACTION_TOOL = {
    "name": "classify_interaction",
    "description": "Classify the interaction quality of a comment pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "enum": [1.0, 0.5, 0.0, -0.5, -1.0],
                "description": "+1=Constructive Disagreement, +0.5=Constructive Agreement, 0=Neutral/Rephrase, -0.5=Destructive Agreement, -1=Destructive Disagreement",
            },
            "label": {
                "type": "string",
                "enum": [
                    "Constructive Disagreement", "Constructive Agreement",
                    "Neutral/Rephrase", "Destructive Agreement", "Destructive Disagreement",
                ],
            },
            "reasoning": {"type": "string"},
        },
        "required": ["score", "label", "reasoning"],
    },
}


# Prompt loaders 

def _load(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("#"):
        return ""
    return text

def load_prompts() -> dict:
    return {
        "stance_migration":        _load(Path("prompts/stance_migration.txt")),
        "stance_climate":          _load(Path("prompts/stance_climate.txt")),
        "parent_stance_migration": _load(Path("prompts/parent_stance_migration.txt")),
        "parent_stance_climate":   _load(Path("prompts/parent_stance_climate.txt")),
        "techniques":              _load(Path("prompts/techniques.txt")),
        "interaction":             _load(Path("prompts/interaction.txt")),
    }

def load_video_topics() -> dict:
    mapping = {}
    if not PAIRS_FILE.exists():
        return mapping
    for line in PAIRS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = __import__("json").loads(line)
            mapping[p["VideoID"]] = p.get("category", "migration")
        except Exception:
            pass
    return mapping


# The annotation functions

def annotate_stance(client, parent_text: str, child_text: str, system_prompt: str,
                    target: str = "child"):
    if not system_prompt:
        return None
    # For parent stance: classify only the parent comment (child is provided for context)
    if target == "parent":
        user_msg = f"Comment to classify: {parent_text}\nContext (response): {child_text}"
    else:
        user_msg = f"Parent Comment: {parent_text}\nResponse Comment: {child_text}"
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=10,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            digit = resp.content[0].text.strip()
            if digit in ("0", "1", "2"):
                return int(digit)
        except Exception as e:
            print(f"    stance error [{target}] (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None  # flagged as AnnotationFailed in annotate_pair


def annotate_techniques(client, child_text: str, system_prompt: str) -> list:
    if not system_prompt:
        return []
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": child_text}],
            )
            text = resp.content[0].text.strip()
            if "no propaganda detected" in text.lower():
                return []
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [l for l in lines if l in VALID_TECHNIQUES]
        except Exception as e:
            print(f"    techniques error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None  # flagged as AnnotationFailed in annotate_pair


def annotate_interaction(client, parent_text: str, child_text: str, system_prompt: str) -> dict:
    if not system_prompt:
        return {"score": None, "label": None, "reasoning": None}
    user_msg = f"PARENT COMMENT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}"
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=512,
                system=system_prompt,
                tools=[INTERACTION_TOOL],
                tool_choice={"type": "tool", "name": "classify_interaction"},
                messages=[{"role": "user", "content": user_msg}],
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return block.input
        except Exception as e:
            print(f"    interaction error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return {"score": None, "label": None, "reasoning": None}  # flagged as AnnotationFailed


def annotate_pair(client, pair: dict, prompts: dict, video_topics: dict) -> dict:
    topic = video_topics.get(pair["VideoID"], pair.get("category", "migration"))
    stance_prompt        = (prompts["stance_climate"]
                            if topic == "climate" else prompts["stance_migration"])
    parent_stance_prompt = (prompts["parent_stance_climate"]
                            if topic == "climate" else prompts["parent_stance_migration"])

    child_stance  = annotate_stance(client, pair["ParentText"], pair["ChildText"],
                                    stance_prompt, target="child")
    time.sleep(CALL_DELAY)
    parent_stance = annotate_stance(client, pair["ParentText"], pair["ChildText"],
                                    parent_stance_prompt, target="parent")
    time.sleep(CALL_DELAY)
    techs    = annotate_techniques(client, pair["ChildText"], prompts["techniques"])
    time.sleep(CALL_DELAY)
    interact = annotate_interaction(client, pair["ParentText"], pair["ChildText"],
                                    prompts["interaction"])
    time.sleep(CALL_DELAY)

    failed = ((child_stance is None) or (parent_stance is None) or
              (techs is None) or (interact.get("score") is None))
    return {
        **pair,
        "StanceLabel":          child_stance,
        "ParentStanceLabel":    parent_stance,
        "Techniques":           techs if techs is not None else [],
        "InteractionScore":     interact.get("score"),
        "InteractionLabel":     interact.get("label"),
        "InteractionReasoning": interact.get("reasoning"),
        "Topic":                topic,
        "PromptVersion":        PROMPT_VERSION,
        "ModelUsed":            CLAUDE_MODEL,
        "AnnotatedAt":          datetime.now(timezone.utc).isoformat(),
        "AnnotationFailed":     failed,
    }


# Annotation loop

def run_annotation(client, gold_pairs: list, prompts: dict, video_topics: dict):
    # Resume: skip pairs that are already fully annotated
    done_complete = set()
    if SUBSET_OUT.exists():
        for r in load_jsonl(str(SUBSET_OUT)):
            if r.get("ParentStanceLabel") is not None:
                done_complete.add(r["PairID"])
    pending = [p for p in gold_pairs if p["PairID"] not in done_complete]
    print(f"\nAnnotation: {len(pending)} pairs to annotate ({len(done_complete)} already complete)")

    for i, pair in enumerate(pending, 1):
        result = annotate_pair(client, pair, prompts, video_topics)
        append_jsonl(result, str(SUBSET_OUT))
        if i % 10 == 0 or i == len(pending):
            ok = "OK" if result.get("InteractionScore") is not None else "PARTIAL"
            print(f"  [{i}/{len(pending)}] {pair['PairID']} — {ok}")


# Evaluation

def align(gold: list, llm: list):
    llm_index = {r["PairID"]: r for r in llm}
    aligned_gold, aligned_llm = [], []
    missing = 0
    for g in gold:
        l = llm_index.get(g["PairID"])
        if l:
            aligned_gold.append(g)
            aligned_llm.append(l)
        else:
            missing += 1
    if missing:
        print(f"[WARNING] {missing} gold pairs not found in LLM output.")
    return aligned_gold, aligned_llm


def _to_ordinal(score: float) -> int:
    return min(range(len(SCORE_ORDINAL)), key=lambda i: abs(SCORE_ORDINAL[i] - score))


def eval_interaction(gold: list, llm: list) -> dict:
    g_raw = [float(r.get("InteractionScore") or 0.0) for r in gold]
    l_raw = [float(r.get("InteractionScore") or 0.0) for r in llm]
    g_ord = [_to_ordinal(s) for s in g_raw]
    l_ord = [_to_ordinal(s) for s in l_raw]
    kappa_q = cohen_kappa_score(g_ord, l_ord, weights="quadratic")
    kappa_u = cohen_kappa_score(g_ord, l_ord, weights=None)
    mae     = float(np.mean(np.abs(np.array(g_raw) - np.array(l_raw))))
    r, p    = pearsonr(g_raw, l_raw)
    return {
        "kappa_quadratic":  round(kappa_q, 4),
        "kappa_unweighted": round(kappa_u, 4),
        "mae":              round(mae, 4),
        "pearson_r":        round(float(r), 4),
        "pearson_p":        round(float(p), 6),
    }


def _eval_stance_labels(gold_labels: list, llm_labels: list) -> dict:
    """Shared computation for child and parent stance evaluation."""
    labels      = [0, 1, 2]
    label_names = ["Against", "Neutral", "Support"]
    kappa  = cohen_kappa_score(gold_labels, llm_labels)
    report = classification_report(gold_labels, llm_labels, labels=labels,
                                   target_names=label_names,
                                   output_dict=True, zero_division=0)
    per_class = {n: round(report[n]["f1-score"], 4) for n in label_names}
    support   = {n: int(report[n]["support"])       for n in label_names}
    return {
        "kappa":        round(kappa, 4),
        "per_class_f1": per_class,
        "macro_f1":     round(report["macro avg"]["f1-score"], 4),
        "support":      support,
    }


def eval_stance(gold: list, llm: list) -> dict:
    """Evaluate child stance (StanceLabel)."""
    g = [int(r.get("StanceLabel"))  if r.get("StanceLabel")  is not None else 1 for r in gold]
    l = [int(r.get("StanceLabel"))  if r.get("StanceLabel")  is not None else 1 for r in llm]
    return _eval_stance_labels(g, l)


def eval_parent_stance(gold: list, llm: list) -> dict:
    """Evaluate parent stance (ParentStanceLabel).
    Only pairs where both gold and LLM have a non-None ParentStanceLabel are included.
    """
    pairs = [(r_g, r_l) for r_g, r_l in zip(gold, llm)
             if r_g.get("ParentStanceLabel") is not None
             and r_l.get("ParentStanceLabel") is not None]
    if not pairs:
        return {"kappa": None, "per_class_f1": {}, "macro_f1": None,
                "support": {}, "n_evaluated": 0}
    g = [int(r_g["ParentStanceLabel"]) for r_g, _ in pairs]
    l = [int(r_l["ParentStanceLabel"]) for _, r_l in pairs]
    result = _eval_stance_labels(g, l)
    result["n_evaluated"] = len(pairs)
    return result


def eval_techniques(gold: list, llm: list) -> dict:
    mlb   = MultiLabelBinarizer(classes=sorted(VALID_TECHNIQUES))
    g_bin = mlb.fit_transform([r.get("Techniques") or [] for r in gold])
    l_bin = mlb.transform(   [r.get("Techniques") or [] for r in llm])
    per_arr   = f1_score(g_bin, l_bin, average=None, zero_division=0)
    per_label = {t: round(float(f), 4) for t, f in zip(mlb.classes_, per_arr)}
    macro     = round(float(f1_score(g_bin, l_bin, average="macro", zero_division=0)), 4)
    h_loss    = round(float(hamming_loss(g_bin, l_bin)), 4)
    return {"per_label_f1": per_label, "macro_f1": macro, "hamming_loss": h_loss}


def print_summary(report: dict):
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    i = report["interaction"]
    print(f"Interaction  kappa_quadratic={i['kappa_quadratic']}  kappa_unweighted={i['kappa_unweighted']}  MAE={i['mae']}  r={i['pearson_r']}")
    s = report["stance"]
    print(f"Stance(child)  kappa={s['kappa']}  macro-F1={s['macro_f1']}")
    for cls in ["Against", "Neutral", "Support"]:
        print(f"  {cls}: F1={s['per_class_f1'].get(cls, 0.0)}  (n={s['support'].get(cls, 0)})")
    ps = report.get("parent_stance", {})
    n_ps = ps.get("n_evaluated", 0)
    print(f"Stance(parent) kappa={ps.get('kappa')}  macro-F1={ps.get('macro_f1')}  (n={n_ps})")
    for cls in ["Against", "Neutral", "Support"]:
        print(f"  {cls}: F1={ps.get('per_class_f1', {}).get(cls, 0.0)}  (n={ps.get('support', {}).get(cls, 0)})")
    te = report["techniques"]
    print(f"Techniques   macro-F1={te['macro_f1']}  Hamming={te['hamming_loss']}")
    print("\nPer-label F1:")
    for label, f1 in te["per_label_f1"].items():
        bar = "█" * int(f1 * 20)
        print(f"  {label:<42} {f1:.2f}  {bar}")
    print("=" * 55)
    print("done")


def run_evaluation():
    gold = load_jsonl(str(GOLD_FILE))
    llm  = load_jsonl(str(SUBSET_OUT))
    aligned_gold, aligned_llm = align(gold, llm)
    n = len(aligned_gold)
    print(f"\nAligned {n} pairs for evaluation.")

    report = {
        "metadata": {
            "n_pairs":         n,
            "model":           CLAUDE_MODEL,
            "prompt_version":  PROMPT_VERSION,
            "evaluation_date": datetime.now(timezone.utc).isoformat(),
        },
        "interaction":    eval_interaction(aligned_gold, aligned_llm),
        "stance":         eval_stance(aligned_gold, aligned_llm),
        "parent_stance":  eval_parent_stance(aligned_gold, aligned_llm),
        "techniques":     eval_techniques(aligned_gold, aligned_llm),
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved in {REPORT_FILE}")
    print_summary(report)


# Main
def main():
    if not GOLD_FILE.exists():
        print("gold_standard.jsonl not found")
        return

    client       = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url="https://api.anthropic.com")
    prompts      = load_prompts()
    video_topics = load_video_topics()
    gold_pairs   = load_jsonl(str(GOLD_FILE))
    print(f"Gold standard: {len(gold_pairs)} pairs")

    run_annotation(client, gold_pairs, prompts, video_topics)
    run_evaluation()


if __name__ == "__main__":
    main()
