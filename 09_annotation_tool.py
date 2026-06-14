"""
Manual annotation tool (Streamlit).
to run: streamlit run 09_annotation_tool.py
"""
import os
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# paths
PAIRS_FILE  = Path("./08_pairs/sampled_pairs.jsonl")
OUT_DIR     = Path("./09_manual_annotations")
OUT_DIR.mkdir(exist_ok=True)
GOLD_FILE   = OUT_DIR / "gold_standard.jsonl"

# 10% of pairs is around 700
ANNOTATION_SAMPLE_SIZE = int(os.getenv("ANNOTATION_SAMPLE_SIZE", 700))
RANDOM_SEED            = int(os.getenv("RANDOM_SEED", 42))

# annotation schema
INTERACTION_OPTIONS = [
    ("+1.0  Constructive Disagreement", "+1.0", 1.0),
    ("+0.5  Constructive Agreement",    "+0.5", 0.5),
    (" 0.0  Neutral / Rephrase",        " 0.0", 0.0),
    ("-0.5  Destructive Agreement",     "-0.5", -0.5),
    ("-1.0  Destructive Disagreement",  "-1.0", -1.0),
]
STANCE_OPTIONS = {
    "migration": [
        ("0 — Against immigration",  0),
        ("1 — Neutral",              1),
        ("2 — Pro immigration",      2),
    ],
    "climate": [
        ("0 — Climate skeptic / Against climate action", 0),
        ("1 — Neutral",                                  1),
        ("2 — Pro climate action",                       2),
    ],
}
RHETORIC_LABELS = [
    "Loaded_Language", "Name_Calling,Labeling", "Repetition",
    "Exaggeration,Minimisation", "Appeal_to_fear-prejudice", "Flag-Waving",
    "Causal_Oversimplification", "Appeal_to_Authority",
    "Slogans/Thought-terminating_Cliches", "Whataboutism,Straw_Men",
    "Black-and-White_Fallacy", "Bandwagon,Reductio_ad_hitlerum",
    "Doubt", "Appeal_to_Time",
]

CATEGORY_COLORS = {"climate": "#d4edda", "migration": "#d1ecf1"}


# helpers
def load_jsonl(path: Path) -> list:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def append_jsonl(record: dict, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@st.cache_data
def load_queue():
    all_pairs = load_jsonl(PAIRS_FILE)

    # filter out very short texts and mismatched targets
    all_pairs = [p for p in all_pairs
                 if len(p.get("ParentText", "")) >= 20
                 and len(p.get("ChildText",  "")) >= 20
                 and not p.get("mismatched_target", False)]

    # stratified sample: equal from each (category × party) stratum
    from collections import defaultdict
    strata = defaultdict(list)
    for p in all_pairs:
        strata[(p["category"], p["party"])].append(p)

    rng = random.Random(RANDOM_SEED)
    per_stratum = ANNOTATION_SAMPLE_SIZE // len(strata)
    sample = []
    for key, items in strata.items():
        rng.shuffle(items)
        sample.extend(items[:per_stratum])
    rng.shuffle(sample)
    sample_by_id = {p["PairID"]: p for p in sample}

    # Load gold standard — identify annotation gaps
    gold_records = {}
    if GOLD_FILE.exists():
        for rec in load_jsonl(GOLD_FILE):
            gold_records[rec["PairID"]] = rec

    # Pairs in gold but missing ParentStanceLabel goes to parent stance update mode
    needs_parent_stance = {
        pid for pid, rec in gold_records.items()
        if "ParentStanceLabel" not in rec or rec["ParentStanceLabel"] is None
    }
    # Fully annotated (all fields)
    fully_annotated_ids = set(gold_records.keys()) - needs_parent_stance

    # Build queue:
    #   1. Update mode: pairs needing only ParentStanceLabel
    #   2. New pairs: not in gold at all thus they need full annotation
    update_queue = [
        {"_mode": "update_parent_stance", **sample_by_id[pid]}
        for pid in needs_parent_stance if pid in sample_by_id
    ]
    new_queue = [p for p in sample if p["PairID"] not in gold_records]
    queue = update_queue + new_queue

    return queue, gold_records, fully_annotated_ids


def update_parent_stance_in_gold(pair_id: str, parent_stance: int) -> bool:
    """Rewrite gold_standard.jsonl with ParentStanceLabel added to the matching record."""
    if not GOLD_FILE.exists():
        return False
    records = load_jsonl(GOLD_FILE)
    updated = False
    for rec in records:
        if rec["PairID"] == pair_id:
            rec["ParentStanceLabel"] = parent_stance
            updated = True
            break
    if updated:
        with open(GOLD_FILE, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return updated


def save_annotation(pair: dict, score: float, label: str,
                    child_stance: int, parent_stance: int,
                    rhetoric: list, note: str):
    record = {
        "PairID":             pair["PairID"],
        "VideoID":            pair["VideoID"],
        "ThreadID":           pair["ThreadID"],
        "category":           pair.get("category", ""),
        "party":              pair.get("party", ""),
        "ParentCommentID":    pair["ParentCommentID"],
        "ParentText":         pair["ParentText"],
        "ChildCommentID":     pair["ChildCommentID"],
        "ChildText":          pair["ChildText"],
        "InteractionScore":   score,
        "InteractionLabel":   label,
        "StanceLabel":        child_stance,
        "ParentStanceLabel":  parent_stance,
        "Techniques":         rhetoric,
        "AnnotatorNote":      note,
        "AnnotatedAt":        datetime.now(timezone.utc).isoformat(),
    }
    try:
        append_jsonl(record, GOLD_FILE)
    except Exception as e:
        st.error(f"Failed to save: {e}")


# UI
def main():
    st.set_page_config(page_title="Comment Pair Annotator", layout="wide")
    st.title("Comment Pair Annotator")

    if "queue" not in st.session_state:
        queue, gold_records, fully_annotated_ids = load_queue()
        st.session_state.queue              = queue
        st.session_state.gold_records       = gold_records
        st.session_state.idx                = 0
        st.session_state.total_done         = len(fully_annotated_ids)

    queue        = st.session_state.queue
    gold_records = st.session_state.gold_records
    idx          = st.session_state.idx
    total_done   = st.session_state.total_done
    total        = total_done + len(queue)

    # progress bar
    progress = (total_done + idx) / total if total > 0 else 0
    st.progress(progress, text=f"{total_done + idx} / {total} annotated  "
                               f"({100*progress:.1f}%)")

    if idx >= len(queue):
        st.success("All pairs in this session have been annotated")
        st.info(f"Gold standard saved to: `{GOLD_FILE}`")
        return

    pair       = queue[idx]
    update_mode = pair.get("_mode") == "update_parent_stance"
    pair_id    = pair["PairID"]
    cat        = pair.get("category", "")
    party      = pair.get("party", "")
    bg         = CATEGORY_COLORS.get(cat, "#f9f9f9")

    # In update mode, fetch existing gold record for pre-filling
    gold_rec = gold_records.get(pair_id, {}) if update_mode else {}

    if update_mode:
        st.info(
            "**Parent stance missing** — "
            "Interaction, child stance, and techniques are pre-filled from your earlier annotation. "
            "Only Task B (parent stance) needs to be completed."
        )

    # context bar
    st.markdown(
        f"<div style='background:{bg};color:#111;padding:8px 12px;border-radius:6px;margin-bottom:8px'>"
        f"<b>Category:</b> {cat.upper()}  &nbsp;|&nbsp; "
        f"<b>Party:</b> {party}  &nbsp;|&nbsp; "
        f"<b>Video:</b> {pair['VideoID']}  &nbsp;|&nbsp; "
        f"<b>Thread:</b> {pair['ThreadID'][:30]}…"
        f"</div>",
        unsafe_allow_html=True,
    )

    # parent / child columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Parent comment**")
        st.markdown(
            f"<div style='background:#f0f0f0;color:#111;padding:10px;border-radius:6px;min-height:100px'>"
            f"<small>{pair['ParentAuthor']} · {str(pair.get('ParentTimestamp',''))[:10]} · "
            f"Level {pair.get('ParentLevel',0)}</small><br><br>"
            f"{pair['ParentText']}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**Child comment**")
        st.markdown(
            f"<div style='background:#dce8f7;color:#111;padding:10px;border-radius:6px;min-height:100px'>"
            f"<small>{pair['ChildAuthor']} · {str(pair.get('ChildTimestamp',''))[:10]} · "
            f"Level {pair.get('ChildLevel',0)}</small><br><br>"
            f"{pair['ChildText']}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    stance_opts = STANCE_OPTIONS.get(cat, STANCE_OPTIONS["migration"])
    stance_labels = [opt[0] for opt in stance_opts]

    if update_mode:
        # Task A — pre-filled, read-only display
        existing_score = float(gold_rec.get("InteractionScore", 0.0))
        existing_label = gold_rec.get("InteractionLabel", "")
        score, label   = existing_score, existing_label
        st.markdown(f"**Task A (pre-filled):** {existing_label} ({existing_score:+.1f})")

        # Task B child — pre-filled, read-only
        existing_child_stance = int(gold_rec.get("StanceLabel", 1))
        child_stance_label = next(
            (l for l, v in stance_opts if v == existing_child_stance), stance_labels[1]
        )
        st.markdown(f"**Task B child (pre-filled):** {child_stance_label}")
        child_stance_value = existing_child_stance

        # Task C — pre-filled, read-only
        existing_techs = gold_rec.get("Techniques") or []
        st.markdown(f"**Task C (pre-filled):** {', '.join(existing_techs) if existing_techs else 'none'}")
        rhetoric = existing_techs

        note = gold_rec.get("AnnotatorNote", "")

    else:
        # Task A — interaction type
        st.markdown("**Task A: Interaction quality** — how does the child respond to the parent?")
        interaction_labels = [opt[0] for opt in INTERACTION_OPTIONS]
        interaction_choice = st.radio(
            "Interaction quality", interaction_labels, key=f"interaction_{idx}",
            label_visibility="collapsed"
        )
        score, label = next(
            (s, lbl.strip()) for lbl, _, s in INTERACTION_OPTIONS if lbl == interaction_choice
        )

        # Task B — child stance
        st.markdown(f"**Task B (child): Stance** of the **child** comment toward **{cat} policy**")
        child_stance_choice = st.radio(
            "Child stance", stance_labels, horizontal=True,
            key=f"child_stance_{idx}", label_visibility="collapsed"
        )
        child_stance_value = next(v for l, v in stance_opts if l == child_stance_choice)

        # Task C — propaganda techniques
        st.markdown("**Task C: Propaganda techniques** in the child comment (select all that apply)")
        rhetoric = []
        cols = st.columns(2)
        for i, lbl in enumerate(RHETORIC_LABELS):
            with cols[i % 2]:
                if st.checkbox(lbl, key=f"rhet_{idx}_{i}"):
                    rhetoric.append(lbl)

        note = st.text_input("Optional note", key=f"note_{idx}", placeholder="Free text…")

    # Task B — PARENT stance 
    st.markdown(f"**Task B (parent): Stance** of the **parent** comment toward **{cat} policy**")
    parent_stance_choice = st.radio(
        "Parent stance", stance_labels, horizontal=True,
        key=f"parent_stance_{idx}", label_visibility="collapsed"
    )
    parent_stance_value = next(v for l, v in stance_opts if l == parent_stance_choice)

    col_submit, col_skip = st.columns([1, 5])
    with col_submit:
        if st.button("Submit & Next", type="primary"):
            if update_mode:
                ok = update_parent_stance_in_gold(pair_id, parent_stance_value)
                if not ok:
                    st.error(f"Failed to update ParentStanceLabel for {pair_id}")
            else:
                save_annotation(pair, score, label, child_stance_value,
                                parent_stance_value, rhetoric, note)
            st.session_state.idx += 1
            st.rerun()
    with col_skip:
        if st.button("Skip"):
            st.session_state.idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
