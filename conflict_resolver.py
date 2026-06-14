#!/usr/bin/env python3
"""
conflict_resolver.py  –  Streamlit tool for adjudicating annotation conflicts.

Run from the project root:
    streamlit run conflict_resolver.py

Workflow
--------
* Identical pairs (all 4 fields agree) are auto-resolved on first launch.
* Conflicted pairs are presented for manual adjudication.
* Filter view: All | Conflicts only | Flagged | Unresolved.
* Flag pairs for later review with the ★ button.
* Progress is auto-saved to  09_manual_annotations/resolution_progress.json
* Export final gold_standard.jsonl with the Export button.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Paths
ANNOT_DIR     = Path("./09_manual_annotations")
FILE_A        = ANNOT_DIR / "gold_standard_a.jsonl"
FILE_Z        = ANNOT_DIR / "gold_standard_z.jsonl"
PROGRESS_FILE = ANNOT_DIR / "resolution_progress.json"
OUTPUT_FILE   = ANNOT_DIR / "gold_standard.jsonl"

# Schema
INTERACTION_OPTIONS = [
    ("+1.0  Constructive Disagreement", 1.0),
    ("+0.5  Constructive Agreement",    0.5),
    (" 0.0  Neutral / Rephrase",        0.0),
    ("-0.5  Destructive Agreement",    -0.5),
    ("-1.0  Destructive Disagreement", -1.0),
]
SCORE_TO_LABEL = {s: lbl for lbl, s in INTERACTION_OPTIONS}
INTER_LABELS   = [lbl for lbl, _ in INTERACTION_OPTIONS]
INTER_SCORES   = [s   for _, s   in INTERACTION_OPTIONS]

STANCE_OPTIONS = {
    "migration": [
        ("0 — Against immigration",                  0),
        ("1 — Neutral",                              1),
        ("2 — Pro immigration",                      2),
    ],
    "climate": [
        ("0 — Climate skeptic / Against action",     0),
        ("1 — Neutral",                              1),
        ("2 — Pro climate action",                   2),
    ],
}

ALL_TECHNIQUES = [
    "Appeal_to_Authority",
    "Appeal_to_Time",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans/Thought-terminating_Cliches",
    "Whataboutism,Straw_Men",
]

# Helpers
def _inter_label(score) -> str:
    return SCORE_TO_LABEL.get(float(score), str(score))

def _stance_label(val, category: str) -> str:
    opts = STANCE_OPTIONS.get(category, STANCE_OPTIONS["migration"])
    for lbl, v in opts:
        if v == val:
            return lbl
    return str(val)

def _tset(rec) -> set:
    return set(rec.get("Techniques") or [])

def _is_identical(a, z) -> bool:
    return (
        a["InteractionScore"]  == z["InteractionScore"]
        and a["StanceLabel"]   == z["StanceLabel"]
        and a["ParentStanceLabel"] == z["ParentStanceLabel"]
        and _tset(a) == _tset(z)
    )

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

#  Data loading (cached)
@st.cache_data
def load_data():
    def _load(path):
        recs = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["PairID"]] = r
        return recs

    a_recs   = _load(FILE_A)
    z_recs   = _load(FILE_Z)
    pair_ids = sorted(set(a_recs) & set(z_recs))
    return a_recs, z_recs, pair_ids

# Progress persistence
def _load_progress() -> tuple[dict, set]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("resolutions", {}), set(data.get("flags", []))
    return {}, set()

def _save_progress(resolutions: dict, flags: set):
    ANNOT_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"resolutions": resolutions,
             "flags": sorted(flags),
             "last_saved": _now()},
            f, indent=2, ensure_ascii=False,
        )

# Session-state initialisation
def _init():
    if "cr_initialized" in st.session_state:
        return
    a_recs, z_recs, pair_ids = load_data()
    resolutions, flags = _load_progress()

    # Auto-resolve identical pairs on first run
    changed = False
    for pid in pair_ids:
        if pid not in resolutions and _is_identical(a_recs[pid], z_recs[pid]):
            rec = deepcopy(a_recs[pid])
            rec["resolved_by"] = "auto"
            resolutions[pid]   = rec
            changed = True
    if changed:
        _save_progress(resolutions, flags)

    st.session_state.resolutions   = resolutions
    st.session_state.flags         = flags
    st.session_state.filter_mode   = "conflicts_only"
    st.session_state.current_idx   = 0
    st.session_state.cr_initialized = True

# Widget state initialisation for a specific pair
def _init_pair_state(pid, a, z, saved):
    """
    Explicitly set session_state for every widget on this pair.
    Called whenever the user navigates to a different pair so that
    st.multiselect / st.radio always reflect the saved resolution (or
    sensible defaults) rather than whatever stale value Streamlit cached.
    """
    a_tech = set(a.get("Techniques") or [])
    z_tech = set(z.get("Techniques") or [])

    # Techniques: radio choice + custom multiselect
    saved_tech_set = set(saved.get("Techniques") or []) if "Techniques" in saved else None

    if saved_tech_set is not None and a_tech != z_tech:
        # Determine which radio option the saved choice corresponds to
        if saved_tech_set == a_tech:
            st.session_state[f"tech_radio_{pid}"] = "Use A"
        elif saved_tech_set == z_tech:
            st.session_state[f"tech_radio_{pid}"] = "Use Z"
        else:
            st.session_state[f"tech_radio_{pid}"] = "Custom"
        # Always set the custom multiselect to the saved value
        st.session_state[f"tech_custom_{pid}"] = [
            t for t in sorted(saved.get("Techniques", [])) if t in ALL_TECHNIQUES
        ]
    else:
        # No save yet: default custom multiselect to the union
        st.session_state[f"tech_custom_{pid}"] = [
            t for t in sorted(a_tech | z_tech) if t in ALL_TECHNIQUES
        ]

    # Interaction radio
    a_inter    = a["InteractionScore"]
    z_inter    = z["InteractionScore"]
    saved_inter = saved.get("InteractionScore")
    if saved_inter is not None and a_inter != z_inter:
        if float(saved_inter) == float(a_inter):
            st.session_state[f"inter_radio_{pid}"] = "Use A"
        elif float(saved_inter) == float(z_inter):
            st.session_state[f"inter_radio_{pid}"] = "Use Z"
        else:
            st.session_state[f"inter_radio_{pid}"] = "Custom"

    # Stance child radio
    a_sc    = a["StanceLabel"]
    z_sc    = z["StanceLabel"]
    saved_sc = saved.get("StanceLabel")
    if saved_sc is not None and a_sc != z_sc:
        if saved_sc == a_sc:
            st.session_state[f"stance_radio_sc_{pid}"] = "Use A"
        elif saved_sc == z_sc:
            st.session_state[f"stance_radio_sc_{pid}"] = "Use Z"
        else:
            st.session_state[f"stance_radio_sc_{pid}"] = "Custom"

    # Stance parent radio
    a_sp    = a["ParentStanceLabel"]
    z_sp    = z["ParentStanceLabel"]
    saved_sp = saved.get("ParentStanceLabel")
    if saved_sp is not None and a_sp != z_sp:
        if saved_sp == a_sp:
            st.session_state[f"stance_radio_sp_{pid}"] = "Use A"
        elif saved_sp == z_sp:
            st.session_state[f"stance_radio_sp_{pid}"] = "Use Z"
        else:
            st.session_state[f"stance_radio_sp_{pid}"] = "Custom"


# Filtered view
def _filtered_ids(pair_ids, a_recs, z_recs) -> list:
    mode = st.session_state.filter_mode
    res  = st.session_state.resolutions
    flg  = st.session_state.flags
    if mode == "all":
        return pair_ids
    elif mode == "conflicts_only":
        # Only conflicts that haven't been resolved yet
        return [p for p in pair_ids
                if not _is_identical(a_recs[p], z_recs[p]) and p not in res]
    elif mode == "all_conflicts":
        # All conflicts, resolved or not
        return [p for p in pair_ids if not _is_identical(a_recs[p], z_recs[p])]
    elif mode == "flagged":
        return [p for p in pair_ids if p in flg]
    elif mode == "unresolved":
        return [p for p in pair_ids if p not in res]
    return pair_ids

# Export
def _export(resolutions: dict, pair_ids: list) -> int:
    records = [resolutions[p] for p in pair_ids if p in resolutions]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)

# Conflict badge
def _badge(match: bool, resolved: bool = False) -> str:
    if match:
        return "✅ match"
    if resolved:
        return "✅ resolved"
    return "⚠️ unresolved"

# Resolution widgets for one field
def _interaction_widget(pid, a_val, z_val, saved_val, category):
    """Renders the interaction score resolution widget. Returns chosen score."""
    match = (a_val == z_val)
    a_lbl = _inter_label(a_val)
    z_lbl = _inter_label(z_val)

    col1, col2 = st.columns(2)
    col1.markdown(f"**A:** {a_lbl}")
    col2.markdown(f"**Z:** {z_lbl}")

    if match:
        st.caption(f"✅ Both annotators agree: {a_lbl}")
        show_override = st.checkbox("Override this value", key=f"override_inter_{pid}")
        if not show_override:
            return a_val

    # Which option was previously saved?
    default_idx = 0  # default: first option (most constructive)
    if saved_val is not None:
        try:
            default_idx = INTER_SCORES.index(float(saved_val))
        except ValueError:
            default_idx = 0

    choice = st.radio(
        "Resolution",
        options=["Use A", "Use Z", "Custom"],
        index=0 if saved_val == a_val else (1 if saved_val == z_val else 2),
        horizontal=True,
        key=f"inter_radio_{pid}",
        label_visibility="collapsed",
    )

    if choice == "Use A":
        return a_val
    elif choice == "Use Z":
        return z_val
    else:  # Custom
        sel = st.selectbox(
            "Custom value",
            options=INTER_LABELS,
            index=default_idx,
            key=f"inter_custom_{pid}",
            label_visibility="collapsed",
        )
        return INTER_SCORES[INTER_LABELS.index(sel)]


def _stance_widget(pid, key_suffix, a_val, z_val, saved_val, category, label):
    """Renders a stance resolution widget. Returns chosen int value."""
    opts  = STANCE_OPTIONS.get(category, STANCE_OPTIONS["migration"])
    lbls  = [l for l, _ in opts]
    vals  = [v for _, v in opts]
    match = (a_val == z_val)

    col1, col2 = st.columns(2)
    col1.markdown(f"**A:** {_stance_label(a_val, category)}")
    col2.markdown(f"**Z:** {_stance_label(z_val, category)}")

    if match:
        st.caption(f"✅ Both annotators agree: {_stance_label(a_val, category)}")
        show_override = st.checkbox("Override this value", key=f"override_{key_suffix}_{pid}")
        if not show_override:
            return a_val

    radio_choice = st.radio(
        "Resolution",
        options=["Use A", "Use Z", "Custom"],
        index=0 if saved_val == a_val else (1 if saved_val == z_val else 2),
        horizontal=True,
        key=f"stance_radio_{key_suffix}_{pid}",
        label_visibility="collapsed",
    )

    if radio_choice == "Use A":
        return a_val
    elif radio_choice == "Use Z":
        return z_val
    else:
        saved_lbl = _stance_label(saved_val, category) if saved_val is not None else lbls[1]
        saved_idx = lbls.index(saved_lbl) if saved_lbl in lbls else 1
        sel = st.radio(
            "Custom stance",
            options=lbls,
            index=saved_idx,
            horizontal=True,
            key=f"stance_custom_{key_suffix}_{pid}",
            label_visibility="collapsed",
        )
        return vals[lbls.index(sel)]


def _technique_widget(pid, a_val, z_val, saved_val):
    """
    Technique resolution widget — mirrors Tasks A & B:
      Use A  |  Use Z  |  Custom (manual multiselect)
    Returns the final list of selected techniques.
    """
    a_set = set(a_val or [])
    z_set = set(z_val or [])
    match = (a_set == z_set)

    #  Side-by-side display of both annotations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**A selected:**")
        if a_set:
            for t in sorted(a_set):
                color = "green" if t in z_set else "orange"
                st.markdown(f"&nbsp;&nbsp;:{color}[{t}]")
        else:
            st.caption("(none)")
    with col2:
        st.markdown("**Z selected:**")
        if z_set:
            for t in sorted(z_set):
                color = "green" if t in a_set else "blue"
                st.markdown(f"&nbsp;&nbsp;:{color}[{t}]")
        else:
            st.caption("(none)")

    st.caption("🟢 both agree  🟠 only A  🔵 only Z")

    # Matched case: auto-resolve with optional override
    if match:
        agreed = ", ".join(sorted(a_set)) if a_set else "(none)"
        st.caption(f"✅ Both annotators agree: {agreed}")
        if not st.checkbox("Override this value", key=f"override_tech_{pid}"):
            return sorted(a_set)

    # Conflict / override: Use A / Use Z / Custom
    choice = st.radio(
        "Resolution",
        options=["Use A", "Use Z", "Custom"],
        horizontal=True,
        key=f"tech_radio_{pid}",
        label_visibility="collapsed",
    )

    if choice == "Use A":
        st.caption(f"→ {sorted(a_set) or '(none)'}")
        return sorted(a_set)

    if choice == "Use Z":
        st.caption(f"→ {sorted(z_set) or '(none)'}")
        return sorted(z_set)

    # Custom: show multiselect  (state pre-set by _init_pair_state)
    selected = st.multiselect(
        "Custom selection",
        options=ALL_TECHNIQUES,
        key=f"tech_custom_{pid}",
        label_visibility="collapsed",
    )
    return selected

# Main app 
def main():
    st.set_page_config(
        page_title="Conflict Resolver",
        page_icon="🟢",
        layout="wide",
    )
    _init()

    a_recs, z_recs, pair_ids = load_data()
    res   = st.session_state.resolutions
    flags = st.session_state.flags

    n_total         = len(pair_ids)
    n_conflict      = sum(1 for p in pair_ids if not _is_identical(a_recs[p], z_recs[p]))
    n_resolved      = len(res)
    n_flagged       = len(flags)
    n_unres_conflict = sum(
        1 for p in pair_ids
        if not _is_identical(a_recs[p], z_recs[p]) and p not in res
    )

    # Sidebar
    with st.sidebar:
        st.title("⚖️ Conflict Resolver")
        st.metric("Total pairs",          n_total)
        st.metric("Conflicts (total)",    n_conflict)
        st.metric("Conflicts (remaining)", n_unres_conflict)
        st.metric("Resolved",             n_resolved)
        st.metric("Flagged",              n_flagged)
        st.progress(
            (n_conflict - n_unres_conflict) / n_conflict if n_conflict else 1.0,
            text=f"{n_conflict - n_unres_conflict}/{n_conflict} conflicts resolved",
        )
        st.divider()

        st.subheader("Filter")
        filter_options = {
            "all":            f"All ({n_total})",
            "conflicts_only": f"Unresolved conflicts ({n_unres_conflict})",
            "all_conflicts":  f"All conflicts ({n_conflict})",
            "flagged":        f"Flagged ({n_flagged})",
            "unresolved":     f"Unresolved pairs ({n_total - n_resolved})",
        }
        chosen_filter = st.radio(
            "Show",
            list(filter_options.keys()),
            format_func=lambda k: filter_options[k],
            index=list(filter_options.keys()).index(st.session_state.filter_mode),
            label_visibility="collapsed",
        )
        if chosen_filter != st.session_state.filter_mode:
            st.session_state.filter_mode = chosen_filter
            st.session_state.current_idx = 0
            st.rerun()

        st.divider()
        st.subheader("Export")
        if st.button("💾 Export gold_standard.jsonl", use_container_width=True):
            n_exp = _export(res, pair_ids)
            st.success(f"Exported {n_exp} records → {OUTPUT_FILE}")

    # Filtered list
    view_ids = _filtered_ids(pair_ids, a_recs, z_recs)
    n_view   = len(view_ids)

    if n_view == 0:
        st.info("No pairs match the current filter.")
        return

    idx = max(0, min(st.session_state.current_idx, n_view - 1))
    st.session_state.current_idx = idx
    pid = view_ids[idx]
    a   = a_recs[pid]
    z   = z_recs[pid]
    saved = res.get(pid, {})
    is_flagged = pid in flags
    category   = a.get("category", "migration")

    # Reset widget state on navigation (MUST happen before any widget renders)
    # Streamlit forbids setting session_state keys after their widget is rendered.
    if st.session_state.get("_cur_pid") != pid:
        st.session_state["_cur_pid"]   = pid
        st.session_state["jump_input"] = idx + 1   # sync jump display
        _init_pair_state(pid, a, z, saved)

    # Navigation bar
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 2, 1])

    with nav_col1:
        if st.button("← Prev", disabled=idx == 0, use_container_width=True):
            st.session_state.current_idx -= 1
            st.session_state.pop("_cur_pid", None)
            st.rerun()

    with nav_col4:
        if st.button("Next →", disabled=idx == n_view - 1, use_container_width=True):
            st.session_state.current_idx += 1
            st.session_state.pop("_cur_pid", None)
            st.rerun()

    with nav_col2:
        #  Jump-to input + explicit Go button
        # Any widget interaction (radio, checkbox, …) triggers a Streamlit
        # rerun. If we auto-navigate on "jump != idx", a stale jump_input
        # value causes spurious jumps back to pair 1 on every radio click.
        # An explicit "Go" button means navigation only fires on purpose.
        jcol1, jcol2 = st.columns([3, 1])
        with jcol1:
            jump = st.number_input(
                "Jump to #",
                min_value=1, max_value=n_view,
                value=idx + 1,
                key="jump_input",
                label_visibility="collapsed",
            )
        with jcol2:
            if st.button("Go", key="jump_go", use_container_width=True):
                target = int(jump) - 1
                if target != idx:
                    st.session_state.current_idx = target
                    st.session_state.pop("_cur_pid", None)
                    st.rerun()

    with nav_col3:
        filter_label = filter_options[st.session_state.filter_mode]
        resolved_mark = "✅" if pid in res else "○"
        st.markdown(
            f"**{resolved_mark} Pair {idx + 1} / {n_view}** &nbsp; ({filter_label})",
            unsafe_allow_html=True,
        )

    st.divider()

    # Pair header
    hcol1, hcol2, hcol3 = st.columns([4, 1, 1])
    with hcol1:
        st.caption(f"PairID: `{pid}`  ·  category: **{category}**  ·  party: **{a.get('party', '')}**")
    with hcol2:
        if st.button("⭐ Flag" if not is_flagged else "★ Unflag", use_container_width=True):
            if is_flagged:
                flags.discard(pid)
            else:
                flags.add(pid)
            st.session_state.flags = flags
            _save_progress(res, flags)
            st.rerun()
    with hcol3:
        if pid in res:
            st.markdown("**Status:** ✅ resolved")
        else:
            st.markdown("**Status:** 🔲 pending")

    # Comment texts
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("**Parent comment**")
        st.text_area("parent_text", value=a.get("ParentText", ""), height=140,
                     disabled=True, key=f"pt_{pid}", label_visibility="collapsed")
    with tcol2:
        st.markdown("**Child comment**")
        st.text_area("child_text", value=a.get("ChildText", ""), height=140,
                     disabled=True, key=f"ct_{pid}", label_visibility="collapsed")

    st.divider()

    # Task A: Interaction Score
    a_inter = a["InteractionScore"]
    z_inter = z["InteractionScore"]
    inter_match = (a_inter == z_inter)
    st.markdown(f"#### Task A — Interaction Quality &nbsp; {_badge(inter_match, pid in res)}")
    chosen_inter = _interaction_widget(
        pid, a_inter, z_inter,
        saved.get("InteractionScore"), category,
    )

    st.divider()

    # Task B (child): Stance
    a_sc    = a["StanceLabel"]
    z_sc    = z["StanceLabel"]
    sc_match = (a_sc == z_sc)
    st.markdown(f"#### Task B (child) — Stance toward **{category}** policy &nbsp; {_badge(sc_match, pid in res)}")
    chosen_sc = _stance_widget(
        pid, "sc", a_sc, z_sc,
        saved.get("StanceLabel"), category,
        "Child stance",
    )

    st.divider()

    # Task B (parent): Stance
    a_sp    = a["ParentStanceLabel"]
    z_sp    = z["ParentStanceLabel"]
    sp_match = (a_sp == z_sp)
    st.markdown(f"#### Task B (parent) — Stance toward **{category}** policy &nbsp; {_badge(sp_match, pid in res)}")
    chosen_sp = _stance_widget(
        pid, "sp", a_sp, z_sp,
        saved.get("ParentStanceLabel"), category,
        "Parent stance",
    )

    st.divider()

    # Task C: Techniques
    a_tech  = a.get("Techniques") or []
    z_tech  = z.get("Techniques") or []
    tech_match = (set(a_tech) == set(z_tech))
    st.markdown(f"#### Task C — Propaganda Techniques &nbsp; {_badge(tech_match, pid in res)}")
    chosen_tech = _technique_widget(
        pid, a_tech, z_tech,
        saved.get("Techniques"),
    )

    st.divider()

    # Note
    note_default = saved.get("AnnotatorNote", "")
    note = st.text_input(
        "📝 Resolver note (optional)",
        value=note_default,
        key=f"note_{pid}",
    )

    # Save buttons
    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])

    def _build_record():
        return {
            "PairID":            pid,
            "VideoID":           a["VideoID"],
            "ThreadID":          a["ThreadID"],
            "category":          a["category"],
            "party":             a["party"],
            "ParentCommentID":   a["ParentCommentID"],
            "ParentText":        a["ParentText"],
            "ChildCommentID":    a["ChildCommentID"],
            "ChildText":         a["ChildText"],
            "InteractionScore":  chosen_inter,
            "InteractionLabel":  _inter_label(chosen_inter),
            "StanceLabel":       chosen_sc,
            "ParentStanceLabel": chosen_sp,
            "Techniques":        sorted(chosen_tech),
            "AnnotatorNote":     note,
            "resolved_by":       "manual",
            "ResolvedAt":        _now(),
        }

    with btn_col1:
        if st.button("💾 Save & Next", type="primary", use_container_width=True):
            res[pid] = _build_record()
            st.session_state.resolutions = res
            _save_progress(res, flags)
            # Force widget reset for the next pair
            st.session_state.pop("_cur_pid", None)
            # In filters that REMOVE resolved pairs (conflicts_only, unresolved),
            # the saved pair disappears and the next one slides into index `idx`
            # automatically — do NOT increment or we skip a pair.
            # In filters that KEEP all pairs (all, all_conflicts, flagged),
            # it must advance explicitly.
            mode = st.session_state.filter_mode
            if mode in ("all", "all_conflicts", "flagged"):
                if idx < n_view - 1:
                    st.session_state.current_idx += 1
            # else: stay at idx — the next unresolved pair will appear there
            st.rerun()

    with btn_col2:
        if st.button("💾 Save", use_container_width=True):
            res[pid] = _build_record()
            st.session_state.resolutions = res
            _save_progress(res, flags)
            st.success("Saved.")
            st.rerun()

    with btn_col3:
        if pid in res:
            if st.button("🗑️ Clear resolution for this pair", use_container_width=True):
                del res[pid]
                st.session_state.resolutions = res
                _save_progress(res, flags)
                st.rerun()

    # Show raw annotations for debugging
    with st.expander("🔍 Raw annotations (debug)"):
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("**Annotator A**")
            st.json(a)
        with dcol2:
            st.markdown("**Annotator Z**")
            st.json(z)


if __name__ == "__main__":
    main()
