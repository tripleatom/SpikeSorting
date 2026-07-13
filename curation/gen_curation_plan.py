"""
gen_curation_plan.py
====================
Build the per-session curation plan CSV consumed by batch_label_CnL42SG.py and
batch_analyzer_CnL42SG.py (e.g. curation_record/CnL42SG_260122_260325_curation_plan.csv).

The plan is a point-in-time snapshot that, for every sort-out session in a date
range, records where its files live, whether each expected artifact exists, the
raw recording folder it pairs with (G: preferred over F:), and the action the
two batch scripts should take. The ``labeling_status`` / ``analyzer_status``
columns are left blank here — the batch scripts stamp them as they run.

Usage
-----
  python gen_curation_plan.py
"""

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

_HERE = Path(__file__).resolve().parent

# ── Configuration ──────────────────────────────────────────────────────────────
ANIMAL_ID   = "CnL42SG"
SORTOUT_TOP = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout")
# Raw recording drives in *preference order* — earlier wins when both match.
# Keep the trailing slash: Path("G:") is drive-*relative* ("G:CnL42SG"), while
# Path("G:/") anchors to the drive root ("G:\CnL42SG").
RAW_DRIVES  = [Path(r"G:/"), Path(r"F:/")]
DATE_START  = "20260122"   # inclusive, YYYYMMDD
DATE_END    = "20260325"   # inclusive, YYYYMMDD

CURATION_LAZY_SCRIPT = _HERE / "curation_lazy.py"
MS_ANALYZER_SCRIPT   = _HERE / "MsCuratedAnalyzer.py"

OUT_CSV = (_HERE / "curation_record" /
           f"{ANIMAL_ID}_{DATE_START[2:]}_{DATE_END[2:]}_curation_plan.csv")

# Column order of the plan CSV.
COLUMNS = [
    "generated_at", "date_short", "session_name",
    "sortout_session_path", "sortout_session_exists",
    "curated_analyzer_path", "curated_analyzer_exists",
    "unit_labels_path", "unit_labels_exists",
    "unit_merge_map_path", "unit_merge_map_exists",
    "detected_shanks", "missing_shank_folders",
    "expected_shank_folder_glob", "expected_latest_sorting_results_glob",
    "expected_sorting_analyzer_glob",
    "chosen_raw_folder_path", "raw_folder_exists",
    "raw_match_status", "raw_match_note",
    "expected_raw_nwb_glob", "expected_bad_channels_path", "bad_channels_exists",
    "curation_lazy_script", "ms_curated_analyzer_script",
    "planned_action", "notes_user_can_edit",
    "labeling_status", "analyzer_status",
]

_DATE_RE = re.compile(r"(\d{8})")


def session_date(folder_name: str):
    """Return the 8-digit YYYYMMDD date embedded in a session folder name, or None."""
    m = _DATE_RE.search(folder_name)
    return m.group(1) if m else None


def detect_shanks(session_folder: Path):
    """Return (sorted shank names present, sorted shank names missing up to the max index)."""
    present = sorted(
        (p.name for p in session_folder.glob("shank*") if p.is_dir()),
        key=lambda n: int(n.replace("shank", "")),
    )
    if not present:
        return [], []
    idxs = [int(n.replace("shank", "")) for n in present]
    missing = [f"shank{i}" for i in range(max(idxs) + 1) if i not in idxs]
    return present, missing


def match_raw_folder(date: str):
    """
    Find the raw recording folder for ``date`` across RAW_DRIVES (preference order).

    Returns (chosen_path | None, is_case_insensitive_match). The match is
    case-insensitive on the ``<ANIMAL_ID>_<date>`` folder name, so e.g.
    ``Cnl42SG_20260319`` still pairs with sort-out ``CnL42SG_20260319``.
    """
    target = f"{ANIMAL_ID}_{date}".lower()
    for drive in RAW_DRIVES:
        animal_dir = drive / ANIMAL_ID
        if not animal_dir.exists():
            continue
        for cand in sorted(animal_dir.iterdir()):
            if cand.is_dir() and cand.name.lower() == target:
                return cand, (cand.name != f"{ANIMAL_ID}_{date}")
    return None, False


def main():
    generated_at = datetime.now().isoformat(timespec="seconds")
    animal_sortout = SORTOUT_TOP / ANIMAL_ID

    # ── Collect in-range sessions ────────────────────────────────────────────
    sessions = []
    for p in sorted(animal_sortout.iterdir()):
        if not p.is_dir():
            continue
        date = session_date(p.name)
        if date is None or not (DATE_START <= date <= DATE_END):
            continue
        sessions.append((date, p))
    sessions.sort(key=lambda t: t[0])

    # ── First pass: resolve raw folders so the global note can list case-
    #    insensitive matches that every "matched" row references. ─────────────
    raw_resolved = {}          # date -> (chosen_path | None, is_ci)
    ci_matches   = []          # (date, actual_folder_name)
    for date, _ in sessions:
        chosen, is_ci = match_raw_folder(date)
        raw_resolved[date] = (chosen, is_ci)
        if chosen is not None and is_ci:
            ci_matches.append((date, chosen.name))

    raw_match_note = "G: preferred over F: when both matched"
    if ci_matches:
        raw_match_note += "; " + "; ".join(
            f"{d} is case-insensitive match {name}" for d, name in ci_matches)
    raw_match_note += "."

    # ── Build rows ───────────────────────────────────────────────────────────
    rows = []
    for date, session_folder in sessions:
        session_name = session_folder.name

        curated   = session_folder / "curated_analyzer"
        labels    = session_folder / "unit_labels.json"
        merge_map = session_folder / "unit_merge_map.json"
        present_shanks, missing_shanks = detect_shanks(session_folder)

        chosen, _ = raw_resolved[date]
        if chosen is not None:
            raw_status   = "matched"
            note         = raw_match_note
            raw_exists   = chosen.exists()
            nwb_glob     = str(chosen / "*sh*.nwb")
            bad_ch_path  = chosen / "bad_channels.txt"
            bad_ch_str   = str(bad_ch_path)
            bad_ch_exist = bad_ch_path.exists()
            chosen_str   = str(chosen)
        else:
            raw_status   = "missing"
            note         = "No F:/G: raw match from precheck."
            raw_exists   = False
            nwb_glob     = ""
            bad_ch_str   = ""
            bad_ch_exist = False
            chosen_str   = ""

        # ── Planned action + editable guidance note ───────────────────────────
        curated_exists = curated.exists()
        notes = []
        if curated_exists:
            action = "skip_existing_curated; run_curation_lazy_only_if_relabel_needed"
            notes.append("curated_analyzer already exists; analyzer step should skip")
        elif raw_status == "matched":
            action = "run_curation_lazy_then_MsCuratedAnalyzer"
        else:
            action = "run_curation_lazy; wait_for_user_to_fill_raw_before_MsCuratedAnalyzer"
        if raw_status != "matched":
            notes.append("fill chosen_raw_folder_path if analyzer is needed")

        rows.append({
            "generated_at": generated_at,
            "date_short": date[2:],
            "session_name": session_name,
            "sortout_session_path": str(session_folder),
            "sortout_session_exists": session_folder.exists(),
            "curated_analyzer_path": str(curated),
            "curated_analyzer_exists": curated_exists,
            "unit_labels_path": str(labels),
            "unit_labels_exists": labels.exists(),
            "unit_merge_map_path": str(merge_map),
            "unit_merge_map_exists": merge_map.exists(),
            "detected_shanks": ",".join(present_shanks),
            "missing_shank_folders": ",".join(missing_shanks),
            "expected_shank_folder_glob": str(session_folder / "shank*"),
            "expected_latest_sorting_results_glob":
                str(session_folder / "shank*" / "sorting_results_*"),
            "expected_sorting_analyzer_glob":
                str(session_folder / "shank*" / "sorting_results_*" / "sorting_analyzer"),
            "chosen_raw_folder_path": chosen_str,
            "raw_folder_exists": raw_exists,
            "raw_match_status": raw_status,
            "raw_match_note": note,
            "expected_raw_nwb_glob": nwb_glob,
            "expected_bad_channels_path": bad_ch_str,
            "bad_channels_exists": bad_ch_exist,
            "curation_lazy_script": str(CURATION_LAZY_SCRIPT),
            "ms_curated_analyzer_script": str(MS_ANALYZER_SCRIPT),
            "planned_action": action,
            "notes_user_can_edit": " | ".join(notes),
            "labeling_status": "",
            "analyzer_status": "",
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_matched = sum(r["raw_match_status"] == "matched" for r in rows)
    print(f"Wrote {len(rows)} session(s) to {OUT_CSV}")
    print(f"  raw matched : {n_matched}")
    print(f"  raw missing : {len(rows) - n_matched}")
    if ci_matches:
        print("  case-insensitive raw matches:")
        for d, name in ci_matches:
            print(f"    {d} -> {name}")


if __name__ == "__main__":
    main()
