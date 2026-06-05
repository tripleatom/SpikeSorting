"""
batch_label_CnL42SG.py
======================
Step 1 of 2 — auto-labeling for CnL42SG sessions 260122-260325.

For each session in the plan CSV, run curation_lazy auto-classification
(no GUI, no HTML review) and record the outcome in the ``labeling_status``
column of the plan CSV.

Pairs with batch_analyzer_CnL42SG.py, which consumes the labels written here
and builds the curated SortingAnalyzer.

Status values written to ``labeling_status``:
  done@<ts>           — units found and labeled
  no_units@<ts>       — no unit images discovered for the session
  failed@<ts>         — curation_lazy raised (see console traceback)
  skipped_done@<ts>   — already labeled and OVERWRITE=False

Usage
-----
  python batch_label_CnL42SG.py
"""

import sys
import traceback
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # for rec2nwb

import curation_lazy
from plan_status import stamp, state_of, update_status

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH = _HERE / "curation_record" / "CnL42SG_260122_260325_curation_plan.csv"
STATUS_COL = "labeling_status"
OVERWRITE = False   # set True to re-label sessions already marked done


# ── curation_lazy runner ───────────────────────────────────────────────────────
def run_curation_lazy(session_sortout_folder: Path) -> bool:
    """
    Run auto-curation on one session by patching curation_lazy's module globals.
    Returns True on success, False if no units were found.
    """
    orig_folder    = curation_lazy.sortout_folder
    orig_overwrite = curation_lazy.OVERWRITE

    try:
        curation_lazy.sortout_folder  = session_sortout_folder
        curation_lazy.OVERWRITE       = False
        curation_lazy._metrics_cache  = {}
        curation_lazy._analyzer_cache = {}

        output_json = session_sortout_folder / "unit_labels.json"

        units = curation_lazy.discover_units(session_sortout_folder)
        if not units:
            print(f"  [curation_lazy] No unit images found — skipping.")
            return False

        print(f"  [curation_lazy] Found {len(units)} unit image(s).")

        labels = curation_lazy.load_labels(output_json)
        already = sum(1 for rec in labels.values()
                      for lbl in rec.values() if lbl in curation_lazy.VALID_LABELS)
        if already:
            print(f"  [curation_lazy] Resuming — {already} unit(s) already labeled.")

        curation_lazy.label_units(
            units, labels, output_json,
            run_merge=curation_lazy.RUN_MERGE,
            run_html_review=False,
        )
        return True

    finally:
        curation_lazy.sortout_folder = orig_folder
        curation_lazy.OVERWRITE      = orig_overwrite


# ── Main batch loop ────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    total       = len(df)
    done        = 0
    no_units    = 0
    skipped     = 0
    errors      = []

    for _, row in df.iterrows():
        session_name    = row["session_name"].strip()
        planned_action  = row.get("planned_action", "").strip()
        sortout_session = Path(row["sortout_session_path"].strip())
        prev_status     = state_of(row.get(STATUS_COL, ""))

        print(f"\n{'='*60}")
        print(f"Session : {session_name}")
        print(f"Action  : {planned_action}")
        print(f"{'='*60}")

        # ── Resume skip: already labeled in a previous run ─────────────────
        if prev_status == "done" and not OVERWRITE:
            print(f"  labeling_status=done → skipping (set OVERWRITE=True to redo).")
            skipped += 1
            continue

        # ── Run curation_lazy ──────────────────────────────────────────────
        try:
            ok = run_curation_lazy(sortout_session)
        except Exception:
            msg = f"{session_name}: curation_lazy FAILED\n{traceback.format_exc()}"
            print(f"\n  ERROR:\n{msg}")
            errors.append(msg)
            update_status(CSV_PATH, session_name, STATUS_COL, stamp("failed"))
            continue

        if ok:
            done += 1
            update_status(CSV_PATH, session_name, STATUS_COL, stamp("done"))
        else:
            no_units += 1
            update_status(CSV_PATH, session_name, STATUS_COL, stamp("no_units"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LABELING COMPLETE  ({total} session(s))")
    print(f"  Labeled           : {done}")
    print(f"  No units found    : {no_units}")
    print(f"  Skipped (done)    : {skipped}")
    print(f"  Errors            : {len(errors)}")
    if errors:
        print("\nFailed sessions:")
        for e in errors:
            print(e)
            print("-" * 40)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
