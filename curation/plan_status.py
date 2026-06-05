"""
plan_status.py
==============
Tiny helper for recording per-session task progress back into a curation
plan CSV (e.g. curation_record/CnL42SG_260122_260325_curation_plan.csv).

Used by the two split batch scripts:
  • batch_label_CnL42SG.py     -> writes the ``labeling_status`` column
  • batch_analyzer_CnL42SG.py  -> writes the ``analyzer_status`` column

Each status value is stamped with a wall-clock time so a half-finished batch
can be resumed and inspected at a glance, e.g. ``done@2026-06-02T21:14:07``.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


def stamp(state: str) -> str:
    """Return ``state@<ISO-seconds>`` for writing into a status cell."""
    return f"{state}@{datetime.now().isoformat(timespec='seconds')}"


def state_of(cell: str) -> str:
    """Extract the bare state from a ``state@timestamp`` cell ("" if empty)."""
    return (cell or "").split("@", 1)[0].strip()


def update_status(csv_path: Path, session_name: str, column: str, value: str):
    """
    Re-read the plan CSV, set ``column`` for the row matching ``session_name``,
    and write it back. The column is created if it does not yet exist.

    Re-reading on every call keeps the on-disk file authoritative, so manual
    edits between sessions are preserved and a crash loses at most one row.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if column not in df.columns:
        df[column] = ""
    mask = df["session_name"].str.strip() == session_name
    if not mask.any():
        print(f"  [plan_status] WARNING: '{session_name}' not found in {csv_path.name}")
        return
    df.loc[mask, column] = value
    df.to_csv(csv_path, index=False)
