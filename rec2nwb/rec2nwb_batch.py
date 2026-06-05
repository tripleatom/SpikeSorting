"""
Batch wrapper for rec2nwb_interp.py
====================================
Reads batch_config.json (in the same directory as this script), then processes
each folder in sequence. If one folder fails, execution continues.
All runtime info is saved to a timestamped TXT log file.

Usage
-----
  Edit batch_config.json, then run:
  python batch_rec2nwb.py
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from rec2nwb.rec2nwb_interp import process_folder, SpikeGadgetsRecToNWB

CONFIG_FILE = Path(__file__).parent / "batch_config.json"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_run(configs: list[dict], log_path: Path) -> None:
    """Process all folders, continue on failure, write runtime log."""
    n = len(configs)
    run_start = datetime.now()

    print(f"\n{'='*60}")
    print(f"Config file : {CONFIG_FILE}")
    print(f"Runtime log : {log_path}")
    print(f"Folders     : {n}")
    print(f"{'='*60}")

    log_lines = [
        f"Batch run started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Config file      : {CONFIG_FILE}",
        f"Runtime log      : {log_path}",
        f"Total folders    : {n}",
        "=" * 60,
    ]

    succeeded = 0
    failed = 0

    for i, cfg in enumerate(configs, 1):
        folder_str = cfg.get('data_folder', '<unknown>')
        print(f"\n{'='*60}")
        print(f"[{i}/{n}] Processing: {folder_str}")
        print(f"{'='*60}")

        t_start = time.time()
        try:
            process_folder(cfg)
            elapsed = time.time() - t_start

            data_folder = Path(folder_str)
            shanks = cfg.get('shanks', [])
            _conv = SpikeGadgetsRecToNWB()
            try:
                sess_desc = _conv.get_session_description(data_folder)
                outputs = ', '.join(f"{sess_desc}sh{s}.nwb" for s in shanks)
            except Exception:
                outputs = '(could not determine output filenames)'

            log_lines += [
                f"[{i}/{n}] {folder_str}",
                f"  Status : SUCCESS",
                f"  Elapsed: {elapsed:.1f}s",
                f"  Outputs: {outputs}",
                "-" * 60,
            ]
            succeeded += 1

        except Exception as e:
            elapsed = time.time() - t_start
            tb = traceback.format_exc()
            print(f"\nERROR processing {folder_str}:\n{tb}")
            log_lines += [
                f"[{i}/{n}] {folder_str}",
                f"  Status : FAILED",
                f"  Elapsed: {elapsed:.1f}s",
                f"  Error  : {type(e).__name__}: {e}",
                "  Traceback:",
            ]
            for tb_line in tb.splitlines():
                log_lines.append(f"    {tb_line}")
            log_lines.append("-" * 60)
            failed += 1

    run_end = datetime.now()
    log_lines += [
        "=" * 60,
        f"Total: {succeeded} succeeded, {failed} failed",
        f"Batch run ended: {run_end.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    log_path.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    print(f"\nRuntime log saved to: {log_path}")
    print(f"Summary: {succeeded}/{n} succeeded, {failed}/{n} failed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not CONFIG_FILE.exists():
        print(f"Config file not found: {CONFIG_FILE}")
        print("Please create batch_config.json next to this script.")
        sys.exit(1)

    with open(CONFIG_FILE, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    configs = data['folders']
    print(f"Loaded {len(configs)} folder(s) from {CONFIG_FILE.name}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(__file__).parent / f"batch_run_{timestamp}.txt"
    batch_run(configs, log_path)


if __name__ == "__main__":
    main()
