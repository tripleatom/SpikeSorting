r"""
sweep_track_counts.py
=====================
Sweep match thresholds and tabulate the resulting unit-track count, so you can
pick the threshold that maximizes it.

Track count is a *peaked* function of the matching threshold: too high and few
units find any partner (few tracks); too low and distinct units fuse into
ambiguous mega-merges that collapse to a handful of components (few tracks
again). The maximum sits in between — this script finds it by running
build_unit_tracks across a set of thresholds and printing count vs threshold,
for both the one-to-one builder (default) and the raw connected-components mode.

For each threshold t it:
  - (optional, --run-matching) runs match_units_CnL42SG.py --thres t, which
    writes <base>/thres_<t>/ ;
  - runs build_unit_tracks.py --output-folder <base>/thres_<t> twice
    (--raw-components, then --one-to-one) and parses the printed Tracks /
    Ambiguous / Longest lines.

The one-to-one run is done last so the unit_tracks.csv / figure left on disk in
each folder are the (preferred) one-to-one version.

Folder naming
-------------
match_units_CnL42SG.py --thres t sets ALL four floors to t, so it writes
thres_<t> (e.g. thres_0.70). This script names folders the same way and passes
them to build_unit_tracks via --output-folder, so build's own MIN_WAVEFORM
default (0.3) does not have to agree with t.

Usage
-----
  python sweep_track_counts.py                          # read existing thres_* folders
  python sweep_track_counts.py --thresholds 0.6 0.7 0.8
  python sweep_track_counts.py --run-matching           # (re)build matches first
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Single source of truth for the base output path (and a light import — no
# network access happens at import time, just Path arithmetic).
from build_unit_tracks import OUTPUT_FOLDER_BASE

_HERE = Path(__file__).resolve().parent
MATCH_SCRIPT = _HERE / "match_units_CnL42SG.py"
BUILD_SCRIPT = _HERE / "build_unit_tracks.py"

DEFAULT_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def _parse_counts(stdout: str) -> dict:
    """Pull the Tracks / Ambiguous / Longest numbers out of build's summary."""
    def grab(pattern: str) -> int:
        match = re.search(pattern, stdout)
        return int(match.group(1)) if match else -1
    return {
        "tracks": grab(r"Tracks\s*:\s*(\d+)"),
        "ambiguous": grab(r"Ambiguous[^:]*:\s*(\d+)"),
        "longest": grab(r"Longest track\s*:\s*(\d+)"),
    }


def _run_build(folder: Path, one_to_one: bool, min_sessions: int) -> dict:
    flag = "--one-to-one" if one_to_one else "--raw-components"
    proc = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output-folder", str(folder),
         "--min-sessions", str(min_sessions), flag],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        print(f"  [error] build failed ({flag}) for {folder.name}: {tail}")
        return {"tracks": -1, "ambiguous": -1, "longest": -1}
    return _parse_counts(proc.stdout)


def _run_match(thres: float) -> bool:
    proc = subprocess.run(
        [sys.executable, str(MATCH_SCRIPT), "--thres", f"{thres:.2f}", "--no-figures"],
        text=True,
    )
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-folder-base", type=Path, default=OUTPUT_FOLDER_BASE)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--run-matching", action="store_true",
                        help="run match_units_CnL42SG.py --thres t before reading each folder")
    args = parser.parse_args()

    rows = []
    for thres in args.thresholds:
        folder = args.output_folder_base / f"thres_{thres:.2f}"
        if args.run_matching:
            print(f"[match] thres {thres:.2f} ...")
            if not _run_match(thres):
                print(f"  [error] matching failed at thres {thres:.2f}; skipping")
                continue
        if not (folder / "matched_units_all_pairs.csv").exists():
            print(f"  [skip] {folder} has no matched_units_all_pairs.csv "
                  f"(run with --run-matching).")
            continue
        # Raw first, one-to-one last so the persisted CSV/figure are one-to-one.
        raw = _run_build(folder, False, args.min_sessions)
        o2o = _run_build(folder, True, args.min_sessions)
        rows.append((thres, o2o, raw))

    if not rows:
        print("No thresholds produced results.")
        return

    print(f"\n{'='*72}")
    print(f"{'thres':>7} | {'1-to-1':>8} {'longest':>8} | "
          f"{'components':>11} {'longest':>8} {'ambig':>6}")
    print(f"{'-'*72}")
    best_thres, best_count = None, -1
    for thres, o2o, raw in rows:
        print(f"{thres:>7.2f} | {o2o['tracks']:>8} {o2o['longest']:>8} | "
              f"{raw['tracks']:>11} {raw['longest']:>8} {raw['ambiguous']:>6}")
        if o2o["tracks"] > best_count:
            best_count, best_thres = o2o["tracks"], thres
    print(f"{'='*72}")
    if best_thres is not None:
        print(f"Peak one-to-one track count: {best_count} at thres {best_thres:.2f}")


if __name__ == "__main__":
    main()
