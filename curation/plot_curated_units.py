r"""
plot_curated_units.py
=====================
Plot a full SpikeInterface unit summary for every unit in every CnL42SG
session's combined ``curated_analyzer``, and save one PNG per unit.

Each figure is ``spikeinterface.widgets.plot_unit_summary``: unit location on
the probe, multichannel template with raw waveforms overlaid, waveform density
map, autocorrelogram, and spike amplitudes over time — the same summary your
PhyCuratedPlot.py produces, but for the post-curation combined analyzer.

Layout / performance notes
--------------------------
- Sessions are enumerated from the curation plan CSV (same discovery as
  match_units_CnL42SG.py), so freshly-built analyzers are included.
- The analyzer is loaded WITHOUT reconstructing the recording (a sentinel
  recording is passed) so the 8 per-shank NWBs are never re-opened over the
  share. The summary reads only stored extensions (templates / waveforms /
  spike_amplitudes / unit_locations) plus an in-memory correlograms.
- The combined analyzers ship without a ``correlograms`` extension, so it is
  computed with ``save=False`` (in memory only) — the on-disk analyzers are
  never modified. Set COMPUTE_CORRELOGRAMS=False to skip the ACG panel.
- Existing PNGs are skipped unless OVERWRITE=True, so the job is resumable.

Output
------
By default one folder per session next to its analyzer:
  <session>/curated_units/unit_<id>_<label>.png
Set OUTPUT_BASE to centralize them instead (<OUTPUT_BASE>/<session>/...).

Usage
-----
  python plot_curated_units.py
  python plot_curated_units.py --overwrite
  python plot_curated_units.py --output-base D:\some\dir
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PNG, no display
import matplotlib.pyplot as plt
import spikeinterface.widgets as sw

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the matcher's session discovery and the recording-skip sentinel so the
# loading behavior (and session set) stays identical across the two tools.
from match_units_CnL42SG import discover_sessions, CSV_PATH, _SKIP_RECORDING
from spikeinterface.core.sortinganalyzer import SortingAnalyzer


# ============================== USER CONFIG ==============================
# Eligible curated labels. The curated_analyzer has already removed Noise and
# tags units "SUA"/"MUA" via the unit_label property.
INCLUDE_LABELS = ("sua", "mua")

# Where PNGs go. None -> <session_path>/curated_units/ (next to each analyzer).
# Set a path to centralize: <OUTPUT_BASE>/<session>/unit_*.png
OUTPUT_BASE: Path | None = None
OUTPUT_SUBDIR = "curated_units"

OVERWRITE = False               # re-render PNGs that already exist
DPI = 150

# Autocorrelogram panel (computed in memory, never written to disk).
COMPUTE_CORRELOGRAMS = True
CORRELOGRAM_WINDOW_MS = 50.0
CORRELOGRAM_BIN_MS = 1.0
# ========================================================================


def _labels_by_unit(analyzer) -> dict[str, str]:
    unit_ids = list(analyzer.unit_ids)
    prop = analyzer.sorting.get_property("unit_label")
    if prop is None:
        return {}
    return {str(uid): str(lbl).lower() for uid, lbl in zip(unit_ids, prop)}


def plot_session(session_name: str, session_path: Path, curated_path: Path,
                 *, include_labels: set[str], output_base: Path | None,
                 overwrite: bool) -> tuple[int, int, Path]:
    """Render unit-summary PNGs for one session. Returns (plotted, skipped, out_dir)."""
    analyzer = SortingAnalyzer.load(curated_path, recording=_SKIP_RECORDING, load_extensions=False)

    if COMPUTE_CORRELOGRAMS and not analyzer.has_extension("correlograms"):
        # save=False keeps it in memory only — the on-disk analyzer is untouched.
        analyzer.compute("correlograms", window_ms=CORRELOGRAM_WINDOW_MS,
                         bin_ms=CORRELOGRAM_BIN_MS, save=False)

    labels = _labels_by_unit(analyzer)
    out_dir = (output_base / session_name) if output_base else (session_path / OUTPUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotted = skipped = 0
    for uid in analyzer.unit_ids:
        label = labels.get(str(uid), "mua")
        if label not in include_labels:
            continue
        png = out_dir / f"unit_{uid}_{label}.png"
        if png.exists() and not overwrite:
            skipped += 1
            continue
        try:
            widget = sw.plot_unit_summary(analyzer, unit_id=uid, backend="matplotlib")
            fig = widget.figure
            fig.suptitle(f"{session_name} | unit {uid} | {label.upper()}", fontsize=13)
            fig.savefig(png, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            plotted += 1
        except Exception:
            plt.close("all")
            print(f"    [error] unit {uid}: failed to plot\n{traceback.format_exc()}")
    return plotted, skipped, out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--analyzer-path", type=Path, default=None,
                        help="plot a single curated_analyzer folder directly, "
                             "bypassing CSV session discovery (e.g. for animals "
                             "not tracked in the CnL42SG curation plan CSV)")
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE,
                        help="centralize PNGs here; default writes next to each analyzer")
    parser.add_argument("--include-labels", nargs="+", default=list(INCLUDE_LABELS))
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_labels = {label.lower() for label in args.include_labels}

    if args.analyzer_path is not None:
        curated_path = args.analyzer_path
        session_path = curated_path.parent
        session_name = session_path.name
        sessions = [(session_name, session_path, curated_path)]
    else:
        sessions = discover_sessions(args.csv_path)
    print(f"Discovered {len(sessions)} session(s) with a curated_analyzer.")

    total_plotted = total_skipped = 0
    for i, (session_name, session_path, curated_path) in enumerate(sessions, start=1):
        print(f"\n[{i}/{len(sessions)}] {session_name}")
        try:
            plotted, skipped, out_dir = plot_session(
                session_name, session_path, curated_path,
                include_labels=include_labels, output_base=args.output_base,
                overwrite=args.overwrite,
            )
            print(f"    {plotted} plotted, {skipped} skipped -> {out_dir}")
            total_plotted += plotted
            total_skipped += skipped
        except Exception:
            print(f"    [error] {session_name}: failed to load/plot\n{traceback.format_exc()}")

    print(f"\n{'='*60}")
    print(f"CURATED UNIT PLOTS COMPLETE")
    print(f"  Sessions : {len(sessions)}")
    print(f"  Plotted  : {total_plotted}")
    print(f"  Skipped  : {total_skipped} (already existed; use --overwrite to redo)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
