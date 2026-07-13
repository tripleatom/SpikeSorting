r"""
match_units_CnL42SG.py
======================
Cross-session unit matching for the CnL42SG series (sessions 260122-260325).

This is the CnL42SG analogue of Centimani's auto-match module
(``D:\Cemtimani\code\Centimani\curation\auto_match_curated_sessions.py``). It
reuses that module's scoring core verbatim (waveform / amplitude / autocorr /
location similarity, the greedy one-to-one matcher, and the CSV/JSON/figure
writers) so the matching logic stays a single source of truth. The only thing
this script changes is *where the units come from*.

Why a separate driver
----------------------
The Centimani module was written for the Xiaorong/LS16 sortout layout, where
each session has per-*recording* ``sorting_results_*`` folders and a
``unit_labels.json`` keyed by recording. CnL42SG is different: each session
already has ONE combined ``curated_analyzer`` at its root (built by
MsCuratedAnalyzer.py) with Noise removed, merges applied, ``unit_label``
(SUA/MUA) set, all 8 shanks aggregated under shank-local sparsity, and
templates / spike_amplitudes / unit_locations / template_similarity
precomputed. So here the curated_analyzer *is* the per-session unit source —
no per-recording descent and no unit_labels.json lookup are needed.

What it does
------------
- Enumerates every session in the curation plan CSV whose curated_analyzer
  exists, sorted chronologically.
- Loads each session's curated_analyzer once into a list of UnitFeatures
  (reusing the Centimani extractors for templates/amplitudes/locations).
  Each unit's shank is recovered from its extremum channel id ("sh{N}_..."),
  and its autocorrelogram is computed locally from the spike train so the
  on-disk analyzers are never modified.
- Runs the Centimani matcher on ALL session pairs (i < j).
- Writes, per pair with >=1 match, a folder with matched_units.csv / .json /
  .pdf / .png, plus top-level summaries:
    matched_units_all_pairs.csv  - every matched pair across all comparisons
    pair_match_counts.csv        - n units / n matches for every pair
    session_units.csv            - every eligible unit (session, shank, label)

Each run writes into a threshold-named subfolder of the output folder (e.g.
``unit_match_all_pairs/thres_0.70``), so sweeping thresholds never overwrites a
previous run. Point build_unit_tracks.py at that subfolder.

Usage
-----
  python match_units_CnL42SG.py                 # all pairs, write figures
  python match_units_CnL42SG.py --no-figures    # skip the per-pair PDFs/PNGs
  python match_units_CnL42SG.py --thres 0.8     # set all four floors -> thres_0.80/
  python match_units_CnL42SG.py --output-folder D:\some\dir

Threshold sweep (PowerShell):
  foreach ($t in 0.70,0.75,0.80,0.85) {
      python match_units_CnL42SG.py --thres $t --no-figures
  }
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from spikeinterface.core.sortinganalyzer import SortingAnalyzer

# ── Reuse the Centimani auto-match module as the scoring core ───────────────────
AUTO_MATCH_MODULE_DIR = Path(r"D:\Cemtimani\code\Centimani\curation")
if str(AUTO_MATCH_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_MATCH_MODULE_DIR))
import auto_match_curated_sessions as am  # noqa: E402

import re  # noqa: E402


# ============================== USER CONFIG ==============================
_HERE = Path(__file__).resolve().parent
CSV_PATH = _HERE / "curation_record" / "CnL42SG_260122_260325_curation_plan.csv"

# Where the matching outputs go. Defaults to a folder next to the sessions on
# the sortout share so the results live with the data.
OUTPUT_FOLDER = Path(
    r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout\CnL42SG\unit_match_all_pairs"
)

# Eligible curated labels (curated_analyzer has already removed Noise). The
# combined analyzer tags units "SUA"/"MUA" via the unit_label property.
INCLUDE_LABELS = ("sua", "mua")

# Matching thresholds / weights — mirror the Centimani auto-match defaults.
thres = 0.6
MIN_TOTAL = thres
MIN_WAVEFORM = 0.3
MIN_AMPLITUDE = thres
MIN_AUTOCORR = thres

# Units on different shanks cannot be the same unit, so require same-shank.
MAX_LOCATION_DISTANCE_UM = 100.0
REQUIRE_SHANK_OVERLAP = True

WAVEFORM_WEIGHT = 0.10
AMPLITUDE_WEIGHT = 0.40
AUTOCORR_WEIGHT = 0.30
LOCATION_WEIGHT = 0.20

# Local autocorrelogram window/bin (ms). Used for every session identically, so
# the cross-session cosine comparison is consistent.
AUTOCORR_WINDOW_MS = 50.0
AUTOCORR_BIN_MS = 1.0
# ========================================================================

# Non-None sentinel passed as the analyzer's recording. Its only purpose is to
# make SortingAnalyzer.load() skip reconstructing the recording from
# recording.json — which for these combined analyzers re-opens all 8 per-shank
# NWBs over the network share (slow, and the source of the "stuck on Loading"
# hang). We never read traces: templates/amplitudes/locations come from stored
# extensions and channel_ids/sampling_frequency from rec_attributes, so the
# sentinel object is never actually used.
_SKIP_RECORDING = "__skip_recording__"


# ── Session enumeration ─────────────────────────────────────────────────────────
def discover_sessions(csv_path: Path) -> list[tuple[str, Path, Path]]:
    """
    Read the plan CSV and return (session_name, sortout_session_path,
    curated_analyzer_path) for every session whose curated_analyzer is present
    ON DISK, sorted chronologically by date_short.

    Existence is checked on disk rather than via the plan's
    ``curated_analyzer_exists`` column: that column is a snapshot from when the
    plan was generated, so sessions whose analyzer was built afterwards (their
    ``analyzer_status`` reads ``done@...``) still show ``False`` there. Trusting
    the column would silently drop those freshly-built sessions.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df.sort_values("date_short")
    sessions: list[tuple[str, Path, Path]] = []
    for _, row in df.iterrows():
        name = row["session_name"].strip()
        curated_str = row.get("curated_analyzer_path", "").strip()
        if not curated_str:
            continue
        session_path = Path(row["sortout_session_path"].strip())
        curated = Path(curated_str)
        if not curated.exists():
            print(f"  [skip] {name}: no curated_analyzer on disk: {curated}")
            continue
        sessions.append((name, session_path, curated))
    return sessions


# ── Per-unit shank + autocorrelogram (local, no on-disk writes) ─────────────────
def shank_of_unit(template: np.ndarray, channel_ids: list) -> int | None:
    """Recover a unit's shank from its extremum channel id ('sh{N}_...').

    The curated analyzer renames channels 'sh{ish}_{ch}' before aggregating
    shanks, and each unit is sparse to its own shank, so the channel carrying
    the largest template energy identifies the shank robustly.
    """
    template = np.asarray(template, dtype=float)
    if template.ndim != 2 or template.size == 0:
        return None
    channel_energy = np.max(np.abs(template), axis=0)
    if channel_energy.size == 0:
        return None
    ext_idx = int(np.nanargmax(channel_energy))
    if ext_idx >= len(channel_ids):
        return None
    match = re.search(r"sh(\d+)_", str(channel_ids[ext_idx]))
    return int(match.group(1)) if match else None


def autocorr_vector(train_samples: np.ndarray, fs: float, *, window_ms: float, bin_ms: float) -> np.ndarray:
    """Compute a normalized autocorrelogram from a (sorted) spike train.

    Replaces am.get_autocorr_vector (which would call analyzer.compute and
    persist a 'correlograms' extension into the on-disk curated_analyzer). The
    center bin is zeroed and the histogram is L1-normalized, matching the way
    the Centimani module conditions its autocorr vector before cosine scoring.

    Lag-based and fully vectorized: for spikes ``lag`` apart in index the time
    gap ``t[lag:] - t[:-lag]`` grows monotonically with ``lag``, so we sweep
    lags and stop as soon as no gap falls within the window. For realistic
    firing rates only a handful of lags land inside a 50 ms window, so this is
    orders of magnitude faster than a per-spike neighbor scan on the
    half-million-spike trains these sessions contain.
    """
    train = np.asarray(train_samples, dtype=np.float64)
    n = train.size
    if n < 2:
        return np.array([], dtype=float)
    times_ms = train / fs * 1000.0  # sorted ascending (spike trains are sorted)

    n_bins = int(round(2.0 * window_ms / bin_ms))
    edges = np.linspace(-window_ms, window_ms, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)
    for lag in range(1, n):
        gaps = times_ms[lag:] - times_ms[:-lag]  # positive, monotonic in lag
        within = gaps[gaps <= window_ms]
        if within.size == 0:
            break  # larger lags only have larger gaps
        counts += np.histogram(within, bins=edges)[0]   # +Δt side
        counts += np.histogram(-within, bins=edges)[0]  # symmetric −Δt side

    center = n_bins // 2
    if 0 <= center < n_bins:
        counts[center] = 0.0
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts


# ── Load one session's curated units as UnitFeatures ────────────────────────────
def load_session_units(
    *, session_name: str, session_path: Path, curated_path: Path,
    include_labels: set[str], window_ms: float, bin_ms: float,
) -> list:
    print(f"\n  Loading {session_name}: {curated_path}")
    # recording skipped (see _SKIP_RECORDING) and load_extensions=False so the
    # large 'waveforms' extension is never read; get_templates_array /
    # get_amplitudes_by_unit / get_locations lazily load only the three
    # extensions they actually use.
    analyzer = SortingAnalyzer.load(curated_path, recording=_SKIP_RECORDING, load_extensions=False)

    templates = am.get_templates_array(analyzer)
    amplitudes = am.get_amplitudes_by_unit(analyzer)
    locations = am.get_locations(analyzer)
    channel_ids = list(analyzer.channel_ids)
    unit_ids = am.get_unit_ids(analyzer)
    fs = float(analyzer.sampling_frequency)

    label_prop = analyzer.sorting.get_property("unit_label")
    label_by_uid: dict[str, str] = {}
    if label_prop is not None:
        for uid, lbl in zip(unit_ids, label_prop):
            label_by_uid[str(uid)] = str(lbl).lower()

    units = []
    for uid in unit_ids:
        label = label_by_uid.get(str(uid), "mua")
        if label not in include_labels:
            continue
        template = am.get_template_for_unit(analyzer, templates, uid)
        amplitude = amplitudes.get(str(uid), float(np.max(np.abs(template))))
        shank = shank_of_unit(template, channel_ids)
        train = analyzer.sorting.get_unit_spike_train(uid)
        units.append(
            am.UnitFeatures(
                session_label=session_name,
                session_root=str(session_path),
                # 'recording' carries the session id so per-pair CSVs are
                # self-describing (session1_recording / session2_recording).
                recording=session_name,
                analyzer_folder=str(curated_path),
                unit_id=int(uid) if str(uid).isdigit() else uid,
                label=label,
                shanks=(shank,) if shank is not None else (),
                template=template,
                waveform_vector=am.get_waveform_vector(template),
                autocorr_vector=autocorr_vector(train, fs, window_ms=window_ms, bin_ms=bin_ms),
                amplitude=float(amplitude),
                location=locations.get(str(uid)),
                num_spikes=int(len(train)),
            )
        )
    n_sua = sum(1 for u in units if u.label == "sua")
    print(f"    {len(units)} eligible units ({n_sua} SUA, {len(units) - n_sua} MUA).")
    return units


# ── Output helpers ───────────────────────────────────────────────────────────────
def write_session_units(units_by_session: dict, output_folder: Path) -> None:
    path = output_folder / "session_units.csv"
    fields = ["session", "unit_id", "label", "shank", "amplitude", "num_spikes"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for session_name, units in units_by_session.items():
            for u in units:
                writer.writerow({
                    "session": session_name,
                    "unit_id": u.unit_id,
                    "label": u.label,
                    "shank": ",".join(str(s) for s in u.shanks),
                    "amplitude": u.amplitude,
                    "num_spikes": u.num_spikes,
                })


def write_summary(all_rows: list, output_folder: Path) -> None:
    path = output_folder / "matched_units_all_pairs.csv"
    if not all_rows:
        path.write_text(",".join(am.MatchRow.__annotations__) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))


def write_counts(count_rows: list[dict], output_folder: Path) -> None:
    path = output_folder / "pair_match_counts.csv"
    fields = ["session1", "session2", "n_session1_units", "n_session2_units", "n_matches"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in count_rows:
            writer.writerow(row)


# ── Driver ───────────────────────────────────────────────────────────────────────
def threshold_tag(min_total: float, min_waveform: float, min_amplitude: float,
                  min_autocorr: float) -> str:
    """A filesystem-safe subfolder name encoding the matching thresholds, so a
    sweep writes each run to its own folder instead of overwriting.

    All four equal -> ``thres_0.70``; otherwise the floors are spelled out, e.g.
    ``t0.70_w0.75_a0.70_ac0.70``.
    """
    vals = (min_total, min_waveform, min_amplitude, min_autocorr)
    if len(set(vals)) == 1:
        return f"thres_{vals[0]:.2f}"
    return (f"t{min_total:.2f}_w{min_waveform:.2f}"
            f"_a{min_amplitude:.2f}_ac{min_autocorr:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--output-folder", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument("--include-labels", nargs="+", default=list(INCLUDE_LABELS))
    # Convenience for threshold sweeps: --thres X sets all four min-* floors at
    # once and overrides any individual --min-* given on the same command line.
    parser.add_argument("--thres", type=float, default=None,
                        help="set min-total/waveform/amplitude/autocorr to this value")
    parser.add_argument("--min-total", type=float, default=MIN_TOTAL)
    parser.add_argument("--min-waveform", type=float, default=MIN_WAVEFORM)
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE)
    parser.add_argument("--min-autocorr", type=float, default=MIN_AUTOCORR)
    parser.add_argument("--max-location-distance-um", type=float, default=MAX_LOCATION_DISTANCE_UM)
    parser.add_argument("--no-require-shank-overlap", action="store_false", dest="require_shank_overlap",
                        default=REQUIRE_SHANK_OVERLAP)
    parser.add_argument("--no-figures", action="store_false", dest="figures", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_labels = {label.lower() for label in args.include_labels}
    weights = {
        "waveform": WAVEFORM_WEIGHT,
        "amplitude": AMPLITUDE_WEIGHT,
        "autocorr": AUTOCORR_WEIGHT,
        "location": LOCATION_WEIGHT,
    }

    # --thres is a shorthand that sets all four floors at once.
    if args.thres is not None:
        args.min_total = args.min_waveform = args.min_amplitude = args.min_autocorr = args.thres

    # Give each threshold setting its own subfolder so a sweep never overwrites.
    tag = threshold_tag(args.min_total, args.min_waveform, args.min_amplitude, args.min_autocorr)
    output_folder = args.output_folder / tag
    pairs_dir = output_folder / "pairs"
    print(f"Threshold tag          : {tag}")
    output_folder.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(args.csv_path)
    print(f"Discovered {len(sessions)} session(s) with a curated_analyzer.")
    if len(sessions) < 2:
        print("Need at least 2 sessions to match. Aborting.")
        return

    # ── Load every session's units once ────────────────────────────────────────
    units_by_session: dict[str, list] = {}
    for session_name, session_path, curated_path in sessions:
        try:
            units_by_session[session_name] = load_session_units(
                session_name=session_name, session_path=session_path,
                curated_path=curated_path, include_labels=include_labels,
                window_ms=AUTOCORR_WINDOW_MS, bin_ms=AUTOCORR_BIN_MS,
            )
        except Exception as exc:  # one bad analyzer shouldn't kill the whole run
            print(f"    [error] failed to load {session_name}: {exc}")
            units_by_session[session_name] = []

    write_session_units(units_by_session, output_folder)

    # ── Match every session pair (i < j) ───────────────────────────────────────
    session_names = [name for name, _, _ in sessions]
    all_rows = []
    count_rows = []
    n_pairs = len(session_names) * (len(session_names) - 1) // 2
    print(f"\nMatching {n_pairs} session pair(s)...")

    for a_name, b_name in combinations(session_names, 2):
        units_a = units_by_session[a_name]
        units_b = units_by_session[b_name]
        matches = am.find_matches(
            units_a, units_b,
            min_total=args.min_total,
            min_waveform=args.min_waveform,
            min_amplitude=args.min_amplitude,
            min_autocorr=args.min_autocorr,
            max_location_distance_um=args.max_location_distance_um,
            require_shank_overlap=args.require_shank_overlap,
            weights=weights,
        )
        rows = am.make_match_rows(matches)
        count_rows.append({
            "session1": a_name, "session2": b_name,
            "n_session1_units": len(units_a), "n_session2_units": len(units_b),
            "n_matches": len(rows),
        })
        all_rows.extend(rows)

        if rows:
            pair_folder = pairs_dir / f"{a_name}__{b_name}"
            config = {
                "session1": a_name, "session2": b_name,
                "include_labels": sorted(include_labels),
                "min_total": args.min_total, "min_waveform": args.min_waveform,
                "min_amplitude": args.min_amplitude, "min_autocorr": args.min_autocorr,
                "max_location_distance_um": args.max_location_distance_um,
                "require_shank_overlap": args.require_shank_overlap,
                "weights": weights,
                "num_session1_units": len(units_a), "num_session2_units": len(units_b),
                "num_matches": len(rows),
            }
            am.write_outputs(rows, pair_folder, config)
            if args.figures:
                am.save_match_figure(matches, pair_folder)
            print(f"  {a_name} <-> {b_name}: {len(rows)} match(es)")

    write_summary(all_rows, output_folder)
    write_counts(count_rows, output_folder)

    total_matches = sum(c["n_matches"] for c in count_rows)
    pairs_with_matches = sum(1 for c in count_rows if c["n_matches"] > 0)
    print(f"\n{'='*60}")
    print(f"ALL-PAIRS MATCHING COMPLETE")
    print(f"  Sessions               : {len(session_names)}")
    print(f"  Pairs compared         : {n_pairs}")
    print(f"  Pairs with >=1 match   : {pairs_with_matches}")
    print(f"  Total matched pairs    : {total_matches}")
    print(f"  Output folder          : {output_folder}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
