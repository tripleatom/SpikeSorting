r"""
match_two_sessions.py
======================
Cross-session unit matching for two arbitrary sessions that each already have
a combined ``curated_analyzer`` (built by MsCuratedAnalyzer.py) at their root.

This is a two-session analogue of match_units_CnL42SG.py: it reuses the same
scoring core (waveform / amplitude / autocorr / location similarity, the
greedy one-to-one matcher, and the CSV/JSON/figure writers) from Centimani's
``auto_match_curated_sessions.py``, but takes the two curated_analyzer
folders directly instead of discovering sessions from a curation-plan CSV.
Use this for animals (e.g. CnL43) that don't have such a CSV.

Output (in --output-folder, default next to session 1):
  matched_units.csv / .json   the matched pairs and their scores
  matched_units.pdf           per-match waveform comparison figure
  simple_matched_units.csv    session1 export id / session2 export id / score
  session_units.csv           every eligible unit from both sessions
  pair_match_counts.csv       n units / n matches summary (one row)

Usage
-----
  python match_two_sessions.py --session1-path <curated_analyzer 1> --session1-name 2026-04-08 \
                               --session2-path <curated_analyzer 2> --session2-name 2026-06-23
  python match_two_sessions.py --session1-path ... --session2-path ... --thres 0.7 --no-figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from spikeinterface.core.sortinganalyzer import SortingAnalyzer

# ── Reuse the Centimani auto-match module as the scoring core ───────────────────
AUTO_MATCH_MODULE_DIR = Path(r"D:\Cemtimani\code\Centimani\curation")
if str(AUTO_MATCH_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_MATCH_MODULE_DIR))
import auto_match_curated_sessions as am  # noqa: E402

import re  # noqa: E402


# ============================== USER CONFIG (defaults) ==============================
# Edit these paths for interactive/script runs. Command-line --session1-path and
# --session2-path still override these values when provided.
SESSION1_PATH: Path | None = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout\CnL43\CnL43_20260408\curated_analyzer")
SESSION2_PATH: Path | None = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout\CnL43\CnL43_20260623\curated_analyzer")
SESSION1_EXPORT_PKL: Path | None = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout\CnL43\CnL43_20260408\CnL43_20260408_curated_units_export.pkl")
SESSION2_EXPORT_PKL: Path | None = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout\CnL43\CnL43_20260623\CnL43_20260623_curated_units_export.pkl")

INCLUDE_LABELS = ("sua", "mua")

thres = 0.0
MIN_TOTAL = thres
MIN_WAVEFORM = 0.3
MIN_AMPLITUDE = thres
MIN_AUTOCORR = thres

# Units on different shanks cannot be the same unit, so require same-shank.
MAX_LOCATION_DISTANCE_UM = 100.0
REQUIRE_SHANK_OVERLAP = True

WAVEFORM_WEIGHT = 0.30
AMPLITUDE_WEIGHT = 0.20
AUTOCORR_WEIGHT = 0.30
LOCATION_WEIGHT = 0.20
# Graded major-channel agreement (same contact = 1, 1 apart = 0.5, ... 0 on a
# different shank). Dropped from the decision (renormalised away) for any pair
# where contact_pitch_um could not be inferred or a position is missing.
MAJOR_CHANNEL_WEIGHT = 0.10
MAJOR_CHANNEL_MATCH_TOL_UM = 30.0

AUTOCORR_WINDOW_MS = 50.0
AUTOCORR_BIN_MS = 1.0
# ====================================================================================

_SKIP_RECORDING = "__skip_recording__"


def shank_of_unit(template: np.ndarray, channel_ids: list) -> int | None:
    """Recover a unit's shank from its extremum channel id ('sh{N}_...')."""
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


def unit_property_lookup(analyzer, property_name: str, unit_ids: list) -> dict[str, str]:
    values = analyzer.sorting.get_property(property_name)
    if values is None:
        return {}
    return {str(uid): str(value) for uid, value in zip(unit_ids, values)}


def resolve_export_id(
    export_id_map: dict[tuple[str, str], int],
    *,
    session_name: str,
    unit_id,
    original_unit_id: str | None,
    recording: str | None,
) -> int | None:
    unit_id_text = str(unit_id)
    candidates = []
    if recording is not None and original_unit_id is not None:
        candidates.append((recording, original_unit_id))
    if recording is not None:
        candidates.append((recording, unit_id_text))
    if original_unit_id is not None:
        candidates.append((session_name, original_unit_id))
        candidates.append(("", original_unit_id))
    candidates.append((session_name, unit_id_text))
    candidates.append(("", unit_id_text))

    for key in candidates:
        if key in export_id_map:
            return export_id_map[key]
    return None


def autocorr_vector(train_samples: np.ndarray, fs: float, *, window_ms: float, bin_ms: float) -> np.ndarray:
    """Normalized autocorrelogram from a spike train, computed locally (no on-disk writes)."""
    train = np.asarray(train_samples, dtype=np.float64)
    n = train.size
    if n < 2:
        return np.array([], dtype=float)
    times_ms = train / fs * 1000.0

    n_bins = int(round(2.0 * window_ms / bin_ms))
    edges = np.linspace(-window_ms, window_ms, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)
    for lag in range(1, n):
        gaps = times_ms[lag:] - times_ms[:-lag]
        within = gaps[gaps <= window_ms]
        if within.size == 0:
            break
        counts += np.histogram(within, bins=edges)[0]
        counts += np.histogram(-within, bins=edges)[0]

    center = n_bins // 2
    if 0 <= center < n_bins:
        counts[center] = 0.0
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts


def load_session_units(
    *, session_name: str, curated_path: Path,
    include_labels: set[str], window_ms: float, bin_ms: float,
    export_id_map: dict[tuple[str, str], int] | None = None,
) -> tuple[list, float | None]:
    print(f"\n  Loading {session_name}: {curated_path}")
    analyzer = SortingAnalyzer.load(curated_path, recording=_SKIP_RECORDING, load_extensions=False)

    templates = am.get_templates_array(analyzer)
    amplitudes = am.get_amplitudes_by_unit(analyzer)
    locations = am.get_locations(analyzer)
    channel_ids = list(analyzer.channel_ids)
    # Templates are dense (n_units, n_samples, n_channels): sparsity zeroes out
    # inactive channels rather than compacting the array, so the channel axis
    # lines up with channel_locations/channel_ids for every unit.
    channel_locations = np.asarray(analyzer.get_channel_locations(), dtype=float)
    channel_depths = channel_locations[:, 1]
    contact_pitch = am.infer_contact_pitch_um(channel_locations)
    unit_ids = am.get_unit_ids(analyzer)
    fs = float(analyzer.sampling_frequency)

    label_prop = analyzer.sorting.get_property("unit_label")
    label_by_uid: dict[str, str] = {}
    if label_prop is not None:
        for uid, lbl in zip(unit_ids, label_prop):
            label_by_uid[str(uid)] = str(lbl).lower()
    original_unit_by_uid = unit_property_lookup(analyzer, "original_unit_id", unit_ids)
    recording_by_uid = unit_property_lookup(analyzer, "recording", unit_ids)

    units = []
    export_id = 0
    n_unresolved = 0
    for uid in unit_ids:
        label = label_by_uid.get(str(uid), "mua")
        if label not in include_labels:
            continue
        if export_id_map is not None:
            resolved_export_id = resolve_export_id(
                export_id_map,
                session_name=session_name,
                unit_id=uid,
                original_unit_id=original_unit_by_uid.get(str(uid)),
                recording=recording_by_uid.get(str(uid)),
            )
            if resolved_export_id is None:
                n_unresolved += 1
                continue
            current_export_id = resolved_export_id
        else:
            current_export_id = export_id
            export_id += 1
        template = am.get_template_for_unit(analyzer, templates, uid)
        amplitude = amplitudes.get(str(uid), float(np.max(np.abs(template))))
        shank = shank_of_unit(template, channel_ids)
        train = analyzer.sorting.get_unit_spike_train(uid)
        unit = am.UnitFeatures(
            session_label=session_name,
            session_root=str(curated_path.parent),
            recording=session_name,
            analyzer_folder=str(curated_path),
            unit_id=int(uid) if str(uid).isdigit() else uid,
            export_id=current_export_id,
            label=label,
            shanks=(shank,) if shank is not None else (),
            template=template,
            channel_depths=channel_depths,
            major_channel_position=am.compute_major_channel_position(template, channel_locations),
            waveform_vector=am.get_waveform_vector(template),
            autocorr_vector=autocorr_vector(train, fs, window_ms=window_ms, bin_ms=bin_ms),
            amplitude=float(amplitude),
            location=locations.get(str(uid)),
            num_spikes=int(len(train)),
            channel_locations=channel_locations,
            channel_ids=channel_ids,
        )
        units.append(unit)
    n_sua = sum(1 for u in units if u.label == "sua")
    print(f"    {len(units)} eligible units ({n_sua} SUA, {len(units) - n_sua} MUA).")
    if n_unresolved:
        print(f"    WARNING: skipped {n_unresolved} eligible units not found in export pkl.")
    return units, contact_pitch


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


def write_counts(count_row: dict, output_folder: Path) -> None:
    path = output_folder / "pair_match_counts.csv"
    fields = ["session1", "session2", "n_session1_units", "n_session2_units", "n_matches"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(count_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session1-path", type=Path, default=SESSION1_PATH,
                        help="curated_analyzer folder for session 1")
    parser.add_argument("--session2-path", type=Path, default=SESSION2_PATH,
                        help="curated_analyzer folder for session 2")
    parser.add_argument("--session1-export-pkl", type=Path, default=SESSION1_EXPORT_PKL,
                        help="optional units_export pkl for session 1")
    parser.add_argument("--session2-export-pkl", type=Path, default=SESSION2_EXPORT_PKL,
                        help="optional units_export pkl for session 2")
    parser.add_argument("--session1-name", default=None,
                        help="defaults to session1-path's parent folder name")
    parser.add_argument("--session2-name", default=None,
                        help="defaults to session2-path's parent folder name")
    parser.add_argument("--output-folder", type=Path, default=None,
                        help="defaults to <session1's parent>/<s1>_to_<s2>_unit_match")
    parser.add_argument("--include-labels", nargs="+", default=list(INCLUDE_LABELS))
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
    if args.session1_path is None or args.session2_path is None:
        raise SystemExit(
            "Set SESSION1_PATH and SESSION2_PATH in USER CONFIG, "
            "or pass --session1-path and --session2-path."
        )

    include_labels = {label.lower() for label in args.include_labels}
    weights = {
        "waveform": WAVEFORM_WEIGHT,
        "amplitude": AMPLITUDE_WEIGHT,
        "autocorr": AUTOCORR_WEIGHT,
        "location": LOCATION_WEIGHT,
        "major_channel": MAJOR_CHANNEL_WEIGHT,
    }

    if args.thres is not None:
        args.min_total = args.min_waveform = args.min_amplitude = args.min_autocorr = args.thres

    session1_name = args.session1_name or args.session1_path.parent.name
    session2_name = args.session2_name or args.session2_path.parent.name
    output_folder = args.output_folder or (
        args.session1_path.parent.parent / f"{session1_name}_to_{session2_name}_unit_match"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}")

    export_map1 = None
    if args.session1_export_pkl is not None:
        export_map1, n_export1 = am.load_export_id_map(args.session1_export_pkl)
        print(f"Loaded {n_export1} export ids from {args.session1_export_pkl.name}")
    export_map2 = None
    if args.session2_export_pkl is not None:
        export_map2, n_export2 = am.load_export_id_map(args.session2_export_pkl)
        print(f"Loaded {n_export2} export ids from {args.session2_export_pkl.name}")

    units1, pitch1 = load_session_units(
        session_name=session1_name, curated_path=args.session1_path,
        include_labels=include_labels, window_ms=AUTOCORR_WINDOW_MS, bin_ms=AUTOCORR_BIN_MS,
        export_id_map=export_map1,
    )
    units2, pitch2 = load_session_units(
        session_name=session2_name, curated_path=args.session2_path,
        include_labels=include_labels, window_ms=AUTOCORR_WINDOW_MS, bin_ms=AUTOCORR_BIN_MS,
        export_id_map=export_map2,
    )
    write_session_units({session1_name: units1, session2_name: units2}, output_folder)

    inferred_pitches = [p for p in (pitch1, pitch2) if p is not None]
    contact_pitch_um = float(np.median(inferred_pitches)) if inferred_pitches else None
    if weights["major_channel"] > 0:
        if contact_pitch_um is not None:
            print(f"Major-channel score using contact pitch = {contact_pitch_um:.1f} um")
        else:
            print("WARNING: contact pitch unavailable; major-channel score disabled for this run")

    matches = am.find_matches(
        units1, units2,
        min_total=args.min_total,
        min_waveform=args.min_waveform,
        min_amplitude=args.min_amplitude,
        min_autocorr=args.min_autocorr,
        max_location_distance_um=args.max_location_distance_um,
        require_shank_overlap=args.require_shank_overlap,
        weights=weights,
        major_channel_match_tol_um=MAJOR_CHANNEL_MATCH_TOL_UM,
        contact_pitch_um=contact_pitch_um,
    )
    rows = am.make_match_rows(matches)

    config = {
        "session1": session1_name, "session2": session2_name,
        "session1_curated_analyzer": str(args.session1_path),
        "session2_curated_analyzer": str(args.session2_path),
        "session1_export_pkl": str(args.session1_export_pkl) if args.session1_export_pkl else None,
        "session2_export_pkl": str(args.session2_export_pkl) if args.session2_export_pkl else None,
        "include_labels": sorted(include_labels),
        "min_total": args.min_total, "min_waveform": args.min_waveform,
        "min_amplitude": args.min_amplitude, "min_autocorr": args.min_autocorr,
        "max_location_distance_um": args.max_location_distance_um,
        "major_channel_match_tol_um": MAJOR_CHANNEL_MATCH_TOL_UM,
        "contact_pitch_um": contact_pitch_um,
        "require_shank_overlap": args.require_shank_overlap,
        "weights": weights,
        "num_session1_units": len(units1), "num_session2_units": len(units2),
        "num_matches": len(rows),
    }
    am.write_outputs(rows, output_folder, config)
    am.write_simple_match_csv(matches, output_folder / "simple_matched_units.csv")
    if args.figures and rows:
        am.save_match_figure(matches, output_folder / "matched_units.pdf")

    write_counts({
        "session1": session1_name, "session2": session2_name,
        "n_session1_units": len(units1), "n_session2_units": len(units2),
        "n_matches": len(rows),
    }, output_folder)

    print(f"\n{'='*60}")
    print("TWO-SESSION MATCHING COMPLETE")
    print(f"  {session1_name}: {len(units1)} eligible units")
    print(f"  {session2_name}: {len(units2)} eligible units")
    print(f"  Matched pairs        : {len(rows)}")
    print(f"  Output folder        : {output_folder}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
