"""
3_curated_analyzer.py
=====================
Post-curation: build a combined SortingAnalyzer that reflects curated
labels (unit_labels.json) and merge decisions (unit_merge_map.json).

Steps performed per shank:
  1. Load the SortingAnalyzer produced by 1_ms_sorting.py
  2. Remove Noise units (from unit_labels.json)
  3. Merge over-split units (from unit_merge_map.json; user overrides auto)

Then across all shanks:
  4. Aggregate recordings and sortings
  5. Create a single SortingAnalyzer with shank-local sparsity
  6. Compute waveforms, templates, quality metrics, etc.

Output: session_folder / 'curated_analyzer/'

Usage
-----
  python 3_curated_analyzer.py              # reads curated_analyzer_files.json
  python 3_curated_analyzer.py --overwrite  # re-create even if folder exists
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from spikeinterface import create_sorting_analyzer
from spikeinterface.core import aggregate_channels, aggregate_units
from rec2nwb.preproc_func import parse_session_info

# ── Configuration (edit when running directly) ─────────────────────────────────
rec_folder     = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260313\CnL42SG_20260313")
sortout_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout")
shanks         = [0, 1, 2, 3, 4, 5, 6, 7]
n_jobs         = 8
overwrite      = False


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cast_uid(uid_str: str, all_unit_ids: list):
    """Resolve a string unit ID to the type used internally by the sorting."""
    if uid_str in all_unit_ids:
        return uid_str
    try:
        uid_int = int(uid_str)
        if uid_int in all_unit_ids:
            return uid_int
    except (ValueError, TypeError):
        pass
    return None


def _resolve_merge_map(shank_key: str, merge_map_json: dict, all_unit_ids: list) -> dict:
    """
    Build a {native_uid: native_canonical} map for one shank.

    - auto map provides the base (from run_merge_pass)
    - user map overrides at the per-unit level
    - units absent from both maps are identity-mapped (no merge)
    - if the canonical unit was labeled Noise and removed, falls back to identity

    Returns
    -------
    dict: native_uid -> native_canonical for every uid in all_unit_ids
    """
    auto_map = merge_map_json.get("auto", {}).get(shank_key, {})
    user_map = merge_map_json.get("user", {}).get(shank_key, {})
    combined_str = {**auto_map, **user_map}   # user overrides auto

    result = {}
    for uid in all_unit_ids:
        uid_str = str(uid)
        if uid_str in combined_str:
            canonical_str = combined_str[uid_str]
            canonical = _cast_uid(canonical_str, all_unit_ids)
            if canonical is None:
                print(f"  [Warning] Canonical '{canonical_str}' not found after "
                      f"Noise removal — identity-mapping {uid}")
                canonical = uid
        else:
            canonical = uid   # not mentioned → no merge
        result[uid] = canonical
    return result


def _merge_map_to_groups(merge_map: dict):
    """
    Convert {uid: canonical} -> (units_to_merge, new_unit_ids).
    Only groups with >= 2 members are returned (actual merges).
    """
    groups_by_canon = defaultdict(list)
    for uid, canonical in merge_map.items():
        groups_by_canon[canonical].append(uid)

    units_to_merge = []
    new_unit_ids   = []
    for canonical, members in groups_by_canon.items():
        if len(members) >= 2:
            units_to_merge.append(members)
            new_unit_ids.append(canonical)
    return units_to_merge, new_unit_ids


# ── Per-shank processing ───────────────────────────────────────────────────────
def process_shank(ish: int, session_folder: Path, rec_folder: Path,
                  labels_json: dict, merge_map_json: dict, n_jobs: int = 1):
    """
    Load one shank, apply curation labels and merges.

    Returns (recording_filtered_renamed, curated_sorting) or None if the
    shank should be skipped (files missing or all units are Noise).
    """
    print(f"\n  shank{ish}: loading data...")

    # ── SortingAnalyzer ────────────────────────────────────────────────────────
    shank_folder = session_folder / f"shank{ish}"
    if not shank_folder.exists():
        print(f"  shank{ish}: shank folder not found: {shank_folder}")
        return None

    sorting_dirs = sorted(shank_folder.glob("sorting_results_*"))
    if not sorting_dirs:
        print(f"  shank{ish}: no sorting_results_* found in {shank_folder}")
        return None

    analyzer_folder = sorting_dirs[-1] / "sorting_analyzer"
    if not analyzer_folder.exists():
        print(f"  shank{ish}: sorting_analyzer not found in {sorting_dirs[-1]}")
        return None

    sa      = si.load_sorting_analyzer(str(analyzer_folder))
    all_ids = list(sa.sorting.get_unit_ids())

    # ── Determine Noise vs. good units ────────────────────────────────────────
    shank_key     = f"shank{ish}"
    shank_labels  = labels_json.get(shank_key, {})
    noise_id_strs = {str(uid) for uid in all_ids
                     if shank_labels.get(str(uid), "MUA") == "Noise"}
    good_ids      = [uid for uid in all_ids if str(uid) not in noise_id_strs]

    if not good_ids:
        print(f"  shank{ish}: all {len(all_ids)} units labeled Noise — skipping.")
        return None

    print(f"  shank{ish}: {len(noise_id_strs)} Noise unit(s) removed, "
          f"{len(good_ids)} remaining.")

    # ── Load recording (needed for hard-mode merge and for the combined analyzer)
    # Glob for *sh{ish}.nwb because the date part of the filename may differ
    # from the folder name (e.g. "CnL42SG_260313sh0.nwb" vs "CnL42SG_20260313").
    nwb_candidates = sorted(rec_folder.glob(f"*sh{ish}.nwb"))
    if not nwb_candidates:
        print(f"  shank{ish}: no *sh{ish}.nwb found in {rec_folder}")
        return None
    recording_file = nwb_candidates[0]
    recording  = se.read_nwb_recording(str(recording_file))
    rec_filt   = sp.bandpass_filter(recording, freq_min=300, freq_max=6000,
                                    dtype=np.float32)

    # Attach the recording so hard-mode merge can recompute noise_levels etc.
    sa.set_temporary_recording(rec_filt)

    # ── Apply merge map via analyzer.merge_units() ────────────────────────────
    # Build groups using only non-noise units (merge map only covers SUA+MUA).
    merge_map      = _resolve_merge_map(shank_key, merge_map_json, good_ids)
    units_to_merge, _ = _merge_map_to_groups(merge_map)

    if units_to_merge:
        merged_count = sum(len(g) for g in units_to_merge) - len(units_to_merge)
        print(f"  shank{ish}: merging {merged_count} unit(s) into "
              f"{len(units_to_merge)} group(s).")
        # Noise units still exist in merged_sa; filter them out afterwards.
        merged_sa   = sa.merge_units(merge_unit_groups=units_to_merge,
                                     merging_mode="hard", n_jobs=n_jobs)
        curated_ids = [uid for uid in merged_sa.sorting.get_unit_ids()
                       if str(uid) not in noise_id_strs]
        sorting_curated = merged_sa.sorting.select_units(curated_ids)
    else:
        sorting_curated = sa.sorting.select_units(good_ids)

    print(f"  shank{ish}: {len(list(sorting_curated.get_unit_ids()))} curated units.")

    # Rename channels to avoid ID collisions when aggregating across shanks.
    new_ch_ids = [f"sh{ish}_{ch}" for ch in rec_filt.get_channel_ids()]
    rec_filt   = rec_filt.rename_channels(new_ch_ids)

    return rec_filt, sorting_curated


# ── Main ───────────────────────────────────────────────────────────────────────
def main(rec_folder, shanks, sortout_folder, animal_id="", overwrite=False, n_jobs=8,
         compute_extensions=True):
    """
    Build a combined curated SortingAnalyzer for all shanks of one session.

    Parameters
    ----------
    rec_folder    : folder containing the per-shank NWB recordings
    shanks        : list of shank indices to include
    sortout_folder: top-level sortout directory (animal/session derived inside)
    animal_id          : optional override; parsed from rec_folder if empty
    overwrite          : re-create the output folder if it already exists
    n_jobs             : parallelism for waveform / metric computation
    compute_extensions : if True, compute waveforms, templates, spike_amplitudes,
                         template_similarity, and unit_locations on the combined
                         analyzer. quality_metrics are intentionally skipped here
                         because per-shank analyzers already have them for unmerged
                         units; compute them manually afterwards if needed.
                         Set False to just create the analyzer shell.
    """
    animal_id_parsed, session_id, _ = parse_session_info(str(rec_folder))
    if not animal_id:
        animal_id = animal_id_parsed

    session_folder = Path(sortout_folder) / animal_id / f"{animal_id}_{session_id}"
    output_folder  = session_folder / "curated_analyzer"

    print(f"Session folder : {session_folder}")
    print(f"Output folder  : {output_folder}")

    if output_folder.exists() and not overwrite:
        print("curated_analyzer already exists. Set overwrite=True to recreate.")
        return

    # ── Load curation JSONs ────────────────────────────────────────────────────
    labels_path    = session_folder / "unit_labels.json"
    merge_map_path = session_folder / "unit_merge_map.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"unit_labels.json not found: {labels_path}\n"
                                "Run curation_lazy.py first.")
    with open(labels_path) as f:
        labels_json = json.load(f)

    if merge_map_path.exists():
        with open(merge_map_path) as f:
            merge_map_json = json.load(f)
    else:
        print("unit_merge_map.json not found — no merges will be applied.")
        merge_map_json = {"auto": {}, "user": {}, "blacklist": {}}

    # ── Process each shank ─────────────────────────────────────────────────────
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
    print(f"\nProcessing {len(shanks)} shank(s): {shanks}")
    recordings, sortings = [], []
    for ish in shanks:
        result = process_shank(ish, session_folder, Path(rec_folder),
                               labels_json, merge_map_json, n_jobs=n_jobs)
        if result is None:
            continue
        rec_sh, sort_sh = result
        recordings.append(rec_sh)
        sortings.append(sort_sh)

    if not recordings:
        print("No shanks produced curated data — aborting.")
        return

    # ── Aggregate across shanks ────────────────────────────────────────────────
    combined_recording = aggregate_channels(recordings)
    combined_sorting   = aggregate_units(sortings)
    total_units = len(list(combined_sorting.get_unit_ids()))
    print(f"\nCombined: {combined_recording.get_num_channels()} channels, "
          f"{total_units} units across {len(recordings)} shank(s).")

    # Each unit's waveforms are extracted only from its own shank's channels.
    # aggregate_units sets a 'group' integer property (0-indexed over sortings
    # list) on every unit; aggregate_channels sets the same 'group' on channels.
    # by_property="group" matches these so waveforms stay shank-local.
    sparsity = si.compute_sparsity(
        combined_sorting, combined_recording,
        method="by_property", by_property="group",
    )

    # ── Create curated SortingAnalyzer ─────────────────────────────────────────
    analyzer = create_sorting_analyzer(
        combined_sorting, combined_recording,
        format="binary_folder",
        folder=output_folder,
        overwrite=overwrite,
        sparsity=sparsity,
    )

    # ── Compute extensions ─────────────────────────────────────────────────────
    # quality_metrics are NOT recomputed here: for unmerged units the per-shank
    # analyzers already have them; for merged units call analyzer.compute(
    # "quality_metrics", ...) manually after this script finishes.
    if compute_extensions:
        print("\nComputing extensions...")
        analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"],
                         n_jobs=n_jobs)
        analyzer.compute("spike_amplitudes", n_jobs=n_jobs)
        analyzer.compute("template_similarity")
        analyzer.compute("unit_locations")
    else:
        print("\nSkipping extension computation (compute_extensions=False).")

    print(f"\nDone. Curated analyzer saved → {output_folder}")
    print(f"  {total_units} units total.")
    return analyzer


# ── Batch processing via JSON config ──────────────────────────────────────────
def process_from_json(json_file="curated_analyzer_files.json"):
    """
    JSON format (same schema as phy_files.json):
    {
        "sortout": "/path/to/sortout",
        "n_jobs": 8,
        "overwrite": false,
        "recordings": [
            {"path": "/path/to/rec_folder", "shanks": [0, 1], "animal_id": "CnL42SG"}
        ]
    }
    """
    script_dir = Path(__file__).parent
    json_path  = script_dir / json_file

    with open(json_path, "r") as f:
        config = json.load(f)

    sortout = config.get("sortout")
    if sortout is None:
        raise ValueError("'sortout' key is required in the JSON config.")

    n_jobs_default = config.get("n_jobs", 8)
    overwrite_cfg  = config.get("overwrite", False)

    for rec in config["recordings"]:
        rec_path  = Path(rec["path"])
        shanks_   = rec.get("shanks", [0])
        animal_id = rec.get("animal_id", "")
        rec_n_jobs = rec.get("n_jobs", n_jobs_default)

        if not rec_path.exists():
            print(f"Recording folder not found: {rec_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {rec_path.name}  shanks={shanks_}")
        print(f"{'='*60}")
        main(rec_path, shanks_, sortout,
             animal_id=animal_id, overwrite=overwrite_cfg, n_jobs=rec_n_jobs)


if __name__ == "__main__":
    import sys
    _ow = "--overwrite" in sys.argv
    _cfg_path = Path(__file__).parent / "curated_analyzer_files.json"
    if _cfg_path.exists():
        if _ow:
            # Temporarily patch overwrite flag without modifying the file
            import tempfile, os
            with open(_cfg_path) as _f:
                _cfg = json.load(_f)
            _cfg["overwrite"] = True
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as _tf:
                json.dump(_cfg, _tf)
                _tmp = _tf.name
            process_from_json(_tmp)
            os.unlink(_tmp)
        else:
            process_from_json()
    else:
        # No JSON config — fall back to the top-level variables above
        main(rec_folder, shanks, sortout_folder, overwrite=_ow, n_jobs=n_jobs)
