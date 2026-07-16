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
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from spikeinterface import create_sorting_analyzer
from spikeinterface.core import aggregate_channels, aggregate_units, ChannelSparsity
# Ensure this repo's root is imported before any similarly-named package that a
# sibling project (e.g. ContinualLearning/rec2nwb) may have put on the path.
# When run as `python curation/MsCuratedAnalyzer.py`, sys.path[0] is curation/,
# not the repo root, so `import rec2nwb` would otherwise pick up the wrong copy.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rec2nwb.preproc_func import parse_session_info
from rec2nwb.utils.file_io import load_bad_ch

# ── Configuration (edit when running directly) ─────────────────────────────────
rec_folder     = Path(r"/Volumes/xieluanlabs2/xl_cl/zebra_noise/CnL43_20260623/CnL43_20260623_133122.rec")  # folder containing the per-shank NWB recordings; also used to parse animal/session ID if not provided explicitly")
sortout_folder = Path(r"/Volumes/xieluanlabs2/xl_cl/sortout")
shanks         = [0,1,2,3,4,5,6,7]  # list of shanks to process; set empty to auto-detect from rec_folder
n_jobs         = 24
overwrite      = True
USE_CACHE      = False     # write temp binary (True) vs compute on lazy recording (False)
# z_offsets: set per-shank z coordinate (µm) when two probes share identical
# x/y geometry (e.g. two 4-shank probes implanted at different depths).
# Shanks from the same probe get the same z; leave as {} for a single probe.
z_offsets      = {0: 0, 1: 0, 2: 0, 3: 0, 4: 500, 5: 500, 6: 500, 7: 500}


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
                  labels_json: dict, merge_map_json: dict, n_jobs: int = 1,
                  device_type: str = None, bad_ch_ids: list = None,
                  impedance_path: Path = None):
    """
    Load one shank, apply curation labels and merges.

    The per-shank recording is sourced NWB-first: an existing ``*sh{ish}.nwb``
    in ``rec_folder`` is preferred; if none is found, the recording is built
    directly from the ``.rec`` parts via
    ``rec2nwb.direct_recording.build_sortable_recording`` (requires
    ``device_type``).

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

    # ── Load recording (needed for hard-mode merge and for the combined analyzer)
    # Loaded FIRST so it can be handed to the analyzer loader below. NWB-first,
    # .rec-fallback. Glob for *sh{ish}.nwb because the date part of the filename
    # may differ from the folder name (e.g. "CnL42SG_260313sh0.nwb" vs
    # "CnL42SG_20260313"). If no NWB exists, build the recording directly from
    # the .rec parts (same object MsSorting's direct_sort path feeds in).
    # read_nwb_recording / bandpass_filter are lazy, so this is ~instant.
    nwb_candidates = sorted(rec_folder.glob(f"*sh{ish}.nwb"))
    if nwb_candidates:
        recording = se.read_nwb_recording(str(nwb_candidates[0]))
    else:
        if device_type is None:
            print(f"  shank{ish}: no *sh{ish}.nwb in {rec_folder} and no "
                  f"device_type for the .rec fallback — skipping.")
            return None
        print(f"  shank{ish}: no NWB found — building recording directly from "
              f".rec (device_type={device_type}).")
        try:
            from rec2nwb.direct_recording import build_sortable_recording
            recording = build_sortable_recording(
                data_folder=rec_folder,
                shank=ish,
                device_type=device_type,
                impedance_path=impedance_path,
                bad_ch_ids=bad_ch_ids,
            )
        except Exception as e:
            print(f"  shank{ish}: build_sortable_recording failed: {e}")
            return None
    rec_filt   = sp.bandpass_filter(recording, freq_min=300, freq_max=6000,
                                    dtype=np.float32)

    # ── Load the SortingAnalyzer WITHOUT reading recording.json ────────────────
    # Passing recording=rec_filt makes load_from_binary_folder skip the
    # recording.{json,pickle} reconstruction entirely. That file is ~1.8 GB for
    # these sessions (the recording graph was serialized with inline data), and
    # parsing it was ~40 s/shank over the network share — the whole "loading
    # data" cost. The provided recording also drives the hard-mode merge below,
    # exactly as the old set_temporary_recording(rec_filt) did. No saved
    # extensions are loaded; main() recomputes them on the combined analyzer.
    from spikeinterface.core.sortinganalyzer import SortingAnalyzer
    sa      = SortingAnalyzer.load_from_binary_folder(str(analyzer_folder), recording=rec_filt)
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

    # ── Apply merge map via analyzer.merge_units() ────────────────────────────
    # Build groups using only non-noise units (merge map only covers SUA+MUA).
    merge_map      = _resolve_merge_map(shank_key, merge_map_json, good_ids)
    units_to_merge, merge_canonicals = _merge_map_to_groups(merge_map)

    if units_to_merge:
        merged_count = sum(len(g) for g in units_to_merge) - len(units_to_merge)
        print(f"  shank{ish}: merging {merged_count} unit(s) into "
              f"{len(units_to_merge)} group(s).")
        # Pass our canonical IDs so SI uses them (makes label lookup trivial).
        # Noise units still exist in merged_sa; filter them out afterwards.
        merged_sa   = sa.merge_units(merge_unit_groups=units_to_merge,
                                     new_unit_ids=merge_canonicals,
                                     merging_mode="hard", n_jobs=n_jobs)
        curated_ids = [uid for uid in merged_sa.sorting.get_unit_ids()
                       if str(uid) not in noise_id_strs]
        sorting_curated = merged_sa.sorting.select_units(curated_ids)

        # For each merged canonical, pick "SUA" if any group member was SUA.
        canonical_labels = {
            str(canon): ("SUA" if any(shank_labels.get(str(uid)) == "SUA" for uid in group)
                         else "MUA")
            for group, canon in zip(units_to_merge, merge_canonicals)
        }
    else:
        sorting_curated = sa.sorting.select_units(good_ids)
        canonical_labels = {}

    # Set unit_label property so the combined analyzer carries SUA/MUA tags.
    unit_labels_list = [
        canonical_labels.get(str(uid), shank_labels.get(str(uid), "MUA"))
        for uid in sorting_curated.get_unit_ids()
    ]
    sorting_curated.set_property("unit_label", unit_labels_list)

    # Ensure is_merged exists on every shank's sorting so aggregate_units can
    # concatenate it.  merge_units() sets it only on shanks that had merges.
    if sorting_curated.get_property("is_merged") is None:
        sorting_curated.set_property(
            "is_merged", np.zeros(sorting_curated.get_num_units(), dtype=bool)
        )

    print(f"  shank{ish}: {len(unit_labels_list)} curated units "
          f"({unit_labels_list.count('SUA')} SUA, {unit_labels_list.count('MUA')} MUA).")

    # Rename channels to avoid ID collisions when aggregating across shanks.
    new_ch_ids = [f"sh{ish}_{ch}" for ch in rec_filt.get_channel_ids()]
    rec_filt   = rec_filt.rename_channels(new_ch_ids)

    return rec_filt, sorting_curated


# ── Main ───────────────────────────────────────────────────────────────────────
def _resolve_device_type(animal_id: str, device_type: str = None):
    """Return device_type, falling back to rec2nwb/device_types.json by animal_id.

    Returns None (rather than raising) when it can't be resolved: it is only
    needed for the .rec fallback, so shanks that have an NWB don't require it.
    """
    if device_type is not None:
        return device_type
    dt_map_path = Path(__file__).resolve().parent.parent / "rec2nwb" / "device_types.json"
    if dt_map_path.exists():
        with open(dt_map_path) as f:
            dt_map = json.load(f)
        device_type = dt_map.get(animal_id)
        if device_type is not None:
            print(f"  device_type for {animal_id!r} resolved from "
                  f"device_types.json -> {device_type}")
    return device_type


def _estimate_int16_gain(recording, n_chunks: int = 20, chunk_sec: float = 1.0,
                         headroom: float = 2.0) -> float:
    """
    Pick ``gain_to_uV`` (µV per int16 step) for the temporary binary cache.

    The bandpassed recording is float32 µV; storing it as int16 halves the temp
    I/O but needs a scale factor that (a) never clips and (b) preserves amplitude
    precision. Sample ``n_chunks`` evenly-spaced 1 s windows to find the peak
    |µV|, then leave ``headroom``x margin for unsampled transients. With the
    returned gain, full-scale int16 (±32767) maps to ±(peak·headroom) µV.
    """
    fs     = recording.get_sampling_frequency()
    n      = max(1, int(chunk_sec * fs))
    total  = recording.get_num_samples()
    starts = ([0] if total <= n
              else np.linspace(0, total - n, n_chunks, dtype=np.int64))
    peak = 0.0
    for s in starts:
        tr = recording.get_traces(start_frame=int(s), end_frame=int(s) + n)
        peak = max(peak, float(np.abs(tr).max()))
    peak = peak or 1.0
    return (peak * headroom) / 32767.0


def _safe_rmtree(path, tries=20, delay=1.5):
    """Remove a folder, retrying through the brief window where Windows still
    holds a just-released memmap file open (PermissionError / WinError 32)."""
    for _ in range(tries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(delay)
    if path.exists():
        print(f"  [warn] temp recording still locked, not deleted: {path}")


# Extensions every curated analyzer should end up with. Also used to detect an
# incomplete/partial build (folder exists but some extensions never computed).
# Keep in sync with _compute_extension_set().
CURATED_EXTENSIONS = (
    "random_spikes", "waveforms", "templates", "noise_levels",
    "spike_amplitudes", "template_similarity", "unit_locations",
)


def _compute_extension_set(analyzer, n_jobs):
    """Compute the curated analyzer's extensions (quality_metrics handled elsewhere)."""
    print("\nComputing extensions...")
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"],
                     n_jobs=n_jobs)
    analyzer.compute("spike_amplitudes", n_jobs=n_jobs)
    analyzer.compute("template_similarity")
    analyzer.compute("unit_locations")


def _compute_curated_extensions(analyzer, combined_recording, *, n_jobs, chunk_duration,
                                compute_chunk_duration, cache_dtype, use_cache, temp_dir, tag):
    """
    Compute extensions either via a temporary on-disk cache of the recording or
    directly on the lazy recording.

    use_cache=True : write the combined recording to a local temp binary once,
        then compute on it. Pays a full-recording write up front to make the
        passes that scan the whole recording (waveforms, spike_amplitudes) read
        from fast local disk instead of re-reading + re-filtering the source NWB.
        ``cache_dtype="int16"`` halves that temp I/O (see _estimate_int16_gain).
    use_cache=False: skip the write and compute straight off the lazy recording,
        re-reading + bandpass-filtering the source NWB on each scanning pass.
        Wins when the source is fast (local NWB) and the write would cost more
        than the extra read passes.

    The extension passes use ``compute_chunk_duration`` (smaller) rather than the
    write's ``chunk_duration``: spike_amplitudes builds a float64 (chunk x
    n_channels) buffer per worker, so a 30 s chunk over 256 channels is ~1.7 GiB
    each and exhausts RAM when many workers / parallel sessions run at once.
    """
    # Smaller chunks for compute (RAM); save() below keeps its own large chunk.
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration=compute_chunk_duration,
                             progress_bar=True)
    if not use_cache:
        print("\nComputing extensions directly on the lazy recording (no temp cache)...")
        _compute_extension_set(analyzer, n_jobs)
        return

    # Write the temp binary on a local fast disk (system temp on C: by default),
    # NOT next to output_folder on the network share. Named by ``tag`` to avoid
    # collisions between concurrent sessions.
    tmp_base = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
    tmp_base.mkdir(parents=True, exist_ok=True)
    binary_recording_folder = tmp_base / f"{tag}_curated_binary_tmp"
    cached_recording = None
    cached_store = None
    try:
        _safe_rmtree(binary_recording_folder)
        if cache_dtype == "int16":
            gain = _estimate_int16_gain(combined_recording)
            print(f"\nWriting temporary binary recording (int16, "
                  f"gain_to_uV={gain:.4g}) -> {binary_recording_folder}")
            # µV -> int16 steps on write; read back as a float32 µV *view*. The
            # analyzer was built on a float32 recording and set_temporary_recording
            # enforces a matching dtype, so the temp recording must also present
            # float32. The store stays int16 on disk (read I/O halved); the upscale
            # to float32 µV happens per-chunk in RAM.
            to_write = sp.scale(combined_recording, gain=1.0 / gain, dtype="float32")
            cached_store = to_write.save(
                format="binary", folder=binary_recording_folder, dtype="int16",
                n_jobs=n_jobs, chunk_duration=chunk_duration, progress_bar=True,
            )
            cached_recording = sp.scale(cached_store, gain=gain, dtype="float32")
        else:
            print(f"\nWriting temporary binary recording -> {binary_recording_folder}")
            cached_recording = combined_recording.save(
                format="binary", folder=binary_recording_folder,
                n_jobs=n_jobs, chunk_duration=chunk_duration, progress_bar=True,
            )
            cached_store = cached_recording
        analyzer.set_temporary_recording(cached_recording)
        _compute_extension_set(analyzer, n_jobs)
    finally:
        analyzer.set_temporary_recording(combined_recording)
        # Drop refs so the int16 memmap is released before we try to delete it.
        cached_recording = None
        cached_store = None
        if binary_recording_folder.exists():
            print(f"\nDeleting temporary binary recording -> {binary_recording_folder}")
            _safe_rmtree(binary_recording_folder)


def main(rec_folder, shanks, sortout_folder, animal_id="", overwrite=False, n_jobs=8,
         compute_extensions=True, z_offsets=None, output_folder=None,
         device_type=None, impedance_path=None, temp_dir=None, chunk_duration="30s",
         cache_dtype="float32", use_cache=True, compute_chunk_duration="10s"):
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
    z_offsets          : dict mapping shank index -> z coordinate (µm).
                         Use when two probes share the same x/y geometry but are
                         implanted at different depths, e.g.
                         {0:0, 1:0, 2:0, 3:0, 4:500, 5:500, 6:500, 7:500}.
                         Shanks with the same offset are combined into one
                         physical Probe in the output ProbeGroup. Shanks not
                         listed get z=0.
    output_folder      : override the default output path (session/curated_analyzer).
                         Useful for testing without touching real data.
    device_type        : probe mapping CSV stem under rec2nwb/mapping/ (e.g.
                         "8shank32"). Only used for shanks that have no NWB and
                         fall back to building the recording from .rec. If
                         omitted, it is looked up from rec2nwb/device_types.json
                         by animal_id.
    impedance_path     : optional impedance CSV, only honoured by the .rec
                         fallback.
    temp_dir           : where to write the temporary binary recording used to
                         speed up extension computation. Defaults to the system
                         temp dir (local C: drive), which is much faster than
                         the network sortout share. The folder is deleted after
                         the analyzer is built. Needs free space ~= the combined
                         recording size (n_channels x duration x 4 bytes).
    chunk_duration     : chunk size for the parallel binary write and for
                         extension computation (e.g. "1s", "10s"). Larger chunks
                         mean fewer HDF5 reads / less filter-margin overhead and
                         usually faster throughput, at the cost of more RAM
                         (~chunk_seconds x n_channels x 4 bytes x n_jobs).
    cache_dtype        : dtype of the temporary binary recording. "float32"
                         (default) writes it verbatim; "int16" halves the temp
                         I/O (the dominant cost for long sessions) by storing a
                         scaled int16 copy with a per-session ``gain_to_uV`` so
                         waveforms/amplitudes are still read back in µV. The
                         gain is chosen from a sampled peak with 2x headroom, so
                         clipping is not expected for bandpassed (artifact-
                         repaired) data; precision is ~peak/16000 µV per step.
                         Ignored when use_cache=False.
    use_cache          : if True (default) write the recording to a temp binary
                         once and compute extensions on it; if False, compute
                         directly on the lazy recording (re-reads + re-filters
                         the source NWB on each scanning pass — no temp write).
    compute_chunk_duration : chunk size for the extension-compute passes, kept
                         smaller than ``chunk_duration`` to bound RAM.
                         spike_amplitudes builds a float64 (chunk x n_channels)
                         buffer per worker, so peak RAM ≈ chunk_seconds x
                         n_channels x 8 bytes x n_jobs x (concurrent sessions).
                         Lower it if you still hit MemoryError with many workers.
    """
    animal_id_parsed, session_id, _ = parse_session_info(str(rec_folder))
    if not animal_id:
        animal_id = animal_id_parsed

    session_folder = Path(sortout_folder) / animal_id / f"{animal_id}_{session_id}"
    if not session_folder.exists():
        # Some sessions' sortout folders are named date-only (no _HHMMSS time
        # suffix) while the .rec name carries the time. Fall back to that form.
        date_only = session_id.split("_")[0]
        alt = Path(sortout_folder) / animal_id / f"{animal_id}_{date_only}"
        if alt.exists():
            print(f"Session folder {session_folder.name!r} not found; using "
                  f"date-only folder {alt.name!r}.")
            session_folder = alt
    if output_folder is None:
        output_folder = session_folder / "curated_analyzer"
    else:
        output_folder = Path(output_folder)

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

    # ── Prep for the .rec fallback (only used by shanks lacking an NWB) ────────
    device_type = _resolve_device_type(animal_id, device_type)
    impedance_path = Path(impedance_path) if impedance_path else None
    bad_ch_ids = load_bad_ch(Path(rec_folder) / "bad_channels.txt")

    # ── Process each shank ─────────────────────────────────────────────────────
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration=chunk_duration, progress_bar=True)
    print(f"\nProcessing {len(shanks)} shank(s): {shanks}")
    recordings, sortings, processed_shanks = [], [], []
    for ish in shanks:
        result = process_shank(ish, session_folder, Path(rec_folder),
                               labels_json, merge_map_json, n_jobs=n_jobs,
                               device_type=device_type, bad_ch_ids=bad_ch_ids,
                               impedance_path=impedance_path)
        if result is None:
            continue
        rec_sh, sort_sh = result
        recordings.append(rec_sh)
        sortings.append(sort_sh)
        processed_shanks.append(ish)

    if not recordings:
        print("No shanks produced curated data — aborting.")
        return

    # ── Separate probes in x before aggregating ───────────────────────────────
    # Two probes that share x/y geometry (grouped by z_offsets, e.g. the second
    # 4-shank probe replicating the first) collide in aggregate_channels()
    # ("Locations are not unique!") and overlap in set_probegroup(). Shift each
    # probe band apart in x by more than the full array width so both checks
    # pass. This is the same offset the ProbeGroup block below uses; the only
    # change is applying it to the recordings here, before aggregation. A pure
    # translation preserves each shank's geometry — extraction is shank-local
    # via the sparsity mask, so the absolute x offset does not affect waveforms,
    # templates, quality metrics, or unit_locations.
    def _z_of(ish):
        if not z_offsets:
            return 0.0
        return float(z_offsets.get(ish, z_offsets.get(str(ish), 0.0)))

    z_order = sorted({_z_of(ish) for ish in processed_shanks})
    if z_offsets and len(z_order) > 1:
        band_of = {z: idx for idx, z in enumerate(z_order)}
        all_x   = np.concatenate([r.get_channel_locations()[:, 0] for r in recordings])
        x_sep   = float(np.ptp(all_x)) + 500.0   # full array width + 500 µm gap
        for i, ish in enumerate(processed_shanks):
            dx = band_of[_z_of(ish)] * x_sep
            if dx:
                probe = recordings[i].get_probe()
                probe.move([dx, 0.0])
                recordings[i] = recordings[i].set_probe(probe, in_place=True)
        print(f"  Separated {len(z_order)} probe band(s) in x by {x_sep:.0f} µm.")

    # ── Tag each shank with a group index before aggregating ──────────────────
    # aggregate_units/aggregate_channels only propagate properties that already
    # exist on the individual objects — they never create a "group" themselves.
    # Setting it here ensures the combined objects carry it for sparsity and
    # for downstream use (e.g. grouping in plots). Done AFTER the probe shift
    # above, because set_probe() resets the channel "group" property.
    for group_idx, (rec, sort) in enumerate(zip(recordings, sortings)):
        rec.set_property("group",  np.full(rec.get_num_channels(),  group_idx, dtype=np.int64))
        sort.set_property("group", np.full(sort.get_num_units(),    group_idx, dtype=np.int64))

    # ── Aggregate across shanks ────────────────────────────────────────────────
    combined_recording = aggregate_channels(recordings)
    combined_sorting   = aggregate_units(sortings)
    total_units = combined_sorting.get_num_units()
    print(f"\nCombined: {combined_recording.get_num_channels()} channels, "
          f"{total_units} units across {len(recordings)} shank(s).")

    # Build shank-local sparsity: unit i is active only on channels that share
    # its group index.  compute_sparsity(by_property) requires a SortingAnalyzer
    # which doesn't exist yet, so we build the boolean mask directly.
    unit_groups    = combined_sorting.get_property("group")
    channel_groups = combined_recording.get_property("group")
    mask = unit_groups[:, np.newaxis] == channel_groups[np.newaxis, :]
    sparsity = ChannelSparsity(
        mask, combined_sorting.get_unit_ids(), combined_recording.get_channel_ids()
    )

    # ── Attach ProbeGroup for multi-probe setups ─────────────────────────────
    # create_sorting_analyzer → get_probegroup() can fall back to
    # create_dummy_probe_from_locations(), which loses physical probe grouping.
    # Build an explicit ProbeGroup (one Probe per z-offset value) from the
    # per-shank locations as they now stand — the x-separation that guarantees
    # unique locations was already applied to each recording above, so here we
    # use rec.get_channel_locations() directly with no further shifting.
    if z_offsets:
        from probeinterface import Probe, ProbeGroup
        ch_start = 0
        z_to_data: dict = {}
        for ish, rec in zip(processed_shanks, recordings):
            z = _z_of(ish)
            n = rec.get_num_channels()
            if z not in z_to_data:
                z_to_data[z] = {"locs": [], "indices": []}
            z_to_data[z]["locs"].append(rec.get_channel_locations()[:, :2])
            z_to_data[z]["indices"].append(np.arange(ch_start, ch_start + n))
            ch_start += n

        probegroup = ProbeGroup()
        for z_val in sorted(z_to_data.keys()):
            d    = z_to_data[z_val]
            locs = np.vstack(d["locs"])
            probe = Probe(ndim=2, si_units="um")
            probe.set_contacts(locs, shapes="circle", shape_params={"radius": 5})
            probe.set_device_channel_indices(np.concatenate(d["indices"]))
            probegroup.add_probe(probe)

        combined_recording.set_probegroup(probegroup, group_mode="by_probe", in_place=True)

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
        _compute_curated_extensions(
            analyzer, combined_recording,
            n_jobs=n_jobs, chunk_duration=chunk_duration,
            compute_chunk_duration=compute_chunk_duration,
            cache_dtype=cache_dtype, use_cache=use_cache, temp_dir=temp_dir,
            tag=f"{animal_id}_{session_id}",
        )
    else:
        print("\nSkipping extension computation (compute_extensions=False).")

    print(f"\nDone. Curated analyzer saved -> {output_folder}")
    print(f"  {total_units} units total.")
    return analyzer


# ── Batch processing via JSON config ──────────────────────────────────────────
def process_from_json(json_file="curated_analyzer_files.json", overwrite=None):
    """
    JSON format:
    {
        "sortout": "/path/to/sortout",
        "n_jobs": 8,
        "overwrite": false,
        "recordings": [
            {"path": "/path/to/rec_folder", "shanks": [0, 1], "animal_id": "CnL42SG",
             "z_offsets": {"0": 0, "1": 0, "4": 500, "5": 500},
             "device_type": "8shank32",   # optional; only used for the .rec
                                          # fallback when a shank has no NWB.
                                          # Auto-looked up from device_types.json
                                          # by animal_id if omitted.
             "impedance_path": null}      # optional; .rec fallback only
        ]
    }
    overwrite kwarg takes precedence over the value in the JSON file.
    """
    json_path = Path(__file__).parent / json_file
    with open(json_path) as f:
        config = json.load(f)

    sortout = config.get("sortout")
    if sortout is None:
        raise ValueError("'sortout' key is required in the JSON config.")

    n_jobs_default = config.get("n_jobs", 8)
    overwrite_cfg  = overwrite if overwrite is not None else config.get("overwrite", False)

    for rec in config["recordings"]:
        rec_path  = Path(rec["path"])
        shanks_   = rec.get("shanks", [0])
        animal_id = rec.get("animal_id", "")
        n_jobs_   = rec.get("n_jobs", n_jobs_default)
        z_offsets = {int(k): v for k, v in rec.get("z_offsets", {}).items()}
        device_type    = rec.get("device_type")
        impedance_path = rec.get("impedance_path")

        if not rec_path.exists():
            print(f"Recording folder not found: {rec_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {rec_path.name}  shanks={shanks_}")
        print(f"{'='*60}")
        main(rec_path, shanks_, sortout,
             animal_id=animal_id, overwrite=overwrite_cfg, n_jobs=n_jobs_,
             z_offsets=z_offsets or None,
             device_type=device_type, impedance_path=impedance_path)


if __name__ == "__main__":
    import sys
    _ow = "--overwrite" in sys.argv
    _cfg_path = Path(__file__).parent / "curated_analyzer_files.json"
    if _cfg_path.exists():
        process_from_json(overwrite=_ow or None)
    else:
        # No JSON config — fall back to the top-level variables above
        main(rec_folder, shanks, sortout_folder, overwrite=_ow or overwrite, n_jobs=n_jobs,
             z_offsets=z_offsets or None, use_cache=USE_CACHE)
