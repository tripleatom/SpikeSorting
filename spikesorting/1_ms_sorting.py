import os
import time
import json
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
import mountainsort5 as ms5
from Timer import Timer
from rec2nwb.preproc_func import parse_session_info

from spikesorting.artifact_utils import (detect_artifacts_recording,
                                          LazyArtifactRepairRecording)


def _load_or_compute_artifacts(rec, out_folder, sorter_params):
    """
    Return (rec_repaired, artifact_timestamps).

    On first run: detects artifacts (reads recording once), saves tiny per-channel
    timestamp arrays to out_folder/_artifact_cache/ (a few KB total), then wraps
    the recording in a LazyArtifactRepairRecording — no memmap, no disk write.
    On subsequent runs: loads cached timestamps instantly and skips detection.
    Cache is invalidated if detection params or recording shape changes.
    """
    cache_dir = Path(out_folder) / '_artifact_cache'
    meta_path = cache_dir / 'cache_meta.json'
    ts_path   = cache_dir / 'artifact_timestamps.npz'

    current_meta = {
        'detection_method': 'rolling_std',
        'rolling_window_size': 100,
        'rolling_z_threshold': 30,
        'time_batch_sec': sorter_params.get('artifact_time_batch_sec', 600),
        'n_samples': int(rec.get_num_frames()),
        'n_channels': int(rec.get_num_channels()),
        'sampling_rate': float(rec.get_sampling_frequency()),
    }

    if meta_path.exists() and ts_path.exists():
        with open(meta_path) as f:
            cached_meta = json.load(f)
        if cached_meta == current_meta:
            print("1. Cache hit — loading artifact timestamps, skipping detection...")
            n_channels = current_meta['n_channels']
            ts_data = np.load(str(ts_path), allow_pickle=True)
            artifact_timestamps = [ts_data[f'ch_{i:03d}'] for i in range(n_channels)]
        else:
            print("1. Cache params mismatch — rerunning artifact detection...")
            artifact_timestamps = None
    else:
        print("1. No cache — running artifact detection and saving timestamps...")
        artifact_timestamps = None

    if artifact_timestamps is None:
        artifact_timestamps = detect_artifacts_recording(
            rec,
            detection_method=current_meta['detection_method'],
            rolling_window_size=current_meta['rolling_window_size'],
            rolling_z_threshold=current_meta['rolling_z_threshold'],
            time_batch_sec=current_meta['time_batch_sec'],
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(current_meta, f, indent=2)
        ts_save = {f'ch_{i:03d}': ts for i, ts in enumerate(artifact_timestamps)}
        np.savez(str(ts_path), **ts_save)

    rec_repaired = LazyArtifactRepairRecording(rec, artifact_timestamps, dither=True)
    return rec_repaired, artifact_timestamps, current_meta


def _process_single_recording(rec, out_folder, sorter_params, folder_name, shank,
                               rec_folder=None, remove_artifacts=True, n_jobs=1):
    """
    Process a single recording through the full pipeline.

    Parameters
    ----------
    rec : BaseRecording
        The recording to process
    out_folder : Path
        Output folder for results
    sorter_params : dict
        Sorting parameters
    folder_name : str
        Name for labeling plots
    shank : int
        Shank identifier
    rec_folder : Path, optional
        Recording folder (needed for artifact removal)
    remove_artifacts : bool
        Whether to run artifact detection and repair (default True)
    """
    print("Recording:", rec)

    # === DIAGNOSTIC: Check data range ===
    print("\n=== Data Quality Check ===")
    traces_sample = rec.get_traces(start_frame=0, end_frame=int(rec.get_sampling_frequency() * 1))
    print(f"Raw data range: {traces_sample.min():.2e} to {traces_sample.max():.2e}")
    print(f"Raw data std: {np.std(traces_sample):.2e}")

    # FIX: If data is too small (e.g., < 1e-6), scale it up
    if np.abs(traces_sample).max() < 1e-6:
        print("WARNING: Data appears to be in wrong units (too small). Rescaling...")
        rec = sp.scale(rec, gain=1e6)
        traces_sample = rec.get_traces(start_frame=0, end_frame=int(rec.get_sampling_frequency() * 1))
        print(f"Rescaled data range: {traces_sample.min():.2e} to {traces_sample.max():.2e}")

    # === PREPROCESSING PIPELINE ===
    print("\n=== Preprocessing Pipeline ===")
    _pipeline_t0 = time.time()

    # 1. Detect and repair artifacts (cached timestamps reused on subsequent runs)
    _t0 = time.time()
    if remove_artifacts:
        rec_rm_artifacts, artifact_timestamps, artifact_meta = _load_or_compute_artifacts(
            rec, out_folder, sorter_params,
        )
    else:
        print("1. Skipping artifact removal (remove_artifacts=False)...")
        rec_rm_artifacts = rec
        artifact_timestamps = None
        artifact_meta = None
    print(f"[TIMING] Step 1 artifact detection/cache: {time.time() - _t0:.2f}s")

    # 2. Common median reference (CMR)
    print("2. Applying common median reference (CMR)...")
    rec_cmr = sp.common_reference(rec_rm_artifacts, reference='global', operator='median')

    # 3. Bandpass filter for spikes (300-6000 Hz)
    print("3. Applying bandpass filter (300-6000 Hz)...")
    rec_filt = sp.bandpass_filter(rec_cmr, freq_min=300, freq_max=6000, dtype=np.float32)

    # 4. Whitening
    print("4. Applying whitening...")
    recording_preprocessed: si.BaseRecording = sp.whiten(rec_filt)

    # Check preprocessed data
    traces_preproc = recording_preprocessed.get_traces(
        start_frame=0, end_frame=int(recording_preprocessed.get_sampling_frequency() * 10))
    print(f"Preprocessed data range: {traces_preproc.min():.2f} to {traces_preproc.max():.2f}")
    print(f"Preprocessed data std: {np.std(traces_preproc):.2f}")

    # Per-channel statistics
    print("\n=== Per-Channel Statistics (first 10 channels) ===")
    for i in range(min(10, traces_preproc.shape[1])):
        ch_std = np.std(traces_preproc[:, i])
        ch_max = np.max(np.abs(traces_preproc[:, i]))
        print(f"Ch {i}: std={ch_std:.2f}, max_abs={ch_max:.2f}")

    # Estimate noise level
    noise_level = np.median(np.abs(traces_preproc)) / 0.6745
    print(f"\nEstimated median noise level: {noise_level:.2f}")
    threshold_val = sorter_params.get('detect_threshold', 5.5)
    print(f"Detection threshold: {threshold_val} (= {threshold_val * noise_level:.2f} absolute units)")

    # Count potential threshold crossings
    threshold_crossings = np.sum(np.abs(traces_preproc) > threshold_val * noise_level)
    total_samples = traces_preproc.size
    print(f"Threshold crossings in sample: {threshold_crossings} / {total_samples} ({100*threshold_crossings/total_samples:.3f}%)")

    # === PREPARE OUTPUT FOLDER ===
    scheme = str(sorter_params.get('scheme', '1'))
    current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    results_folder_name = f"sorting_results_{current_time}_scheme{scheme}"
    sort_out_folder = out_folder / results_folder_name
    sort_out_folder.mkdir(parents=True, exist_ok=True)

    # Save artifact timestamps
    if artifact_timestamps is not None:
        ts_save = {f'ch_{i:03d}': ts for i, ts in enumerate(artifact_timestamps)}
        ts_save['channel_ids'] = np.array(rec.get_channel_ids())
        np.savez(str(sort_out_folder / 'artifact_timestamps.npz'), **ts_save)
        n_flagged = sum(len(ts) for ts in artifact_timestamps)
        print(f"Artifact timestamps saved: {n_flagged} flagged samples across {len(artifact_timestamps)} channels")

    # === VISUALIZATION ===
    n_snippets = 5
    print(f"\n=== Saving {n_snippets} data snippet(s) visualization ===")
    fs = recording_preprocessed.get_sampling_frequency()
    snippet_duration = 1.0  # seconds per snippet
    snippet_frames = int(snippet_duration * fs)
    total_frames = recording_preprocessed.get_num_frames()
    n_channels_to_plot = min(4, recording_preprocessed.get_num_channels())

    # Calculate start frames for each snippet (evenly spaced)
    if n_snippets == 1:
        start_frames = [0]
    else:
        start_frames = np.linspace(0, total_frames - snippet_frames, n_snippets, dtype=int)

    fig, axes = plt.subplots(n_channels_to_plot, n_snippets,
                             figsize=(5 * n_snippets, 3 * n_channels_to_plot),
                             squeeze=False)

    for col, start_frame in enumerate(start_frames):
        traces_snippet = recording_preprocessed.get_traces(
            start_frame=start_frame, end_frame=start_frame + snippet_frames)
        time_vec = np.arange(snippet_frames) / fs + start_frame / fs

        for row in range(n_channels_to_plot):
            ax = axes[row, col]
            ax.plot(time_vec, traces_snippet[:, row], 'k-', linewidth=0.5)

            # Mark threshold
            noise = np.median(np.abs(traces_snippet[:, row])) / 0.6745
            thresh = threshold_val * noise
            ax.axhline(thresh, color='r', linestyle='--', alpha=0.5, label=f'thresh={thresh:.1f}')
            ax.axhline(-thresh, color='r', linestyle='--', alpha=0.5)

            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f'Ch {row}')
                ax.legend()
            if row == n_channels_to_plot - 1:
                ax.set_xlabel('Time (s)')
            if row == 0:
                ax.set_title(f't={start_frame/fs:.1f}s')

    plt.suptitle(f'{folder_name} - Shank {shank} - Preprocessed Data ({n_snippets} snippets)')
    plt.tight_layout()
    plt.savefig(sort_out_folder / 'data_snippet.png', dpi=150)
    plt.close()
    print(f"Data snippet saved to: {sort_out_folder / 'data_snippet.png'}")

    # === SPIKE SORTING ===
    timer = Timer("ms5")
    print(f"[TIMING] Total preprocessing wall time: {time.time() - _pipeline_t0:.2f}s")
    print("\n=== Starting MountainSort5 ===")

    if scheme == '1':
        print("Using Scheme 1 sorting...")
        sorting_params = ms5.Scheme1SortingParameters(
            detect_sign=sorter_params.get('detect_sign', 0),
            detect_time_radius_msec=sorter_params.get('detect_time_radius_msec', 0.5),
            detect_threshold=sorter_params.get('detect_threshold', 5.5),
            snippet_T1=sorter_params.get('snippet_T1', 20),
            snippet_T2=sorter_params.get('snippet_T2', 20),
            npca_per_channel=sorter_params.get('npca_per_channel', 3),
            npca_per_subdivision=sorter_params.get('npca_per_subdivision', 10),
            snippet_mask_radius=sorter_params.get('snippet_mask_radius', 250),
            detect_channel_radius=sorter_params.get('scheme1_detect_channel_radius', 150),
        )
        sorting = ms5.sorting_scheme1(
            recording=recording_preprocessed,
            sorting_parameters=sorting_params)
        timer.report()

    elif scheme == '2':
        print("Using Scheme 2 sorting...")
        sorting_params = ms5.Scheme2SortingParameters(
            detect_sign=sorter_params.get('detect_sign', 0),
            detect_time_radius_msec=sorter_params.get('detect_time_radius_msec', 0.5),
            detect_threshold=sorter_params.get('detect_threshold', 5.5),
            snippet_T1=sorter_params.get('snippet_T1', 20),
            snippet_T2=sorter_params.get('snippet_T2', 20),
            snippet_mask_radius=sorter_params.get('snippet_mask_radius', 250),
            phase1_detect_channel_radius=sorter_params.get('scheme2_phase1_detect_channel_radius', 200),
            detect_channel_radius=sorter_params.get('scheme2_detect_channel_radius', 120),
            max_num_snippets_per_training_batch=sorter_params.get('scheme2_max_num_snippets_per_training_batch', 200),
            training_duration_sec=sorter_params.get('scheme2_training_duration_sec', 300),
            training_recording_sampling_mode=sorter_params.get('scheme2_training_recording_sampling_mode', 'uniform'),
            phase1_detect_threshold=sorter_params.get('scheme2_phase1_detect_threshold', 5.5),
            phase1_npca_per_channel=sorter_params.get('scheme2_phase1_npca_per_channel', 3),
            phase1_npca_per_subdivision=sorter_params.get('scheme2_phase1_npca_per_subdivision', 10),
        )
        sorting = ms5.sorting_scheme2(
            recording=recording_preprocessed,
            sorting_parameters=sorting_params)
        timer.report()
    else:
        raise ValueError(f"Invalid scheme: {scheme}. Must be '1' or '2'")

    if remove_artifacts and hasattr(rec_rm_artifacts, 'print_timing_report'):
        rec_rm_artifacts.print_timing_report()

    # Save parameters
    params_to_save = {
        'scheme': scheme,
        'sorter_params': sorter_params,
        'sorting_params': sorting_params.__dict__,
        'preprocessing': {
            'cmr': {'reference': 'global', 'operator': 'median'},
            'artifact_removal': artifact_meta,
            'bandpass': {'freq_min': 300, 'freq_max': 6000},
            'whitening': True,
        },
    }
    with open(sort_out_folder / "sorting_params.json", "w") as f:
        json.dump(params_to_save, f, indent=2)

    print("\n=== Sorting Results ===")
    print(f"Number of units found: {len(sorting.unit_ids)}")
    print(f"Unit IDs: {sorting.unit_ids}")
    spike_counts = sorting.count_num_spikes_per_unit()
    print("Spike counts per unit:")
    for unit_id in sorting.unit_ids:
        count = spike_counts[unit_id]
        rate = count / recording_preprocessed.get_total_duration()
        print(f"  Unit {unit_id}: {count} spikes ({rate:.2f} Hz)")

    # Register recording and create a sorting analyzer
    sorting.register_recording(recording_preprocessed)

    analyzer_folder = sort_out_folder / "sorting_analyzer"
    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording_preprocessed,
        format="binary_folder",
        folder=str(analyzer_folder)
    )
    print("Sorting analyzer:", sorting_analyzer)

    # Compute metrics
    try:
        print("\n=== Computing Waveforms and Metrics ===")
        si.set_global_job_kwargs(n_jobs=n_jobs, progress_bar=True)
        sorting_analyzer.compute(['random_spikes', 'waveforms', 'noise_levels'], n_jobs=n_jobs)
        sorting_analyzer.compute('templates')
        _ = sorting_analyzer.compute('template_similarity')
        _ = sorting_analyzer.compute('spike_amplitudes', n_jobs=n_jobs)
        _ = sorting_analyzer.compute('correlograms', n_jobs=n_jobs)
        _ = sorting_analyzer.compute('unit_locations')

        out_fig_folder = sort_out_folder / 'raw_units'
        out_fig_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Generating Unit Summary Plots ===")
        for unit_id in sorting.get_unit_ids():
            print(f"  Plotting unit {unit_id}...")
            sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
            plt.savefig(out_fig_folder / f'unit_summary_{unit_id}.png', dpi=150)
            plt.close()

        print(f"Summary plots saved to: {out_fig_folder}")

    except Exception as e:
        print(f"Error during metrics computation: {e}")
        traceback.print_exc()

    print(f"\n=== Shank {shank} Complete ===")
    print(f"Results saved to: {sort_out_folder}")



def main(rec_folder=None, sorter_params=None, shanks=None, animal_id="", sortout=None,
         remove_artifacts=True, n_jobs=1):
    """
    Main spike sorting function.

    Parameters
    ----------
    rec_folder : str or Path
        Path to recording folder
    sorter_params : dict
        Dictionary containing sorting parameters
    shanks : list
        List of shank indices to process
    animal_id : str
        Animal identifier
    sortout : str or Path
        Output folder for sorting results.
    remove_artifacts : bool
        Whether to run artifact detection and repair (default True)
    """
    # Default parameters if none provided
    if shanks is None:
        shanks = [0]

    if sorter_params is None:
        sorter_params = {
            "scheme": "1",
            "detect_threshold": 5.5,
            "detect_sign": 0,
            "detect_time_radius_msec": 0.5,
            "npca_per_channel": 3,
            "npca_per_subdivision": 10
        }

    if sortout is None:
        raise ValueError("sortout must be provided (set 'sortout' in the JSON config)")
    sortout = Path(sortout)

    if rec_folder is None:
        raise ValueError("rec_folder must be provided")

    rec_folder = Path(rec_folder)
    _, session_id, folder_name = parse_session_info(str(rec_folder))

    for shank in shanks:
        # Construct paths for NWB file and output folder
        nwb_folder = rec_folder / f"{folder_name}sh{shank}.nwb"
        if not nwb_folder.exists():
            print(f"NWB file not found: {nwb_folder}")
            continue

        out_folder = Path(sortout) / animal_id / \
            f"{animal_id}_{session_id}" / f"shank{shank}"
        out_folder.mkdir(parents=True, exist_ok=True)

        # Load recording from NWB file
        rec = se.NwbRecordingExtractor(str(nwb_folder))

        # Process this single recording
        _process_single_recording(
            rec=rec,
            out_folder=out_folder,
            sorter_params=sorter_params,
            folder_name=folder_name,
            shank=shank,
            rec_folder=rec_folder,
            remove_artifacts=remove_artifacts,
            n_jobs=n_jobs,
        )


def process_from_json(json_file="sorting_files.json"):
    """Read JSON configuration and process recordings.

    JSON format:
    {
        "recordings": [
            {"path": "/path/to/folder", "shanks": [0, 1], "animal_id": "M001"}
        ],
        "sortout": "/path/to/output",
        "sorter_params": {...}
    }
    """

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    json_path = script_dir / json_file

    # Read JSON file
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Get sorter parameters (can be global or per-recording)
    global_sorter_params = config.get('sorter_params', None)

    # Choose sortout folder once for the whole run (outside shank/recording loops)
    sortout = config.get('sortout')
    if sortout is None:
        raise ValueError("'sortout' key is required in the JSON config")

    global_n_jobs = config.get('n_jobs', 1)

    # Process each recording
    for i, rec in enumerate(config['recordings'], 1):
        animal_id = rec.get('animal_id', '')

        # Use recording-specific params if available, otherwise use global
        sorter_params = rec.get('sorter_params', global_sorter_params)

        rec_folder = Path(rec['path'])
        shanks = rec['shanks']
        remove_artifacts = rec.get('remove_artifacts', True)
        n_jobs = rec.get('n_jobs', global_n_jobs)

        print(f"\n{'='*60}")
        print(f"[{i}/{len(config['recordings'])}] Processing: {rec_folder.name}")
        print(f"  Animal ID: {animal_id}")
        print(f"  Shanks: {shanks}")
        print(f"  Scheme: {sorter_params.get('scheme', '1')}")
        print(f"  Threshold: {sorter_params.get('detect_threshold', 5.5)}")
        print(f"  Remove artifacts: {remove_artifacts}")
        print(f"  n_jobs: {n_jobs}")
        print(f"{'='*60}")

        try:
            main(rec_folder=rec_folder,
                 sorter_params=sorter_params,
                 shanks=shanks,
                 animal_id=animal_id,
                 sortout=sortout,
                 remove_artifacts=remove_artifacts,
                 n_jobs=n_jobs)
        except Exception as e:
            print(f"ERROR processing {rec_folder.name}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    process_from_json()