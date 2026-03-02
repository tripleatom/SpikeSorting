import os
import time
import json
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

from spikesorting.artifact_utils import repair_artifacts_recording


def load_and_concatenate_recordings(nwb_files):
    """
    Load multiple NWB files and concatenate them, tracking sample boundaries.

    Parameters
    ----------
    nwb_files : list
        List of paths to NWB files

    Returns
    -------
    rec : BaseRecording
        Concatenated recording
    boundary_info : list of dict
        List containing boundary information for each recording
    """
    recordings = []
    boundary_info = []
    cumulative_samples = 0

    print("\n=== Loading and Concatenating Recordings ===")
    for i, nwb_file in enumerate(nwb_files):
        nwb_path = Path(nwb_file)
        print(f"\nLoading [{i+1}/{len(nwb_files)}]: {nwb_path.name}")

        rec = se.NwbRecordingExtractor(str(nwb_path))
        n_samples = rec.get_num_samples()
        duration = rec.get_total_duration()
        fs = rec.get_sampling_frequency()

        info = {
            'index': i,
            'file': str(nwb_path),
            'filename': nwb_path.name,
            'sampling_rate': fs,
            'n_samples': n_samples,
            'duration_sec': duration,
            'start_sample': cumulative_samples,
            'end_sample': cumulative_samples + n_samples - 1,
        }
        boundary_info.append(info)

        print(f"  Sampling rate: {fs} Hz")
        print(f"  Samples: {n_samples:,}")
        print(f"  Duration: {duration:.2f} sec")
        print(f"  Sample range: {cumulative_samples:,} - {cumulative_samples + n_samples - 1:,}")

        recordings.append(rec)
        cumulative_samples += n_samples

    # Concatenate all recordings
    if len(recordings) == 1:
        rec_concat = recordings[0]
    else:
        rec_concat = si.concatenate_recordings(recordings)

    print(f"\n=== Concatenation Summary ===")
    print(f"Total recordings: {len(recordings)}")
    print(f"Total samples: {cumulative_samples:,}")
    print(f"Total duration: {cumulative_samples / fs:.2f} sec")

    return rec_concat, boundary_info


def save_boundary_info(boundary_info, output_folder, filename="recording_boundaries.txt"):
    """
    Save recording boundary information to a text file.

    Parameters
    ----------
    boundary_info : list of dict
        Boundary information from load_and_concatenate_recordings
    output_folder : Path
        Folder to save the file
    filename : str
        Output filename
    """
    output_path = Path(output_folder) / filename

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RECORDING BOUNDARIES FOR SPIKE TIME TRACKING\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of recordings: {len(boundary_info)}\n\n")

        for info in boundary_info:
            f.write("-" * 50 + "\n")
            f.write(f"Recording {info['index'] + 1}:\n")
            f.write(f"  File: {info['filename']}\n")
            f.write(f"  Full path: {info['file']}\n")
            f.write(f"  Sampling rate: {info['sampling_rate']} Hz\n")
            f.write(f"  Number of samples: {info['n_samples']:,}\n")
            f.write(f"  Duration: {info['duration_sec']:.2f} sec\n")
            f.write(f"  Start sample (inclusive): {info['start_sample']:,}\n")
            f.write(f"  End sample (inclusive): {info['end_sample']:,}\n")
            f.write("\n")

        # Summary table for quick reference
        f.write("=" * 70 + "\n")
        f.write("QUICK REFERENCE TABLE\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Rec#':<6}{'Start Sample':<18}{'End Sample':<18}{'Filename'}\n")
        f.write("-" * 70 + "\n")
        for info in boundary_info:
            f.write(f"{info['index']+1:<6}{info['start_sample']:<18,}{info['end_sample']:<18,}{info['filename']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("HOW TO USE:\n")
        f.write("  To find which recording a spike belongs to:\n")
        f.write("  - Get the spike's sample index from sorting results\n")
        f.write("  - Find which recording's [start_sample, end_sample] range contains it\n")
        f.write("  - Subtract start_sample to get the original sample index in that recording\n")
        f.write("=" * 70 + "\n")

    print(f"Boundary info saved to: {output_path}")
    return output_path


def _process_single_recording(rec, out_folder, sorter_params, folder_name, shank,
                               rec_folder=None, boundary_info=None):
    """
    Process a single recording (or concatenated recording) through the full pipeline.

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
    shank : int or str
        Shank identifier (or "concat" for concatenated)
    rec_folder : Path, optional
        Recording folder (needed for artifact removal)
    boundary_info : list of dict, optional
        Boundary information for concatenated recordings
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

    # 1. Detect and repair artifacts
    print("1. Detecting and repairing artifacts (rolling_std, window=50, z=30)...")
    rec_rm_artifacts, _artifact_cleanup, artifact_timestamps = repair_artifacts_recording(
        rec,
        detection_method='rolling_std',
        rolling_window_size=50,
        rolling_z_threshold=30,
        n_jobs=-1,
        time_batch_sec=sorter_params.get('artifact_time_batch_sec', None),
    )

    # 3. Bandpass filter for spikes (300-6000 Hz)
    print("3. Applying bandpass filter (300-6000 Hz)...")
    rec_filt = sp.bandpass_filter(rec_rm_artifacts, freq_min=300, freq_max=6000, dtype=np.float32)

    # Check filtered data
    traces_filt = rec_filt.get_traces(start_frame=0, end_frame=int(rec_filt.get_sampling_frequency() * 1))
    print(f"Filtered data range: {traces_filt.min():.2f} to {traces_filt.max():.2f}")
    print(f"Filtered data std: {np.std(traces_filt):.2f}")

    # 4. Whitening (optional but recommended)
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
    ts_save = {f'ch_{i:03d}': ts for i, ts in enumerate(artifact_timestamps)}
    ts_save['channel_ids'] = np.array(rec.get_channel_ids())
    np.savez(str(sort_out_folder / 'artifact_timestamps.npz'), **ts_save)
    n_flagged = sum(len(ts) for ts in artifact_timestamps)
    print(f"Artifact timestamps saved: {n_flagged} flagged samples across {len(artifact_timestamps)} channels")

    # Save boundary info if this is a concatenated recording
    if boundary_info is not None:
        save_boundary_info(boundary_info, sort_out_folder)

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

    # Save parameters
    params_to_save = {
        'scheme': scheme,
        'sorter_params': sorter_params,
        'sorting_params': sorting_params.__dict__,
        'preprocessing': {
            'car': False,
            'artifact_removal': {
                'method': 'rolling_std',
                'rolling_window_size': 50,
                'rolling_z_threshold': 30,
            },
            'bandpass': {'freq_min': 300, 'freq_max': 6000},
            'whitening': True,
        },
        'concatenated': boundary_info is not None,
        'source_files': [info['file'] for info in boundary_info] if boundary_info else None
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
        sorting_analyzer.compute(['random_spikes', 'waveforms', 'noise_levels'])
        sorting_analyzer.compute('templates')
        _ = sorting_analyzer.compute('template_similarity')
        _ = sorting_analyzer.compute('spike_amplitudes')
        _ = sorting_analyzer.compute('correlograms')
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
        import traceback
        traceback.print_exc()

    print(f"\n=== Shank {shank} Complete ===")
    print(f"Results saved to: {sort_out_folder}")

    # Release all objects that hold the artifact memmap open before deleting it.
    # Chain: sorting_analyzer → sorting → recording_preprocessed → rec_filt → rec_rm_artifacts → memmap
    import gc
    del sorting_analyzer, sorting, recording_preprocessed, rec_filt, rec_rm_artifacts
    gc.collect()
    _artifact_cleanup()


def main(rec_folder=None, nwb_files=None, sorter_params=None, shanks=[0], animal_id="", shank_id=None, sortout=None):
    """
    Main spike sorting function.

    Parameters
    ----------
    rec_folder : str or Path, optional
        Path to recording folder (for single recording mode)
    nwb_files : list, optional
        List of NWB file paths to concatenate and sort together
        If provided, rec_folder and shanks are ignored
    sorter_params : dict
        Dictionary containing sorting parameters
    shanks : list
        List of shank indices to process (only used with rec_folder)
    animal_id : str
        Animal identifier
    shank_id : int, optional
        Shank identifier for concatenated mode folder naming
    sortout : str or Path, optional
        Output folder for sorting results. If None, user is prompted once per main() call.
    """
    # Default parameters if none provided
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

    # Mode 1: Concatenate multiple NWB files
    if nwb_files is not None and len(nwb_files) > 0:
        print("\n" + "=" * 60)
        print("CONCATENATED RECORDING MODE")
        print("=" * 60)

        # Load and concatenate recordings
        rec, boundary_info = load_and_concatenate_recordings(nwb_files)

        # Create output folder for concatenated sorting
        # Structure: sortout/animal_id/rec1_rec2_concat_sh{N}/shank{N}/
        # Extract folder names from nwb file paths (parent.parent is the .rec folder)
        rec_folder_names = []
        for nwb_file in nwb_files:
            nwb_path = Path(nwb_file)
            # Parent is the .rec folder, get its name without .rec extension
            rec_folder_name = nwb_path.parent.stem  # .stem removes .rec extension
            rec_folder_names.append(rec_folder_name)

        concat_name = "_".join(rec_folder_names) + "_concat"
        shank_folder = f"shank{shank_id}" if shank_id is not None else "shank_concat"
        out_folder = Path(sortout) / animal_id / concat_name / shank_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        # Process this single concatenated recording
        _process_single_recording(
            rec=rec,
            out_folder=out_folder,
            sorter_params=sorter_params,
            boundary_info=boundary_info,
            folder_name=f"{animal_id}_concatenated",
            shank=shank_id if shank_id is not None else "concat"
        )
        return

    # Mode 2: Original single folder mode
    if rec_folder is None:
        raise ValueError("Either rec_folder or nwb_files must be provided")

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
            boundary_info=None
        )


def process_from_json(json_file="sorting_files.json"):
    """Read JSON configuration and process recordings.

    JSON format supports three modes:

    1. Single folder mode (original):
    {
        "recordings": [
            {"path": "/path/to/folder", "shanks": [0, 1], "animal_id": "M001"}
        ]
    }

    2. Multiple paths with concatenation option:
    {
        "recordings": [
            {
                "paths": ["/path/to/folder1", "/path/to/folder2"],
                "shanks": [0, 1, 2, 3],
                "animal_id": "M001",
                "concatenate": true  // or false
            }
        ]
    }

    When concatenate=true: for each shank, concatenate NWB files from all paths
    When concatenate=false: process each path separately (same as calling with single path multiple times)
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

    # Process each recording
    for i, rec in enumerate(config['recordings'], 1):
        animal_id = rec.get('animal_id', '')

        # Use recording-specific params if available, otherwise use global
        sorter_params = rec.get('sorter_params', global_sorter_params)

        # Check for multiple paths mode
        paths = rec.get('paths', None)

        if paths is not None:
            # Multiple paths mode
            shanks = rec['shanks']
            concatenate = rec.get('concatenate', False)

            if concatenate:
                # Concatenate mode: for each shank, concatenate NWB files from all paths
                print(f"\n{'='*60}")
                print(f"[{i}/{len(config['recordings'])}] Processing CONCATENATED (multiple paths):")
                print(f"  Animal ID: {animal_id}")
                print(f"  Paths: {len(paths)}")
                for p in paths:
                    print(f"    - {Path(p).name}")
                print(f"  Shanks: {shanks}")
                print(f"  Scheme: {sorter_params.get('scheme', '1')}")
                print(f"  Threshold: {sorter_params.get('detect_threshold', 5.5)}")
                print(f"{'='*60}")

                # Process each shank with concatenated recordings
                for shank in shanks:
                    print(f"\n--- Processing Shank {shank} (concatenated) ---")

                    # Build list of NWB files for this shank from all paths
                    nwb_files = []
                    for p in paths:
                        rec_folder = Path(p)
                        _, _, folder_name = parse_session_info(str(rec_folder))
                        nwb_file = rec_folder / f"{folder_name}sh{shank}.nwb"

                        if not nwb_file.exists():
                            print(f"WARNING: NWB file not found: {nwb_file}")
                            continue
                        nwb_files.append(str(nwb_file))

                    if len(nwb_files) < 2:
                        print(f"ERROR: Need at least 2 NWB files to concatenate, found {len(nwb_files)}")
                        continue

                    try:
                        main(nwb_files=nwb_files,
                             sorter_params=sorter_params,
                             animal_id=animal_id,
                             shank_id=shank,
                             sortout=sortout)
                    except Exception as e:
                        print(f"ERROR processing shank {shank}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                # Non-concatenate mode: process each path separately
                for p in paths:
                    rec_folder = Path(p)

                    print(f"\n{'='*60}")
                    print(f"[{i}/{len(config['recordings'])}] Processing: {rec_folder.name}")
                    print(f"  Animal ID: {animal_id}")
                    print(f"  Shanks: {shanks}")
                    print(f"  Scheme: {sorter_params.get('scheme', '1')}")
                    print(f"  Threshold: {sorter_params.get('detect_threshold', 5.5)}")
                    print(f"{'='*60}")

                    try:
                        main(rec_folder=rec_folder,
                             sorter_params=sorter_params,
                             shanks=shanks,
                             animal_id=animal_id,
                             sortout=sortout)
                    except Exception as e:
                        print(f"ERROR processing {rec_folder.name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        else:
            # Single folder mode (original) - "path" instead of "paths"
            rec_folder = Path(rec['path'])
            shanks = rec['shanks']

            print(f"\n{'='*60}")
            print(f"[{i}/{len(config['recordings'])}] Processing: {rec_folder.name}")
            print(f"  Animal ID: {animal_id}")
            print(f"  Shanks: {shanks}")
            print(f"  Scheme: {sorter_params.get('scheme', '1')}")
            print(f"  Threshold: {sorter_params.get('detect_threshold', 5.5)}")
            print(f"{'='*60}")

            try:
                main(rec_folder=rec_folder,
                     sorter_params=sorter_params,
                     shanks=shanks,
                     animal_id=animal_id,
                     sortout=sortout)
            except Exception as e:
                print(f"ERROR processing {rec_folder.name}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    process_from_json()