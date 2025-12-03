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
from rec2nwb.preproc_func import rm_artifacts, parse_session_info
from spikesorting.ss_proc_func import get_sortout_folder


def main(rec_folder, sorter_params=None, shanks=[0], animal_id=""):
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
    
    # Define recording folder and parse session info
    rec_folder = Path(rec_folder)
    _, session_id, folder_name = parse_session_info(str(rec_folder))

    sortout = get_sortout_folder()
    
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
        print("Recording:", rec)
        
        # === DIAGNOSTIC: Check data range ===
        print("\n=== Data Quality Check ===")
        traces_sample = rec.get_traces(start_frame=0, end_frame=int(rec.get_sampling_frequency() * 1))
        print(f"Raw data range: {traces_sample.min():.2e} to {traces_sample.max():.2e}")
        print(f"Raw data std: {np.std(traces_sample):.2e}")
        
        # FIX: If data is too small (e.g., < 1e-6), scale it up
        if np.abs(traces_sample).max() < 1e-6:
            print("WARNING: Data appears to be in wrong units (too small). Rescaling...")
            rec = sp.scale(rec, gain=1e6)  # Scale up by 1 million
            traces_sample = rec.get_traces(start_frame=0, end_frame=int(rec.get_sampling_frequency() * 1))
            print(f"Rescaled data range: {traces_sample.min():.2e} to {traces_sample.max():.2e}")

        # === PREPROCESSING PIPELINE ===
        print("\n=== Preprocessing Pipeline ===")
        
        # 1. Common Average Reference FIRST
        print("1. Applying common average reference...")
        rec_car = sp.common_reference(rec, reference="global", operator="median")
        
        # 2. Bandpass filter for spikes (300-6000 Hz)
        print("2. Applying bandpass filter (300-6000 Hz)...")
        rec_filt = sp.bandpass_filter(rec_car, freq_min=300, freq_max=6000, dtype=np.float32)
        
        # Check filtered data
        traces_filt = rec_filt.get_traces(start_frame=0, end_frame=int(rec_filt.get_sampling_frequency() * 1))
        print(f"Filtered data range: {traces_filt.min():.2f} to {traces_filt.max():.2f}")
        print(f"Filtered data std: {np.std(traces_filt):.2f}")
        
        # 3. Remove artifacts
        print("3. Removing artifacts...")
        chunk_time = 0.02
        artifacts_thres = 6.0
        rec_rm_artifacts = rm_artifacts(
            rec_filt, rec_folder, shank,
            chunk_time=chunk_time, threshold=artifacts_thres,
            overwrite=False)
        
        # 4. Whitening (optional but recommended)
        print("4. Applying whitening...")
        recording_preprocessed: si.BaseRecording = sp.whiten(rec_rm_artifacts)
        
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

        # === VISUALIZATION ===
        print("\n=== Saving data snippet visualization ===")
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        fs = recording_preprocessed.get_sampling_frequency()
        snippet_duration = 1.0  # seconds
        snippet_frames = int(snippet_duration * fs)

        traces_snippet = recording_preprocessed.get_traces(start_frame=0, end_frame=snippet_frames)
        time_vec = np.arange(snippet_frames) / fs

        for i in range(min(4, traces_snippet.shape[1])):
            axes[i].plot(time_vec, traces_snippet[:, i], 'k-', linewidth=0.5)
            
            # Mark threshold
            noise = np.median(np.abs(traces_snippet[:, i])) / 0.6745
            thresh = threshold_val * noise
            axes[i].axhline(thresh, color='r', linestyle='--', alpha=0.5, label=f'thresh={thresh:.1f}')
            axes[i].axhline(-thresh, color='r', linestyle='--', alpha=0.5)
            
            axes[i].set_ylabel(f'Ch {i}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(f'{folder_name} - Shank {shank} - Preprocessed Data')
        plt.tight_layout()
        plt.savefig(out_folder / 'data_snippet.png', dpi=150)
        plt.close()
        print(f"Data snippet saved to: {out_folder / 'data_snippet.png'}")

        # === SPIKE SORTING ===
        timer = Timer("ms5")
        print("\n=== Starting MountainSort5 ===")
        
        scheme = str(sorter_params.get('scheme', '1'))
        
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

        # Create sorting results folder
        current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        results_folder_name = f"sorting_results_{current_time}_scheme{scheme}"
        sort_out_folder = out_folder / results_folder_name
        sort_out_folder.mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        params_to_save = {
            'scheme': scheme,
            'sorter_params': sorter_params,
            'sorting_params': sorting_params.__dict__,
            'preprocessing': {
                'car': True,
                'bandpass': {'freq_min': 300, 'freq_max': 6000},
                'artifact_removal': {'chunk_time': chunk_time, 'threshold': artifacts_thres},
                'whitening': True
            }
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


def process_from_json(json_file="sorting_files.json"):
    """Read JSON configuration and process recordings."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    json_path = script_dir / json_file

    # Read JSON file
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Get sorter parameters (can be global or per-recording)
    global_sorter_params = config.get('sorter_params', None)

    # Process each recording
    for i, rec in enumerate(config['recordings'], 1):
        rec_folder = Path(rec['path'])
        shanks = rec['shanks']
        animal_id = rec.get('animal_id', '')
        
        # Use recording-specific params if available, otherwise use global
        sorter_params = rec.get('sorter_params', global_sorter_params)

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
                 animal_id=animal_id)
        except Exception as e:
            print(f"ERROR processing {rec_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    process_from_json()