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


def main(rec_folder, threshold=5.5, scheme=1, shanks=[0]):
    # Define recording folder and parse session info
    rec_folder = Path(rec_folder)
    animal_id, session_id, folder_name = parse_session_info(str(rec_folder))

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

        # Preprocessing: bandpass filter
        rec_filt = sp.bandpass_filter(
            rec, freq_min=300, freq_max=6000, dtype=np.float32)

        # Remove artifacts using a chunk-based approach
        chunk_time = 0.02
        artifacts_thres = 6.0
        rec_rm_artifacts = rm_artifacts(
            rec_filt, rec_folder, shank,
            chunk_time=chunk_time, threshold=artifacts_thres,
            overwrite=True)

        # Apply common reference and whitening
        rec_cr = sp.common_reference(
            rec_rm_artifacts, reference="global", operator="median")
        recording_preprocessed: si.BaseRecording = sp.whiten(rec_cr)

        # Define sorting parameters
        detect_time_radius_msec = 0.5
        npca_per_channel = 3
        npca_per_subdivision = 10

        timer = Timer("ms5")
        print("Starting ms5 sorting...")

        if scheme == 1:
            print("Using Scheme 1 sorting...")
            sorting_params = ms5.Scheme1SortingParameters(
                detect_sign=0,  # 0 for all, 1 for positive, -1 for negative
                detect_time_radius_msec=detect_time_radius_msec,
                detect_threshold=threshold,
                npca_per_channel=npca_per_channel,
                npca_per_subdivision=npca_per_subdivision
            )
            sorting = ms5.sorting_scheme1(
                recording=recording_preprocessed, sorting_parameters=sorting_params)
            timer.report()
        elif scheme == 2:
            # FIXME: finish setting parameters for scheme 2
            print("Using Scheme 2 sorting...")
            sorting_params = ms5.Scheme2SortingParameters(
                phase1_detect_channel_radius=150,
                detect_channel_radius=120,
                training_duration_sec=300,
                detect_sign=0,  # 0 for all, 1 for positive, -1 for negative
                detect_time_radius_msec=detect_time_radius_msec,
                detect_threshold=threshold,
                npca_per_channel=npca_per_channel,
                npca_per_subdivision=npca_per_subdivision,
            )
            sorting = ms5.sorting_scheme2(
                recording=recording_preprocessed, sorting_parameters=sorting_params)
            timer.report()

        # Create sorting results folder
        current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        results_folder_name = f"sorting_results_{current_time}"
        sort_out_folder = out_folder / results_folder_name
        sort_out_folder.mkdir(parents=True, exist_ok=True)
        with open(sort_out_folder / "sorting_params.json", "w") as f:
            json.dump(sorting_params.__dict__, f)

        print("Unit IDs:", sorting.unit_ids)
        print("Spike counts per unit:", sorting.count_num_spikes_per_unit())

        # Register recording and create a sorting analyzer
        sorting.register_recording(rec_cr)
        analyzer_folder = sort_out_folder / "sorting_analyzer"
        sorting_analyzer = si.create_sorting_analyzer(sorting=sorting,
                                                      recording=rec_rm_artifacts,
                                                      format="binary_folder",
                                                      folder=str(analyzer_folder))
        print("Sorting analyzer:", sorting_analyzer)

        # Compute metrics

        try:
            sorting_analyzer.compute(
                ['random_spikes', 'waveforms', 'noise_levels'])
            sorting_analyzer.compute('templates')
            _ = sorting_analyzer.compute('template_similarity')
            _ = sorting_analyzer.compute('spike_amplitudes')
            _ = sorting_analyzer.compute('correlograms')
            _ = sorting_analyzer.compute('unit_locations')

            out_fig_folder = sort_out_folder / 'raw_units'
            out_fig_folder.mkdir(parents=True, exist_ok=True)

            for unit_id in sorting.get_unit_ids():
                sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
                plt.savefig(out_fig_folder / f'unit_summary_{unit_id}.png')
                plt.close()

        except Exception as e:
            print(f"Error during metrics computation: {e}")


def process_from_json(json_file="sorting_files.json"):
    """Simple function to read JSON and process recordings."""

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Construct the full path to the JSON file
    json_path = script_dir / json_file

    # Read JSON file
    with open(json_path, 'r') as f:
        config = json.load(f)

    threshold = config['threshold']

    # Process each recording
    for i, rec in enumerate(config['recordings'], 1):
        rec_folder = Path(rec['path'])
        shanks = rec['shanks']
        scheme = rec.get('scheme', 1)

        print(f"\n[{i}/{len(config['recordings'])}] Processing: {rec_folder.name}")
        print(f"  Shanks: {shanks}")

        # Call your main function
        main(threshold=threshold, rec_folder=rec_folder,
             shanks=shanks, scheme=scheme)


if __name__ == "__main__":
    process_from_json()
