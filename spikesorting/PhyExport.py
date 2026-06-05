import os
import shutil
from pathlib import Path
import json
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.exporters as sexp
from spikeinterface import create_sorting_analyzer
from spikeinterface.core import aggregate_channels, aggregate_units
import numpy as np
import spikeinterface.preprocessing as sp
from rec2nwb.preproc_func import parse_session_info


def main(rec_folder, shanks, sortout_folder, animal_id="", overwrite=False, n_jobs=8):
    animal_id, session_id, folder_name = parse_session_info(str(rec_folder))
    session_folder = Path(sortout_folder) / animal_id / f"{animal_id}_{session_id}"

    for ish in shanks:
        print(f"\nProcessing {animal_id} {session_id} shank {ish}...")

        recording_file = rec_folder / f"{folder_name}sh{ish}.nwb"
        if not recording_file.exists():
            print(f"  Recording file not found: {recording_file}")
            continue

        shank_folder = session_folder / f"shank{ish}"
        if not shank_folder.exists():
            print(f"  Shank folder not found: {shank_folder}")
            continue

        sorting_results_folders = [
            Path(root) / d
            for root, dirs, _ in os.walk(shank_folder)
            for d in dirs
            if d.startswith('sorting_results_')
        ]
        if not sorting_results_folders:
            print(f"  No sorting results folder found in {shank_folder}")
            continue

        recording = se.read_nwb_recording(str(recording_file))
        rec_filt = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)

        for sorting_results_folder in sorting_results_folders:
            output_folder = sorting_results_folder / 'phy'
            if output_folder.exists() and not overwrite:
                print(f"  Phy folder already exists: {output_folder}")
                continue

            analyzer_folder = sorting_results_folder / 'sorting_analyzer'
            if not analyzer_folder.exists():
                print(f"  Sorting analyzer folder not found: {analyzer_folder}")
                continue

            temp_folder = sorting_results_folder / 'sorting_analyzer_temp'

            try:
                sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
                sorting = sorting_analyzer.sorting

                # Create temp analyzer on the bandpass-filtered (non-whitened) recording
                # Phy requires non-whitened traces for its waveform view
                sorting_analyzer = create_sorting_analyzer(
                    sorting, rec_filt,
                    format="binary_folder",
                    folder=temp_folder,
                    overwrite=True,
                )

                si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration='1s', progress_bar=True)
                sorting_analyzer.compute(
                    ['random_spikes', 'waveforms', 'templates', 'noise_levels'], n_jobs=n_jobs)
                sorting_analyzer.compute('spike_amplitudes', n_jobs=n_jobs)
                sorting_analyzer.compute(
                    'principal_components', n_components=5, mode='by_channel_local', n_jobs=n_jobs)

                sexp.export_to_phy(sorting_analyzer, output_folder=output_folder, copy_binary=False)
                print(f"  Exported to: {output_folder}")

            except Exception as e:
                print(f"  Error processing {sorting_results_folder.name}: {e}")

            finally:
                if temp_folder.exists():
                    shutil.rmtree(temp_folder)


def main_combined(rec_folder, shanks, sortout_folder, animal_id="", overwrite=False, n_jobs=8):
    """Export all shanks combined into a single Phy folder.

    Each unit's waveforms are computed only from its own shank's channels
    (by_property sparsity on 'group'), keeping memory manageable.
    """
    animal_id, session_id, folder_name = parse_session_info(str(rec_folder))
    session_folder = Path(sortout_folder) / animal_id / f"{animal_id}_{session_id}"
    output_folder = session_folder / 'phy_combined'
    temp_folder   = session_folder / 'phy_combined_temp'

    if output_folder.exists() and not overwrite:
        print(f"Combined phy folder already exists: {output_folder}")
        return

    recordings, sortings = [], []
    for ish in shanks:
        recording_file = rec_folder / f"{folder_name}sh{ish}.nwb"
        if not recording_file.exists():
            print(f"  Recording not found, skipping shank {ish}: {recording_file}")
            continue

        shank_folder = session_folder / f"shank{ish}"
        sorting_results_folders = sorted([
            Path(root) / d
            for root, dirs, _ in os.walk(shank_folder)
            for d in dirs
            if d.startswith('sorting_results_')
        ])
        if not sorting_results_folders:
            print(f"  No sorting results for shank {ish}, skipping")
            continue

        analyzer_folder = sorting_results_folders[0] / 'sorting_analyzer'
        if not analyzer_folder.exists():
            print(f"  No sorting_analyzer for shank {ish}, skipping")
            continue

        recording = se.read_nwb_recording(str(recording_file))
        rec_filt = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
        # Rename channels to avoid ID collisions across shanks
        new_ids = [f"sh{ish}_{ch}" for ch in rec_filt.get_channel_ids()]
        rec_filt = rec_filt.rename_channels(new_ids)
        recordings.append(rec_filt)

        analyzer = si.load_sorting_analyzer(analyzer_folder)
        sortings.append(analyzer.sorting)
        print(f"  Loaded shank {ish}: {len(analyzer.sorting.get_unit_ids())} units")

    if not recordings:
        print("No data loaded, aborting.")
        return

    combined_recording = aggregate_channels(recordings)
    combined_sorting   = aggregate_units(sortings)
    print(f"\nCombined: {combined_recording.get_num_channels()} channels, "
          f"{len(combined_sorting.get_unit_ids())} units")

    # Each unit gets waveforms only from its own shank (group property)
    sparsity = si.compute_sparsity(
        combined_sorting, combined_recording,
        method="by_property", by_property="group"
    )

    try:
        combined_analyzer = create_sorting_analyzer(
            combined_sorting, combined_recording,
            format="binary_folder",
            folder=temp_folder,
            overwrite=True,
            sparsity=sparsity,
        )

        si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration='1s', progress_bar=True)
        combined_analyzer.compute(
            ['random_spikes', 'waveforms', 'templates', 'noise_levels'], n_jobs=n_jobs)
        combined_analyzer.compute('spike_amplitudes', n_jobs=n_jobs)
        combined_analyzer.compute(
            'principal_components', n_components=5, mode='by_channel_local', n_jobs=n_jobs)

        sexp.export_to_phy(combined_analyzer, output_folder=output_folder, copy_binary=False)
        print(f"\nCombined export done: {output_folder}")

    except Exception as e:
        print(f"Error during combined export: {e}")
        raise

    finally:
        if temp_folder.exists():
            shutil.rmtree(temp_folder)


def process_from_json(json_file="PhyFiles.json", combined=False):
    """
    JSON format (mirrors MSSortingFiles.json):
    {
        "sortout": "/path/to/sortout",
        "n_jobs": 8,
        "overwrite": false,
        "recordings": [
            {"path": "/path/to/folder", "shanks": [0, 1], "animal_id": "M001"}
        ]
    }
    Set combined=True to export all shanks merged into one Phy folder.
    """
    script_dir = Path(__file__).parent
    json_path = script_dir / json_file

    with open(json_path, 'r') as f:
        config = json.load(f)

    sortout = config.get('sortout')
    if sortout is None:
        raise ValueError("'sortout' key is required in phy_files.json")

    n_jobs   = config.get('n_jobs', 8)
    overwrite = config.get('overwrite', False)

    fn = main_combined if combined else main

    for rec in config['recordings']:
        rec_path = Path(rec['path'])
        shanks   = rec.get('shanks', [0])
        animal_id = rec.get('animal_id', '')
        rec_n_jobs = rec.get('n_jobs', n_jobs)

        if not rec_path.exists():
            print(f"Recording folder not found: {rec_path}")
            continue

        print(f"\nProcessing: {rec_path.name}  shanks={shanks}")
        fn(rec_path, shanks, sortout, animal_id=animal_id,
           overwrite=overwrite, n_jobs=rec_n_jobs)


if __name__ == "__main__":
    process_from_json(combined=True)
