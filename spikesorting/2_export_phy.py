import os
from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.exporters as sexp
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import apply_sortingview_curation
from spikeinterface.widgets import plot_sorting_summary
import numpy as np
import spikeinterface.preprocessing as sp

from rec2nwb.preproc_func import parse_session_info
from spikesorting.ss_proc_func import get_sortout_folder

# Constants
BASE_FOLDER = r"D:\cl\ephys"
# BASE_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed"  
DATES = ['sleep']
ANIMAL_IDS = ['CnL39SG']
# ISHS = [ 0, 1, 2, 3, 4, 5, 6, 7]  # List of shank numbers to process
ISHS = [ 0, 1, 2, 3]  # List of shank numbers to process

SORTOUT_FOLDER = get_sortout_folder()
# TODO: add overwrite option, if the phy folder exists, skip it if overwrite is False
OVERWRITE = False

for date in DATES:
    for animal_id in ANIMAL_IDS:
        # Construct experiment folder path
        experiment_folder = Path(BASE_FOLDER) / f"{date}/{animal_id}"
        if not experiment_folder.exists():
            print(f"Experiment folder not found: {experiment_folder}")
            continue
        # Select the first subdirectory found (if any)
        rec_folders = [p for p in experiment_folder.iterdir() if p.is_dir()]
        if rec_folders is None:
            continue
        print(f"Found {len(rec_folders)} recording folders in {experiment_folder}.")

        for rec_folder in rec_folders:

            # Parse session info (returns animal_id, session_id, folder_name)
            animal_id, session_id, folder_name = parse_session_info(rec_folder)
            session_folder = SORTOUT_FOLDER / f"{animal_id}/{animal_id}_{session_id}"

            for ish in ISHS:
                print(f"Processing {animal_id} {session_id} shank {ish}...")
                # Build recording file path
                recording_file = rec_folder / f"{folder_name}sh{ish}.nwb"
                if not recording_file.exists():
                    print(f"Recording file not found: {recording_file}")
                    continue
                
                # Create the NWB recording extractor
                recording = se.NwbRecordingExtractor(str(recording_file))
                rec_filt = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
                
                shank_folder = session_folder / f"shank{ish}"
                # Find folders starting with 'sorting_results_'
                sorting_results_folders = [
                    os.path.join(root, d)
                    for root, dirs, _ in os.walk(shank_folder)
                    for d in dirs
                    if d.startswith('sorting_results_')
                ]
                if not sorting_results_folders:
                    print(f"No sorting results folder found in {shank_folder}")
                    continue

                for sorting_results_folder in sorting_results_folders:
                    sorting_results_folder = Path(sorting_results_folder)
                    output_folder = sorting_results_folder / 'phy'
                    if output_folder.exists() and not OVERWRITE:
                        print(f"Phy folder already exists: {output_folder}")
                        continue
                    analyzer_folder = sorting_results_folder / 'sorting_analyzer'
                    
                    # Load the sorting analyzer
                    sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
                    sorting = sorting_analyzer.sorting
                    sorting_analyzer = create_sorting_analyzer(sorting, rec_filt)

                    sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
                    _ = sorting_analyzer.compute('spike_amplitudes')
                    _ = sorting_analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")
                    
                    # Export sorting results to Phy format
                    sexp.export_to_phy(sorting_analyzer, output_folder=output_folder)
