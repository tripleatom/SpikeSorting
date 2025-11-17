import os
from pathlib import Path
from matplotlib import pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface import create_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
import numpy as np
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
from spikesorting.ss_proc_func import get_sortout_folder
from rec2nwb.preproc_func import parse_session_info


#FIXME: problematic code, didn't change according to data refactoring

# Constants
# BASE_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_spinal_cord_electrode\CoI"
BASE_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed"  
DATES = ['250912']
# ANIMAL_IDS = ['CoI06', 'CoI07', 'CoI08', 'CoI09', 'CoI10']
ANIMAL_IDS = [ 'CnL38', 'CnL39']

ISHS = [0, 1, 2, 3]
SORTOUT_FOLDER = get_sortout_folder()
overwrite = False

for date in DATES:
    for animal_id in ANIMAL_IDS:
        # Construct experiment folder path
        experiment_folder = Path(BASE_FOLDER) / f"{date}/{animal_id}"
        if not experiment_folder.exists():
            print(f"Experiment folder not found: {experiment_folder}")
            continue
        # Select the first subdirectory found (if any)
        rec_folder = next((p for p in experiment_folder.iterdir() if p.is_dir()), None)
        print(f"Recording folder: {rec_folder}")
        if rec_folder is None:
            continue

        # Parse session info (returns animal_id, session_id, folder_name)
        animal_id, session_id, folder_name = parse_session_info(rec_folder)
        session_folder = SORTOUT_FOLDER / f"{animal_id}/{animal_id}_{session_id}"

        for ish in ISHS:

            print(f"Processing {animal_id} {session_id} shank {ish}...")
            # Build recording file path
            recording_file = rec_folder / f"{folder_name}sh{ish}.nwb"
            
            # Create the NWB recording extractor
            recording = se.NwbRecordingExtractor(str(recording_file))
            rec_filt = sp.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
            
            shank_folder = session_folder / f'shank{ish}'
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
                phy_folder = Path(sorting_results_folder) / 'phy'
                sorting = PhySortingExtractor(phy_folder)
                qualities = sorting.get_property('quality')
                curated_sorting_folder = Path(sorting_results_folder) / 'curated_sorting_analyzer'

                if curated_sorting_folder.exists() and not overwrite:
                    sorting_analyzer = si.load_sorting_analyzer(curated_sorting_folder)
                else:
                    sorting_analyzer = create_sorting_analyzer(sorting, rec_filt, 
                                                      format="binary_folder",
                                                      folder=str(curated_sorting_folder))
                    sorting_analyzer.compute(['random_spikes', 'waveforms', 'noise_levels'])
                    sorting_analyzer.compute('templates')
                    _ = sorting_analyzer.compute('template_similarity')
                    _ = sorting_analyzer.compute('spike_amplitudes')
                    _ = sorting_analyzer.compute('correlograms')
                    _ = sorting_analyzer.compute('unit_locations')


                out_fig_folder = Path(sorting_results_folder) / 'curated_units'
                out_fig_folder.mkdir(parents=True, exist_ok=True)

                for index, unit_id in enumerate(sorting.get_unit_ids()):
                    w = sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id,  backend="matplotlib",)
                    fig = w.figure
                    fig.suptitle(f"Unit {unit_id}: {qualities[index]}")
                    plt.savefig(out_fig_folder / f'unit_summary_{unit_id}.png')
                    plt.close()
