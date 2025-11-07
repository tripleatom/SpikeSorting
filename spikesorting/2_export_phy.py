import os
from pathlib import Path
import json
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

def main(rec_folder, ISHS, OVERWRITE, SORTOUT_FOLDER):

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
            
            # Check if shank folder exists
            if not shank_folder.exists():
                print(f"Shank folder not found: {shank_folder}")
                continue
            
            # Find folders starting with 'sorting_results_'
            sorting_results_folders = [
                Path(root) / d
                for root, dirs, _ in os.walk(shank_folder)
                for d in dirs
                if d.startswith('sorting_results_')
            ]
            if not sorting_results_folders:
                print(f"No sorting results folder found in {shank_folder}")
                continue

            for sorting_results_folder in sorting_results_folders:
                output_folder = sorting_results_folder / 'phy'
                if output_folder.exists() and not OVERWRITE:
                    print(f"Phy folder already exists: {output_folder}")
                    continue
                
                analyzer_folder = sorting_results_folder / 'sorting_analyzer'
                
                # Check if analyzer folder exists
                if not analyzer_folder.exists():
                    print(f"Sorting analyzer folder not found: {analyzer_folder}")
                    continue
                
                try:
                    # Load the sorting analyzer
                    sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)
                    sorting = sorting_analyzer.sorting
                    
                    # Create new analyzer with filtered recording
                    sorting_analyzer = create_sorting_analyzer(sorting, rec_filt, format="binary_folder", folder=analyzer_folder.parent / "sorting_analyzer_temp")

                    # Compute extensions
                    sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
                    sorting_analyzer.compute('spike_amplitudes')
                    sorting_analyzer.compute('principal_components', n_components=5, mode="by_channel_local")
                    
                    # Export sorting results to Phy format
                    sexp.export_to_phy(sorting_analyzer, output_folder=output_folder)
                    
                    print(f"Successfully exported to: {output_folder}")
                    
                except Exception as e:
                    print(f"Error processing {sorting_results_folder}: {e}")
                    continue


def process_from_json(json_file="phy_files.json"):
    """
    Process recordings from a JSON configuration file.
    
    JSON format:
    {
        "recordings": [
            {
                "path": "path/to/recording.rec",
                "shanks": [0, 1, 2]
            }
        ]
    }
    """
    
    OVERWRITE = False
    SORTOUT_FOLDER = get_sortout_folder()
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    recs = data.get('recordings', [])
    if not recs:
        print("No recordings found in JSON file")
        return
    
    for rec in recs:
        rec_path = Path(rec['path'])
        shanks = rec.get('shanks', [0])

        if not rec_path.exists():
            print(f"Recording folder not found: {rec_path}")
            continue
        
        print(f"\nProcessing recording: {rec_path}")
        print(f"Shanks: {shanks}")
        main(rec_path, shanks, OVERWRITE, SORTOUT_FOLDER)


if __name__ == "__main__":
    json_path = Path(__file__).parent / "phy_files.json"
    process_from_json(json_path)