import os
import re
import sys
import numpy as np
import spikeinterface.preprocessing as spre
import json
import tkinter as tk
from pathlib import Path

# Lives in its own stdlib-only module so callers that just need to read a
# session name (pipeline_gui.py) do not have to import spikeinterface.
from rec2nwb.session_id import parse_session_info

def get_animal_id(data_folder):
    """Get animal ID with user confirmation."""
    # Auto-detect animal ID
    detected_id = data_folder.stem.split('_')[0]
    print(f"\nDetected animal ID: {detected_id}")
    
    while True:
        confirm = input(f"Is '{detected_id}' correct? (y/n): ").strip().lower()
        
        if confirm == 'y' or confirm == 'yes':
            animal_id = detected_id
            print(f"✓ Using animal ID: {animal_id}")
            break
        elif confirm == 'n' or confirm == 'no':
            animal_id = input("Enter the correct animal ID: ").strip()
            if animal_id:  # Check if not empty
                print(f"✓ Animal ID set to: {animal_id}")
                break
            else:
                print("Animal ID cannot be empty. Please try again.")
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    return animal_id

def choose_device_type(animal_id: str) -> str:
    """
    Pop up a small Tk window to let the user pick one of the
    CSV‐stems inside ./mapping (ignores any files starting with ._).
    Returns the chosen stem (string).
    """
    mapping_dir = Path(__file__).resolve().parent / "mapping"
    if not mapping_dir.is_dir():
        raise FileNotFoundError(f"Mapping folder not found: {mapping_dir}")

    choices = sorted(
        p.stem for p in mapping_dir.glob("*.csv")
        if not p.name.startswith("._")
    )
    if not choices:
        raise FileNotFoundError(f"No valid .csv files in {mapping_dir}")

    root = tk.Tk()
    root.title(f"Choose device type for {animal_id}")
    root.geometry("300x150")
    tk.Label(root, text="Device type:").pack(padx=10, pady=(10, 0))

    var = tk.StringVar(value=choices[0])
    tk.OptionMenu(root, var, *choices).pack(padx=10, pady=5)

    def on_ok():
        root.quit()

    tk.Button(root, text="OK", command=on_ok).pack(pady=(0,10))

    # center window
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")

    root.mainloop()
    selection = var.get()
    root.destroy()
    return selection


def get_or_set_device_type(animal_id: str) -> str:
    """
    Load (or create) device_types.json, return the device_type for this animal_id.
    If missing, pop up chooser and write it back to JSON.
    """
    code_dir  = Path(__file__).resolve().parent
    json_path = code_dir / "device_types.json"

    # load existing map (or start fresh)
    if json_path.exists():
        device_map = json.loads(json_path.read_text())
    else:
        device_map = {}

    # if we already know this animal, return it
    if animal_id in device_map:
        return device_map[animal_id]

    # else, ask the user
    dt = choose_device_type(animal_id)
    device_map[animal_id] = dt
    json_path.write_text(json.dumps(device_map, indent=4))
    print(f"Saved {animal_id} → {dt} in {json_path}")
    return dt




def get_bad_ch_id(rec, folder, ish, load_if_exists=True):
    r"""
    Retrieve or detect bad channel IDs.

    Args:
        rec: Recording object.
        folder: Parent folder for the NWB file.
        ish: Shank identifier.
        load_if_exists (bool): If True, load from file if available.

    Returns:
        np.ndarray: Array of bad channel IDs.
    """
    bad_ch_file = folder / f'bad_ch_id_sh{ish}.npy'
    if load_if_exists and os.path.exists(bad_ch_file):
        bad_ch_id = np.load(bad_ch_file)
    else:
        bad_ch_id, _ = spre.detect_bad_channels(
            rec, num_random_chunks=400, n_neighbors=5, dead_channel_threshold=-0.2
        )
        np.save(bad_ch_file, bad_ch_id)

    print('Bad channel IDs:', bad_ch_id)
    return bad_ch_id


def rm_artifacts(rec_filtered, folder, ish, mode="cubic", threshold=6, chunk_time=0.05, overwrite=False):
    r"""
    Remove artifacts from the filtered recording.

    Args:
        rec_filtered: The filtered recording object.
        folder: Parent folder for saving results.
        ish: Shank identifier.
        bad_ch_id: List of bad channel IDs (optional).
        threshold: Threshold for artifact detection.
        chunk_time: Chunk size in seconds.
        overwrite: If True, recompute artifact indices even if they already exist.

    Returns:
        Recording object with artifacts removed.
    """
    fs = rec_filtered.get_sampling_frequency()
    chunk_size = int(chunk_time * fs)
    n_timepoints = rec_filtered.get_num_frames()
    n_channels = rec_filtered.get_num_channels()
    num_chunks = int(np.ceil(n_timepoints / chunk_size))

    artifact_file = folder / f'artifact_indices_sh{ish}.npy'
    if not overwrite and os.path.exists(artifact_file):
        artifact_indices = np.load(artifact_file)
    else:
        # Compute norm of traces per chunk and channel using batch reads.
        # Read large batches (e.g. 10s) at once, then reshape into chunks
        # to avoid thousands of tiny I/O calls.
        batch_time = 10.0  # seconds per batch read
        batch_size = int(batch_time * fs)
        norms = np.zeros((num_chunks, n_channels))

        n_batches = int(np.ceil(n_timepoints / batch_size))
        bar_width = 40
        for b in range(n_batches):
            batch_start = b * batch_size
            batch_end = min((b + 1) * batch_size, n_timepoints)
            traces = rec_filtered.get_traces(
                start_frame=batch_start, end_frame=batch_end, return_scaled=True)

            # Figure out which chunks fall in this batch
            chunk_idx_start = batch_start // chunk_size
            n_samples_in_batch = batch_end - batch_start
            n_chunks_in_batch = int(np.ceil(n_samples_in_batch / chunk_size))

            for c in range(n_chunks_in_batch):
                local_start = c * chunk_size
                local_end = min((c + 1) * chunk_size, n_samples_in_batch)
                norms[chunk_idx_start + c] = np.linalg.norm(
                    traces[local_start:local_end], axis=0)

            # Progress bar
            pct = (b + 1) / n_batches
            filled = int(bar_width * pct)
            bar = '#' * filled + '-' * (bar_width - filled)
            sys.stdout.write(f"\r  Computing norms [{bar}] {pct*100:.0f}%")
            sys.stdout.flush()
        print()  # newline after progress bar

        # Determine which chunks to discard based on threshold (vectorized).
        means = np.mean(norms, axis=0)
        stds = np.std(norms, axis=0)
        artifact_mask = norms > means[np.newaxis, :] + threshold * stds[np.newaxis, :]
        bad_chunks = np.any(artifact_mask, axis=1)

        # Also mark neighbors of bad chunks
        bad_indices = np.where(bad_chunks)[0]
        use_chunk = ~bad_chunks
        if bad_indices.size > 0:
            use_chunk[bad_indices[bad_indices > 0] - 1] = False
            use_chunk[bad_indices[bad_indices < num_chunks - 1] + 1] = False

        # Summary
        n_bad = int(np.sum(~use_chunk))
        bad_duration = n_bad * chunk_time
        total_duration = n_timepoints / fs
        print(f"  Artifacts: {n_bad} chunks ({bad_duration:.2f}s / {total_duration:.1f}s, "
              f"{bad_duration/total_duration*100:.2f}%)")

        # Convert chunk indices to timepoints.
        artifact_indices = np.where(~use_chunk)[0] * chunk_size
        np.save(artifact_file, artifact_indices)

    # Convert chunk size to milliseconds.
    chunk_time_ms = chunk_size / fs * 1000
    if artifact_indices.size > 0:
        #FIXME: how this handles the connection point. will this set all channels to 0?
        # mode“zeros”, “linear”, “cubic”, “average”, “median”, default: “zeros”
        rec_rm_artifacts = spre.remove_artifacts(
            rec_filtered, list_triggers=artifact_indices, ms_before=0, ms_after=chunk_time_ms,
            mode=mode
        )
    else:
        rec_rm_artifacts = rec_filtered

    return rec_rm_artifacts