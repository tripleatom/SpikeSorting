import spikeinterface.extractors as se
import pickle
import numpy as np
from pathlib import Path
import spikeinterface as si
import spikeinterface.preprocessing as sp
from rec2nwb.preproc_func import rm_artifacts, parse_session_info
from kilosort import io

shanks = [0, 2]

for shank in shanks:

    print(f"\n=== Processing Shank {shank} ===")


    file_name = rf"c:\Users\xz106\data\CnL42SG_20251115_133046sh{shank}.nwb"

    rec = se.read_nwb(file_name)
    rec
    rec_folder = Path(file_name).parent

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
    recording_preprocessed: si.BaseRecording = rec_rm_artifacts

    DATA_DIRECTORY = rf"C:\Users\xz106\data\kilosort\sh{shank}"
    DATA_DIRECTORY = Path(DATA_DIRECTORY)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)



    dtype = recording_preprocessed.dtype
    filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
        recording_preprocessed, DATA_DIRECTORY, data_name='data.bin', dtype=dtype,
        chunksize=60000, export_probe=True, probe_name='probe.prb'
        )