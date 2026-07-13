#%%
import re
import time
import json
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
from kilosort import run_kilosort
import spikeinterface.extractors as se
from rec2nwb.preproc_func import parse_session_info



class NWBLazyArray:
    """
    Wraps an NWB ElectricalSeries as a file_object for kilosort4's run_kilosort().
    Only removes bad channels; all preprocessing (CAR, filtering) is done by kilosort.
    Returns int16 data as kilosort expects.
    No data is written to disk.

    kilosort4 accesses data via file_object[start:end, :] expecting shape (n_samples, n_channels).
    """
    def __init__(self, electrical_series, good_indices=None):
        self._data  = electrical_series.data        # h5py.Dataset (n_samples, n_channels)
        self._good  = good_indices                  # int array of good column indices
        n_good = len(good_indices) if good_indices is not None else self._data.shape[1]
        self.shape  = (self._data.shape[0], n_good)
        self.dtype  = np.dtype('int16')

    def __getitem__(self, idx):
        chunk = self._data[idx]
        if self._good is not None:
            chunk = chunk[:, self._good]
        return chunk.astype(np.int16)

    def __len__(self):
        return self.shape[0]

# ============================================================
# Configuration
# ============================================================
nwb_path = r"F:\CnL42SG\CnL42SG_20260325\CnL42SG_20260325sh7.nwb"
sortout   = r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout"

kilosort_params = {
    'batch_size': 60000,  # default=60000; increase to use more GPU memory per batch
}

# ============================================================
# Derive animal_id / session_id / shank from the NWB path
# ============================================================
nwb_path = Path(nwb_path)

# parse_session_info works on folder names; use the parent dir
animal_id, session_id, folder_name = parse_session_info(str(nwb_path.parent))

# Extract shank index from filename, e.g. CnL42SG_20260325sh0.nwb → 0
shank_match = re.search(r'sh(\d+)', nwb_path.stem)
shank = int(shank_match.group(1)) if shank_match else 0

# ============================================================
# Build output folder  (mirrors 1_ms_sorting.py structure)
#   sortout / animal_id / animal_id_session_id / shank{N} /
#       kilosort4_{YYYYMMDD_HHMM}/
# ============================================================
current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
out_folder = (
    Path(sortout)
    / animal_id
    / f"{animal_id}_{session_id}"
    / f"shank{shank}"
)
sort_out_folder = out_folder / f"kilosort4_{current_time}"
sort_out_folder.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {sort_out_folder}")

# ============================================================
# Load recording
# ============================================================
print(f"\nLoading NWB: {nwb_path}")
rec = se.read_nwb_recording(str(nwb_path))
fs = rec.get_sampling_frequency()
print(f"  {rec.get_num_channels()} channels, {fs} Hz, {rec.get_total_duration():.1f} s")

#%%

# ============================================================
# Bad channel detection
# ============================================================
bad_channel_file = r"F:\CnL42SG\CnL42SG_20260325\bad_channels.txt"

# Read SpikeGadgets channel numbers from bad_channels.txt
with open(bad_channel_file, "r") as f:
    bad_channel_spikegadget_id = [int(line.strip()) for line in f if line.strip()]
print(f"Bad SpikeGadgets channel IDs from file: {bad_channel_spikegadget_id}")

# channel_label looks like "shankX: YY" where YY is the SpikeGadgets channel number.
# Match any channel whose label's numeric suffix is in the bad list.
channel_labels = rec.get_property('label')   # array of strings, one per channel
channel_ids    = rec.get_channel_ids()

bad_channel_ids = []
for ch_id, label in zip(channel_ids, channel_labels):
    m = re.search(r':\s*(\d+)', str(label))
    if m and int(m.group(1)) in bad_channel_spikegadget_id:
        bad_channel_ids.append(ch_id)
bad_channel_ids = np.array(bad_channel_ids)
print(f"Matched bad channel IDs ({len(bad_channel_ids)}): {bad_channel_ids.tolist()}")

#%%
# Remove bad channels
if len(bad_channel_ids) > 0:
    rec_good = rec.remove_channels(remove_channel_ids=bad_channel_ids)
    print(f"  Channels remaining: {rec_good.get_num_channels()}")
else:
    rec_good = rec
    print("  No bad channels removed.")

# ============================================================
# Build probe from good-channel locations only
# ============================================================
all_channel_ids = rec.get_channel_ids().tolist()
all_locations   = rec.get_channel_locations()   # (n_channels, 2) — x, y in µm

# Integer indices of bad channels in the full array
bad_channel_indices = [all_channel_ids.index(ch) for ch in bad_channel_ids.tolist()]
good_channel_indices = np.array([i for i in range(len(all_channel_ids))
                                  if i not in bad_channel_indices])
good_locations = all_locations[good_channel_indices]
print(f"Bad channel indices removed: {bad_channel_indices}")
print(f"Good channels remaining: {len(good_channel_indices)}")

n_good = len(good_channel_indices)
probe = {
    'xc':      good_locations[:, 0].astype(np.float32),
    'yc':      good_locations[:, 1].astype(np.float32),
    'kcoords': np.zeros(n_good, dtype=np.float32),   # all on shank 0
    'chanMap': np.arange(n_good, dtype=np.int32),
    'n_chan':  n_good,
}

#%%
# ============================================================
# Save parameters
# ============================================================
params_to_save = {
    "nwb_path": str(nwb_path),
    "animal_id": animal_id,
    "session_id": session_id,
    "shank": shank,
    "bad_channels": bad_channel_ids.tolist(),
    "bad_channel_indices": bad_channel_indices,
    "bad_spikegadget_ids": bad_channel_spikegadget_id,
    "kilosort_params": kilosort_params,
}
with open(sort_out_folder / "sorting_params.json", "w") as f:
    json.dump(params_to_save, f, indent=2)

#%%
# ============================================================
# Run KiloSort4 via in-memory file_object (no binary written)
# ============================================================
print("\nRunning KiloSort4...")
kilosort_folder = sort_out_folder / "kilosort4_output"
kilosort_folder.mkdir(parents=True, exist_ok=True)

io = NWBHDF5IO(str(nwb_path), 'r')
nwb_file = io.read()
es_key = list(nwb_file.acquisition.keys())[0]
es = nwb_file.acquisition[es_key]
print(f"  ElectricalSeries: '{es_key}', shape {es.data.shape}")

file_object = NWBLazyArray(es, good_indices=good_channel_indices)
print(f"  Shape seen by Kilosort: {file_object.shape}")

settings = {
    'n_chan_bin': file_object.shape[1],
    'fs': float(fs),
    **kilosort_params,
}

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nvmlInit()
best_gpu = min(range(torch.cuda.device_count()),
               key=lambda i: nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).used)
device = torch.device(f'cuda:{best_gpu}')
print(f"  Using device: {device} ({torch.cuda.get_device_name(device)})")

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(
    settings=settings,
    filename=nwb_path,       # required by set_files even when file_object is used
    file_object=file_object,
    data_dtype='int16',
    do_CAR=True,
    results_dir=str(kilosort_folder),
    probe=probe,
    device=device,
)
io.close()

print(f"\n=== Sorting Results ===")
n_units = len(np.unique(clu))
print(f"Units found: {n_units}")
print(f"Total spikes: {len(st)}")
print(f"Results saved to: {kilosort_folder}")
print(f"\nAll results saved to: {sort_out_folder}")
