#%%
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as si

n_part = 5
spikegadgets_file = Path(rf"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260227\CnL42SG_20260227\CnL42_20260227_143319.rec\CnL42_20260227_143319.part{n_part}.rec")

recording = si.read_spikegadgets(spikegadgets_file)

# Set channel locations from device mapping CSV
# spikegadget column == .rec channel ID (bare numeric string)
device_type = "8shank32"
mapping_file = Path(__file__).resolve().parent / "mapping" / f"{device_type}.csv"
channel_map = pd.read_csv(mapping_file)
ch_to_pos = {str(int(row['spikegadget'])): (row['xcoord'], row['ycoord'])
             for _, row in channel_map.iterrows()}
ch_ids = recording.get_channel_ids()
locations = np.array([ch_to_pos[ch] for ch in ch_ids])  # shape (N, 2): col0=x, col1=depth
recording.set_channel_locations(locations)

print(recording)

# %%
rec_filt = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
rec_cmr = sp.common_reference(rec_filt, reference='global')

sw.plot_traces(
    rec_cmr,
    time_range=(0, 1),
    channel_ids=['0', '1', '2', '3', '4', '5', '6'],
    backend="matplotlib"
)
plt.show()
# %%
