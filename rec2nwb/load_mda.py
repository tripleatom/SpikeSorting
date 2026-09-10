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

# Attach the full ProbeInterface group in recording channel order.
from rec2nwb.probes import load_probes
from probeinterface import ProbeGroup
device_type = "8shank32"
channel_positions = {str(ch): i for i, ch in enumerate(recording.get_channel_ids())}
group = ProbeGroup()
for probe in load_probes(device_type):
    probe.set_device_channel_indices([
        channel_positions.get(str(ch), -1) for ch in probe.device_channel_indices
    ])
    group.add_probe(probe)
recording = recording.set_probegroup(group, in_place=False)

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
