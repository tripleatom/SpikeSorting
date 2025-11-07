import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.widgets as sw
import spikeinterface as si
from pathlib import Path
import time
import matplotlib.pyplot as plt
from Timer import Timer
from rec2nwb.preproc_func import get_bad_ch_id, rm_artifacts, parse_session_info
import numpy as np
import os

rec_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CNL36\CNL36_250305_194558")

rec_folder = Path(rec_folder)
animal_id, session_id, folder_name = parse_session_info(str(rec_folder))
ishs = ['0', '1', '2', '3']

# ishs = ['0']

rec_folder = Path(rec_folder)

for ish in ishs:
    nwb_folder = rec_folder / f"{folder_name}sh{ish}.nwb"
    out_folder = Path('sortout') / animal_id / session_id / ish
    if not out_folder.exists():
        out_folder.mkdir(parents=True)


    rec = se.NwbRecordingExtractor(nwb_folder)
    print(rec)
    # print(rec.get_channel_gains())
    # print(rec.get_channel_locations())


    #%%
    rec_filt = sp.bandpass_filter(rec, freq_min=300, freq_max=6000, dtype='int32')



    bad_ch_id = get_bad_ch_id(rec, rec_folder, ish)
    remaining_ch = np.array([ch for ch in rec.get_channel_ids() if ch not in bad_ch_id])
    np.save(os.path.join(rec_folder, f'remaining_ch_sh{ish}.npy'), remaining_ch)
    print('Remaining channel IDs:', remaining_ch)
    chunk_size = 900
    rec_rm_artifacts = rm_artifacts(rec_filt, rec_folder, ish, bad_ch_id=bad_ch_id, chunk_size=chunk_size)
    # rec_clean = rec_rm_artifacts.channel_slice(remaining_ch)

    rec_cr = sp.common_reference(rec_rm_artifacts, reference='global', operator='median')

    rec_whiten = sp.whiten(rec_cr, dtype='float32')

    rec_preprocessed = rec_whiten

    import numpy as np
    start_times = np.arange(0, rec.get_total_duration(), 10)

    ts_whiten_out_folder = out_folder / 'timeseries_whiten'
    ts_cr_out_folder = out_folder / 'timeseries_cr'
    ts_filter_out_folder = out_folder / 'timeseries_filter'

    if not ts_whiten_out_folder.exists():
        ts_whiten_out_folder.mkdir(parents=True)
    if not ts_cr_out_folder.exists():
        ts_cr_out_folder.mkdir(parents=True)
    if not ts_filter_out_folder.exists():
        ts_filter_out_folder.mkdir(parents=True)

    for i, start_time in enumerate(start_times):
        time_range = [start_time, start_time + .2]
        sw.plot_traces(rec_whiten, backend='matplotlib', time_range=time_range,
                    order_channel_by_depth=True,)
        plt.savefig(ts_whiten_out_folder / f'timeseries_{i}.png')
        plt.close()
        sw.plot_traces(rec_cr, backend='matplotlib', time_range=time_range,
                    order_channel_by_depth=True,)
        plt.savefig(ts_cr_out_folder / f'timeseries_{i}.png')
        plt.close()

        sw.plot_traces(rec_filt, backend='matplotlib', time_range=time_range,
                    order_channel_by_depth=True,)
        plt.savefig(ts_filter_out_folder / f'timeseries_{i}.png')
        plt.close()