#%%
from spikeinterface.extractors import PhySortingExtractor
from pathlib import Path
from spikeinterface import create_sorting_analyzer
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import numpy as np
import spikeinterface.widgets as sw
from matplotlib import pyplot as plt

shank = [0,2]

for shank in shank:
    phy_folder = rf"C:\Users\xz106\data\kilosort\sh{shank}\kilosort4"
    sorting = PhySortingExtractor(phy_folder)

    file_name = rf"c:\Users\xz106\data\CnL42SG_20251115_133046sh{shank}.nwb"
    rec = se.read_nwb(file_name)

    rec_car = sp.common_reference(rec, reference="global", operator="median")
    rec_filt = sp.bandpass_filter(rec_car, freq_min=300, freq_max=6000, dtype=np.float32)

    sorting = PhySortingExtractor(phy_folder)
    qualities = sorting.get_property('quality')
    # curated_sorting_folder = Path(sorting_results_folder) / 'curated_sorting_analyzer'

    # if curated_sorting_folder.exists() and not overwrite:
    #     sorting_analyzer = si.load_sorting_analyzer(curated_sorting_folder)
    # else:
    curated_sorting_folder = rf"C:\Users\xz106\data\kilosort\sh{shank}\sortout"
    sorting_analyzer = create_sorting_analyzer(sorting, rec_filt, 
                                        format="binary_folder",
                                        folder=str(curated_sorting_folder), overwrite=True)
    sorting_analyzer.compute(['random_spikes', 'waveforms', 'noise_levels'])
    sorting_analyzer.compute('templates')
    sorting_analyzer.compute('template_metrics')
    _ = sorting_analyzer.compute('template_similarity')
    _ = sorting_analyzer.compute('spike_amplitudes')
    _ = sorting_analyzer.compute('correlograms')
    _ = sorting_analyzer.compute('unit_locations')
    _ = sorting_analyzer.compute('isi_histograms')

#%%

    out_fig_folder = rf"C:\Users\xz106\data\kilosort\sh{shank}\unit"
    out_fig_folder = Path(out_fig_folder)
    out_fig_folder.mkdir(parents=True, exist_ok=True)

    for unit_id in sorting.get_unit_ids():
        sw.plot_unit_summary(sorting_analyzer, unit_id=unit_id)
        plt.savefig(out_fig_folder / f"unit_{unit_id}_summary.png")
        plt.close()
