import pynwb
import numpy as np
from datetime import datetime

nwb_file = r"D:\cl\ephys\CnL42SG_20251115_133046.rec\CnL42SG_20251115_133046sh0.nwb"
# Read the original file
with pynwb.NWBHDF5IO(nwb_file, 'r') as io:
    nwbfile_in = io.read()
    
    # Get the electrical series data
    electrical_series = nwbfile_in.acquisition['ElectricalSeries']  # adjust name
    data = electrical_series.data[:]
    timestamps = electrical_series.timestamps[:]
    
    # Define the noisy region to remove (in seconds or sample indices)
    noise_start_time = 100.0  # seconds
    noise_end_time = 150.0    # seconds
    
    # Find indices
    mask = (timestamps < noise_start_time) | (timestamps > noise_end_time)
    clean_data = data[mask]
    clean_timestamps = timestamps[mask]
    
    # Create new NWB file with clean data
    nwbfile_out = pynwb.NWBFile(
        session_description=nwbfile_in.session_description,
        identifier=f"{nwbfile_in.identifier}_cleaned",
        session_start_time=nwbfile_in.session_start_time
    )
    
    # Add cleaned electrical series
    clean_series = pynwb.ecephys.ElectricalSeries(
        name='ElectricalSeries',
        data=clean_data,
        electrodes=electrical_series.electrodes,
        timestamps=clean_timestamps,
        description='Cleaned data with noisy chunk removed'
    )
    
    nwbfile_out.add_acquisition(clean_series)
    
    # Write to new file
    with pynwb.NWBHDF5IO('output_cleaned.nwb', 'w') as io_out:
        io_out.write(nwbfile_out)