import pynwb
import numpy as np
from datetime import datetime
from pathlib import Path

nwb_file = r"D:\cl\ephys\CnL42SG_20251115_133046.rec\CnL42SG_20251115_133046sh6_before_clean.nwb"
nwb_file = Path(nwb_file)
nwb_folder = nwb_file.parent

# Read the original file
with pynwb.NWBHDF5IO(nwb_file, 'r') as io:
    nwbfile_in = io.read()
    
    # Get the electrical series data
    electrical_series = nwbfile_in.acquisition['ElectricalSeries']
    data = electrical_series.data[:]
    timestamps = electrical_series.timestamps[:]
    
    # Define the noisy region to remove (in seconds)
    noise_start_time = 4190.0
    noise_end_time = 4570.0
    
    # Find indices
    mask = (timestamps < noise_start_time) | (timestamps > noise_end_time)
    clean_data = data[mask]
    clean_timestamps = timestamps[mask]
    
    # Create new NWB file with all metadata
    nwbfile_out = pynwb.NWBFile(
        session_description=nwbfile_in.session_description,
        identifier=f"{nwbfile_in.identifier}_cleaned",
        session_start_time=nwbfile_in.session_start_time,
        experimenter=nwbfile_in.experimenter,
        lab=nwbfile_in.lab,
        institution=nwbfile_in.institution,
        experiment_description=nwbfile_in.experiment_description,
        session_id=nwbfile_in.session_id,
    )
    
    # Copy devices
    for device_name, device in nwbfile_in.devices.items():
        nwbfile_out.create_device(
            name=device.name,
            description=device.description if hasattr(device, 'description') else '--',
            manufacturer=device.manufacturer if hasattr(device, 'manufacturer') else '--'
        )
    
    # Copy electrode groups
    electrode_group_map = {}
    for group_name, group in nwbfile_in.electrode_groups.items():
        new_group = nwbfile_out.create_electrode_group(
            name=group.name,
            description=group.description,
            location=group.location,
            device=nwbfile_out.devices[group.device.name]
        )
        electrode_group_map[group_name] = new_group
    
    # Add custom electrode columns BEFORE adding any electrodes
    for col in nwbfile_in.electrodes.colnames:
        if col not in ['location', 'group', 'group_name', 'x', 'y', 'z', 'imp', 'filtering', 
                       'rel_x', 'rel_y', 'rel_z', 'reference']:
            nwbfile_out.add_electrode_column(
                name=col,
                description=nwbfile_in.electrodes[col].description
            )
    
    # Copy electrodes
    for idx in range(len(nwbfile_in.electrodes)):
        electrode_kwargs = {
            'group': electrode_group_map[nwbfile_in.electrodes['group'][idx].name],
            'location': nwbfile_in.electrodes['location'][idx],
        }
        
        # Add all standard and custom columns
        for col in nwbfile_in.electrodes.colnames:
            if col not in ['location', 'group', 'group_name']:
                value = nwbfile_in.electrodes[col][idx]
                # Convert to appropriate type
                if col in ['x', 'y', 'z', 'rel_x', 'rel_y', 'rel_z', 'imp']:
                    electrode_kwargs[col] = float(value)
                else:
                    electrode_kwargs[col] = value
        
        nwbfile_out.add_electrode(**electrode_kwargs)
    
    # Create electrode table region
    electrode_table_region = nwbfile_out.create_electrode_table_region(
        region=list(range(len(nwbfile_out.electrodes))),
        description=electrical_series.electrodes.description
    )
    
    # Add cleaned electrical series
    clean_series = pynwb.ecephys.ElectricalSeries(
        name='ElectricalSeries',
        data=clean_data,
        electrodes=electrode_table_region,
        timestamps=clean_timestamps,
        conversion=electrical_series.conversion,
        offset=electrical_series.offset,
        description='Cleaned data with noisy chunk removed'
    )
    
    nwbfile_out.add_acquisition(clean_series)
    
    # Write to new file
    output_path = nwb_folder / f'{nwb_file.stem}_cleaned.nwb'
    with pynwb.NWBHDF5IO(output_path, 'w') as io_out:
        io_out.write(nwbfile_out)
    
    print(f"Cleaned file saved to: {output_path}")
    print(f"Original data points: {len(timestamps)}")
    print(f"Cleaned data points: {len(clean_timestamps)}")
    print(f"Removed {len(timestamps) - len(clean_timestamps)} data points")