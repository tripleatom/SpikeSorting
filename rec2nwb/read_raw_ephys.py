import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import sys
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import neo.rawio
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO
from rec2nwb.preproc_func import get_or_set_device_type, get_animal_id
import shutil


class EphysToNWBConverter:
    """Unified converter for Intan and SpikeGadgets data to NWB format with optimized memory handling."""
    
    def __init__(self, recording_method: str, chunk_duration: float = 60.0):
        """
        Initialize converter with recording method.
        
        Args:
            recording_method: Either 'intan' or 'spikegadget'
            chunk_duration: Duration in seconds for each chunk when processing large files (default: 60s)
        """
        if recording_method not in ['intan', 'spikegadget']:
            raise ValueError("Recording method must be 'intan' or 'spikegadget'")
        self.recording_method = recording_method
        self.chunk_duration = chunk_duration
    
    def get_stream_ids(self, file_path: str) -> any:
        """
        Get the stream ids from an Intan file.
        Only applies to Intan recordings.
        """
        if self.recording_method != 'intan':
            return None
            
        file_path = str(file_path)
        reader = neo.rawio.IntanRawIO(filename=file_path)
        reader.parse_header()
        header = reader.header
        return header['signal_streams']['id']

    def get_timestamp(self, file_path: Path) -> datetime:
        """
        Extract the recording start time from the filename.
        Handles both Intan and SpikeGadgets formats.
        """
        if self.recording_method == 'intan':
            # Expected format: <prefix>_yymmdd_HHMMSS.rh[s|d]
            match = re.match(
                r"([a-zA-Z0-9_]+)_([0-9]+_[0-9]+).rh(?:s|d)", file_path.name)
            if match:
                rec_datetimestr = match.group(2)  # yymmdd_HHMMSS
                return datetime.strptime(rec_datetimestr, "%y%m%d_%H%M%S")
        else:
            # Expected format: contains _YYYYMMDD_HHMMSS
            match = re.search(r"_(\d{8})_(\d{6})", file_path.name)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        
        raise ValueError("Filename does not match expected pattern.")

    def get_ch_index_on_shank(self, ish: int, device_type: str) -> tuple:
        """
        Return the channel indices on a given shank.
        Returns: (channel indices, x-coordinates, y-coordinates)
        """
        script_dir = Path(__file__).resolve().parent
        mapping_file = script_dir / "mapping" / f"{device_type}.csv"

        channel_map = pd.read_csv(mapping_file)
        xcoord = channel_map['xcoord'].astype(float).to_numpy()
        ycoord = channel_map['ycoord'].astype(float).to_numpy()
        sh = channel_map['sh'].astype(int).to_numpy()

        ch_index = np.where(sh == ish)[0]
        return ch_index, xcoord[ch_index], ycoord[ch_index]

    def _setup_spikegadget_files(self, data_file: Path, selected_geom: Path = None):
        """Setup required files for SpikeGadgets reading."""
        if self.recording_method != 'spikegadget':
            return
            
        mda_folder = data_file.parent
        script_dir = Path(__file__).resolve().parent
        params_path = script_dir / "params.json"
        
        # Use selected geom file or default
        if selected_geom is None:
            geom_path = script_dir / "geom.csv"
        else:
            geom_path = selected_geom
        
        shutil.copy2(params_path, mda_folder)
        shutil.copy2(geom_path, mda_folder / "geom.csv")  # Always copy as geom.csv

    def _get_recording_info(self, data_file: Path, selected_geom: Path = None):
        """
        Get recording object and metadata without loading all data.
        Returns: recording object, sampling frequency, number of frames, conversion, offset
        """
        if self.recording_method == 'intan':
            recording = se.read_intan(data_file, stream_id='0')
            conversion = recording.get_channel_gains()[0] / 1e6
            offset = recording.get_channel_offsets()[0] / 1e6
        else:  # spikegadget
            self._setup_spikegadget_files(data_file, selected_geom)
            mda_folder = data_file.parent
            mda_file = data_file.name
            recording = se.read_mda_recording(mda_folder, mda_file, 
                                            params_fname="params.json",
                                            geom_fname="geom.csv")
            conversion = 0.195 / 1e6  # Convert to V
            offset = 0.0 / 1e6
        
        sampling_freq = recording.get_sampling_frequency()
        num_frames = recording.get_num_frames()
        
        return recording, sampling_freq, num_frames, conversion, offset

    def _read_recording_chunk(self, recording, channel_ids: list = None, 
                             start_frame: int = 0, end_frame: int = None):
        """
        Read a chunk of recording data.
        
        Args:
            recording: SpikeInterface recording object
            channel_ids: List of channel IDs to read
            start_frame: Starting frame for chunk
            end_frame: Ending frame for chunk
        """
        if end_frame is None:
            end_frame = recording.get_num_frames()
        
        if channel_ids:
            trace = recording.get_traces(channel_ids=channel_ids, 
                                        start_frame=start_frame, 
                                        end_frame=end_frame)
        else:
            trace = recording.get_traces(start_frame=start_frame, 
                                        end_frame=end_frame)
        
        return trace

    def _estimate_file_size_gb(self, num_frames: int, num_channels: int, dtype_size: int = 2):
        """
        Estimate file size in GB.
        
        Args:
            num_frames: Number of time frames
            num_channels: Number of channels
            dtype_size: Size of data type in bytes (default: 2 for int16)
        """
        total_bytes = num_frames * num_channels * dtype_size
        return total_bytes / (1024**3)

    def _should_use_chunked_processing(self, num_frames: int, num_channels: int, 
                                      threshold_gb: float = 3.0):
        """
        Determine if chunked processing should be used based on estimated data size.
        
        Args:
            num_frames: Number of time frames
            num_channels: Number of channels
            threshold_gb: Size threshold in GB for using chunked processing (default: 3.0)
        """
        estimated_size = self._estimate_file_size_gb(num_frames, num_channels)
        return estimated_size > threshold_gb

    def initiate_nwb(self, data_file: Path, nwb_path: Path, ishank: int = 0,
                     impedance_path: str = None, bad_ch_ids: list = None,
                     metadata: dict = None, has_multiple_files: bool = False) -> list:
        """
        Create and write an NWB file from recording data with optimized memory handling.
        
        Args:
            has_multiple_files: If True, forces chunked dataset creation even for smaller files
                               to allow appending additional files later
        """
        metadata = metadata or {}
        print("Initiating NWB file...")
        
        session_start_time = self.get_timestamp(data_file)
        nwb_description = metadata.get("session_desc", f"NWB file for {self.recording_method} data")
        experimenter = metadata.get("experimenter", "Zhang, Xiaorong")
        lab = metadata.get("lab", "XL Lab")
        institution = metadata.get("institution", "Rice University")
        exp_desc = metadata.get("exp_desc", "None")
        session_id = metadata.get("session_id", "None")
        electrode_location = metadata.get("electrode_location", None)
        device_type = metadata.get("device_type", "4shank16intan" if self.recording_method == 'intan' else "4shank16")
        selected_geom = metadata.get("selected_geom", None)
        
        nwbfile = NWBFile(
            session_description=nwb_description,
            identifier=str(uuid4()),
            session_start_time=session_start_time,
            experimenter=[experimenter],
            lab=lab,
            institution=institution,
            experiment_description=exp_desc,
            session_id=session_id,
        )

        print("Adding device...")
        channel_index, xcoord, ycoord = self.get_ch_index_on_shank(ishank, device_type)
        
        # Create a device and add electrode metadata
        device = nwbfile.create_device(
            name="--", description="--", manufacturer="--")
        nwbfile.add_electrode_column(
            name="label", description="label of electrode")

        electrode_group = nwbfile.create_electrode_group(
            name=f"shank{ishank}",
            description=f"electrode group for shank {ishank}",
            device=device,
            location=electrode_location,
        )

        # Handle impedance and channel names
        impedance_sh = None
        channel_name_sh = None
        if impedance_path is not None:
            impedance_table = pd.read_csv(impedance_path)
            impedance = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()
            impedance_sh = impedance[channel_index]
            channel_name = impedance_table['Channel Name'].to_numpy()
            channel_name_sh = channel_name[channel_index]
        else:
            # Create default channel names and impedances
            channel_name_sh = [f"ch{i}" for i in channel_index]
            impedance_sh = [np.nan] * len(channel_index)

        # Create electrode DataFrame
        electrode_df = pd.DataFrame({
            'channel_name': channel_name_sh,
            'impedance': impedance_sh,
            'x': xcoord,
            'y': ycoord,
            'channel_index': channel_index
        })

        # Remove bad channels from the DataFrame
        if bad_ch_ids is not None:
            electrode_df = electrode_df[~electrode_df['channel_name'].isin(bad_ch_ids)]

        n_electrodes = len(electrode_df)
        print(f"Number of good electrodes: {n_electrodes}")
        
        # Add electrodes to NWB file
        for idx, row in electrode_df.iterrows():
            nwbfile.add_electrode(
                group=electrode_group,
                label=f"shank{ishank}:{row['channel_name']}",
                location=electrode_location,
                rel_x=float(row['x']),
                rel_y=float(row['y']),
                imp=float(row['impedance']) if not np.isnan(row['impedance']) else 0.0,
            )

        electrode_table_region = nwbfile.create_electrode_table_region(
            list(range(n_electrodes)), "all electrodes"
        )

        # Get recording info
        print("Getting recording information...")
        recording, sampling_freq, num_frames, conversion, offset = self._get_recording_info(
            data_file, selected_geom)
        
        # Determine channel IDs to use
        if impedance_path is not None:
            good_channel_ids = electrode_df['channel_name'].tolist()
        else:
            if self.recording_method == 'intan':
                good_channel_ids = electrode_df['channel_name'].tolist()
            else:
                good_channel_ids = electrode_df['channel_index'].tolist()

        # Check if chunked processing is needed
        use_chunked = self._should_use_chunked_processing(num_frames, n_electrodes)
        estimated_size = self._estimate_file_size_gb(num_frames, n_electrodes)
        
        # Force chunking if multiple files need to be appended
        if has_multiple_files and not use_chunked:
            print("Multiple files detected - enabling chunked dataset for appending")
            use_chunked = True
        
        print(f"File size: ~{estimated_size:.2f} GB")
        print(f"Duration: {num_frames / sampling_freq:.2f} seconds")
        print(f"Processing mode: {'Chunked' if use_chunked else 'Direct'}")

        # Read and write electrical data
        print("Adding electrical data...")
        
        if use_chunked:
            # Process in chunks
            chunk_frames = int(self.chunk_duration * sampling_freq)
            num_chunks = int(np.ceil(num_frames / chunk_frames))
            
            print(f"Processing {num_chunks} chunks of {self.chunk_duration}s each...")
            
            # Read first chunk to initialize
            first_chunk = self._read_recording_chunk(
                recording, good_channel_ids, 0, min(chunk_frames, num_frames))
            
            electrical_series = ElectricalSeries(
                name="ElectricalSeries",
                data=H5DataIO(data=first_chunk, maxshape=(None, first_chunk.shape[1]),
                            compression='gzip', compression_opts=4, chunks=True),
                electrodes=electrode_table_region,
                starting_time=0.0,
                rate=sampling_freq,
                conversion=conversion,
                offset=offset,
            )
            
            nwbfile.add_acquisition(electrical_series)
            
            # Write initial NWB file
            print(f"Writing chunk 1/{num_chunks}...")
            with NWBHDF5IO(nwb_path, "w") as io:
                io.write(nwbfile)
            
            # Append remaining chunks
            for i in range(1, num_chunks):
                start_frame = i * chunk_frames
                end_frame = min((i + 1) * chunk_frames, num_frames)
                
                print(f"Processing chunk {i+1}/{num_chunks} (frames {start_frame}-{end_frame})...")
                chunk_data = self._read_recording_chunk(
                    recording, good_channel_ids, start_frame, end_frame)
                
                # Append to NWB file
                with NWBHDF5IO(nwb_path, "a") as io:
                    nwb_obj = io.read()
                    self._append_nwb_dset(
                        nwb_obj.acquisition['ElectricalSeries'].data, chunk_data, 0)
                    io.write(nwb_obj)
                
                # Clean up
                del chunk_data
        else:
            # Direct processing for smaller files (but still needs chunking if multiple files)
            trace = self._read_recording_chunk(recording, good_channel_ids)
            
            electrical_series = ElectricalSeries(
                name="ElectricalSeries",
                data=H5DataIO(data=trace, maxshape=(None, trace.shape[1]), chunks=True),
                electrodes=electrode_table_region,
                starting_time=0.0,
                rate=sampling_freq,
                conversion=conversion,
                offset=offset,
            )
            
            nwbfile.add_acquisition(electrical_series)
            
            print("Writing NWB file...")
            with NWBHDF5IO(nwb_path, "w") as io:
                io.write(nwbfile)
                
        # Handle digital input for Intan
        if self.recording_method == 'intan':
            stream_ids = self.get_stream_ids(data_file)
            if '4' in stream_ids:
                print("Found digital input channels...")
                # TODO: Implement digital input handling if needed
                pass

        return good_channel_ids

    def _append_nwb_dset(self, dset, data_to_append, append_axis: int) -> None:
        """
        Append data along a specified axis in an HDF5 dataset.
        """
        dset_shape = dset.shape
        dset_len = dset_shape[append_axis]
        app_len = data_to_append.shape[append_axis]
        new_len = dset_len + app_len

        # Prepare slicer to index the appended region
        slicer = [slice(None)] * len(dset_shape)
        slicer[append_axis] = slice(-app_len, None)

        dset.resize(new_len, axis=append_axis)
        dset[tuple(slicer)] = data_to_append

    def append_nwb(self, nwb_path: Path, data_file: Path, channel_ids: list = None,
                   metadata: dict = None) -> None:
        """
        Append additional recording data to an existing NWB file with optimized memory handling.
        """
        metadata = metadata or {}
        selected_geom = metadata.get("selected_geom", None)
        
        # Get recording info
        recording, sampling_freq, num_frames, conversion, offset, timestamps = self._get_recording_info(
            data_file, selected_geom)
        
        # Check if chunked processing is needed
        n_channels = len(channel_ids) if channel_ids else recording.get_num_channels()
        use_chunked = self._should_use_chunked_processing(num_frames, n_channels)
        estimated_size = self._estimate_file_size_gb(num_frames, n_channels)
        
        print(f"Appending file size: ~{estimated_size:.2f} GB")
        print(f"Processing mode: {'Chunked' if use_chunked else 'Direct'}")
        
        if use_chunked:
            # Process in chunks
            chunk_frames = int(self.chunk_duration * sampling_freq)
            num_chunks = int(np.ceil(num_frames / chunk_frames))
            
            print(f"Appending {num_chunks} chunks...")
            
            for i in range(num_chunks):
                start_frame = i * chunk_frames
                end_frame = min((i + 1) * chunk_frames, num_frames)
                
                print(f"Appending chunk {i+1}/{num_chunks} (frames {start_frame}-{end_frame})...")
                chunk_data = self._read_recording_chunk(
                    recording, channel_ids, start_frame, end_frame)
                
                with NWBHDF5IO(nwb_path, "a") as io:
                    nwb_obj = io.read()
                    self._append_nwb_dset(
                        nwb_obj.acquisition['ElectricalSeries'].data, chunk_data, 0)
                    io.write(nwb_obj)
                
                del chunk_data
        else:
            # Direct processing
            trace = self._read_recording_chunk(recording, channel_ids)
            with NWBHDF5IO(nwb_path, "a") as io:
                nwb_obj = io.read()
                self._append_nwb_dset(
                    nwb_obj.acquisition['ElectricalSeries'].data, trace, 0)
                io.write(nwb_obj)

    def get_data_files(self, data_folder: Path) -> list:
        """Get list of data files based on recording method."""
        if self.recording_method == 'intan':
            data_files = sorted(
                p for p in data_folder.iterdir()
                if p.suffix.lower() in ('.rhd', '.rhs') and not p.name.startswith("._"))
        else:  # spikegadget / mountainsort
            ms_folders = list(data_folder.glob('*.mountainsort'))
            if not ms_folders:
                raise FileNotFoundError("No .mountainsort folders found in the specified folder.")
            
            # Base folder (no .part) → (0,0); .partN → (1,N)
            def part_key(f: Path):
                m = re.search(r'\.part(\d+)\.mountainsort$', f.name)
                return (1, int(m.group(1))) if m else (0, 0)
            
            ms_folders.sort(key=part_key)
            
            data_files = []
            for folder in ms_folders:
                group_files = list(folder.glob('*group0.mda'))
                data_files.extend(group_files)
        
        if not data_files:
            file_types = ".rhd/.rhs" if self.recording_method == 'intan' else "group0.mda"
            raise FileNotFoundError(f"No {file_types} files found in the specified folder.")
        
        return data_files

    def get_session_description(self, data_folder: Path) -> str:
        """Extract session description from folder path."""
        if self.recording_method == 'spikegadget':
            # For SpikeGadgets: extract from .rec folder
            folder_str = str(data_folder)
            pattern = r'[\\\/]([^\\\/]+)\.rec$'
            match = re.search(pattern, folder_str)
            if match:
                return match.group(1)
            return data_folder.stem
        else:
            # For Intan: use folder name
            return data_folder.name


def load_bad_ch(bad_file: Path) -> list:
    """
    Load bad channels from a file.
    """
    if not bad_file.exists():
        print(f"No bad channels file found at {bad_file}. Using all channels.")
        return []
    with open(bad_file, "r") as f:
        bad_channels = [line.strip() for line in f.readlines()]
    return bad_channels

def get_geom_files(geom_folder: Path) -> list:
    """Get list of available geom.csv files in the geom folder."""
    if not geom_folder.exists():
        return []
    geom_files = sorted(geom_folder.glob("*.csv"))
    return geom_files


def main():
    """Main function to run the unified converter."""
    # Choose recording method
    print("Choose recording method:")
    print("1. Intan")
    print("2. SpikeGadgets")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        recording_method = 'intan'
        folder_prompt = "Please enter the full path to the Intan data folder: "
    elif choice == '2':
        recording_method = 'spikegadget'
        folder_prompt = "Please enter the full path to the .rec folder: "
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Get chunk duration for large files
    chunk_duration_input = input("Enter chunk duration in seconds for large files (default: 60): ").strip()
    chunk_duration = float(chunk_duration_input) if chunk_duration_input else 60.0

    # Initialize converter
    converter = EphysToNWBConverter(recording_method, chunk_duration=chunk_duration)
    
    # Get inputs
    data_folder = Path(input(folder_prompt).strip().strip("'").strip('"'))
    impedance_file_input = input("Please enter the full path to the impedance file (optional, press Enter to skip): ").strip().strip("'").strip('"')
    impedance_file = Path(impedance_file_input) if impedance_file_input else None
    electrode_location = input("Please enter the electrode location: ").strip()
    exp_desc = input("Please enter the experiment description: ").strip() or "None"
    
    # Get device type
    animal_id = get_animal_id(data_folder)
    device_type = get_or_set_device_type(animal_id)

    # GEOM FILE SELECTION (only for SpikeGadgets)
    selected_geom = None
    if recording_method == 'spikegadget':
        script_dir = Path(__file__).resolve().parent
        geom_folder = script_dir / "geom"
        geom_files = get_geom_files(geom_folder)
        
        if not geom_files:
            print(f"Warning: No .csv files found in {geom_folder}. Will use default geom.csv if available.")
        else:
            print("\nAvailable geom files:")
            for idx, gfile in enumerate(geom_files, 1):
                print(f"{idx}. {gfile.name}")
            
            geom_choice = input("Select geom file (enter number): ").strip()
            try:
                geom_idx = int(geom_choice) - 1
                if 0 <= geom_idx < len(geom_files):
                    selected_geom = geom_files[geom_idx]
                    print(f"Selected: {selected_geom.name}")
                else:
                    print("Invalid selection. Using default geom.csv if available.")
            except ValueError:
                print("Invalid input. Using default geom.csv if available.")

    raw = input("Please enter the shank numbers (e.g. 0,1,2,3 or [0,1,2,3]): ")    
    shanks = [int(x) for x in re.findall(r'\d+', raw)]
    print(f"Processing shanks: {shanks}")
    
    session_description = converter.get_session_description(data_folder)
    
    if not data_folder.exists():
        print(f"Folder {data_folder} does not exist, exiting")
        sys.exit(1)

    # Get data files
    data_files = converter.get_data_files(data_folder)
    first_file = data_files[0]
    print(f"all data files: {data_files}")

    # Load bad channels
    bad_file = data_folder / "bad_channels.txt"
    bad_ch_ids = load_bad_ch(bad_file)

    # Process each shank
    for ish in shanks:
        nwb_path = data_folder / f"{session_description}sh{ish}.nwb"
        print(f"\n{'='*60}")
        print(f"Creating NWB file {nwb_path.name}")
        print(f"{'='*60}")

        # Create the NWB from first file
        good_ch = converter.initiate_nwb(
            first_file, nwb_path, ishank=ish,
            impedance_path=impedance_file,
            bad_ch_ids=bad_ch_ids,
            metadata={
                'device_type': device_type,
                'session_desc': session_description,
                'n_channels_per_shank': 32,
                'electrode_location': electrode_location,
                'exp_desc': exp_desc,
                'selected_geom': selected_geom if recording_method == 'spikegadget' else None
            },
            has_multiple_files=(len(data_files) > 1)  # Add this flag
        )

        if len(data_files) == 1:
            print(f"Only one file ({first_file.name}) found, skipping appending.")
            continue

        # Append the rest
        for idx, f in enumerate(data_files[1:], 2):
            print(f"\n{'-'*60}")
            print(f"Appending file {idx}/{len(data_files)}: {f.name}")
            print(f"{'-'*60}")
            converter.append_nwb(
                nwb_path, f,
                channel_ids=good_ch,
                metadata={'device_type': device_type,
                          'selected_geom': selected_geom if recording_method == 'spikegadget' else None}
            )

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()