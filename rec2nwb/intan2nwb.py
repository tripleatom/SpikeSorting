import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import neo.rawio
from pynwb import NWBHDF5IO

from rec2nwb.preproc_func import get_or_set_device_type, get_animal_id
from rec2nwb.utils.file_io import get_data_files, load_bad_ch
from rec2nwb.utils.electrode import (
    get_ch_index_on_shank, build_electrode_df, resolve_good_channel_ids,
)
from rec2nwb.utils.nwb_helpers import (
    make_nwbfile, add_electrodes_to_nwb, make_electrical_series,
    append_nwb_dset, append_chunk_to_nwb,
)

HOUSTON_TZ = ZoneInfo("America/Chicago")


class EphysToNWBConverter:
    """Converts Intan recordings to NWB format."""

    VALID_METHODS = ('intan',)

    def __init__(self, recording_method: str, chunk_duration: float = 60.0,
                 parallelShank: bool = False):
        """
        Args:
            recording_method: 'intan'.
            chunk_duration: Seconds per chunk for large-file processing (default 60 s).
            parallelShank: If True, all shanks share one get_traces() call per chunk.
        """
        if recording_method not in self.VALID_METHODS:
            raise ValueError(f"recording_method must be one of {self.VALID_METHODS}")
        self.recording_method = recording_method
        self.chunk_duration = chunk_duration
        self.parallelShank = parallelShank

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_data_files(self, data_folder: Path) -> list:
        return get_data_files(data_folder, self.recording_method)

    def get_session_description(self, data_folder: Path) -> str:
        return data_folder.name

    def get_stream_ids(self, file_path: Path):
        """Return Intan stream IDs."""
        reader = neo.rawio.IntanRawIO(filename=str(file_path))
        reader.parse_header()
        return reader.header['signal_streams']['id']

    def get_timestamp(self, file_path: Path) -> datetime:
        """Parse recording start datetime from the filename as Houston local time."""
        m = re.match(r"[a-zA-Z0-9_]+_([0-9]+_[0-9]+)\.rh(?:s|d)", file_path.name)
        if m:
            dt = datetime.strptime(m.group(1), "%y%m%d_%H%M%S")
            return dt.replace(tzinfo=HOUSTON_TZ)
        raise ValueError(f"Cannot parse timestamp from filename: {file_path.name}")

    # ------------------------------------------------------------------
    # Recording I/O
    # ------------------------------------------------------------------

    def _get_recording_info(self, data_file: Path):
        """
        Open the recording file and return metadata without loading all data.

        Returns:
            (recording, sampling_freq, num_frames, conversion_V, offset_V)
        """
        recording = se.read_intan(data_file, stream_id='0')
        conversion = recording.get_channel_gains()[0] / 1e6
        offset = recording.get_channel_offsets()[0] / 1e6
        sampling_freq = recording.get_sampling_frequency()
        return recording, sampling_freq, recording.get_num_frames(), conversion, offset

    def _read_chunk(self, recording, channel_ids: list = None,
                    start_frame: int = 0, end_frame: int = None):
        """Read a slice of recording data."""
        if end_frame is None:
            end_frame = recording.get_num_frames()
        kwargs = dict(start_frame=start_frame, end_frame=end_frame)
        if channel_ids:
            kwargs['channel_ids'] = channel_ids
        return recording.get_traces(**kwargs)

    def _chunk_generator(self, recording, num_frames: int, channel_ids: list, file_label: str):
        """Yield data chunks for the full recording duration."""
        chunk_frames = int(self.chunk_duration * self._last_sampling_freq)
        n_chunks = int(np.ceil(num_frames / chunk_frames))
        for i in range(n_chunks):
            start = i * chunk_frames
            end = min((i + 1) * chunk_frames, num_frames)
            print(f"  Chunk {i+1}/{n_chunks} (frames {start}-{end}) — {file_label}")
            yield self._read_chunk(recording, channel_ids, start, end)

    # ------------------------------------------------------------------
    # Size / chunking helpers
    # ------------------------------------------------------------------

    def _estimate_gb(self, num_frames: int, num_channels: int, dtype_size: int = 2) -> float:
        return num_frames * num_channels * dtype_size / (1024 ** 3)

    def _needs_chunking(self, num_frames: int, num_channels: int,
                        threshold_gb: float = 3.0) -> bool:
        return self._estimate_gb(num_frames, num_channels) > threshold_gb

    # ------------------------------------------------------------------
    # Single-shank NWB creation + append
    # ------------------------------------------------------------------

    def initiate_nwb(self, data_file: Path, nwb_path: Path, ishank: int = 0,
                     impedance_path: str = None, bad_ch_ids: list = None,
                     metadata: dict = None, has_multiple_files: bool = False) -> list:
        """
        Write a new NWB file for one shank from the first data file.

        Returns:
            good_channel_ids used (needed for subsequent append_nwb calls).
        """
        metadata = metadata or {}
        device_type = metadata.get("device_type", "4shank16intan")
        electrode_location = metadata.get("electrode_location", None)

        print("Initiating NWB file...")
        session_start_time = self.get_timestamp(data_file)

        # --- Electrode table ---
        channel_index, xcoord, ycoord = get_ch_index_on_shank(ishank, device_type)
        impedance_table = pd.read_csv(impedance_path) if impedance_path else None
        electrode_df = build_electrode_df(
            channel_index, xcoord, ycoord,
            self.recording_method, impedance_table, bad_ch_ids,
        )
        print(f"Good electrodes on shank {ishank}: {len(electrode_df)}")

        # --- Recording info ---
        print("Getting recording information...")
        recording, sampling_freq, num_frames, conversion, offset = \
            self._get_recording_info(data_file)
        self._last_sampling_freq = sampling_freq

        good_channel_ids = resolve_good_channel_ids(
            electrode_df, self.recording_method,
            has_impedance=(impedance_table is not None),
            actual_channel_ids=recording.get_channel_ids(),
        )

        # --- Chunking decision ---
        use_chunked = self._needs_chunking(num_frames, len(electrode_df))
        if has_multiple_files and not use_chunked:
            print("Multiple files detected — enabling chunked dataset for appending.")
            use_chunked = True

        est_gb = self._estimate_gb(num_frames, len(electrode_df))
        print(f"File size: ~{est_gb:.2f} GB | Duration: {num_frames/sampling_freq:.1f}s "
              f"| Mode: {'Chunked' if use_chunked else 'Direct'}")

        # --- Build NWB shell ---
        nwbfile = make_nwbfile(session_start_time, metadata)
        electrode_table_region = add_electrodes_to_nwb(
            nwbfile, electrode_df, ishank, electrode_location)

        # --- Write data ---
        print("Adding electrical data...")
        if use_chunked:
            chunk_frames = int(self.chunk_duration * sampling_freq)
            num_chunks = int(np.ceil(num_frames / chunk_frames))
            print(f"Processing {num_chunks} chunk(s) of {self.chunk_duration}s...")

            first_chunk = self._read_chunk(
                recording, good_channel_ids, 0, min(chunk_frames, num_frames))
            nwbfile.add_acquisition(
                make_electrical_series(first_chunk, electrode_table_region,
                                       sampling_freq, conversion, offset, chunked=True))

            print(f"Writing chunk 1/{num_chunks}...")
            with NWBHDF5IO(nwb_path, "w") as io:
                io.write(nwbfile)
            del first_chunk

            for i in range(1, num_chunks):
                start = i * chunk_frames
                end = min((i + 1) * chunk_frames, num_frames)
                print(f"Processing chunk {i+1}/{num_chunks} (frames {start}-{end})...")
                chunk = self._read_chunk(recording, good_channel_ids, start, end)
                append_chunk_to_nwb(nwb_path, chunk)
                del chunk
        else:
            trace = self._read_chunk(recording, good_channel_ids)
            nwbfile.add_acquisition(
                make_electrical_series(trace, electrode_table_region,
                                       sampling_freq, conversion, offset, chunked=False))
            print("Writing NWB file...")
            with NWBHDF5IO(nwb_path, "w") as io:
                io.write(nwbfile)

        # Digital input
        stream_ids = self.get_stream_ids(data_file)
        if stream_ids is not None and '4' in stream_ids:
            print("Found digital input channels (not yet implemented).")

        return good_channel_ids

    def append_nwb(self, nwb_path: Path, data_file: Path,
                   channel_ids: list = None, metadata: dict = None) -> None:
        """Append one more data file to an existing NWB file."""
        recording, sampling_freq, num_frames, _, _ = self._get_recording_info(data_file)
        self._last_sampling_freq = sampling_freq

        n_channels = len(channel_ids) if channel_ids else recording.get_num_channels()
        use_chunked = self._needs_chunking(num_frames, n_channels)
        est_gb = self._estimate_gb(num_frames, n_channels)
        print(f"Appending ~{est_gb:.2f} GB | Mode: {'Chunked' if use_chunked else 'Direct'}")

        if use_chunked:
            chunk_frames = int(self.chunk_duration * sampling_freq)
            n_chunks = int(np.ceil(num_frames / chunk_frames))
            print(f"Appending {n_chunks} chunk(s)...")
            for i in range(n_chunks):
                start = i * chunk_frames
                end = min((i + 1) * chunk_frames, num_frames)
                print(f"  Chunk {i+1}/{n_chunks} (frames {start}-{end})...")
                chunk = self._read_chunk(recording, channel_ids, start, end)
                append_chunk_to_nwb(nwb_path, chunk)
                del chunk
        else:
            trace = self._read_chunk(recording, channel_ids)
            append_chunk_to_nwb(nwb_path, trace)

    # ------------------------------------------------------------------
    # Multi-shank parallel conversion
    # ------------------------------------------------------------------

    def convert_all_shanks(self, data_files: list, shanks: list,
                            session_description: str, data_folder: Path,
                            impedance_path: Path = None, bad_ch_ids: list = None,
                            metadata: dict = None) -> dict:
        """
        Convert all shanks simultaneously. One get_traces() call per chunk covers
        all shanks, then slices are written to per-shank NWB files.
        Only used when parallelShank=True.

        Returns:
            {shank_index: good_channel_ids}
        """
        metadata = metadata or {}
        device_type = metadata.get("device_type", "4shank16intan")
        electrode_location = metadata.get("electrode_location", None)
        first_file = data_files[0]

        # --- Per-shank electrode setup ---
        impedance_table = pd.read_csv(impedance_path) if impedance_path else None
        shank_setups = {}
        for ish in shanks:
            channel_index, xcoord, ycoord = get_ch_index_on_shank(ish, device_type)
            electrode_df = build_electrode_df(
                channel_index, xcoord, ycoord,
                self.recording_method, impedance_table, bad_ch_ids,
            )
            good_ids = resolve_good_channel_ids(
                electrode_df, self.recording_method,
                has_impedance=(impedance_table is not None),
            )
            print(f"Shank {ish}: {len(electrode_df)} good electrodes")
            shank_setups[ish] = {
                'electrode_df': electrode_df,
                'good_channel_ids': good_ids,
                'nwb_path': data_folder / f"{session_description}sh{ish}.nwb",
            }

        # --- Combined channel list (union, order-preserving) ---
        seen, all_channel_ids = set(), []
        for ish in shanks:
            for ch in shank_setups[ish]['good_channel_ids']:
                if str(ch) not in seen:
                    all_channel_ids.append(ch)
                    seen.add(str(ch))

        # Map each shank's channels to column positions in the combined array
        ch_to_col = {str(ch): i for i, ch in enumerate(all_channel_ids)}
        for ish in shanks:
            shank_setups[ish]['col_indices'] = [
                ch_to_col[str(ch)] for ch in shank_setups[ish]['good_channel_ids']
            ]

        # --- Recording info (once, from first file) ---
        print("\nGetting recording information...")
        recording, sampling_freq, num_frames, conversion, offset = \
            self._get_recording_info(first_file)
        self._last_sampling_freq = sampling_freq

        chunk_frames = int(self.chunk_duration * sampling_freq)
        est_gb = self._estimate_gb(num_frames, len(all_channel_ids))
        n_chunks = int(np.ceil(num_frames / chunk_frames))
        print(f"Total channels: {len(all_channel_ids)} | "
              f"Duration: {num_frames/sampling_freq:.1f}s | ~{est_gb:.2f} GB")
        print(f"Processing {n_chunks} chunk(s) of {self.chunk_duration}s each")

        session_start_time = self.get_timestamp(first_file)

        def _write_initial_nwb(chunk_all, file_index, total_files):
            print(f"\nWriting initial NWB for all shanks (file {file_index}/{total_files})...")
            for ish in shanks:
                setup = shank_setups[ish]
                chunk_sh = chunk_all[:, setup['col_indices']]
                nwbfile = make_nwbfile(session_start_time, metadata)
                electrode_table_region = add_electrodes_to_nwb(
                    nwbfile, setup['electrode_df'], ish, electrode_location)
                nwbfile.add_acquisition(
                    make_electrical_series(chunk_sh, electrode_table_region,
                                           sampling_freq, conversion, offset, chunked=True))
                print(f"  Writing shank {ish} NWB...")
                with NWBHDF5IO(setup['nwb_path'], "w") as io:
                    io.write(nwbfile)

        def _append_to_all_shanks(chunk_all):
            for ish in shanks:
                setup = shank_setups[ish]
                append_chunk_to_nwb(setup['nwb_path'], chunk_all[:, setup['col_indices']])

        def _iter_chunks(rec, n_frames, label):
            n = int(np.ceil(n_frames / chunk_frames))
            for i in range(n):
                start = i * chunk_frames
                end = min((i + 1) * chunk_frames, n_frames)
                print(f"  Chunk {i+1}/{n} (frames {start}-{end}) — {label}")
                chunk = rec.get_traces(channel_ids=all_channel_ids,
                                       start_frame=start, end_frame=end)
                yield chunk
                del chunk

        # First file: write shells then append remaining chunks
        gen = _iter_chunks(recording, num_frames, first_file.name)
        first_chunk = next(gen)
        _write_initial_nwb(first_chunk, file_index=1, total_files=len(data_files))
        del first_chunk
        for chunk in gen:
            _append_to_all_shanks(chunk)
        print(f"\nFile 1/{len(data_files)} ({first_file.name}) done.")

        # Additional files
        for file_idx, f in enumerate(data_files[1:], 2):
            print(f"\n{'='*60}\nProcessing file {file_idx}/{len(data_files)}: {f.name}\n{'='*60}")
            t0 = time.time()
            rec2, _, num_frames2, _, _ = self._get_recording_info(f)
            for chunk in _iter_chunks(rec2, num_frames2, f.name):
                _append_to_all_shanks(chunk)
            print(f"File {file_idx}/{len(data_files)} done in {time.time()-t0:.1f}s")

        return {ish: shank_setups[ish]['good_channel_ids'] for ish in shanks}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    recording_method = 'intan'
    converter = EphysToNWBConverter(
        recording_method,
        chunk_duration=float(input("Chunk duration in seconds (default 60): ").strip() or 60.0),
        parallelShank=input("Process all shanks in parallel per chunk? [y/N]: ").strip().lower() == 'y',
    )

    data_folder = Path(input("Path to the Intan data folder: ").strip().strip("'\""))
    if not data_folder.exists():
        print(f"Folder {data_folder} does not exist, exiting.")
        sys.exit(1)

    impedance_input = input("Path to impedance file (optional, Enter to skip): ").strip().strip("'\"")
    impedance_file = Path(impedance_input) if impedance_input else None
    electrode_location = input("Electrode location: ").strip()
    exp_desc = input("Experiment description: ").strip() or "None"

    animal_id = get_animal_id(data_folder)
    device_type = get_or_set_device_type(animal_id)

    raw = input("Shank numbers (e.g. 0,1,2,3 or [0,1,2,3]): ")
    shanks = [int(x) for x in re.findall(r'\d+', raw)]
    print(f"Processing shanks: {shanks}")

    # Discover and log data files
    data_files = converter.get_data_files(data_folder)
    first_file = data_files[0]
    session_description = converter.get_session_description(data_folder)

    print(f"\n{'='*60}")
    print(f"Conversion order ({len(data_files)} file(s)):")
    print(f"{'='*60}")
    log_lines = [f"Conversion order ({len(data_files)} file(s)):"]
    for i, f in enumerate(data_files, 1):
        try:
            rec = se.read_intan(str(f), stream_id='0')
            line = f"  {i:2d}. {f.name}: {rec.get_num_frames()} timestamps"
        except Exception as e:
            line = f"  {i:2d}. {f.name}: ERROR — {e}"
        print(line)
        log_lines.append(line)
    print(f"{'='*60}\n")

    log_path = data_folder / "conversion_list.txt"
    log_path.write_text('\n'.join(log_lines) + '\n')
    print(f"Conversion list saved to: {log_path}\n")

    bad_ch_ids = load_bad_ch(data_folder / "bad_channels.txt")

    shared_metadata = {
        'device_type': device_type,
        'session_desc': session_description,
        'electrode_location': electrode_location,
        'exp_desc': exp_desc,
    }

    t0 = time.time()
    if converter.parallelShank:
        print("\nMode: parallelShank=True — processing all shanks simultaneously")
        converter.convert_all_shanks(
            data_files=data_files, shanks=shanks,
            session_description=session_description,
            data_folder=data_folder, impedance_path=impedance_file,
            bad_ch_ids=bad_ch_ids, metadata=shared_metadata,
        )
    else:
        print("\nMode: parallelShank=False — processing shanks sequentially")
        for ish in shanks:
            nwb_path = data_folder / f"{session_description}sh{ish}.nwb"
            print(f"\n{'='*60}\nCreating {nwb_path.name}\n{'='*60}")

            good_ch = converter.initiate_nwb(
                first_file, nwb_path, ishank=ish,
                impedance_path=impedance_file, bad_ch_ids=bad_ch_ids,
                metadata=shared_metadata, has_multiple_files=(len(data_files) > 1),
            )

            if len(data_files) == 1:
                print(f"Single file — no appending needed.")
                continue

            for idx, f in enumerate(data_files[1:], 2):
                print(f"\n{'-'*60}\nAppending file {idx}/{len(data_files)}: {f.name}\n{'-'*60}")
                converter.append_nwb(nwb_path, f, channel_ids=good_ch,
                                     metadata=shared_metadata)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
