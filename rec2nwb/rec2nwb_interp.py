"""
Expected data structure for spikegadget_rec mode
=================================================

When converting multiple sessions, point data_folder at the parent directory
containing one or more session folders (each with a .rec extension):

    data_folder/
    ├── CnL42SG_passive_20260304_142720.rec/       <- session folder (date 1)
    │   ├── CnL42SG_passive_20260304_142720.rec    <- base file (no part number)
    │   ├── CnL42SG_passive_20260304_142720.part2.rec
    │   └── CnL42SG_passive_20260304_142720.part3.rec
    ├── CnL42SG_passive_20260305_091000.rec/       <- session folder (date 2)
    │   ├── CnL42SG_passive_20260305_091000.rec
    │   └── CnL42SG_passive_20260305_091000.part2.rec
    └── bad_channels.txt                           <- optional

All .rec files are collected across all session folders, then sorted by:
  1. Recording datetime (parsed from filename: _YYYYMMDD_HHMMSS)
  2. Part number (base file first, then part2, part3, ...)

The sorted files are concatenated and written into a single NWB file per shank.

Filename convention:
  <AnimalID>_<OptionalTag>_<YYYYMMDD>_<HHMMSS>[.partN].rec
  Example: CnL42SG_passive_20260304_142720.part2.rec
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from queue import Queue
import sys
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from pynwb import NWBHDF5IO
from scipy.interpolate import PchipInterpolator

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from rec2nwb.preproc_func import get_or_set_device_type, get_animal_id
from rec2nwb.process_func.DIO import get_dio_folders, extract_DIN
from rec2nwb.utils.file_io import (
    get_data_files, setup_spikegadget_files, get_sampling_rate_from_params,
    load_bad_ch,
)
from rec2nwb.utils.electrode import (
    get_ch_index_on_shank, build_electrode_df, resolve_good_channel_ids,
)
from tqdm import tqdm

from rec2nwb.utils.nwb_helpers import (
    make_nwbfile, add_electrodes_to_nwb, make_electrical_series,
    append_chunk_to_nwb, nwb_direct_writer,
)


def _prefetch_chunks(gen, prefetch: int = 1):
    """Wrap a chunk generator to read `prefetch` chunks ahead in a background thread.

    Lets disk reads of the next chunk overlap with HDF5 writes of the current one.
    """
    q = Queue(maxsize=prefetch + 1)
    _DONE = object()

    def _producer():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(_DONE)

    threading.Thread(target=_producer, daemon=True).start()
    while True:
        item = q.get()
        if item is _DONE:
            break
        yield item


class SpikeGadgetsRecToNWB:
    """Converts SpikeGadgets .rec recordings to NWB format."""

    def __init__(self, chunk_duration: float = None, parallelShank: bool = False):
        """
        Args:
            chunk_duration: Seconds per chunk.  None (default) = auto-size to
                            use ~30 % of available RAM (requires psutil), falling
                            back to 60 s if psutil is not installed.
            parallelShank: If True, all shanks share one get_traces() call per chunk.
        """
        self.recording_method = 'spikegadget_rec'
        self.chunk_duration = chunk_duration
        self.parallelShank = parallelShank

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_data_files(self, data_folder: Path) -> list:
        return get_data_files(data_folder, self.recording_method)

    def get_session_description(self, data_folder: Path) -> str:
        return data_folder.name

    def get_timestamp(self, file_path: Path) -> datetime:
        """Parse the recording start datetime from the filename."""
        m = re.search(r"_(\d{8})_(\d{6})", file_path.name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        raise ValueError(f"Cannot parse timestamp from filename: {file_path.name}")

    # ------------------------------------------------------------------
    # Recording I/O
    # ------------------------------------------------------------------

    def _get_recording_info(self, data_file: Path):
        """
        Open the .rec file and return metadata without loading all data.

        Returns:
            (recording, sampling_freq, num_frames, conversion_V, offset_V)
        """
        setup_spikegadget_files(data_file, self.recording_method)
        recording = se.read_spikegadgets(data_file)
        conversion, offset = 0.195 / 1e6, 0.0
        sampling_freq = get_sampling_rate_from_params()
        return recording, sampling_freq, recording.get_num_frames(), conversion, offset

    def _read_chunk(self, recording, channel_ids: list = None,
                    start_frame: int = 0, end_frame: int = None):
        """Read a slice of recording data. Channel IDs are cast to strings for .rec files."""
        if end_frame is None:
            end_frame = recording.get_num_frames()
        kwargs = dict(start_frame=start_frame, end_frame=end_frame)
        if channel_ids:
            kwargs['channel_ids'] = [str(ch) for ch in channel_ids]
        return recording.get_traces(**kwargs)

    def _iter_chunks(self, recording, num_frames: int, sampling_freq: float,
                     channel_ids: list, label: str):
        """Yield data chunks across the full recording duration."""
        n_ch = len(channel_ids) if channel_ids else recording.get_num_channels()
        chunk_frames = self._chunk_frames(sampling_freq, n_ch)
        n = int(np.ceil(num_frames / chunk_frames))
        tqdm.write(f"  {chunk_frames/sampling_freq:.0f}s/chunk "
                   f"({self._estimate_gb(chunk_frames, n_ch):.2f} GB) × {n}")
        for i in tqdm(range(n), desc=f"  {label}", unit="chunk", leave=True):
            start = i * chunk_frames
            end = min((i + 1) * chunk_frames, num_frames)
            yield self._read_chunk(recording, channel_ids, start, end)

    # ------------------------------------------------------------------
    # Size / memory helpers
    # ------------------------------------------------------------------

    def _estimate_gb(self, num_frames: int, num_channels: int, dtype_size: int = 2) -> float:
        return num_frames * num_channels * dtype_size / (1024 ** 3)

    def _chunk_frames(self, sampling_freq: float, num_channels: int) -> int:
        """Return chunk size in frames.

        If chunk_duration was set explicitly, honour it.  Otherwise use 30 % of
        available RAM (via psutil), clamped to [30 s, 600 s].
        """
        if self.chunk_duration is not None:
            return int(self.chunk_duration * sampling_freq)
        if _psutil is not None:
            avail = _psutil.virtual_memory().available
            # Reserve 20 % for OS + HDF5/gzip buffers + Python overhead.
            # The remaining 80 % is split equally between the chunk being written
            # and the one being prefetched (2 chunks in memory at peak).
            frames = int(avail * 0.80 / 2 / (num_channels * 2))
            lo, hi = int(30 * sampling_freq), int(600 * sampling_freq)
            return max(lo, min(frames, hi))
        return int(60.0 * sampling_freq)

    # ------------------------------------------------------------------
    # Gap / packet-loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gap_file(txt_path: Path) -> list:
        """Parse a sidecar .txt file for packet-loss gap records.

        Expected line format (any order, mixed with other lines):
            Interpolating data during gap of N points after timestamp T

        Returns:
            Sorted list of (abs_timestamp: int, n_missing: int).
            Empty list if the file does not exist or contains no gap lines.
        """
        txt_path = Path(txt_path)
        if not txt_path.exists():
            return []
        pattern = re.compile(r'gap of (\d+) points after timestamp (\d+)')
        gaps = []
        with open(txt_path, 'r') as fh:
            for line in fh:
                m = pattern.search(line)
                if m:
                    gaps.append((int(m.group(2)), int(m.group(1))))  # (timestamp, n_missing)
        return sorted(gaps)

    @staticmethod
    def _get_first_timestamp(rec_folder: Path) -> int:
        """Return the first hardware timestamp of the recording session.

        Reads Din1.dat from the first DIO folder inside *rec_folder*.
        All .rec parts in the same rec_folder share this origin timestamp.
        """
        rec_folder = Path(rec_folder)
        dio_folders = get_dio_folders(rec_folder)
        if not dio_folders:
            raise FileNotFoundError(f"No .DIO folder found in {rec_folder}")
        result = extract_DIN(dio_folders[0], channel_id=1)
        if result is None:
            raise FileNotFoundError(f"Din1.dat not found in {dio_folders[0]}")
        time_array, _ = result
        return int(time_array[0])

    def _pchip_fill(self, pre: np.ndarray, post, n_missing: int) -> np.ndarray:
        """PCHIP-interpolate *n_missing* frames between *pre* and *post* context.

        Args:
            pre:       Context frames before the gap, shape (ctx, C).
            post:      Context frames after the gap, shape (ctx, C), or None.
            n_missing: Number of frames to synthesize.

        Returns:
            Array of shape (n_missing, C) with same dtype as *pre*.
        """
        dtype = pre.dtype
        x_gap = np.arange(n_missing, dtype=float)

        if post is not None and len(post) > 0:
            x_known = np.concatenate([
                np.arange(-len(pre), 0, dtype=float),
                np.arange(n_missing, n_missing + len(post), dtype=float),
            ])
            y_known = np.concatenate([pre, post], axis=0).astype(float)
        else:
            x_known = np.arange(-len(pre), 0, dtype=float)
            y_known = pre.astype(float)

        # PchipInterpolator supports y of shape (n_points, n_channels): all channels
        # are fitted and evaluated in one vectorised call — no Python loop needed.
        info = np.iinfo(dtype)
        filled = PchipInterpolator(x_known, y_known)(x_gap)  # (n_missing, n_channels)
        return np.clip(np.round(filled), info.min, info.max).astype(dtype)

    def _precompute_gap_fills(self, recording, channel_ids: list, gaps: list,
                              num_frames: int, ctx: int = 10) -> dict:
        """Read context frames and PCHIP-fill all gaps in parallel.

        SciPy's PchipInterpolator releases the GIL, so threads genuinely run
        concurrently.  Each gap's context reads are also independent.

        Returns:
            {last_good_frame: filled_array (n_missing, C)}
        """
        def _fill_one(last_good, n_missing):
            pre = self._read_chunk(recording, channel_ids,
                                   max(0, last_good - ctx + 1), last_good + 1)
            post_start = last_good + 1
            post_end = min(num_frames, post_start + ctx)
            post = (self._read_chunk(recording, channel_ids, post_start, post_end)
                    if post_start < post_end else None)
            return last_good, self._pchip_fill(pre, post, n_missing)

        filled = {}
        with ThreadPoolExecutor(max_workers=len(gaps)) as ex:
            for last_good, arr in (f.result() for f in
                                   as_completed(ex.submit(_fill_one, lg, nm)
                                                for lg, nm in gaps)):
                filled[last_good] = arr
        return filled

    def _iter_chunks_with_gaps(self, recording, num_frames: int, sampling_freq: float,
                               channel_ids: list, label: str, gaps: list,
                               ctx: int = 10):
        """Like _iter_chunks but inserts PCHIP-interpolated frames at each gap.

        All gap fills are pre-computed in parallel before the chunk loop starts,
        so the streaming loop never blocks on interpolation.

        Args:
            gaps: List of (last_good_src_frame, n_missing) sorted ascending.
            ctx:  Context samples on each side of a gap for PCHIP.
        """
        if not gaps:
            yield from self._iter_chunks(recording, num_frames, sampling_freq,
                                         channel_ids, label)
            return

        # Pre-compute every gap fill in parallel before streaming begins.
        precomputed = self._precompute_gap_fills(
            recording, channel_ids, gaps, num_frames, ctx)

        n_ch = len(channel_ids) if channel_ids else recording.get_num_channels()
        chunk_frames = self._chunk_frames(sampling_freq, n_ch)

        # Build ordered segment list
        segments = []
        src_pos = 0
        for last_good, n_missing in sorted(gaps):
            seg_end = last_good + 1
            if seg_end > src_pos:
                segments.append(('read', src_pos, seg_end))
            segments.append(('gap', last_good, n_missing))
            src_pos = seg_end
        if src_pos < num_frames:
            segments.append(('read', src_pos, num_frames))

        n_total = len(gaps) + sum(
            int(np.ceil((end - start) / chunk_frames))
            for kind, start, end in segments if kind == 'read'
        )
        tqdm.write(f"  {chunk_frames/sampling_freq:.0f}s/chunk "
                   f"({self._estimate_gb(chunk_frames, n_ch):.2f} GB) × ~{n_total} "
                   f"({len(gaps)} gap(s) pre-filled)")
        with tqdm(total=n_total, desc=f"  {label}", unit="chunk", leave=True) as pbar:
            for seg in segments:
                if seg[0] == 'read':
                    _, start, end = seg
                    pos = start
                    while pos < end:
                        chunk_end = min(pos + chunk_frames, end)
                        yield self._read_chunk(recording, channel_ids, pos, chunk_end)
                        pbar.update(1)
                        pos = chunk_end
                else:
                    yield precomputed[seg[1]]
                    pbar.update(1)

    # ------------------------------------------------------------------
    # Single-shank NWB creation + append
    # ------------------------------------------------------------------

    def initiate_nwb(self, data_file: Path, nwb_path: Path, ishank: int = 0,
                     impedance_path: str = None, bad_ch_ids: list = None,
                     metadata: dict = None, has_multiple_files: bool = False,
                     gaps: list = None) -> list:
        """
        Write a new NWB file for one shank from the first data file.

        Returns:
            good_channel_ids used (needed for subsequent append_nwb calls).
        """
        metadata = metadata or {}
        device_type = metadata.get("device_type", "4shank16")
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

        good_channel_ids = resolve_good_channel_ids(
            electrode_df, self.recording_method,
            has_impedance=(impedance_table is not None),
            actual_channel_ids=recording.get_channel_ids(),
        )

        # --- Chunking decision ---
        use_chunked = self._estimate_gb(num_frames, len(electrode_df)) > 3.0 or has_multiple_files
        est_gb = self._estimate_gb(num_frames, len(electrode_df))
        print(f"File size: ~{est_gb:.2f} GB | Duration: {num_frames/sampling_freq:.1f}s "
              f"| Mode: {'Chunked' if use_chunked else 'Direct'}")

        # --- Build NWB shell ---
        nwbfile = make_nwbfile(session_start_time, metadata)
        electrode_table_region = add_electrodes_to_nwb(
            nwbfile, electrode_df, ishank, electrode_location)

        # --- Write data ---
        print("Adding electrical data...")
        _chunk_iter = (self._iter_chunks_with_gaps if gaps else self._iter_chunks)
        _iter_kwargs = dict(gaps=gaps) if gaps else {}
        if use_chunked or gaps:
            chunk_gen = _chunk_iter(
                recording, num_frames, sampling_freq, good_channel_ids, data_file.name,
                **_iter_kwargs)
            first_chunk = next(chunk_gen)
            nwbfile.add_acquisition(
                make_electrical_series(first_chunk, electrode_table_region,
                                       sampling_freq, conversion, offset, chunked=True))
            with NWBHDF5IO(nwb_path, "w") as io:
                io.write(nwbfile)
            del first_chunk
            with nwb_direct_writer(nwb_path) as append_fn:
                for chunk in _prefetch_chunks(chunk_gen):
                    append_fn(chunk)
        else:
            trace = self._read_chunk(recording, good_channel_ids)
            nwbfile.add_acquisition(
                make_electrical_series(trace, electrode_table_region,
                                       sampling_freq, conversion, offset, chunked=False))
            print("Writing NWB file...")
            with NWBHDF5IO(nwb_path, "w") as io:
                io.write(nwbfile)

        return good_channel_ids

    def append_nwb(self, nwb_path: Path, data_file: Path,
                   channel_ids: list = None, gaps: list = None) -> None:
        """Append one more .rec file to an existing NWB file."""
        recording, sampling_freq, num_frames, _, _ = self._get_recording_info(data_file)
        n_channels = len(channel_ids) if channel_ids else recording.get_num_channels()
        print(f"Appending ~{self._estimate_gb(num_frames, n_channels):.2f} GB")
        _chunk_iter = (self._iter_chunks_with_gaps if gaps else self._iter_chunks)
        _iter_kwargs = dict(gaps=gaps) if gaps else {}
        with nwb_direct_writer(nwb_path) as append_fn:
            for chunk in _prefetch_chunks(
                    _chunk_iter(recording, num_frames, sampling_freq,
                                channel_ids, data_file.name, **_iter_kwargs)):
                append_fn(chunk)

    # ------------------------------------------------------------------
    # Multi-shank parallel conversion
    # ------------------------------------------------------------------

    def convert_all_shanks(self, data_files: list, shanks: list,
                            session_description: str, data_folder: Path,
                            impedance_path: Path = None, bad_ch_ids: list = None,
                            metadata: dict = None, file_gaps: list = None) -> dict:
        """
        Convert all shanks simultaneously. One get_traces() call per chunk covers
        all shanks, then slices are written to per-shank NWB files.
        Only used when parallelShank=True.

        Returns:
            {shank_index: good_channel_ids}
        """
        metadata = metadata or {}
        device_type = metadata.get("device_type", "4shank16")
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

        est_gb = self._estimate_gb(num_frames, len(all_channel_ids))
        print(f"Total channels: {len(all_channel_ids)} | "
              f"Duration: {num_frames/sampling_freq:.1f}s | ~{est_gb:.2f} GB")

        session_start_time = self.get_timestamp(first_file)

        def _write_initial_nwb(chunk_all, file_index):
            print(f"\nWriting initial NWB for all shanks (file {file_index}/{len(data_files)})...")
            for ish in shanks:
                setup = shank_setups[ish]
                nwbfile = make_nwbfile(session_start_time, metadata)
                electrode_table_region = add_electrodes_to_nwb(
                    nwbfile, setup['electrode_df'], ish, electrode_location)
                nwbfile.add_acquisition(
                    make_electrical_series(chunk_all[:, setup['col_indices']],
                                           electrode_table_region,
                                           sampling_freq, conversion, offset, chunked=True))
                print(f"  Writing shank {ish} NWB...")
                with NWBHDF5IO(setup['nwb_path'], "w") as io:
                    io.write(nwbfile)

        def _append_to_all_shanks(chunk_all):
            for ish in shanks:
                setup = shank_setups[ish]
                append_chunk_to_nwb(setup['nwb_path'], chunk_all[:, setup['col_indices']])

        # First file: write shells then append remaining chunks
        # Pass already-stringified IDs; _read_chunk handles str(str(x)) == str(x)
        gaps0 = (file_gaps[0] if file_gaps else None)
        _iter0 = self._iter_chunks_with_gaps if gaps0 else self._iter_chunks
        _kw0 = dict(gaps=gaps0) if gaps0 else {}
        t0_file1 = time.time()
        gen = _iter0(recording, num_frames, sampling_freq,
                     all_channel_ids, first_file.name, **_kw0)
        first_chunk = next(gen)
        _write_initial_nwb(first_chunk, file_index=1)
        del first_chunk
        with ExitStack() as stack:
            append_fns = {
                ish: stack.enter_context(nwb_direct_writer(shank_setups[ish]['nwb_path']))
                for ish in shanks
            }
            for chunk in _prefetch_chunks(gen):
                for ish in shanks:
                    append_fns[ish](chunk[:, shank_setups[ish]['col_indices']])
        print(f"\nFile 1/{len(data_files)} ({first_file.name}) done in {time.time()-t0_file1:.1f}s.")

        # Additional files
        for file_idx, f in enumerate(data_files[1:], 2):
            print(f"\n{'='*60}\nProcessing file {file_idx}/{len(data_files)}: {f.name}\n{'='*60}")
            t0 = time.time()
            rec2, sampling_freq2, num_frames2, _, _ = self._get_recording_info(f)
            gaps_k = (file_gaps[file_idx - 1] if file_gaps else None)
            _iter_k = self._iter_chunks_with_gaps if gaps_k else self._iter_chunks
            _kw_k = dict(gaps=gaps_k) if gaps_k else {}
            with ExitStack() as stack:
                append_fns = {
                    ish: stack.enter_context(nwb_direct_writer(shank_setups[ish]['nwb_path']))
                    for ish in shanks
                }
                for chunk in _prefetch_chunks(
                        _iter_k(rec2, num_frames2, sampling_freq2,
                                all_channel_ids, f.name, **_kw_k)):
                    for ish in shanks:
                        append_fns[ish](chunk[:, shank_setups[ish]['col_indices']])
            print(f"File {file_idx}/{len(data_files)} done in {time.time()-t0:.1f}s")

        return {ish: shank_setups[ish]['good_channel_ids'] for ish in shanks}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    chunk_input = input("Chunk duration in seconds (Enter for auto based on available RAM): ").strip()
    chunk_duration = float(chunk_input) if chunk_input else None

    parallel_shank = input("Process all shanks in parallel per chunk? [y/N]: ").strip().lower() == 'y'

    converter = SpikeGadgetsRecToNWB(chunk_duration=chunk_duration,
                                     parallelShank=parallel_shank)

    data_folder = Path(input("Path to the folder containing .rec files: ").strip().strip("'\""))
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
    part_frame_counts = []
    for i, f in enumerate(data_files, 1):
        try:
            rec = se.read_spikegadgets(str(f))
            nf = rec.get_num_frames()
            part_frame_counts.append(nf)
            line = f"  {i:2d}. {f.name}: {nf} timestamps"
        except Exception as e:
            part_frame_counts.append(0)
            line = f"  {i:2d}. {f.name}: ERROR — {e}"
        print(line)
        log_lines.append(line)
    print(f"{'='*60}\n")

    log_path = data_folder / "conversion_list.txt"
    log_path.write_text('\n'.join(log_lines) + '\n')
    print(f"Conversion list saved to: {log_path}\n")

    # --- Gap pre-computation from sidecar .txt files ---
    # Each session folder (f.parent) has its own DIO and its own first timestamp.
    # Track per-folder: first_timestamp and cumulative frame offset within that folder.
    _folder_first_ts = {}   # rec_folder -> first hardware timestamp (or None)
    _folder_cum = {}        # rec_folder -> cumulative frames seen so far in that folder

    file_gaps = []
    for f, nf in zip(data_files, part_frame_counts):
        rec_folder = f.parent

        if rec_folder not in _folder_first_ts:
            try:
                ts = converter._get_first_timestamp(rec_folder)
                print(f"First hardware timestamp for {rec_folder.name}: {ts}")
                _folder_first_ts[rec_folder] = ts
            except FileNotFoundError as e:
                print(f"WARNING: could not read DIO timestamp for {rec_folder.name} "
                      f"({e}). Gap interpolation disabled for this folder.")
                _folder_first_ts[rec_folder] = None
            _folder_cum[rec_folder] = 0

        first_timestamp = _folder_first_ts[rec_folder]
        cum_offset = _folder_cum[rec_folder]

        txt_path = f.parent / (f.name + '.txt')
        if first_timestamp is not None:
            raw_gaps = converter._parse_gap_file(txt_path)
            gaps = [
                (t - first_timestamp - cum_offset, n)
                for t, n in raw_gaps
                if 0 <= t - first_timestamp - cum_offset < nf
            ]
        else:
            gaps = []
        file_gaps.append(gaps)
        _folder_cum[rec_folder] = cum_offset + nf

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
            file_gaps=file_gaps,
        )
    else:
        print("\nMode: parallelShank=False — processing shanks sequentially")
        for ish in shanks:
            nwb_path = data_folder / f"{session_description}sh{ish}.nwb"
            print(f"\n{'='*60}\nCreating {nwb_path.name}\n{'='*60}")

            t_file = time.time()
            good_ch = converter.initiate_nwb(
                first_file, nwb_path, ishank=ish,
                impedance_path=impedance_file, bad_ch_ids=bad_ch_ids,
                metadata=shared_metadata, has_multiple_files=(len(data_files) > 1),
                gaps=file_gaps[0],
            )
            print(f"File 1/{len(data_files)} ({first_file.name}) done in {time.time()-t_file:.1f}s.")

            if len(data_files) == 1:
                print("Single file — no appending needed.")
                continue

            for idx, f in enumerate(data_files[1:], 2):
                print(f"\n{'-'*60}\nAppending file {idx}/{len(data_files)}: {f.name}\n{'-'*60}")
                t_file = time.time()
                converter.append_nwb(nwb_path, f, channel_ids=good_ch,
                                     gaps=file_gaps[idx - 1])
                print(f"File {idx}/{len(data_files)} ({f.name}) done in {time.time()-t_file:.1f}s.")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
