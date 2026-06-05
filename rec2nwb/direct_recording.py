"""
Build a SpikeInterface ``BaseRecording`` for one shank directly from
SpikeGadgets ``.rec`` parts, with no intermediate NWB file.

Mirrors what ``rec2nwb_interp.SpikeGadgetsRecToNWB`` writes into NWB, but
keeps everything lazy: get_traces() reads only the requested window, and
PCHIP-interpolated frames at packet-loss gaps are spliced in on demand.

Public entry point:
    build_sortable_recording(data_folder, shank, device_type, ...) -> BaseRecording
"""

from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from probeinterface import Probe
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from rec2nwb.rec2nwb_interp import SpikeGadgetsRecToNWB
from rec2nwb.utils.electrode import (
    build_electrode_df,
    get_ch_index_on_shank,
    resolve_good_channel_ids,
)
from rec2nwb.utils.file_io import (
    get_data_files,
    load_bad_ch,
    setup_spikegadget_files,
)


# ---------------------------------------------------------------------------
# Lazy PCHIP gap-fill wrapper
# ---------------------------------------------------------------------------

class _LazyGapInterpolatedSegment(BaseRecordingSegment):
    """Segment that injects pre-computed PCHIP fills at gap positions."""

    def __init__(self, parent, gaps, fills, n_out, n_src, channel_ids, dtype):
        super().__init__(sampling_frequency=parent.get_sampling_frequency())
        self._parent = parent
        self._channel_ids = np.asarray(channel_ids)
        self._dtype = dtype
        self._n_out = int(n_out)
        self._n_src = int(n_src)

        # Build a coordinate table mapping the synthetic output timeline to the
        # source recording. Each row: (out_start, out_end, kind, src_or_fill_id).
        #   kind 0 = read from parent[src_start:src_start + len]
        #   kind 1 = use precomputed fill array
        rows = []
        src_pos = 0
        for gi, (last_good, n_missing) in enumerate(sorted(gaps)):
            seg_end = last_good + 1
            if seg_end > src_pos:
                length = seg_end - src_pos
                rows.append((None, None, 0, src_pos, length))  # filled after offset accum
                src_pos = seg_end
            rows.append((None, None, 1, gi, int(n_missing)))
        if src_pos < self._n_src:
            rows.append((None, None, 0, src_pos, self._n_src - src_pos))

        out_pos = 0
        self._segments = []
        for _, _, kind, payload, length in rows:
            self._segments.append((out_pos, out_pos + length, kind, payload, length))
            out_pos += length
        assert out_pos == self._n_out, (out_pos, self._n_out)

        self._seg_starts = np.array([s[0] for s in self._segments], dtype=np.int64)
        self._fills = fills  # list[np.ndarray] (n_missing, n_channels)

    def get_num_samples(self):
        return self._n_out

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._n_out
        start_frame = int(start_frame)
        end_frame = int(end_frame)

        if channel_indices is None or (isinstance(channel_indices, slice)
                                       and channel_indices == slice(None)):
            ch_ids = self._channel_ids
            ch_local = slice(None)
        elif isinstance(channel_indices, slice):
            ch_ids = self._channel_ids[channel_indices]
            ch_local = channel_indices
        else:
            ch_idx = np.asarray(channel_indices)
            ch_ids = self._channel_ids[ch_idx]
            ch_local = ch_idx

        n_ch = len(ch_ids)
        out = np.empty((end_frame - start_frame, n_ch), dtype=self._dtype)

        # Find first segment overlapping the requested window via binary search
        first = int(np.searchsorted(self._seg_starts, start_frame, side='right') - 1)
        first = max(first, 0)

        for s_start, s_end, kind, payload, length in self._segments[first:]:
            if s_start >= end_frame:
                break
            if s_end <= start_frame:
                continue
            lo = max(s_start, start_frame)
            hi = min(s_end, end_frame)
            out_lo = lo - start_frame
            out_hi = hi - start_frame

            if kind == 0:
                src_start = payload + (lo - s_start)
                src_end = src_start + (hi - lo)
                data = self._parent.get_traces(
                    start_frame=src_start, end_frame=src_end,
                    channel_ids=list(ch_ids),
                )
                out[out_lo:out_hi] = data.astype(self._dtype, copy=False)
            else:
                fill = self._fills[payload]  # (n_missing, n_channels_full)
                local_lo = lo - s_start
                local_hi = hi - s_start
                out[out_lo:out_hi] = fill[local_lo:local_hi, ch_local].astype(
                    self._dtype, copy=False
                )

        return out


class LazyGapInterpolatedRecording(BaseRecording):
    """Wrap a SpikeGadgets recording, inserting PCHIP-filled frames at packet-loss gaps.

    Parameters
    ----------
    parent : BaseRecording
        Source recording (single segment).
    gaps : list[tuple[int, int]]
        ``(last_good_src_frame, n_missing)`` records, source-frame indexed.
    fills : list[np.ndarray]
        Per-gap precomputed fill arrays, shape ``(n_missing, n_parent_channels)``,
        ordered to match ``sorted(gaps)``.
    """

    def __init__(self, parent: BaseRecording, gaps, fills):
        channel_ids = parent.get_channel_ids()
        dtype = parent.get_dtype()
        super().__init__(
            sampling_frequency=parent.get_sampling_frequency(),
            channel_ids=channel_ids,
            dtype=dtype,
        )

        # Copy parent properties (gain/offset, locations if any, etc.)
        for key in parent.get_property_keys():
            val = parent.get_property(key)
            if val is not None:
                self.set_property(key, val)

        n_src = parent.get_num_frames()
        n_out = n_src + sum(int(n) for _, n in gaps)
        # Wrap the single segment (SpikeGadgets rec is always one segment).
        assert parent.get_num_segments() == 1
        self.add_recording_segment(
            _LazyGapInterpolatedSegment(
                parent=parent,
                gaps=gaps,
                fills=fills,
                n_out=n_out,
                n_src=n_src,
                channel_ids=channel_ids,
                dtype=dtype,
            )
        )

        self._kwargs = dict(
            parent=parent,
            gaps=list(gaps),
            fills=[f.tolist() for f in fills],
        )


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------

def _make_probe(electrode_df: pd.DataFrame, channel_id_strings: list[str]) -> Probe:
    """Build a probeinterface Probe matching the sliced recording.

    ``electrode_df`` has columns: channel_name, x, y, channel_index. Rows are
    already in the order we sliced the recording, so positions[i] corresponds
    to channel_id_strings[i].
    """
    positions = np.column_stack([
        electrode_df['x'].to_numpy(dtype=float),
        electrode_df['y'].to_numpy(dtype=float),
    ])
    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(
        positions=positions,
        shapes='circle',
        shape_params={'radius': 6.0},
    )
    probe.set_contact_ids(channel_id_strings)
    # device_channel_indices: which channel of the (already sliced) recording
    # each contact corresponds to. Since the recording was sliced to exactly
    # these channels in this order, it is just 0..N-1.
    probe.set_device_channel_indices(np.arange(len(channel_id_strings)))
    return probe


# ---------------------------------------------------------------------------
# Multi-part concatenation with optional gap interpolation
# ---------------------------------------------------------------------------

def _open_part(data_file: Path) -> BaseRecording:
    setup_spikegadget_files(data_file, 'spikegadget_rec')
    return se.read_spikegadgets(str(data_file))


def _wrap_with_gaps(parent: BaseRecording, data_file: Path,
                    converter: SpikeGadgetsRecToNWB,
                    first_timestamp: int, cum_offset: int,
                    ctx: int = 10) -> tuple[BaseRecording, int]:
    """Wrap a single-part recording in LazyGapInterpolatedRecording when needed.

    Returns (wrapped_recording_or_parent, advance_in_src_frames).
    The advance value is the *parent* (source) frame count, used to update the
    running cumulative offset for the next part in the same session folder.
    """
    txt_path = data_file.parent / (data_file.name + '.txt')
    raw_gaps = converter._parse_gap_file(txt_path)
    n_src = parent.get_num_frames()

    if first_timestamp is None or not raw_gaps:
        return parent, n_src

    gaps = [
        (t - first_timestamp - cum_offset, n)
        for t, n in raw_gaps
        if 0 <= t - first_timestamp - cum_offset < n_src
    ]
    if not gaps:
        return parent, n_src

    # Precompute PCHIP fills against the full-channel parent. This reuses the
    # existing helper verbatim; it returns {last_good: arr(n_missing, n_ch_all)}.
    channel_ids = list(parent.get_channel_ids())
    precomp = converter._precompute_gap_fills(
        parent, channel_ids, gaps, n_src, ctx=ctx,
    )
    gaps_sorted = sorted(gaps)
    fills = [precomp[lg] for lg, _ in gaps_sorted]

    print(f"  {data_file.name}: {len(gaps_sorted)} gap(s) pre-filled "
          f"({sum(n for _, n in gaps_sorted)} samples)")
    wrapped = LazyGapInterpolatedRecording(parent, gaps_sorted, fills)
    return wrapped, n_src


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_sortable_recording(
    data_folder: Path,
    shank: int,
    device_type: str,
    impedance_path: Path | None = None,
    bad_ch_ids: list | None = None,
    verbose: bool = True,
) -> BaseRecording:
    """Return a per-shank SpikeInterface recording sourced directly from .rec files.

    Equivalent to what ``rec2nwb_interp.process_folder`` writes into a per-shank
    NWB, minus the disk write: parts are discovered, concatenated, sliced to the
    shank's good channels, given probe geometry, and (when sidecar .txt files
    report packet-loss gaps) wrapped in a lazy PCHIP-interpolation layer.

    Parameters
    ----------
    data_folder : Path
        Folder containing one or more ``*.rec`` session folders (matches the
        layout consumed by ``rec2nwb_interp.process_folder``).
    shank : int
        Shank index (e.g. 0–7 for an 8-shank probe).
    device_type : str
        Mapping CSV stem under ``rec2nwb/mapping/`` (e.g. ``"8shank32"``).
    impedance_path : Path, optional
        Optional impedance CSV used to attach impedance values / verify channel
        names. Same semantics as the NWB pipeline.
    bad_ch_ids : list, optional
        Channel names to exclude (matches what ``load_bad_ch`` returns).
    verbose : bool
        Print a short summary.

    Returns
    -------
    BaseRecording
        Ready to feed into ``MsSorting._sort_shank``.
    """
    data_folder = Path(data_folder)
    converter = SpikeGadgetsRecToNWB()  # only for its gap/timestamp helpers

    # --- Discover parts ---
    data_files = get_data_files(data_folder, 'spikegadget_rec')
    if verbose:
        print(f"Direct-sort: {len(data_files)} part(s) for shank {shank} in {data_folder}")

    # --- Electrode table for this shank ---
    channel_index, xcoord, ycoord = get_ch_index_on_shank(shank, device_type)
    impedance_table = pd.read_csv(impedance_path) if impedance_path else None
    electrode_df = build_electrode_df(
        channel_index, xcoord, ycoord, 'spikegadget_rec',
        impedance_table, bad_ch_ids,
    )
    if verbose:
        print(f"  Good electrodes on shank {shank}: {len(electrode_df)}")

    # --- Per-part open + optional gap wrap, then concatenate ---
    _folder_first_ts: dict[Path, int | None] = {}
    _folder_cum: dict[Path, int] = {}
    wrapped_parts: list[BaseRecording] = []

    for f in data_files:
        rec_folder = f.parent
        if rec_folder not in _folder_first_ts:
            try:
                _folder_first_ts[rec_folder] = converter._get_first_timestamp(rec_folder)
            except FileNotFoundError as e:
                if verbose:
                    print(f"  WARNING: no DIO timestamp for {rec_folder.name} ({e}); "
                          f"gap interpolation disabled here.")
                _folder_first_ts[rec_folder] = None
            _folder_cum[rec_folder] = 0

        parent = _open_part(f)
        wrapped, advance = _wrap_with_gaps(
            parent, f, converter,
            first_timestamp=_folder_first_ts[rec_folder],
            cum_offset=_folder_cum[rec_folder],
        )
        wrapped_parts.append(wrapped)
        _folder_cum[rec_folder] += advance

    if len(wrapped_parts) == 1:
        full_rec = wrapped_parts[0]
    else:
        full_rec = si.concatenate_recordings(wrapped_parts)

    # --- Slice to good channels for this shank ---
    actual_ids = full_rec.get_channel_ids()
    good_indices = resolve_good_channel_ids(
        electrode_df, 'spikegadget_rec',
        has_impedance=(impedance_table is not None),
        actual_channel_ids=actual_ids,
    )
    good_str_ids = [str(i) for i in good_indices]
    sliced = full_rec.select_channels(channel_ids=good_str_ids)

    # --- Attach probe geometry ---
    # Reorder electrode_df rows to follow good_indices so positions[i] matches
    # the sliced recording's channel order.
    ed_indexed = electrode_df.set_index('channel_index').loc[list(good_indices)].reset_index()
    probe = _make_probe(ed_indexed, good_str_ids)
    sliced = sliced.set_probe(probe, in_place=False)

    if verbose:
        fs = sliced.get_sampling_frequency()
        n_frames = sliced.get_num_frames()
        print(f"  Built recording: {sliced.get_num_channels()} ch, "
              f"{n_frames} frames ({n_frames / fs:.1f} s)")

    return sliced


def build_sortable_recording_intan(
    data_folder: Path,
    shank: int,
    device_type: str,
    impedance_path: Path | None = None,
    bad_ch_ids: list | None = None,
    verbose: bool = True,
) -> BaseRecording:
    """Return a per-shank SpikeInterface recording sourced directly from Intan files.

    Intan (``.rhd``/``.rhs``) analogue of :func:`build_sortable_recording`. Mirrors
    the channel/electrode setup that ``intan2nwb.EphysToNWBConverter`` writes into a
    per-shank NWB, minus the disk write: every ``.rhd``/``.rhs`` part in
    ``data_folder`` is opened, concatenated in time, sliced to the shank's good
    channels, and given probe geometry.

    Unlike the SpikeGadgets path there is no packet-loss gap interpolation — Intan
    recordings have no ``.rec`` packet-loss ``.txt`` sidecars, so parts are simply
    concatenated.

    Parameters
    ----------
    data_folder : Path
        Folder containing one or more ``*.rhd``/``*.rhs`` files (the layout consumed
        by ``intan2nwb``). Multiple files are concatenated in sorted (time) order.
    shank : int
        Shank index.
    device_type : str
        Mapping CSV stem under ``rec2nwb/mapping/`` (e.g. ``"4shank16intan"``).
    impedance_path : Path, optional
        Optional impedance CSV whose ``Channel Name`` column supplies channel IDs
        (e.g. ``"A-000"``) to select, mirroring the NWB pipeline. **Not required:**
        when omitted, channels are selected by native position — the mapping CSV's
        row order is the Intan native channel order, so ``channel_index`` indexes the
        recording directly.
    bad_ch_ids : list, optional
        Channel names to exclude (matches what ``load_bad_ch`` returns). With an
        impedance CSV these are the impedance ``Channel Name`` values (e.g.
        ``"A-005"``); without one they are the mapping's ``"chN"`` names where ``N``
        is the Intan channel index.
    verbose : bool
        Print a short summary.

    Returns
    -------
    BaseRecording
        Ready to feed into ``MsSorting._sort_shank``.
    """
    data_folder = Path(data_folder)

    # --- Discover parts ---
    data_files = get_data_files(data_folder, 'intan')
    if verbose:
        print(f"Direct-sort (intan): {len(data_files)} part(s) for shank {shank} "
              f"in {data_folder}")

    # --- Electrode table for this shank ---
    channel_index, xcoord, ycoord = get_ch_index_on_shank(shank, device_type)
    impedance_table = pd.read_csv(impedance_path) if impedance_path else None
    electrode_df = build_electrode_df(
        channel_index, xcoord, ycoord, 'intan',
        impedance_table, bad_ch_ids,
    )
    if verbose:
        print(f"  Good electrodes on shank {shank}: {len(electrode_df)}")

    # --- Open each part and concatenate in time (no gap interpolation for Intan) ---
    parts = [se.read_intan(str(f), stream_id='0') for f in data_files]
    full_rec = parts[0] if len(parts) == 1 else si.concatenate_recordings(parts)

    # --- Slice to good channels for this shank ---
    actual_ids = list(full_rec.get_channel_ids())

    if impedance_table is not None:
        # Impedance CSV supplies channel names ("A-000"-style); match them against
        # the recording's IDs, mirroring the intan2nwb -> NWB pipeline exactly.
        good_ids = resolve_good_channel_ids(
            electrode_df, 'intan', has_impedance=True,
            actual_channel_ids=full_rec.get_channel_ids(),
        )
        good_str_ids = [str(i) for i in good_ids]
        actual_set = set(map(str, actual_ids))
        missing = [c for c in good_str_ids if c not in actual_set]
        if missing:
            raise ValueError(
                f"{len(missing)} channel(s) for shank {shank} not found in the Intan "
                f"recording (e.g. {missing[:5]}). The recording's channel IDs look "
                f"like {sorted(actual_set)[:3]}. The impedance CSV's 'Channel Name' "
                f"values must match the recording IDs, or device_type={device_type!r} "
                f"is the wrong mapping."
            )
        # electrode_df rows are already in good_ids order; reindex defensively so
        # probe positions[i] line up with the sliced recording's channel order.
        ed_ordered = (electrode_df.set_index('channel_name')
                      .loc[good_str_ids].reset_index())
    else:
        # No impedance file: select by NATIVE POSITION. The mapping CSV's row order
        # is the Intan native channel order, so channel_index indexes the recording
        # directly — no channel-name matching (and thus no impedance CSV) needed.
        positions = electrode_df['channel_index'].to_numpy()
        n_rec = len(actual_ids)
        oob = positions[(positions < 0) | (positions >= n_rec)]
        if len(oob):
            raise ValueError(
                f"Channel index/indices {list(oob[:5])} on shank {shank} are out of "
                f"range for this {n_rec}-channel Intan recording. device_type="
                f"{device_type!r} likely doesn't match this recording."
            )
        good_ids = [actual_ids[p] for p in positions]
        good_str_ids = [str(i) for i in good_ids]
        # electrode_df is already in selection order (positions order).
        ed_ordered = electrode_df.reset_index(drop=True)
        if verbose:
            print(f"  No impedance CSV — selecting {len(good_ids)} channel(s) by "
                  f"native position; recording IDs look like {good_str_ids[:3]}")

    sliced = full_rec.select_channels(channel_ids=good_ids)

    # --- Attach probe geometry ---
    # select_channels preserves the given order, so positions[i] <-> good_ids[i].
    probe = _make_probe(ed_ordered, good_str_ids)
    sliced = sliced.set_probe(probe, in_place=False)

    if verbose:
        fs = sliced.get_sampling_frequency()
        n_frames = sliced.get_num_frames()
        print(f"  Built recording: {sliced.get_num_channels()} ch, "
              f"{n_frames} frames ({n_frames / fs:.1f} s)")

    return sliced
