"""MUA detection by reusing MountainSort5's spike detector.

This wraps ``mountainsort5.core.detect_spikes.detect_spikes`` — the exact
function that produces the "Detected N spikes" line in an MS5 run — so that
threshold events can be obtained without any of the downstream clustering.

The MS5 detector is a pure-python loop over threshold crossings, so it is run
here chunk-by-chunk with overlap instead of on the whole recording at once.
That keeps memory flat and lets long sessions run without loading all traces.
"""

import contextlib
import io
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import spikeinterface as si
import spikeinterface.preprocessing as sp

from mountainsort5.core.detect_spikes import detect_spikes


@dataclass
class MuaEvents:
    """Threshold events found across a recording.

    times, channel_indices and amplitudes are parallel arrays sorted by time.
    ``amplitudes`` are read off the same (scaled) traces the detector saw, so
    with ``scale_mode='whiten'`` or ``'zscore'`` they are in noise-sigma units.
    """
    times: npt.NDArray[np.int64]
    channel_indices: npt.NDArray[np.int32]
    amplitudes: npt.NDArray[np.float32]
    channel_ids: npt.NDArray = field(default_factory=lambda: np.array([]))
    sampling_frequency: float = 0.0
    num_frames: int = 0

    @property
    def times_sec(self) -> npt.NDArray[np.float64]:
        return self.times / self.sampling_frequency

    def per_channel_times(self):
        """Dict of channel_id -> event times in seconds."""
        return {
            cid: self.times[self.channel_indices == m] / self.sampling_frequency
            for m, cid in enumerate(self.channel_ids)
        }

    def rate_histogram(self, bin_size_sec: float = 0.010, per_channel: bool = False):
        """Binned MUA rate in Hz.

        Returns (bin_edges_sec, rate). ``rate`` is shape (n_bins,) when
        ``per_channel`` is False, else (n_channels, n_bins).
        """
        duration = self.num_frames / self.sampling_frequency
        edges = np.arange(0, duration + bin_size_sec, bin_size_sec)
        t = self.times_sec
        if not per_channel:
            counts, _ = np.histogram(t, bins=edges)
            return edges, counts / bin_size_sec
        n_ch = len(self.channel_ids)
        counts = np.zeros((n_ch, len(edges) - 1), dtype=np.float64)
        for m in range(n_ch):
            counts[m], _ = np.histogram(t[self.channel_indices == m], bins=edges)
        return edges, counts / bin_size_sec


def preprocess_for_mua(
    recording: si.BaseRecording, *,
    freq_min: float = 300,
    freq_max: float = 6000,
    common_reference: bool = True,
    scale_mode: str = 'zscore',
) -> si.BaseRecording:
    """Bandpass (+CMR) then put the traces in noise-sigma units.

    ``scale_mode``:
      - ``'whiten'``  : spikeinterface whitening, i.e. what MsSorting.py feeds
        MS5. Decorrelates channels, so a threshold of 5.5 means the same thing
        it does in a sorting run.
      - ``'zscore'``  : per-channel divide by a MAD-based noise estimate. No
        cross-channel mixing, which is usually what you want for per-channel
        MUA — whitening can move a large spike onto its neighbours.
      - ``'none'``    : leave the traces alone; ``detect_threshold`` is then in
        raw units (e.g. uV).
    """
    if scale_mode not in ('whiten', 'zscore', 'none'):
        raise ValueError(f"scale_mode must be 'whiten', 'zscore' or 'none', got {scale_mode!r}")

    rec = recording
    if rec.get_dtype().kind == 'u':
        rec = sp.unsigned_to_signed(rec)
    if common_reference:
        rec = sp.common_reference(rec, reference='global', operator='median')
    rec = sp.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max, dtype=np.float32)

    if scale_mode == 'whiten':
        rec = sp.whiten(rec)
    elif scale_mode == 'zscore':
        # median/MAD rather than mean/std so that the spikes themselves do not
        # inflate the noise estimate.
        rec = sp.zscore(rec, mode='median+mad')
    return rec


def detect_mua(
    recording: si.BaseRecording, *,
    detect_threshold: float = 5.0,
    detect_sign: int = -1,
    detect_time_radius_msec: float = 0.5,
    detect_channel_radius: Optional[float] = 0.0,
    preprocess: bool = True,
    scale_mode: str = 'zscore',
    freq_min: float = 300,
    freq_max: float = 6000,
    common_reference: bool = True,
    chunk_duration_sec: float = 60.0,
    collapse_simultaneous: bool = False,
    verbose: bool = True,
) -> MuaEvents:
    """Detect MUA threshold events using MountainSort5's detector.

    Parameters mirror ``ms5.Scheme1SortingParameters`` where they overlap.

    detect_sign
        -1 negative-going peaks (the usual choice for extracellular MUA),
        +1 positive, 0 both (detector works on -|trace|).
    detect_time_radius_msec
        An event must be the most extreme sample within +/- this window on its
        own channel and on every channel inside ``detect_channel_radius``.
        Doubles as the refractory period, so 0.5 ms caps one channel at 2 kHz.
    detect_channel_radius
        Radius in channel-location units for cross-channel peak suppression.
        ``0.0`` (default) compares each channel only against itself, giving
        independent per-channel MUA — one physical spike can then be counted on
        several channels. Set it to your inter-site spacing (or ``None`` for
        all channels, the MS5 default) to keep only the event on the channel
        where it is largest.
    collapse_simultaneous
        MS5's scheme 1 drops all but one event per identical frame after
        detection. That is there to keep isosplit happy, and it throws away
        genuinely simultaneous spikes on distant channels, so it is off here.
    """
    if recording.get_num_segments() > 1:
        recording = si.concatenate_recordings(recording_list=[recording])

    rec = (preprocess_for_mua(recording, freq_min=freq_min, freq_max=freq_max,
                              common_reference=common_reference, scale_mode=scale_mode)
           if preprocess else recording)

    fs = rec.get_sampling_frequency()
    N = rec.get_num_frames()
    M = rec.get_num_channels()
    channel_locations = np.asarray(rec.get_channel_locations(), dtype=np.float32)

    time_radius = int(math.ceil(detect_time_radius_msec / 1000 * fs))
    # One extra sample of context so an event sitting on a chunk edge sees the
    # same neighbourhood it would have seen in a single-shot run.
    pad = time_radius + 1
    chunk_size = max(int(chunk_duration_sec * fs), 10 * pad)

    all_times, all_channels, all_amps = [], [], []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        load_start = max(0, start - pad)
        load_end = min(N, end + pad)
        traces = np.asarray(rec.get_traces(start_frame=load_start, end_frame=load_end),
                            dtype=np.float32)

        # detect_spikes drops events within margin_left/right of the array
        # edges. Use those margins to drop the overlap region instead: only
        # events belonging to [start, end) survive, so chunks never double-count.
        margin_left = start - load_start
        margin_right = load_end - end
        if traces.shape[0] <= margin_left + margin_right:
            continue

        # detect_spikes prints its adjacency table on every call.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            times, channel_indices = detect_spikes(
                traces=traces,
                channel_locations=channel_locations,
                time_radius=time_radius,
                channel_radius=detect_channel_radius,
                detect_threshold=detect_threshold,
                detect_sign=detect_sign,
                margin_left=margin_left,
                margin_right=margin_right,
                verbose=False,
            )

        if len(times):
            amps = traces[times, channel_indices]
            all_times.append(times.astype(np.int64) + load_start)
            all_channels.append(channel_indices)
            all_amps.append(amps)

        if verbose:
            n = int(sum(len(t) for t in all_times))
            print(f'  MUA detect: {end / fs:8.1f} s / {N / fs:.1f} s  ({n} events)')

    if all_times:
        times = np.concatenate(all_times)
        channel_indices = np.concatenate(all_channels)
        amplitudes = np.concatenate(all_amps).astype(np.float32)
        order = np.argsort(times, kind='stable')
        times, channel_indices, amplitudes = times[order], channel_indices[order], amplitudes[order]
    else:
        times = np.array([], dtype=np.int64)
        channel_indices = np.array([], dtype=np.int32)
        amplitudes = np.array([], dtype=np.float32)

    if collapse_simultaneous and len(times):
        keep = np.concatenate([[0], np.nonzero(np.diff(times) > 0)[0] + 1])
        times, channel_indices, amplitudes = times[keep], channel_indices[keep], amplitudes[keep]

    if verbose:
        duration = N / fs
        print(f'Detected {len(times)} MUA events over {duration:.1f} s '
              f'({len(times) / duration:.1f} Hz across {M} channels)')

    return MuaEvents(
        times=times,
        channel_indices=channel_indices.astype(np.int32),
        amplitudes=amplitudes,
        channel_ids=np.asarray(rec.get_channel_ids()),
        sampling_frequency=fs,
        num_frames=N,
    )


def detect_mua_from_traces(
    traces: npt.NDArray[np.float32], *,
    channel_locations: npt.NDArray[np.float32],
    sampling_frequency: float,
    detect_threshold: float = 5.0,
    detect_sign: int = -1,
    detect_time_radius_msec: float = 0.5,
    detect_channel_radius: Optional[float] = 0.0,
):
    """Single-shot detection on an in-memory (num_samples, num_channels) array.

    Traces are used as-is — filter and scale them yourself first. Returns
    (times, channel_indices, amplitudes).
    """
    traces = np.asarray(traces, dtype=np.float32)
    time_radius = int(math.ceil(detect_time_radius_msec / 1000 * sampling_frequency))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        times, channel_indices = detect_spikes(
            traces=traces,
            channel_locations=np.asarray(channel_locations, dtype=np.float32),
            time_radius=time_radius,
            channel_radius=detect_channel_radius,
            detect_threshold=detect_threshold,
            detect_sign=detect_sign,
            margin_left=time_radius,
            margin_right=time_radius,
            verbose=False,
        )
    amps = traces[times, channel_indices] if len(times) else np.array([], dtype=np.float32)
    return times, channel_indices, amps
