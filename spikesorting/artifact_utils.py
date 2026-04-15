import os
import tempfile
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from joblib import Parallel, delayed
from spikeinterface.core import BaseRecording, BaseRecordingSegment, NumpyRecording


class MemmapCleanupHandle:
    """
    Cleanup handle for temporary memmap files.

    Call cleanup() or just call the handle directly to remove the memmap file
    and optionally its parent directory. Safe to call multiple times.

    Example
    -------
    >>> rec, cleanup = repair_artifacts_recording(...)
    >>> # ... use rec ...
    >>> cleanup()  # or cleanup.cleanup()
    """

    def __init__(self, file_path, folder_path=None, remove_folder=False):
        self.file_path = file_path
        self.folder_path = folder_path
        self.remove_folder = remove_folder
        self._cleaned = False

    def cleanup(self):
        """Clean up temporary files. Returns True if successful, False otherwise."""
        if self._cleaned:
            return True

        file_removed = False

        if self.file_path and os.path.isfile(self.file_path):
            try:
                os.remove(self.file_path)
                file_removed = True
            except OSError as e:
                print(f"Warning: Could not remove memmap file: {self.file_path}")
                print(f"  Error: {e}")
        else:
            file_removed = True

        if self.remove_folder and self.folder_path:
            try:
                if os.path.isdir(self.folder_path) and not os.listdir(self.folder_path):
                    os.rmdir(self.folder_path)
            except OSError as e:
                print(f"Warning: Could not remove folder: {self.folder_path}")
                print(f"  Error: {e}")

        self._cleaned = file_removed
        return file_removed

    __call__ = cleanup


def copy_recording_properties(source: BaseRecording, target: BaseRecording) -> None:
    """
    Copy all properties from source recording to target recording.

    Parameters
    ----------
    source : BaseRecording
        Recording to copy properties from.
    target : BaseRecording
        Recording to copy properties to.
    """
    for key in source.get_property_keys():
        target.set_property(key, source.get_property(key))


def get_artifact_timestamps_per_channel(data, slope_threshold=500):
    """
    Detects artifact positions for each channel independently.

    Args:
        data (np.ndarray): Your (n_timepoints, n_channels) array (e.g., 6000x30).
        slope_threshold (float): The max allowed change between two timepoints.

    Returns:
        artifact_timestamps (list[np.ndarray]): A list of length n_channels, where each
            element is a 1D array of time indices where artifacts were detected.
    """
    # 1. Calculate the derivative (slope) along time axis
    diff_data = np.diff(data, axis=0, prepend=data[0:1, :])

    # 2. Take absolute value
    abs_slope = np.abs(diff_data)

    # 3. Get per-channel artifact time indices
    artifact_mask = abs_slope > slope_threshold
    n_channels = data.shape[1]
    artifact_timestamps = [np.where(artifact_mask[:, ch])[0] for ch in range(n_channels)]

    return artifact_timestamps


def repair_artifacts_per_channel(
    data,
    artifact_timestamps,
    window=50,
    gap_pad_samples=2,
    correct_dc_offset=True,
    dither=True,
    fs=None,
    rng=None,
):
    """
    Repairs artifacts per channel using DC offset correction + windowed PCHIP interpolation.

    Per artifact per channel:
      1. Cluster consecutive flagged samples into contiguous groups.
      2. Optionally correct DC step offset by shifting all post-artifact data.
      3. Use window samples on each side as PCHIP anchors to interpolate across the gap.
      4. Optionally add Gaussian dither scaled to the channel's high-frequency noise σ so
         the repaired gap resembles quiet neural activity to downstream sorters (e.g.
         MountainSort) rather than a suspiciously flat segment.

    Args:
        data                (np.ndarray): Shape (n_timepoints, n_channels).
        artifact_timestamps (list[np.ndarray]): Output of get_artifact_timestamps_per_channel().
        window              (int): Number of clean samples on each side used as PCHIP anchors.
        gap_pad_samples     (int): Extra samples excluded on each side of the artifact window
                                   to avoid edge transients entering the anchor fit.
        correct_dc_offset   (bool): If True, shift post-artifact data to remove DC step before
                                    interpolating. Recommended for stimulation artifacts.
        dither              (bool): If True, add Gaussian noise whose σ matches the channel's
                                    high-frequency (300–6000 Hz) noise floor.
        fs                  (float | None): Sampling rate in Hz. Required when dither=True to
                                    design the bandpass filter for σ estimation.
        rng                 (np.random.Generator | int | None): Random generator or seed for
                                    reproducible dither. None uses a fresh default RNG.

    Returns:
        cleaned_data (np.ndarray): Shape (n_timepoints, n_channels), artifact-repaired.
    """
    if dither and fs is None:
        raise ValueError("fs (sampling rate) must be provided when dither=True.")

    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    if dither:
        # Bandpass 300–6000 Hz to isolate high-frequency noise; clamp to Nyquist.
        nyq = fs / 2.0
        highcut = min(6000.0, nyq * 0.95)
        sos = butter(4, [300.0 / nyq, highcut / nyq], btype='bandpass', output='sos')

    n_timepoints, n_channels = data.shape
    cleaned_data = data.copy().astype(float)

    for ch in range(n_channels):
        timestamps = artifact_timestamps[ch]
        if len(timestamps) == 0:
            continue

        # Cluster consecutive artifact samples into contiguous groups
        gaps = np.diff(timestamps)
        split_indices = np.where(gaps > 1)[0] + 1
        artifact_groups = np.split(timestamps, split_indices)

        # Estimate σ from the high-frequency band of all clean samples on this channel.
        # Bandpassing removes LFP/slow drifts so σ reflects only the thermal + MUA noise
        # floor — the "fuzz" we want to replicate in the dithered gap.
        if dither:
            artifact_mask_ch = np.zeros(n_timepoints, dtype=bool)
            artifact_mask_ch[timestamps] = True
            clean_samples = cleaned_data[~artifact_mask_ch, ch]
            # sosfiltfilt requires at least padlen samples (default 3*(2*order)=24 for order-4)
            if len(clean_samples) > 27:
                hp_samples = sosfiltfilt(sos, clean_samples)
            else:
                hp_samples = clean_samples
            # MAD-based σ estimate: robust to residual spikes in the clean segment.
            # The 0.6745 factor converts MAD to a consistent σ estimate for Gaussian noise.
            baseline_sigma = np.median(np.abs(hp_samples - np.median(hp_samples))) / 0.6745

        for group in artifact_groups:
            art_start = int(group[0])
            art_end   = int(group[-1])

            # Expand window slightly to exclude edge transients
            win_start = max(0, art_start - gap_pad_samples)
            win_end   = min(n_timepoints - 1, art_end + gap_pad_samples)

            # Anchor regions
            pre_start = max(0, win_start - window)
            post_end  = min(n_timepoints, win_end + 1 + window)

            # Need enough anchor samples on both sides
            if (win_start - pre_start) < 2 or (post_end - win_end - 1) < 2:
                continue

            seg = cleaned_data[:, ch]

            # DC offset correction: align post-artifact baseline to pre-artifact baseline
            if correct_dc_offset:
                pre_level  = np.median(seg[pre_start:win_start])
                post_level = np.median(seg[win_end + 1:post_end])
                offset = post_level - pre_level
                seg[win_end + 1:] -= offset

            # Build anchor arrays from both flanking windows
            anchor_x = np.concatenate([
                np.arange(pre_start, win_start),
                np.arange(win_end + 1, post_end),
            ])
            anchor_y = seg[anchor_x]

            # PCHIP interpolation across the gap
            interp_fn = PchipInterpolator(anchor_x, anchor_y)
            gap_idx   = np.arange(win_start, win_end + 1)
            seg[gap_idx] = interp_fn(gap_idx)

            # Dither: add Gaussian noise matching this channel's baseline σ so the
            # repaired region resembles quiet neural activity rather than a flat gap.
            if dither and baseline_sigma > 0:
                seg[gap_idx] += rng.normal(0.0, baseline_sigma, size=len(gap_idx))

    return cleaned_data


def detect_artifact_rolling_std(data, window_size=50, z_threshold=5):
    """
    Detects artifacts using rolling standard deviation (Z-score method).

    Args:
        data (np.ndarray): Input data of shape (n_timepoints, n_channels).
        window_size (int): Size of the rolling window (e.g., 50 samples).
        z_threshold (float): How many std deviations above baseline to flag (e.g., 5).

    Returns:
        artifact_timestamps (list[np.ndarray]): A list of length n_channels, where each
            element is a 1D array of time indices where artifacts were detected.
    """
    df = pd.DataFrame(data)
    rolling_std = df.rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.fillna(0)
    median_std = rolling_std.median()
    q75 = rolling_std.quantile(0.75)
    q25 = rolling_std.quantile(0.25)
    iqr = q75 - q25
    if iqr.sum() == 0:
        scale_factor = 1
    else:
        scale_factor = iqr / 1.35
    z_scores = (rolling_std - median_std) / scale_factor
    artifact_mask = z_scores.values > z_threshold
    n_channels = data.shape[1]
    artifact_timestamps = [np.where(artifact_mask[:, ch])[0] for ch in range(n_channels)]
    return artifact_timestamps


# ---------------------------------------------------------------------------
# Recording-level API (handles large recordings via memmap + batched loading)
# ---------------------------------------------------------------------------

def _detect_rolling_std_single_channel(ch_data, window_size=50, z_threshold=5):
    """
    Rolling std Z-score artifact detection for a single 1-D channel.

    Returns artifact time indices (same format as slope detection).
    """
    s = pd.Series(ch_data.astype(np.float64))
    rolling_std = s.rolling(window=window_size, center=True).std().fillna(0).values
    median_std = np.median(rolling_std)
    q75 = np.percentile(rolling_std, 75)
    q25 = np.percentile(rolling_std, 25)
    iqr = q75 - q25
    scale_factor = iqr / 1.35 if iqr > 0 else 1.0
    z_scores = (rolling_std - median_std) / scale_factor
    return np.where(z_scores > z_threshold)[0]


def _repair_artifacts_single_channel(ch_data, slope_threshold=500, window=50,
                                      gap_pad_samples=2, correct_dc_offset=True,
                                      detection_method='slope',
                                      rolling_window_size=50, rolling_z_threshold=5):
    """
    Detect artifacts and repair them with PCHIP interpolation for one channel.

    Fuses detection + repair into a single pass over a 1-D channel array.
    Supports two detection methods selectable via ``detection_method``.

    Parameters
    ----------
    ch_data : np.ndarray, shape (n_samples,)
        Single-channel trace. Converted internally to float64.
    slope_threshold : float
        Max allowed sample-to-sample change (used when detection_method='slope').
    window : int
        PCHIP anchor samples on each side of the artifact gap.
    gap_pad_samples : int
        Extra samples excluded around artifact edges before anchoring.
    correct_dc_offset : bool
        Shift post-artifact baseline to pre-artifact level before interpolating.
    detection_method : str, default 'slope'
        'slope'       — flag samples where |diff| > slope_threshold.
        'rolling_std' — flag samples where rolling-std Z-score > rolling_z_threshold.
    rolling_window_size : int, default 50
        Rolling window size in samples (used when detection_method='rolling_std').
    rolling_z_threshold : float, default 5
        Z-score threshold (used when detection_method='rolling_std').

    Returns
    -------
    seg : np.ndarray, float64, same length as ch_data
    """
    seg = ch_data.astype(np.float64)
    n = len(seg)

    # Detection
    if detection_method == 'rolling_std':
        timestamps = _detect_rolling_std_single_channel(
            seg, rolling_window_size, rolling_z_threshold,
        )
    else:  # 'slope'
        diff = np.diff(seg, prepend=seg[0:1])
        timestamps = np.where(np.abs(diff) > slope_threshold)[0]

    if len(timestamps) == 0:
        return seg, np.array([], dtype=np.int64)

    # Cluster consecutive flagged samples into contiguous groups
    split_indices = np.where(np.diff(timestamps) > 1)[0] + 1
    artifact_groups = np.split(timestamps, split_indices)

    for group in artifact_groups:
        art_start = int(group[0])
        art_end   = int(group[-1])
        win_start = max(0, art_start - gap_pad_samples)
        win_end   = min(n - 1, art_end + gap_pad_samples)
        pre_start = max(0, win_start - window)
        post_end  = min(n, win_end + 1 + window)

        if (win_start - pre_start) < 2 or (post_end - win_end - 1) < 2:
            continue

        if correct_dc_offset:
            pre_level  = np.median(seg[pre_start:win_start])
            post_level = np.median(seg[win_end + 1:post_end])
            seg[win_end + 1:] -= (post_level - pre_level)

        anchor_x = np.concatenate([
            np.arange(pre_start, win_start),
            np.arange(win_end + 1, post_end),
        ])
        anchor_y = seg[anchor_x]
        interp_fn = PchipInterpolator(anchor_x, anchor_y)
        gap_idx = np.arange(win_start, win_end + 1)
        seg[gap_idx] = interp_fn(gap_idx)

    return seg, timestamps


def _repair_artifacts_channel_worker(local_idx, traces_batch, slope_threshold,
                                      window, gap_pad_samples, correct_dc_offset,
                                      dtype, detection_method,
                                      rolling_window_size, rolling_z_threshold):
    """Joblib worker: process one channel slice from a batch array."""
    ch = traces_batch[:, local_idx].copy()
    result, timestamps = _repair_artifacts_single_channel(
        ch, slope_threshold, window, gap_pad_samples, correct_dc_offset,
        detection_method, rolling_window_size, rolling_z_threshold,
    )
    return local_idx, result.astype(dtype), timestamps


def _detect_batch_vectorized(traces, detection_method, slope_threshold,
                              rolling_window_size, rolling_z_threshold,
                              global_median=None, global_scale=None):
    """
    Vectorized artifact detection on a (T, C) array. No per-channel Python loop.

    Parameters
    ----------
    global_median, global_scale : np.ndarray (C,), optional
        Pre-computed global normalization statistics. When provided, Z-scores are
        computed relative to the full-recording baseline rather than the current
        batch, preventing threshold drift across batches.

    Returns
    -------
    artifact_mask : np.ndarray, bool, shape (T, C)
    """
    # Fix: float32 is sufficient — no need to upcast to float64
    data = traces.astype(np.float32)

    if detection_method == 'slope':
        diff = np.empty_like(data)
        diff[0, :] = 0.0
        diff[1:, :] = np.abs(np.diff(data, axis=0))
        return diff > slope_threshold

    # rolling_std — use scipy uniform_filter1d for O(T*C) computation
    # Var(x) = E[x²] - E[x]² computed with a uniform (boxcar) filter.
    mean_x  = uniform_filter1d(data,      size=rolling_window_size, axis=0, mode='nearest')
    mean_x2 = uniform_filter1d(data ** 2, size=rolling_window_size, axis=0, mode='nearest')
    rolling_std = np.sqrt(np.maximum(0.0, mean_x2 - mean_x ** 2))

    if global_median is not None and global_scale is not None:
        # Fix: use recording-wide stats so the threshold doesn't shift per batch
        z_scores = (rolling_std - global_median) / global_scale
    else:
        # Fallback: per-batch normalization (original behaviour)
        median_std = np.median(rolling_std, axis=0)
        q75 = np.percentile(rolling_std, 75, axis=0)
        q25 = np.percentile(rolling_std, 25, axis=0)
        iqr = q75 - q25
        scale = np.where(iqr > 0, iqr / 1.35, 1.0)
        z_scores = (rolling_std - median_std) / scale

    return z_scores > rolling_z_threshold


def detect_artifacts_recording(
    rec_obj,
    detection_method='slope',
    slope_threshold=500,
    rolling_window_size=50,
    rolling_z_threshold=5,
    batch_size=None,
    time_batch_sec=None,
):
    """
    Detect artifact timestamps without repairing or writing to disk.

    Reads the recording in time batches and runs fully vectorized detection
    across all channels simultaneously (no per-channel Python loop).
    Use LazyArtifactRepairRecording to apply repairs on-the-fly.

    Parameters
    ----------
    rec_obj : BaseRecording
    detection_method : str
        'slope' or 'rolling_std'
    slope_threshold : float
    rolling_window_size : int
    rolling_z_threshold : float
    batch_size : int, optional
        Channels per get_traces call. None = all channels at once.
    time_batch_sec : float, optional
        Process in time chunks of this many seconds. None = all at once.

    Returns
    -------
    artifact_timestamps : list of np.ndarray
        Per-channel detected artifact sample indices, length n_channels.
    """
    fs = rec_obj.get_sampling_frequency()
    n_samples = rec_obj.get_num_frames()
    n_channels = rec_obj.get_num_channels()
    channel_ids = rec_obj.get_channel_ids()

    if batch_size is None:
        batch_size = n_channels

    # Fix: slope needs overlap=1 so a jump at a batch boundary is not missed
    overlap_samples = rolling_window_size if detection_method == 'rolling_std' else 1

    if time_batch_sec is not None:
        time_batch_samples = int(time_batch_sec * fs)
    else:
        time_batch_samples = n_samples

    if detection_method == 'slope':
        det_desc = f"slope={slope_threshold}"
    else:
        det_desc = f"rolling_std w={rolling_window_size} z={rolling_z_threshold}"
    print(f"Artifact detection: {n_channels} ch, {n_samples} samples, {det_desc}", flush=True)

    start_time = time.time()
    time_batches = list(range(0, n_samples, time_batch_samples))

    # Fix: compute global per-channel rolling_std statistics in a lightweight first
    # pass so the Z-score threshold is anchored to the full-recording baseline rather
    # than fluctuating with each 600 s batch.
    global_median = global_scale = None
    if detection_method == 'rolling_std':
        print("Pass 1/2: computing global normalization statistics...", flush=True)
        batch_medians, batch_q25s, batch_q75s = [], [], []
        for t_start in tqdm(time_batches, desc="Stats pass", unit="batch"):
            t_end      = min(t_start + time_batch_samples, n_samples)
            load_start = max(0, t_start - overlap_samples)
            load_end   = min(n_samples, t_end + overlap_samples)
            traces_all = rec_obj.get_traces(
                start_frame=load_start, end_frame=load_end,
                channel_ids=list(channel_ids), return_scaled=True,
            ).astype(np.float32)
            mean_x  = uniform_filter1d(traces_all,          size=rolling_window_size, axis=0, mode='nearest')
            mean_x2 = uniform_filter1d(traces_all ** 2,     size=rolling_window_size, axis=0, mode='nearest')
            rs = np.sqrt(np.maximum(0.0, mean_x2 - mean_x ** 2))
            batch_medians.append(np.median(rs, axis=0))
            batch_q25s.append(np.percentile(rs, 25, axis=0))
            batch_q75s.append(np.percentile(rs, 75, axis=0))
        # Aggregate: median-of-per-batch-medians converges to the true global median
        global_median = np.median(np.stack(batch_medians), axis=0)   # (n_channels,)
        global_iqr    = np.median(np.stack(batch_q75s), axis=0) - np.median(np.stack(batch_q25s), axis=0)
        global_scale  = np.where(global_iqr > 0, global_iqr / 1.35, 1.0)
        print("Pass 2/2: detecting artifacts with global thresholds...", flush=True)

    artifact_timestamps = [[] for _ in range(n_channels)]

    pbar = tqdm(time_batches, desc="Detecting artifacts", unit="batch")
    for t_start in pbar:
        t_end = min(t_start + time_batch_samples, n_samples)
        load_start = max(0, t_start - overlap_samples)
        load_end = min(n_samples, t_end + overlap_samples)
        local_write_start = t_start - load_start
        local_write_end = t_end - load_start
        pbar.set_postfix_str(f"{t_start/fs:.0f}s – {t_end/fs:.0f}s")

        for batch_start in range(0, n_channels, batch_size):
            batch_end = min(batch_start + batch_size, n_channels)
            batch_ch_ids = [channel_ids[i] for i in range(batch_start, batch_end)]

            traces_batch = rec_obj.get_traces(
                start_frame=load_start, end_frame=load_end,
                channel_ids=batch_ch_ids, return_scaled=True,
            )

            # Slice global stats to match the current channel batch
            gm = global_median[batch_start:batch_end] if global_median is not None else None
            gs = global_scale[batch_start:batch_end]  if global_scale  is not None else None

            artifact_mask = _detect_batch_vectorized(
                traces_batch, detection_method, slope_threshold,
                rolling_window_size, rolling_z_threshold,
                global_median=gm, global_scale=gs,
            )
            # Trim to core window, convert local indices to global
            core_mask = artifact_mask[local_write_start:local_write_end, :]
            for local_ch in range(batch_end - batch_start):
                ts_core = np.where(core_mask[:, local_ch])[0] + t_start
                artifact_timestamps[batch_start + local_ch].append(ts_core)

    artifact_timestamps = [
        np.concatenate(ts_list) if ts_list else np.array([], dtype=np.int64)
        for ts_list in artifact_timestamps
    ]

    elapsed = time.time() - start_time
    print(f"Artifact detection done: {n_channels} channels in {elapsed:.2f}s", flush=True)
    return artifact_timestamps


class LazyArtifactRepairRecordingSegment(BaseRecordingSegment):
    """
    Recording segment that patches artifact windows on-the-fly via PCHIP interpolation.

    No data is read or written at construction time. Repair happens only within
    the chunk requested by get_traces(), so the overhead is proportional to the
    number of artifact samples in that chunk, not the full recording length.
    """

    def __init__(self, parent_recording, seg_idx, artifact_timestamps, n_samples,
                 channel_ids, window, gap_pad_samples, correct_dc_offset, overlap,
                 dither=True, rng=None):
        super().__init__(sampling_frequency=parent_recording.get_sampling_frequency())
        self._parent = parent_recording
        self._seg_idx = seg_idx
        self._artifact_timestamps = artifact_timestamps  # list[np.ndarray], indexed by global channel pos
        self._n_samples = n_samples
        self._channel_ids = np.asarray(channel_ids)
        self._window = window
        self._gap_pad_samples = gap_pad_samples
        self._correct_dc_offset = correct_dc_offset
        self._overlap = overlap
        self._dither = dither
        self._rng = rng if rng is not None else np.random.default_rng()
        if dither:
            fs = parent_recording.get_sampling_frequency()
            nyq = fs / 2.0
            highcut = min(6000.0, nyq * 0.95)
            sos = butter(4, [300.0 / nyq, highcut / nyq], btype='bandpass', output='sos')

            # Cache per-channel baseline sigma once at construction from a short
            # representative chunk (~10 s) so sosfiltfilt is never called during sorting.
            probe_frames = min(n_samples, int(10.0 * fs))
            probe = parent_recording.get_traces(
                start_frame=0, end_frame=probe_frames,
                segment_index=seg_idx,
                channel_ids=list(channel_ids),
                return_scaled=True,
            ).astype(np.float32)
            self._baseline_sigmas = np.empty(len(channel_ids), dtype=np.float32)
            for ci in range(len(channel_ids)):
                ts = artifact_timestamps[ci]
                art_mask = np.zeros(probe_frames, dtype=bool)
                in_probe = ts[ts < probe_frames]
                if len(in_probe):
                    art_mask[in_probe] = True
                clean = probe[~art_mask, ci]
                if len(clean) > 27:
                    hp = sosfiltfilt(sos, clean)
                    self._baseline_sigmas[ci] = np.median(np.abs(hp - np.median(hp))) / 0.6745
                else:
                    self._baseline_sigmas[ci] = np.std(clean) if len(clean) > 1 else 0.0

        # Accumulated timing counters (seconds)
        self._t_io          = 0.0   # parent get_traces (disk/NWB read)
        self._t_pchip       = 0.0   # PCHIP interpolation
        self._t_dither      = 0.0   # noise injection
        self._n_repair_calls = 0    # get_traces calls that hit the repair path

    def get_num_samples(self):
        return self._n_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None or (isinstance(channel_indices, slice) and channel_indices == slice(None)):
            ch_ids = self._channel_ids
            global_indices = np.arange(len(self._channel_ids))
        elif isinstance(channel_indices, slice):
            global_indices = np.arange(len(self._channel_ids))[channel_indices]
            ch_ids = self._channel_ids[global_indices]
        else:
            global_indices = np.asarray(channel_indices)
            ch_ids = self._channel_ids[global_indices]

        load_start = max(0, start_frame - self._overlap)
        load_end = min(self._n_samples, end_frame + self._overlap)

        # Fast path: check whether any artifact touches this extended window
        has_artifacts = False
        for gi in global_indices:
            ts = self._artifact_timestamps[gi]
            if len(ts) > 0:
                lo = np.searchsorted(ts, load_start)
                hi = np.searchsorted(ts, load_end)
                if lo < hi:
                    has_artifacts = True
                    break

        if not has_artifacts:
            raw = self._parent.get_traces(
                start_frame=start_frame, end_frame=end_frame,
                segment_index=self._seg_idx,
                channel_ids=ch_ids,
                return_scaled=True,
            )
            return raw.astype(np.float32)

        # Load with context needed for anchor windows and DC correction
        _t0 = time.perf_counter()
        raw = self._parent.get_traces(
            start_frame=load_start, end_frame=load_end,
            segment_index=self._seg_idx,
            channel_ids=ch_ids,
            return_scaled=True,
        )
        self._t_io += time.perf_counter() - _t0

        traces = raw.copy().astype(np.float32)
        n = traces.shape[0]
        write_start = start_frame - load_start
        write_end = end_frame - load_start
        self._n_repair_calls += 1

        for local_ch_idx, gi in enumerate(global_indices):
            ts_global = self._artifact_timestamps[gi]
            if len(ts_global) == 0:
                continue
            lo = np.searchsorted(ts_global, load_start)
            hi = np.searchsorted(ts_global, load_end)
            if lo >= hi:
                continue
            ts_local = ts_global[lo:hi] - load_start

            seg = traces[:, local_ch_idx]
            split_indices = np.where(np.diff(ts_local) > 1)[0] + 1
            groups = np.split(ts_local, split_indices)

            # Look up the pre-cached per-channel sigma (estimated once at construction).
            if self._dither:
                baseline_sigma = float(self._baseline_sigmas[gi])

            for group in groups:
                art_start = int(group[0])
                art_end   = int(group[-1])
                win_start = max(0, art_start - self._gap_pad_samples)
                win_end   = min(n - 1, art_end + self._gap_pad_samples)
                pre_start = max(0, win_start - self._window)
                post_end  = min(n, win_end + 1 + self._window)

                if (win_start - pre_start) < 2 or (post_end - win_end - 1) < 2:
                    continue

                if self._correct_dc_offset:
                    pre_level  = np.median(seg[pre_start:win_start])
                    post_level = np.median(seg[win_end + 1:post_end])
                    seg[win_end + 1:] -= (post_level - pre_level)

                anchor_x = np.concatenate([
                    np.arange(pre_start, win_start),
                    np.arange(win_end + 1, post_end),
                ])
                gap_idx = np.arange(win_start, win_end + 1)
                _t0 = time.perf_counter()
                interp_fn = PchipInterpolator(anchor_x, seg[anchor_x])
                seg[gap_idx] = interp_fn(gap_idx)
                self._t_pchip += time.perf_counter() - _t0

                if self._dither and baseline_sigma > 0:
                    _t0 = time.perf_counter()
                    seg[gap_idx] += self._rng.normal(0.0, baseline_sigma, size=len(gap_idx))
                    self._t_dither += time.perf_counter() - _t0

        return traces[write_start:write_end].astype(np.float32)


class LazyArtifactRepairRecording(BaseRecording):
    """
    Lazily repairs pre-detected artifact windows on-the-fly.

    Wraps an existing recording and patches artifact samples with PCHIP
    interpolation only in the chunks actually requested by downstream code
    (sorter, bandpass filter, CMR, etc.). No memmap file is written and no
    data is read at construction time.

    Parameters
    ----------
    recording : BaseRecording
    artifact_timestamps : list of np.ndarray
        Per-channel artifact sample indices (output of detect_artifacts_recording).
    window : int
        PCHIP anchor samples on each side.
    gap_pad_samples : int
        Extra exclusion samples around artifact edges.
    correct_dc_offset : bool
        Shift post-artifact data within each chunk to match pre-artifact baseline.
    """

    def __init__(self, recording, artifact_timestamps, window=50,
                 gap_pad_samples=2, correct_dc_offset=True, dither=True, rng=None):
        # Normalise to numpy arrays (input may be lists when reconstructed from _kwargs)
        artifact_timestamps = [np.asarray(ts) for ts in artifact_timestamps]
        channel_ids = recording.get_channel_ids()
        super().__init__(
            sampling_frequency=recording.get_sampling_frequency(),
            channel_ids=channel_ids,
            dtype=np.float32,
        )

        # Copy spatial/group properties but not gain/offset.
        # Our segment returns pre-scaled float32, so we set trivial gain/offset.
        for key in recording.get_property_keys():
            if key not in ('gain_to_uV', 'offset_to_uV'):
                val = recording.get_property(key)
                if val is not None:
                    self.set_property(key, val)
        n_ch = len(channel_ids)
        self.set_property('gain_to_uV', np.ones(n_ch, dtype=np.float32))
        self.set_property('offset_to_uV', np.zeros(n_ch, dtype=np.float32))

        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()

        overlap = window + gap_pad_samples
        for seg_idx in range(recording.get_num_segments()):
            n_samples = recording.get_num_samples(segment_index=seg_idx)
            self.add_recording_segment(
                LazyArtifactRepairRecordingSegment(
                    parent_recording=recording,
                    seg_idx=seg_idx,
                    artifact_timestamps=artifact_timestamps,
                    n_samples=n_samples,
                    channel_ids=channel_ids,
                    window=window,
                    gap_pad_samples=gap_pad_samples,
                    correct_dc_offset=correct_dc_offset,
                    overlap=overlap,
                    dither=dither,
                    rng=rng,
                )
            )

        self._kwargs = dict(
            recording=recording,
            artifact_timestamps=[ts.tolist() for ts in artifact_timestamps],
            window=window,
            gap_pad_samples=gap_pad_samples,
            correct_dc_offset=correct_dc_offset,
            dither=dither,
        )

    def print_timing_report(self):
        """Print accumulated timing breakdown across all segments."""
        t_io = t_pchip = t_dither = n_calls = 0
        for seg in self._recording_segments:
            t_io    += seg._t_io
            t_pchip += seg._t_pchip
            t_dither += seg._t_dither
            n_calls += seg._n_repair_calls
        print("\n[TIMING] LazyArtifactRepair — accumulated over sorting run")
        print(f"  repair get_traces calls : {n_calls}")
        print(f"  IO (parent get_traces)  : {t_io:.2f}s")
        print(f"  PCHIP interpolation     : {t_pchip:.2f}s")
        print(f"  dither noise injection  : {t_dither:.2f}s")
        print(f"  total accounted         : {t_io + t_pchip + t_dither:.2f}s")


def repair_from_timestamps(
    rec_obj,
    artifact_timestamps,
    window=50,
    gap_pad_samples=2,
    correct_dc_offset=True,
    memmap_folder=None,
    dtype=np.float32,
    n_jobs=-1,
    batch_size=None,
    time_batch_sec=None,
):
    """
    Repair artifacts using pre-computed timestamps (skip detection).

    Faster than repair_artifacts_recording when timestamps are already known
    (e.g. loaded from a cache). Uses the same memmap + joblib pipeline.

    Parameters
    ----------
    rec_obj : BaseRecording
    artifact_timestamps : list of np.ndarray
        Per-channel artifact sample indices (length n_channels).
    All other parameters match repair_artifacts_recording.

    Returns
    -------
    rec_repaired : NumpyRecording
    cleanup_handle : MemmapCleanupHandle
    """
    fs = rec_obj.get_sampling_frequency()
    n_samples = rec_obj.get_num_frames()
    n_channels = rec_obj.get_num_channels()
    channel_ids = rec_obj.get_channel_ids()

    created_temp_dir = memmap_folder is None
    if created_temp_dir:
        memmap_folder = tempfile.mkdtemp(prefix="artifact_repair_memmap_")
    os.makedirs(memmap_folder, exist_ok=True)
    memmap_path = os.path.join(memmap_folder, "repaired_traces.dat")
    traces_out = np.memmap(memmap_path, mode="w+", dtype=dtype,
                           shape=(n_samples, n_channels))

    if batch_size is None:
        batch_size = n_channels

    if time_batch_sec is not None:
        time_batch_samples = int(time_batch_sec * fs)
        overlap_samples = window + gap_pad_samples
    else:
        time_batch_samples = n_samples
        overlap_samples = 0

    use_parallel = n_jobs != 1
    print(f"Artifact repair (from cached timestamps): {n_channels} channels, {n_samples} samples")

    def _repair_worker(local_idx, traces_batch, ts_global, load_start, t_start, t_end):
        ch = traces_batch[:, local_idx].copy().astype(np.float64)
        n = len(ch)
        local_write_start = t_start - load_start
        local_write_end = t_end - load_start
        # Translate global timestamps to local batch indices
        ts_local = ts_global[(ts_global >= load_start) & (ts_global < t_end + overlap_samples)]
        ts_local = ts_local - load_start
        if len(ts_local) == 0:
            return local_idx, ch[local_write_start:local_write_end].astype(dtype)
        split_indices = np.where(np.diff(ts_local) > 1)[0] + 1
        groups = np.split(ts_local, split_indices)
        for group in groups:
            art_start = int(group[0])
            art_end = int(group[-1])
            win_start = max(0, art_start - gap_pad_samples)
            win_end = min(n - 1, art_end + gap_pad_samples)
            pre_start = max(0, win_start - window)
            post_end = min(n, win_end + 1 + window)
            if (win_start - pre_start) < 2 or (post_end - win_end - 1) < 2:
                continue
            if correct_dc_offset:
                pre_level = np.median(ch[pre_start:win_start])
                post_level = np.median(ch[win_end + 1:post_end])
                ch[win_end + 1:] -= (post_level - pre_level)
            anchor_x = np.concatenate([
                np.arange(pre_start, win_start),
                np.arange(win_end + 1, post_end),
            ])
            anchor_y = ch[anchor_x]
            interp_fn = PchipInterpolator(anchor_x, anchor_y)
            ch[np.arange(win_start, win_end + 1)] = interp_fn(np.arange(win_start, win_end + 1))
        return local_idx, ch[local_write_start:local_write_end].astype(dtype)

    start_time = time.time()
    time_batches = list(range(0, n_samples, time_batch_samples))
    pbar = tqdm(time_batches, desc="Repairing artifacts", unit="batch")
    for t_start in pbar:
        t_end = min(t_start + time_batch_samples, n_samples)
        load_start = max(0, t_start - overlap_samples)
        load_end = min(n_samples, t_end + overlap_samples)
        pbar.set_postfix_str(f"{t_start/fs:.0f}s – {t_end/fs:.0f}s")
        for batch_start in range(0, n_channels, batch_size):
            batch_end = min(batch_start + batch_size, n_channels)
            batch_ch_ids = [channel_ids[i] for i in range(batch_start, batch_end)]
            traces_batch = rec_obj.get_traces(
                start_frame=load_start, end_frame=load_end,
                channel_ids=batch_ch_ids, return_scaled=True,
            )
            if use_parallel:
                results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(_repair_worker)(
                        i - batch_start, traces_batch,
                        artifact_timestamps[i], load_start, t_start, t_end,
                    )
                    for i in range(batch_start, batch_end)
                )
                for local_idx, data in results:
                    traces_out[t_start:t_end, batch_start + local_idx] = data
            else:
                for local_idx in range(batch_end - batch_start):
                    global_idx = batch_start + local_idx
                    _, data = _repair_worker(
                        local_idx, traces_batch,
                        artifact_timestamps[global_idx], load_start, t_start, t_end,
                    )
                    traces_out[t_start:t_end, global_idx] = data

    elapsed = time.time() - start_time
    print(f"Artifact repair done: {n_channels} channels in {elapsed:.2f}s")

    rec_repaired = NumpyRecording(traces_out, sampling_frequency=fs, channel_ids=channel_ids)
    copy_recording_properties(rec_obj, rec_repaired)
    cleanup_handle = MemmapCleanupHandle(
        memmap_path, memmap_folder, remove_folder=created_temp_dir,
    )
    return rec_repaired, cleanup_handle


def repair_artifacts_recording(
    rec_obj,
    detection_method='slope',
    slope_threshold=500,
    rolling_window_size=50,
    rolling_z_threshold=5,
    window=50,
    gap_pad_samples=2,
    correct_dc_offset=True,
    memmap_folder=None,
    dtype=np.float32,
    n_jobs=-1,
    batch_size=None,
    time_batch_sec=None,
    sequential_channels=False,
):
    """
    Detect and repair artifacts in a large recording, reading it only once.

    Combines detection and repair into a single memmap-backed pipeline suitable
    for recordings that do not fit in RAM. Channels are loaded in batches (one
    get_traces call per batch) and processed in parallel with joblib.

    Two detection methods are supported:
    - ``'slope'``: flags samples where the absolute sample-to-sample difference
      exceeds ``slope_threshold``. Fast and simple.
    - ``'rolling_std'``: flags samples where the rolling-std Z-score exceeds
      ``rolling_z_threshold``. More robust to varying baseline noise.

    Parameters
    ----------
    rec_obj : BaseRecording
        Input SpikeInterface recording.
    detection_method : str, default 'slope'
        Artifact detection method: ``'slope'`` or ``'rolling_std'``.
    slope_threshold : float, default 500
        Max allowed sample-to-sample change (used when detection_method='slope').
    rolling_window_size : int, default 50
        Rolling window size in samples (used when detection_method='rolling_std').
    rolling_z_threshold : float, default 5
        Z-score threshold (used when detection_method='rolling_std').
    window : int, default 50
        Number of clean samples on each side of the artifact used as PCHIP anchors.
    gap_pad_samples : int, default 2
        Extra samples excluded on each side of the artifact window to avoid
        edge transients entering the anchor fit.
    correct_dc_offset : bool, default True
        Shift post-artifact baseline to match pre-artifact level before
        interpolating. Recommended for stimulation artifacts.
    memmap_folder : str, optional
        Directory for the output memmap file. If None, a temporary directory
        is created automatically and removed on cleanup.
    dtype : numpy dtype, default np.float32
        Output data type.
    n_jobs : int, default -1
        Number of parallel joblib workers. -1 = all CPU cores, 1 = serial.
    batch_size : int, optional
        Number of channels to load per get_traces call. None = all channels
        at once. Use a smaller value when memory is limited.
    time_batch_sec : float, optional
        Process the recording in time chunks of this many seconds. None = all
        samples at once. Use (e.g. 1200) when the full recording does not fit
        in RAM. Adjacent chunks overlap by ``max(window+gap_pad, rolling_window)``
        samples so artifacts at chunk boundaries are handled correctly.
        Note: DC offset correction is applied within each chunk independently.
    sequential_channels : bool, default False
        If True, process each channel's full recording (across all time batches)
        before moving to the next channel. Always runs serially (ignores n_jobs).
        Use this when memory per channel is tight or when you want predictable,
        ordered I/O. If False (default), the outer loop is time batches and the
        inner loop is channel batches (can run in parallel with joblib).

    Returns
    -------
    rec_repaired : NumpyRecording
        Artifact-repaired recording with the same sampling rate and channel IDs.
    cleanup_handle : MemmapCleanupHandle
        Call cleanup_handle() to delete the temporary memmap file when done.
    artifact_timestamps : list of np.ndarray
        Per-channel detected artifact sample indices, length n_channels.
    """
    fs = rec_obj.get_sampling_frequency()
    n_samples = rec_obj.get_num_frames()
    n_channels = rec_obj.get_num_channels()
    channel_ids = rec_obj.get_channel_ids()

    # Setup output memmap
    created_temp_dir = memmap_folder is None
    if created_temp_dir:
        memmap_folder = tempfile.mkdtemp(prefix="artifact_repair_memmap_")
    os.makedirs(memmap_folder, exist_ok=True)
    memmap_path = os.path.join(memmap_folder, "repaired_traces.dat")
    traces_out = np.memmap(memmap_path, mode="w+", dtype=dtype,
                           shape=(n_samples, n_channels))

    if batch_size is None:
        batch_size = n_channels
    use_parallel = n_jobs != 1

    print(f"Artifact detection + repair: {n_channels} channels, {n_samples} samples")
    print(f"  detection_method={detection_method}", end="")
    if detection_method == 'slope':
        print(f", slope_threshold={slope_threshold}")
    else:
        print(f", rolling_window={rolling_window_size}, z_threshold={rolling_z_threshold}")
    print(f"  window={window}, gap_pad={gap_pad_samples}, dc_offset={correct_dc_offset}")
    print(f"  Mode: {'sequential_channels' if sequential_channels else ('parallel' if use_parallel else 'serial')}")
    print(f"  Memmap: {memmap_path}")

    # Time batch setup
    if time_batch_sec is not None:
        time_batch_samples = int(time_batch_sec * fs)
        overlap_samples = max(
            window + gap_pad_samples,
            rolling_window_size if detection_method == 'rolling_std' else 0,
        )
    else:
        time_batch_samples = n_samples
        overlap_samples = 0

    artifact_timestamps = [[] for _ in range(n_channels)]

    start_time = time.time()

    if sequential_channels:
        # Channel-outer, time-inner: finish each channel's whole recording before next.
        # Always serial — one get_traces call per (channel, time-batch) pair.
        for global_idx in tqdm(range(n_channels), desc="Repairing channels", unit="ch"):
            ch_id = channel_ids[global_idx]

            for t_start in range(0, n_samples, time_batch_samples):
                t_end = min(t_start + time_batch_samples, n_samples)
                load_start = max(0, t_start - overlap_samples)
                load_end = min(n_samples, t_end + overlap_samples)
                local_write_start = t_start - load_start
                local_write_end = t_end - load_start

                traces_batch = rec_obj.get_traces(
                    start_frame=load_start, end_frame=load_end,
                    channel_ids=[ch_id], return_scaled=True,
                )
                ch = traces_batch[:, 0].copy()
                result, ts = _repair_artifacts_single_channel(
                    ch, slope_threshold, window, gap_pad_samples, correct_dc_offset,
                    detection_method, rolling_window_size, rolling_z_threshold,
                )
                traces_out[t_start:t_end, global_idx] = result[local_write_start:local_write_end].astype(dtype)
                ts_core = ts[(ts >= local_write_start) & (ts < local_write_end)]
                artifact_timestamps[global_idx].append(ts_core + load_start)

    else:
        # Time-outer, channel-inner: load a channel batch for each time window.
        # Supports parallel processing via joblib.
        time_batches = list(range(0, n_samples, time_batch_samples))
        pbar = tqdm(time_batches, desc="Repairing artifacts", unit="batch")
        for t_start in pbar:
            t_end = min(t_start + time_batch_samples, n_samples)
            load_start = max(0, t_start - overlap_samples)
            load_end = min(n_samples, t_end + overlap_samples)
            local_write_start = t_start - load_start
            local_write_end = t_end - load_start
            pbar.set_postfix_str(f"{t_start/fs:.0f}s – {t_end/fs:.0f}s")

            for batch_start in range(0, n_channels, batch_size):
                batch_end = min(batch_start + batch_size, n_channels)
                batch_ch_ids = [channel_ids[i] for i in range(batch_start, batch_end)]
                batch_indices = list(range(batch_start, batch_end))

                traces_batch = rec_obj.get_traces(
                    start_frame=load_start, end_frame=load_end,
                    channel_ids=batch_ch_ids, return_scaled=True,
                )

                if use_parallel:
                    results = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(_repair_artifacts_channel_worker)(
                            i - batch_start, traces_batch,
                            slope_threshold, window, gap_pad_samples,
                            correct_dc_offset, dtype,
                            detection_method, rolling_window_size, rolling_z_threshold,
                        )
                        for i in batch_indices
                    )
                    for local_idx, data, ts in results:
                        global_idx = batch_start + local_idx
                        traces_out[t_start:t_end, global_idx] = data[local_write_start:local_write_end]
                        ts_core = ts[(ts >= local_write_start) & (ts < local_write_end)]
                        artifact_timestamps[global_idx].append(ts_core + load_start)
                else:
                    for local_idx in range(batch_end - batch_start):
                        global_idx = batch_start + local_idx
                        ch = traces_batch[:, local_idx].copy()
                        result, ts = _repair_artifacts_single_channel(
                            ch, slope_threshold, window, gap_pad_samples, correct_dc_offset,
                            detection_method, rolling_window_size, rolling_z_threshold,
                        )
                        traces_out[t_start:t_end, global_idx] = result[local_write_start:local_write_end].astype(dtype)
                        ts_core = ts[(ts >= local_write_start) & (ts < local_write_end)]
                        artifact_timestamps[global_idx].append(ts_core + load_start)

                        if (local_idx + 1) % 10 == 0 or local_idx == batch_end - batch_start - 1:
                            print(f"  Processed {global_idx + 1}/{n_channels} channels",
                                  end="\r")

    artifact_timestamps = [
        np.concatenate(ts_list) if ts_list else np.array([], dtype=np.int64)
        for ts_list in artifact_timestamps
    ]

    elapsed = time.time() - start_time
    print(f"\nArtifact repair done: {n_channels} channels in {elapsed:.2f}s")

    rec_repaired = NumpyRecording(
        traces_out, sampling_frequency=fs, channel_ids=channel_ids,
    )
    copy_recording_properties(rec_obj, rec_repaired)
    cleanup_handle = MemmapCleanupHandle(
        memmap_path, memmap_folder, remove_folder=created_temp_dir,
    )
    return rec_repaired, cleanup_handle, artifact_timestamps