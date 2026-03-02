import os
import tempfile
import time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from joblib import Parallel, delayed
from spikeinterface.core import BaseRecording, NumpyRecording


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
):
    """
    Repairs artifacts per channel using DC offset correction + windowed PCHIP interpolation.

    Per artifact per channel:
      1. Cluster consecutive flagged samples into contiguous groups.
      2. Optionally correct DC step offset by shifting all post-artifact data.
      3. Use window samples on each side as PCHIP anchors to interpolate across the gap.

    Args:
        data                (np.ndarray): Shape (n_timepoints, n_channels).
        artifact_timestamps (list[np.ndarray]): Output of get_artifact_timestamps_per_channel().
        window              (int): Number of clean samples on each side used as PCHIP anchors.
        gap_pad_samples     (int): Extra samples excluded on each side of the artifact window
                                   to avoid edge transients entering the anchor fit.
        correct_dc_offset   (bool): If True, shift post-artifact data to remove DC step before
                                    interpolating. Recommended for stimulation artifacts.

    Returns:
        cleaned_data (np.ndarray): Shape (n_timepoints, n_channels), artifact-repaired.
    """
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
        n_time_batches = (n_samples + time_batch_samples - 1) // time_batch_samples
        print(f"  Time batching: {n_time_batches} batches of ~{time_batch_sec}s "
              f"({time_batch_samples} samples), overlap={overlap_samples} samples")
    else:
        time_batch_samples = n_samples
        overlap_samples = 0
        n_time_batches = 1

    artifact_timestamps = [[] for _ in range(n_channels)]

    start_time = time.time()

    if sequential_channels:
        # Channel-outer, time-inner: finish each channel's whole recording before next.
        # Always serial — one get_traces call per (channel, time-batch) pair.
        for global_idx in range(n_channels):
            ch_id = channel_ids[global_idx]
            print(f"  Channel {global_idx + 1}/{n_channels} (id={ch_id})", end="\r")

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

        print()  # newline after \r progress

    else:
        # Time-outer, channel-inner: load a channel batch for each time window.
        # Supports parallel processing via joblib.
        for t_batch_idx, t_start in enumerate(range(0, n_samples, time_batch_samples)):
            t_end = min(t_start + time_batch_samples, n_samples)
            load_start = max(0, t_start - overlap_samples)
            load_end = min(n_samples, t_end + overlap_samples)
            local_write_start = t_start - load_start
            local_write_end = t_end - load_start

            if n_time_batches > 1:
                print(f"\n  Time batch {t_batch_idx + 1}/{n_time_batches}: "
                      f"samples {t_start}-{t_end} ({t_start / fs:.1f}s - {t_end / fs:.1f}s)")

            for batch_start in range(0, n_channels, batch_size):
                batch_end = min(batch_start + batch_size, n_channels)
                batch_ch_ids = [channel_ids[i] for i in range(batch_start, batch_end)]
                batch_indices = list(range(batch_start, batch_end))

                print(f"  Loading channels {batch_start}-{batch_end - 1}...", end="\r")
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