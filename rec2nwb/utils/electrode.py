"""
Channel-map loading and electrode DataFrame construction.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


def mapping_dir() -> Path:
    """Folder holding the per-device channel-map CSVs.

    Inside a PyInstaller build ``__file__`` points into the unpacked temp dir,
    so resolve against the .exe instead -- that keeps the maps editable on disk
    rather than frozen into the binary.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "rec2nwb" / "mapping"
    return Path(__file__).resolve().parent.parent / "mapping"


def _load_channel_map(device_type: str) -> pd.DataFrame:
    return pd.read_csv(mapping_dir() / f"{device_type}.csv")


def get_all_shanks(device_type: str) -> list:
    """
    Return every shank index present in the device's channel map, sorted.

    Used as the default when the user does not name specific shanks.
    """
    return sorted(int(s) for s in _load_channel_map(device_type)['sh'].unique())


def get_ch_index_on_shank(ishank: int, device_type: str) -> tuple:
    """
    Return channel indices and probe coordinates for a given shank.

    Returns:
        (channel_indices, x_coords, y_coords)
    """
    channel_map = _load_channel_map(device_type)

    xcoord = channel_map['xcoord'].astype(float).to_numpy()
    ycoord = channel_map['ycoord'].astype(float).to_numpy()
    sh = channel_map['sh'].astype(int).to_numpy()

    ch_index = np.where(sh == ishank)[0]
    return ch_index, xcoord[ch_index], ycoord[ch_index]


def build_electrode_df(channel_index: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray,
                       recording_method: str, impedance_table: pd.DataFrame = None,
                       bad_ch_ids: list = None) -> pd.DataFrame:
    """
    Build an electrode DataFrame for one shank, optionally filtering bad channels.

    Args:
        channel_index: Indices of channels on the shank.
        xcoord: X probe coordinates for those channels.
        ycoord: Y probe coordinates for those channels.
        recording_method: 'intan', 'spikegadget', or 'spikegadget_rec'.
        impedance_table: DataFrame from an impedance CSV (optional).
        bad_ch_ids: Channel names to exclude (optional).

    Returns:
        DataFrame with columns: channel_name, impedance, x, y, channel_index.
    """
    if impedance_table is not None:
        impedance_sh = impedance_table['Impedance Magnitude at 1000 Hz (ohms)'].to_numpy()[channel_index]
        channel_name_sh = impedance_table['Channel Name'].to_numpy()[channel_index]
    else:
        # spikegadget_rec uses bare numeric strings; others use "chN"
        if recording_method == 'spikegadget_rec':
            channel_name_sh = [str(i) for i in channel_index]
        else:
            channel_name_sh = [f"ch{i}" for i in channel_index]
        impedance_sh = [np.nan] * len(channel_index)

    electrode_df = pd.DataFrame({
        'channel_name': channel_name_sh,
        'impedance': impedance_sh,
        'x': xcoord,
        'y': ycoord,
        'channel_index': channel_index,
    })

    if bad_ch_ids:
        electrode_df = electrode_df[~electrode_df['channel_name'].isin(bad_ch_ids)]

    return electrode_df.reset_index(drop=True)


def resolve_good_channel_ids(electrode_df: pd.DataFrame, recording_method: str,
                              has_impedance: bool, actual_channel_ids=None) -> list:
    """
    Return the list of channel IDs to pass to recording.get_traces().

    For spikegadget_rec, validates against what the recording actually exposes.
    """
    if recording_method == 'spikegadget_rec':
        good_ids = []
        for idx in electrode_df['channel_index'].tolist():
            if actual_channel_ids is not None and str(idx) not in actual_channel_ids:
                print(f"Warning: Channel index {idx} not found in recording, skipping.")
                continue
            good_ids.append(idx)
        return good_ids

    if recording_method == 'intan':
        if not has_impedance and actual_channel_ids is not None:
            return [actual_channel_ids[i] for i in electrode_df['channel_index'].tolist()]
        return electrode_df['channel_name'].tolist()

    if has_impedance:
        return electrode_df['channel_name'].tolist()

    return electrode_df['channel_index'].tolist()
