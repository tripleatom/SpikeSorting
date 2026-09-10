"""
Channel-map loading and electrode DataFrame construction.
"""

import json
import numpy as np
import pandas as pd
from rec2nwb.probes import load_probes, probe_for_channels


def get_all_shanks(device_type: str) -> list:
    """
    Return every shank index present in the device's channel map, sorted.

    Used as the default when the user does not name specific shanks.
    """
    return sorted({int(sh) for p in load_probes(device_type)
                   for sh in p.shank_ids[p.device_channel_indices >= 0]})


def get_ch_index_on_shank(ishank: int, device_type: str) -> tuple:
    """
    Return channel indices and probe coordinates for a given shank.

    Returns:
        (channel_indices, x_coords, y_coords)
    """
    rows = []
    for probe in load_probes(device_type):
        selected = (probe.shank_ids.astype(int) == ishank) & (probe.device_channel_indices >= 0)
        rows.extend(zip(probe.device_channel_indices[selected], *probe.contact_positions[selected].T))
    rows.sort()
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    channels, x, y = zip(*rows)
    return np.array(channels, dtype=int), np.array(x), np.array(y)


def build_electrode_df(channel_index: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray,
                       recording_method: str, impedance_table: pd.DataFrame = None,
                       bad_ch_ids: list = None, device_type: str = None) -> pd.DataFrame:
    """
    Build an electrode DataFrame for one shank, optionally filtering bad channels.

    Args:
        channel_index: Indices of channels on the shank.
        xcoord: X probe coordinates for those channels.
        ycoord: Y probe coordinates for those channels.
        recording_method: 'intan', 'spikegadget', or 'spikegadget_rec'.
        impedance_table: DataFrame from an impedance CSV (optional).
        bad_ch_ids: Channel names to exclude (optional).
        device_type: Probe map name; includes contact geometry when supplied.

    Returns:
        DataFrame with channel_name, impedance, x, y, channel_index, plus
        contact_id, contact_shape, contact_shape_params, shank_id when requested.
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

    if device_type is not None and len(channel_index):
        probe = probe_for_channels(device_type, channel_index)
        electrode_df['contact_id'] = probe.contact_ids
        electrode_df['contact_shape'] = probe.contact_shapes
        electrode_df['contact_shape_params'] = [json.dumps(p) for p in probe.contact_shape_params]
        electrode_df['shank_id'] = probe.shank_ids

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
