"""
NWB file construction helpers: creating NWBFile objects, adding electrodes,
building ElectricalSeries, and appending HDF5 datasets.
"""

from contextlib import contextmanager
from uuid import uuid4
import h5py
import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO

# Path to the ElectricalSeries data dataset inside an NWB HDF5 file.
_NWB_DSET_PATH = "acquisition/ElectricalSeries/data"


# ---------------------------------------------------------------------------
# NWBFile construction
# ---------------------------------------------------------------------------

def make_nwbfile(session_start_time, metadata: dict) -> NWBFile:
    """Create a bare NWBFile from a metadata dict."""
    return NWBFile(
        session_description=metadata.get("session_desc", "NWB recording"),
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        experimenter=[metadata.get("experimenter", "Zhang, Xiaorong")],
        lab=metadata.get("lab", "XL Lab"),
        institution=metadata.get("institution", "Rice University"),
        experiment_description=metadata.get("exp_desc", "None"),
        session_id=metadata.get("session_id", "None"),
    )


def add_electrodes_to_nwb(nwbfile: NWBFile, electrode_df: pd.DataFrame,
                           ishank: int, electrode_location: str):
    """
    Add a device, electrode group, and all electrodes from electrode_df to nwbfile.

    Returns:
        electrode_table_region for use in ElectricalSeries.
    """
    device = nwbfile.create_device(name="--", description="--", manufacturer="--")
    nwbfile.add_electrode_column(name="label", description="label of electrode")

    electrode_group = nwbfile.create_electrode_group(
        name=f"shank{ishank}",
        description=f"electrode group for shank {ishank}",
        device=device,
        location=electrode_location,
    )

    for _, row in electrode_df.iterrows():
        imp = float(row['impedance']) if not np.isnan(row['impedance']) else 0.0
        nwbfile.add_electrode(
            group=electrode_group,
            label=f"shank{ishank}:{row['channel_name']}",
            location=electrode_location,
            rel_x=float(row['x']),
            rel_y=float(row['y']),
            imp=imp,
        )

    return nwbfile.create_electrode_table_region(
        list(range(len(electrode_df))), "all electrodes"
    )


def make_electrical_series(data, electrode_table_region, sampling_freq: float,
                            conversion: float, offset: float,
                            chunked: bool = True) -> ElectricalSeries:
    """
    Wrap data in an ElectricalSeries with HDF5 compression.

    Args:
        data: numpy array (frames × channels).
        chunked: If True, use gzip compression and allow future resizing.
    """
    if chunked:
        h5_data = H5DataIO(data=data, maxshape=(None, data.shape[1]),
                           compression='gzip', compression_opts=4, chunks=True)
    else:
        h5_data = H5DataIO(data=data, maxshape=(None, data.shape[1]), chunks=True)

    return ElectricalSeries(
        name="ElectricalSeries",
        data=h5_data,
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=float(sampling_freq),
        conversion=conversion,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# HDF5 dataset append
# ---------------------------------------------------------------------------

def append_nwb_dset(dset, data_to_append, append_axis: int = 0) -> None:
    """Resize an HDF5 dataset and append data along append_axis."""
    old_len = dset.shape[append_axis]
    app_len = data_to_append.shape[append_axis]
    dset.resize(old_len + app_len, axis=append_axis)

    slicer = [slice(None)] * len(dset.shape)
    slicer[append_axis] = slice(old_len, None)
    dset[tuple(slicer)] = data_to_append


def append_chunk_to_nwb(nwb_path, chunk_data) -> None:
    """Open an NWB file in append mode and extend its ElectricalSeries."""
    with NWBHDF5IO(nwb_path, "a") as io:
        nwb_obj = io.read()
        append_nwb_dset(nwb_obj.acquisition['ElectricalSeries'].data, chunk_data)
        io.write(nwb_obj)


@contextmanager
def nwb_direct_writer(nwb_path):
    """Keep the NWB HDF5 file open for the duration, yielding a fast append callable.

    Avoids the per-chunk open / read-full-NWB-object / write / close overhead of
    append_chunk_to_nwb.  The file is opened once with h5py and kept open until
    the context exits.

    Usage::

        with nwb_direct_writer(path) as append_fn:
            for chunk in chunk_generator:
                append_fn(chunk)
    """
    with h5py.File(nwb_path, 'a') as f:
        dset = f[_NWB_DSET_PATH]
        def _append(chunk: np.ndarray) -> None:
            old = dset.shape[0]
            dset.resize(old + chunk.shape[0], axis=0)
            dset[old:] = chunk
        yield _append
