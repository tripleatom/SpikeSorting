"""
Re-exports, resolved on first use (PEP 562).

Importing these eagerly meant that ``import rec2nwb.utils.electrode`` -- which
only needs pandas to read a channel map -- also pulled in nwb_helpers, and with
it pynwb, h5py and hdmf. pipeline_gui.py reads channel maps to fill in the shank
list, so that made the GUI depend on the whole NWB stack. The names below still
import exactly as before; each submodule is now loaded when one of its names is
first touched rather than at package import.
"""

from importlib import import_module

_EXPORTS = {
    "get_data_files": "file_io",
    "setup_spikegadget_files": "file_io",
    "get_sampling_rate_from_params": "file_io",
    "load_bad_ch": "file_io",
    "get_geom_files": "file_io",
    "get_ch_index_on_shank": "electrode",
    "build_electrode_df": "electrode",
    "resolve_good_channel_ids": "electrode",
    "make_nwbfile": "nwb_helpers",
    "add_electrodes_to_nwb": "nwb_helpers",
    "make_electrical_series": "nwb_helpers",
    "append_nwb_dset": "nwb_helpers",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value  # cache, so the submodule loads once
    return value


def __dir__():
    return sorted(__all__)
