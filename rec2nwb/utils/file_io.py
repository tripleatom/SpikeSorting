"""
File discovery and SpikeGadgets setup utilities.
"""

import re
import json
import shutil
from datetime import datetime
from pathlib import Path


def get_data_files(data_folder: Path, recording_method: str) -> list:
    """
    Return a sorted list of data files in data_folder for the given recording_method.

    Args:
        data_folder: Folder to search.
        recording_method: 'intan', 'spikegadget', or 'spikegadget_rec'.
    """
    if recording_method == 'intan':
        data_files = sorted(
            p for p in data_folder.iterdir()
            if p.suffix.lower() in ('.rhd', '.rhs') and not p.name.startswith("._")
        )
        if not data_files:
            raise FileNotFoundError("No .rhd/.rhs files found in the specified folder.")
        return data_files

    if recording_method == 'spikegadget':
        ms_folders = list(data_folder.glob('*.mountainsort'))
        if not ms_folders:
            raise FileNotFoundError("No .mountainsort folders found in the specified folder.")

        def _part_key(f: Path):
            m = re.search(r'\.part(\d+)\.mountainsort$', f.name)
            return (1, int(m.group(1))) if m else (0, 0)

        ms_folders.sort(key=_part_key)
        data_files = [f for folder in ms_folders for f in folder.glob('*group0.mda')]
        if not data_files:
            raise FileNotFoundError("No group0.mda files found in the specified folder.")
        return data_files

    # spikegadget_rec
    rec_files = [f for f in data_folder.glob('*.rec') if f.is_file()]
    for rec_dir in data_folder.glob('*.rec'):
        if rec_dir.is_dir():
            rec_files.extend(f for f in rec_dir.glob('*.rec') if f.is_file())

    if not rec_files:
        raise FileNotFoundError("No .rec files found in the specified folder.")

    def _rec_sort_key(f: Path):
        m_dt = re.search(r'_(\d{8})_(\d{6})', f.name)
        dt = datetime.strptime(m_dt.group(1) + m_dt.group(2), "%Y%m%d%H%M%S") if m_dt else datetime.min
        m_part = re.search(r'\.part(\d+)\.rec$', f.name)
        part = (1, int(m_part.group(1))) if m_part else (0, 0)
        return (dt, part)

    return sorted(rec_files, key=_rec_sort_key)


def setup_spikegadget_files(data_file: Path, recording_method: str,
                             selected_geom: Path = None) -> None:
    """
    Copy params.json (and optionally geom.csv) next to the data file so
    SpikeInterface can find them.
    """
    script_dir = Path(__file__).resolve().parent.parent  # rec2nwb/
    params_path = script_dir / "params.json"

    if recording_method == 'spikegadget':
        mda_folder = data_file.parent
        geom_path = selected_geom if selected_geom is not None else script_dir / "geom.csv"
        shutil.copy2(params_path, mda_folder)
        shutil.copy2(geom_path, mda_folder / "geom.csv")
    elif recording_method == 'spikegadget_rec':
        shutil.copy2(params_path, data_file.parent)


def get_sampling_rate_from_params() -> float:
    """Read sampling rate from params.json located in rec2nwb/."""
    params_file = Path(__file__).resolve().parent.parent / "params.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
        return float(params.get("samplerate", 30000))
    print(f"Warning: params.json not found at {params_file}, using default 30000 Hz")
    return 30000.0


def load_bad_ch(bad_file: Path) -> list:
    """Load bad channel names from a text file (one per line)."""
    if not bad_file.exists():
        print(f"No bad channels file found at {bad_file}. Using all channels.")
        return []
    with open(bad_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_geom_files(geom_folder: Path) -> list:
    """Return sorted list of .csv geometry files in geom_folder."""
    if not geom_folder.exists():
        return []
    return sorted(geom_folder.glob("*.csv"))
