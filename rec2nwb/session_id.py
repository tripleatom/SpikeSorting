"""
Session-name parsing, kept free of heavy imports.

``parse_session_info`` is pure regex, but it used to live in preproc_func.py,
whose module-level ``import spikeinterface.preprocessing`` meant that anything
wanting to turn a folder name into (animal, session) had to load the whole
sorting stack. pipeline_gui.py needs exactly this one function, so it lives
here; preproc_func re-exports it and every existing caller is unaffected.
"""

from __future__ import annotations

import os
import re


def parse_session_info(rec_folder: str) -> tuple:
    r"""
    Extract animal ID, session ID, and folder name from a recording folder path.

    Supports folder names such as:
      1. \\10.129.151.108\xieluanlabs\xl_cl\ephys\CnL14_20240915_161250.rec
      2. \\10.129.151.108\xieluanlabs\xl_cl\rf_reconstruction\head_fixed\CNL35\CNL35_250305_191757

    Args:
        rec_folder (str): Path to the recording folder.

    Returns:
        tuple: (animal_id, session_id, folder_name)
    """
    rec_folder = str(rec_folder)
    basename = os.path.basename(rec_folder.rstrip("\\/"))
    pattern = r'([A-Za-z]+\d+)_(\d{6,8}_\d{6})(?:\.rec)?$'
    match = re.search(pattern, basename)
    if match:
        animal_id, session_id = match.groups()
        return animal_id, session_id, f"{animal_id}_{session_id}"

    # Fallback: remove '.rec' if present and split by underscore
    parts = basename.replace('.rec', '').split('_')
    if len(parts) >= 2:
        animal_id = parts[0]
        session_id = '_'.join(parts[1:])
        return animal_id, session_id, f"{animal_id}_{session_id}"

    raise ValueError("Recording folder name doesn't match the expected format.")
