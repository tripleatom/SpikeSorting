"""
List all .rec files in a folder, sorted by recording date then part number,
and export a txt file with the number of timestamps in each file.

Usage:
    python list_rec_timestamps.py <folder_path> [output_txt_path]

Example:
    python list_rec_timestamps.py "\\\\server\\path\\to\\rec_folder"
    python list_rec_timestamps.py "\\\\server\\path\\to\\rec_folder" "output.txt"
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import spikeinterface.extractors as se


def _parse_datetime(filename: str) -> datetime:
    """Parse datetime from .rec filename. Expects _YYYYMMDD_HHMMSS pattern."""
    match = re.search(r'_(\d{8})_(\d{6})', filename)
    if match:
        return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return datetime.min


def _part_key(filename: str) -> tuple:
    """Sort key for part number: base file → (0, 0), partN → (1, N)."""
    match = re.search(r'\.part(\d+)\.rec$', filename)
    return (1, int(match.group(1))) if match else (0, 0)


def list_rec_timestamps(folder_path: str, output_txt: str = None) -> None:
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    # Collect .rec files directly in folder, or inside .rec subdirectories
    rec_files = [f for f in folder.glob('*.rec') if f.is_file()]
    for rec_dir in folder.glob('*.rec'):
        if rec_dir.is_dir():
            rec_files.extend(f for f in rec_dir.glob('*.rec') if f.is_file())

    if not rec_files:
        print(f"No .rec files found in {folder}")
        return

    # Sort: primary by datetime, secondary by part number
    rec_files.sort(key=lambda f: (_parse_datetime(f.name), _part_key(f.name)))

    output_path = Path(output_txt) if output_txt else folder / 'rec_timestamps.txt'

    lines = []
    for rec_file in rec_files:
        try:
            recording = se.read_spikegadgets(str(rec_file))
            n_frames = recording.get_num_frames()
            line = f"{rec_file.name}: {n_frames} timestamps"
        except Exception as e:
            line = f"{rec_file.name}: ERROR - {e}"
        print(line)
        lines.append(line)

    output_path.write_text('\n'.join(lines) + '\n')
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python list_rec_timestamps.py <folder_path> [output_txt_path]")
        sys.exit(1)
    list_rec_timestamps(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
