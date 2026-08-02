"""
Child-process entry points for pipeline_gui.py
==============================================

The GUI writes a JSON config to disk, then runs one of:

    python pipeline_runner.py rec2nwb <config.json>
    python pipeline_runner.py mssort  <config.json>

Both heavy steps run out-of-process so the window stays responsive, a run can
be stopped by killing the process tree, and SpikeInterface's worker pool is
spawned from a plain module instead of from a Tk app.

Config formats are exactly the ones the existing scripts already accept:
    rec2nwb -> the dict rec2nwb_interp.process_folder() takes
               (see rec2nwb/batch_config.json for an example)
    mssort  -> the MsSortingFiles.json schema MsSorting.process_from_json() reads
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Both scripts have imports that only resolve with their own folder on the
# path, not just the repo root: MsSorting.py does `from Timer import Timer`,
# and rec2nwb/process_func/DIO.py does `from process_func...`.
for _p in (REPO_ROOT, REPO_ROOT / "spikesorting", REPO_ROOT / "rec2nwb"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def run_rec2nwb(config_path: Path) -> None:
    from rec2nwb.rec2nwb_interp import process_folder

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    process_folder(config)


def run_mssort(config_path: Path) -> None:
    from spikesorting.MsSorting import process_from_json

    # process_from_json() joins its argument onto the spikesorting folder;
    # an absolute path passes straight through.
    process_from_json(str(config_path))


STAGES = {"rec2nwb": run_rec2nwb, "mssort": run_mssort}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in STAGES:
        print(__doc__)
        return 2
    STAGES[argv[1]](Path(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
