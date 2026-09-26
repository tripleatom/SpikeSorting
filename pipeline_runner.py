"""
Child-process entry points for pipeline_gui.py
==============================================

The GUI writes a JSON config to disk, then runs one of:

    python pipeline_runner.py rec2nwb <config.json>
    python pipeline_runner.py mssort  <config.json>
    python pipeline_runner.py archive <config.json>

The heavy steps run out-of-process so the window stays responsive, a run can
be stopped by killing the process tree, and SpikeInterface's worker pool is
spawned from a plain module instead of from a Tk app.

Config formats are exactly the ones the existing scripts already accept:
    rec2nwb -> the dict rec2nwb_interp.process_folder() takes
               (see rec2nwb/batch_config.json for an example)
    mssort  -> the MsSortingFiles.json schema MsSorting.process_from_json() reads
    archive -> {"data_folder", "device_type", "impedance_path",
                "tools_root", "delete_raw"}; see run_archive()
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


def run_dio(config_path: Path) -> None:
    """``trodesexport -dio`` over every .rec, writing the gap .txt sidecars.

    The same work as the GUI's step 1, driven headlessly so a batch can run it.

    Config keys:
        data_folder    the recording folder
        trodes_exe     path to trodesexport.exe
        interp         gap size (packets) above which a gap is called out
        skip_existing  leave .rec files that already have a .txt sidecar
    """
    import queue as _queue

    from rec2nwb.trodes_dio_gui import (ExtractWorker, discover_rec_files,
                                        write_gap_files)

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    folder = Path(config["data_folder"])
    exe = config["trodes_exe"]
    if not exe or not Path(exe).exists():
        raise FileNotFoundError(f"trodesexport.exe not found: {exe or '(not set)'}")

    rec_files = discover_rec_files(folder)
    if not rec_files:
        raise RuntimeError(f"No .rec files in {folder} (or its *.rec subfolders)")
    skip_existing = bool(config.get("skip_existing", True))
    todo = ([f for f in rec_files if not (f.parent / (f.name + ".txt")).exists()]
            if skip_existing else rec_files)
    if not todo:
        print("All .rec files already have gap sidecars; nothing to export.")
        return
    print(f"Exporting DIO + timestamps from {len(todo)} of {len(rec_files)} .rec file(s)")

    out_q: _queue.Queue = _queue.Queue()
    worker = ExtractWorker(exe, todo, int(config.get("interp", 100)), out_q)
    worker.start()
    results = []
    while True:
        kind, payload = out_q.get()
        if kind == "log":
            print(payload, flush=True)
        elif kind == "file":
            results.append(payload)
            print(f"  {len(results)}/{len(todo)} done", flush=True)
        elif kind == "done":
            break

    failed = [r for r in results if r["returncode"] != 0]
    if failed:
        raise RuntimeError("trodesexport failed on: "
                           + ", ".join(f"{r['rec'].name} (exit {r['returncode']})"
                                       for r in failed))
    total_gaps = sum(len(r["gaps"]) for r in results)
    missing = sum(n for r in results for _, n in r["gaps"])
    print(f"{total_gaps} gap(s) across {len(results)} file(s); "
          f"{missing} sample(s) will be PCHIP-filled during conversion.")
    written, skipped, errors = write_gap_files(results, overwrite=not skip_existing)
    print(f"Wrote {written} .txt file(s); skipped {skipped}; {len(errors)} error(s).")
    for err in errors:
        print(f"  {err}")
    if errors:
        raise RuntimeError(f"{len(errors)} gap file(s) could not be written")


def run_archive(config_path: Path) -> None:
    """Move a finished session to the server, then drop its raw .rec traces.

    Everything but the raw traces is copied to the session's server folder and
    each NWB is re-read there against conversion_list.txt; the traces are deleted
    only once every one of them checks out. This has to run after sorting, which
    reads the NWBs from the recording folder this step empties.

    The copy/verify/delete logic is ContinualLearning's
    ``data_collection/migrate_local_session.py``, and the destination comes from
    its ``server_fallback.server_session_folder`` -- the same local -> server
    mapping the sleep pipeline reads from -- so both stay single sources of truth.

    Config keys:
        data_folder     the recording folder, named <animal>_<YYYYMMDD>
        device_type     probe map, used to decide which shanks must have an NWB
        impedance_path  optional impedance CSV, as for rec2nwb
        tools_root      the ContinualLearning repository folder
        delete_raw      delete the raw traces once the NWBs verify (else keep them)
    """
    import pandas as pd
    from rec2nwb.utils.electrode import get_all_shanks, good_electrode_counts
    from rec2nwb.utils.file_io import load_bad_ch

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    folder = Path(config["data_folder"])
    device_type = config["device_type"]
    tools_root = Path(config["tools_root"])
    tools = tools_root / "data_collection"
    if not (tools / "migrate_local_session.py").is_file():
        raise FileNotFoundError(f"migrate_local_session.py not found under {tools}")
    for p in (tools_root, tools):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import migrate_local_session
    from server_fallback import server_session_folder

    # Never invent a destination: server_session_folder returns None rather than
    # guess at an animal folder it cannot find on the share.
    dest = server_session_folder(folder)
    if dest is None:
        raise RuntimeError(
            f"No server folder could be identified for {folder.name}: no animal folder "
            f"on the share matches it. Add it to SESSION_SERVER_FOLDERS in "
            f"{tools_root / 'server_fallback.py'} -- nothing was moved.")

    # A shank may be missing only if the conversion had nothing to write for it:
    # every one of its channels is in bad_channels.txt. Judging "expected" from
    # the shanks picked for this run instead would let a one-shank re-sort
    # authorise deleting the raw data behind every other shank.
    # bad_channels.txt travels with everything else, so after an interrupted run
    # it may already be on the server; without it every dead shank would look
    # live, and the session could never be finished.
    bad_file = folder / "bad_channels.txt"
    if not bad_file.is_file() and (dest / "bad_channels.txt").is_file():
        bad_file = dest / "bad_channels.txt"
    impedance = config.get("impedance_path")
    table = pd.read_csv(impedance) if impedance else None
    counts = good_electrode_counts(device_type, load_bad_ch(bad_file), table)
    shank_ids = get_all_shanks(device_type)
    expect = max(shank_ids) + 1
    dead = sorted(s for s, n in counts.items() if n == 0)
    ignore = frozenset(dead) | (frozenset(range(expect)) - frozenset(shank_ids))

    print(f"Good electrodes per shank ({device_type}): "
          + ", ".join(f"sh{s}={n}" for s, n in sorted(counts.items())))
    if dead:
        print(f"Dead shank(s) {dead}: every channel is in bad_channels.txt, "
              f"so no NWB is expected for them.")

    delete_raw = bool(config.get("delete_raw"))
    ok = migrate_local_session.migrate(folder, expect, ignore, apply=True,
                                       allow_delete=delete_raw, dest=dest)
    if not ok:
        raise RuntimeError("archive did not complete (see above) -- the raw .rec "
                           "traces were left in place.")


STAGES = {"dio": run_dio, "rec2nwb": run_rec2nwb, "mssort": run_mssort,
          "archive": run_archive}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in STAGES:
        print(__doc__)
        return 2
    STAGES[argv[1]](Path(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
