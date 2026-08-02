"""
Daily pipeline GUI:  DIO gaps  ->  NWB  ->  MountainSort5
=========================================================

One window for the three steps that get run on every day's recording:

  Step 1  rec2nwb/trodes_dio_gui.py    ``trodesexport -dio`` on every .rec file;
                                       writes the ``<recfile>.rec.txt`` gap
                                       sidecars *and* the .DIO folders that
                                       step 2 reads its first timestamp from.
  Step 2  rec2nwb/rec2nwb_interp.py    process_folder() -> ``<session>sh<N>.nwb``
  Step 3  spikesorting/MsSorting.py    process_from_json() -> sorting results
                                       under ``<sortout>/<animal>/...``

Recording folder, animal ID, device type and shank list are entered once and
shared by all three steps; per-step options live on their own tabs.  Any
subset of the steps can be run, in order, from one button.

Steps 2 and 3 run as child processes (pipeline_runner.py) so that a multi-hour
sort can be stopped without taking the window down.  Their console output —
progress bars included — is streamed into the log pane.  They run under the
interpreter named in "Python (steps 2, 3)", which defaults to this one but can
point at whichever conda environment has spikeinterface + mountainsort5;
"Check setup" verifies it before anything runs.

Every run writes ``pipeline_logs/<timestamp>_<session>/`` containing the log
and the exact JSON configs handed to each step.  Widget values are remembered
in pipeline_gui_settings.json.

    python pipeline_gui.py
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from rec2nwb.trodes_dio_gui import (
    ExtractWorker, ReviewDialog, discover_rec_files, find_trodesexport,
    write_gap_files,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Folder the repo's files are read from.

    Frozen by PyInstaller, ``__file__`` points inside the temporary unpack
    directory, so everything below (pipeline_runner.py, the channel maps,
    device_types.json) would be looked up in the wrong place. The .exe is meant
    to sit in the repo, so resolve against it instead.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


REPO_ROOT = _repo_root()
RUNNER = REPO_ROOT / "pipeline_runner.py"
SETTINGS_PATH = REPO_ROOT / "pipeline_gui_settings.json"
LOG_ROOT = REPO_ROOT / "pipeline_logs"
MAPPING_DIR = REPO_ROOT / "rec2nwb" / "mapping"
DEVICE_TYPES_PATH = REPO_ROOT / "rec2nwb" / "device_types.json"
MSSORT_DEFAULTS = REPO_ROOT / "spikesorting" / "MsSortingFiles.json"

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Keep the log pane from growing without bound over a multi-hour run.
LOG_MAX_LINES = 8000
LOG_TRIM_LINES = 2000


def child_python() -> str:
    """Interpreter for child processes (pythonw.exe cannot pipe stdout).

    Frozen, ``sys.executable`` is this .exe, which cannot run pipeline_runner.py
    -- steps 2 and 3 need the conda environment holding spikeinterface. Offer
    whatever python is on PATH as a starting guess; the saved setting normally
    replaces it and "Check setup" verifies the choice either way.
    """
    if getattr(sys, "frozen", False):
        return shutil.which("python") or ""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def missing_modules(python_exe: str, modules: list[str]) -> list[str] | None:
    """Return which of *modules* ``python_exe`` cannot import.

    None means the interpreter itself could not be run.  Steps 2 and 3 need a
    specific conda environment (spikeinterface, pynwb, mountainsort5), which is
    not necessarily the one this window was started from.
    """
    code = ("import importlib.util as u, sys;"
            "print(' '.join(m for m in sys.argv[1:] if u.find_spec(m) is None))")
    try:
        proc = subprocess.run([python_exe, "-c", code, *modules], capture_output=True,
                              text=True, timeout=120, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.split() if proc.returncode == 0 else None


def list_device_types() -> list[str]:
    return sorted(p.stem for p in MAPPING_DIR.glob("*.csv")
                  if not p.name.startswith("._"))


def load_device_map() -> dict:
    if DEVICE_TYPES_PATH.exists():
        return json.loads(DEVICE_TYPES_PATH.read_text())
    return {}


def remember_device_type(animal_id: str, device_type: str) -> bool:
    """Persist animal_id -> device_type. Returns True if the file changed."""
    device_map = load_device_map()
    if device_map.get(animal_id) == device_type:
        return False
    device_map[animal_id] = device_type
    DEVICE_TYPES_PATH.write_text(json.dumps(device_map, indent=4))
    return True


def all_shanks(device_type: str) -> list[int]:
    from rec2nwb.utils.electrode import get_all_shanks
    return get_all_shanks(device_type)


def parse_shanks(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def fmt_dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


def session_names(data_folder: Path) -> tuple[str, str]:
    """Return (name step 2 writes, name step 3 looks for) for the NWB files.

    These are derived by different code paths — ``data_folder.name`` versus
    ``parse_session_info()`` — and disagree for some folder names, which would
    silently leave step 3 with nothing to sort.
    """
    from rec2nwb.session_id import parse_session_info
    written = data_folder.name
    try:
        read = parse_session_info(str(data_folder))[2]
    except ValueError:
        read = ""
    return written, read


# ---------------------------------------------------------------------------
# Worker: run the enabled steps in order
# ---------------------------------------------------------------------------

class PipelineWorker(threading.Thread):
    """Runs the selected steps sequentially in a background thread.

    Messages put on the queue are tuples:
        ("log",    (text, tag, inplace))   -- a line for the log pane
        ("status", str)                    -- status bar text
        ("review", dict)                   -- pause: show the gap review dialog
        ("done",   bool)                   -- pipeline finished (True = all ok)
    """

    def __init__(self, cfg: dict, run_dir: Path, out_q: "queue.Queue"):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.run_dir = run_dir
        self.q = out_q
        self._stop_evt = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._dio_worker: ExtractWorker | None = None

    # -- messaging ----------------------------------------------------------

    def _log(self, msg: str = "", tag: str = None, inplace: bool = False):
        self.q.put(("log", (msg, tag, inplace)))

    def _status(self, msg: str):
        self.q.put(("status", msg))

    # -- lifecycle ----------------------------------------------------------

    def stop(self):
        self._stop_evt.set()
        if self._dio_worker is not None:
            self._dio_worker.stop()
        self._kill_child()

    def _kill_child(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if os.name == "nt":
            # Kill the whole tree: SpikeInterface/HDF5 leave worker processes.
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=_NO_WINDOW)
        else:
            proc.terminate()

    def run(self):
        steps = [(name, fn) for name, fn, enabled in (
            ("Trodes DIO -> gap .txt", self._step_dio, self.cfg["do_dio"]),
            ("rec -> NWB", self._step_nwb, self.cfg["do_nwb"]),
            ("MountainSort5", self._step_sort, self.cfg["do_sort"]),
        ) if enabled]

        ok = True
        t_all = time.time()
        try:
            for i, (name, fn) in enumerate(steps, 1):
                if self._stop_evt.is_set():
                    ok = False
                    break
                label = f"Step {i}/{len(steps)}  {name}"
                self._log("")
                self._log("=" * 70, "hdr")
                self._log(label, "hdr")
                self._log("=" * 70, "hdr")
                self._status(label)
                t0 = time.time()
                try:
                    step_ok = fn()
                except Exception as e:  # noqa: BLE001 - report, don't crash the GUI
                    self._log(f"{type(e).__name__}: {e}", "err")
                    for line in traceback.format_exc().splitlines():
                        self._log(f"  {line}", "err")
                    step_ok = False
                self._log(f"-- {name}: {'done' if step_ok else 'FAILED'} "
                          f"in {fmt_dur(time.time() - t0)}",
                          None if step_ok else "err")
                if not step_ok:
                    ok = False
                    if self._stop_evt.is_set():
                        self._log("Stopped by user.", "err")
                    else:
                        self._log("Pipeline aborted — later steps not run.", "err")
                    break
            self._log("")
            self._log(f"Total pipeline time: {fmt_dur(time.time() - t_all)}", "hdr")
        finally:
            self.q.put(("done", ok))

    # -- child process streaming -------------------------------------------

    def _run_child(self, stage: str, config: dict, config_name: str) -> bool:
        """Write *config*, run pipeline_runner.py *stage* on it, stream output."""
        config_path = self.run_dir / config_name
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._log(f"config: {config_path}")

        cmd = [self.cfg["python"], str(RUNNER), stage, str(config_path)]
        self._log(f"run: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Pin the child's encoding: it prints characters (->, µV, —) that a
        # cp1252 pipe cannot encode, and this must match the decode below.
        env["PYTHONIOENCODING"] = "utf-8"

        self._proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW,
        )
        # newline="" keeps the bare \r that tqdm uses, so progress bars can be
        # redrawn in place instead of flooding the log with one line per update.
        stream = io.TextIOWrapper(self._proc.stdout, encoding="utf-8",
                                  errors="replace", newline="")
        buf: list[str] = []
        pending_cr = False
        while True:
            ch = stream.read(1)
            if not ch:
                break
            if ch == "\r":
                pending_cr = True
                continue
            if ch == "\n":
                self._log("".join(buf))
                buf.clear()
                pending_cr = False
                continue
            if pending_cr:
                if buf:
                    self._log("".join(buf), inplace=True)
                    buf.clear()
                pending_cr = False
            buf.append(ch)
        if buf:
            self._log("".join(buf))

        stream.close()
        self._proc.wait()
        rc = self._proc.returncode
        self._proc = None
        if rc != 0:
            self._log(f"{stage} exited with code {rc}", "err")
        return rc == 0

    # -- step 1: Trodes DIO -------------------------------------------------

    def _step_dio(self) -> bool:
        folder = Path(self.cfg["data_folder"])
        exe = self.cfg["trodes_exe"]
        if not exe or not Path(exe).exists():
            self._log(f"trodesexport.exe not found: {exe or '(not set)'}", "err")
            return False

        rec_files = discover_rec_files(folder)
        if not rec_files:
            self._log("No .rec files found in that folder (or its *.rec subfolders).", "err")
            return False

        skip_existing = self.cfg["dio_existing"] == "skip"
        todo = rec_files
        if skip_existing:
            todo = [f for f in rec_files
                    if not (f.parent / (f.name + ".txt")).exists()]
            n_skipped = len(rec_files) - len(todo)
            if n_skipped:
                self._log(f"{n_skipped}/{len(rec_files)} .rec file(s) already have a "
                          f".txt sidecar — not re-exporting those.")
        if not todo:
            self._log("All .rec files already have gap sidecars; nothing to export.")
            return True

        self._log(f"Exporting DIO + timestamps from {len(todo)} .rec file(s); "
                  f"gaps read from the raw timestamps, flagged above "
                  f"{self.cfg['dio_interp']} packets")

        q: queue.Queue = queue.Queue()
        self._dio_worker = ExtractWorker(exe, todo, self.cfg["dio_interp"], q)
        self._dio_worker.start()
        results: list[dict] = []
        while True:
            kind, payload = q.get()
            if kind == "log":
                self._log(payload)
            elif kind == "file":
                results.append(payload)
                self._status(f"Step 1  DIO export {len(results)}/{len(todo)}")
            elif kind == "done":
                break
        self._dio_worker = None
        if self._stop_evt.is_set():
            return False

        failed = [r for r in results if r["returncode"] != 0]
        if failed:
            self._log(f"{len(failed)} file(s) returned a non-zero trodesexport exit code:",
                      "err")
            for r in failed:
                self._log(f"  {r['rec'].name}  (exit {r['returncode']})", "err")
            return False

        total_gaps = sum(len(r["gaps"]) for r in results)
        total_missing = sum(n for r in results for _, n in r["gaps"])
        self._log(f"{total_gaps} gap(s) across {len(results)} file(s); "
                  f"{total_missing} sample(s) will be PCHIP-filled in step 2.",
                  "warn" if total_gaps else None)

        if self.cfg["dio_review"]:
            holder = {"results": results,
                      "overwrite": not skip_existing,
                      "event": threading.Event(),
                      "saved": False}
            self._status("Step 1  waiting for gap review...")
            self.q.put(("review", holder))
            holder["event"].wait()
            if not holder["saved"]:
                self._log("Gap review cancelled — .txt files not written.", "err")
                return False
            return True

        written, skipped, errors = write_gap_files(results, overwrite=not skip_existing)
        self._log(f"Wrote {written} .txt file(s); skipped {skipped} "
                  f"(already existed); {len(errors)} error(s).")
        for err in errors:
            self._log(f"  {err}", "err")
        return not errors

    # -- step 2: rec -> NWB -------------------------------------------------

    def _step_nwb(self) -> bool:
        folder = Path(self.cfg["data_folder"])
        shanks = list(self.cfg["shanks"])
        nwb_stem, _ = session_names(folder)

        if self.cfg["nwb_skip_existing"]:
            missing = [s for s in shanks
                       if not (folder / f"{nwb_stem}sh{s}.nwb").exists()]
            done = [s for s in shanks if s not in missing]
            if done:
                self._log(f"Already converted, skipping shank(s): "
                          f"{', '.join(map(str, done))}")
            if not missing:
                self._log("All requested shanks already have an NWB file.")
                return True
            shanks = missing

        self._log(f"Converting shank(s): {', '.join(map(str, shanks))}  ->  "
                  f"{folder / (nwb_stem + 'sh<N>.nwb')}")

        config = {
            "chunk_duration": self.cfg["chunk_duration"],
            "parallel_shank": self.cfg["parallel_shank"],
            "data_folder": str(folder),
            "impedance_path": self.cfg["impedance_path"] or None,
            "electrode_location": self.cfg["electrode_location"],
            "experiment_description": self.cfg["experiment_description"],
            "animal_id": self.cfg["animal_id"],
            "device_type": self.cfg["device_type"],
            "shanks": shanks,
        }
        return self._run_child("rec2nwb", config, "rec2nwb_config.json")

    # -- step 3: MountainSort5 ----------------------------------------------

    def _step_sort(self) -> bool:
        recording = {
            "path": self.cfg["data_folder"],
            "shanks": list(self.cfg["shanks"]),
            "animal_id": self.cfg["animal_id"],
            "device_type": self.cfg["device_type"],
            "direct_sort": self.cfg["direct_sort"],
            "remove_artifacts": self.cfg["remove_artifacts"],
            "impedance_path": self.cfg["impedance_path"] or None,
        }
        config = {
            "sortout": self.cfg["sortout"],
            "n_jobs": self.cfg["n_jobs"],
            "sorter_params": self.cfg["sorter_params"],
            "recordings": [recording],
        }
        self._log(f"Results -> {Path(self.cfg['sortout']) / self.cfg['animal_id']}")
        return self._run_child("mssort", config, "mssort_config.json")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spike sorting pipeline  —  DIO gaps -> NWB -> MountainSort5")
        self.geometry("1000x820")

        self.q: queue.Queue = queue.Queue()
        self.worker: PipelineWorker | None = None
        self.log_file = None
        self._inplace_dirty = False
        self._auto_shanks = ""

        self._build_ui()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=6, pady=3)
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        # --- Session (shared by all steps) ---
        ses = ttk.LabelFrame(root, text="Session", padding=6)
        ses.grid(row=0, column=0, sticky="ew")
        ses.columnconfigure(1, weight=1)

        ttk.Label(ses, text="Recording folder *:").grid(row=0, column=0, sticky="w", **pad)
        self.folder_var = tk.StringVar()
        ttk.Entry(ses, textvariable=self.folder_var).grid(
            row=0, column=1, columnspan=4, sticky="ew", **pad)
        ttk.Button(ses, text="Browse...", command=self._browse_folder).grid(
            row=0, column=5, **pad)

        ttk.Label(ses, text="Animal ID *:").grid(row=1, column=0, sticky="w", **pad)
        self.animal_var = tk.StringVar()
        ttk.Entry(ses, textvariable=self.animal_var, width=14).grid(
            row=1, column=1, sticky="w", **pad)

        ttk.Label(ses, text="Device type *:").grid(row=1, column=2, sticky="e", **pad)
        self.device_var = tk.StringVar()
        self.device_cb = ttk.Combobox(ses, textvariable=self.device_var,
                                      values=list_device_types(), width=22)
        self.device_cb.grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(ses, text="Shanks *:").grid(row=1, column=4, sticky="e", **pad)
        sh = ttk.Frame(ses)
        sh.grid(row=1, column=5, sticky="w", **pad)
        self.shanks_var = tk.StringVar()
        ttk.Entry(sh, textvariable=self.shanks_var, width=18).pack(side="left")
        ttk.Button(sh, text="All", width=4, command=self._fill_all_shanks).pack(
            side="left", padx=(4, 0))

        self.remember_device_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ses, text="Save animal -> device type in device_types.json",
                        variable=self.remember_device_var).grid(
            row=2, column=1, columnspan=4, sticky="w", **pad)

        ttk.Label(ses, text="Python (steps 2, 3) *:").grid(row=3, column=0, sticky="w", **pad)
        self.python_var = tk.StringVar(value=child_python())
        ttk.Entry(ses, textvariable=self.python_var).grid(
            row=3, column=1, columnspan=4, sticky="ew", **pad)
        ttk.Button(ses, text="Browse...", command=self._browse_python).grid(
            row=3, column=5, **pad)

        ttk.Label(ses, foreground="#555", text=(
            "* must be filled in (here and on the step tabs) — 'Check setup' and "
            "'Run pipeline' stop before starting if one is blank.")).grid(
            row=4, column=0, columnspan=6, sticky="w", **pad)

        self.animal_var.trace_add("write", lambda *_: self._on_animal_change())
        self.device_var.trace_add("write", lambda *_: self._on_device_change())

        # --- Steps to run ---
        steps = ttk.LabelFrame(root, text="Steps to run", padding=6)
        steps.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.do_dio_var = tk.BooleanVar(value=True)
        self.do_nwb_var = tk.BooleanVar(value=True)
        self.do_sort_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps, text="1. Trodes DIO -> gap .txt",
                        variable=self.do_dio_var).pack(side="left", padx=(0, 16))
        self.nwb_check = ttk.Checkbutton(steps, text="2. rec -> NWB",
                                         variable=self.do_nwb_var)
        self.nwb_check.pack(side="left", padx=(0, 16))
        ttk.Checkbutton(steps, text="3. MountainSort5",
                        variable=self.do_sort_var).pack(side="left", padx=(0, 16))
        self.notify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps, text="Pop up when finished",
                        variable=self.notify_var).pack(side="right")

        # --- Per-step options ---
        nb = ttk.Notebook(root)
        nb.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        nb.add(self._build_dio_tab(nb), text="1. DIO")
        nb.add(self._build_nwb_tab(nb), text="2. NWB")
        nb.add(self._build_sort_tab(nb), text="3. Sorting")

        # --- Actions / status ---
        act = ttk.Frame(root)
        act.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.check_btn = ttk.Button(act, text="Check setup", command=self._preflight)
        self.check_btn.pack(side="left")
        self.run_btn = ttk.Button(act, text="Run pipeline", command=self._run)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(act, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left")
        ttk.Button(act, text="Clear log", command=self._clear_log).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(act, textvariable=self.status_var, foreground="#555").pack(
            side="left", padx=12)

        # --- Log ---
        log_frame = ttk.Frame(root)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="none", font=("Consolas", 9), height=18)
        ys = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.log.tag_configure("hdr", font=("Consolas", 9, "bold"))
        self.log.tag_configure("err", foreground="#c00000")
        self.log.tag_configure("warn", foreground="#b06000")
        self.log.tag_configure("ok", foreground="#007000")

    def _build_dio_tab(self, parent) -> ttk.Frame:
        pad = dict(padx=6, pady=4)
        f = ttk.Frame(parent, padding=8)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="trodesexport.exe *:").grid(row=0, column=0, sticky="w", **pad)
        self.exe_var = tk.StringVar(value=find_trodesexport())
        ttk.Entry(f, textvariable=self.exe_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(f, text="Browse...", command=self._browse_exe).grid(row=0, column=2, **pad)

        opt = ttk.Frame(f)
        opt.grid(row=1, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(opt, text="Warn above (dropped packets):").pack(side="left")
        self.interp_var = tk.StringVar(value="100")
        ttk.Entry(opt, textvariable=self.interp_var, width=8).pack(side="left", padx=(4, 16))
        ttk.Label(opt, text=".rec files that already have a .txt:").pack(side="left")
        self.dio_existing_var = tk.StringVar(value="skip")
        ttk.Combobox(opt, textvariable=self.dio_existing_var, width=26, state="readonly",
                     values=["skip", "re-export and overwrite"]).pack(side="left", padx=4)

        self.dio_review_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Review captured gaps before writing the .txt files "
                               "(pauses the pipeline)",
                        variable=self.dio_review_var).grid(
            row=2, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(f, foreground="#555", text=(
            "Runs trodesexport -dio on every .rec file: writes the gap sidecars step 2 "
            "interpolates from,\nand the .DIO folders it reads the session's first "
            "hardware timestamp from.")).grid(
            row=3, column=0, columnspan=3, sticky="w", **pad)
        return f

    def _build_nwb_tab(self, parent) -> ttk.Frame:
        pad = dict(padx=6, pady=4)
        f = ttk.Frame(parent, padding=8)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Electrode location *:").grid(row=0, column=0, sticky="w", **pad)
        self.location_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.location_var).grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(f, text="Experiment description:").grid(row=1, column=0, sticky="w", **pad)
        self.expdesc_var = tk.StringVar(value="None")
        ttk.Entry(f, textvariable=self.expdesc_var).grid(row=1, column=1, sticky="ew", **pad)

        ttk.Label(f, text="Impedance file:").grid(row=2, column=0, sticky="w", **pad)
        self.impedance_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.impedance_var).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(f, text="Browse...", command=self._browse_impedance).grid(
            row=2, column=2, **pad)

        opt = ttk.Frame(f)
        opt.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(opt, text="Chunk duration (s, blank = auto from RAM):").pack(side="left")
        self.chunk_var = tk.StringVar()
        ttk.Entry(opt, textvariable=self.chunk_var, width=8).pack(side="left", padx=(4, 16))
        self.parallel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="All shanks per chunk (parallelShank)",
                        variable=self.parallel_var).pack(side="left")

        self.nwb_skip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Skip shanks whose .nwb file already exists",
                        variable=self.nwb_skip_var).grid(
            row=4, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(f, foreground="#555", text=(
            "Electrode location is the recording site (e.g. V1, CA1). pynwb rejects a "
            "blank one on\nevery electrode, so leaving it empty fails the conversion "
            "after it has already started.")).grid(
            row=5, column=0, columnspan=3, sticky="w", **pad)
        return f

    def _build_sort_tab(self, parent) -> ttk.Frame:
        pad = dict(padx=6, pady=4)
        f = ttk.Frame(parent, padding=8)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(2, weight=1)

        ttk.Label(f, text="Sorting output (sortout) *:").grid(row=0, column=0, sticky="w", **pad)
        self.sortout_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.sortout_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(f, text="Browse...", command=self._browse_sortout).grid(
            row=0, column=2, **pad)

        opt = ttk.Frame(f)
        opt.grid(row=1, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(opt, text="n_jobs (metrics):").pack(side="left")
        self.njobs_var = tk.StringVar(value="8")
        ttk.Entry(opt, textvariable=self.njobs_var, width=6).pack(side="left", padx=(4, 16))
        self.rm_artifacts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Remove artifacts",
                        variable=self.rm_artifacts_var).pack(side="left", padx=(0, 16))
        self.direct_sort_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="direct_sort (sort .rec directly, skips step 2)",
                        variable=self.direct_sort_var,
                        command=self._on_direct_sort_toggle).pack(side="left")

        params = ttk.LabelFrame(f, text="sorter_params (JSON)", padding=4)
        params.grid(row=2, column=0, columnspan=3, sticky="nsew", **pad)
        params.rowconfigure(0, weight=1)
        params.columnconfigure(0, weight=1)
        self.params_text = tk.Text(params, height=11, wrap="none", font=("Consolas", 9))
        pys = ttk.Scrollbar(params, orient="vertical", command=self.params_text.yview)
        self.params_text.configure(yscrollcommand=pys.set)
        self.params_text.grid(row=0, column=0, sticky="nsew")
        pys.grid(row=0, column=1, sticky="ns")
        ttk.Button(params, text=f"Reload from {MSSORT_DEFAULTS.name}",
                   command=self._load_sorter_defaults).grid(
            row=1, column=0, sticky="w", pady=(4, 0))
        return f

    # -- Field helpers ------------------------------------------------------

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select the day's recording folder")
        if d:
            self.folder_var.set(str(Path(d)))
            self._autofill_from_folder()

    def _browse_exe(self):
        f = filedialog.askopenfilename(
            title="Locate trodesexport.exe",
            filetypes=[("trodesexport", "trodesexport.exe"), ("Executables", "*.exe"),
                       ("All files", "*.*")])
        if f:
            self.exe_var.set(f)

    def _browse_impedance(self):
        f = filedialog.askopenfilename(title="Select impedance CSV",
                                       filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if f:
            self.impedance_var.set(f)

    def _browse_python(self):
        f = filedialog.askopenfilename(
            title="Select the python.exe of the environment that has "
                  "spikeinterface + mountainsort5",
            filetypes=[("python.exe", "python.exe"), ("Executables", "*.exe"),
                       ("All files", "*.*")])
        if f:
            self.python_var.set(f)

    def _browse_sortout(self):
        d = filedialog.askdirectory(title="Select the sorting output folder")
        if d:
            self.sortout_var.set(str(Path(d)))

    def _autofill_from_folder(self):
        folder = Path(self.folder_var.get().strip().strip('"'))
        if folder.name:
            self.animal_var.set(folder.stem.split("_")[0])

    def _on_animal_change(self):
        """Fill in the device type this animal was last recorded with."""
        dt = load_device_map().get(self.animal_var.get().strip())
        if dt and dt != self.device_var.get():
            self.device_var.set(dt)

    def _on_device_change(self):
        # Only refill shanks while the field still holds what we auto-filled.
        if self.shanks_var.get().strip() in ("", self._auto_shanks):
            self._fill_all_shanks(quiet=True)

    def _fill_all_shanks(self, quiet: bool = False):
        device_type = self.device_var.get().strip()
        if not device_type:
            return
        try:
            shanks = all_shanks(device_type)
        except Exception as e:  # noqa: BLE001 - missing/unreadable mapping csv
            if not quiet:
                messagebox.showerror("Device type",
                                     f"Cannot read the channel map for {device_type}:\n\n{e}")
            return
        self._auto_shanks = ",".join(map(str, shanks))
        self.shanks_var.set(self._auto_shanks)

    def _on_direct_sort_toggle(self):
        if self.direct_sort_var.get():
            self.do_nwb_var.set(False)
            self.nwb_check.configure(state="disabled")
            self._log("direct_sort is on: step 2 (rec -> NWB) is not needed and was "
                      "switched off.", "warn")
        else:
            self.nwb_check.configure(state="normal")

    def _mssort_defaults(self) -> dict:
        try:
            return json.loads(MSSORT_DEFAULTS.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._log(f"[warn] cannot read {MSSORT_DEFAULTS}: {e}", "warn")
            return {}

    def _load_sorter_defaults(self):
        self._set_params_text(self._mssort_defaults().get("sorter_params", {}))

    def _set_params_text(self, params: dict):
        self.params_text.delete("1.0", "end")
        self.params_text.insert("1.0", json.dumps(params, indent=4))

    # -- Log ----------------------------------------------------------------

    def _log(self, msg: str = "", tag: str = None, inplace: bool = False):
        self.log.configure(state="normal")
        if self._inplace_dirty:
            self.log.delete("end-1c linestart", "end-1c")
            self._inplace_dirty = False
        self.log.insert("end", msg if inplace else msg + "\n", tag or ())
        self._inplace_dirty = inplace
        if not inplace:
            if self.log_file is not None:
                self.log_file.write(msg + "\n")
                self.log_file.flush()
            if int(self.log.index("end-1c").split(".")[0]) > LOG_MAX_LINES:
                self.log.delete("1.0", f"{LOG_TRIM_LINES}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._inplace_dirty = False

    # -- Config collection --------------------------------------------------

    def _collect(self) -> dict:
        """Validate every field and return the config for PipelineWorker.

        Raises ValueError with a message meant for the user.
        """
        folder = Path(self.folder_var.get().strip().strip('"'))
        if not folder.is_dir():
            raise ValueError(f"Recording folder does not exist:\n{folder}")

        animal_id = self.animal_var.get().strip()
        if not animal_id:
            raise ValueError("Animal ID is empty.")

        device_type = self.device_var.get().strip()
        do_nwb, do_sort = self.do_nwb_var.get(), self.do_sort_var.get()
        if (do_nwb or (do_sort and self.direct_sort_var.get())) and not device_type:
            raise ValueError("Device type is required for steps 2 and direct_sort.")
        if device_type and not (MAPPING_DIR / f"{device_type}.csv").exists():
            raise ValueError(f"No channel map for device type {device_type!r}:\n"
                             f"{MAPPING_DIR / (device_type + '.csv')}")

        shanks = parse_shanks(self.shanks_var.get())
        if (do_nwb or do_sort) and not shanks:
            raise ValueError("No shanks selected.")

        # pynwb treats "" the same as a missing location and rejects it on every
        # electrode, so a blank field only surfaces once conversion is under way.
        electrode_location = self.location_var.get().strip()
        if do_nwb and not electrode_location:
            raise ValueError("Electrode location is empty (NWB tab).\n\n"
                             "Enter the recording site, e.g. V1 or CA1.")

        chunk_txt = self.chunk_var.get().strip()
        try:
            chunk_duration = float(chunk_txt) if chunk_txt else None
        except ValueError:
            raise ValueError("Chunk duration must be a number (or blank for auto).")

        try:
            interp = int(self.interp_var.get().strip())
        except ValueError:
            raise ValueError("'Warn above' must be an integer number of packets "
                             "(e.g. 100).")

        try:
            n_jobs = int(self.njobs_var.get().strip())
        except ValueError:
            raise ValueError("n_jobs must be an integer.")

        sortout = self.sortout_var.get().strip().strip('"')
        if do_sort and not sortout:
            raise ValueError("Sorting output folder (sortout) is empty.")

        python_exe = self.python_var.get().strip().strip('"')
        if (do_nwb or do_sort) and not Path(python_exe).exists():
            raise ValueError(f"Python interpreter not found:\n{python_exe}")

        try:
            sorter_params = json.loads(self.params_text.get("1.0", "end").strip() or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"sorter_params is not valid JSON:\n{e}")

        return {
            "python": python_exe,
            "data_folder": str(folder),
            "animal_id": animal_id,
            "device_type": device_type,
            "shanks": shanks,
            "do_dio": self.do_dio_var.get(),
            "do_nwb": do_nwb,
            "do_sort": do_sort,
            "trodes_exe": self.exe_var.get().strip().strip('"'),
            "dio_interp": interp,
            "dio_existing": "skip" if self.dio_existing_var.get() == "skip" else "overwrite",
            "dio_review": self.dio_review_var.get(),
            "chunk_duration": chunk_duration,
            "parallel_shank": self.parallel_var.get(),
            "impedance_path": self.impedance_var.get().strip().strip('"'),
            "electrode_location": electrode_location,
            "experiment_description": self.expdesc_var.get().strip() or "None",
            "nwb_skip_existing": self.nwb_skip_var.get(),
            "sortout": sortout,
            "n_jobs": n_jobs,
            "remove_artifacts": self.rm_artifacts_var.get(),
            "direct_sort": self.direct_sort_var.get(),
            "sorter_params": sorter_params,
        }

    # -- Preflight ----------------------------------------------------------

    def _preflight(self) -> bool:
        """Log what each enabled step will read and write. Returns False on errors."""
        try:
            cfg = self._collect()
        except ValueError as e:
            self._log(f"[ERROR] {e}", "err")
            return False

        folder = Path(cfg["data_folder"])
        self._log("")
        self._log("--- Check setup ---", "hdr")
        self._log(f"Folder      : {folder}")
        self._log(f"Animal      : {cfg['animal_id']}   device: {cfg['device_type'] or '(none)'}"
                  f"   shanks: {cfg['shanks']}")
        errors = 0

        if cfg["do_nwb"] or cfg["do_sort"]:
            needed = ["spikeinterface", "pynwb"] + (["mountainsort5"] if cfg["do_sort"] else [])
            missing = missing_modules(cfg["python"], needed)
            if missing is None:
                self._log(f"[ERROR] could not run {cfg['python']}", "err")
                errors += 1
            elif missing:
                self._log(f"[ERROR] {cfg['python']}\n        cannot import "
                          f"{', '.join(missing)} — point 'Python (steps 2, 3)' at the "
                          f"environment you normally run these scripts from.", "err")
                errors += 1
            else:
                self._log(f"[ok] python: {cfg['python']}", "ok")

        rec_files = discover_rec_files(folder)
        with_txt = sum(1 for f in rec_files if (f.parent / (f.name + ".txt")).exists())
        if rec_files:
            self._log(f"[ok] {len(rec_files)} .rec file(s); {with_txt} already have a "
                      f"gap .txt sidecar.", "ok")
        elif cfg["do_dio"] or cfg["do_nwb"] or cfg["direct_sort"]:
            self._log("[ERROR] No .rec files in that folder (or its *.rec subfolders).", "err")
            errors += 1

        if cfg["do_dio"]:
            exe = cfg["trodes_exe"]
            if exe and Path(exe).exists():
                self._log(f"[ok] trodesexport: {exe}", "ok")
            else:
                self._log(f"[ERROR] trodesexport.exe not found: {exe or '(not set)'}", "err")
                errors += 1

        nwb_stem, sort_stem = session_names(folder)
        existing = [s for s in cfg["shanks"] if (folder / f"{nwb_stem}sh{s}.nwb").exists()]
        if cfg["do_nwb"]:
            self._log(f"[ok] step 2 writes {nwb_stem}sh<N>.nwb in the recording folder", "ok")
            self._log(f"[ok] electrode location: {cfg['electrode_location']}", "ok")
            if existing:
                what = ("skipped" if cfg["nwb_skip_existing"] else "OVERWRITTEN")
                self._log(f"[warn] shank(s) {existing} already converted — will be {what}.",
                          "warn")

        if cfg["do_sort"] and not cfg["direct_sort"]:
            if sort_stem != nwb_stem:
                self._log(f"[ERROR] name mismatch: step 2 writes '{nwb_stem}sh<N>.nwb' but "
                          f"step 3 looks for '{sort_stem}sh<N>.nwb'. Rename the folder "
                          f"(e.g. drop a trailing '.rec') before running.", "err")
                errors += 1
            else:
                missing = [s for s in cfg["shanks"] if s not in existing]
                if missing and not cfg["do_nwb"]:
                    self._log(f"[ERROR] no .nwb for shank(s) {missing} and step 2 is off — "
                              f"step 3 would skip them.", "err")
                    errors += 1
                else:
                    self._log(f"[ok] step 3 reads {sort_stem}sh<N>.nwb", "ok")

        if cfg["do_sort"]:
            sortout = Path(cfg["sortout"])
            if sortout.is_dir():
                self._log(f"[ok] results -> {sortout / cfg['animal_id']}", "ok")
            else:
                self._log(f"[warn] sortout does not exist yet (it will be created): "
                          f"{sortout}", "warn")
            self._log(f"[ok] scheme {cfg['sorter_params'].get('scheme', '1')}, "
                      f"detect_threshold {cfg['sorter_params'].get('detect_threshold')}, "
                      f"remove_artifacts {cfg['remove_artifacts']}", "ok")

        if cfg["impedance_path"] and not Path(cfg["impedance_path"]).exists():
            self._log(f"[ERROR] impedance file not found: {cfg['impedance_path']}", "err")
            errors += 1

        bad_ch = folder / "bad_channels.txt"
        found = "found" if bad_ch.exists() else "none (all channels used)"
        self._log(f"[ok] bad_channels.txt: {found}", "ok")

        self._log(f"--- {errors} problem(s) found ---" if errors
                  else "--- ready to run ---", "err" if errors else "ok")
        return errors == 0

    # -- Run / stop ---------------------------------------------------------

    def _run(self):
        try:
            cfg = self._collect()
        except ValueError as e:
            messagebox.showerror("Check the settings", str(e))
            return
        if not (cfg["do_dio"] or cfg["do_nwb"] or cfg["do_sort"]):
            messagebox.showerror("Nothing to run", "Select at least one step.")
            return
        if not self._preflight():
            if not messagebox.askyesno(
                    "Problems found",
                    "The check found problems (see the log).\n\nRun anyway?"):
                return

        self._save_settings()
        if cfg["device_type"] and self.remember_device_var.get():
            if remember_device_type(cfg["animal_id"], cfg["device_type"]):
                self._log(f"Saved {cfg['animal_id']} -> {cfg['device_type']} in "
                          f"{DEVICE_TYPES_PATH.name}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = LOG_ROOT / f"{stamp}_{Path(cfg['data_folder']).name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(run_dir / "pipeline.log", "w", encoding="utf-8")

        self._log("")
        self._log("#" * 70, "hdr")
        self._log(f"# Pipeline started {datetime.now():%Y-%m-%d %H:%M:%S}", "hdr")
        self._log(f"# {cfg['data_folder']}", "hdr")
        self._log(f"# log: {run_dir}", "hdr")
        self._log("#" * 70, "hdr")

        self.run_btn.configure(state="disabled")
        self.check_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.worker = PipelineWorker(cfg, run_dir, self.q)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.status_var.set("Stopping...")
            self._log("Stop requested — killing the running step.", "warn")
            self.worker.stop()

    # -- Queue polling ------------------------------------------------------

    def _poll_queue(self):
        try:
            for _ in range(500):  # cap per tick so the UI stays responsive
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    msg, tag, inplace = payload
                    self._log(msg, tag, inplace)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "review":
                    self._show_review(payload)
                elif kind == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _show_review(self, holder: dict):
        dlg = ReviewDialog(self, holder["results"], holder["overwrite"])
        self.wait_window(dlg)
        holder["saved"] = dlg.saved
        holder["event"].set()

    def _on_done(self, ok: bool):
        self.worker = None
        self.run_btn.configure(state="normal")
        self.check_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Finished." if ok else "Stopped / failed — see the log.")
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.notify_var.get():
            if ok:
                messagebox.showinfo("Pipeline finished", "All selected steps completed.")
            else:
                messagebox.showwarning("Pipeline stopped",
                                       "The run stopped early — see the log for details.")

    # -- Settings persistence ----------------------------------------------

    def _save_settings(self):
        settings = {
            "folder": self.folder_var.get(),
            "animal": self.animal_var.get(),
            "device": self.device_var.get(),
            "shanks": self.shanks_var.get(),
            "python": self.python_var.get(),
            "remember_device": self.remember_device_var.get(),
            "do_dio": self.do_dio_var.get(),
            "do_nwb": self.do_nwb_var.get(),
            "do_sort": self.do_sort_var.get(),
            "notify": self.notify_var.get(),
            "exe": self.exe_var.get(),
            "interp": self.interp_var.get(),
            "dio_existing": self.dio_existing_var.get(),
            "dio_review": self.dio_review_var.get(),
            "location": self.location_var.get(),
            "expdesc": self.expdesc_var.get(),
            "impedance": self.impedance_var.get(),
            "chunk": self.chunk_var.get(),
            "parallel": self.parallel_var.get(),
            "nwb_skip": self.nwb_skip_var.get(),
            "sortout": self.sortout_var.get(),
            "njobs": self.njobs_var.get(),
            "remove_artifacts": self.rm_artifacts_var.get(),
            "direct_sort": self.direct_sort_var.get(),
            "sorter_params": self.params_text.get("1.0", "end").strip(),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except OSError as e:
            self._log(f"[warn] could not save settings: {e}", "warn")

    def _load_settings(self):
        settings = {}
        if SETTINGS_PATH.exists():
            try:
                settings = json.loads(SETTINGS_PATH.read_text())
            except (OSError, json.JSONDecodeError):
                settings = {}

        # shanks before device: setting the device auto-fills the shank list,
        # and a saved list should win over that.
        for key, var in (
            ("folder", self.folder_var), ("animal", self.animal_var),
            ("shanks", self.shanks_var), ("device", self.device_var),
            ("python", self.python_var), ("remember_device", self.remember_device_var),
            ("do_dio", self.do_dio_var), ("do_nwb", self.do_nwb_var),
            ("do_sort", self.do_sort_var), ("notify", self.notify_var),
            ("exe", self.exe_var), ("interp", self.interp_var),
            ("dio_existing", self.dio_existing_var), ("dio_review", self.dio_review_var),
            ("location", self.location_var), ("expdesc", self.expdesc_var),
            ("impedance", self.impedance_var), ("chunk", self.chunk_var),
            ("parallel", self.parallel_var), ("nwb_skip", self.nwb_skip_var),
            ("sortout", self.sortout_var), ("njobs", self.njobs_var),
            ("remove_artifacts", self.rm_artifacts_var),
            ("direct_sort", self.direct_sort_var),
        ):
            if key in settings:
                var.set(settings[key])

        self._auto_shanks = self.shanks_var.get()
        if settings.get("sorter_params"):
            self.params_text.insert("1.0", settings["sorter_params"])
        else:
            # First run: seed the sorting tab from the JSON config used so far.
            defaults = self._mssort_defaults()
            self._set_params_text(defaults.get("sorter_params", {}))
            if defaults.get("sortout"):
                self.sortout_var.set(defaults["sortout"])
            if defaults.get("n_jobs"):
                self.njobs_var.set(str(defaults["n_jobs"]))
        if self.direct_sort_var.get():
            self.nwb_check.configure(state="disabled")
        self._log("Pick the day's recording folder, check the per-step tabs, then "
                  "'Run pipeline'.")

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("Quit", "A run is in progress. Stop it and quit?"):
                return
            self.worker.stop()
        self._save_settings()
        if self.log_file is not None:
            self.log_file.close()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
