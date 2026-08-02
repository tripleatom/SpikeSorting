"""
Trodes DIO extractor  ->  gap .txt files for rec2nwb_interp.py
==============================================================

A small GUI that, for every .rec file in a folder:
    1. runs Trodes' DIO export (which rec2nwb_interp.py reads the session's
       first hardware timestamp from), and
    2. writes a sidecar <recfile>.rec.txt listing every dropped packet.

rec2nwb_interp.py reads those sidecar .txt files (regex
``gap of (\\d+) points after timestamp (\\d+)``) to know where packets
were dropped, so it can PCHIP-interpolate the same gaps when it builds
the NWB files.

How the gaps are found
----------------------
NOT by scraping the console. ``trodesexport -dio`` is a *state change* export:
it never reconstructs a continuous signal, so it never interpolates and never
prints an "Interpolating data during gap of..." line. Watching its stdout --
which is what this tool used to do -- therefore found zero gaps on recordings
that had thousands, and every sidecar came out empty.

Instead we add ``-time`` to the same invocation (Trodes allows several
processing modes per run, so this costs one pass, not two) with ``-interp 0``
so timestamps come back raw. A dropped packet is then simply a jump in that
sequence, which is unambiguous and needs no message parsing.

``-interp 0`` is required: with the default 100 Trodes silently fills gaps of
up to 100 packets, and the timestamps come back perfectly continuous with
nothing to detect. It does not change the DIO output (verified byte-identical).

Counting convention
-------------------
Trodes' own message reports the timestamp *delta*, so a single dropped packet
is printed as "gap of 2 points". The sidecars written here record the number of
samples actually MISSING (that same gap is "gap of 1 points"), which is what
rec2nwb_interp.py's ``_pchip_fill`` synthesizes. Do not paste raw Trodes
console output into a sidecar: every gap would be over-filled by one sample.

Requires numpy (for the timestamp scan) and tkinter.  Run:

    python trodes_dio_gui.py
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------------

# The exact pattern rec2nwb_interp.py._parse_gap_file() looks for.  Emitting
# lines in this shape guarantees whatever we write is parseable downstream.
GAP_LINE = "Interpolating data during gap of {n} points after timestamp {t}"
GAP_RE = re.compile(r"gap of \d+ points after timestamp \d+")

# Trodes writes its extracted-data files as an ASCII settings block followed by
# packed binary records; the timestamp export is <time uint32><systime int64>.
TS_HEADER_END = b"<End settings>\n"
TS_DTYPE = np.dtype([("time", "<u4"), ("systime", "<i8")])

# Candidate locations for trodesexport.exe (first existing one wins).
TRODES_CANDIDATES = [
    Path(r"C:\Users\Windows\Desktop\Trodes_2-6-0_Windows11\trodesexport.exe"),
    Path(r"C:\Program Files\Trodes\trodesexport.exe"),
]

# On Windows, keep child consoles from flashing up for every file.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_timestamps(ts_path: Path) -> np.ndarray:
    """Return the raw hardware timestamps from a Trodes .timestamps.dat.

    The file is mapped rather than read whole -- a long session's export is well
    over a gigabyte and two thirds of it is the systime column we do not want --
    but the timestamps are copied out and the mapping closed before returning.
    Windows will not delete a file that still has a live mapping, and the caller
    deletes this one.
    """
    with open(ts_path, "rb") as fh:
        head = fh.read(8192)
    offset = head.find(TS_HEADER_END)
    if offset < 0:
        raise ValueError(f"no '<End settings>' block in {ts_path}")
    offset += len(TS_HEADER_END)
    n = (ts_path.stat().st_size - offset) // TS_DTYPE.itemsize
    if n <= 0:
        return np.empty(0, dtype=np.uint32)

    mm = np.memmap(ts_path, dtype=TS_DTYPE, mode="r", offset=offset, shape=(n,))
    try:
        return np.array(mm["time"])  # copy: a view would keep the file mapped
    finally:
        mm._mmap.close()


def find_gaps(timestamps: np.ndarray) -> list[tuple[int, int]]:
    """Locate dropped packets in a raw timestamp sequence.

    Returns:
        [(last_good_timestamp, n_missing), ...] in recording order, where
        *n_missing* is how many samples are absent -- one less than the
        timestamp delta Trodes prints.
    """
    if timestamps.size < 2:
        return []
    delta = np.diff(timestamps.astype(np.int64))
    breaks = np.flatnonzero(delta != 1)
    return [(int(timestamps[i]), int(delta[i] - 1)) for i in breaks]


def gap_file_text(rec_name: str, gaps: list[tuple[int, int]]) -> str:
    """Render the sidecar .txt contents for one .rec file."""
    total = sum(n for _, n in gaps)
    header = [
        f"# Gap list for {rec_name}",
        "# Written by trodes_dio_gui.py from 'trodesexport -time -interp 0'.",
        "# 'gap of N points' here is the number of samples MISSING. Trodes' own",
        "# console prints the timestamp delta instead, which is one larger.",
        f"# {len(gaps)} gap(s), {total} sample(s) missing.",
    ]
    body = [GAP_LINE.format(n=n, t=t) for t, n in gaps]
    return "\n".join(header + body) + "\n"


def find_trodesexport() -> str:
    for c in TRODES_CANDIDATES:
        if c.exists():
            return str(c)
    return ""


# ---------------------------------------------------------------------------
# .rec file discovery  (mirrors rec2nwb.utils.file_io.get_data_files)
# ---------------------------------------------------------------------------

def _rec_sort_key(f: Path):
    m_dt = re.search(r"_(\d{8})_(\d{6})", f.name)
    dt = (datetime.strptime(m_dt.group(1) + m_dt.group(2), "%Y%m%d%H%M%S")
          if m_dt else datetime.min)
    m_part = re.search(r"\.part(\d+)\.rec$", f.name)
    part = (1, int(m_part.group(1))) if m_part else (0, 0)
    return (dt, part)


def discover_rec_files(data_folder: Path) -> list[Path]:
    """Return .rec files directly in *data_folder* plus those inside any
    ``*.rec`` session subfolders, sorted by datetime then part number.

    Matches how rec2nwb_interp.py collects its files, so the sidecar .txt we
    write lands next to exactly the files it will later read.
    """
    rec_files = [f for f in data_folder.glob("*.rec") if f.is_file()]
    for rec_dir in data_folder.glob("*.rec"):
        if rec_dir.is_dir():
            rec_files.extend(f for f in rec_dir.glob("*.rec") if f.is_file())
    return sorted(rec_files, key=_rec_sort_key)


# ---------------------------------------------------------------------------
# Worker: run trodesexport per file, capture gap lines
# ---------------------------------------------------------------------------

class ExtractWorker(threading.Thread):
    """Runs ``trodesexport -dio`` over a list of .rec files in a background
    thread and reports progress / gap lines back through a queue.

    Messages put on the queue are tuples:
        ("log",   str)                       -- a line for the log pane
        ("file",  dict)                       -- per-file result dict
        ("done",  None)                       -- batch finished (or stopped)
    """

    def __init__(self, exe: str, rec_files: list[Path], interp: int,
                 out_q: "queue.Queue"):
        super().__init__(daemon=True)
        self.exe = exe
        self.rec_files = rec_files
        # Not passed to trodesexport (the export must run at -interp 0 to expose
        # gaps at all); it is the size above which a gap is called out as large.
        self.interp = interp
        self.q = out_q
        self._stop_evt = threading.Event()
        self._proc: subprocess.Popen | None = None

    def stop(self):
        self._stop_evt.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _log(self, msg: str):
        self.q.put(("log", msg))

    def run(self):
        try:
            for idx, rec in enumerate(self.rec_files, 1):
                if self._stop_evt.is_set():
                    self._log("Stopped by user.")
                    break
                self._process_one(idx, rec)
        finally:
            self.q.put(("done", None))

    def _process_one(self, idx: int, rec: Path):
        n = len(self.rec_files)
        self._log("")
        self._log(f"[{idx}/{n}] {rec.name}")
        self._log(f"        folder: {rec.parent}")

        # Both modes in one invocation = one pass over the .rec. -interp 0 is
        # what makes the dropped packets visible in the timestamp sequence.
        cmd = [self.exe, "-dio", "-time", "-rec", str(rec), "-interp", "0"]
        gaps: list[tuple[int, int]] = []
        rc = None
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=_NO_WINDOW,
            )
            for line in self._proc.stdout:  # drained so the pipe cannot fill
                if self._stop_evt.is_set():
                    break
            self._proc.wait()
            rc = self._proc.returncode
        except FileNotFoundError:
            self._log("        ERROR: trodesexport.exe not found.")
            rc = -1
        except Exception as e:  # noqa: BLE001 - surface any failure to the log
            self._log(f"        ERROR: {e}")
            rc = -1
        finally:
            self._proc = None

        if rc == 0 and not self._stop_evt.is_set():
            try:
                gaps = self._scan_timestamps(rec)
            except Exception as e:  # noqa: BLE001
                self._log(f"        ERROR reading timestamps: {e}")
                rc = -1

        status = "ok" if rc == 0 else f"exit {rc}"
        total = sum(g for _, g in gaps)
        self._log(f"        {len(gaps)} gap(s), {total} sample(s) missing  ({status})")
        large = [(t, g) for t, g in gaps if g > self.interp]
        if large:
            self._log(f"        WARNING: {len(large)} gap(s) exceed {self.interp} "
                      f"packets (largest {max(g for _, g in large)}) - "
                      f"interpolating these may not be meaningful")

        txt_path = rec.parent / (rec.name + ".txt")
        self.q.put(("file", {
            "rec": rec,
            "txt_path": txt_path,
            "gaps": gaps,
            "returncode": rc,
            "existed": txt_path.exists(),
        }))

    def _scan_timestamps(self, rec: Path) -> list[tuple[int, int]]:
        """Read the .time export beside *rec*, find the gaps, then discard it.

        The timestamp export is only a means of finding dropped packets, and is
        large enough (over a gigabyte for a long session) that leaving it next
        to the recording would be a nuisance.
        """
        stem = rec.name[:-4] if rec.name.endswith(".rec") else rec.stem
        time_dir = rec.parent / f"{stem}.time"
        ts_path = time_dir / f"{stem}.timestamps.dat"
        if not ts_path.exists():
            found = sorted(time_dir.glob("*.timestamps.dat")) if time_dir.is_dir() else []
            if not found:
                raise FileNotFoundError(f"no timestamps export at {ts_path}")
            ts_path = found[0]
        try:
            timestamps = read_timestamps(ts_path)
            self._log(f"        {timestamps.size:,} timestamps scanned")
            return find_gaps(timestamps)
        finally:
            shutil.rmtree(time_dir, onerror=self._report_cleanup_failure)

    def _report_cleanup_failure(self, _func, path, exc_info):
        """rmtree callback: say so rather than silently leaving GBs behind."""
        self._log(f"        WARNING: could not remove {path} ({exc_info[1]}); "
                  f"delete it by hand")


# ---------------------------------------------------------------------------
# Writing the sidecar .txt files
# ---------------------------------------------------------------------------

def write_gap_files(results: list[dict], overwrite: bool) -> tuple[int, int, list[str]]:
    """Write each result's gap lines to its ``<recfile>.rec.txt`` sidecar.

    Returns:
        (written, skipped, errors) — *errors* holds one message per file that
        could not be written.
    """
    written, skipped, errors = 0, 0, []
    for r in results:
        txt_path: Path = r["txt_path"]
        if r["existed"] and not overwrite:
            skipped += 1
            continue
        try:
            txt_path.write_text(gap_file_text(r["rec"].name, r["gaps"]),
                                encoding="utf-8")
            written += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{txt_path}: {e}")
    return written, skipped, errors


# ---------------------------------------------------------------------------
# Review dialog: show captured gaps per file, confirm before writing
# ---------------------------------------------------------------------------

class ReviewDialog(tk.Toplevel):
    MAX_SHOWN = 50  # per file, in the dialog only

    def __init__(self, master, results: list[dict], overwrite: bool):
        super().__init__(master)
        self.title("Review detected gaps")
        self.geometry("820x560")
        self.results = results
        self.overwrite = overwrite
        self.saved = False

        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)

        total_gaps = sum(len(r["gaps"]) for r in results)
        total_missing = sum(n for r in results for _, n in r["gaps"])
        summary = (f"{len(results)} file(s) processed  |  "
                   f"{total_gaps} gap(s), {total_missing} sample(s) missing  |  "
                   f"{sum(1 for r in results if r['gaps'])} file(s) with gaps")
        ttk.Label(top, text=summary, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="Counts are samples MISSING, one less than the delta Trodes' own "
                 "console prints.\nThey go into each <recfile>.rec.txt, which "
                 "rec2nwb_interp.py PCHIP-fills. Review, then Save.",
            foreground="#555",
        ).pack(anchor="w", pady=(0, 6))

        txt_frame = ttk.Frame(top)
        txt_frame.pack(fill="both", expand=True)
        self.text = tk.Text(txt_frame, wrap="none", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(txt_frame, orient="vertical", command=self.text.yview)
        xscroll = ttk.Scrollbar(txt_frame, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)

        self._populate()
        self.text.configure(state="disabled")

        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(8, 0))
        note = "existing .txt files WILL be overwritten" if overwrite \
            else "files that already have a .txt will be SKIPPED"
        ttk.Label(btns, text=f"Save mode: {note}", foreground="#555").pack(side="left")
        ttk.Button(btns, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(btns, text="Save all .txt files", command=self._save).pack(
            side="right", padx=6)

        self.transient(master)
        self.grab_set()

    def _populate(self):
        self.text.tag_configure("hdr", font=("Consolas", 9, "bold"))
        self.text.tag_configure("nogap", foreground="#888")
        self.text.tag_configure("skip", foreground="#b06000")
        for r in self.results:
            rc = r["returncode"]
            flag = "" if rc == 0 else f"  [trodesexport exit {rc}]"
            self.text.insert("end", f"{r['rec'].name}{flag}\n", "hdr")
            self.text.insert("end", f"  -> {r['txt_path']}\n")
            if r["existed"] and not self.overwrite:
                self.text.insert("end", "  (already has .txt - will be skipped)\n", "skip")
            gaps = r["gaps"]
            if gaps:
                missing = sum(n for _, n in gaps)
                self.text.insert("end", f"  {len(gaps)} gap(s), {missing} sample(s) "
                                        f"missing, largest {max(n for _, n in gaps)}\n")
                # A bad session runs to thousands of gaps; listing every one just
                # makes the dialog unusable. The .txt still gets all of them.
                for t, n in gaps[:self.MAX_SHOWN]:
                    self.text.insert("end", f"    {GAP_LINE.format(n=n, t=t)}\n")
                if len(gaps) > self.MAX_SHOWN:
                    self.text.insert("end", f"    ... {len(gaps) - self.MAX_SHOWN} "
                                            f"more (all will be written)\n", "nogap")
            else:
                self.text.insert("end", "    (no gaps found)\n", "nogap")
            self.text.insert("end", "\n")

    def _save(self):
        written, skipped, errors = write_gap_files(self.results, self.overwrite)
        for err in errors:
            messagebox.showerror("Write error", err, parent=self)
        self.saved = True
        messagebox.showinfo(
            "Saved",
            f"Wrote {written} .txt file(s).\n"
            f"Skipped {skipped} (already existed).\n"
            f"Errors: {len(errors)}.",
            parent=self,
        )
        self.destroy()

    def _cancel(self):
        self.destroy()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trodes DIO -> gap txt  (for rec2nwb_interp.py)")
        self.geometry("860x600")

        self.q: queue.Queue = queue.Queue()
        self.worker: ExtractWorker | None = None
        self.results: list[dict] = []
        self.rec_files: list[Path] = []

        self._build_ui()
        self.after(100, self._poll_queue)

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=6, pady=4)
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Folder row
        ttk.Label(frm, text="Recording folder:").grid(row=0, column=0, sticky="w", **pad)
        self.folder_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.folder_var, width=70).grid(
            row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self._browse_folder).grid(
            row=0, column=2, **pad)

        # trodesexport row
        ttk.Label(frm, text="trodesexport.exe:").grid(row=1, column=0, sticky="w", **pad)
        self.exe_var = tk.StringVar(value=find_trodesexport())
        ttk.Entry(frm, textvariable=self.exe_var, width=70).grid(
            row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self._browse_exe).grid(
            row=1, column=2, **pad)

        # Options row
        opt = ttk.Frame(frm)
        opt.grid(row=2, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(opt, text="Warn above (dropped packets):").pack(side="left")
        self.interp_var = tk.StringVar(value="100")
        ttk.Entry(opt, textvariable=self.interp_var, width=8).pack(side="left", padx=(4, 16))
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Overwrite existing .txt files",
                        variable=self.overwrite_var).pack(side="left")

        # Action buttons
        act = ttk.Frame(frm)
        act.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        self.scan_btn = ttk.Button(act, text="Scan folder", command=self._scan)
        self.scan_btn.pack(side="left")
        self.run_btn = ttk.Button(act, text="Run DIO extraction",
                                  command=self._run, state="disabled")
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(act, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Pick a folder and click 'Scan folder'.")
        ttk.Label(act, textvariable=self.status_var, foreground="#555").pack(
            side="left", padx=12)

        # Progress bar
        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)

        # Log pane
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)
        self.log = tk.Text(log_frame, wrap="none", font=("Consolas", 9), height=20)
        ys = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=ys.set, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

    # -- Helpers ------------------------------------------------------------

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select folder containing .rec files")
        if d:
            self.folder_var.set(d)

    def _browse_exe(self):
        f = filedialog.askopenfilename(
            title="Locate trodesexport.exe",
            filetypes=[("trodesexport", "trodesexport.exe"), ("Executables", "*.exe"),
                       ("All files", "*.*")])
        if f:
            self.exe_var.set(f)

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # -- Actions ------------------------------------------------------------

    def _scan(self):
        folder = self.folder_var.get().strip().strip('"')
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("Invalid folder", "Please select a valid folder.")
            return
        self.rec_files = discover_rec_files(Path(folder))
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        if not self.rec_files:
            self.status_var.set("No .rec files found.")
            self.run_btn.configure(state="disabled")
            self._log("No .rec files found in that folder (or its *.rec subfolders).")
            return
        self._log(f"Found {len(self.rec_files)} .rec file(s):")
        for i, f in enumerate(self.rec_files, 1):
            self._log(f"  {i:2d}. {f.relative_to(folder) if str(f).startswith(folder) else f}")
        self.status_var.set(f"{len(self.rec_files)} file(s) ready. Click 'Run DIO extraction'.")
        self.run_btn.configure(state="normal")

    def _run(self):
        exe = self.exe_var.get().strip().strip('"')
        if not exe or not Path(exe).exists():
            messagebox.showerror("trodesexport not found",
                                 "Set a valid path to trodesexport.exe.")
            return
        try:
            interp = int(self.interp_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid threshold",
                                 "'Warn above' must be an integer number of "
                                 "packets (e.g. 100).")
            return
        if not self.rec_files:
            self._scan()
            if not self.rec_files:
                return

        self.results = []
        self.progress.configure(maximum=len(self.rec_files), value=0)
        self.run_btn.configure(state="disabled")
        self.scan_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running trodesexport...")
        self._log("")
        self._log("=" * 60)
        self._log(f"Running trodesexport -dio -time -interp 0  on "
                  f"{len(self.rec_files)} file(s); gaps read from the raw "
                  f"timestamps, flagged above {interp} packets")
        self._log("=" * 60)

        self.worker = ExtractWorker(exe, self.rec_files, interp, self.q)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.status_var.set("Stopping...")

    # -- Queue polling ------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "file":
                    self.results.append(payload)
                    self.progress.step(1)
                elif kind == "done":
                    self._on_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _on_done(self):
        self.run_btn.configure(state="normal")
        self.scan_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.worker = None
        if not self.results:
            self.status_var.set("No results.")
            return
        self.status_var.set(
            f"Done. {sum(len(r['gaps']) for r in self.results)} gap(s), "
            f"{sum(n for r in self.results for _, n in r['gaps'])} sample(s) missing "
            f"across {len(self.results)} file(s). Review to save.")
        ReviewDialog(self, self.results, self.overwrite_var.get())


if __name__ == "__main__":
    App().mainloop()
