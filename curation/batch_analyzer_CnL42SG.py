"""
batch_analyzer_CnL42SG.py
=========================
Step 2 of 2 — build curated SortingAnalyzers for CnL42SG sessions 260122-260325.

Consumes the labels written by batch_label_CnL42SG.py and runs
MsCuratedAnalyzer.main() per session, recording the outcome in the
``analyzer_status`` column of the plan CSV.

Sessions are independent, so up to ``MAX_PARALLEL_SESSIONS`` of them are built
concurrently. Each build runs in its own subprocess (``--build <session>``):
MsCuratedAnalyzer calls ``si.set_global_job_kwargs`` and spawns its own worker
pool, so a separate process is the clean way to run two at once without the
global job config or the nested worker pools colliding. The orchestrator owns
all CSV writes (guarded by a lock); workers never touch the CSV.

Status values written to ``analyzer_status``:
  done@<ts>            — curated_analyzer built
  skipped_exists@<ts>  — curated_analyzer folder already present
  skipped_no_raw@<ts>  — no matched/existing raw folder in the plan
  no_labels@<ts>       — unit_labels.json missing (run labeling first)
  failed@<ts>          — the build subprocess exited non-zero (see its log)

Usage
-----
  python batch_analyzer_CnL42SG.py                 # orchestrate the whole plan
  python batch_analyzer_CnL42SG.py --build <name>  # (internal) build one session
"""

import os
import sys
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # for rec2nwb

import MsCuratedAnalyzer
from MsCuratedAnalyzer import CURATED_EXTENSIONS
from plan_status import stamp, update_status

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH    = _HERE / "curation_record" / "CnL42SG_260122_260325_curation_plan.csv"
LOG_DIR     = _HERE / "curation_record" / "build_logs"
STATUS_COL  = "analyzer_status"
TOP_SORTOUT = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout")
ANIMAL_ID   = "CnL42SG"

N_JOBS                = 24        # worker pool size for ONE session's build
MAX_PARALLEL_SESSIONS = 1         # build sessions one at a time (avoids RAM contention)
USE_CACHE             = True      # write temp binary (True) vs compute on lazy recording (False)
CACHE_DTYPE           = "int16"   # temp binary dtype when USE_CACHE; int16 halves its I/O

# Split the cores across concurrent sessions so two builds don't oversubscribe
# (32 logical cores / 2 sessions -> 16 each). Capped by N_JOBS.
_CPU             = os.cpu_count() or N_JOBS
PER_SESSION_JOBS = max(1, min(N_JOBS, _CPU // MAX_PARALLEL_SESSIONS))

# Two 4-shank probes (8shank32): shanks 0-3 on first probe, 4-7 on second.
# z values group shanks into separate physical probes for ProbeGroup construction.
Z_OFFSETS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 500, 5: 500, 6: 500, 7: 500}

_csv_lock = threading.Lock()


# ── Completeness check ──────────────────────────────────────────────────────────
def _analyzer_complete(curated_path: Path) -> bool:
    """
    True only if curated_analyzer has every required extension computed.

    A build that created the analyzer folder but crashed before/while computing
    extensions leaves curated_analyzer/extensions/ empty or partial. Such a
    folder must be rebuilt, not skipped.
    """
    ext_dir = curated_path / "extensions"
    if not ext_dir.is_dir():
        return False
    for ext in CURATED_EXTENSIONS:
        sub = ext_dir / ext
        if not sub.is_dir() or not any(sub.iterdir()):
            return False
    return True


# ── Worker: build a single session (runs in its own subprocess) ─────────────────
def _build_session(row) -> None:
    """Build one curated analyzer from a plan row. Raises on failure."""
    raw_folder = Path(row["chosen_raw_folder_path"].strip())
    shanks = [int(s.strip().replace("shank", ""))
              for s in row.get("detected_shanks", "shank0").strip().split(",") if s.strip()]
    # overwrite=True: the orchestrator only dispatches sessions that are missing
    # or incomplete, so a build should always (re)create the analyzer folder.
    analyzer = MsCuratedAnalyzer.main(
        rec_folder=raw_folder,
        shanks=shanks,
        sortout_folder=TOP_SORTOUT,
        animal_id=ANIMAL_ID,
        overwrite=True,
        n_jobs=PER_SESSION_JOBS,
        z_offsets=Z_OFFSETS or None,
        use_cache=USE_CACHE,
        cache_dtype=CACHE_DTYPE,
    )
    # main() returns None when it aborts without building (e.g. no shank had a
    # usable NWB/.rec). Treat that as a failure so it isn't stamped "done".
    if analyzer is None:
        raise RuntimeError(
            f"no curated analyzer produced for {raw_folder} — no usable shank "
            "data (missing NWB and no .rec fallback?)")


def _run_worker(session_name: str) -> int:
    """--build entry point: build the one row matching ``session_name``."""
    import traceback
    df  = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    sel = df[df["session_name"].str.strip() == session_name]
    if sel.empty:
        print(f"[worker] session not found in plan: {session_name}")
        return 2
    try:
        _build_session(sel.iloc[0])
        return 0
    except Exception:
        traceback.print_exc()
        return 1


# ── Orchestrator ────────────────────────────────────────────────────────────────
def _set_status(session_name: str, value: str):
    with _csv_lock:
        update_status(CSV_PATH, session_name, STATUS_COL, value)


def _dispatch(session_name: str):
    """Run one build subprocess, streaming its output to a per-session log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{session_name}.log"
    print(f"  [{session_name}] building (jobs={PER_SESSION_JOBS}) → {log_path}")
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--build", session_name],
            stdout=lf, stderr=subprocess.STDOUT,
        )
    return session_name, proc.returncode, log_path


def main():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    total          = len(df)
    skipped_exist  = 0
    skipped_no_raw = 0
    skipped_no_lbl = 0
    to_build       = []

    # ── Classify: stamp skips here, queue the rest for parallel building ──────
    for _, row in df.iterrows():
        session_name   = row["session_name"].strip()
        curated_path   = Path(row["curated_analyzer_path"].strip())
        session_folder = Path(row["sortout_session_path"].strip())
        raw_folder_str = row.get("chosen_raw_folder_path", "").strip()
        raw_match      = row.get("raw_match_status", "").strip()

        if _analyzer_complete(curated_path):
            print(f"  [{session_name}] curated_analyzer complete → skip.")
            skipped_exist += 1
            _set_status(session_name, stamp("skipped_exists"))
            continue

        if curated_path.exists():
            print(f"  [{session_name}] curated_analyzer present but missing "
                  f"extensions → rebuild.")

        if not (session_folder / "unit_labels.json").exists():
            print(f"  [{session_name}] unit_labels.json missing → run labeling first.")
            skipped_no_lbl += 1
            _set_status(session_name, stamp("no_labels"))
            continue

        if not raw_folder_str or raw_match != "matched" or not Path(raw_folder_str).exists():
            print(f"  [{session_name}] no usable raw folder → skip.")
            skipped_no_raw += 1
            _set_status(session_name, stamp("skipped_no_raw"))
            continue

        to_build.append(session_name)

    # ── Build queued sessions, MAX_PARALLEL_SESSIONS at a time ────────────────
    print(f"\nBuilding {len(to_build)} session(s), {MAX_PARALLEL_SESSIONS} at a time "
          f"({PER_SESSION_JOBS} jobs each, cache_dtype={CACHE_DTYPE}).")
    done   = 0
    failed = []
    if to_build:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SESSIONS) as ex:
            futures = [ex.submit(_dispatch, s) for s in to_build]
            for fut in as_completed(futures):
                session_name, rc, log_path = fut.result()
                if rc == 0:
                    done += 1
                    _set_status(session_name, stamp("done"))
                    print(f"  [{session_name}] DONE")
                else:
                    failed.append((session_name, log_path))
                    _set_status(session_name, stamp("failed"))
                    print(f"  [{session_name}] FAILED (rc={rc}) → see {log_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ANALYZER BUILD COMPLETE  ({total} session(s))")
    print(f"  Built                       : {done}")
    print(f"  Skipped — already exists    : {skipped_exist}")
    print(f"  Skipped — no labels         : {skipped_no_lbl}")
    print(f"  Skipped — no raw folder     : {skipped_no_raw}")
    print(f"  Failed                      : {len(failed)}")
    if failed:
        print("\nFailed sessions (see logs):")
        for session_name, log_path in failed:
            print(f"  {session_name}  → {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--build":
        sys.exit(_run_worker(sys.argv[2]))
    main()
