"""
build_exe.py
============
Package pipeline_gui.py as ``pipeline_gui.exe`` so the daily run can be started
by double-clicking instead of activating a conda env first.

What the .exe is (and is not)
-----------------------------
It is a launcher, not a self-contained distribution. Steps 2 and 3 have always
run in a *separate* interpreter (pipeline_runner.py under the env named in
"Python (steps 2, 3)"), and that does not change. So the .exe still needs:

  * to sit in the repo root, beside pipeline_runner.py, rec2nwb/ and
    spikesorting/ -- it reads the channel maps, device_types.json and
    MsSortingFiles.json from disk, next to itself, at run time; and
  * a conda environment with spikeinterface / pynwb / mountainsort5 for the
    child process. "Check setup" verifies that env before anything runs.

Only the window itself is frozen: tkinter, plus pandas for reading a channel
map. spikeinterface, pynwb, h5py and mountainsort5 are deliberately absent,
which is what keeps the build small and reliable.

The build runs in a throwaway venv rather than in the sorting environment, so
PyInstaller cannot sweep the whole scientific stack into the binary and nothing
gets installed into the env you sort with.

    python build_exe.py              # build
    python build_exe.py --clean      # build after deleting the cached venv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build"
VENV_DIR = BUILD_DIR / "exe_venv"
WORK_DIR = BUILD_DIR / "pyinstaller"
ENTRY = REPO_ROOT / "pipeline_gui.py"
EXE_NAME = "pipeline_gui"

# Only what the window itself needs. Everything heavy stays in the conda env.
BUILD_REQUIREMENTS = ["pyinstaller", "pandas"]

# Belt and braces: these are not installed in the build venv, but naming them
# means a future stray import fails the build loudly instead of adding 500 MB.
EXCLUDED = [
    "spikeinterface", "pynwb", "hdmf", "h5py", "mountainsort5", "neo", "zarr",
    "numba", "torch", "kilosort", "matplotlib", "scipy", "IPython", "pytest",
]


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def venv_python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe"


def ensure_venv(clean: bool) -> Path:
    if clean and VENV_DIR.exists():
        print(f"removing {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    python = venv_python(VENV_DIR)
    if not python.exists():
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        # Built from this interpreter, but without its site-packages.
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        run([str(python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        run([str(python), "-m", "pip", "install", *BUILD_REQUIREMENTS, "--quiet"])
    else:
        print(f"reusing build venv at {VENV_DIR}")
    return python


def build(python: Path) -> Path:
    cmd = [
        str(python), "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",                    # no console window behind the GUI
        "--name", EXE_NAME,
        "--distpath", str(REPO_ROOT),    # land beside pipeline_runner.py
        "--workpath", str(WORK_DIR),
        "--specpath", str(WORK_DIR),
        "--paths", str(REPO_ROOT),
    ]
    for module in EXCLUDED:
        cmd += ["--exclude-module", module]
    cmd.append(str(ENTRY))
    run(cmd, cwd=str(REPO_ROOT))
    return REPO_ROOT / f"{EXE_NAME}.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true",
                        help="delete the cached build venv first")
    args = parser.parse_args()

    if not ENTRY.exists():
        print(f"cannot find {ENTRY}", file=sys.stderr)
        return 1

    python = ensure_venv(args.clean)
    exe = build(python)

    if not exe.exists():
        print("\nbuild finished but the .exe is missing", file=sys.stderr)
        return 1
    print(f"\nBuilt {exe}  ({exe.stat().st_size / 1e6:.1f} MB)")
    print("Keep it in the repo root: it reads pipeline_runner.py, "
          "rec2nwb/mapping/ and\nspikesorting/MsSortingFiles.json from the "
          "folder it sits in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
