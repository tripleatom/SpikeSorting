"""Native-looking launcher for pipeline_gui.py.

This module is frozen as pipeline_gui.exe.  The actual Tk application runs in
the configured Conda interpreter, avoiding PyInstaller/Tcl conflicts on hosts
with multiple Conda installations.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, "Pipeline GUI", 0x10)


def configured_python(root: Path) -> Path | None:
    settings_path = root / "pipeline_gui_settings.json"
    if settings_path.exists():
        try:
            configured = json.loads(settings_path.read_text(encoding="utf-8")).get("python")
            if configured and Path(configured).is_file():
                return Path(configured)
        except (OSError, json.JSONDecodeError):
            pass

    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "python.exe")
    candidates.append(Path.home() / ".conda" / "envs" / "ms10" / "python.exe")
    on_path = shutil.which("python")
    if on_path:
        candidates.append(Path(on_path))
    return next((path for path in candidates if path.is_file()), None)


def main() -> int:
    root = repo_root()
    script = root / "pipeline_gui.py"
    python = configured_python(root)
    if not script.is_file():
        show_error(f"Cannot find the GUI script:\n{script}\n\nKeep pipeline_gui.exe in the repository root.")
        return 1
    if python is None:
        show_error(
            "Cannot find a Python interpreter. Set the 'python' value in "
            f"{root / 'pipeline_gui_settings.json'} to the ms10 environment."
        )
        return 1

    log_dir = root / "pipeline_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "pipeline_gui_startup.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] launching with {python}\n")
            log.flush()
            process = subprocess.Popen(
                [str(python), str(script)],
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                return_code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                return 0
    except OSError as exc:
        show_error(f"Could not start the Pipeline GUI:\n{exc}")
        return 1

    show_error(
        f"The Pipeline GUI exited during startup (code {return_code}).\n\n"
        f"Details were written to:\n{log_path}"
    )
    return return_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
