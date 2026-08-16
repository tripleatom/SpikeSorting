#!/usr/bin/env python3
r"""
Move bad-channel txt files from the local NWB tree to the server .rec tree.

    G:\CnL46\<session>\bad_channels.txt
        ->  \\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\<session>\bad_channels.txt

The two trees do NOT share a session path. The server nests one extra level:

    local   G:\CnL46\CnL46_20260801\bad_channels.txt
    server  ...\CnL46\260801\CnL46_20260801\bad_channels.txt
                      ^^^^^^ YYMMDD level

so the destination is resolved rather than assumed:
  * outer level: the folder whose name ENDS in the session's yymmdd. Matches
    "260801", "CnL46_260801" and "CnL46_20260801" alike.
  * inner level: the subfolder named exactly like the local session; failing
    that, the sole non-"video" subfolder (session 260805 is really named
    CnL46_202060805 on the server -- a typo -- and only this fallback finds it).
    If several candidates remain it is reported as ambiguous and skipped.

Other assumptions:
  * the txt lives at the session folder ROOT on both sides, beside the .rec files
  * MOVE semantics: source is deleted only after a byte-for-byte verified copy

Safety design:
  * dry run by default -- prints the full plan and changes nothing.
    Add --apply to actually move.
  * never creates a session folder on the server. If the matching session
    doesn't exist there, it is reported and skipped (a missing folder almost
    always means a name mismatch, not a folder you actually want created).
  * copy -> sha256 verify -> delete source. If verification fails the source
    is kept and the partial destination is removed.
  * if the destination already exists: identical content -> source deleted
    (the move is effectively already done); different content -> skipped
    unless --overwrite.

Usage:
    python move_bad_channels.py                 # dry run
    python move_bad_channels.py --apply
    python move_bad_channels.py --apply --overwrite
    python move_bad_channels.py --pattern "*bad*chan*.txt"
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(r"G:\CnL46")
DST_ROOT = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46")
DEFAULT_PATTERN = "bad_channel*.txt"  # catches bad_channel.txt / bad_channels.txt


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_sources(src_root: Path, pattern: str) -> list[tuple[str, Path]]:
    """Return (session_name, txt_path) for every session folder holding a match.

    Only the session folder root is searched (not recursively), matching the
    layout you described. Case-insensitive on Windows already, but we lower()
    the glob results anyway so this behaves the same if run from Linux/macOS.
    """
    found: list[tuple[str, Path]] = []
    for session_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        hits = sorted(
            p for p in session_dir.glob("*")
            if p.is_file() and Path(p.name.lower()).match(pattern.lower())
        )
        if not hits:
            continue
        if len(hits) > 1:
            print(f"  ! {session_dir.name}: {len(hits)} matches "
                  f"({', '.join(h.name for h in hits)}) -- moving all")
        for h in hits:
            found.append((session_dir.name, h))
    return found


def session_yymmdd(session: str) -> str | None:
    """'CnL46_20260801' -> '260801'. None if the name carries no 8-digit date."""
    m = re.search(r"(?<!\d)(\d{8})(?!\d)", session)
    return m.group(1)[2:] if m else None


def index_server(dst_root: Path) -> dict[str, list[Path]]:
    """Map yymmdd -> outer server folders whose name ends in that date.

    Taking the LAST six digits makes '260801', 'CnL46_260801' and
    'CnL46_20260801' all index under '260801', so the layout can drift without
    this script needing another edit.
    """
    index: dict[str, list[Path]] = {}
    for outer in sorted(p for p in dst_root.iterdir() if p.is_dir()):
        m = re.search(r"(\d{6})$", outer.name)
        if m:
            index.setdefault(m.group(1), []).append(outer)
    return index


def resolve_dst_dir(session: str, index: dict[str, list[Path]]) -> tuple[Path | None, str]:
    """Find the server session folder for a local session name.

    Returns (folder, note); folder is None when it cannot be resolved safely,
    and note then explains why. A non-empty note alongside a folder means the
    match was inexact and deserves a look before --apply.
    """
    yymmdd = session_yymmdd(session)
    if yymmdd is None:
        return None, "SKIP: no yyyymmdd in local session name"

    outers = index.get(yymmdd, [])
    if not outers:
        return None, f"SKIP: no {yymmdd} folder on server"
    if len(outers) > 1:
        names = ", ".join(o.name for o in outers)
        return None, f"SKIP: {len(outers)} server folders match {yymmdd} ({names})"
    outer = outers[0]

    subs = [p for p in outer.iterdir() if p.is_dir() and p.name.lower() != "video"]
    for p in subs:
        if p.name.lower() == session.lower():
            return p, ""
    if not subs:
        return None, f"SKIP: {outer.name} has no session folder inside"
    if len(subs) == 1:
        return subs[0], f"via {subs[0].name} (name differs from local)"
    names = ", ".join(p.name for p in subs)
    return None, f"SKIP: ambiguous inside {outer.name} ({names})"


def move_one(src: Path, dst: Path, apply: bool, overwrite: bool) -> str:
    """Return a one-word status string."""
    if dst.exists():
        if sha256(src) == sha256(dst):
            if apply:
                src.unlink()
            return "already-there (source removed)" if apply else "already-there"
        if not overwrite:
            return "SKIP: destination exists with different content"
        # fall through and overwrite

    if not apply:
        return "would move"

    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copy2(src, tmp)
    if sha256(tmp) != sha256(src):
        tmp.unlink(missing_ok=True)
        return "FAIL: checksum mismatch, source kept"
    tmp.replace(dst)          # atomic-ish swap on the same volume
    src.unlink()              # only now is the source dropped
    return "moved"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a differing file already on the server")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--src", type=Path, default=SRC_ROOT)
    ap.add_argument("--dst", type=Path, default=DST_ROOT)
    args = ap.parse_args()

    for root, label in ((args.src, "source"), (args.dst, "destination")):
        if not root.is_dir():
            print(f"ERROR: {label} root not reachable: {root}")
            return 2

    print(f"{'APPLY' if args.apply else 'DRY RUN'}   pattern={args.pattern}")
    print(f"  from {args.src}\n    to {args.dst}\n")

    sources = find_sources(args.src, args.pattern)
    if not sources:
        print("No bad-channel files found.")
        return 0

    index = index_server(args.dst)

    counts: dict[str, int] = {}
    for session, src in sources:
        dst_dir, note = resolve_dst_dir(session, index)
        if dst_dir is None:
            status, where = note, ""
        else:
            status = move_one(src, dst_dir / src.name, args.apply, args.overwrite)
            where = str(dst_dir.relative_to(args.dst)).replace("\\", "/")
        key = status.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if note and dst_dir is not None:
            status = f"{status}  [{note}]"
        print(f"  {session}/{src.name:<24} -> {where:<30} {status}")

    print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not args.apply:
        print("nothing was changed -- rerun with --apply once the plan looks right")
    return 0


if __name__ == "__main__":
    sys.exit(main())