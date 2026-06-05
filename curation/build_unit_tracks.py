r"""
build_unit_tracks.py
====================
Assemble cross-session unit *tracks* from the all-pairs match output of
match_units_CnL42SG.py, and visualize them.

match_units_CnL42SG.py writes pairwise matches only: each row of
``matched_units_all_pairs.csv`` is one edge linking a unit in session A to a
unit in session B. This script turns those edges into tracks so questions like
"which units appear on days 1, 2 and 5 but not 3 and 4?" can be read directly.

Method
------
Treat every ``(session, unit_id)`` as a node and every matched pair as an
undirected edge, then take connected components (union-find). Each component is
a track: the same physical unit followed across sessions. Because all-pairs
tests non-adjacent sessions too, a unit that skips days still links across the
gap (the (1,5) and (2,5) edges survive even with no day-3/4 match) — which a
consecutive-only scheme would have lost.

Two consistency flags are reported because independent per-pair matching does
not guarantee a globally clean track:
  - max_units_per_session > 1  -> the component pulled in >1 unit from some day
    (ambiguous merge; inspect before trusting).
  - mean_score / n_sessions    -> long tracks built from borderline edges are
    where transitive drift would show up.

Outputs (written next to the input CSV)
---------------------------------------
  unit_tracks.csv        - one row per track; one column per session (chrono),
                           cell = matched unit id ("" where the unit is absent,
                           so day-3/4 gaps render as blanks), plus track_id,
                           shank, n_sessions, n_units, max_units_per_session,
                           mean_score, first_session, last_session, span_days.
  unit_tracks_presence.pdf/.png
                         - presence matrix: tracks (rows) x sessions (cols),
                           cells colored by shank where present, blank where
                           absent; top panel = active-track count per session.

Usage
-----
Set THRES in the USER CONFIG block to the match run you want to read, then:
  python build_unit_tracks.py
It reads OUTPUT_FOLDER_BASE / thres_<THRES> (the subfolder match_units_CnL42SG
wrote). CLI flags still exist (--output-folder, --min-sessions, --one-to-one /
--raw-components) but are optional.

Tracks are built one-to-one by default (ONE_TO_ONE): each track holds at most
one unit per session, so the ambiguous mega-merges are split into clean tracks
(more, and more trustworthy, tracks). Pass --raw-components for the old behavior.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# ============================== USER CONFIG ==============================
# Base output folder — must match match_units_CnL42SG.OUTPUT_FOLDER. The actual
# results live in a threshold-named subfolder under here (e.g. thres_0.70).
OUTPUT_FOLDER_BASE = Path(
    r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\unit_match_all_pairs"
)

# Threshold of the match run to read. Set THRES to the value you passed to
# match_units_CnL42SG (--thres / its `thres` constant); the four floors below
# follow it. For a mixed-threshold run, edit the four individually to match.
THRES = 0.6
MIN_TOTAL = THRES
MIN_WAVEFORM = 0.3
MIN_AMPLITUDE = THRES
MIN_AUTOCORR = THRES

# Keep tracks spanning at least this many sessions.
MIN_SESSIONS = 2

# Split over-merged components so each track holds at most ONE unit per session.
# Per-pair matching is not globally one-to-one, so one unit can bridge many
# others into ambiguous mega-merges (max_units_per_session up to ~20) that each
# collapse to a single track. With this on, those blobs fragment into clean
# single-unit-per-session tracks — which both cleans the output and *increases*
# the track count. Set False to keep the raw connected components of the graph.
ONE_TO_ONE = True
# ========================================================================


def threshold_tag(min_total: float, min_waveform: float, min_amplitude: float,
                  min_autocorr: float) -> str:
    """Threshold subfolder name. MUST stay identical to the same-named function
    in match_units_CnL42SG.py so this script reads the folder that one wrote."""
    vals = (min_total, min_waveform, min_amplitude, min_autocorr)
    if len(set(vals)) == 1:
        return f"thres_{vals[0]:.2f}"
    return (f"t{min_total:.2f}_w{min_waveform:.2f}"
            f"_a{min_amplitude:.2f}_ac{min_autocorr:.2f}")


# Resolved input/output folder for this threshold (code-driven; no CLI needed).
OUTPUT_FOLDER = OUTPUT_FOLDER_BASE / threshold_tag(
    MIN_TOTAL, MIN_WAVEFORM, MIN_AMPLITUDE, MIN_AUTOCORR
)


# ── Union-find ───────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ── Graph → components ─────────────────────────────────────────────────────────────
def build_components(matches: pd.DataFrame, one_to_one: bool):
    """Union matched units into tracks (connected components).

    Returns ``(uf, node_shank, scores_by_root)``: the final union-find, a map
    from each (session, unit_id) node to its shank string, and the total_score
    of every edge kept in each component (for the mean_score diagnostic).

    one_to_one=False reproduces the original behavior — every matched pair is
    unioned, so each component is a raw connected component of the match graph.
    Because per-pair matching is not globally one-to-one, one unit can bridge
    many others into ambiguous mega-merges (max_units_per_session up to ~20),
    and each such blob counts as a single track.

    one_to_one=True consumes edges highest-score-first and rejects any merge
    whose two components already share a session, so no track ever holds two
    units from the same day. This splits those blobs into many clean
    single-unit-per-session tracks — which both cleans the output and raises the
    track count.
    """
    uf = UnionFind()
    node_shank: dict[tuple, str] = {}
    edges: list[tuple[float, tuple, tuple]] = []
    for _, row in matches.iterrows():
        a = (row["session1_recording"], row["session1_unit_id"])
        b = (row["session2_recording"], row["session2_unit_id"])
        node_shank.setdefault(a, row.get("session1_shanks", ""))
        node_shank.setdefault(b, row.get("session2_shanks", ""))
        try:
            score = float(row["total_score"])
        except (ValueError, KeyError):
            score = float("nan")
        edges.append((score, a, b))

    accepted: list[tuple[float, tuple, tuple]]
    if one_to_one:
        # Each component carries the set of sessions it already covers; a merge
        # that would collide on a session is dropped so the unit stays separate.
        # Nodes start as their own root, so keying by the node IS keying by root.
        sessions_of: dict[tuple, set] = {node: {node[0]} for node in node_shank}
        # Best edges claim their pairing first; NaN-scored edges are tried last.
        order = sorted(edges, key=lambda e: -e[0] if e[0] == e[0] else float("inf"))
        accepted = []
        for score, a, b in order:
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                accepted.append((score, a, b))  # already together; keep its score
                continue
            if sessions_of[ra] & sessions_of[rb]:
                continue  # merge would put two units in one session -> split off
            merged = sessions_of[ra] | sessions_of[rb]
            uf.union(a, b)
            sessions_of[uf.find(a)] = merged  # union() makes rb the survivor
            accepted.append((score, a, b))
    else:
        for _, a, b in edges:
            uf.union(a, b)
        accepted = edges

    scores_by_root: dict[tuple, list[float]] = {}
    for score, a, b in accepted:
        if score == score:  # drop NaN
            scores_by_root.setdefault(uf.find(a), []).append(score)
    return uf, node_shank, scores_by_root


# ── Helpers ──────────────────────────────────────────────────────────────────────
def session_date(session_name: str) -> int:
    """Sortable YYYYMMDD int parsed from a session name; 0 if none found."""
    match = re.search(r"(\d{8})", str(session_name))
    return int(match.group(1)) if match else 0


def session_order(matches: pd.DataFrame, units: pd.DataFrame | None) -> list[str]:
    """Chronological session list. Prefer session_units.csv (includes sessions
    with zero matches); fall back to whatever appears in the edges."""
    names: set[str] = set()
    if units is not None and "session" in units.columns:
        names.update(units["session"].astype(str))
    names.update(matches["session1_recording"].astype(str))
    names.update(matches["session2_recording"].astype(str))
    return sorted(names, key=session_date)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-folder", type=Path, default=OUTPUT_FOLDER,
                        help="defaults to the threshold subfolder from the config block")
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS,
                        help="keep tracks spanning at least this many sessions")
    parser.add_argument("--one-to-one", dest="one_to_one", action="store_true",
                        default=ONE_TO_ONE,
                        help="split components so each track has <=1 unit/session (default)")
    parser.add_argument("--raw-components", dest="one_to_one", action="store_false",
                        help="keep raw connected components (may merge many units/session)")
    args = parser.parse_args()

    folder = args.output_folder
    matches_path = folder / "matched_units_all_pairs.csv"
    units_path = folder / "session_units.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Not found: {matches_path}\nRun match_units_CnL42SG.py first.")

    matches = pd.read_csv(matches_path, dtype=str).fillna("")
    units = pd.read_csv(units_path, dtype=str).fillna("") if units_path.exists() else None
    sessions = session_order(matches, units)
    sess_index = {name: i for i, name in enumerate(sessions)}

    if matches.empty:
        print("No matched pairs in the CSV — no tracks to build.")
        return

    # ── Build the graph and connected components ───────────────────────────────
    uf, node_shank, scores_by_root = build_components(matches, args.one_to_one)

    # Group nodes by component root.
    comps: dict[tuple, list[tuple]] = {}
    for node in node_shank:
        comps.setdefault(uf.find(node), []).append(node)

    # ── Build the track table ───────────────────────────────────────────────────
    records = []
    for root, nodes in comps.items():
        by_session: dict[str, list[str]] = {}
        shanks: list[str] = []
        for sess, uid in nodes:
            by_session.setdefault(sess, []).append(str(uid))
            sh = node_shank.get((sess, uid), "")
            if sh != "":
                shanks.append(sh)
        n_sessions = len(by_session)
        if n_sessions < args.min_sessions:
            continue
        present = sorted(by_session, key=session_date)
        shank = pd.Series(shanks).mode().iloc[0] if shanks else ""
        scores = scores_by_root.get(root, [])
        rec = {
            "shank": shank,
            "n_sessions": n_sessions,
            "n_units": len(nodes),
            "max_units_per_session": max(len(v) for v in by_session.values()),
            "mean_score": round(float(np.nanmean(scores)), 4) if scores else float("nan"),
            "first_session": present[0],
            "last_session": present[-1],
            "span_days": _span_days(present[0], present[-1]),
            "_by_session": {s: "|".join(sorted(v, key=_as_num)) for s, v in by_session.items()},
        }
        records.append(rec)

    if not records:
        print(f"No tracks span >= {args.min_sessions} sessions.")
        return

    # Sort tracks: by shank, then first appearance, then longer tracks first.
    records.sort(key=lambda r: (_as_num(r["shank"]), session_date(r["first_session"]),
                                -r["n_sessions"]))

    # ── Write unit_tracks.csv ───────────────────────────────────────────────────
    meta_cols = ["track_id", "shank", "n_sessions", "n_units", "max_units_per_session",
                 "mean_score", "first_session", "last_session", "span_days"]
    rows_out = []
    for tid, rec in enumerate(records, start=1):
        out = {c: rec.get(c, "") for c in meta_cols}
        out["track_id"] = tid
        for sess in sessions:
            out[sess] = rec["_by_session"].get(sess, "")
        rows_out.append(out)
    tracks_df = pd.DataFrame(rows_out, columns=meta_cols + sessions)
    tracks_csv = folder / "unit_tracks.csv"
    tracks_df.to_csv(tracks_csv, index=False)

    # ── Visualize: presence matrix ──────────────────────────────────────────────
    _plot_presence(records, sessions, sess_index, folder)

    # ── Summary ─────────────────────────────────────────────────────────────────
    multi = [r for r in records if r["n_sessions"] >= args.min_sessions]
    ambiguous = sum(1 for r in records if r["max_units_per_session"] > 1)
    longest = max(records, key=lambda r: r["n_sessions"])
    print(f"\n{'='*60}")
    print(f"UNIT TRACKS  (>= {args.min_sessions} sessions)")
    print(f"  Mode                   : {'one-to-one' if args.one_to_one else 'raw components'}")
    print(f"  Tracks                 : {len(multi)}")
    print(f"  Ambiguous (>1 unit/day): {ambiguous}")
    print(f"  Longest track          : {longest['n_sessions']} sessions "
          f"(shank {longest['shank']}, {longest['first_session']} -> {longest['last_session']})")
    print(f"  Table                  : {tracks_csv}")
    print(f"  Figure                 : {folder / 'unit_tracks_presence.pdf'}")
    print(f"{'='*60}")


def _as_num(value):
    """Sort key tolerant of non-numeric ids/shanks."""
    try:
        return (0, float(value))
    except (ValueError, TypeError):
        return (1, str(value))


def _span_days(first: str, last: str) -> int:
    a, b = session_date(first), session_date(last)
    if not a or not b:
        return 0
    from datetime import date
    try:
        da = date(a // 10000, (a // 100) % 100, a % 100)
        db = date(b // 10000, (b // 100) % 100, b % 100)
        return (db - da).days
    except ValueError:
        return 0


def _plot_presence(records, sessions, sess_index, folder) -> None:
    """Presence matrix: tracks (rows) x sessions (cols), colored by shank."""
    n_tracks = len(records)
    n_sess = len(sessions)

    # Matrix of shank value where present, NaN where absent (masked -> blank).
    mat = np.full((n_tracks, n_sess), np.nan)
    for r_idx, rec in enumerate(records):
        try:
            sh = int(float(rec["shank"]))
        except (ValueError, TypeError):
            sh = 0
        for sess in rec["_by_session"]:
            if sess in sess_index:
                mat[r_idx, sess_index[sess]] = sh
    masked = np.ma.masked_invalid(mat)

    shank_vals = sorted({int(v) for v in masked.compressed()}) if masked.count() else [0]
    base = plt.get_cmap("tab10")
    cmap = ListedColormap([base(s % 10) for s in range(max(shank_vals) + 1)])
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(-0.5, max(shank_vals) + 1.5, 1), cmap.N)

    fig_h = max(4.0, 0.16 * n_tracks + 2.0)
    fig = plt.figure(figsize=(max(8.0, 0.42 * n_sess + 3.0), fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, max(4, n_tracks)], hspace=0.04)

    # Top: active-track count per session.
    ax_top = fig.add_subplot(gs[0])
    active = np.array([np.sum(~masked.mask[:, j]) for j in range(n_sess)])
    ax_top.bar(np.arange(n_sess), active, color="#555555", width=0.85)
    ax_top.set_ylabel("# active\ntracks", fontsize=8)
    ax_top.set_xlim(-0.5, n_sess - 0.5)
    ax_top.set_xticks([])
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.tick_params(labelsize=7)

    # Bottom: the presence matrix.
    ax = fig.add_subplot(gs[1])
    ax.imshow(masked, aspect="auto", interpolation="none", cmap=cmap, norm=norm,
              extent=(-0.5, n_sess - 0.5, n_tracks - 0.5, -0.5))
    ax.set_xticks(np.arange(n_sess))
    ax.set_xticklabels([str(session_date(s))[2:] for s in sessions], rotation=90, fontsize=7)
    ax.set_xlabel("session (YYMMDD)")
    ax.set_ylabel(f"track (n={n_tracks}, sorted by shank then first appearance)")
    ax.set_yticks([])
    ax.set_xlim(-0.5, n_sess - 0.5)
    for x in np.arange(-0.5, n_sess, 1):
        ax.axvline(x, color="#dddddd", linewidth=0.4)

    handles = [Patch(facecolor=cmap(s), edgecolor="none", label=f"shank {s}") for s in shank_vals]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
              fontsize=8, title="shank", title_fontsize=8, frameon=False)

    fig.suptitle("CnL42SG cross-session unit tracks — presence matrix", fontsize=12)
    fig.savefig(folder / "unit_tracks_presence.pdf", bbox_inches="tight")
    fig.savefig(folder / "unit_tracks_presence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
