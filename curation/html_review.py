"""
html_review.py
==============
Generate an interactive HTML review for spike-sorting units, served by a
local HTTP server so that label and merge changes are written back to disk
automatically.

Sortout root
------------
The folder you pass must contain (at any depth) directories matching
``**/sorting_results_*/raw_units/unit_summary_*.png`` (MountainSort / 1_ms_sorting
layout). That can be a session folder (…/animal/session), the animal folder above
it, or any parent that contains those shank runs.

Override the built-in default path with::

    python html_review.py --serve --sortout "D:\\data\\sortout\\Mouse1\\Mouse1_20260101"

or environment variable ``SPIKE_SORTOUT`` / ``SORTOUT_FOLDER``.

Two modes
---------
Static export (original behaviour):
    python html_review.py            # writes review.html to sortout_folder
    python html_review.py --open     # also opens the browser

Interactive server (recommended):
    python html_review.py --serve    # starts server, opens browser, auto-saves
    python html_review.py --serve --port 8080

Each card shows:
  • Unit summary image  •  Key quality metrics  •  Auto-classification label
  •  Label buttons: SUA / MUA / Noise

Merge mode (sidebar):
  • Toggle Merge Mode  •  Click cards to select units for merging
  • Merge Preview panel — appears automatically when 2+ units are selected:
      – Probe-layout waveform overlay (all top channels at real positions)
      – ACG for each unit and CCG between them
      – Metrics table: distance, Pearson R, amplitude ratio, post-merge ISI,
        ACG similarity — with pass/fail per criterion (same thresholds as auto-curation)
  • Merged pairs page (header tab): all unit pairs in current merge groups, each
    with the same preview plots (lazy-loaded; requires --serve).

Sync strategy
-------------
All classification logic, metrics infrastructure, and threshold constants are
imported directly from curation_lazy.py.

Usage from curation_lazy
------------------------
    from html_review import launch_server
    launch_server(open_browser=True)   # blocks until Ctrl-C
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import spikeinterface as si

# ── Import everything shared from curation_lazy ───────────────────────────────
from curation_lazy import (
    # thresholds
    NOISE_SNR_THRESHOLD, NOISE_PRESENCE_THRESHOLD, NOISE_ISI_THRESHOLD,
    SUA_SNR_THRESHOLD, SUA_ISI_RATIO_THRESHOLD, SUA_FIRING_RATE_MIN,
    SUA_RP_THRESHOLD, SUA_AMPLITUDE_CUTOFF_THRESHOLD,
    # merge thresholds
    MERGE_DISTANCE_THRESH_UM, MERGE_PEARSON_THRESH,
    MERGE_AMPLITUDE_RATIO_THRESH, MERGE_ISI_THRESH,
    # classification
    auto_classify, _get_reject_reasons, _get_sua_failures,
    # metrics infrastructure
    _load_or_compute_qm, compute_bleed_flag, _get_templates,
    # merge helpers
    _cast_uid, _isi_ratio_from_spike_train,
    discover_units,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
_DEFAULT_SORTOUT = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL43\CnL43_20260408")


def _sortout_from_env_or_default() -> Path:
    for key in ("SPIKE_SORTOUT", "SORTOUT_FOLDER"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return Path(_DEFAULT_SORTOUT)


sortout_folder = _sortout_from_env_or_default()
output_json    = sortout_folder / "unit_labels.json"
output_html    = sortout_folder / "review.html"
merge_json     = sortout_folder / "unit_merge_map.json"


def configure_sortout(path: str | Path | None) -> None:
    """Point review I/O at a session or animal folder (contains shank* / sorting_results_*)."""
    global sortout_folder, output_json, output_html, merge_json
    if path is None:
        return
    sortout_folder = Path(path).expanduser().resolve()
    output_json = sortout_folder / "unit_labels.json"
    output_html = sortout_folder / "review.html"
    merge_json = sortout_folder / "unit_merge_map.json"


# ── Analyzer / metrics ────────────────────────────────────────────────────────
_analyzer_cache: dict = {}
_metrics_cache:  dict = {}


def _get_analyzer(rec_name: str):
    if rec_name in _analyzer_cache:
        return _analyzer_cache[rec_name]
    rec_dir = sortout_folder / rec_name
    sorting_dirs = sorted(rec_dir.glob("sorting_results_*"))
    if not sorting_dirs:
        return None
    analyzer_folder = sorting_dirs[-1] / "sorting_analyzer"
    if not analyzer_folder.is_dir():
        return None
    try:
        sa = si.load_sorting_analyzer(str(analyzer_folder))
        _analyzer_cache[rec_name] = sa
        return sa
    except Exception as e:
        print(f"[Analyzer] {analyzer_folder}: {e}")
        return None


def load_metrics(rec_name: str) -> dict:
    if rec_name in _metrics_cache:
        return _metrics_cache[rec_name]
    print(f"[Metrics] Loading {rec_name} ...")
    sa = _get_analyzer(rec_name)
    if sa is None:
        _metrics_cache[rec_name] = {}
        return {}
    try:
        _get_templates(sa)
        df = _load_or_compute_qm(sa)
        df["bleed_flag"] = [compute_bleed_flag(sa, uid) for uid in df.index]
        result = {str(uid): row.to_dict() for uid, row in df.iterrows()}
        _metrics_cache[rec_name] = result
        return result
    except Exception as e:
        print(f"[Metrics] Error {rec_name}: {e}")
        _metrics_cache[rec_name] = {}
        return {}


# ── Merge map helpers ─────────────────────────────────────────────────────────
def _merge_map_to_groups(merge_map_per_rec: dict) -> dict:
    groups = {}
    for rec, uid_map in merge_map_per_rec.items():
        canonical_to_members: dict = {}
        for uid, canonical in uid_map.items():
            canonical_to_members.setdefault(canonical, [])
            if uid not in canonical_to_members[canonical]:
                canonical_to_members[canonical].append(uid)
        non_trivial = [m for m in canonical_to_members.values() if len(m) > 1]
        if non_trivial:
            groups[rec] = non_trivial
    return groups


def _groups_to_merge_map(groups_per_rec: dict) -> dict:
    result = {}
    for rec, groups in groups_per_rec.items():
        result[rec] = {}
        for group in groups:
            canonical = group[0]
            for uid in group:
                result[rec][uid] = canonical
    return result


# ── Merge preview helpers ─────────────────────────────────────────────────────
def _compute_correlogram(st_a: np.ndarray, st_b: np.ndarray, fs: float,
                          window_ms: float = 15.0, bin_ms: float = 0.5,
                          is_acg: bool = False):
    """Return (bin_centers_ms, counts) for a cross- or auto-correlogram."""
    window_samp = int(window_ms / 1000.0 * fs)
    lags = []
    for ta in st_a:
        lo = int(np.searchsorted(st_b, ta - window_samp, "left"))
        hi = int(np.searchsorted(st_b, ta + window_samp, "right"))
        chunk = st_b[lo:hi].astype(np.int64) - np.int64(ta)
        if is_acg:
            chunk = chunk[chunk != 0]
        if len(chunk):
            lags.append(chunk)

    n_bins = int(2 * window_ms / bin_ms)
    if not lags:
        centers = np.linspace(-window_ms + bin_ms / 2, window_ms - bin_ms / 2, n_bins)
        return centers.tolist(), [0] * n_bins

    all_ms = np.concatenate(lags).astype(np.float64) / fs * 1000.0
    edges = np.linspace(-window_ms, window_ms, n_bins + 1)
    counts, _ = np.histogram(all_ms, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers.tolist(), counts.tolist()


def _compute_merge_preview(rec_name: str, uid_a_str: str, uid_b_str: str) -> dict:
    """
    Compute everything needed by the merge preview panel:
      - Template waveforms on the top channels (probe-layout positions included)
      - ACG for each unit, CCG between them
      - Merge-criteria metrics (distance, Pearson R, amplitude ratio, ISI merged, ACG sim)
    """
    from scipy.stats import pearsonr

    sa = _get_analyzer(rec_name)
    if sa is None:
        return {"error": f"No analyzer for {rec_name}"}

    templates = _get_templates(sa)
    if templates is None:
        return {"error": "Templates not available"}

    unit_ids = list(sa.unit_ids)
    uid_a = _cast_uid(uid_a_str, unit_ids)
    uid_b = _cast_uid(uid_b_str, unit_ids)
    if uid_a is None or uid_b is None:
        return {"error": f"Unit not found: {uid_a_str} or {uid_b_str}"}

    idx_a = unit_ids.index(uid_a)
    idx_b = unit_ids.index(uid_b)
    tmpl_a = templates[idx_a]   # (n_samples, n_channels)
    tmpl_b = templates[idx_b]

    sorting = sa.sorting
    fs      = float(sorting.get_sampling_frequency())
    n_samp  = tmpl_a.shape[0]
    n_chan   = tmpl_a.shape[1]

    p2p_a = tmpl_a.max(0) - tmpl_a.min(0)   # (n_channels,)
    p2p_b = tmpl_b.max(0) - tmpl_b.min(0)
    pri_a = int(p2p_a.argmax())
    pri_b = int(p2p_b.argmax())

    # Primary channel + 2 nearest neighbors by probe distance
    p2p_comb    = p2p_a + p2p_b
    pri_combined = int(p2p_comb.argmax())
    ch_locs  = sa.get_channel_locations()    # (n_channels, 2)
    dists    = np.linalg.norm(ch_locs - ch_locs[pri_combined], axis=1)
    top_chs  = np.argsort(dists)[:3]        # primary + 2 nearest
    time_ms  = ((np.arange(n_samp) - n_samp // 2) / fs * 1000.0).tolist()

    channels = [
        {
            "ch":  int(ch),
            "x":   float(ch_locs[ch, 0]),
            "y":   float(ch_locs[ch, 1]),
            "pri_a": bool(ch == pri_a),
            "pri_b": bool(ch == pri_b),
            "wv_a": tmpl_a[:, ch].tolist(),
            "wv_b": tmpl_b[:, ch].tolist(),
        }
        for ch in top_chs
    ]

    # Spike trains
    st_a = sorting.get_unit_spike_train(unit_id=uid_a)
    st_b = sorting.get_unit_spike_train(unit_id=uid_b)

    acg_a_bins, acg_a_cnt = _compute_correlogram(st_a, st_a, fs, is_acg=True)
    acg_b_bins, acg_b_cnt = _compute_correlogram(st_b, st_b, fs, is_acg=True)
    ccg_bins,   ccg_cnt   = _compute_correlogram(st_a, st_b, fs, is_acg=False)

    # Metrics
    vec_a = tmpl_a.flatten().astype(float)
    vec_b = tmpl_b.flatten().astype(float)
    pr = float(pearsonr(vec_a, vec_b)[0]) if (np.std(vec_a) > 0 and np.std(vec_b) > 0) else 0.0

    amp_a    = float(p2p_a[pri_a])
    amp_b    = float(p2p_b[pri_b])
    amp_ratio = float(max(amp_a, amp_b) / (min(amp_a, amp_b) + 1e-9))

    dist = float(np.linalg.norm(ch_locs[pri_a] - ch_locs[pri_b]))

    st_merged   = np.sort(np.concatenate([st_a, st_b]))
    isi_merged  = _isi_ratio_from_spike_train(st_merged, fs)

    # ACG similarity: Pearson correlation of normalised ACG histograms
    na = np.array(acg_a_cnt, dtype=float);  na /= (na.sum() or 1)
    nb = np.array(acg_b_cnt, dtype=float);  nb /= (nb.sum() or 1)
    acg_sim = float(np.corrcoef(na, nb)[0, 1]) if (np.std(na) > 0 and np.std(nb) > 0) else 0.0

    criteria = {
        "distance":       dist       <  MERGE_DISTANCE_THRESH_UM,
        "pearson_r":      pr         >  MERGE_PEARSON_THRESH,
        "amplitude_ratio":amp_ratio  <  MERGE_AMPLITUDE_RATIO_THRESH,
        "isi_merged":     isi_merged <  MERGE_ISI_THRESH,
    }

    return {
        "uid_a": uid_a_str,
        "uid_b": uid_b_str,
        "channels": channels,
        "time_ms":  time_ms,
        "acg_a": {"bins": acg_a_bins, "counts": acg_a_cnt, "n": int(len(st_a))},
        "acg_b": {"bins": acg_b_bins, "counts": acg_b_cnt, "n": int(len(st_b))},
        "ccg":   {"bins": ccg_bins,   "counts": ccg_cnt},
        "metrics": {
            "distance_um":     dist,
            "pearson_r":       pr,
            "amplitude_ratio": amp_ratio,
            "isi_merged":      isi_merged,
            "acg_similarity":  acg_sim,
            "n_spikes_a":      int(len(st_a)),
            "n_spikes_b":      int(len(st_b)),
        },
        "criteria": criteria,
        "n_pass": sum(criteria.values()),
        "thresholds": {
            "distance_um":     MERGE_DISTANCE_THRESH_UM,
            "pearson_r":       MERGE_PEARSON_THRESH,
            "amplitude_ratio": MERGE_AMPLITUDE_RATIO_THRESH,
            "isi_merged":      MERGE_ISI_THRESH,
        },
    }


def _compute_merge_group_preview(rec_name: str, uid_strs: list) -> dict:
    """
    Compute a full group-level merge preview for N units:
      - Template waveforms overlaid on probe layout (all N units, each a different colour)
      - ACG for each unit
      - CCG for every unique pair
      - Pairwise merge-criteria metrics (distance, Pearson R, amplitude ratio, post-merge ISI,
        ACG similarity) with pass/fail flags
      - Per-unit quality metrics (SNR, FR, ISI, etc.)
    """
    from scipy.stats import pearsonr

    sa = _get_analyzer(rec_name)
    if sa is None:
        return {"error": f"No analyzer for {rec_name}"}

    templates = _get_templates(sa)
    if templates is None:
        return {"error": "Templates not available"}

    unit_ids   = list(sa.unit_ids)
    uids       = [_cast_uid(s, unit_ids) for s in uid_strs]
    if any(u is None for u in uids):
        missing = [s for s, u in zip(uid_strs, uids) if u is None]
        return {"error": f"Units not found: {missing}"}

    idxs    = [unit_ids.index(u) for u in uids]
    sorting = sa.sorting
    fs      = float(sorting.get_sampling_frequency())
    n_samp  = templates.shape[1]
    n_chan  = templates.shape[2]

    p2ps    = [templates[i].max(0) - templates[i].min(0) for i in idxs]  # (n_chan,) each
    pri_chs = [int(p.argmax()) for p in p2ps]

    # Primary channel + 2 nearest neighbors by probe distance
    p2p_comb     = sum(p2ps)
    pri_combined = int(p2p_comb.argmax())
    ch_locs      = sa.get_channel_locations()
    dists        = np.linalg.norm(ch_locs - ch_locs[pri_combined], axis=1)
    top_chs      = np.argsort(dists)[:3]    # primary + 2 nearest
    time_ms  = ((np.arange(n_samp) - n_samp // 2) / fs * 1000.0).tolist()

    channels = [
        {
            "ch":        int(ch),
            "x":         float(ch_locs[ch, 0]),
            "y":         float(ch_locs[ch, 1]),
            "waveforms": [templates[i][:, ch].tolist() for i in idxs],
            "pri":       [bool(ch == pc) for pc in pri_chs],
        }
        for ch in top_chs
    ]

    # Spike trains & ACGs
    spike_trains = [sorting.get_unit_spike_train(unit_id=u) for u in uids]
    acg_counts_list = []
    acgs = []
    for k, (uid_str, st) in enumerate(zip(uid_strs, spike_trains)):
        bins, cnts = _compute_correlogram(st, st, fs, is_acg=True)
        acg_counts_list.append(cnts)
        acgs.append({"uid": uid_str, "bins": bins, "counts": cnts, "n": int(len(st))})

    # CCGs (all unique pairs)
    ccgs = []
    for i in range(len(uid_strs)):
        for j in range(i + 1, len(uid_strs)):
            bins, cnts = _compute_correlogram(spike_trains[i], spike_trains[j], fs, is_acg=False)
            ccgs.append({"uid_a": uid_strs[i], "uid_b": uid_strs[j], "bins": bins, "counts": cnts})

    # Pairwise criteria
    pair_criteria = []
    for i in range(len(uid_strs)):
        for j in range(i + 1, len(uid_strs)):
            vec_a = templates[idxs[i]].flatten().astype(float)
            vec_b = templates[idxs[j]].flatten().astype(float)
            pr    = float(pearsonr(vec_a, vec_b)[0]) if (np.std(vec_a) > 0 and np.std(vec_b) > 0) else 0.0

            amp_a     = float(p2ps[i][pri_chs[i]])
            amp_b     = float(p2ps[j][pri_chs[j]])
            amp_ratio = float(max(amp_a, amp_b) / (min(amp_a, amp_b) + 1e-9))
            dist      = float(np.linalg.norm(ch_locs[pri_chs[i]] - ch_locs[pri_chs[j]]))

            st_merged  = np.sort(np.concatenate([spike_trains[i], spike_trains[j]]))
            isi_merged = _isi_ratio_from_spike_train(st_merged, fs)

            na = np.array(acg_counts_list[i], dtype=float); na /= (na.sum() or 1)
            nb = np.array(acg_counts_list[j], dtype=float); nb /= (nb.sum() or 1)
            acg_sim = float(np.corrcoef(na, nb)[0, 1]) if (np.std(na) > 0 and np.std(nb) > 0) else 0.0

            criteria = {
                "distance":        dist      <  MERGE_DISTANCE_THRESH_UM,
                "pearson_r":       pr        >  MERGE_PEARSON_THRESH,
                "amplitude_ratio": amp_ratio <  MERGE_AMPLITUDE_RATIO_THRESH,
                "isi_merged":      isi_merged < MERGE_ISI_THRESH,
            }
            pair_criteria.append({
                "uid_a":           uid_strs[i],
                "uid_b":           uid_strs[j],
                "distance_um":     dist,
                "pearson_r":       pr,
                "amplitude_ratio": amp_ratio,
                "isi_merged":      isi_merged,
                "acg_similarity":  acg_sim,
                "n_pass":          sum(criteria.values()),
                "criteria":        criteria,
            })

    # Per-unit quality metrics
    qm = load_metrics(rec_name)

    def _fmt(v, d=3, sfx=""):
        if v is None: return "n/a"
        try: return f"{float(v):.{d}f}{sfx}"
        except: return str(v)

    unit_metrics = {}
    for uid_str in uid_strs:
        m = qm.get(uid_str, {})
        unit_metrics[uid_str] = {
            "snr":      _fmt(m.get("snr"), 2),
            "fr":       _fmt(m.get("firing_rate"), 3, " Hz"),
            "isi":      _fmt(m.get("isi_violations_ratio"), 4),
            "rp_cont":  _fmt(m.get("rp_contamination"), 4),
            "amp_cut":  _fmt(m.get("amplitude_cutoff"), 3),
            "presence": _fmt(m.get("presence_ratio"), 2),
            "bleed":    "YES" if m.get("bleed_flag") else "no",
            "auto":     auto_classify(m),
        }

    return {
        "uids":         uid_strs,
        "rec":          rec_name,
        "channels":     channels,
        "time_ms":      time_ms,
        "acgs":         acgs,
        "ccgs":         ccgs,
        "pair_criteria": pair_criteria,
        "unit_metrics": unit_metrics,
        "thresholds": {
            "distance_um":     MERGE_DISTANCE_THRESH_UM,
            "pearson_r":       MERGE_PEARSON_THRESH,
            "amplitude_ratio": MERGE_AMPLITUDE_RATIO_THRESH,
            "isi_merged":      MERGE_ISI_THRESH,
        },
    }


def _compute_candidates(rec_name: str, group_uid_strs: list, n: int = 15) -> dict:
    """
    For units NOT in group_uid_strs, compute Pearson R of their flattened template
    against the group centroid template (mean of group members), plus distance and
    amplitude ratio. Return top-N ranked by Pearson R descending.
    ISI criterion is omitted here (expensive); it runs when user clicks Preview.
    """
    from scipy.stats import pearsonr

    sa = _get_analyzer(rec_name)
    if sa is None:
        return {"error": f"No analyzer for {rec_name}"}
    templates = _get_templates(sa)
    if templates is None:
        return {"error": "Templates not available"}

    unit_ids  = list(sa.unit_ids)
    group_ids = [_cast_uid(s, unit_ids) for s in group_uid_strs]
    if any(u is None for u in group_ids):
        return {"error": "One or more group units not found"}

    idxs    = [unit_ids.index(u) for u in group_ids]
    ch_locs = sa.get_channel_locations()  # (n_chan, 2)

    # Mean template across group members as centroid
    centroid  = np.stack([templates[i] for i in idxs]).mean(axis=0)  # (n_samp, n_chan)
    cent_flat = centroid.flatten().astype(float)
    cent_p2p  = centroid.max(0) - centroid.min(0)
    cent_pri  = int(cent_p2p.argmax())
    cent_amp  = float(cent_p2p[cent_pri])

    idx_set = set(idxs)
    candidates = []
    for i, uid in enumerate(unit_ids):
        if i in idx_set:
            continue
        tmpl      = templates[i]
        flat      = tmpl.flatten().astype(float)
        p2p       = tmpl.max(0) - tmpl.min(0)
        pri       = int(p2p.argmax())
        amp       = float(p2p[pri])
        dist      = float(np.linalg.norm(ch_locs[pri] - ch_locs[cent_pri]))
        pr        = float(pearsonr(cent_flat, flat)[0]) \
                    if np.std(cent_flat) > 0 and np.std(flat) > 0 else 0.0
        amp_ratio = float(max(amp, cent_amp) / (min(amp, cent_amp) + 1e-9))
        criteria  = {
            "distance":        dist      < MERGE_DISTANCE_THRESH_UM,
            "pearson_r":       pr        > MERGE_PEARSON_THRESH,
            "amplitude_ratio": amp_ratio < MERGE_AMPLITUDE_RATIO_THRESH,
        }
        candidates.append({
            "uid":         str(uid),
            "pearson_r":   round(pr, 3),
            "distance_um": round(dist, 1),
            "amp_ratio":   round(amp_ratio, 2),
            "n_pass":      sum(criteria.values()),
            "criteria":    criteria,
        })

    candidates.sort(key=lambda x: x["pearson_r"], reverse=True)
    return {
        "rec":        rec_name,
        "group_uids": group_uid_strs,
        "candidates": candidates[:n],
        "thresholds": {
            "distance_um":     MERGE_DISTANCE_THRESH_UM,
            "pearson_r":       MERGE_PEARSON_THRESH,
            "amplitude_ratio": MERGE_AMPLITUDE_RATIO_THRESH,
        },
    }


# ── Build unit records ────────────────────────────────────────────────────────
def _fmt(val, decimals=3, suffix="") -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _get_primary_channel(rec_name: str, uid: str) -> "int | None":
    """Primary channel index (highest peak-to-peak amplitude) for one unit."""
    sa = _get_analyzer(rec_name)
    if sa is None:
        return None
    templates = _get_templates(sa)
    if templates is None:
        return None
    uid_list = list(sa.unit_ids)
    uid_cast = _cast_uid(uid, uid_list)
    if uid_cast is None:
        return None
    idx = uid_list.index(uid_cast)
    p2p = templates[idx].max(axis=0) - templates[idx].min(axis=0)
    return int(p2p.argmax())


def _get_unit_depth(rec_name: str, ch: "int | None") -> float:
    """Y-coordinate (µm) of the primary channel — used for depth ordering."""
    if ch is None:
        return 0.0
    sa = _get_analyzer(rec_name)
    if sa is None:
        return 0.0
    try:
        return float(sa.get_probe().contact_positions[ch, 1])
    except Exception:
        return 0.0


def build_unit_records(units: list, labels: dict) -> list:
    records = []
    for rec_name, uid, img_path in units:
        m    = load_metrics(rec_name).get(uid, {})
        auto = auto_classify(m)
        existing_label = labels.get(rec_name, {}).get(uid, "")
        reasons = _get_reject_reasons(m) if auto == "Noise" else (
                  _get_sua_failures(m)   if auto == "MUA"   else [])
        ch = _get_primary_channel(rec_name, uid)
        records.append({
            "rec":     rec_name,
            "uid":     uid,
            "img":     base64.b64encode(img_path.read_bytes()).decode("ascii"),
            "auto":    auto,
            "label":   existing_label,
            "reasons": reasons,
            "contact": ch,
            "depth":   _get_unit_depth(rec_name, ch),
            "m": {
                "snr":      _fmt(m.get("snr"),                  2),
                "fr":       _fmt(m.get("firing_rate"),          3, " Hz"),
                "isi":      _fmt(m.get("isi_violations_ratio"), 4),
                "rp_cont":  _fmt(m.get("rp_contamination"),     4),
                "amp_cut":  _fmt(m.get("amplitude_cutoff"),     3),
                "presence": _fmt(m.get("presence_ratio"),       2),
                "bleed":    "YES" if m.get("bleed_flag") else "no",
            },
        })
    return records


# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spike Sorting Review</title>
<style>
  :root {
    --sua-color:      #2e7d32;
    --mua-color:      #e65100;
    --noise-color:    #c62828;
    --unlabeled-color:#546e7a;
    --merge-color:    #1565c0;
    --card-w: 320px;
    --col-a:  #2196F3;
    --col-b:  #FF5722;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; font-size: 13px; background: #f0f2f5;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  #boot-error {
    display: none;
    flex-shrink: 0;
    background: #ffebee;
    color: #b71c1c;
    padding: 10px 14px;
    font-size: 12px;
    font-family: ui-monospace, monospace;
    border-bottom: 2px solid #c62828;
    white-space: pre-wrap;
  }

  /* ── Header ── */
  #header { background: #1a237e; color: white; padding: 8px 16px;
            display: flex; align-items: center; gap: 12px; flex-shrink: 0; flex-wrap: wrap; }
  #header h1 { font-size: 16px; font-weight: 600; }
  #unit-count { font-size: 13px; opacity: 0.85; margin-left: auto; }
  #save-status { font-size: 12px; padding: 3px 8px; border-radius: 4px;
                 transition: background 0.3s; white-space: nowrap; }
  #save-status.saving { background: rgba(255,213,79,0.35); }
  #save-status.saved  { background: rgba(105,240,174,0.2); }
  #save-status.error  { background: rgba(255,82,82,0.35); }
  #save-btn { background: transparent; color: white;
              border: 1px solid rgba(255,255,255,0.5); padding: 5px 12px;
              border-radius: 4px; cursor: pointer; font-size: 13px; flex-shrink: 0; }
  #save-btn:hover { background: rgba(255,255,255,0.12); }

  #view-nav { display: flex; gap: 4px; margin-left: 4px; flex-shrink: 0; }
  .view-tab {
    background: rgba(255,255,255,0.14);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.4);
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
  }
  .view-tab:hover { background: rgba(255,255,255,0.22); }
  .view-tab.active { background: #fff; color: #1a237e; border-color: #fff; }

  /* ── Layout ── */
  #body-wrap {
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    transition: padding-bottom 0.2s;
  }
  body.preview-open #body-wrap { padding-bottom: 330px; }

  /* ── Sidebar ── */
  #sidebar { width: 210px; flex-shrink: 0; background: #fff; border-right: 1px solid #ddd;
             padding: 12px; overflow-y: auto; }
  #sidebar h2 { font-size: 12px; text-transform: uppercase; color: #888; margin: 12px 0 6px;
                letter-spacing: 0.5px; }
  #sidebar h2:first-child { margin-top: 0; }
  .filter-group { display: flex; flex-direction: column; gap: 4px; }
  .filter-item { display: flex; align-items: center; gap: 6px; cursor: pointer;
                 padding: 3px 4px; border-radius: 3px; }
  .filter-item:hover { background: #f5f5f5; }
  .filter-item input { cursor: pointer; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .dot-sua      { background: var(--sua-color); }
  .dot-mua      { background: var(--mua-color); }
  .dot-noise    { background: var(--noise-color); }
  .dot-unlabeled{ background: var(--unlabeled-color); }
  .count-badge { margin-left: auto; font-size: 11px; color: #888; background: #eee;
                 padding: 1px 5px; border-radius: 8px; }
  #rec-filter-list { display: flex; flex-direction: column; gap: 4px;
                     max-height: 200px; overflow-y: auto; }

  /* ── Merge sidebar controls ── */
  .merge-toggle { width: 100%; padding: 5px; cursor: pointer; border-radius: 4px;
                  font-size: 12px; font-weight: 600; border: 1.5px solid var(--merge-color);
                  color: var(--merge-color); background: transparent; }
  .merge-toggle.on { background: var(--merge-color); color: #fff; }
  .merge-toggle:hover { opacity: 0.85; }
  #merge-panel { margin-top: 6px; display: none; }
  #merge-pending-text { font-size: 11px; color: #555; margin-bottom: 4px; }
  .merge-action-btn { width: 100%; padding: 4px; border: none; border-radius: 3px;
                      font-size: 11px; font-weight: 600; cursor: pointer; margin: 2px 0; }
  .merge-action-btn.commit { background: var(--merge-color); color: #fff; }
  .merge-action-btn.commit:disabled { background: #bbb; cursor: default; }
  .merge-action-btn.cancel { background: #eee; color: #555; }
  #merge-selected-thumbs { margin: 6px 0 4px; display: flex; flex-direction: column; gap: 6px; }
  .msel-thumb { background: #f0f4ff; border: 1px solid #c5d0f0; border-radius: 4px;
                overflow: hidden; cursor: pointer; }
  .msel-thumb:hover { border-color: var(--merge-color); }
  .msel-thumb img { width: 100%; display: block; }
  .msel-thumb-info { padding: 3px 5px; font-size: 9px; color: #333; line-height: 1.5; }
  .msel-thumb-title { font-weight: 700; font-size: 10px; margin-bottom: 1px; }
  #merge-groups-list { margin-top: 6px; }
  .merge-group-row { font-size: 10px; color: #444; background: #e8f0fe; border-radius: 3px;
                     padding: 3px 6px; margin: 3px 0; display: flex; align-items: flex-start; gap: 4px; }
  .merge-group-uids { flex: 1; word-break: break-all; }
  .mg-disband { cursor: pointer; color: #c62828; font-weight: 700; flex-shrink: 0; }
  .merge-badge { display: inline-flex; align-items: center; gap: 2px; font-size: 9px;
                 padding: 1px 5px; border-radius: 8px; background: #e3f2fd;
                 color: var(--merge-color); cursor: pointer; flex-shrink: 0; }
  .merge-badge:hover { background: #bbdefb; }

  /* ── Main grid ── */
  #main { flex: 1; min-height: 0; overflow-y: auto; padding: 12px; }
  #grid { display: flex; flex-wrap: wrap; gap: 10px; }

  /* ── Unit card ── */
  .card { width: var(--card-w); background: #fff; border-radius: 6px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.12); overflow: hidden;
          border: 2px solid transparent; transition: border-color 0.15s;
          display: flex; flex-direction: column; }
  .card.merge-sel     { outline: 3px solid var(--merge-color); outline-offset: -1px; }
  .card.merge-mode-on { cursor: pointer; }
  .card.merge-mode-on:hover { outline: 1px dashed #90caf9; }
  .card-header { padding: 6px 8px; font-size: 11px; display: flex;
                 justify-content: space-between; align-items: center;
                 background: #fafafa; border-bottom: 1px solid #eee; gap: 4px; }
  .card-rec { color: #444; font-weight: 600; overflow: hidden; text-overflow: ellipsis;
              white-space: nowrap; max-width: 130px; }
  .card-uid { color: #888; }
  .auto-badge { font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 10px;
                text-transform: uppercase; flex-shrink: 0; }
  .auto-SUA   { background: #e8f5e9; color: var(--sua-color); }
  .auto-MUA   { background: #fff3e0; color: var(--mua-color); }
  .auto-Noise { background: #ffebee; color: var(--noise-color); }
  .card-img-wrap { position: relative; cursor: pointer; background: #111; }
  .card-img-wrap img { width: 100%; display: block; }
  .card-metrics { padding: 5px 8px; font-size: 11px; color: #555;
                  display: grid; grid-template-columns: 1fr 1fr; gap: 2px 8px;
                  border-bottom: 1px solid #eee; }
  .metric-name { color: #888; }
  .metric-val  { font-weight: 600; }
  .metric-val.fail { color: var(--noise-color); }
  .metric-val.warn { color: #f57c00; }
  .reasons { padding: 3px 8px 4px; font-size: 10px; color: #888; font-style: italic; }
  .reasons.noise-reason { color: var(--noise-color); }
  .reasons.mua-reason   { color: var(--mua-color); }

  /* ── Manual label buttons ── */
  .label-btn-row { display: flex; gap: 4px; padding: 5px 8px 6px; align-items: center;
                   border-top: 1px solid #f0f0f0; }
  .label-btn { font-size: 9px; font-weight: 700; padding: 2px 9px; border-radius: 10px;
               cursor: pointer; border: 1.5px solid; text-transform: uppercase; background: #fff; }
  .label-btn.lb-SUA   { border-color: var(--sua-color);   color: var(--sua-color); }
  .label-btn.lb-SUA.active   { background: var(--sua-color);   color: #fff; }
  .label-btn.lb-MUA   { border-color: var(--mua-color);   color: var(--mua-color); }
  .label-btn.lb-MUA.active   { background: var(--mua-color);   color: #fff; }
  .label-btn.lb-Noise { border-color: var(--noise-color); color: var(--noise-color); }
  .label-btn.lb-Noise.active { background: var(--noise-color); color: #fff; }
  .label-clear-btn { font-size: 10px; color: #bbb; background: transparent; border: none;
                     cursor: pointer; margin-left: auto; padding: 1px 3px; line-height: 1; }
  .label-clear-btn:hover { color: #666; }
  .auto-badge.manual { box-shadow: 0 0 0 1.5px #888; }

  /* ── Merge section header ── */
  .merge-section-header {
    font-size: 11px; font-weight: 700; color: var(--merge-color);
    padding: 12px 4px 5px; text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 2px solid #c5cae9; margin-bottom: 8px;
  }

  /* ── Lightbox ── */
  #lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85);
              z-index: 1000; align-items: center; justify-content: center; }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 95vw; max-height: 95vh; border-radius: 4px; }
  #lightbox-close { position: absolute; top: 12px; right: 20px; font-size: 32px;
                    color: #fff; cursor: pointer; line-height: 1; }

  /* ── Empty state ── */
  #empty-msg { display: none; padding: 40px; color: #aaa; font-size: 15px; }

  /* ── Merge Preview panel ── */
  #merge-preview {
    position: fixed; bottom: 0; left: 0; right: 0;
    height: 0; background: #fff;
    border-top: 2px solid var(--merge-color);
    z-index: 200; display: flex; flex-direction: column;
    overflow: hidden; transition: height 0.2s ease;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.13);
  }
  #merge-preview.open { height: 330px; }

  #mp-header {
    background: #e8f0fe; padding: 5px 12px; flex-shrink: 0;
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid #c5cae9;
  }
  #mp-title   { font-weight: 700; font-size: 13px; color: var(--merge-color); }
  #mp-verdict { flex: 1; font-size: 12px; }
  #mp-merge-btn { background: var(--merge-color); color: #fff; border: none;
                  padding: 4px 14px; border-radius: 4px; cursor: pointer;
                  font-weight: 600; font-size: 12px; }
  #mp-merge-btn:hover { opacity: 0.85; }
  #mp-close-btn { background: transparent; border: none; font-size: 20px;
                  cursor: pointer; color: #666; line-height: 1; padding: 0 2px; }

  #mp-body { display: flex; flex: 1; overflow: hidden; }

  #mp-loading { flex: 1; display: flex; align-items: center; justify-content: center;
                color: #888; font-size: 13px; font-style: italic; }

  #mp-content { display: none; flex: 1; overflow: hidden; }

  /* Waveform column */
  #mp-waveforms { width: 280px; flex-shrink: 0; padding: 6px 8px;
                  border-right: 1px solid #eee;
                  display: flex; flex-direction: column; gap: 4px; }
  #mp-wv-canvas { flex: 1; width: 100%; display: block; }
  #mp-wv-legend { display: flex; gap: 14px; font-size: 11px; flex-shrink: 0; }
  .mp-dot { display: inline-block; width: 16px; height: 3px; border-radius: 2px;
            vertical-align: middle; margin-right: 3px; }

  /* Correlogram column */
  #mp-correlos { flex: 0 0 160px; min-width: 0; padding: 6px 8px;
                 border-right: 1px solid #eee; overflow-y: auto;
                 display: flex; flex-direction: column; gap: 4px; }
  #mp-correlos canvas { flex-shrink: 0; width: 100%; display: block; }

  /* Metrics column */
  #mp-metrics { width: 250px; flex-shrink: 0; padding: 10px 12px; overflow-y: auto; }
  #mp-metrics h3 { font-size: 11px; text-transform: uppercase; color: #888;
                   letter-spacing: 0.5px; margin-bottom: 8px; }
  #mp-metrics-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  #mp-metrics-table tr { border-bottom: 1px solid #f0f0f0; }
  #mp-metrics-table td { padding: 4px 4px; vertical-align: middle; }
  .mname { color: #666; }
  .mval  { font-weight: 600; font-family: monospace; }
  .mpass { color: var(--sua-color); }
  .mfail { color: var(--noise-color); }
  .minfo { color: #444; }
  .mthr  { color: #bbb; font-size: 10px; font-weight: normal; }

  /* ── Merged pairs full-page view ── */
  #merge-pairs-page {
    display: none;
    flex: 1;
    min-height: 0;
    flex-direction: column;
    overflow: hidden;
    background: #eceff1;
  }
  #merge-pairs-page.open { display: flex; }
  #merge-pairs-toolbar {
    flex-shrink: 0;
    padding: 8px 14px;
    background: #fff;
    border-bottom: 1px solid #ddd;
    font-size: 12px;
    color: #555;
  }
  #merge-pairs-scroll { flex: 1; min-height: 0; overflow-y: auto; padding: 12px; }
  #merge-pairs-empty { display: none; padding: 40px 20px; text-align: center; color: #888; font-size: 14px; }
  .mpair-card {
    background: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    margin-bottom: 14px;
    border: 1px solid #c5cae9;
    overflow: hidden;
  }
  .mpair-head {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    padding: 8px 12px;
    background: #e8f0fe;
    border-bottom: 1px solid #c5cae9;
    font-weight: 600;
    font-size: 12px;
    color: var(--merge-color);
  }
  .mpair-rec {
    font-weight: 500;
    color: #444;
    max-width: 42%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .mpair-verdict { margin-left: auto; font-size: 11px; }
  .mpair-body { position: relative; min-height: 100px; }
  .mpair-loading, .mpair-static-hint {
    padding: 20px;
    text-align: center;
    color: #888;
    font-style: italic;
    font-size: 12px;
  }
  .mpair-static-hint { font-style: normal; color: #666; }
  .mpair-err {
    display: none;
    padding: 16px;
    text-align: center;
    color: #c62828;
    font-size: 12px;
  }
  .mpair-inner {
    display: none;
    flex-direction: row;
    align-items: stretch;
    min-height: 300px;
  }
  .mpair-wv {
    width: 360px;
    flex-shrink: 0;
    padding: 6px;
    border-right: 1px solid #eee;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    max-height: 300px;
  }
  .mpair-wv canvas { flex-shrink: 0; min-height: 460px; width: 100%; display: block; }
  .mpair-legend { display: flex; gap: 10px; font-size: 10px; flex-shrink: 0; margin-top: 4px; }
  .mpair-corr {
    flex: 1;
    min-width: 140px;
    max-width: 160px;
    padding: 6px;
    border-right: 1px solid #eee;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .mpair-corr canvas { flex: 1; min-height: 72px; width: 100%; display: block; }
  .mpair-met {
    width: 220px;
    flex-shrink: 0;
    padding: 8px 10px;
    overflow-y: auto;
    font-size: 11px;
  }
  .mpair-met h4 {
    font-size: 10px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
    letter-spacing: 0.4px;
  }
  .mpair-met table { width: 100%; border-collapse: collapse; }

  /* ── Auto-merge group cards ── */
  .mgroup-inner {
    display: flex; flex-direction: row; align-items: stretch;
  }
  .mgroup-wv {
    width: 380px; flex-shrink: 0; padding: 6px;
    border-right: 1px solid #eee;
    display: flex; flex-direction: column;
  }
  .mgroup-wv canvas { flex: 1; min-height: 220px; width: 100%; display: block; }
  .mgroup-legend { display: flex; gap: 8px; flex-wrap: wrap; font-size: 10px; flex-shrink: 0; margin-top: 4px; }
  .mgroup-corr {
    width: 160px; flex-shrink: 0; max-height: 400px; overflow-y: auto;
    padding: 4px 6px; border-right: 1px solid #eee;
    display: flex; flex-direction: column; gap: 2px;
  }
  .mgroup-corr canvas { width: 100%; display: block; flex-shrink: 0; }
  .mgroup-corr-sep { font-size: 9px; font-weight: 700; color: #90a4ae;
                     text-transform: uppercase; letter-spacing: 0.4px;
                     margin: 4px 0 1px; padding-left: 2px; flex-shrink: 0; }
  .mgroup-met {
    width: 260px; flex-shrink: 0; padding: 8px 10px;
    overflow-y: auto; max-height: 340px; font-size: 11px;
  }
  .mgroup-met h4 {
    font-size: 10px; text-transform: uppercase; color: #888;
    margin: 6px 0 3px; letter-spacing: 0.4px;
  }
  .mgroup-met h4:first-child { margin-top: 0; }
  .mgroup-met table { width: 100%; border-collapse: collapse; }
  .mgroup-met tr { border-bottom: 1px solid #f0f0f0; }
  .mgroup-met td { padding: 3px 3px; vertical-align: middle; }
  .mgroup-uid-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                    margin-right: 4px; vertical-align: middle; flex-shrink: 0; }
  .mgroup-pair-sep { height: 5px; }

  /* ── Candidate unit list ── */
  .cand-section h4 { font-size: 10px; text-transform: uppercase; color: #888;
                     margin: 8px 0 3px; letter-spacing: 0.4px; }
  .cand-table { width: 100%; border-collapse: collapse; }
  .cand-table tr { border-bottom: 1px solid #f0f0f0; }
  .cand-table td { padding: 2px 3px; vertical-align: middle; font-size: 10px; }
  .cand-score { font-family: monospace; font-weight: 600; color: #1565c0; }
  .cand-preview-btn { background: transparent; border: 1px solid #90a4ae;
    border-radius: 3px; padding: 1px 5px; cursor: pointer;
    font-size: 9px; color: #455a64; }
  .cand-preview-btn:hover { background: #eceff1; }
  /* ── Unmerge button — card header ── */
  .unmerge-auto-btn { background: transparent; border: 1px solid #c62828;
    color: #c62828; border-radius: 3px; padding: 2px 7px;
    font-size: 10px; font-weight: 600; cursor: pointer; flex-shrink: 0; }
  .unmerge-auto-btn:hover { background: #ffebee; }
</style>
</head>
<body>
<div id="boot-error"></div>

<!-- Header -->
<div id="header">
  <h1>Spike Sorting Review</h1>
  <nav id="view-nav" aria-label="Page">
    <button type="button" class="view-tab active" data-view="units">Units</button>
    <button type="button" class="view-tab" data-view="merged-pairs">Merge</button>
  </nav>
  <span id="unit-count" style="font-size:13px;opacity:0.85;"></span>
  <span id="save-status" class="saved">Saved &#10003;</span>
  <button id="save-btn" onclick="saveNow()">Save</button>
</div>

<!-- Body -->
<div id="body-wrap">
  <!-- Sidebar -->
  <div id="sidebar">
    <h2>Auto-class Filter</h2>
    <div class="filter-group">
      <label class="filter-item"><input type="checkbox" class="af" value="SUA" checked>
        <span class="dot dot-sua"></span>Auto SUA<span class="count-badge" id="cnt-SUA">0</span></label>
      <label class="filter-item"><input type="checkbox" class="af" value="MUA" checked>
        <span class="dot dot-mua"></span>Auto MUA<span class="count-badge" id="cnt-MUA">0</span></label>
      <label class="filter-item"><input type="checkbox" class="af" value="Noise" checked>
        <span class="dot dot-noise"></span>Auto Noise<span class="count-badge" id="cnt-Noise">0</span></label>
    </div>

    <h2>Recording</h2>
    <div id="rec-filter-list"></div>

    <h2 style="margin-top:16px;">Merge</h2>
    <button class="merge-toggle" id="merge-toggle-btn" onclick="toggleMergeMode()">Merge Mode: OFF</button>
    <div id="merge-panel">
      <div id="merge-pending-text">0 units selected</div>
      <div id="merge-selected-thumbs"></div>
      <button class="merge-action-btn commit" id="commit-btn" onclick="commitMergeFromPanel()" disabled>
        Create Merge Group</button>
      <button class="merge-action-btn cancel" onclick="clearMergeSelect()">Cancel Selection</button>
    </div>
    <div id="merge-groups-list"></div>

    <h2 style="margin-top:16px;">Keyboard (focused card)</h2>
    <div style="font-size:11px;color:#666;line-height:1.8;">
      <b>&#8592; &#8594;</b> = prev / next card<br>
      <b>Space</b> = select for merge<br>
      <b>Esc</b> = close lightbox / preview
    </div>
  </div>

  <!-- Grid -->
  <div id="main">
    <div id="grid"></div>
    <div id="empty-msg">No units match current filters.</div>
  </div>
</div>

<!-- Merged pairs (all pairs within each merge group) -->
<div id="merge-pairs-page" aria-label="Merged pairs">
  <div id="merge-pairs-toolbar"></div>
  <div id="merge-pairs-scroll"></div>
  <div id="merge-pairs-empty">No pairs found. Auto-merged pairs appear here after running the merge pass. Turn on Merge Mode on the Units tab to create potential pairs.</div>
</div>

<!-- Lightbox -->
<div id="lightbox" onclick="closeLightbox()">
  <span id="lightbox-close">&times;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<!-- Merge Preview Panel -->
<div id="merge-preview">
  <div id="mp-header">
    <span id="mp-title">Merge Preview</span>
    <span id="mp-verdict"></span>
    <button id="mp-merge-btn" onclick="commitMergeFromPreview()">Merge</button>
    <button id="mp-close-btn" onclick="closeMergePreview()">&times;</button>
  </div>
  <div id="mp-body">
    <div id="mp-loading">Loading preview&hellip;</div>
    <div id="mp-content">
      <div id="mp-waveforms">
        <canvas id="mp-wv-canvas"></canvas>
        <div id="mp-wv-legend"></div>
      </div>
      <div id="mp-correlos">
        <canvas id="mp-acg-a"></canvas>
        <canvas id="mp-acg-b"></canvas>
        <canvas id="mp-ccg"></canvas>
      </div>
      <div id="mp-metrics">
        <h3>Merge Criteria</h3>
        <table id="mp-metrics-table"></table>
      </div>
    </div>
  </div>
</div>

<script>
window.addEventListener('error', function(ev) {
  const el = document.getElementById('boot-error');
  if (!el || !ev.message) return;
  el.style.display = 'block';
  el.textContent = (ev.message || 'Script error')
    + (ev.filename ? (String.fromCharCode(10) + ev.filename + ':' + (ev.lineno || '?')) : '');
});

// ── Injected constants ─────────────────────────────────────────────────────────
const T = __THRESHOLDS_JSON__;
const UNITS = Array.isArray(__UNITS_JSON__) ? __UNITS_JSON__ : [];

function normalizeInitMerges(raw) {
  const out = {};
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
  for (const [rec, groups] of Object.entries(raw)) {
    if (!Array.isArray(groups)) continue;
    const rows = [];
    if (groups.length >= 2 && !groups.some(Array.isArray)) {
      rows.push(groups.map(x => String(x)));
    } else {
      for (const g of groups) {
        let arr = [];
        if (Array.isArray(g)) arr = g.map(x => String(x));
        else if (g != null && g !== '') arr = [String(g)];
        if (arr.length >= 2) rows.push(arr);
      }
    }
    if (rows.length) out[String(rec)] = rows;
  }
  return out;
}
// AUTO_MERGES: pairs found by run_merge_pass (read-only, shown in "Auto merged" section)
const AUTO_MERGES = normalizeInitMerges(__AUTO_MERGES_JSON__);
// INIT_MERGES: user-defined potential pairs loaded from disk
const INIT_MERGES = normalizeInitMerges(__MERGES_JSON__);

// ── State ──────────────────────────────────────────────────────────────────────
const mergeGroups = {};          // rec → [[uid1, uid2, ...], ...]  (user-defined potential pairs)
let mergeMode     = false;
const mergeSelected = [];        // [{rec, uid}, ...]
let focusIdx = -1;
let _previewUnits = null;        // {rec, uid_a, uid_b} currently shown
let activeView    = 'units';     // 'units' | 'merged-pairs'
let _mergePairsIO = null;

const SERVER_MODE = window.location.protocol === 'http:' || window.location.protocol === 'https:';

// ── Init ───────────────────────────────────────────────────────────────────────
// Populate user merge groups from disk (potential pairs)
for (const [rec, groups] of Object.entries(INIT_MERGES))
  mergeGroups[rec] = groups.map(g => [...g]);

// ── Sidebar recording filter ───────────────────────────────────────────────────
function buildRecFilter() {
  const recs = [...new Set(UNITS.map(u => u.rec))].sort();
  const container = document.getElementById('rec-filter-list');
  recs.forEach(rec => {
    const lbl = document.createElement('label');
    lbl.className = 'filter-item';
    lbl.innerHTML = `<input type="checkbox" class="rf" value="${rec}" checked>
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;">${rec}</span>`;
    container.appendChild(lbl);
  });
  container.querySelectorAll('.rf').forEach(cb => cb.addEventListener('change', renderGrid));
}

function activeFilters(cls) {
  return [...document.querySelectorAll('.' + cls + ':checked')].map(c => c.value);
}

// ── Render grid ────────────────────────────────────────────────────────────────
let visibleCards = [];

function renderGrid() {
  const autoShow  = new Set(activeFilters('af'));
  const recShow   = new Set(activeFilters('rf'));
  const grid      = document.getElementById('grid');
  grid.innerHTML  = '';
  visibleCards    = [];
  focusIdx        = -1;

  const sorted = [...UNITS].sort((a, b) => {
    if (a.rec < b.rec) return -1;
    if (a.rec > b.rec) return 1;
    const ca = (a.contact != null) ? a.contact : Infinity;
    const cb = (b.contact != null) ? b.contact : Infinity;
    if (ca !== cb) return ca - cb;
    return parseInt(a.uid) - parseInt(b.uid);
  });
  sorted.forEach((u, i) => {
    if (!autoShow.has(u.auto))  return;
    if (!recShow.has(u.rec))    return;
    const card = buildCard(u, i);
    grid.appendChild(card);
    visibleCards.push(card);
  });

  document.getElementById('empty-msg').style.display = visibleCards.length ? 'none' : 'block';
  updateCounts();
  updateMergeGroupsDisplay();
  if (visibleCards.length > 0) setFocus(0);
}

// ── Build a unit card ──────────────────────────────────────────────────────────
function buildCard(u, dataIdx) {
  const isSel      = mergeSelected.some(s => s.rec === u.rec && s.uid === u.uid);
  const gIdx       = getMergeGroupIdx(u.rec, u.uid);
  const modeCls    = mergeMode ? ' merge-mode-on' : '';
  const selCls     = isSel    ? ' merge-sel'     : '';
  const m          = u.m || {};
  const reasons    = Array.isArray(u.reasons) ? u.reasons : [];

  const div = document.createElement('div');
  div.className    = `card${modeCls}${selCls}`;
  div.dataset.rec  = u.rec;
  div.dataset.uid  = u.uid;
  div.dataset.idx  = dataIdx;
  div.tabIndex     = 0;

  const snrF = parseFloat(m.snr),  frF  = parseFloat(m.fr),
        isiF = parseFloat(m.isi),  rpF  = parseFloat(m.rp_cont),
        acF  = parseFloat(m.amp_cut), prF = parseFloat(m.presence);
  const snrCls  = snrF <  T.noise_snr      ? 'fail' : snrF < T.sua_snr  ? 'warn' : '';
  const frCls   = frF  <  T.sua_fr         ? 'warn' : '';
  const isiCls  = isiF >= T.sua_isi        ? 'warn' : '';
  const rpCls   = rpF  >= T.sua_rp         ? 'warn' : '';
  const acCls   = acF  >= T.sua_ampcut     ? 'warn' : '';
  const prCls   = prF  <  T.noise_presence ? 'fail' : '';
  const bleedCls = m.bleed === 'YES'     ? 'fail' : '';

  let reasonHtml = '';
  if (reasons.length) {
    const cls    = u.auto === 'Noise' ? 'noise-reason' : 'mua-reason';
    const prefix = u.auto === 'Noise' ? '&#9940; ' : '&#9888; SUA failed: ';
    reasonHtml = `<div class="reasons ${cls}">${prefix}${reasons.join(' | ')}</div>`;
  }

  const mergeBadge = gIdx >= 0
    ? `<span class="merge-badge" title="Group ${gIdx+1} — click to unmerge"
             onclick="event.stopPropagation();unmergeUnit('${u.rec}','${u.uid}')">
         g${gIdx+1}&nbsp;&times;</span>`
    : '';

  const effectiveLbl = u.label || u.auto;
  const badgeCls = `auto-badge auto-${effectiveLbl}${u.label ? ' manual' : ''}`;
  const badgeTxt = u.label ? `${u.label} ✎` : u.auto;
  const labelBtns = ['SUA','MUA','Noise'].map(lbl =>
    `<button class="label-btn lb-${lbl}${effectiveLbl===lbl?' active':''}"
             onclick="event.stopPropagation();setLabel('${u.rec}','${u.uid}','${lbl}')">${lbl}</button>`
  ).join('');
  const clearBtn = u.label
    ? `<button class="label-clear-btn" title="Clear manual label"
               onclick="event.stopPropagation();setLabel('${u.rec}','${u.uid}','')">&#10005;</button>`
    : '';

  div.innerHTML = `
    <div class="card-header">
      <span class="card-rec" title="${u.rec}">${u.rec}</span>
      <span class="card-uid">u${u.uid}</span>
      <span class="${badgeCls}">${badgeTxt}</span>
      ${mergeBadge}
    </div>
    <div class="card-img-wrap">
      <img src="data:image/png;base64,${u.img}" loading="lazy"
           onclick="openLightbox(this.src)" title="Click to enlarge">
    </div>
       <div class="card-metrics">
      ${mRow('SNR',     m.snr,      snrCls)}
      ${mRow('FR',      m.fr,       frCls)}
      ${mRow('ISI',     m.isi,      isiCls)}
      ${mRow('RPCont',  m.rp_cont,  rpCls)}
      ${mRow('AmpCut',  m.amp_cut,  acCls)}
      ${mRow('Presence',m.presence, prCls)}
      ${mRow('Bleed',   m.bleed,    bleedCls)}
    </div>
    ${reasonHtml}
    <div class="label-btn-row">${labelBtns}${clearBtn}</div>`;

  div.addEventListener('click', e => {
    if (!mergeMode) return;
    if (e.target.closest('button') || e.target.closest('.merge-badge')) return;
    toggleMergeSelect(u.rec, u.uid);
  });
  div.addEventListener('focus', () => {
    const ci = visibleCards.indexOf(div);
    if (ci >= 0) focusIdx = ci;
  });
  return div;
}

function mRow(name, val, cls) {
  return `<span class="metric-name">${name}</span><span class="metric-val ${cls}">${val}</span>`;
}

// ── Manual labelling ───────────────────────────────────────────────────────────
function setLabel(rec, uid, lbl) {
  const u = UNITS.find(x => x.rec === rec && x.uid === uid);
  if (!u) return;
  u.label = lbl;
  const effectiveLbl = lbl || u.auto;
  // Update the card in place — no full re-render needed
  const card = document.querySelector(`.card[data-rec="${CSS.escape(rec)}"][data-uid="${CSS.escape(uid)}"]`);
  if (card) {
    const badge = card.querySelector('.auto-badge');
    if (badge) {
      badge.className = `auto-badge auto-${effectiveLbl}${lbl ? ' manual' : ''}`;
      badge.textContent = lbl ? `${lbl} ✎` : u.auto;
    }
    card.querySelectorAll('.label-btn').forEach(btn => {
      btn.classList.toggle('active', btn.textContent.trim() === effectiveLbl);
    });
    const row = card.querySelector('.label-btn-row');
    if (row) {
      let clr = row.querySelector('.label-clear-btn');
      if (lbl && !clr) {
        clr = document.createElement('button');
        clr.className = 'label-clear-btn';
        clr.title = 'Clear manual label';
        clr.innerHTML = '&#10005;';
        clr.onclick = e => { e.stopPropagation(); setLabel(rec, uid, ''); };
        row.appendChild(clr);
      } else if (!lbl && clr) {
        clr.remove();
      }
    }
  }
  autoSave();
}

function _collectLabels() {
  const out = {};
  UNITS.forEach(u => {
    if (u.label) {
      if (!out[u.rec]) out[u.rec] = {};
      out[u.rec][u.uid] = u.label;
    }
  });
  return out;
}

// ── Merge mode ─────────────────────────────────────────────────────────────────
function getMergeGroupIdx(rec, uid) {
  return (mergeGroups[rec] || []).findIndex(g => g.includes(uid));
}

function toggleMergeMode() {
  mergeMode = !mergeMode;
  const btn = document.getElementById('merge-toggle-btn');
  btn.textContent = mergeMode ? 'Merge Mode: ON' : 'Merge Mode: OFF';
  btn.classList.toggle('on', mergeMode);
  document.getElementById('merge-panel').style.display = mergeMode ? 'block' : 'none';
  if (!mergeMode) { clearMergeSelect(); closeMergePreview(); }
  // Toggle cursor class on existing cards without rebuilding the grid
  document.querySelectorAll('#grid .card').forEach(card => {
    card.classList.toggle('merge-mode-on', mergeMode);
  });
}

function toggleMergeSelect(rec, uid) {
  const idx = mergeSelected.findIndex(s => s.rec === rec && s.uid === uid);
  if (idx >= 0) mergeSelected.splice(idx, 1);
  else          mergeSelected.push({ rec, uid });
  updateMergePendingUI();
  // Toggle selection highlight in-place — no full re-render so scroll position is preserved
  const card = document.querySelector(`.card[data-rec="${CSS.escape(rec)}"][data-uid="${CSS.escape(uid)}"]`);
  if (card) card.classList.toggle('merge-sel', mergeSelected.some(s => s.rec === rec && s.uid === uid));
}

function updateMergePendingUI() {
  const n = mergeSelected.length;
  document.getElementById('merge-pending-text').textContent =
    `${n} unit${n !== 1 ? 's' : ''} selected`;
  document.getElementById('commit-btn').disabled = n < 2;

  // Unit summary thumbnails
  const thumbsEl = document.getElementById('merge-selected-thumbs');
  if (n === 0) {
    thumbsEl.innerHTML = '';
  } else {
    thumbsEl.innerHTML = mergeSelected.map(({ rec, uid }) => {
      const u = UNITS.find(x => x.rec === rec && x.uid === uid);
      if (!u) return '';
      const lbl = u.label || u.auto || '';
      const lblCol = lbl === 'SUA' ? '#2e7d32' : lbl === 'Noise' ? '#c62828' : '#e65100';
      const m = u.m || {};
      return `<div class="msel-thumb" onclick="openLightbox('data:image/png;base64,${u.img}')">
        <img src="data:image/png;base64,${u.img}" alt="u${uid}">
        <div class="msel-thumb-info">
          <div class="msel-thumb-title">u${uid}
            <span style="color:${lblCol};margin-left:4px;">${lbl}</span>
          </div>
          SNR ${m.snr || 'n/a'} &nbsp; FR ${m.fr || 'n/a'}<br>
          ISI ${m.isi || 'n/a'} &nbsp; PR ${m.presence || 'n/a'}
        </div>
      </div>`;
    }).join('');
  }

  // Auto-show preview when ≥2 from same recording
  if (SERVER_MODE && n >= 2) {
    const s0 = mergeSelected[mergeSelected.length - 2];
    const s1 = mergeSelected[mergeSelected.length - 1];
    if (s0.rec === s1.rec)
      showMergePreview(s0.rec, s0.uid, s1.uid);
  } else if (n < 2) {
    closeMergePreview();
  }
}

function clearMergeSelect() {
  mergeSelected.length = 0;
  updateMergePendingUI();
}

// Commit from sidebar (when >2 units may be selected)
function commitMergeFromPanel() {
  _doMerge(mergeSelected.map(s => ({ rec: s.rec, uid: s.uid })));
  clearMergeSelect();
}

// Update only the merge-badge inside each card header — no grid rebuild, no scroll reset
function _refreshMergeBadges() {
  document.querySelectorAll('#grid .card').forEach(card => {
    const rec = card.dataset.rec, uid = card.dataset.uid;
    const gIdx = getMergeGroupIdx(rec, uid);
    const header = card.querySelector('.card-header');
    const existing = header.querySelector('.merge-badge');
    if (existing) existing.remove();
    if (gIdx >= 0) {
      const span = document.createElement('span');
      span.className = 'merge-badge';
      span.title = `Group ${gIdx+1} \u2014 click to unmerge`;
      span.innerHTML = `g${gIdx+1}&nbsp;&times;`;
      span.onclick = e => { e.stopPropagation(); unmergeUnit(rec, uid); };
      header.appendChild(span);
    }
  });
}

function _doMerge(pairs) {
  const byRec = {};
  pairs.forEach(({ rec, uid }) => {
    if (!byRec[rec]) byRec[rec] = [];
    byRec[rec].push(uid);
  });
  for (const [rec, uids] of Object.entries(byRec)) {
    if (!mergeGroups[rec]) mergeGroups[rec] = [];
    const hitIdxs = new Set();
    uids.forEach(uid => {
      const i = mergeGroups[rec].findIndex(g => g.includes(uid));
      if (i >= 0) hitIdxs.add(i);
    });
    const newGroup = [...uids];
    [...hitIdxs].sort((a, b) => b - a).forEach(i => {
      mergeGroups[rec][i].forEach(uid => {
        if (!newGroup.includes(uid)) newGroup.push(uid);
      });
      mergeGroups[rec].splice(i, 1);
    });
    mergeGroups[rec].push(newGroup);
  }
  _refreshMergeBadges();
  if (activeView === 'merged-pairs') renderMergePairsPage();
  autoSave();
}

function unmergeUnit(rec, uid) {
  const groups = mergeGroups[rec];
  if (!groups) return;
  const idx = groups.findIndex(g => g.includes(uid));
  if (idx < 0) return;
  if (groups[idx].length <= 2) groups.splice(idx, 1);
  else groups[idx] = groups[idx].filter(u => u !== uid);
  if (groups.length === 0) delete mergeGroups[rec];
  _refreshMergeBadges();
  if (activeView === 'merged-pairs') renderMergePairsPage();
  autoSave();
}

function disbandGroup(rec, canonical) {
  if (!mergeGroups[rec]) return;
  const idx = mergeGroups[rec].findIndex(g => g[0] === canonical);
  if (idx >= 0) mergeGroups[rec].splice(idx, 1);
  if (!mergeGroups[rec].length) delete mergeGroups[rec];
  _refreshMergeBadges();
  if (activeView === 'merged-pairs') renderMergePairsPage();
  autoSave();
}

function unmergeAutoGroup(rec, uids) {
  if (!SERVER_MODE) return;
  // 1. Remove from in-memory AUTO_MERGES
  const groups = AUTO_MERGES[rec];
  if (groups) {
    const idx = groups.findIndex(g =>
      g.length === uids.length && uids.every(u => g.includes(u)));
    if (idx >= 0) groups.splice(idx, 1);
    if (groups.length === 0) delete AUTO_MERGES[rec];
  }
  // 2. Remove card from DOM
  const scroll = document.getElementById('merge-pairs-scroll');
  if (scroll) {
    const card = [...scroll.querySelectorAll('.mpair-card[data-uids]')]
      .find(c => c.dataset.rec === rec && c.dataset.uids === uids.join(','));
    if (card) card.remove();
  }
  // 3. Update section header count
  const autoCount = Object.values(AUTO_MERGES).reduce((s, gs) => s + gs.length, 0);
  const hdr = [...document.querySelectorAll('#merge-pairs-scroll .merge-section-header')]
    .find(h => h.textContent.startsWith('Auto merged'));
  if (hdr) {
    if (autoCount === 0) hdr.remove();
    else hdr.textContent = `Auto merged  (${autoCount} group${autoCount !== 1 ? 's' : ''})`;
  }
  // 4. Persist to disk (optimistic update)
  fetch('/api/unmerge-auto', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rec, uids }),
  }).catch(err => console.error('unmerge-auto failed:', err));
}

function updateMergeGroupsDisplay() {
  const el = document.getElementById('merge-groups-list');
  let html = '';
  for (const [rec, groups] of Object.entries(mergeGroups)) {
    groups.forEach(group => {
      html += `<div class="merge-group-row">
        <span class="merge-group-uids" title="${rec}">
          ${rec.slice(-10)}: [${group.map(u => 'u' + u).join(', ')}]</span>
        <span class="mg-disband" onclick="disbandGroup('${rec}','${group[0]}')"
              title="Disband">&times;</span>
      </div>`;
    });
  }
  el.innerHTML = html;
}

// ── Keyboard navigation ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (document.getElementById('lightbox').classList.contains('open')) {
    if (e.key === 'Escape') closeLightbox();
    return;
  }
  if (e.key === 'Escape') { closeMergePreview(); return; }
  if (focusIdx < 0 || focusIdx >= visibleCards.length) return;
  const card = visibleCards[focusIdx];
  const rec = card.dataset.rec, uid = card.dataset.uid;
  if (e.key === ' ' && mergeMode) { toggleMergeSelect(rec, uid); e.preventDefault(); }
  if (e.key === 'ArrowRight') { setFocus(Math.min(focusIdx + 1, visibleCards.length - 1)); e.preventDefault(); }
  if (e.key === 'ArrowLeft')  { setFocus(Math.max(focusIdx - 1, 0));                       e.preventDefault(); }
});

function setFocus(idx) {
  focusIdx = idx;
  if (visibleCards[idx]) {
    visibleCards[idx].focus({ preventScroll: false });
    visibleCards[idx].scrollIntoView({ block: 'nearest' });
  }
}

// ── Counts ─────────────────────────────────────────────────────────────────────
function updateCounts() {
  const counts = { SUA: 0, MUA: 0, Noise: 0 };
  UNITS.forEach(u => { counts[u.auto] = (counts[u.auto] || 0) + 1; });
  ['SUA', 'MUA', 'Noise'].forEach(k => {
    const el = document.getElementById('cnt-' + k);
    if (el) el.textContent = counts[k] || 0;
  });
  const total = UNITS.length;
  const el = document.getElementById('unit-count');
  if (el) el.textContent = `${total} units`;
}

// ── Lightbox ───────────────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}

// ── Save / auto-save ───────────────────────────────────────────────────────────
let _saveTimer = null;

function setSaveStatus(cls, text) {
  const el = document.getElementById('save-status');
  el.className = cls;
  el.textContent = text;
}
function autoSave() {
  if (!SERVER_MODE) return;
  if (_saveTimer) clearTimeout(_saveTimer);
  setSaveStatus('saving', 'Saving\u2026');
  _saveTimer = setTimeout(saveNow, 800);
}
function saveNow() {
  if (_saveTimer) { clearTimeout(_saveTimer); _saveTimer = null; }
  if (SERVER_MODE) {
    setSaveStatus('saving', 'Saving\u2026');
    fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ merges: mergeGroups, labels: _collectLabels() }),
    })
      .then(r => r.json())
      .then(() => setSaveStatus('saved', 'Saved \u2713'))
      .catch(() => setSaveStatus('error', 'Error \u2717'));
  } else {
    const blob = new Blob([JSON.stringify(mergeGroups, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'unit_user_merges.json';
    a.click();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── Merge Preview panel ────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

function showMergePreview(rec, uid_a, uid_b) {
  if (!SERVER_MODE) return;
  _previewUnits = { rec, uid_a, uid_b };

  const panel = document.getElementById('merge-preview');
  panel.classList.add('open');
  document.body.classList.add('preview-open');

  document.getElementById('mp-title').textContent = `Merge Preview \u2014 u${uid_a} \u2194 u${uid_b}`;
  document.getElementById('mp-verdict').textContent = '';
  document.getElementById('mp-loading').style.display = 'flex';
  document.getElementById('mp-content').style.display = 'none';

  fetch(`/api/merge-preview?rec=${encodeURIComponent(rec)}&uid_a=${uid_a}&uid_b=${uid_b}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        document.getElementById('mp-loading').textContent = 'Error: ' + data.error;
        return;
      }
      _renderMergePreview(data);
    })
    .catch(() => {
      document.getElementById('mp-loading').textContent = 'Failed to load preview.';
    });
}

function closeMergePreview() {
  document.getElementById('merge-preview').classList.remove('open');
  document.body.classList.remove('preview-open');
  _previewUnits = null;
}

function commitMergeFromPreview() {
  if (!_previewUnits) return;
  const { rec, uid_a, uid_b } = _previewUnits;
  _doMerge([{ rec, uid: uid_a }, { rec, uid: uid_b }]);
  // Remove these two from mergeSelected
  [uid_a, uid_b].forEach(uid => {
    const idx = mergeSelected.findIndex(s => s.rec === rec && s.uid === uid);
    if (idx >= 0) mergeSelected.splice(idx, 1);
  });
  updateMergePendingUI();
  closeMergePreview();
}

// ── Render preview content (bottom panel + merged-pairs cards) ─────────────────
function _mergePreviewMetricsRowsHTML(data) {
  const m = data.metrics, c = data.criteria, thr = data.thresholds;
  const rows = [
    ['Distance',    m.distance_um.toFixed(1) + ' \u00b5m', c.distance,
      `< ${thr.distance_um}\u00b5m`],
    ['Pearson R',   m.pearson_r.toFixed(3),                c.pearson_r,
      `> ${thr.pearson_r}`],
    ['Amp ratio',   m.amplitude_ratio.toFixed(2),          c.amplitude_ratio,
      `< ${thr.amplitude_ratio}`],
    ['ISI merged',  (m.isi_merged * 100).toFixed(2) + '%', c.isi_merged,
      `< ${(thr.isi_merged * 100).toFixed(0)}%`],
    null,
    ['ACG sim',     m.acg_similarity.toFixed(3),           null, ''],
    ['Spikes A',    m.n_spikes_a.toLocaleString(),         null, ''],
    ['Spikes B',    m.n_spikes_b.toLocaleString(),         null, ''],
  ];
  return rows.map(row => {
    if (!row) return '<tr><td colspan="2" style="height:6px"></td></tr>';
    const [name, val, pass, threshold] = row;
    const valCls = pass === null ? 'minfo' : (pass ? 'mpass' : 'mfail');
    const icon   = pass === null ? '' : (pass ? ' \u2713' : ' \u2717');
    const thrSpan = threshold
      ? `<span class="mthr"> (${threshold})</span>`
      : '';
    return `<tr>
      <td class="mname">${name}</td>
      <td class="mval ${valCls}">${val}${icon}${thrSpan}</td>
    </tr>`;
  }).join('');
}

function _setMergeVerdictHtml(el, data) {
  if (!el) return;
  const n = data.n_pass;
  if (n === 4)
    el.innerHTML = '<span style="color:#2e7d32;font-weight:700;">&#10003; All 4 criteria met</span>';
  else
    el.innerHTML = `<span style="color:#c62828;font-weight:700;">${n}/4 criteria met</span>`;
}

function applyMergePreviewToTargets(data, tgt) {
  _setMergeVerdictHtml(tgt.verdictEl, data);

  const wvCanvas = tgt.wvCanvas;
  const pw = wvCanvas.parentElement;
  const cOff = tgt.compact ? 12 : 16;
  const hOff = tgt.compact ? 24 : 32;
  wvCanvas.width  = Math.max(100, (pw ? pw.clientWidth : 360) - cOff);
  wvCanvas.height = tgt.compact ? 460 : Math.max(72, (pw ? pw.clientHeight : 140) - hOff);
  _drawProbeWaveforms(wvCanvas, data);

  tgt.legendEl.innerHTML =
    `<span><span class="mp-dot" style="background:var(--col-a)"></span>u${data.uid_a}</span>` +
    `<span><span class="mp-dot" style="background:var(--col-b)"></span>u${data.uid_b}</span>`;

  const cH = tgt.compact ? 75 : 130;
  _drawCorrelogram(tgt.acgA, data.acg_a.bins, data.acg_a.counts,
    '#2196F3', `ACG  u${data.uid_a}  (${data.acg_a.n.toLocaleString()} spk)`, true, cH);
  _drawCorrelogram(tgt.acgB, data.acg_b.bins, data.acg_b.counts,
    '#FF5722', `ACG  u${data.uid_b}  (${data.acg_b.n.toLocaleString()} spk)`, true, cH);
  _drawCorrelogram(tgt.ccg, data.ccg.bins, data.ccg.counts,
    '#9C27B0', `CCG  u${data.uid_a} \u2192 u${data.uid_b}`, false, cH);

  tgt.metricsTable.innerHTML = _mergePreviewMetricsRowsHTML(data);
}

function _renderMergePreview(data) {
  document.getElementById('mp-loading').style.display  = 'none';
  document.getElementById('mp-content').style.display  = 'flex';
  applyMergePreviewToTargets(data, {
    verdictEl: document.getElementById('mp-verdict'),
    wvCanvas: document.getElementById('mp-wv-canvas'),
    legendEl: document.getElementById('mp-wv-legend'),
    acgA: document.getElementById('mp-acg-a'),
    acgB: document.getElementById('mp-acg-b'),
    ccg: document.getElementById('mp-ccg'),
    metricsTable: document.getElementById('mp-metrics-table'),
    compact: false,
  });
}

// ── Canvas: probe-layout waveform overlay ──────────────────────────────────────
function _drawProbeWaveforms(canvas, data) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  const { channels, time_ms, uid_a, uid_b } = data;
  if (!channels || channels.length === 0) return;

  const COL_A = '#2196F3', COL_B = '#FF5722';

  // Bounding box of channel positions
  const xs = channels.map(c => c.x), ys = channels.map(c => c.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;

  // Global max amplitude for consistent scaling
  const maxAmp = Math.max(
    ...channels.flatMap(c => [...c.wv_a, ...c.wv_b].map(Math.abs)), 1e-9
  );

  // Minimum inter-channel distance in µm (determines waveform display size)
  let minDist = Infinity;
  for (let i = 0; i < channels.length; i++)
    for (let j = i + 1; j < channels.length; j++) {
      const d = Math.hypot(channels[i].x - channels[j].x, channels[i].y - channels[j].y);
      if (d > 0) minDist = Math.min(minDist, d);
    }
  if (!isFinite(minDist)) minDist = 25;

  // Canvas mapping — leave padding for waveforms extending outside channel bbox
  const PAD = 28;
  const plotW = W - 2 * PAD, plotH = H - 2 * PAD;
  // Uniform scale; waveform display occupies ~50% of inter-channel gap
  const sc = Math.min(
    xRange > 0 ? plotW * 0.55 / xRange : 9999,
    yRange > 0 ? plotH * 0.85 / yRange : 9999
  );

  const toX = x => PAD + plotW / 2 + (x - (xMin + xMax) / 2) * sc;
  const toY = y => PAD + plotH / 2 - (y - (yMin + yMax) / 2) * sc; // y-flip

  const wvW = minDist * sc * 0.90;  // waveform time-axis width in px
  const wvH = minDist * sc * 0.75;  // waveform amplitude half-height in px
  const n   = time_ms.length;

  channels.forEach(ch => {
    const cx = toX(ch.x), cy = toY(ch.y);

    // Channel dot — primary channel gets unit colour
    const dotCol = ch.pri_a ? COL_A : (ch.pri_b ? COL_B : '#bbb');
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = dotCol;
    ctx.fill();

    // Draw waveforms
    [[ch.wv_a, COL_A], [ch.wv_b, COL_B]].forEach(([wv, col]) => {
      ctx.beginPath();
      ctx.strokeStyle = col;
      ctx.lineWidth   = 2.5;
      ctx.globalAlpha = 0.88;
      for (let i = 0; i < n; i++) {
        const px = cx + (i / (n - 1) - 0.5) * wvW;
        const py = cy - (wv[i] / maxAmp) * wvH;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    // Channel label
    ctx.fillStyle = '#888';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('ch' + ch.ch, cx - wvW / 2 - 4, cy + 4);
  });

  // Legend (top-left)
  [[COL_A, uid_a], [COL_B, uid_b]].forEach(([col, uid], i) => {
    const lx = 6, ly = 11 + i * 14;
    ctx.beginPath();
    ctx.strokeStyle = col; ctx.lineWidth = 2.5;
    ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.fillStyle = col;
    ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('u' + uid, lx + 22, ly + 3);
  });
}

// ── Canvas: correlogram bar chart ──────────────────────────────────────────────
function _drawCorrelogram(canvasRef, bins, counts, fillColor, title, isAcg, fixedH) {
  const canvas = typeof canvasRef === 'string' ? document.getElementById(canvasRef) : canvasRef;
  if (!canvas) return;
  const par = canvas.parentElement;
  const pw = (par && par.clientWidth > 8) ? par.clientWidth - 16 : 260;
  canvas.width  = pw;
  canvas.height = fixedH || 75;

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  const pad = { t: 15, r: 4, b: 17, l: 30 };
  const plotW = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const maxC = Math.max(...counts, 1);
  const n = bins.length, bw = plotW / n;

  // Bars
  counts.forEach((c, i) => {
    const zeroBin = isAcg && Math.abs(bins[i]) < 1.5;
    ctx.fillStyle = zeroBin ? 'rgba(150,150,150,0.35)' : (fillColor + 'cc');
    const bh = c / maxC * ph;
    ctx.fillRect(pad.l + i * bw, pad.t + ph - bh, Math.max(bw - 0.5, 1), bh);
  });

  // Zero-lag dashed line
  const zIdx = bins.findIndex(b => b >= 0);
  if (zIdx >= 0) {
    const zx = pad.l + zIdx * bw;
    ctx.strokeStyle = 'rgba(0,0,0,0.22)'; ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(zx, pad.t); ctx.lineTo(zx, pad.t + ph); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Axis border
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph);
  ctx.lineTo(pad.l + plotW, pad.t + ph);
  ctx.stroke();

  // X ticks: −window, 0, +window
  ctx.fillStyle = '#888'; ctx.font = '8px sans-serif'; ctx.textAlign = 'center';
  const winMs = Math.round(Math.abs(bins[0]));
  [[-winMs, 0], [0, zIdx >= 0 ? zIdx : Math.floor(n/2)], [winMs, n - 1]].forEach(([label, i]) => {
    ctx.fillText(label + 'ms', pad.l + i * bw, pad.t + ph + 12);
  });

  // Y max
  ctx.textAlign = 'right';
  ctx.fillText(maxC, pad.l - 2, pad.t + 8);

  // Title
  ctx.fillStyle = fillColor; ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(title, pad.l, pad.t - 3);
}

// ===============================================================================
// ── Merged pairs page (all pairs within each merge group) ──────────────────────
// ===============================================================================

function _uidNum(u) {
  const n = parseInt(u, 10);
  return Number.isFinite(n) ? n : 0;
}

function _groupsToPairs(groupsObj) {
  const out = [];
  for (const [rec, groups] of Object.entries(groupsObj)) {
    for (const group of groups) {
      if (!group || group.length < 2) continue;
      const g = [...group].sort((a, b) => _uidNum(a) - _uidNum(b) || String(a).localeCompare(String(b)));
      for (let i = 0; i < g.length; i++)
        for (let j = i + 1; j < g.length; j++)
          out.push({ rec, uid_a: g[i], uid_b: g[j] });
    }
  }
  out.sort((a, b) => `${a.rec}|${a.uid_a}|${a.uid_b}`.localeCompare(`${b.rec}|${b.uid_a}|${b.uid_b}`));
  return out;
}

function allAutoMergedPairs()   { return _groupsToPairs(AUTO_MERGES); }
function allMergedUnitPairs()   { return _groupsToPairs(mergeGroups); }

function setAppView(view) {
  activeView = view;
  const bw = document.getElementById('body-wrap');
  const mpp = document.getElementById('merge-pairs-page');
  document.querySelectorAll('#view-nav .view-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === view);
  });
  if (view === 'units') {
    bw.style.display = 'flex';
    mpp.classList.remove('open');
    if (_mergePairsIO) { _mergePairsIO.disconnect(); _mergePairsIO = null; }
  } else {
    bw.style.display = 'none';
    mpp.classList.add('open');
    renderMergePairsPage();
  }
}

function _buildMergePairCard(p) {
  const card = document.createElement('div');
  card.className = 'mpair-card';
  card.dataset.rec = p.rec;
  card.dataset.uidA = p.uid_a;
  card.dataset.uidB = p.uid_b;

  if (SERVER_MODE) {
    card.innerHTML = `
      <div class="mpair-head">
        <span class="mpair-pair">u${p.uid_a} \u2194 u${p.uid_b}</span>
        <span class="mpair-rec"></span>
        <span class="mpair-verdict"></span>
      </div>
      <div class="mpair-body">
        <div class="mpair-loading">Loading\u2026</div>
        <div class="mpair-inner" style="display:none">
          <div class="mpair-wv"><canvas></canvas><div class="mpair-legend"></div></div>
          <div class="mpair-corr"><canvas></canvas><canvas></canvas><canvas></canvas></div>
          <div class="mpair-met"><h4>Merge Criteria</h4><table class="mpair-metrics-table"></table></div>
        </div>
        <div class="mpair-err"></div>
      </div>`;
  } else {
    card.innerHTML = `
      <div class="mpair-head">
        <span class="mpair-pair">u${p.uid_a} \u2194 u${p.uid_b}</span>
        <span class="mpair-rec"></span>
        <span class="mpair-verdict"></span>
      </div>
      <div class="mpair-body">
        <div class="mpair-static-hint">Open with <b>python html_review.py --serve</b> to load waveform previews.</div>
      </div>`;
  }
  const recEl = card.querySelector('.mpair-rec');
  recEl.textContent = p.rec;
  recEl.title = p.rec;
  return card;
}

// ── Colour palette for N-unit group overlays ──────────────────────────────────
const GROUP_COLORS = [
  '#2196F3','#FF5722','#4CAF50','#9C27B0',
  '#FF9800','#00BCD4','#E91E63','#795548',
];

// ── Auto-merge group (N units) — canvas waveform overlay ──────────────────────
function _drawProbeWaveformsGroup(canvas, data) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  const { channels, time_ms, uids } = data;
  if (!channels || channels.length === 0) return;

  const xs = channels.map(c => c.x), ys = channels.map(c => c.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;

  const maxAmp = Math.max(
    ...channels.flatMap(c => c.waveforms.flatMap(wv => wv.map(Math.abs))), 1e-9
  );

  let minDist = Infinity;
  for (let i = 0; i < channels.length; i++)
    for (let j = i + 1; j < channels.length; j++) {
      const d = Math.hypot(channels[i].x - channels[j].x, channels[i].y - channels[j].y);
      if (d > 0) minDist = Math.min(minDist, d);
    }
  if (!isFinite(minDist)) minDist = 25;

  const PAD = 28;
  const plotW = W - 2 * PAD, plotH = H - 2 * PAD;
  const sc = Math.min(
    xRange > 0 ? plotW * 0.55 / xRange : 9999,
    yRange > 0 ? plotH * 0.85 / yRange : 9999
  );
  const toX = x => PAD + plotW / 2 + (x - (xMin + xMax) / 2) * sc;
  const toY = y => PAD + plotH / 2 - (y - (yMin + yMax) / 2) * sc;
  const wvW = minDist * sc * 0.90;
  const wvH = minDist * sc * 0.75;
  const n   = time_ms.length;

  channels.forEach(ch => {
    const cx = toX(ch.x), cy = toY(ch.y);
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#bbb';
    ctx.fill();

    ch.waveforms.forEach((wv, k) => {
      ctx.beginPath();
      ctx.strokeStyle  = GROUP_COLORS[k % GROUP_COLORS.length];
      ctx.lineWidth    = 2.5;
      ctx.globalAlpha  = 0.88;
      for (let i = 0; i < n; i++) {
        const px = cx + (i / (n - 1) - 0.5) * wvW;
        const py = cy - (wv[i] / maxAmp) * wvH;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    ctx.fillStyle = '#888'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText('ch' + ch.ch, cx - wvW / 2 - 4, cy + 4);
  });

  uids.forEach((uid, k) => {
    const col = GROUP_COLORS[k % GROUP_COLORS.length];
    const lx = 6, ly = 11 + k * 14;
    ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 2.5;
    ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.fillStyle = col; ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('u' + uid, lx + 22, ly + 3);
  });
}

// ── Render a loaded group preview into a card ──────────────────────────────────
function _renderGroupPreview(card, data) {
  const loading = card.querySelector('.mpair-loading');
  const inner   = card.querySelector('.mgroup-inner');
  if (loading) loading.style.display = 'none';
  inner.style.display = 'flex';

  // ── Verdict in header ──
  const verdictEl = card.querySelector('.mpair-verdict');
  const allPass = data.pair_criteria.every(pc => pc.n_pass === 4);
  const anyPass = data.pair_criteria.some(pc => pc.n_pass === 4);
  if (verdictEl) {
    if (allPass)
      verdictEl.innerHTML = '<span style="color:#2e7d32;font-weight:700;">\u2713 All pairs pass</span>';
    else if (anyPass)
      verdictEl.innerHTML = `<span style="color:#e65100;font-weight:700;">${data.pair_criteria.filter(p=>p.n_pass===4).length}/${data.pair_criteria.length} pairs pass</span>`;
    else
      verdictEl.innerHTML = `<span style="color:#c62828;font-weight:700;">0/${data.pair_criteria.length} pairs pass</span>`;
  }

  // ── Waveform canvas ──
  const wvCanvas = inner.querySelector('.mgroup-wv canvas');
  const pw = wvCanvas.parentElement;
  wvCanvas.width  = Math.max(150, (pw ? pw.clientWidth  : 380) - 12);
  wvCanvas.height = Math.max(200, (pw ? pw.clientHeight : 260) - 28);
  _drawProbeWaveformsGroup(wvCanvas, data);

  // Legend dots
  const legendEl = inner.querySelector('.mgroup-legend');
  legendEl.innerHTML = data.uids.map((uid, k) =>
    `<span><span class="mp-dot" style="background:${GROUP_COLORS[k % GROUP_COLORS.length]}"></span>u${uid}</span>`
  ).join('');

  // ── Correlograms ──
  const corrEl = inner.querySelector('.mgroup-corr');
  corrEl.innerHTML = '';
  const CH = 80;

  const acgSep = document.createElement('div');
  acgSep.className = 'mgroup-corr-sep'; acgSep.textContent = 'ACG';
  corrEl.appendChild(acgSep);
  data.acgs.forEach((acg, k) => {
    const cv = document.createElement('canvas');
    cv.style.height = CH + 'px';
    corrEl.appendChild(cv);
    const col = GROUP_COLORS[k % GROUP_COLORS.length];
    requestAnimationFrame(() =>
      _drawCorrelogram(cv, acg.bins, acg.counts, col,
        `ACG  u${acg.uid}  (${acg.n.toLocaleString()} spk)`, true, CH));
  });

  const ccgSep = document.createElement('div');
  ccgSep.className = 'mgroup-corr-sep'; ccgSep.textContent = 'CCG';
  corrEl.appendChild(ccgSep);
  data.ccgs.forEach(ccg => {
    const cv = document.createElement('canvas');
    cv.style.height = CH + 'px';
    corrEl.appendChild(cv);
    requestAnimationFrame(() =>
      _drawCorrelogram(cv, ccg.bins, ccg.counts, '#78909C',
        `CCG  u${ccg.uid_a} \u2192 u${ccg.uid_b}`, false, CH));
  });

  // ── Metrics panel ──
  const metEl = inner.querySelector('.mgroup-met');
  metEl.innerHTML = '';

  // Pairwise criteria
  const thr = data.thresholds;
  const pairHdr = document.createElement('h4');
  pairHdr.textContent = 'Pair criteria';
  metEl.appendChild(pairHdr);

  const pairTable = document.createElement('table');
  data.pair_criteria.forEach((pc, idx) => {
    const ia = data.uids.indexOf(pc.uid_a), ib = data.uids.indexOf(pc.uid_b);
    const colA = GROUP_COLORS[ia % GROUP_COLORS.length];
    const colB = GROUP_COLORS[ib % GROUP_COLORS.length];
    const rows = [
      ['Distance',    pc.distance_um.toFixed(1) + '\u00b5m', pc.criteria.distance,       `<${thr.distance_um}\u00b5m`],
      ['Pearson R',   pc.pearson_r.toFixed(3),               pc.criteria.pearson_r,       `>${thr.pearson_r}`],
      ['Amp ratio',   pc.amplitude_ratio.toFixed(2),         pc.criteria.amplitude_ratio, `<${thr.amplitude_ratio}`],
      ['ISI merged',  (pc.isi_merged*100).toFixed(2)+'%',    pc.criteria.isi_merged,      `<${(thr.isi_merged*100).toFixed(0)}%`],
      ['ACG sim',     pc.acg_similarity.toFixed(3),          null, ''],
    ];
    // Pair header row
    if (idx > 0) {
      const sepRow = document.createElement('tr');
      sepRow.className = 'mgroup-pair-sep';
      sepRow.innerHTML = '<td colspan="2"></td>';
      pairTable.appendChild(sepRow);
    }
    const hRow = document.createElement('tr');
    hRow.innerHTML =
      `<td colspan="2" style="font-size:10px;font-weight:700;padding:3px 0 1px;">` +
      `<span class="mgroup-uid-dot" style="background:${colA}"></span>u${pc.uid_a}` +
      ` \u2194 ` +
      `<span class="mgroup-uid-dot" style="background:${colB}"></span>u${pc.uid_b}` +
      ` <span style="color:${pc.n_pass===4?'#2e7d32':'#c62828'};margin-left:4px;">${pc.n_pass}/4</span>` +
      `</td>`;
    pairTable.appendChild(hRow);
    rows.forEach(([name, val, pass, thrStr]) => {
      const tr = document.createElement('tr');
      const valCls = pass === null ? 'minfo' : (pass ? 'mpass' : 'mfail');
      const icon   = pass === null ? '' : (pass ? ' \u2713' : ' \u2717');
      tr.innerHTML =
        `<td class="mname" style="font-size:10px;color:#888;">${name}</td>` +
        `<td class="mval ${valCls}" style="font-size:10px;">${val}${icon}` +
        (thrStr ? `<span style="color:#bbb;font-size:9px;"> (${thrStr})</span>` : '') +
        `</td>`;
      pairTable.appendChild(tr);
    });
  });
  metEl.appendChild(pairTable);

  // Per-unit quality metrics
  const unitHdr = document.createElement('h4');
  unitHdr.textContent = 'Unit metrics';
  metEl.appendChild(unitHdr);

  const unitTable = document.createElement('table');
  const metricNames = [['SNR','snr'],['FR','fr'],['ISI','isi'],['RPCont','rp_cont'],['AmpCut','amp_cut'],['Presence','presence'],['Bleed','bleed']];
  // Header row
  const thRow = document.createElement('tr');
  thRow.innerHTML = '<td style="font-size:9px;color:#aaa;"></td>' +
    data.uids.map((uid, k) =>
      `<td style="font-size:9px;font-weight:700;color:${GROUP_COLORS[k%GROUP_COLORS.length]};text-align:center;">u${uid}</td>`
    ).join('');
  unitTable.appendChild(thRow);
  metricNames.forEach(([label, key]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td style="font-size:10px;color:#888;">${label}</td>` +
      data.uids.map(uid => {
        const m = data.unit_metrics[uid] || {};
        return `<td style="font-size:10px;font-weight:600;text-align:center;">${m[key] ?? 'n/a'}</td>`;
      }).join('');
    unitTable.appendChild(tr);
  });
  metEl.appendChild(unitTable);

  // ── Candidate units (ranked by waveform similarity) ──
  if (SERVER_MODE) {
    _loadCandidates(card, data.rec, data.uids);
  }
}

// ── Build & load an auto-merge group card ─────────────────────────────────────
function _buildAutoMergeGroupCard(rec, group) {
  const card = document.createElement('div');
  card.className = 'mpair-card';
  card.dataset.rec  = rec;
  card.dataset.uids = group.join(',');

  const uidBadges = group.map((uid, k) =>
    `<span style="color:${GROUP_COLORS[k%GROUP_COLORS.length]};font-weight:700;">u${uid}</span>`
  ).join(' <span style="color:#aaa">\u2295</span> ');

  if (SERVER_MODE) {
    card.innerHTML = `
      <div class="mpair-head">
        <span class="mpair-pair">${uidBadges}</span>
        <span class="mpair-rec" title="${rec}">${rec}</span>
        <span class="mpair-verdict" style="margin-left:auto;"></span>
        <button class="unmerge-auto-btn"
                data-rec="${rec}" data-uids="${group.join(',')}"
                onclick="unmergeAutoGroup(this.dataset.rec,this.dataset.uids.split(','))">
          Unmerge</button>
      </div>
      <div class="mpair-body">
        <div class="mpair-loading">Loading\u2026</div>
        <div class="mgroup-inner" style="display:none">
          <div class="mgroup-wv"><canvas></canvas><div class="mgroup-legend"></div></div>
          <div class="mgroup-corr"></div>
          <div class="mgroup-met"></div>
        </div>
        <div class="mpair-err" style="display:none;padding:12px;color:#c62828;font-size:12px;"></div>
      </div>`;
  } else {
    card.innerHTML = `
      <div class="mpair-head">
        <span class="mpair-pair">${uidBadges}</span>
        <span class="mpair-rec" title="${rec}" style="margin-left:auto;">${rec}</span>
      </div>
      <div class="mpair-body">
        <div class="mpair-static-hint">Run <b>python html_review.py --serve</b> to load previews.</div>
      </div>`;
  }
  return card;
}

function _loadAutoMergeGroupCard(card) {
  if (card.dataset.loadState === 'ok' || card.dataset.loadState === 'loading') return;
  card.dataset.loadState = 'loading';
  const rec     = card.dataset.rec;
  const uids    = card.dataset.uids;
  const loading = card.querySelector('.mpair-loading');
  const errEl   = card.querySelector('.mpair-err');

  fetch(`/api/merge-group-preview?rec=${encodeURIComponent(rec)}&uids=${encodeURIComponent(uids)}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        card.dataset.loadState = 'err';
        if (loading) loading.style.display = 'none';
        errEl.textContent = data.error; errEl.style.display = 'block';
        return;
      }
      card.dataset.loadState = 'ok';
      requestAnimationFrame(() => requestAnimationFrame(() => _renderGroupPreview(card, data)));
    })
    .catch(e => {
      card.dataset.loadState = 'err';
      if (loading) loading.style.display = 'none';
      errEl.textContent = 'Failed to load preview.'; errEl.style.display = 'block';
    });
}

function _loadCandidates(card, rec, uids) {
  const metEl = card.querySelector('.mgroup-met');
  if (!metEl) return;
  const existing = metEl.querySelector('.cand-section');
  if (existing) existing.remove();

  const section = document.createElement('div');
  section.className = 'cand-section';
  const hdr = document.createElement('h4');
  hdr.textContent = 'Candidate units';
  section.appendChild(hdr);
  const loadMsg = document.createElement('div');
  loadMsg.style.cssText = 'font-size:10px;color:#888;font-style:italic;padding:4px 0;';
  loadMsg.textContent = 'Loading\u2026';
  section.appendChild(loadMsg);
  metEl.appendChild(section);

  fetch(`/api/candidates?rec=${encodeURIComponent(rec)}&uids=${encodeURIComponent(uids.join(','))}&n=15`)
    .then(r => r.json())
    .then(data => {
      loadMsg.remove();
      if (data.error || !data.candidates || data.candidates.length === 0) {
        const msg = document.createElement('div');
        msg.style.cssText = 'font-size:10px;color:#888;padding:4px 0;';
        msg.textContent = data.error ? 'Error: ' + data.error : 'No candidates found.';
        section.appendChild(msg);
        return;
      }
      const table = document.createElement('table');
      table.className = 'cand-table';
      const thead = document.createElement('tr');
      thead.innerHTML =
        `<td style="color:#aaa;font-size:9px;">Unit</td>` +
        `<td style="color:#aaa;font-size:9px;">R</td>` +
        `<td style="color:#aaa;font-size:9px;">Dist</td>` +
        `<td style="color:#aaa;font-size:9px;">D/R/A</td>` +
        `<td></td>`;
      table.appendChild(thead);
      data.candidates.forEach(c => {
        const tr = document.createElement('tr');
        const icon = pass => pass ? '\u2713' : '\u2717';
        const cls  = pass => pass ? 'mpass' : 'mfail';
        tr.innerHTML =
          `<td style="font-weight:600;">u${c.uid}</td>` +
          `<td class="cand-score">${c.pearson_r.toFixed(3)}</td>` +
          `<td style="color:#555;">${c.distance_um}\u00b5m</td>` +
          `<td>` +
            `<span class="${cls(c.criteria.distance)}" style="font-size:9px;">${icon(c.criteria.distance)}</span>` +
            `<span class="${cls(c.criteria.pearson_r)}" style="font-size:9px;">${icon(c.criteria.pearson_r)}</span>` +
            `<span class="${cls(c.criteria.amplitude_ratio)}" style="font-size:9px;">${icon(c.criteria.amplitude_ratio)}</span>` +
          `</td>` +
          `<td><button class="cand-preview-btn"` +
               ` onclick="showMergePreview('${rec}','${data.group_uids[0]}','${c.uid}')">` +
               `&#9654;</button></td>`;
        table.appendChild(tr);
      });
      section.appendChild(table);
    })
    .catch(() => { loadMsg.textContent = 'Failed to load candidates.'; });
}

// ── Section helpers ───────────────────────────────────────────────────────────
function _appendGroupsSection(scroll, rec, groups) {
  // Auto merged: one card per group (shows all units together)
  groups.forEach(group => {
    const card = _buildAutoMergeGroupCard(rec, group);
    scroll.appendChild(card);
    if (SERVER_MODE && _mergePairsIO) _mergePairsIO.observe(card);
  });
}

function _appendPairsSection(scroll, pairs, title) {
  // Potential pairs: one card per pair (existing pairwise view)
  if (pairs.length === 0) return;
  const hdr = document.createElement('div');
  hdr.className = 'merge-section-header';
  hdr.textContent = `${title}  (${pairs.length} pair${pairs.length !== 1 ? 's' : ''})`;
  scroll.appendChild(hdr);
  pairs.forEach(p => {
    const card = _buildMergePairCard(p);
    scroll.appendChild(card);
    if (SERVER_MODE && _mergePairsIO) _mergePairsIO.observe(card);
  });
}

function renderMergePairsPage() {
  const scroll   = document.getElementById('merge-pairs-scroll');
  const toolbar  = document.getElementById('merge-pairs-toolbar');
  const empty    = document.getElementById('merge-pairs-empty');
  scroll.innerHTML = '';
  if (_mergePairsIO) { _mergePairsIO.disconnect(); _mergePairsIO = null; }

  // Auto merged: count total groups across all recordings
  const autoGroupCount = Object.values(AUTO_MERGES).reduce((s, gs) => s + gs.length, 0);
  const userPairs = allMergedUnitPairs();

  if (autoGroupCount === 0 && userPairs.length === 0) {
    toolbar.textContent = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  toolbar.innerHTML =
    `<b>${autoGroupCount}</b> auto-merged group${autoGroupCount !== 1 ? 's' : ''}` +
    ` &nbsp;|&nbsp; ` +
    `<b>${userPairs.length}</b> potential pair${userPairs.length !== 1 ? 's' : ''}` +
    (SERVER_MODE ? ' &nbsp;\u2014 scroll to load previews' : '');

  if (SERVER_MODE) {
    _mergePairsIO = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (!en.isIntersecting) return;
        _mergePairsIO.unobserve(en.target);
        // Route to the correct loader based on card type
        if (en.target.dataset.uids) _loadAutoMergeGroupCard(en.target);
        else                        loadMergePairCard(en.target);
      });
    }, { root: scroll, rootMargin: '200px' });
  }

  // Auto merged section header + one card per group
  if (autoGroupCount > 0) {
    const hdr = document.createElement('div');
    hdr.className = 'merge-section-header';
    hdr.textContent = `Auto merged  (${autoGroupCount} group${autoGroupCount !== 1 ? 's' : ''})`;
    scroll.appendChild(hdr);
    for (const [rec, groups] of Object.entries(AUTO_MERGES))
      _appendGroupsSection(scroll, rec, groups);
  }

  _appendPairsSection(scroll, userPairs, 'Potential pairs');
}

function loadMergePairCard(card) {
  if (card.dataset.loadState === 'ok' || card.dataset.loadState === 'loading') return;
  card.dataset.loadState = 'loading';
  const rec = card.dataset.rec;
  const uid_a = card.dataset.uidA;
  const uid_b = card.dataset.uidB;
  const loading = card.querySelector('.mpair-loading');
  const inner = card.querySelector('.mpair-inner');
  const errEl = card.querySelector('.mpair-err');

  fetch(`/api/merge-preview?rec=${encodeURIComponent(rec)}&uid_a=${uid_a}&uid_b=${uid_b}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        card.dataset.loadState = 'err';
        if (loading) loading.style.display = 'none';
        errEl.textContent = data.error;
        errEl.style.display = 'block';
        return;
      }
      card.dataset.loadState = 'ok';
      if (loading) loading.style.display = 'none';
      inner.style.display = 'flex';
      const [ca, cb, cc] = inner.querySelectorAll('.mpair-corr canvas');
      const run = () => applyMergePreviewToTargets(data, {
        verdictEl: card.querySelector('.mpair-verdict'),
        wvCanvas: inner.querySelector('.mpair-wv canvas'),
        legendEl: inner.querySelector('.mpair-legend'),
        acgA: ca, acgB: cb, ccg: cc,
        metricsTable: inner.querySelector('.mpair-metrics-table'),
        compact: true,
      });
      requestAnimationFrame(() => requestAnimationFrame(run));
    })
    .catch(() => {
      card.dataset.loadState = 'err';
      if (loading) loading.style.display = 'none';
      errEl.textContent = 'Failed to load preview.';
      errEl.style.display = 'block';
    });
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────
document.querySelectorAll('#view-nav .view-tab').forEach(btn => {
  btn.addEventListener('click', () => setAppView(btn.dataset.view));
});
buildRecFilter();
document.querySelectorAll('.af').forEach(cb => cb.addEventListener('change', renderGrid));
renderGrid();
</script>
</body>
</html>
"""


# ── Shared data-building helper ───────────────────────────────────────────────
def _build_html_content(labels: dict, auto_merge_groups: dict, user_merge_groups: dict) -> str:
    units = discover_units(sortout_folder)
    if not units:
        raise RuntimeError(f"No unit images found under {sortout_folder}")

    print(f"Found {len(units)} units across {len(set(r for r,_,_ in units))} recordings.")
    print("Encoding images and loading metrics (first run may take a moment) ...")

    records = build_unit_records(units, labels)
    records.sort(key=lambda r: (r["rec"], r.get("depth", 0.0)))
    thresholds = {
        "noise_snr":     NOISE_SNR_THRESHOLD,
        "noise_presence":NOISE_PRESENCE_THRESHOLD,
        "sua_snr":       SUA_SNR_THRESHOLD,
        "sua_isi":       SUA_ISI_RATIO_THRESHOLD,
        "sua_fr":        SUA_FIRING_RATE_MIN,
        "sua_rp":        SUA_RP_THRESHOLD,
        "sua_ampcut":    SUA_AMPLITUDE_CUTOFF_THRESHOLD,
    }
    html = HTML_TEMPLATE.replace("__THRESHOLDS_JSON__",   json.dumps(thresholds))
    html = html.replace("__UNITS_JSON__",                  json.dumps(records, separators=(',', ':')))
    html = html.replace("__AUTO_MERGES_JSON__",            json.dumps(auto_merge_groups, separators=(',', ':')))
    html = html.replace("__MERGES_JSON__",                 json.dumps(user_merge_groups, separators=(',', ':')))
    return html


_USER_MERGE_JSON_NAME = "unit_user_merges.json"


def _load_initial_state():
    labels = {}
    if output_json.exists():
        with open(output_json) as f:
            labels = json.load(f)
        total = sum(len(v) for v in labels.values())
        print(f"Loaded {total} existing labels from {output_json.name}")
    else:
        print("No existing labels — starting fresh.")

    # Combined merge file: { "auto": {...}, "user": {...}, "blacklist": {...} }
    # Backward compat: if no "auto"/"user" keys, treat whole file as old flat auto map.
    auto_merge_groups = {}
    user_merge_groups = {}
    if merge_json.exists():
        with open(merge_json) as f:
            combined = json.load(f)
        if "auto" in combined or "user" in combined:
            auto_merge_groups = _merge_map_to_groups(combined.get("auto", {}))
            user_merge_groups = _merge_map_to_groups(combined.get("user", {}))
        else:
            auto_merge_groups = _merge_map_to_groups(combined)
            user_merge_groups = {}
        total_g = sum(len(v) for v in auto_merge_groups.values())
        total_u = sum(len(v) for v in user_merge_groups.values())
        print(f"Loaded {total_g} auto + {total_u} user merge group(s) from {merge_json.name}")

    # Backward compat: absorb old unit_user_merges.json if it exists and no user groups yet
    old_user_json = sortout_folder / _USER_MERGE_JSON_NAME
    if old_user_json.exists() and not user_merge_groups:
        with open(old_user_json) as f:
            user_merge_groups = _merge_map_to_groups(json.load(f))
        total_u = sum(len(v) for v in user_merge_groups.values())
        print(f"Loaded {total_u} user merge group(s) from {old_user_json.name} (legacy)")

    return labels, auto_merge_groups, user_merge_groups


# ── Static export (original behaviour) ───────────────────────────────────────
def generate_html(open_browser: bool = False, sortout_folder: Path | str | None = None):
    configure_sortout(sortout_folder)
    labels, auto_merge_groups, user_merge_groups = _load_initial_state()
    html = _build_html_content(labels, auto_merge_groups, user_merge_groups)
    output_html.write_text(html, encoding="utf-8")
    size_mb = output_html.stat().st_size / 1e6
    print(f"\nHTML written → {output_html}  ({size_mb:.1f} MB)")
    if open_browser:
        webbrowser.open(output_html.as_uri())


# ── Interactive server ────────────────────────────────────────────────────────
def launch_server(
    port: int = 7979,
    open_browser: bool = True,
    sortout_folder: Path | str | None = None,
):
    """
    Serve the interactive HTML review at http://localhost:<port>.
    All label and merge changes POST to /api/save → written to disk.
    /api/merge-preview?rec=&uid_a=&uid_b= returns waveforms, ACG, CCG, metrics.
    The Merged pairs tab lists every unit pair in each merge group and lazy-loads
    the same preview via that endpoint.
    Blocks until Ctrl-C.

    sortout_folder:
        Session folder (…/animal/session) or any ancestor tree that contains
        ``**/sorting_results_*/raw_units/unit_summary_*.png``. If omitted, uses
        the module default or ``SPIKE_SORTOUT`` / ``SORTOUT_FOLDER``.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    configure_sortout(sortout_folder)
    labels, auto_merge_groups, user_merge_groups = _load_initial_state()
    print("Building HTML (encoding images, loading metrics) ...")
    html_bytes = _build_html_content(labels, auto_merge_groups, user_merge_groups).encode("utf-8")
    print(f"HTML ready ({len(html_bytes)/1e6:.1f} MB).")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", html_bytes)
            elif self.path.startswith("/api/merge-group-preview"):
                self._handle_group_preview()
            elif self.path.startswith("/api/merge-preview"):
                self._handle_preview()
            elif self.path.startswith("/api/candidates"):
                self._handle_candidates()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/save":
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length))

                # Save user-defined merge groups into the combined merge file
                existing = json.load(open(merge_json)) if merge_json.exists() else {}
                groups_per_rec = body.get("merges", {})
                existing["user"] = _groups_to_merge_map(groups_per_rec) if groups_per_rec else {}
                with open(merge_json, "w") as f:
                    json.dump(existing, f, indent=2)

                # Save manual labels (only units with a non-empty override)
                if "labels" in body:
                    with open(output_json, "w") as f:
                        json.dump(body["labels"], f, indent=2)

                self._send(200, "application/json", b'{"ok":true}')
            elif self.path == "/api/unmerge-auto":
                length  = int(self.headers.get("Content-Length", 0))
                body    = json.loads(self.rfile.read(length))
                rec     = body.get("rec", "")
                uids    = body.get("uids", [])
                if not (rec and uids):
                    self._send(400, "application/json", b'{"error":"missing rec or uids"}')
                    return
                existing = json.load(open(merge_json)) if merge_json.exists() else {}

                # Remove from "auto" section (handle both new combined and old flat format)
                if "auto" in existing:
                    auto_map = existing["auto"]
                else:
                    # Old flat format: the whole dict is the auto map
                    auto_map = existing
                rec_map  = auto_map.get(rec, {})
                uids_set = set(str(u) for u in uids)
                for k in [k for k, v in rec_map.items()
                          if str(k) in uids_set or str(v) in uids_set]:
                    del rec_map[k]
                if rec_map:
                    auto_map[rec] = rec_map
                elif rec in auto_map:
                    del auto_map[rec]
                existing["auto"] = auto_map

                # Add group to blacklist so it is never re-merged
                bl = existing.setdefault("blacklist", {})
                bl.setdefault(rec, [])
                group_sorted = sorted(str(u) for u in uids)
                if group_sorted not in bl[rec]:
                    bl[rec].append(group_sorted)

                with open(merge_json, "w") as f:
                    json.dump(existing, f, indent=2)
                self._send(200, "application/json", b'{"ok":true}')
            else:
                self.send_error(404)

        def _handle_preview(self):
            parsed = urlparse(self.path)
            qs     = parse_qs(parsed.query)
            rec    = qs.get("rec",   [""])[0]
            uid_a  = qs.get("uid_a", [""])[0]
            uid_b  = qs.get("uid_b", [""])[0]
            if not (rec and uid_a and uid_b):
                self._send(400, "application/json", b'{"error":"missing params"}')
                return
            try:
                result = _compute_merge_preview(rec, uid_a, uid_b)
            except Exception as e:
                result = {"error": str(e)}
            payload = json.dumps(result, separators=(',', ':')).encode("utf-8")
            self._send(200, "application/json", payload)

        def _handle_group_preview(self):
            parsed    = urlparse(self.path)
            qs        = parse_qs(parsed.query)
            rec       = qs.get("rec",  [""])[0]
            uids_raw  = qs.get("uids", [""])[0]
            if not (rec and uids_raw):
                self._send(400, "application/json", b'{"error":"missing params"}')
                return
            uid_list = [u.strip() for u in uids_raw.split(",") if u.strip()]
            if len(uid_list) < 2:
                self._send(400, "application/json", b'{"error":"need at least 2 units"}')
                return
            try:
                result = _compute_merge_group_preview(rec, uid_list)
            except Exception as e:
                result = {"error": str(e)}
            payload = json.dumps(result, separators=(',', ':')).encode("utf-8")
            self._send(200, "application/json", payload)

        def _handle_candidates(self):
            parsed   = urlparse(self.path)
            qs       = parse_qs(parsed.query)
            rec      = qs.get("rec",  [""])[0]
            uids_raw = qs.get("uids", [""])[0]
            n_raw    = qs.get("n",    ["15"])[0]
            if not (rec and uids_raw):
                self._send(400, "application/json", b'{"error":"missing params"}')
                return
            uid_list = [u.strip() for u in uids_raw.split(",") if u.strip()]
            try:
                n = int(n_raw)
            except ValueError:
                n = 15
            try:
                result = _compute_candidates(rec, uid_list, n)
            except Exception as e:
                result = {"error": str(e)}
            payload = json.dumps(result, separators=(',', ':')).encode("utf-8")
            self._send(200, "application/json", payload)

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("localhost", port), Handler)
    url    = f"http://localhost:{port}"
    print(f"\nReview server running at {url}")
    print("Label/merge changes auto-save to disk.  Press Ctrl-C to stop.\n")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sortout_arg = None
    if "--sortout" in sys.argv:
        i = sys.argv.index("--sortout")
        if i + 1 < len(sys.argv):
            sortout_arg = sys.argv[i + 1]

    if "--serve" in sys.argv:
        port_arg = next(
            (int(sys.argv[i + 1]) for i, a in enumerate(sys.argv)
             if a == "--port" and i + 1 < len(sys.argv)),
            7979,
        )
        launch_server(port=port_arg, open_browser=True, sortout_folder=sortout_arg)
    else:
        generate_html(open_browser="--open" in sys.argv, sortout_folder=sortout_arg)
