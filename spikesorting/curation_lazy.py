"""
curation_lazy.py
================
Auto + manual spike sorting curation pipeline.

Auto-curation stage (Wu et al. 2024, STAR Methods):
  - Rejection:  SNR, amplitude, firing rate, ISI violations, noise overlap,
                presence ratio, far-electrode amplitude bleed
  - Merging:    centroid distance, waveform Pearson R, amplitude ratio,
                post-merge ISI violation check

Only borderline units (metrics in the gray zone) are shown in the GUI.
All metrics are displayed in the title so each manual decision takes seconds.

Usage
-----
  python curation_lazy.py              # auto-classify + manual review
  python curation_lazy.py --merge      # also run merge pass after labeling
"""

import itertools
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import spikeinterface as si

# ── Configuration ──────────────────────────────────────────────────────────────
sortout_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313")
output_json = sortout_folder / "unit_labels.json"
RUN_MERGE          = True   # set True to run merge pass after labeling
REVIEW_AUTO        = False  # set True to open GUI for final review of auto-classified units
OVERWRITE          = False   # set True to discard existing labels and re-classify from scratch
LAUNCH_HTML_REVIEW = True   # set True to launch interactive HTML review after auto-curation

# ── Auto-curation thresholds ───────────────────────────────────────────────────
#
# Three-tier labeling: Noise → SUA → MUA (default fallback).
# Priority: Noise gate evaluated first (hard reject), then SUA gate
# (all conditions must pass simultaneously), remainder → MUA.
#
# Noise gate — either condition fires → Noise
NOISE_SNR_THRESHOLD          = 3.0    # below this → Noise
NOISE_PRESENCE_THRESHOLD     = 0.5    # below this → Noise (unit not stable)
NOISE_ISI_THRESHOLD          = 2   # ISI violations ratio above this → Noise

# SUA gate — ALL conditions must pass
SUA_SNR_THRESHOLD            = 5.0    # Allen/IBL convention
SUA_ISI_RATIO_THRESHOLD      = 0.2    # < 20% contamination (Hill metric)
SUA_FIRING_RATE_MIN          = 0.1    # Hz — exclude near-silent units
SUA_RP_THRESHOLD             = 0.1    # sliding refractory period violations (Llobet)
SUA_AMPLITUDE_CUTOFF_THRESHOLD = 0.1  # amplitude cutoff (fraction of spikes clipped)

ISI_THRESHOLD_MS             = 1.5    # biophysical refractory period used for ISI metric

# GMM sanity-check parameters (used in compute_and_label_units)
GMM_N_COMPONENTS  = 3
GMM_RANDOM_STATE  = 42

# Far-electrode bleed check (Wu et al. criterion e)
BLEED_DISTANCE_THRESH_UM   = 140.0   # channels >= this distance are "far"
BLEED_AMPLITUDE_RATIO_THRESH = 0.40  # flag if far-channel amplitude > 40% of primary

# Merge criteria (Wu et al. STAR Methods — all four must be satisfied simultaneously)
MERGE_DISTANCE_THRESH_UM    = 25.0   # centroid distance between candidate clusters
MERGE_PEARSON_THRESH        = 0.8    # Pearson R on template waveforms
MERGE_AMPLITUDE_RATIO_THRESH = 1.5   # amplitude of one cluster < 1.5x the other
MERGE_ISI_THRESH            = 0.07   # post-merge ISI violation must remain < 7%


# ── Folder discovery ───────────────────────────────────────────────────────────
def discover_units(sortout_folder: Path):
    """
    Return list of (recording_name, unit_id, image_path) tuples.

    Supports both layouts:
      • session/shankN/sorting_results_*/raw_units/unit_summary_*.png
      • animal/session/shankN/sorting_results_*/raw_units/...

    ``recording_name`` is the path from ``sortout_folder`` to the folder that
    *contains* ``sorting_results_*`` (e.g. ``shank0`` or ``CnL43_20260408/shank0``).
    """
    units = []
    folder = Path(sortout_folder).expanduser().resolve()
    if not folder.is_dir():
        return units

    # parent_dir -> latest sorting_results_* path (lexicographic max on folder name)
    latest_by_parent: Dict[Path, Path] = {}
    for cand in folder.rglob("*"):
        if not cand.is_dir() or not cand.name.startswith("sorting_results_"):
            continue
        raw_units_dir = cand / "raw_units"
        if not raw_units_dir.is_dir():
            continue
        if not any(raw_units_dir.glob("unit_summary_*.png")):
            continue
        parent_key = cand.parent.resolve()
        prev = latest_by_parent.get(parent_key)
        if prev is None or cand.name > prev.name:
            latest_by_parent[parent_key] = cand

    def _sort_key(p: Path) -> str:
        try:
            return p.relative_to(folder).as_posix()
        except ValueError:
            return p.as_posix()

    for sort_parent in sorted(latest_by_parent.keys(), key=_sort_key):
        sorting_dir = latest_by_parent[sort_parent]
        raw_units_dir = sorting_dir / "raw_units"
        try:
            rec_name = sort_parent.relative_to(folder).as_posix()
        except ValueError:
            rec_name = sort_parent.name
        images = sorted(
            raw_units_dir.glob("unit_summary_*.png"),
            key=lambda p: int(re.search(r"unit_summary_(\d+)", p.stem).group(1)),
        )
        for img_path in images:
            uid = re.search(r"unit_summary_(\d+)", img_path.stem).group(1)
            units.append((rec_name, uid, img_path))

    if not units:
        try:
            subs = sorted(p.name for p in folder.iterdir() if p.is_dir())
        except OSError:
            subs = []
        print(
            f"[discover_units] No unit_summary_*.png found under {folder}.\n"
            f"  Expected **/sorting_results_*/raw_units/unit_summary_*.png\n"
            f"  Subfolders here ({len(subs)}): {subs[:25]}"
            + (" …" if len(subs) > 25 else "")
        )
    return units


# ── JSON load / save ───────────────────────────────────────────────────────────
def load_labels(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict, path: Path):
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


# ── Quality metrics ────────────────────────────────────────────────────────────
_metrics_cache: dict = {}   # rec_name -> dict[str(unit_id) -> {metric: value}]
_analyzer_cache: dict = {}  # rec_name -> SortingAnalyzer (kept alive for merge pass)

# Maps SI compute name → expected DataFrame column name after normalisation.
# Used to detect which metrics are already present before recomputing.
_REQUIRED_METRICS: dict = {
    "snr":             "snr",
    "firing_rate":     "firing_rate",
    "presence_ratio":  "presence_ratio",
    "amplitude_cutoff":"amplitude_cutoff",
    "isi_violation":   "isi_violations_ratio",   # SI stores as isi_violations_ratio
    # rp_contamination is derived from spike trains (Llobet), not an SI metric
}


def _normalize_qm_columns(df):
    """Normalise ISI and RP column names across SpikeInterface versions."""
    rename = {}
    if "isi_violations_ratio" not in df.columns:
        for cand in ("isi_violation", "isi_violation_ratio"):
            if cand in df.columns:
                rename[cand] = "isi_violations_ratio"
                break
    return df.rename(columns=rename) if rename else df


def _load_or_compute_qm(sa, isi_threshold_ms: float = ISI_THRESHOLD_MS):
    """
    Return a quality-metrics DataFrame for *sa* containing all _REQUIRED_METRICS.
    Only missing metrics are computed; already-present columns are reused as-is.
    """
    qm_ext = sa.get_extension("quality_metrics")
    df = _normalize_qm_columns(qm_ext.get_data()) if qm_ext is not None else None

    existing_cols = set(df.columns) if df is not None else set()
    missing = [k for k, v in _REQUIRED_METRICS.items() if v not in existing_cols]

    if missing:
        print(f"[Metrics] Computing missing metrics: {missing}")
        ext_params = {}
        if "isi_violation" in missing:
            ext_params["isi_violation"] = {"isi_threshold_ms": isi_threshold_ms}
        new_ext = sa.compute("quality_metrics", metric_names=missing,
                             extension_params=ext_params)
        new_df = _normalize_qm_columns(new_ext.get_data())
        if df is None:
            df = new_df
        else:
            # Merge: add only newly computed columns into the existing frame
            new_cols = [c for c in new_df.columns if c not in existing_cols]
            df = df.join(new_df[new_cols])
    else:
        print("[Metrics] All required metrics already present, skipping computation.")

    # Llobet rp_contamination is derived from spike trains, not an SI metric
    if "rp_contamination" not in df.columns:
        df["rp_contamination"] = _compute_rp_contamination(sa, isi_threshold_ms)

    return df


def _compute_rp_contamination(sa, t_rp_ms: float = ISI_THRESHOLD_MS) -> "pd.Series":
    """
    Llobet refractory-period contamination for every unit in *sa*.

    Solves the quadratic  n_v = 2·N²·c·(1-c)·t_rp / T  for c:
        c = (1 - sqrt(1 - 2·n_v·T / (N²·t_rp))) / 2

    Bounded [0, 1] — saturates at 1.0 when the discriminant goes negative.
    Compare with the Hill metric (isi_violations_ratio), which is the raw
    ratio of violation rate / total rate and is unbounded.
    """
    sorting = sa.sorting
    fs = float(sorting.get_sampling_frequency())
    t_rp = t_rp_ms / 1000.0  # convert to seconds

    values = {}
    for uid in sa.unit_ids:
        st = sorting.get_unit_spike_train(unit_id=uid).astype(float)
        N = len(st)
        if N < 2:
            values[uid] = 0.0
            continue
        T = (st[-1] - st[0]) / fs  # recording span in seconds
        if T <= 0:
            values[uid] = 0.0
            continue
        n_v = int(np.sum(np.diff(st) / fs < t_rp))
        discriminant = 1.0 - 2.0 * n_v * T / (N ** 2 * t_rp)
        if discriminant < 0:
            values[uid] = 1.0
        else:
            values[uid] = float(np.clip((1.0 - np.sqrt(discriminant)) / 2.0, 0.0, 1.0))

    return pd.Series(values, name="rp_contamination")


def _get_analyzer(rec_name: str) -> Optional[object]:
    """Load and cache the SortingAnalyzer for a recording."""
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
        print(f"[Analyzer] Failed to load {analyzer_folder}: {e}")
        return None


def _get_templates(sa) -> Optional[np.ndarray]:
    """
    Return templates array (n_units, n_samples, n_channels).
    Computes and stores the extension if not already present.
    """
    ext = sa.get_extension("templates")
    if ext is None:
        try:
            ext = sa.compute("templates")
        except Exception as e:
            print(f"[Templates] Could not compute: {e}")
            return None
    # SpikeInterface >= 0.101 uses get_templates(); older uses get_data()
    try:
        return ext.get_templates()
    except AttributeError:
        try:
            return ext.get_data()
        except Exception:
            return None


def compute_bleed_flag(sa, unit_id) -> bool:
    """
    Wu et al. criterion (e): returns True (artifact flag) if any channel
    >= BLEED_DISTANCE_THRESH_UM µm from the primary channel carries amplitude
    > BLEED_AMPLITUDE_RATIO_THRESH * primary amplitude.

    Rationale: real neurons decay steeply with distance (<50 µm detection range).
    A strong signal on a far channel means the waveform is motion artifact or
    electrical crosstalk, not a biological spike.
    """
    try:
        templates = _get_templates(sa)
        if templates is None:
            return False

        unit_ids = list(sa.unit_ids)
        uid_cast = _cast_uid(unit_id, unit_ids)
        if uid_cast is None:
            return False

        idx = unit_ids.index(uid_cast)
        template = templates[idx]                          # (n_samples, n_channels)
        p2p = template.max(axis=0) - template.min(axis=0) # (n_channels,)

        primary_ch = int(p2p.argmax())
        primary_amp = float(p2p[primary_ch])
        if primary_amp == 0:
            return False

        channel_positions = sa.get_channel_locations()     # (n_channels, 2)
        primary_pos = channel_positions[primary_ch]
        distances = np.linalg.norm(channel_positions - primary_pos, axis=1)

        far_mask = distances >= BLEED_DISTANCE_THRESH_UM
        if not np.any(far_mask):
            return False

        far_ratio = p2p[far_mask] / primary_amp
        return bool(np.any(far_ratio > BLEED_AMPLITUDE_RATIO_THRESH))

    except Exception as e:
        print(f"[Bleed] Could not compute bleed flag for unit {unit_id}: {e}")
        return False


def load_metrics(rec_name: str) -> dict:
    """
    Load or compute all quality metrics for one recording.
    Returns dict: str(unit_id) -> {metric_name: value, ...}

    Results are memory-cached so the analyzer is only loaded once per session.
    The quality_metrics extension is also persisted inside the SortingAnalyzer
    folder, so subsequent runs load instantly without recomputation.
    """
    if rec_name in _metrics_cache:
        return _metrics_cache[rec_name]

    print(f"[Metrics] Loading metrics for {rec_name} ...")
    sa = _get_analyzer(rec_name)
    if sa is None:
        _metrics_cache[rec_name] = {}
        return {}

    try:
        # Ensure templates exist (needed for bleed check)
        _get_templates(sa)

        df = _load_or_compute_qm(sa)

        # Attach far-electrode bleed flag (not a standard SI metric)
        df["bleed_flag"] = [compute_bleed_flag(sa, uid) for uid in df.index]

        result = {str(uid): row.to_dict() for uid, row in df.iterrows()}
        print(f"[Metrics] Done — {len(result)} units for {rec_name}.")
        _metrics_cache[rec_name] = result
        return result

    except Exception as e:
        print(f"[Metrics] Error for {rec_name}: {e}")
        _metrics_cache[rec_name] = {}
        return {}


# ── Auto-classification logic ──────────────────────────────────────────────────
def _get_reject_reasons(m: dict) -> list:
    """
    Return human-readable strings explaining why a unit was classified as Noise.
    Empty list means no noise-gate criterion fired (unit is SUA or MUA).
    """
    reasons = []
    snr = m.get("snr")
    presence = m.get("presence_ratio")
    if snr is not None and snr < NOISE_SNR_THRESHOLD:
        reasons.append(f"snr={snr:.3f} < {NOISE_SNR_THRESHOLD}")
    if presence is not None and presence < NOISE_PRESENCE_THRESHOLD:
        reasons.append(f"presence_ratio={presence:.3f} < {NOISE_PRESENCE_THRESHOLD}")
    isi = m.get("isi_violations_ratio")
    if isi is not None and isi > NOISE_ISI_THRESHOLD:
        reasons.append(f"isi={isi:.3f} > {NOISE_ISI_THRESHOLD}")
    if m.get("bleed_flag", False):
        reasons.append("bleed_flag triggered")
    return reasons


def _get_sua_failures(m: dict) -> list:
    """
    For a non-Noise unit, return which SUA gate conditions it failed.
    Empty list means all SUA criteria passed (unit should be SUA, not MUA).
    """
    failures = []
    checks = [
        ("snr",                 m.get("snr",                 0),   ">=", SUA_SNR_THRESHOLD),
        ("isi_violations_ratio",m.get("isi_violations_ratio",1),   "<",  SUA_ISI_RATIO_THRESHOLD),
        ("firing_rate",         m.get("firing_rate",          0),  ">=", SUA_FIRING_RATE_MIN),
        ("rp_contamination",    m.get("rp_contamination",     1),  "<",  SUA_RP_THRESHOLD),
        ("amplitude_cutoff",    m.get("amplitude_cutoff",     1),  "<",  SUA_AMPLITUDE_CUTOFF_THRESHOLD),
    ]
    for name, val, op, thresh in checks:
        if op == ">=" and val < thresh:
            failures.append(f"{name}={val:.3f} (need >={thresh})")
        elif op == "<" and val >= thresh:
            failures.append(f"{name}={val:.3f} (need <{thresh})")
    return failures


def auto_classify(m: dict) -> str:
    """
    Three-tier classification from a quality-metrics dict.

    Returns
    -------
    "Noise"  — noise gate fired (low SNR, low presence, or bleed)
    "SUA"    — all SUA criteria satisfied simultaneously
    "MUA"    — real activity present but does not meet SUA bar
    """
    if not m:
        return "MUA"  # no metrics → conservative default

    # --- Noise gate (highest priority) ---
    if m.get("snr", 0) < NOISE_SNR_THRESHOLD:
        print(f"    auto-Noise: snr={m['snr']:.3f} < {NOISE_SNR_THRESHOLD}")
        return "Noise"
    if m.get("presence_ratio", 0) < NOISE_PRESENCE_THRESHOLD:
        print(f"    auto-Noise: presence_ratio={m['presence_ratio']:.3f} < {NOISE_PRESENCE_THRESHOLD}")
        return "Noise"
    if m.get("isi_violations_ratio", 0) > NOISE_ISI_THRESHOLD:
        print(f"    auto-Noise: isi={m['isi_violations_ratio']:.3f} > {NOISE_ISI_THRESHOLD}")
        return "Noise"
    if m.get("bleed_flag", False):
        print("    auto-Noise: bleed_flag triggered")
        return "Noise"

    # --- SUA gate (all conditions must pass simultaneously) ---
    if (m.get("snr", 0) >= SUA_SNR_THRESHOLD and
            m.get("isi_violations_ratio", 1) < SUA_ISI_RATIO_THRESHOLD and
            m.get("firing_rate", 0) >= SUA_FIRING_RATE_MIN and
            m.get("rp_contamination", 1) < SUA_RP_THRESHOLD and
            m.get("amplitude_cutoff", 1) < SUA_AMPLITUDE_CUTOFF_THRESHOLD):
        return "SUA"

    return "MUA"


# ── Auto-classification review GUI ────────────────────────────────────────────
def review_auto_labels(auto_units: list, labels: dict, output_path: Path) -> dict:
    """
    Show every auto-classified unit (good + noise) for a final human check.

    auto_units : list of (rec_name, uid, img_path, metrics_dict, auto_decision)

    For each unit the title shows:
      • The auto decision (green = good, red = noise)
      • Every criterion that triggered rejection (noise units only)
      • All metric values vs. their thresholds
    Buttons: Confirm (keep auto) | Good | MUA | Noise | Back | Skip
    """
    if not auto_units:
        print("No auto-classified units to review.")
        return labels

    print(f"\n── Auto-label review: {len(auto_units)} units ────────────────────────")
    state = {"idx": 0}
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.14)

    def fmt(val, decimals: int = 3, suffix: str = "") -> str:
        if val is None:
            return "n/a"
        try:
            return f"{float(val):.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return str(val)

    def show_current():
        ax.clear()
        rec_name, uid, img_path, m, auto_dec = auto_units[state["idx"]]
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()

        if auto_dec == "Noise":
            failures = _get_reject_reasons(m)
            reason_line = ("  NOISE GATE: " + " | ".join(failures)) if failures else ""
        elif auto_dec == "MUA":
            failures = _get_sua_failures(m)
            reason_line = ("  SUA failed: " + " | ".join(failures)) if failures else ""
        else:
            reason_line = ""
        dec_color = "darkgreen" if auto_dec == "SUA" else ("darkorange" if auto_dec == "MUA" else "darkred")

        line1 = (
            f"[{rec_name}]  unit {uid}  ({state['idx'] + 1} / {len(auto_units)})  "
            f"── AUTO: {auto_dec}{reason_line}"
        )
        line2 = (
            f"SNR={fmt(m.get('snr'), 2)}  |  "
            f"FR={fmt(m.get('firing_rate'), 3, ' Hz')}  |  "
            f"ISI={fmt(m.get('isi_violations_ratio'), 4)}  |  "
            f"RPCont={fmt(m.get('rp_contamination'), 4)}  |  "
            f"AmpCut={fmt(m.get('amplitude_cutoff'), 3)}  |  "
            f"Presence={fmt(m.get('presence_ratio'), 2)}  |  "
            f"Bleed={'YES' if m.get('bleed_flag') else 'no'}"
        )
        ax.set_title(f"{line1}\n{line2}", fontsize=9, color=dec_color,
                     loc="left", pad=6)
        fig.canvas.draw_idle()

    def _apply_label(label_value: str):
        rec_name, uid, _, _, _ = auto_units[state["idx"]]
        labels.setdefault(rec_name, {})[uid] = label_value
        save_labels(labels, output_path)
        state["idx"] += 1
        if state["idx"] >= len(auto_units):
            print("Auto-label review complete!")
            plt.close(fig)
        else:
            show_current()

    def on_confirm(_):
        _, _, _, _, auto_dec = auto_units[state["idx"]]
        _apply_label(auto_dec)

    def on_back(_):
        if state["idx"] > 0:
            state["idx"] -= 1
            show_current()

    def on_skip(_):
        state["idx"] += 1
        if state["idx"] >= len(auto_units):
            print("Reached end (some units skipped).")
            plt.close(fig)
        else:
            show_current()

    btn_specs = [
        (0.05, "Confirm",  "lightblue",   on_confirm),
        (0.22, "SUA",      "lightgreen",  lambda _: _apply_label("SUA")),
        (0.35, "MUA",      "khaki",       lambda _: _apply_label("MUA")),
        (0.48, "Noise",    "lightsalmon", lambda _: _apply_label("Noise")),
        (0.66, "Back",     "lightgray",   on_back),
        (0.79, "Skip",     "whitesmoke",  on_skip),
    ]
    buttons = []
    for x, text, color, cb in btn_specs:
        ax_btn = fig.add_axes([x, 0.02, 0.12, 0.06])
        btn = Button(ax_btn, text, color=color, hovercolor="lightskyblue")
        btn.on_clicked(cb)
        buttons.append(btn)

    show_current()
    plt.show()
    return labels


# ── Merge pass (Wu et al. 2024, STAR Methods p.20) ────────────────────────────
def _cast_uid(uid_str, all_unit_ids: list):
    """Resolve a string unit ID to the type used internally by the sorting."""
    if uid_str in all_unit_ids:
        return uid_str
    try:
        uid_int = int(uid_str)
        if uid_int in all_unit_ids:
            return uid_int
    except (ValueError, TypeError):
        pass
    return None


def _isi_ratio_from_spike_train(spike_train: np.ndarray, fs: float,
                                 refractory_ms: float = 2.0) -> float:
    """Fraction of ISIs below the refractory period."""
    if spike_train.size < 2:
        return 0.0
    isi_ms = np.diff(spike_train.astype(float)) / fs * 1000.0
    return float(np.sum(isi_ms < refractory_ms)) / len(isi_ms)


def run_merge_pass(rec_name: str, good_unit_ids: list,
                   blacklist: list | None = None) -> dict:
    """
    Iteratively merge over-split units among all non-noise units (SUA + MUA)
    using the four Wu et al. criteria (ALL must be satisfied simultaneously):

      a) Centroid distance between clusters < MERGE_DISTANCE_THRESH_UM µm
      b) Pearson R between template waveforms > MERGE_PEARSON_THRESH
      c) Amplitude ratio of the two clusters < MERGE_AMPLITUDE_RATIO_THRESH
      d) Post-merge ISI violation % < MERGE_ISI_THRESH

    Parameters
    ----------
    blacklist : list of groups (each group is a list of uid strings)
        Pairs within any blacklisted group are never re-merged.

    Returns
    -------
    dict: str(unit_id) -> str(canonical_unit_id)
        Units that are NOT merged map to themselves.
        Units merged into another map to their canonical ID.
    """
    from scipy.stats import pearsonr

    if not good_unit_ids:
        return {}

    sa = _get_analyzer(rec_name)
    if sa is None:
        return {uid: uid for uid in good_unit_ids}

    templates = _get_templates(sa)
    if templates is None:
        print(f"[Merge] No templates for {rec_name}, skipping merge.")
        return {uid: uid for uid in good_unit_ids}

    all_unit_ids = list(sa.unit_ids)
    channel_positions = sa.get_channel_locations()  # (n_channels, 2)
    sorting = sa.sorting
    fs = float(sorting.get_sampling_frequency())

    # ── Helper closures ──────────────────────────────────────────────────────
    def get_centroid(uid_str: str) -> np.ndarray:
        uid = _cast_uid(uid_str, all_unit_ids)
        if uid is None:
            return np.zeros(2)
        idx = all_unit_ids.index(uid)
        p2p = templates[idx].max(axis=0) - templates[idx].min(axis=0)
        total = p2p.sum()
        weights = p2p / total if total > 0 else np.ones(len(p2p)) / len(p2p)
        return (channel_positions * weights[:, None]).sum(axis=0)

    def get_primary_amp(uid_str: str) -> float:
        uid = _cast_uid(uid_str, all_unit_ids)
        if uid is None:
            return 0.0
        idx = all_unit_ids.index(uid)
        p2p = templates[idx].max(axis=0) - templates[idx].min(axis=0)
        return float(p2p.max())

    def get_template_flat(uid_str: str) -> np.ndarray:
        uid = _cast_uid(uid_str, all_unit_ids)
        if uid is None:
            return np.zeros(templates.shape[1] * templates.shape[2])
        idx = all_unit_ids.index(uid)
        return templates[idx].flatten()

    def get_spike_train(uid_str: str) -> np.ndarray:
        uid = _cast_uid(uid_str, all_unit_ids)
        if uid is None:
            return np.array([])
        return sorting.get_unit_spike_train(unit_id=uid)

    def combined_isi_ratio(uid_a: str, uid_b: str) -> float:
        st = np.sort(np.concatenate([get_spike_train(uid_a), get_spike_train(uid_b)]))
        return _isi_ratio_from_spike_train(st, fs)

    # ── Build blacklisted pair set ───────────────────────────────────────────
    bl_pairs: set[frozenset] = set()
    for group in (blacklist or []):
        g = [str(u) for u in group]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                bl_pairs.add(frozenset([g[i], g[j]]))

    # ── Iterative merge ──────────────────────────────────────────────────────
    merge_map = {uid: uid for uid in good_unit_ids}
    n_merges = 0
    changed = True

    while changed:
        changed = False
        groups = list(set(merge_map.values()))
        for uid_a, uid_b in itertools.combinations(groups, 2):

            # Skip pairs the user explicitly unmerged
            if frozenset([str(uid_a), str(uid_b)]) in bl_pairs:
                continue

            # Criterion a: centroid distance
            dist = np.linalg.norm(get_centroid(uid_a) - get_centroid(uid_b))
            if dist >= MERGE_DISTANCE_THRESH_UM:
                continue

            # Criterion b: template waveform similarity
            r, _ = pearsonr(get_template_flat(uid_a), get_template_flat(uid_b))
            if r < MERGE_PEARSON_THRESH:
                continue

            # Criterion c: amplitude ratio
            amp_a, amp_b = get_primary_amp(uid_a), get_primary_amp(uid_b)
            if amp_a == 0 or amp_b == 0:
                continue
            ratio = max(amp_a, amp_b) / min(amp_a, amp_b)
            if ratio >= MERGE_AMPLITUDE_RATIO_THRESH:
                continue

            # Criterion d: post-merge ISI violation
            if combined_isi_ratio(uid_a, uid_b) >= MERGE_ISI_THRESH:
                continue

            # All four criteria met → merge uid_b into uid_a
            for uid in list(merge_map.keys()):
                if merge_map[uid] == uid_b:
                    merge_map[uid] = uid_a
            print(f"  [Merge] {uid_b} → {uid_a}  "
                  f"(dist={dist:.1f}µm, r={r:.2f}, amp_ratio={ratio:.2f})")
            n_merges += 1
            changed = True
            break  # restart after each merge to keep groups consistent

    print(f"[Merge] {rec_name}: {n_merges} merge(s) across {len(good_unit_ids)} non-noise units.")
    return merge_map


# ── Matplotlib labeling GUI ────────────────────────────────────────────────────
def label_units(units: list, labels: dict, output_path: Path,
                run_merge: bool = False, review_auto: bool = False,
                run_html_review: bool = False) -> dict:
    """
    1. Auto-classify all unlabeled units (SUA / MUA / Noise).
    2. Optional GUI review of all auto-classified units (review_auto=True).
    3. Optionally run merge pass on SUA units.
    4. Open GUI for any units that remain unlabeled after step 1.
    """

    # ── Stage 1: auto-classification ────────────────────────────────────────
    print("\n── Auto-classification pass ──────────────────────────────────────")
    counts: dict = {"SUA": 0, "MUA": 0, "Noise": 0}
    auto_units = []  # (rec_name, uid, img_path, metrics_dict, decision)

    for rec_name, uid, img_path in units:
        m = load_metrics(rec_name).get(str(uid), {})

        if rec_name in labels and uid in labels[rec_name]:
            # Already labeled — queue for review if REVIEW_AUTO is on
            if review_auto:
                existing = labels[rec_name][uid]
                auto_units.append((rec_name, uid, img_path, m, existing))
            continue

        decision = auto_classify(m)
        labels.setdefault(rec_name, {})[uid] = decision
        auto_units.append((rec_name, uid, img_path, m, decision))
        counts[decision] = counts.get(decision, 0) + 1

    save_labels(labels, output_path)
    print(f"Auto-labeled:  {counts['SUA']} SUA,  {counts['MUA']} MUA,  {counts['Noise']} Noise.")

    # ── Stage 2: optional auto-label review GUI ──────────────────────────────
    if review_auto and auto_units:
        labels = review_auto_labels(auto_units, labels, output_path)

    # ── Stage 3: optional merge pass ────────────────────────────────────────
    if run_merge:
        print("── Merge pass ────────────────────────────────────────────────────")
        merge_json = output_path.parent / "unit_merge_map.json"
        existing: dict = {}
        if merge_json.exists():
            with open(merge_json) as f:
                existing = json.load(f)

        by_rec: dict = {}
        for rec_name, rec_labels in labels.items():
            good_ids = [uid for uid, lbl in rec_labels.items() if lbl in ("SUA", "MUA")]
            if good_ids:
                by_rec[rec_name] = good_ids

        if existing.get("auto"):
            merge_results = existing["auto"]
            print(f"Auto-merge results found in {merge_json.name} — skipping merge pass. "
                  f"Delete the file or clear the 'auto' section to re-run.")
        else:
            merge_results = {}
            blacklist = existing.get("blacklist", {})
            for rec_name, good_ids in by_rec.items():
                merge_results[rec_name] = run_merge_pass(
                    rec_name, good_ids,
                    blacklist=blacklist.get(rec_name, [])
                )

        combined = {
            "auto":      merge_results,
            "user":      existing.get("user", {}),
            "blacklist": existing.get("blacklist", {}),
        }
        with open(merge_json, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Merge map saved → {merge_json}\n")

    print("All units auto-classified.")

    # ── Stage 4: optional HTML review (interactive, auto-saves to disk) ──────
    if run_html_review:
        from html_review import launch_server
        launch_server(open_browser=True, sortout_folder=sortout_folder)

    return labels


# ── GMM sanity-check (standalone, operates on a SortingAnalyzer) ──────────────
def compute_and_label_units(
    sorting_analyzer,
    noise_snr_threshold=NOISE_SNR_THRESHOLD,
    noise_presence_threshold=NOISE_PRESENCE_THRESHOLD,
    sua_snr_threshold=SUA_SNR_THRESHOLD,
    sua_isi_ratio_threshold=SUA_ISI_RATIO_THRESHOLD,
    sua_firing_rate_min=SUA_FIRING_RATE_MIN,
    sua_rp_threshold=SUA_RP_THRESHOLD,
    sua_amplitude_cutoff_threshold=SUA_AMPLITUDE_CUTOFF_THRESHOLD,
    gmm_n_components=GMM_N_COMPONENTS,
    gmm_random_state=GMM_RANDOM_STATE,
    isi_threshold_ms=ISI_THRESHOLD_MS,
):
    """
    Compute quality metrics on a SortingAnalyzer, apply fixed SUA/MUA/Noise
    thresholds, then run a GMM as a sanity check.

    Writes 'unit_label', 'gmm_label', and 'gmm_mismatch' as unit properties
    directly into the SortingAnalyzer.

    Returns a summary DataFrame with all metrics plus the two label columns.
    """
    feature_cols = [
        "snr", "isi_violations_ratio", "firing_rate",
        "presence_ratio", "rp_contamination", "amplitude_cutoff",
    ]

    # --- 1. Load existing metrics, compute only what's missing ---
    qm = _load_or_compute_qm(sorting_analyzer, isi_threshold_ms=isi_threshold_ms)

    unit_ids = sorting_analyzer.unit_ids
    print(f"Metrics computed for {len(unit_ids)} units.")

    # --- 2. Fixed-threshold labeling ---
    labels_fixed = pd.Series("MUA", index=unit_ids, dtype=str)

    noise_mask = (
        (qm["snr"] < noise_snr_threshold) |
        (qm["presence_ratio"] < noise_presence_threshold)
    )
    labels_fixed[noise_mask] = "Noise"

    non_noise = ~noise_mask
    sua_mask = non_noise & (
        (qm["isi_violations_ratio"] < sua_isi_ratio_threshold) &
        (qm["snr"] >= sua_snr_threshold) &
        (qm["firing_rate"] >= sua_firing_rate_min) &
        (qm["rp_contamination"] < sua_rp_threshold) &
        (qm["amplitude_cutoff"] < sua_amplitude_cutoff_threshold)
    )
    labels_fixed[sua_mask] = "SUA"

    print(f"\nFixed threshold results:\n{labels_fixed.value_counts()}")

    # --- 3. GMM ---
    X = qm[feature_cols].copy()
    for col in ["snr", "firing_rate", "isi_violations_ratio", "rp_contamination"]:
        X[col] = np.log1p(X[col].clip(lower=0))
    X = X.fillna(X.median())

    X_scaled = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(
        n_components=gmm_n_components,
        covariance_type="full",
        random_state=gmm_random_state,
        n_init=5,
    )
    gmm.fit(X_scaled)
    gmm_cluster_ids = gmm.predict(X_scaled)

    # --- 4. Map GMM clusters to SUA / MUA / Noise by median metrics ---
    cluster_medians = pd.DataFrame({
        "cluster":        range(gmm_n_components),
        "median_snr":     [np.median(qm["snr"][gmm_cluster_ids == k]) for k in range(gmm_n_components)],
        "median_isi":     [np.median(qm["isi_violations_ratio"][gmm_cluster_ids == k]) for k in range(gmm_n_components)],
        "median_presence":[np.median(qm["presence_ratio"][gmm_cluster_ids == k]) for k in range(gmm_n_components)],
    })
    print(f"\nGMM cluster medians:\n{cluster_medians}")

    def _cluster_label(row):
        if row["median_snr"] == cluster_medians["median_snr"].min():
            return "Noise"
        if (row["median_snr"] == cluster_medians["median_snr"].max() and
                row["median_isi"] == cluster_medians["median_isi"].min()):
            return "SUA"
        return "MUA"

    cluster_label_map = {row["cluster"]: _cluster_label(row) for _, row in cluster_medians.iterrows()}
    labels_gmm = pd.Series(
        [cluster_label_map[c] for c in gmm_cluster_ids],
        index=unit_ids, dtype=str,
    )
    print(f"\nGMM label distribution:\n{labels_gmm.value_counts()}")

    # --- 5. Flag mismatches ---
    mismatch = labels_fixed != labels_gmm
    print(f"\nMismatched units ({mismatch.sum()} / {len(unit_ids)}):")
    if mismatch.any():
        print(qm[mismatch][feature_cols].assign(
            fixed_label=labels_fixed[mismatch],
            gmm_label=labels_gmm[mismatch],
        ))

    # --- 6. Write back as unit properties ---
    sorting_analyzer.set_unit_property(key="unit_label",   values=labels_fixed.values)
    sorting_analyzer.set_unit_property(key="gmm_label",    values=labels_gmm.values)
    sorting_analyzer.set_unit_property(key="gmm_mismatch", values=mismatch.values)
    print("\nUnit properties written: 'unit_label', 'gmm_label', 'gmm_mismatch'")

    summary = qm[feature_cols].copy()
    summary["unit_label"]   = labels_fixed.values
    summary["gmm_label"]    = labels_gmm.values
    summary["gmm_mismatch"] = mismatch.values
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Sortout folder: {sortout_folder}")
    units = discover_units(sortout_folder)
    print(f"Found {len(units)} unit images across recordings.")

    if OVERWRITE:
        labels = {}
        print("OVERWRITE=True — discarding existing labels, re-classifying from scratch.")
    else:
        labels = load_labels(output_json)
        already = sum(len(v) for v in labels.values())
        if already:
            print(f"Resuming — {already} units already labeled, "
                  f"{len(units) - already} remaining.")

    labels = label_units(units, labels, output_json,
                         run_merge=RUN_MERGE, review_auto=REVIEW_AUTO,
                         run_html_review=LAUNCH_HTML_REVIEW)
    print(f"\nLabels saved → {output_json}")