import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import spikeinterface as si

# ── Configuration ──────────────────────────────────────────────────────────────
sortout_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260310")
output_json = sortout_folder / "unit_labels.json"


# ── Folder discovery ──────────────────────────────────────────────────────────
def discover_units(sortout_folder: Path):
    """Return list of (recording_name, unit_id, image_path) tuples."""
    units = []
    recordings = sorted(
        [d for d in sortout_folder.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    for rec_dir in recordings:
        # find all sortingresults* folders, pick the latest (sorted by name)
        sorting_dirs = sorted(rec_dir.glob("sorting_results_*"))
        if not sorting_dirs:
            continue
        latest_sorting = sorting_dirs[-1]
        raw_units_dir = latest_sorting / "raw_units"
        if not raw_units_dir.is_dir():
            continue
        # collect unit images
        images = sorted(
            raw_units_dir.glob("unit_summary_*.png"),
            key=lambda p: int(re.search(r"unit_summary_(\d+)", p.stem).group(1)),
        )
        for img_path in images:
            uid = re.search(r"unit_summary_(\d+)", img_path.stem).group(1)
            units.append((rec_dir.name, uid, img_path))
    return units


# ── JSON load / save ──────────────────────────────────────────────────────────
def load_labels(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict, path: Path):
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


# ── Matplotlib labeling GUI ──────────────────────────────────────────────────
def label_units(units, labels, output_path):
    """Show each unit image with clickable buttons for labeling."""
    # filter out already-labeled units
    todo = []
    for rec_name, uid, img_path in units:
        if rec_name in labels and uid in labels[rec_name]:
            continue
        todo.append((rec_name, uid, img_path))

    if not todo:
        print("All units are already labeled. Nothing to do.")
        return labels

    state = {"idx": 0}
    # Cache firing-rate metrics per recording so we don't reload the analyzer
    # from disk for every unit.
    fr_cache = {}

    def get_firing_rate(rec_name, uid):
        """Return firing rate (Hz) for this unit, or None if unavailable.

        Uses only spike times and the sorter sampling frequency; does NOT rely
        on having a recording object available.

        Also prints diagnostic information to the terminal.
        """
        # Use per-recording cache
        if rec_name not in fr_cache:
            try:
                rec_dir = sortout_folder / rec_name
                print(f"[FR] Looking for sorting results in: {rec_dir}")
                sorting_dirs = sorted(rec_dir.glob("sorting_results_*"))
                if not sorting_dirs:
                    print(f"[FR] No 'sorting_results_*' folders found for {rec_name}")
                    return None
                latest_sorting = sorting_dirs[-1]
                analyzer_folder = latest_sorting / "sorting_analyzer"
                print(f"[FR] Using analyzer folder: {analyzer_folder}")
                if not analyzer_folder.is_dir():
                    print(f"[FR] Analyzer folder does not exist: {analyzer_folder}")
                    return None

                # Load existing sorting analyzer created by the sorter
                sa = si.load_sorting_analyzer(str(analyzer_folder))
                sorting = sa.sorting

                spike_counts = sorting.count_num_spikes_per_unit()
                fs = sorting.get_sampling_frequency()
                if fs is None or fs <= 0:
                    print(f"[FR] Invalid sampling frequency for {rec_name}: {fs}")
                    return None

                # Estimate recording duration from spike times only
                first_frame = None
                last_frame = None
                for unit_id in sorting.get_unit_ids():
                    st = sorting.get_unit_spike_train(unit_id=unit_id)
                    if st.size == 0:
                        continue
                    u_first = int(st[0])
                    u_last = int(st[-1])
                    first_frame = u_first if first_frame is None else min(first_frame, u_first)
                    last_frame = u_last if last_frame is None else max(last_frame, u_last)

                if first_frame is None or last_frame is None or last_frame <= first_frame:
                    print(f"[FR] Could not infer duration from spike trains for {rec_name}")
                    return None

                duration_sec = float(last_frame - first_frame) / float(fs)
                print(f"[FR] Estimated duration for {rec_name}: {duration_sec:.3f} s (from spike times)")
                if duration_sec <= 0:
                    print(f"[FR] Non-positive duration for {rec_name}, cannot compute FR")
                    return None

                unit_fr = {}
                for unit_id, count in spike_counts.items():
                    try:
                        count_int = int(count)
                    except Exception:
                        print(f"[FR] Could not cast spike count for unit {unit_id}: {count}")
                        continue
                    rate_hz = count_int / duration_sec
                    unit_fr[str(unit_id)] = rate_hz
                print(f"[FR] Computed firing rates for {len(unit_fr)} units in {rec_name}")

                fr_cache[rec_name] = unit_fr
            except Exception as e:
                print(f"[FR] Error while loading analyzer for {rec_name}: {e}")
                fr_cache[rec_name] = {}

        # Look up this particular unit
        unit_map = fr_cache.get(rec_name, {})
        fr = unit_map.get(str(uid))
        if fr is None and isinstance(uid, str) and uid.isdigit():
            fr = unit_map.get(str(int(uid)))
        if fr is None:
            print(f"[FR] No firing rate found for recording '{rec_name}', unit '{uid}'")
        else:
            try:
                print(f"[FR] Firing rate for recording '{rec_name}', unit '{uid}': {float(fr):.3f} Hz")
            except Exception:
                print(f"[FR] Firing rate for recording '{rec_name}', unit '{uid}': {fr}")
        return fr

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.12)

    def show_current():
        ax.clear()
        rec_name, uid, img_path = todo[state["idx"]]
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()

        # Try to load firing rate (Hz) for this unit from the existing
        # spikeinterface SortingAnalyzer (no need to re-run sorting).
        fr = get_firing_rate(rec_name, uid)
        if fr is not None:
            try:
                fr_str = f"{float(fr):.2f} Hz"
            except (TypeError, ValueError):
                fr_str = "null"
        else:
            fr_str = "null"

        ax.set_title(
            f"[{rec_name}]  unit {uid}    ({state['idx'] + 1}/{len(todo)})  —  FR: {fr_str}",
            fontsize=13,
        )
        fig.canvas.draw_idle()

    def make_callback(label_value):
        def callback(event):
            rec_name, uid, _ = todo[state["idx"]]
            labels.setdefault(rec_name, {})[uid] = label_value
            save_labels(labels, output_path)
            state["idx"] += 1
            if state["idx"] >= len(todo):
                print("Labeling complete!")
                plt.close(fig)
            else:
                show_current()
        return callback

    def on_back(event):
        if state["idx"] > 0:
            state["idx"] -= 1
            show_current()

    def on_skip(event):
        state["idx"] += 1
        if state["idx"] >= len(todo):
            print("Reached end (some units skipped).")
            plt.close(fig)
        else:
            show_current()

    # buttons
    btn_specs = [
        (0.15, "Good",  "lightgreen", make_callback("good")),
        (0.35, "MUA",   "khaki",      make_callback("mua")),
        (0.55, "Noise", "lightsalmon", make_callback("noise")),
        (0.72, "Back",  "lightgray",  on_back),
        (0.85, "Skip",  "whitesmoke", on_skip),
    ]
    buttons = []
    for x, text, color, cb in btn_specs:
        ax_btn = fig.add_axes([x, 0.02, 0.1, 0.05])
        btn = Button(ax_btn, text, color=color, hovercolor="lightskyblue")
        btn.on_clicked(cb)
        buttons.append(btn)  # prevent GC

    show_current()
    plt.show()
    return labels


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Sortout folder: {sortout_folder}")
    units = discover_units(sortout_folder)
    print(f"Found {len(units)} unit images across recordings.")

    labels = load_labels(output_json)
    already = sum(len(v) for v in labels.values())
    if already:
        print(f"Resuming — {already} units already labeled, {len(units) - already} remaining.")

    labels = label_units(units, labels, output_json)
    print(f"Labels saved to {output_json}")