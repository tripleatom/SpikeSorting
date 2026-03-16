import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

# ── Configuration ──────────────────────────────────────────────────────────────
sortout_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260304")
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

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.12)

    def show_current():
        ax.clear()
        rec_name, uid, img_path = todo[state["idx"]]
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            f"[{rec_name}]  unit {uid}    ({state['idx'] + 1}/{len(todo)})",
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