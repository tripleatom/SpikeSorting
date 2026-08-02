"""
export_curated_units.py
========================
Export per-unit spike trains and metadata from a combined ``curated_analyzer``
folder produced by MsCuratedAnalyzer.py.

Unlike Centimani's export_units.py (which re-applies unit_labels.json /
unit_merge_map.json across per-recording sorting_results_* folders),
MsCuratedAnalyzer.py has already merged shanks, removed Noise units, and
applied merges into ONE combined SortingAnalyzer. This script just reads
that result and writes it out in a portable form.

Output (next to the curated_analyzer folder, i.e. in the session folder):
  <session>_curated_units_export.pkl              per-unit dict (see below)
  <session>_curated_units_export_spikevector.npy   SpikeInterface-style spike vector
  <session>_curated_units_export_spikevector.json  spike-vector metadata

Each entry in the pickle's "units" dict:
  {
    "spike_train": np.ndarray[int64]   # sample indices, this unit's own recording fs
    "n_spikes": int,
    "label": "SUA" | "MUA",
    "shank_group": int                 # 'group' property; matches shank index when
                                        # every configured shank was processed
    "position": np.ndarray | None       # unit_locations extension row, if computed
    "is_merged": bool,
  }

Usage
-----
  python export_curated_units.py --analyzer-path <path to curated_analyzer>
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from spikeinterface.core.sortinganalyzer import SortingAnalyzer

_SKIP_RECORDING = "__skip_recording__"


def export_units(analyzer) -> dict:
    """Return {unit_id: {spike_train, n_spikes, label, shank_group, position, is_merged}}."""
    unit_ids = list(analyzer.unit_ids)
    labels = analyzer.sorting.get_property("unit_label")
    groups = analyzer.sorting.get_property("group")
    is_merged = analyzer.sorting.get_property("is_merged")

    loc_ext = analyzer.get_extension("unit_locations")
    locations = loc_ext.get_data() if loc_ext is not None else None

    units = {}
    for i, uid in enumerate(unit_ids):
        spike_train = np.asarray(analyzer.sorting.get_unit_spike_train(uid), dtype=np.int64)
        units[uid] = {
            "spike_train": spike_train,
            "n_spikes": int(spike_train.size),
            "label": str(labels[i]) if labels is not None else "MUA",
            "shank_group": int(groups[i]) if groups is not None else None,
            "position": np.asarray(locations[i]) if locations is not None else None,
            "is_merged": bool(is_merged[i]) if is_merged is not None else False,
        }
    return units


def build_spike_vector(units: dict, unit_ids: list) -> np.ndarray:
    """Flat SpikeInterface-style spike vector (sample_index, unit_index, segment_index)."""
    unit_index_of = {uid: i for i, uid in enumerate(unit_ids)}
    total_spikes = sum(u["n_spikes"] for u in units.values())
    spike_vector = np.empty(
        total_spikes,
        dtype=[("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")],
    )
    cursor = 0
    for uid, unit in units.items():
        n = unit["n_spikes"]
        spike_vector["sample_index"][cursor:cursor + n] = unit["spike_train"]
        spike_vector["unit_index"][cursor:cursor + n] = unit_index_of[uid]
        spike_vector["segment_index"][cursor:cursor + n] = 0
        cursor += n
    if spike_vector.size:
        order = np.lexsort((spike_vector["unit_index"], spike_vector["sample_index"]))
        spike_vector = spike_vector[order]
    return spike_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyzer-path", type=Path, required=True,
                        help="path to a combined curated_analyzer folder")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to the curated_analyzer's parent (session) folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curated_path = args.analyzer_path
    session_path = curated_path.parent
    session_name = session_path.name
    output_dir = args.output_dir or session_path

    print(f"Loading curated analyzer: {curated_path}")
    analyzer = SortingAnalyzer.load(curated_path, recording=_SKIP_RECORDING, load_extensions=True)
    unit_ids = list(analyzer.unit_ids)
    print(f"  {len(unit_ids)} units, fs={analyzer.sampling_frequency}")

    units = export_units(analyzer)
    label_counts = {
        label: sum(1 for u in units.values() if u["label"] == label)
        for label in ("SUA", "MUA")
    }
    print(f"  labels: {label_counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pkl = output_dir / f"{session_name}_curated_units_export.pkl"
    spikevector_npy = output_dir / f"{session_name}_curated_units_export_spikevector.npy"
    spikevector_json = output_dir / f"{session_name}_curated_units_export_spikevector.json"

    payload = {
        "metadata": {
            "session_name": session_name,
            "curated_analyzer_path": str(curated_path),
            "sampling_frequency": float(analyzer.sampling_frequency),
            "n_units": len(unit_ids),
            "label_counts": label_counts,
        },
        "units": units,
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)

    spike_vector = build_spike_vector(units, unit_ids)
    np.save(spikevector_npy, spike_vector)
    with open(spikevector_json, "w", encoding="utf-8") as f:
        json.dump({
            "format": "spikeinterface_spike_vector",
            "dtype": str(spike_vector.dtype),
            "fields": list(spike_vector.dtype.names),
            "unit_index": "index into sorted(unit_ids); matches keys in the units export pickle",
            "unit_ids_in_order": [str(uid) for uid in unit_ids],
            "n_spikes": int(spike_vector.size),
            "units_export_pkl": str(output_pkl),
        }, f, indent=2)

    print(f"\nExported {len(unit_ids)} unit(s).")
    print(f"  Pickle:       {output_pkl}")
    print(f"  Spike vector: {spikevector_npy} ({spike_vector.size} spikes)")
    print(f"  Metadata:     {spikevector_json}")


if __name__ == "__main__":
    main()
