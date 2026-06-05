"""Convert NWB recording to int16 DAT plus rich sidecars for re-loading & analysis.

Outputs (next to the .dat):
    <stem>.dat              channel-interleaved int16 binary (Kilosort/Phy convention)
    <stem>_meta.json        full metadata: gain/offset, fs, dtype, channel ids, segments, etc.
    <stem>_probe.csv        per-channel: id, x, y, [z], impedance, group
    <stem>_probe.json       probeinterface JSON (load with `probeinterface.read_probeinterface`)
    params.py               Phy/Kilosort-compatible parameter file
    load_dat.py             tiny helper showing how to memmap & scale to µV
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from spikeinterface import extractors as se
from spikeinterface.core import write_binary_recording
from spikeinterface.preprocessing import unsigned_to_signed


def _get_property(rec, *keys):
    for k in keys:
        try:
            v = rec.get_property(k)
        except Exception:
            v = None
        if v is not None:
            return np.asarray(v)
    return None


def _to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def nwb_to_dat(nwb_file: str | Path,
               out_dir: str | Path | None = None,
               dtype: str = "int16",
               n_jobs: int = 4,
               chunk_duration: str = "1s",
               skip_dat_if_exists: bool = True,
               contact_shape: str = "square",
               contact_size_um: float = 15.0):
    nwb_file = Path(nwb_file)
    out_dir = Path(out_dir) if out_dir is not None else nwb_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = se.read_nwb_recording(str(nwb_file))
    src_dtype = np.dtype(rec.get_dtype())
    if src_dtype.kind == "u" and dtype == "int16":
        # Re-center unsigned data so the resulting int16 .dat has offset_to_uV = 0.
        print(f"  source dtype is {src_dtype}; applying unsigned_to_signed")
        old_gains = np.asarray(rec.get_channel_gains(), dtype=np.float64)
        midpoint = 2 ** (src_dtype.itemsize * 8 - 1)  # 32768 for uint16
        rec = unsigned_to_signed(rec)
        # unsigned_to_signed doesn't always propagate gain/offset — set explicitly.
        rec.set_channel_gains(old_gains)
        rec.set_channel_offsets(np.zeros_like(old_gains))
        # Sanity: uV = signed * gain + 0 ≡ unsigned * gain + (-gain * midpoint)
        _ = midpoint  # only kept for clarity
    fs = float(rec.sampling_frequency)
    n_ch = int(rec.get_num_channels())
    n_samp = int(rec.get_num_samples())
    ch_ids = rec.get_channel_ids()

    print(f"Loaded {nwb_file.name}: {n_ch} ch, {n_samp/fs:.1f} s @ {fs:.0f} Hz, "
          f"dtype src={src_dtype} -> out={dtype}")

    stem = nwb_file.stem
    dat_path = out_dir / f"{stem}.dat"
    meta_json = out_dir / f"{stem}_meta.json"
    probe_csv = out_dir / f"{stem}_probe.csv"
    probe_json = out_dir / f"{stem}_probe.json"
    params_py = out_dir / "params.py"
    loader_py = out_dir / "load_dat.py"

    # --- write binary (skip if already on disk) ---
    if skip_dat_if_exists and dat_path.exists():
        print(f"  {dat_path.name} already exists — skipping binary write")
    else:
        print(f"Writing {dat_path.name} ({dtype})...")
        write_binary_recording(rec, file_paths=[dat_path], dtype=dtype,
                               n_jobs=n_jobs, chunk_duration=chunk_duration,
                               progress_bar=True)
    file_size = dat_path.stat().st_size

    # --- probe info ---
    try:
        locations = np.asarray(rec.get_channel_locations())
    except Exception:
        locations = np.full((n_ch, 2), np.nan)

    impedance = _get_property(rec, "impedance", "imp", "Impedance")
    group = _get_property(rec, "group", "shank", "group_name")
    inter_sample_shift = _get_property(rec, "inter_sample_shift")

    df = pd.DataFrame({"channel_id": [str(c) for c in ch_ids]})
    df["x"] = locations[:, 0] if locations.ndim == 2 else np.nan
    df["y"] = locations[:, 1] if locations.ndim == 2 and locations.shape[1] >= 2 else np.nan
    if locations.ndim == 2 and locations.shape[1] >= 3:
        df["z"] = locations[:, 2]
    if impedance is not None:
        df["impedance_ohm"] = impedance
    if group is not None:
        df["group"] = group
    df.to_csv(probe_csv, index=False)

    # probeinterface JSON (round-trippable probe geometry)
    try:
        probegroup = rec.get_probegroup()
        # Override the dummy circle/radius=1 with the real contact geometry.
        if contact_shape == "square":
            shape_params = {"width": float(contact_size_um)}
        elif contact_shape == "circle":
            shape_params = {"radius": float(contact_size_um) / 2}
        elif contact_shape == "rect":
            shape_params = {"width": float(contact_size_um),
                            "height": float(contact_size_um)}
        else:
            shape_params = {"width": float(contact_size_um)}
        for probe in probegroup.probes:
            n = probe.get_contact_count()
            probe.set_contacts(positions=probe.contact_positions,
                               shapes=[contact_shape] * n,
                               shape_params=[shape_params] * n,
                               plane_axes=probe.contact_plane_axes)
        from probeinterface import write_probeinterface
        write_probeinterface(probe_json, probegroup)
    except Exception as e:
        print(f"  (skipping probeinterface JSON: {e})")
        probe_json = None

    # --- meta JSON: everything needed to reload & analyze ---
    gains = rec.get_channel_gains()
    offsets = rec.get_channel_offsets()
    seg_lengths = [int(rec.get_num_samples(segment_index=i))
                   for i in range(rec.get_num_segments())]
    try:
        ann_src = rec.get_annotations()
    except AttributeError:
        ann_src = getattr(rec, "_annotations", {}) or {}
    annotations = {k: _to_jsonable(v) for k, v in ann_src.items()}

    meta = {
        "source_nwb": str(nwb_file),
        "dat_file": dat_path.name,
        "dtype": dtype,
        "byte_order": "little",
        "layout": "channel-interleaved (sample-major): "
                  "samples[t, ch] = file_offset(t*n_channels + ch)",
        "n_channels": n_ch,
        "n_samples_total": n_samp,
        "n_segments": rec.get_num_segments(),
        "segment_n_samples": seg_lengths,
        "sampling_rate_hz": fs,
        "duration_s": n_samp / fs,
        "file_size_bytes": file_size,
        "channel_ids": [str(c) for c in ch_ids],
        "gain_to_uV_per_channel": [float(g) for g in gains],
        "offset_to_uV_per_channel": [float(o) for o in offsets],
        "uniform_gain_to_uV": (float(gains[0]) if np.allclose(gains, gains[0]) else None),
        "uniform_offset_to_uV": (float(offsets[0]) if np.allclose(offsets, offsets[0]) else None),
        "channel_locations": locations.tolist() if locations.ndim == 2 else None,
        "channel_groups": (group.tolist() if group is not None else None),
        "channel_impedance_ohm": (impedance.tolist() if impedance is not None else None),
        "inter_sample_shift": (inter_sample_shift.tolist()
                               if inter_sample_shift is not None else None),
        "annotations": annotations,
        "probe_csv": probe_csv.name,
        "probe_json": probe_json.name if probe_json else None,
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Phy/Kilosort params.py for tool compatibility ---
    uniform_gain = meta["uniform_gain_to_uV"]
    with open(params_py, "w") as f:
        f.write(f"dat_path = r'{dat_path.name}'\n")
        f.write(f"n_channels_dat = {n_ch}\n")
        f.write(f"dtype = '{dtype}'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write(f"hp_filtered = False\n")
        if uniform_gain is not None:
            f.write(f"# uV = sample * {uniform_gain}\n")

    # --- minimal loader snippet ---
    with open(loader_py, "w") as f:
        f.write(_LOADER_TEMPLATE.format(
            dat=dat_path.name, meta=meta_json.name, n_ch=n_ch, dtype=dtype))

    print(f"Saved:\n  {dat_path}\n  {meta_json}\n  {probe_csv}"
          + (f"\n  {probe_json}" if probe_json else "")
          + f"\n  {params_py}\n  {loader_py}")
    return dat_path, meta_json, probe_csv


_LOADER_TEMPLATE = '''"""Load the .dat file back as a memmapped array and (optionally) a SpikeInterface recording."""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
META = json.loads((HERE / "{meta}").read_text())

n_ch = META["n_channels"]
dtype = np.dtype(META["dtype"])
n_samp = META["n_samples_total"]

# Memmap as (n_samples, n_channels). Element [t, c] is sample t on channel c.
data = np.memmap(HERE / "{dat}", dtype=dtype, mode="r", shape=(n_samp, n_ch))

# Convert a slice to microvolts:
gain = np.array(META["gain_to_uV_per_channel"], dtype=np.float32)
offset = np.array(META["offset_to_uV_per_channel"], dtype=np.float32)
def to_uV(slice_):
    return slice_.astype(np.float32) * gain + offset

# As a SpikeInterface recording (preserves probe + scaling):
# from spikeinterface.core import BinaryRecordingExtractor
# rec = BinaryRecordingExtractor(
#     file_paths=[HERE / "{dat}"],
#     sampling_frequency=META["sampling_rate_hz"],
#     num_channels=n_ch,
#     dtype=dtype,
#     gain_to_uV=META["gain_to_uV_per_channel"],
#     offset_to_uV=META["offset_to_uV_per_channel"],
#     channel_ids=META["channel_ids"],
# )
'''


def folder_to_dat(folder: str | Path, pattern: str = "*sh*.nwb", **kwargs):
    folder = Path(folder)
    nwb_files = sorted(folder.glob(pattern))
    if not nwb_files:
        raise FileNotFoundError(f"No NWB files matching {pattern} in {folder}")
    print(f"Converting {len(nwb_files)} NWB file(s) in {folder}")
    out = {}
    for nwb in nwb_files:
        print(f"\n=== {nwb.name} ===")
        try:
            out[nwb.name] = nwb_to_dat(nwb, **kwargs)
        except Exception as e:
            print(f"  failed: {e}")
            out[nwb.name] = None
    return out


if __name__ == "__main__":
    folder = r"\\10.129.151.108\xieluanlabs\xl_cl\V1Tuning\head_fixed\250912\CnL39\CnL39_250912_171515"
    folder_to_dat(folder)
