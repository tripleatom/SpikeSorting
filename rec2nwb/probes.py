"""ProbeInterface maps and compatibility with legacy acquisition-row CSV maps."""

from pathlib import Path
import json
import sys


def mapping_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "rec2nwb" / "mapping"
    return Path(__file__).resolve().parent / "mapping"


def list_device_types(directory=None):
    directory = Path(directory) if directory is not None else mapping_dir()
    names = {p.name.removesuffix(".probeinterface.json") if
                   p.name.endswith(".probeinterface.json") else p.stem
                   for pattern in ("*.csv", "*.probeinterface.json")
                   for p in directory.glob(pattern) if not p.name.startswith("._")}
    for path in directory.glob('*.json'):
        if path.name.startswith('._') or path.name.endswith('.probeinterface.json'):
            continue
        if _is_probeinterface_json(path):
            names.add(path.stem)
    return sorted(names)


def _is_probeinterface_json(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return isinstance(data, dict) and data.get('specification') == 'probeinterface'
    except (OSError, ValueError):
        return False


def load_probes(device_type):
    """Load 2D, micrometre probes; device indices are acquisition positions.

    JSON takes precedence over CSV. CSV row order is the historical wiring,
    including maps whose hardware labels are one-based (e.g. Ripple).
    """
    import numpy as np
    import pandas as pd
    from probeinterface import Probe, read_probeinterface

    requested = Path(device_type)
    if requested.suffix.lower() == '.json':
        path = requested if requested.is_absolute() else mapping_dir() / requested
        if not path.is_file():
            raise FileNotFoundError(f"Probe JSON not found: {path}")
    else:
        path = mapping_dir() / f"{device_type}.probeinterface.json"
        plain_json = mapping_dir() / f"{device_type}.json"
        if not path.exists() and plain_json.exists() and _is_probeinterface_json(plain_json):
            path = plain_json
    if path.exists():
        group = read_probeinterface(path)
        probes = group.probes
    else:
        table = pd.read_csv(mapping_dir() / f"{device_type}.csv")
        probes = []
        # Some legacy maps repeat local coordinates between shanks.
        for _, rows in table.groupby('sh', sort=False):
            probe = Probe(ndim=2, si_units="um")
            probe.set_contacts(rows[["xcoord", "ycoord"]].to_numpy(float),
                               shapes="circle", shape_params={"radius": 6.0},
                               shank_ids=rows["sh"].astype(int).astype(str).to_numpy())
            probe.set_contact_ids(rows.index.to_numpy().astype(str))
            probe.set_device_channel_indices(rows.index.to_numpy())
            probe.annotate(name=device_type, geometry_source="legacy CSV; contact size unspecified")
            probes.append(probe)
    if not probes:
        raise ValueError(f"{device_type}: empty probe group")
    for probe in probes:
        indices = probe.device_channel_indices
        if probe.ndim != 2 or probe.si_units != "um":
            raise ValueError(f"{device_type}: expected 2D coordinates in um")
        if indices is None or np.any(indices < -1):
            raise ValueError(f"{device_type}: device_channel_indices must be wired (or -1)")
        try:
            probe.shank_ids.astype(int)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{device_type}: integer shank IDs are required") from exc
    connected = np.concatenate([p.device_channel_indices[p.device_channel_indices >= 0] for p in probes])
    if len(np.unique(connected)) != len(connected):
        raise ValueError(f"{device_type}: duplicate device channel indices")
    return probes


def probe_for_channels(device_type, channel_indices):
    """Select contacts in recording order and wire to the sliced recording."""
    import numpy as np

    channels = set(int(ch) for ch in channel_indices)
    candidates = [p for p in load_probes(device_type)
                  if channels.issubset(set(p.device_channel_indices))]
    if len(candidates) != 1:
        raise ValueError(f"{device_type}: selected channels must belong to one probe")
    probe = candidates[0]
    lookup = {int(ch): i for i, ch in enumerate(probe.device_channel_indices) if ch >= 0}
    selection = np.array([lookup[int(ch)] for ch in channel_indices], dtype=int)
    if not len(selection):
        raise ValueError("Cannot attach a probe with no selected channels")
    probe = probe.get_slice(selection)
    probe.set_device_channel_indices(np.arange(len(selection)))
    return probe
