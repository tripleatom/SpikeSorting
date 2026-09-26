# Probe maps

The pipeline prefers `<device_type>.probeinterface.json`, then a standard
ProbeInterface `<device_type>.json`, over a matching CSV. `device_type` may also
be an explicit JSON filename or absolute path when configuring direct sorting.
Keep using the existing device names in the GUI, configuration, and command line.
These JSON files use the standard ProbeInterface format and can be opened with
`probeinterface.read_probeinterface(path)`, which returns a `ProbeGroup`.

| Device name | Contact shape | Side length |
| --- | --- | --- |
| `4shank16intan` | square | 15 µm |
| `4shank32` | square | 20 µm |
| `4shank32intan` | square | 20 µm |
| `4shank32rp` | square | 20 µm |
| `8shank32` | square | 20 µm |

Contact count, shank IDs, and acquisition wiring are preserved from the
corresponding CSVs. Coordinates are preserved except for the specified
`8shank32` offset below. The name does not determine the contact count: for
example, the existing `4shank16intan` map contains 128 rows, all retained.
`device_channel_indices` are zero-based acquisition positions (the original CSV
row numbers), not hardware labels. Original hardware labels are retained as
contact annotations, including the one-based Ripple labels. Contact IDs remain
stable when bad channels are removed; sliced probes are rewired to the sliced
recording's channel positions.

`8shank32` repeats coordinates between shanks 0–3 and 4–7 in the original CSV.
Its JSON contains one eight-shank probe, with the requested +1,000 µm x offset
applied to shanks 4–7. Shank x positions are 0, 300, 600, 900, 1,000, 1,300,
1,600, and 1,900 µm. Y coordinates and wiring are unchanged.

The converted devices use their ProbeInterface JSON files as the sole mapping
source; their superseded CSVs have been removed. The `source_csv` annotation in
each JSON records the original conversion source for provenance. Other devices,
including `4shank16` without the `intan` suffix, still use CSV geometry, with the
previous generic 6 µm-radius circular contacts for direct sorting. The older
`4shank16.json` is a Kilosort map, not ProbeInterface, and is not loaded as one.

New ProbeInterface maps can use `.probeinterface.json` or `.json`, with 2D positions
in `um`, integer shank IDs, and explicit device channel indices (`-1` denotes an
unconnected contact). Device channel indices must be unique across the group.
Use globally unique shank and contact IDs when supplying multiple probes.

NWB conversion saves contact ID, shape, JSON shape parameters, shank ID, and
original acquisition index as electrode columns. Pipeline NWB readers restore
the ProbeInterface geometry using `rec2nwb.nwb_recording.read_nwb_recording`.
Older NWBs continue to load with their existing locations; they do not gain
contact geometry retroactively.

Run regression checks from the repository root:

```console
python -m unittest rec2nwb.test_probe_mapping -v
```
