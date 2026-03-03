# rec2nwb & Spikesorting Toolkit

A pipeline for converting Intan (`.rhd`/`.rhs`) and SpikeGadget (`.rec`) recordings into NWB files, screening bad channels, and running a Mountainsort5-based spike sorting workflow.

SOP by Albert Li: [SOP for SpikeGadget System](https://docs.google.com/document/d/1WlWxgnbquz-oRtNNQ2TsxFg1TqT9nE_1mHZNpW5bork/edit?usp=sharing)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tripleatom/code
   cd SpikeSorting
   ```

2. Create and activate the conda environment:
   ```bash
   conda create --name spikesorting --file requirements.txt
   conda activate spikesorting
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

---

## Data Organization

Place electrophysiology data under a top-level directory organized by animal and session:

```
ephys/
└── sleep/
    ├── CnL42SG/                          # animal ID
    │   └── CnL42SG_20251115_133046.rec   # SpikeGadget recording
    └── CoI10/
        └── session1/
            └── *.rhd                      # Intan recording files
```

Output from spike sorting is saved under:

```
sortout/
└── [ANIMAL_ID]/
    └── [ANIMAL_ID]_[SESSION]/
        └── shank[N]/
            └── sorting_results_[TIMESTAMP]/
                └── phy/                  # Phy-format export for curation
```

---

## Project Structure

### `rec2nwb/`

| File | Description |
| ---- | ----------- |
| `read_raw_ephys.py` | Main converter: reads Intan or SpikeGadget recordings and writes per-shank NWB files |
| `read_raw_ephys_parallel.py` | Parallel batch version of the above |
| `screen_bad_ch.py` | Interactive matplotlib GUI for reviewing traces and marking bad channels |
| `preproc_func.py` | Utilities: animal ID detection, electrode probe selection, channel indexing |
| `nwb2bin.py` | Converts NWB to binary format (`.bin`) for Kilosort input |
| `trim_nwb.py` | Removes noisy/artifact time windows from an NWB file |
| `load_mda.py` | Loads and visualizes raw SpikeGadget `.mda` data for inspection |
| `params.json` | Default recording parameters (`samplerate: 30000`, `spike_sign: -1`) |
| `device_types.json` | Maps animal IDs to electrode probe types |
| `mapping/` | Channel mapping CSV files for each supported probe configuration |
| `geom/` | Electrode geometry CSV files |

### `spikesorting/`

| File | Description |
| ---- | ----------- |
| `1_ms_sorting.py` | Mountainsort5 spike sorting: loads NWB(s), preprocesses, runs sorter |
| `2_export_phy.py` | Exports sorting results to Phy format for manual curation |
| `3_plot_curated_summary.py` | Generates summary plots for curated units |
| `artifact_utils.py` | Per-channel artifact detection and removal |
| `ss_proc_func.py` | Utilities: sortout folder selection |
| `Timer.py` | Simple elapsed-time helper |
| `sorting_files.json` | Mountainsort5 run configuration (recordings, shanks, sorter params) |
| `sorting_files_kilosort.json` | Kilosort4 run configuration |
| `phy_files.json` | Phy export configuration |

---

## Pipeline

```
Raw data (.rhd / .rec)
        |
        v
[1] Screen bad channels        rec2nwb/screen_bad_ch.py
        |
        v
[2] Convert to NWB             rec2nwb/read_raw_ephys.py
        |                      -> one NWB file per shank
        v
[3] (Optional) Trim NWB        rec2nwb/trim_nwb.py
        |                      -> removes artifact epochs
        v
[4] Spike sorting              spikesorting/1_ms_sorting.py
        |                      -> CAR, bandpass 300-6000 Hz,
        |                         artifact removal, Mountainsort5
        v
[5] Export to Phy              spikesorting/2_export_phy.py
        |
        v
[6] Manual curation            Phy GUI (external)
        |
        v
[7] Plot curated summary       spikesorting/3_plot_curated_summary.py
```

---

## Usage

### 1. Screen bad channels

Run the interactive GUI to mark bad or dead channels before conversion:

```bash
python rec2nwb/screen_bad_ch.py
```

On launch you will be prompted to select the recording folder and probe type.

### 2. Convert recording to NWB

Edit the paths in `read_raw_ephys.py` or run it directly. It auto-detects the recording format (Intan vs SpikeGadget) and outputs one NWB file per shank.

```bash
python rec2nwb/read_raw_ephys.py
```

For batch conversion of multiple recordings in parallel:

```bash
python rec2nwb/read_raw_ephys_parallel.py
```

### 3. (Optional) Trim noisy regions

Edit the time window in `trim_nwb.py` and run:

```bash
python rec2nwb/trim_nwb.py
```

### 4. Run Mountainsort5

Edit `spikesorting/sorting_files.json` to specify recordings, shanks, and sorter parameters, then:

```bash
python spikesorting/1_ms_sorting.py
```

#### `sorting_files.json` — global parameters

```json
{
    "sorter_name": "mountainsort5",
    "sorter_params": {
        "scheme": "2",
        "detect_threshold": 5.5,
        "detect_sign": 0,
        "npca_per_channel": 3
    },
    "recordings": [
        {
            "path": "\\\\server\\ephys\\sleep\\CnL42SG\\CnL42SG_20251115_133046.rec",
            "shanks": [0, 1, 2, 3],
            "animal_id": "CnL42SG"
        }
    ]
}
```

#### `sorting_files.json` — per-recording parameters

```json
{
    "sorter_name": "mountainsort5",
    "recordings": [
        {
            "path": "\\\\server\\ephys\\sleep\\CnL42SG\\CnL42SG_20251115_133046.rec",
            "shanks": [0, 1, 2, 3],
            "animal_id": "CnL42SG",
            "sorter_params": { "scheme": "2", "detect_threshold": 5.5, "detect_sign": 0 }
        },
        {
            "path": "\\\\server\\ephys\\sleep\\CnL39SG\\CnL39SG_20251102_210043.rec",
            "shanks": [0],
            "animal_id": "CnL39SG",
            "sorter_params": { "scheme": "1", "detect_threshold": 4.5, "detect_sign": -1 }
        }
    ]
}
```

Multiple recordings listed together are concatenated before sorting.

### 5. Export to Phy

Edit `spikesorting/phy_files.json` to point to the sorting output, then:

```bash
python spikesorting/2_export_phy.py
```

### 6. Plot curated summary

After curation in Phy, generate per-unit summary plots:

```bash
python spikesorting/3_plot_curated_summary.py
```

Output PNGs are saved to `curated_units/` in the sort output folder.

---

## Configuration

### `rec2nwb/device_types.json`

Maps animal IDs to probe configurations (auto-updated when you run `read_raw_ephys.py`):

```json
{
    "CnL42SG": "8shank32",
    "CnL39SG": "4shank16",
    "CoI10":   "pin32LinearIntan"
}
```

### `rec2nwb/params.json`

Default recording parameters used by SpikeGadget conversions:

```json
{
    "samplerate": 30000,
    "spike_sign": -1
}
```

---

## Supported Probe Types

Probe mapping files live in `rec2nwb/mapping/`. Currently supported configurations include:

- `4shank16` / `4shank16intan` — 4-shank, 16 ch/shank
- `4shank32` / `4shank32intan` — 4-shank, 32 ch/shank
- `8shank32` — 8-shank, 32 ch/shank
- `1shank128` — single-shank, 128 channels
- `pin32` variants — 32-channel single-shank Intan probes

---

## License

MIT © Xiaorong Zhang / Luan Lab
