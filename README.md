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
| `intan2nwb.py` | Class-based Intan converter (`EphysToNWBConverter`): reads `.rhd`/`.rhs` files and writes per-shank NWB files; supports chunked writes and parallel-shank mode |
| `rec2nwb_interp.py` | SpikeGadgets `.rec` converter with PCHIP interpolation across multi-part sessions; supports prefetch-based I/O overlap and DIO extraction |
| `batch_rec2nwb.py` | Batch wrapper for `rec2nwb_interp.py`: reads `batch_config.json`, processes multiple folders sequentially, writes a timestamped run log |
| `screen_bad_ch.py` | Interactive matplotlib GUI for reviewing traces and marking bad channels |
| `preproc_func.py` | Utilities: animal ID detection, electrode probe selection, channel indexing |
| `nwb2bin.py` | Converts NWB to binary format (`.bin`) for Kilosort input |
| `trim_nwb.py` | Removes noisy/artifact time windows from an NWB file |
| `add_txt.py` | Creates an empty `.txt` sidecar file next to each `.rec` file in a folder (required by SpikeGadgets reader) |
| `load_mda.py` | Loads and visualizes raw SpikeGadget `.mda` data for inspection |
| `batch_config.json` | Folder list and per-folder parameters for batch SpikeGadgets conversion |
| `params.json` | Default recording parameters (`samplerate: 30000`, `spike_sign: -1`) |
| `device_types.json` | Maps animal IDs to electrode probe types |
| `mapping/` | Channel mapping CSV files for each supported probe configuration |
| `geom/` | Electrode geometry CSV files |
| `utils/` | Shared helpers: `file_io.py`, `electrode.py`, `nwb_helpers.py` |
| `process_func/` | Recording-format-specific processing (e.g. `DIO.py` for digital input extraction) |

### `spikesorting/`

| File | Description |
| ---- | ----------- |
| `1_ms_sorting.py` | Mountainsort5 spike sorting: loads NWB(s), preprocesses, runs sorter |
| `2_export_phy.py` | Exports sorting results to Phy format for manual curation |
| `3_plot_curated_summary.py` | Generates summary plots for curated units |
| `curation_lazy.py` | matplotlib GUI for rapid unit labeling: shows unit summary images one-by-one with Good / MUA / Noise buttons; caches firing rates from the SortingAnalyzer and saves labels to `unit_labels.json` |
| `sort_kilo.py` | KiloSort4 runner that reads an NWB file in-memory via a lazy wrapper (no binary written to disk); selects the least-used GPU automatically |
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
        +-- Intan (.rhd/.rhs) ---------> rec2nwb/intan2nwb.py
        |
        +-- SpikeGadgets (.rec) -------> rec2nwb/rec2nwb_interp.py
        |   (batch: batch_rec2nwb.py)     -> one NWB file per shank
        |
        v
[1] Screen bad channels        rec2nwb/screen_bad_ch.py
        |
        v
[2] Convert to NWB             (see above)
        |
        v
[3] (Optional) Trim NWB        rec2nwb/trim_nwb.py
        |                      -> removes artifact epochs
        v
[4] Spike sorting              spikesorting/1_ms_sorting.py  (Mountainsort5)
        |                   or spikesorting/sort_kilo.py      (KiloSort4)
        |                      -> CAR, bandpass 300-6000 Hz, artifact removal
        v
[5] Export to Phy              spikesorting/2_export_phy.py
        |
        v
[6] Manual curation            Phy GUI (external)
        |                   or spikesorting/curation_lazy.py  (quick GUI labeling)
        v
[7] Plot curated summary       spikesorting/3_plot_curated_summary.py
```

---

## Usage

### 0. One window for the whole daily run (SpikeGadgets)

The three steps run on every day's recording — DIO gap export, NWB conversion,
MountainSort5 — are wrapped in a single GUI:

```bash
python pipeline_gui.py
```

Pick the recording folder once; animal ID, device type and shank list are filled
in from `device_types.json` and the probe map, then shared by all three steps.
Per-step options (trodesexport path, electrode location, `sorter_params`, ...)
live on their own tabs, and any subset of the steps can be run. **Check setup**
reports what will be read and written before anything starts.

Steps 2 and 3 run as child processes under the interpreter named in
*Python (steps 2, 3)* — point it at the conda env with spikeinterface +
mountainsort5 if the window was started from a different one. Each run writes
`pipeline_logs/<timestamp>_<session>/` with the log and the exact JSON configs
used; widget values are remembered in `pipeline_gui_settings.json`.

The steps are equivalent to running `rec2nwb/trodes_dio_gui.py`,
`rec2nwb/rec2nwb_interp.py` and `spikesorting/MsSorting.py` by hand, which is
still supported — see below.

### 1. Screen bad channels

Run the interactive GUI to mark bad or dead channels before conversion:

```bash
python rec2nwb/screen_bad_ch.py
```

On launch you will be prompted to select the recording folder and probe type.

### 2. Convert recording to NWB

**Intan (`.rhd` / `.rhs`)** — run the interactive converter, which prompts for folder path, shanks, chunk size, and optionally processes all shanks in a single pass:

```bash
python rec2nwb/intan2nwb.py
```

**SpikeGadgets (`.rec`)** — single session:

```bash
python rec2nwb/rec2nwb_interp.py
```

**SpikeGadgets — batch** — edit `rec2nwb/batch_config.json` with the list of folders and parameters, then:

```bash
python rec2nwb/batch_rec2nwb.py
```

A timestamped log file (`batch_run_YYYYMMDD_HHMMSS.txt`) is written alongside the script.

> **SpikeGadgets note:** some readers require a `.txt` sidecar next to each `.rec` file. Run `python rec2nwb/add_txt.py <folder>` to create them if missing.

### 3. (Optional) Trim noisy regions

Edit the time window in `trim_nwb.py` and run:

```bash
python rec2nwb/trim_nwb.py
```

### 4. Run spike sorting

**Option A — Mountainsort5:** edit `spikesorting/sorting_files.json` to specify recordings, shanks, and sorter parameters, then:

```bash
python spikesorting/1_ms_sorting.py
```

**Option B — KiloSort4:** set `nwb_path`, `sortout`, and `kilosort_params` at the top of the script, then run it as a notebook (`.py` with `#%%` cells) or plain script. Data is streamed directly from the NWB file with no intermediate binary:

```bash
python spikesorting/sort_kilo.py
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

### 6. Curate units

**Option A — Phy GUI** (standard manual curation, external tool)

**Option B — Lazy curation GUI** — review unit summary images directly in matplotlib without opening Phy. Set `sortout_folder` at the top of the script, then:

```bash
python spikesorting/curation_lazy.py
```

Click **Good**, **MUA**, or **Noise** for each unit. Labels are saved incrementally to `unit_labels.json` in the sortout folder. The firing rate (from the SortingAnalyzer) is shown in the title. Use **Back** / **Skip** to navigate. Already-labeled units are skipped on re-run.

### 7. Plot curated summary

After curation in Phy, generate per-unit summary plots:

```bash
python spikesorting/3_plot_curated_summary.py
```

Output PNGs are saved to `curated_units/` in the sort output folder.

---

## Configuration

### `rec2nwb/device_types.json`

Maps animal IDs to probe configurations (auto-updated when you run the converter scripts):

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
