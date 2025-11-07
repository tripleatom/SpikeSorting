# rec2nwb & Spikesorting Toolkit

A lightweight pipeline for converting Intan (*.rhd*) and SpikeGadget (*.mda*) recordings into NWB files, screening bad channels, and running a Mountainsort‑based spike sorting workflow.

SOP by Albert Li: [SOP for SpikeGadget System](https://docs.google.com/document/d/1WlWxgnbquz-oRtNNQ2TsxFg1TqT9nE_1mHZNpW5bork/edit?usp=sharing)

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tripleatom/code
   cd code
   ```
2. Install in editable mode:
   ```bash
   pip install -e .
   ```
   This makes all local modules (e.g. rec2nwb) available on your `PYTHONPATH`.

---

## 🗂️ Data Organization

Place your electrophysiology data under a top‑level `data/` directory, organized by date and animal:

```
data/
└── 2025-05-06/             # experiment date (YYYY-MM-DD)
    ├── MouseA/             # animal ID
    │   ├── session1/       # recording folder
    │   │   ├── *.rhd       # Intan files
    │   │   └── *.mda       # SpikeGadget files
    └── RatB/
        └── session2/
            ├── *.rhd
            └── *.mda
```

---

## 📁 Project Structure

### `rec2nwb/`

- **`screen_bad_ch.py`**  
  Interactive GUI for reviewing traces, marking bad channels, and saving results.  
  ![Bad channel screening](images/bad_channel.png)

- **`read_intan.py`**  
  Convert Intan `.rhd`/`.rhs` recordings to NWB.

- **`read_spikegadget.py`**  
  Convert SpikeGadget `.mda` recordings to NWB.

### `spikesorting/`

- Full spike‑sorting pipeline:
  1. Run Mountainsort on NWB-extracted data.
  2. Export to Phy for manual curation.
  3. Re-import curated clusters back into NWB.

---

## ⚙️ Usage Examples

### 1. Screen bad channels
```bash
python -m rec2nwb.screen_bad_ch   --data-folder data/2025-05-06/MouseA/session1   --impedance-file data/2025-05-06/MouseA/session1/impedance.csv
```

### 2. Convert Intan → NWB
```bash
python -m rec2nwb.read_intan   --input-folder data/2025-05-06/MouseA/session1   --output-file outputs/MouseA_session1.nwb
```

### 3. Run spike sorting
```bash
python -m spikesorting.run_sorting   --nwb-file outputs/MouseA_session1.nwb
```

---

## 🙋‍♂️ Contributing

1. Fork the repo & create a feature branch  
2. Write tests & ensure all CI checks pass  
3. Open a pull request describing your changes

---

## 📄 License

MIT © Xiaorong Zhang / Luan Lab
