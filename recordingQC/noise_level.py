"""Evaluate per-channel RMS noise (µV) and power spectrum for NWB recordings."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from spikeinterface import extractors as se
import spikeinterface.preprocessing as sp


def _random_chunks(rec, n_chunks: int, chunk_size: int, seed: int = 0):
    """Concatenate n random chunks of (chunk_size) samples, shape (n_chunks*chunk_size, n_ch)."""
    rng = np.random.default_rng(seed)
    n_samp = rec.get_num_samples()
    max_start = max(n_samp - chunk_size, 1)
    starts = rng.integers(0, max_start, size=n_chunks)
    return np.concatenate(
        [rec.get_traces(start_frame=int(s), end_frame=int(s) + chunk_size,
                        return_in_uV=True) for s in starts], axis=0)


def _format_impedance(imp):
    if imp is None or (isinstance(imp, float) and np.isnan(imp)) or imp == 0:
        return ""
    if imp >= 1e6:
        return f"{imp/1e6:.2f}MΩ"
    if imp >= 1e3:
        return f"{imp/1e3:.0f}kΩ"
    return f"{imp:.0f}Ω"


def _get_impedance(rec):
    """Try common property keys; return None if absent."""
    for key in ("impedance", "imp", "Impedance"):
        try:
            vals = rec.get_property(key)
        except Exception:
            vals = None
        if vals is not None:
            return np.asarray(vals, dtype=float)
    return None


def evaluate_noise(nwb_file: str | Path,
                   freq_min: float = 300.0,
                   freq_max: float = 6000.0,
                   n_chunks: int = 20,
                   chunk_duration_s: float = 1.0,
                   psd_nperseg: int = 4096,
                   save: bool = True):
    nwb_file = Path(nwb_file)
    rec = se.read_nwb_recording(str(nwb_file))
    fs = rec.sampling_frequency
    n_ch = rec.get_num_channels()
    ch_ids = rec.get_channel_ids()
    impedance = _get_impedance(rec)
    print(f"Loaded {nwb_file.name}: {n_ch} ch, "
          f"{rec.get_num_samples()/fs:.1f} s @ {fs:.0f} Hz, "
          f"impedance={'yes' if impedance is not None else 'no'}")

    rec_f = sp.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max,
                               dtype=np.float32)

    chunk_size = int(chunk_duration_s * fs)
    traces_f = _random_chunks(rec_f, n_chunks, chunk_size)
    rms = np.sqrt(np.mean(traces_f ** 2, axis=0))

    traces_raw = _random_chunks(rec, n_chunks, chunk_size)
    f, pxx = welch(traces_raw, fs=fs, nperseg=min(psd_nperseg, traces_raw.shape[0]),
                   axis=0)

    df = pd.DataFrame({"channel_id": ch_ids, "rms_uV": rms})
    if impedance is not None:
        df["impedance_ohm"] = impedance
    print(f"median RMS = {np.median(rms):.2f} µV, "
          f"min = {rms.min():.2f}, max = {rms.max():.2f}")

    if save:
        out_dir = nwb_file.parent
        stem = nwb_file.stem
        out_csv = out_dir / f"{stem}_rms.csv"
        out_bar = out_dir / f"{stem}_rms.png"
        out_psd = out_dir / f"{stem}_psd.png"
        df.to_csv(out_csv, index=False)

        # RMS bar plot
        fig, ax = plt.subplots(figsize=(max(6, n_ch * 0.2), 5))
        ax.bar(np.arange(n_ch), rms, color="steelblue")
        ax.axhline(np.median(rms), color="red", ls="--",
                   label=f"median={np.median(rms):.1f} µV")
        ax.set_ylabel("RMS (µV)")
        ax.set_title(f"{stem} — bandpass {freq_min:.0f}-{freq_max:.0f} Hz")
        ax.set_xticks(np.arange(n_ch))
        ax.set_xticklabels([str(c) for c in ch_ids], rotation=90, fontsize=7)
        ax.legend(loc="upper right")

        if impedance is not None:
            ax.set_xlabel("")
            # Place impedance labels below the rotated channel-id tick labels
            for i, imp in enumerate(impedance):
                ax.text(i, -0.22, _format_impedance(imp),
                        ha="center", va="top", fontsize=6, rotation=90,
                        color="dimgray",
                        transform=ax.get_xaxis_transform())
            fig.text(0.5, 0.01, "channel id (top) / impedance (bottom)",
                     ha="center", fontsize=9)
            fig.subplots_adjust(bottom=0.32)
        else:
            ax.set_xlabel("channel id")
            fig.tight_layout()

        fig.savefig(out_bar, dpi=150)
        plt.close(fig)

        # Power spectrum (log-log)
        mask = f > 0
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(f[mask], pxx[mask], color="gray", alpha=0.4, lw=0.6)
        ax.loglog(f[mask], np.median(pxx[mask], axis=1), color="red", lw=1.5,
                  label="median")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (µV²/Hz)")
        ax.set_xlim(f[mask].min(), fs / 2)
        ax.set_title(f"{stem} — power spectrum (raw)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_psd, dpi=150)
        plt.close(fig)

        print(f"Saved {out_csv.name}, {out_bar.name}, {out_psd.name} in {out_dir}")

    return df, (f, pxx)


def evaluate_folder(folder: str | Path, pattern: str = "*sh*.nwb", **kwargs):
    """Run evaluate_noise on every NWB file in `folder` matching `pattern`."""
    folder = Path(folder)
    nwb_files = sorted(folder.glob(pattern))
    if not nwb_files:
        raise FileNotFoundError(f"No NWB files matching {pattern} in {folder}")
    print(f"Found {len(nwb_files)} NWB file(s) in {folder}")
    results = {}
    for nwb in nwb_files:
        print(f"\n=== {nwb.name} ===")
        try:
            results[nwb.name] = evaluate_noise(nwb, **kwargs)
        except Exception as e:
            print(f"  failed: {e}")
            results[nwb.name] = None
    return results


if __name__ == "__main__":
    folder = r"\\10.129.151.108\xieluanlabs\xl_cl\V1Tuning\head_fixed\250912\CnL39\CnL39_250912_171515"
    evaluate_folder(folder)
