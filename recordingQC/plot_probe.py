"""Plot probe layout from probeinterface JSON sidecars (all shanks in a folder)."""
from pathlib import Path
import matplotlib.pyplot as plt
from probeinterface import read_probeinterface
from probeinterface.plotting import plot_probe


def plot_folder_probes(folder: str | Path, pattern: str = "*sh*_probe.json",
                       save: bool = True, show: bool = False):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No probe JSONs matching {pattern} in {folder}")

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, pj in zip(axes, files):
        pg = read_probeinterface(pj)
        probe = pg.probes[0]
        plot_probe(probe, ax=ax, with_contact_id=True,
                   contacts_kargs={"alpha": 0.7})
        ax.set_title(pj.stem.replace("_probe", ""), fontsize=9)
        ax.set_xlabel("x (µm)")

    axes[0].set_ylabel("y (µm)")
    fig.suptitle(folder.name)
    fig.tight_layout()

    if save:
        out = folder / f"{folder.name}_probe_layout.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    folder = r"\\10.129.151.108\xieluanlabs\xl_cl\V1Tuning\head_fixed\250912\CnL39\CnL39_250912_171515"
    plot_folder_probes(folder)
