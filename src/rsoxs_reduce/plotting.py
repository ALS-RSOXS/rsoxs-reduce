"""Plot generation: ISI curve, I-vs-q overlays/waterfalls, and 2D detector frames.

Uses the non-interactive ``Agg`` backend so the CLI runs without a display.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

logger = logging.getLogger(__name__)

DETECTOR_VMIN: float = 1e-3
DETECTOR_VMAX: float = 1e5


def _save(fig: "plt.Figure", path: Path, dry_run: bool, dpi: int = 150) -> None:
    """Save (or, in dry-run, skip) a figure and always close it."""
    if dry_run:
        logger.info(f"[dry-run] Would save plot: {path}")
        plt.close(fig)
        return
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {path}")


def plot_isi(
    isi_df: pd.DataFrame, path: Path, dpi: int = 150, dry_run: bool = False
) -> None:
    """Plot integrated scattering intensity versus energy.

    Args:
        isi_df: DataFrame with ``energy`` and ``ISI`` columns.
        path: Destination image path.
        dpi: Output resolution.
        dry_run: If True, do not write the file.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(isi_df["energy"], isi_df["ISI"], marker="o", markersize=3, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"ISI = $\int I(q)\,q^{2}\,dq$")
    ax.set_title("Integrated scattering intensity")
    _save(fig, path, dry_run, dpi=dpi)


def plot_iq_curves(
    iqe: xr.DataArray,
    path: Path,
    scale: str = "raw",
    mode: str = "overlay",
    dry_run: bool = False,
    waterfall_factor: float = 10.0,
    colormap: str = "viridis",
    dpi: int = 150,
) -> None:
    """Plot I-vs-q for every energy, either overlaid or waterfalled.

    NaN and non-positive points are dropped per energy so the log axes are
    well-defined.

    Args:
        iqe: Reduced intensity with dimensions ``(q, energy)``.
        path: Destination image path.
        scale: ``"raw"`` for I, or ``"q2"`` for the q**2-scaled intensity.
        mode: ``"overlay"`` for shared axes, or ``"waterfall"`` for a per-energy
            multiplicative offset (low to high energy).
        dry_run: If True, do not write the file.
        waterfall_factor: Multiplicative offset between successive curves.
        colormap: Matplotlib colormap used to color curves by energy.
        dpi: Output resolution.
    """
    energies = np.sort(np.asarray(iqe["energy"].values, dtype=float))
    q = np.asarray(iqe["q"].values, dtype=float)
    norm = Normalize(vmin=float(energies.min()), vmax=float(energies.max()))
    cmap = plt.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for index, energy in enumerate(energies):
        intensity = np.asarray(iqe.sel(energy=energy).values, dtype=float)
        values = intensity * q**2 if scale == "q2" else intensity
        valid = ~np.isnan(values) & (values > 0.0)
        if not valid.any():
            continue
        q_valid = q[valid]
        y_valid = values[valid]
        if mode == "waterfall":
            y_valid = y_valid * (waterfall_factor**index)
        ax.plot(q_valid, y_valid, color=cmap(norm(energy)), linewidth=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q (PyHyperScattering units)")
    ylabel = r"$I \cdot q^{2}$" if scale == "q2" else "I"
    if mode == "waterfall":
        ylabel += " (offset)"
    ax.set_ylabel(ylabel)

    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax)
    colorbar.set_label("Energy (eV)")

    scale_label = "q² scaled" if scale == "q2" else "raw"
    ax.set_title(f"I vs q — {mode}, {scale_label}")
    _save(fig, path, dry_run, dpi=dpi)


def save_detector_frames(
    data: xr.DataArray,
    out_dir: Path,
    polarization: str = "0",
    vmin: float = DETECTOR_VMIN,
    vmax: float = DETECTOR_VMAX,
    dpi: int = 72,
    dry_run: bool = False,
) -> None:
    """Save a low-resolution LogNorm image of the detector at each energy.

    Args:
        data: Unstacked detector image stack with an ``energy`` dimension.
        out_dir: Directory to write the per-energy PNGs into.
        polarization: Polarization coordinate to select, if present.
        vmin: Lower bound of the log color scale.
        vmax: Upper bound of the log color scale.
        dpi: Output resolution (kept low for quick previews).
        dry_run: If True, do not create the directory or write files.
    """
    out_dir = Path(out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    energies = np.unique(np.asarray(data["energy"].values, dtype=float))
    for energy in energies:
        try:
            frame = data.sel(energy=energy, method="nearest")
            if "polarization" in frame.coords:
                frame = frame.sel(polarization=polarization)
            fig, ax = plt.subplots(figsize=(4.0, 4.0))
            frame.plot(ax=ax, norm=LogNorm(vmin=vmin, vmax=vmax), add_colorbar=True)
            ax.set_title(f"E = {energy:.1f} eV")
            _save(fig, out_dir / f"E{energy:.1f}eV.png", dry_run, dpi=dpi)
        except Exception as exc:
            logger.warning(f"Skipping 2D frame for energy {energy}: {exc}")
            continue
