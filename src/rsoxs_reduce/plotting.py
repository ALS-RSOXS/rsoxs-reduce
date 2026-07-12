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


_SLICE_LINESTYLES: tuple[str, ...] = ("--", ":", "-.")


def plot_iq_curves(
    iqe: xr.DataArray,
    path: Path,
    scale: str = "raw",
    mode: str = "overlay",
    dry_run: bool = False,
    waterfall_factor: float = 10.0,
    colormap: str = "viridis",
    dpi: int = 150,
    slices: dict[str, xr.DataArray] | None = None,
) -> None:
    """Plot I-vs-q for every energy, either overlaid or waterfalled.

    NaN and non-positive points are dropped per energy so the log axes are
    well-defined. The chi-average is drawn as a solid line; each optional chi
    slice is overlaid in the same per-energy color but a distinct linestyle.

    Args:
        iqe: Chi-average intensity with dimensions ``(q, energy)`` (solid line).
        path: Destination image path.
        scale: ``"raw"`` for I, or ``"q2"`` for the q**2-scaled intensity.
        mode: ``"overlay"`` for shared axes, or ``"waterfall"`` for a per-energy
            multiplicative offset (low to high energy).
        dry_run: If True, do not write the file.
        waterfall_factor: Multiplicative offset between successive curves.
        colormap: Matplotlib colormap used to color curves by energy.
        dpi: Output resolution.
        slices: Optional mapping of chi-slice label to a ``(q, energy)`` array,
            each overlaid with its own linestyle.
    """
    energies = np.sort(np.asarray(iqe["energy"].values, dtype=float))
    q = np.asarray(iqe["q"].values, dtype=float)
    norm = Normalize(vmin=float(energies.min()), vmax=float(energies.max()))
    cmap = plt.get_cmap(colormap)

    # (label, array, linestyle): the chi-average first, then each slice.
    curve_sets: list[tuple[str, xr.DataArray, str]] = [("chi-average", iqe, "-")]
    for offset, (label, sliced) in enumerate(sorted((slices or {}).items())):
        curve_sets.append((label, sliced, _SLICE_LINESTYLES[offset % len(_SLICE_LINESTYLES)]))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for _label, array, linestyle in curve_sets:
        for index, energy in enumerate(energies):
            intensity = np.asarray(array.sel(energy=energy).values, dtype=float)
            values = intensity * q**2 if scale == "q2" else intensity
            valid = ~np.isnan(values) & (values > 0.0)
            if not valid.any():
                continue
            y_valid = values[valid]
            if mode == "waterfall":
                y_valid = y_valid * (waterfall_factor**index)
            ax.plot(q[valid], y_valid, color=cmap(norm(energy)), linewidth=0.8, linestyle=linestyle)

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

    # Legend mapping linestyle -> chi-slice label (color carries energy).
    if len(curve_sets) > 1:
        handles = [
            plt.Line2D([], [], color="black", linestyle=linestyle, label=label)
            for label, _array, linestyle in curve_sets
        ]
        ax.legend(handles=handles, fontsize="small", title="chi")

    scale_label = "q² scaled" if scale == "q2" else "raw"
    ax.set_title(f"I vs q — {mode}, {scale_label}")
    _save(fig, path, dry_run, dpi=dpi)


def plot_anisotropy(
    anisotropy: xr.DataArray,
    path: Path,
    colormap: str = "viridis",
    dpi: int = 150,
    dry_run: bool = False,
) -> None:
    """Plot anisotropy A vs q for every energy, overlaid, with y bounded to [-1, 1].

    Args:
        anisotropy: Anisotropy with dimensions ``(q, energy)``.
        path: Destination image path.
        colormap: Matplotlib colormap used to color curves by energy.
        dpi: Output resolution.
        dry_run: If True, do not write the file.
    """
    energies = np.sort(np.asarray(anisotropy["energy"].values, dtype=float))
    q = np.asarray(anisotropy["q"].values, dtype=float)
    norm = Normalize(vmin=float(energies.min()), vmax=float(energies.max()))
    cmap = plt.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for energy in energies:
        values = np.asarray(anisotropy.sel(energy=energy).values, dtype=float)
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        ax.plot(q[valid], values[valid], color=cmap(norm(energy)), linewidth=0.8)

    ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("q (PyHyperScattering units)")
    ax.set_ylabel("Anisotropy A")

    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax)
    colorbar.set_label("Energy (eV)")

    ax.set_title("Anisotropy vs q")
    _save(fig, path, dry_run, dpi=dpi)


def plot_integrated_anisotropy(
    int_df: pd.DataFrame, path: Path, dpi: int = 150, dry_run: bool = False
) -> None:
    """Plot the q-integrated anisotropy versus energy.

    Args:
        int_df: DataFrame with ``energy`` and ``int_A`` columns.
        path: Destination image path.
        dpi: Output resolution.
        dry_run: If True, do not write the file.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(int_df["energy"], int_df["int_A"], marker="o", markersize=3, linewidth=1.0)
    ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\int A(q)\,dq$")
    ax.set_title("Integrated anisotropy")
    _save(fig, path, dry_run, dpi=dpi)


def plot_iqchi_maps(
    iqce: xr.DataArray,
    out_dir: Path,
    vmin: float = DETECTOR_VMIN,
    vmax: float = DETECTOR_VMAX,
    colormap: str = "viridis",
    dpi: int = 150,
    dry_run: bool = False,
) -> None:
    """Save a 2D I(q, chi) heatmap at each energy.

    Args:
        iqce: Chi-resolved intensity with dimensions including ``(q, chi, energy)``.
        out_dir: Directory to write the per-energy PNGs into.
        vmin: Lower bound of the log color scale.
        vmax: Upper bound of the log color scale.
        colormap: Matplotlib colormap for the intensity.
        dpi: Output resolution.
        dry_run: If True, do not create the directory or write files.
    """
    out_dir = Path(out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    energies = np.unique(np.asarray(iqce["energy"].values, dtype=float))
    for energy in energies:
        try:
            frame = iqce.sel(energy=energy, method="nearest").transpose("chi", "q")
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            frame.plot(
                ax=ax,
                x="q",
                y="chi",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap=colormap,
                add_colorbar=True,
            )
            ax.set_xscale("log")
            ax.set_xlabel("q (PyHyperScattering units)")
            ax.set_ylabel("chi (deg)")
            ax.set_title(f"I(q, chi) — E = {energy:.1f} eV")
            _save(fig, out_dir / f"Iqchi_E{energy:.1f}eV.png", dry_run, dpi=dpi)
        except Exception as exc:
            logger.warning(f"Skipping I(q,chi) map for energy {energy}: {exc}")
            continue


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
