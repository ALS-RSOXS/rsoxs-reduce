"""PyHyperScattering pipeline steps: load, integrate, reduce to I(q, E).

This module imports PyHyperScattering at module load time, which is relatively
slow and emits optional-dependency warnings; import it only when a real
reduction is about to run.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import PyHyperScattering.ALS11012RSoXSLoader as als
from PyHyperScattering.PFEnergySeriesIntegrator import PFEnergySeriesIntegrator

from rsoxs_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)

# One (path, energy, polarization) record per scan file, from a metadata peek.
ScanEntry = tuple[Path, float, float]


def build_loader(cfg: ReductionConfig) -> als.ALS11012RSoXSLoader:
    """Construct the ALS 11.0.1.2 RSoXS loader from the configuration.

    Args:
        cfg: Reduction configuration.

    Returns:
        A configured loader instance.
    """
    return als.ALS11012RSoXSLoader(
        corr_mode=cfg.corr_mode,
        exposure_offset=cfg.exposure_offset,
        data_collected_after_mar2021=cfg.data_collected_after_mar2021,
        dark_subtract=cfg.dark_subtract,
        dark_pedestal=cfg.dark_pedestal,
    )


def prepare_loader(
    loader: als.ALS11012RSoXSLoader,
    file_filter: int,
    cfg: ReductionConfig,
) -> None:
    """Load the scan's darks, I0, and I1 onto the loader.

    These are scan-global and small, so they are loaded once and reused across
    all batches.

    Args:
        loader: A configured loader.
        file_filter: Scan number selecting the darks.
        cfg: Reduction configuration.
    """
    file_filter_str = str(file_filter)
    logger.info(f"Loading darks from {cfg.dark_path} for scan {file_filter_str}")
    loader.loadSampleSpecificDarks(f"{cfg.dark_path}\\", file_filter=file_filter_str)
    logger.info(f"Loading I0 from {cfg.i0_path}")
    loader.loadSampleSpecificI0(cfg.i0_path)
    logger.info(f"Loading I1 from {cfg.i1_path}")
    loader.loadSampleSpecificI1(cfg.i1_path)


def load_stack(
    loader: als.ALS11012RSoXSLoader,
    file_filter: int,
    cfg: ReductionConfig,
) -> xr.DataArray:
    """Load darks, I0, I1, and the full FITS series for one scan.

    Args:
        loader: A configured loader.
        file_filter: Scan number selecting the files to load.
        cfg: Reduction configuration.

    Returns:
        The unstacked detector image stack (system dimension expanded).

    Raises:
        RuntimeError: If loading the file series fails.
    """
    prepare_loader(loader, file_filter, cfg)
    file_filter_str = str(file_filter)
    logger.info(f"Loading FITS series from {cfg.fits_path}")
    try:
        data_stack = loader.loadFileSeries(
            cfg.fits_path,
            list(cfg.stack_dims),
            file_filter=file_filter_str,
            md_filter=cfg.md_filter,
        )
    except Exception as exc:
        logger.error(f"Failed to load file series for scan {file_filter_str}")
        raise RuntimeError(f"loadFileSeries failed for scan {file_filter_str}") from exc

    return data_stack.unstack("system")


def enumerate_scan(
    loader: als.ALS11012RSoXSLoader,
    file_filter: int,
    cfg: ReductionConfig,
) -> tuple[list[ScanEntry], list[float]]:
    """Peek at the scan's FITS headers to map each file to (energy, polarization).

    Only headers are read (no image data), so this is cheap enough to plan
    batches up front. Files failing ``md_filter`` are dropped, matching the
    behaviour of ``loadFileSeries``.

    Args:
        loader: A configured loader.
        file_filter: Scan number selecting the files.
        cfg: Reduction configuration.

    Returns:
        A tuple of the per-file entries and the sorted unique energies.
    """
    file_filter_str = str(file_filter)
    entries: list[ScanEntry] = []
    for path in sorted(cfg.fits_path.glob(f"*{file_filter_str}*")):
        if re.match(loader.file_ext, path.name) is None:
            continue
        try:
            md = loader.peekAtMd(str(path))
        except Exception as exc:
            logger.warning(f"Skipping {path.name}: could not read header ({exc})")
            continue
        if any(md.get(key) != value for key, value in cfg.md_filter.items()):
            continue
        entries.append((path, float(md["energy"]), float(md["polarization"])))

    energies = sorted({energy for _, energy, _ in entries})
    return entries, energies


def filter_scan_energies(
    entries: list[ScanEntry],
    energies: list[float],
    wanted: Sequence[float],
    decimals: int,
) -> tuple[list[ScanEntry], list[float]]:
    """Restrict scan entries to a requested set of energies (for ``--energy``).

    Matching mirrors ``analysis.select_energies``: both stored and requested
    energies are rounded to ``decimals`` before comparison.

    Args:
        entries: Per-file entries from ``enumerate_scan``.
        energies: Sorted unique energies from ``enumerate_scan``.
        wanted: Requested energies (empty means keep all).
        decimals: Decimal places used when matching.

    Returns:
        The filtered entries and their sorted unique energies.
    """
    if not wanted:
        return entries, energies

    wanted_rounded = {round(float(value), decimals) for value in wanted}
    kept = [
        entry for entry in entries if round(entry[1], decimals) in wanted_rounded
    ]
    kept_energies = sorted({entry[1] for entry in kept})

    found = {round(entry[1], decimals) for entry in kept}
    missing = [value for value in sorted(wanted_rounded) if value not in found]
    if missing:
        logger.warning(f"Requested energies not found (to {decimals} dp): {missing}")

    return kept, kept_energies


def iter_file_batches(
    entries: list[ScanEntry],
    energies: list[float],
    batch_size: int,
) -> Iterator[tuple[list[float], list[Path]]]:
    """Yield ``(batch_energies, batch_files)`` in ascending-energy batches.

    Each batch covers ``batch_size`` energies and includes every polarization
    file for those energies.

    Args:
        entries: Per-file entries from ``enumerate_scan``.
        energies: Sorted unique energies to process.
        batch_size: Number of energies per batch (must be > 0).

    Yields:
        The energies in the batch and the FITS files covering them.
    """
    for start in range(0, len(energies), batch_size):
        batch_energies = energies[start : start + batch_size]
        batch_set = set(batch_energies)
        batch_files = [path for path, energy, _ in entries if energy in batch_set]
        yield batch_energies, batch_files


def load_file_batch(
    loader: als.ALS11012RSoXSLoader,
    files: Sequence[Path],
    dims: Sequence[str],
) -> xr.DataArray:
    """Build an unstacked image stack from an explicit list of FITS files.

    This mirrors ``FileLoader.loadFileSeries``' assembly of the ``system``
    MultiIndex but for a caller-provided subset, so only one batch of images is
    resident at a time. It is the single point of coupling to the loader's
    internal stack layout.

    Args:
        loader: A configured loader with darks/I0/I1 already prepared.
        files: The FITS files for this batch.
        dims: The stacked dimensions (e.g. ``["energy", "polarization"]``).

    Returns:
        The unstacked detector image stack for this batch.

    Raises:
        RuntimeError: If no images in the batch could be loaded.
    """
    data_rows: list[xr.DataArray] = []
    dest_coords: dict[str, list[float]] = defaultdict(list)
    for path in files:
        img = loader.loadSingleImage(str(path), coords={})
        data_rows.append(img)
        for dim in dims:
            dest_coords[dim].append(img.attrs[dim])

    if not data_rows:
        raise RuntimeError("Batch contained no loadable images.")

    dest_sorted = sorted(dest_coords.items())
    values = [value for _, value in dest_sorted]
    keys = [key for key, _ in dest_sorted]
    index = pd.MultiIndex.from_arrays(values, names=keys)
    index.name = "system"

    out = xr.concat(data_rows, dim="system").assign_coords(
        {"system": ("system", index)}
    )
    out.attrs.update({"dims_unpacked": list(dims)})
    out = out.assign_coords(
        pix_x=("pix_x", np.arange(0, len(out.pix_x))),
        pix_y=("pix_y", np.arange(0, len(out.pix_y))),
    )
    return out.unstack("system")


def setup_common_qgrid(
    integrator: PFEnergySeriesIntegrator, energies: Sequence[float]
) -> None:
    """Pin the integrator's output q-grid from the full scan's energies.

    ``PFEnergySeriesIntegrator`` derives its q-grid from the median energy of
    the set it is given and caches it. Pinning it once from all energies makes
    every batch interpolate onto the same grid, so batched output is identical
    to a whole-scan run.

    Args:
        integrator: The energy-series integrator to pin.
        energies: All energies in the scan.

    Raises:
        RuntimeError: If log-ish q binning is enabled (which re-transforms the
            pinned grid per batch and would corrupt it).
    """
    if getattr(integrator, "use_log_ish_binning", False):
        raise RuntimeError(
            "Batched processing requires use_log_ish_binning=False so the "
            "pinned q-grid is not re-transformed on each batch."
        )
    energy_list = list(energies)
    integrator.setupIntegrators(energy_list)
    integrator.setupDestQ(energy_list)
    logger.info(
        f"Pinned common q-grid from {len(energy_list)} energies (median-based)."
    )


def build_integrator(cfg: ReductionConfig) -> PFEnergySeriesIntegrator:
    """Construct the energy-series azimuthal integrator.

    Args:
        cfg: Reduction configuration.

    Returns:
        A configured integrator instance.
    """
    return PFEnergySeriesIntegrator(
        maskmethod="nika",
        maskpath=str(cfg.mask_path),
        geomethod="nika",
        NIdistance=cfg.ni_distance,
        NIbcx=cfg.ni_bcx,
        NIbcy=cfg.ni_bcy,
        NIpixsizex=cfg.ni_pixsize_x,
        NIpixsizey=cfg.ni_pixsize_y,
        integration_method=cfg.integration_method,
    )


def reduce_to_iqce(
    integrator: PFEnergySeriesIntegrator,
    data: xr.DataArray,
    cfg: ReductionConfig,
) -> xr.DataArray:
    """Integrate the image stack to the chi-resolved I(q, chi, E).

    This keeps the azimuthal (``chi``) dimension so that the chi-average, chi
    slices, anisotropy, and full 2D outputs can all be derived downstream from
    a single array.

    Args:
        integrator: A configured integrator.
        data: The unstacked detector image stack.
        cfg: Reduction configuration.

    Returns:
        Reduced intensity with dimensions ``(chi, q, energy)`` for the selected
        polarization.
    """
    logger.info("Integrating image stack (this can take several minutes)")
    integrated = integrator.integrateImageStack(data)
    return integrated.sel(polarization=cfg.polarization)
