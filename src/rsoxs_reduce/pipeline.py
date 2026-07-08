"""PyHyperScattering pipeline steps: load, integrate, reduce to I(q, E).

This module imports PyHyperScattering at module load time, which is relatively
slow and emits optional-dependency warnings; import it only when a real
reduction is about to run.
"""

import logging

import xarray as xr

import PyHyperScattering.ALS11012RSoXSLoader as als
from PyHyperScattering.PFEnergySeriesIntegrator import PFEnergySeriesIntegrator

from rsoxs_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)


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


def load_stack(
    loader: als.ALS11012RSoXSLoader,
    file_filter: int,
    cfg: ReductionConfig,
) -> xr.DataArray:
    """Load darks, I0, I1, and the FITS series for one scan.

    Args:
        loader: A configured loader.
        file_filter: Scan number selecting the files to load.
        cfg: Reduction configuration.

    Returns:
        The unstacked detector image stack (system dimension expanded).

    Raises:
        RuntimeError: If loading the file series fails.
    """
    file_filter_str = str(file_filter)
    logger.info(f"Loading darks from {cfg.dark_path} for scan {file_filter_str}")
    loader.loadSampleSpecificDarks(f"{cfg.dark_path}\\", file_filter=file_filter_str)

    logger.info(f"Loading I0 from {cfg.i0_path}")
    loader.loadSampleSpecificI0(cfg.i0_path)
    logger.info(f"Loading I1 from {cfg.i1_path}")
    loader.loadSampleSpecificI1(cfg.i1_path)

    logger.info(f"Loading FITS series from {cfg.fits_path}")
    try:
        data_stack = loader.loadFileSeries(
            cfg.fits_path,
            ["energy", "polarization"],
            file_filter=file_filter_str,
            md_filter=cfg.md_filter,
        )
    except Exception as exc:
        logger.error(f"Failed to load file series for scan {file_filter_str}")
        raise RuntimeError(f"loadFileSeries failed for scan {file_filter_str}") from exc

    return data_stack.unstack("system")


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


def reduce_to_iqe(
    integrator: PFEnergySeriesIntegrator,
    data: xr.DataArray,
    cfg: ReductionConfig,
) -> xr.DataArray:
    """Integrate the image stack and average over chi to obtain I(q, E).

    Args:
        integrator: A configured integrator.
        data: The unstacked detector image stack.
        cfg: Reduction configuration.

    Returns:
        Reduced intensity with dimensions ``(q, energy)`` for the selected
        polarization.
    """
    logger.info("Integrating image stack (this can take several minutes)")
    integrated = integrator.integrateImageStack(data)
    return integrated.sel(polarization=cfg.polarization).mean(dim="chi")
