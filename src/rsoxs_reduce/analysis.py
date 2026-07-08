"""Pure reduction math: I(q,E) tabulation and the ISI integral.

These functions take/return plain xarray/pandas objects and have no dependency
on PyHyperScattering, so they are cheap to import and easy to unit-test.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)


def select_energies(
    data: xr.DataArray,
    energies: Sequence[float],
    decimals: int = 1,
) -> xr.DataArray:
    """Select the subset of a stack whose energies match a requested list.

    Both the data's energy coordinate and the requested energies are rounded to
    ``decimals`` decimal places before matching, so callers can specify energies
    to the nearest tenth of an eV without knowing the exact stored values.

    Args:
        data: Array with an ``energy`` dimension.
        energies: Requested energies (eV) to keep.
        decimals: Decimal places used when matching (default: tenths).

    Returns:
        The subset of ``data`` at the matching energies, sorted ascending.

    Raises:
        ValueError: If none of the requested energies are present in the data.
    """
    coord = np.asarray(data["energy"].values, dtype=float)
    coord_rounded = np.round(coord, decimals)
    wanted = sorted({round(float(energy), decimals) for energy in energies})

    mask = np.isin(coord_rounded, wanted)
    matched = np.sort(coord[mask])
    if matched.size == 0:
        raise ValueError(
            f"None of the requested energies {wanted} were found in the data "
            f"(available: {sorted(set(coord_rounded.tolist()))})."
        )

    found = set(coord_rounded[mask].tolist())
    missing = [energy for energy in wanted if energy not in found]
    if missing:
        logger.warning(
            f"Requested energies not found (to {decimals} dp): {missing}"
        )

    return data.sel(energy=matched)


def iqe_to_table(iqe: xr.DataArray) -> pd.DataFrame:
    """Flatten I(q, E) into a wide table with q as the first column.

    The full, shared q-grid is preserved and NaNs are left in place so every
    energy column shares one set of q-values. Energy columns are labelled
    ``R_{energy}`` with the energy rounded to the nearest tenth of an eV.

    Args:
        iqe: Reduced intensity with dimensions ``(q, energy)``.

    Returns:
        DataFrame with a leading ``q`` column followed by one ``R_{energy}``
        column per energy.
    """
    table = iqe.to_pandas()
    table.index.name = "q"
    table.columns = [f"R_{float(energy):.1f}" for energy in table.columns]
    return table.reset_index()


def compute_isi(
    iqe: xr.DataArray,
    q_min: float | None = None,
    q_max: float | None = None,
) -> pd.DataFrame:
    """Compute the integrated scattering intensity per energy.

    For each energy, points that are NaN or negative are dropped (they indicate
    reduction artifacts and are excluded from analysis), the surviving intensity
    is scaled by ``q**2``, and the result is integrated over q:

        ISI(E) = integral of I(q, E) * q**2 dq

    Args:
        iqe: Reduced intensity with dimensions ``(q, energy)``.
        q_min: Lower q bound applied before trimming, or None for no lower bound.
        q_max: Upper q bound applied before trimming, or None for no upper bound.

    Returns:
        DataFrame with columns ``energy`` and ``ISI``; energies with fewer than
        two valid points yield NaN.
    """
    q = np.asarray(iqe["q"].values, dtype=float)
    energies = np.asarray(iqe["energy"].values, dtype=float)

    q_range = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        q_range &= q >= q_min
    if q_max is not None:
        q_range &= q <= q_max

    isi_values: list[float] = []
    for energy in energies:
        intensity = np.asarray(iqe.sel(energy=energy).values, dtype=float)
        q_e = q[q_range]
        i_e = intensity[q_range]

        valid = ~np.isnan(i_e) & (i_e >= 0.0)
        q_e = q_e[valid]
        i_e = i_e[valid]

        if q_e.size < 2:
            logger.warning(
                f"Energy {energy:.2f} eV: fewer than 2 valid points after "
                f"trimming; ISI set to NaN."
            )
            isi_values.append(float("nan"))
            continue

        order = np.argsort(q_e)
        q_e = q_e[order]
        i_e = i_e[order]
        integrand = i_e * q_e**2
        isi_values.append(float(trapezoid(integrand, q_e)))

    return pd.DataFrame({"energy": energies, "ISI": isi_values})
