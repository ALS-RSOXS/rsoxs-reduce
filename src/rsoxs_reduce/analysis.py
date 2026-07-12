"""Pure reduction math: I(q,E) tabulation and the ISI integral.

These functions take/return plain xarray/pandas objects and have no dependency
on PyHyperScattering, so they are cheap to import and easy to unit-test.
"""

import logging
import math
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


def iqe_to_table(iqe: xr.DataArray, prefix: str = "R") -> pd.DataFrame:
    """Flatten a ``(q, energy)`` array into a wide table with q as the first column.

    The full, shared q-grid is preserved and NaNs are left in place so every
    energy column shares one set of q-values. Energy columns are labelled
    ``{prefix}_{energy}`` with the energy rounded to the nearest tenth of an eV.

    Args:
        iqe: Reduced quantity with dimensions ``(q, energy)`` (e.g. intensity
            or anisotropy).
        prefix: Column-name prefix for the per-energy columns.

    Returns:
        DataFrame with a leading ``q`` column followed by one
        ``{prefix}_{energy}`` column per energy.
    """
    table = iqe.transpose("q", "energy").to_pandas()
    table.index.name = "q"
    table.columns = [f"{prefix}_{float(energy):.1f}" for energy in table.columns]
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


def chi_average(iqce: xr.DataArray) -> xr.DataArray:
    """Average the chi-resolved intensity over all chi to get isotropic I(q, E).

    Args:
        iqce: Chi-resolved intensity with a ``chi`` dimension.

    Returns:
        Intensity with ``chi`` removed, dimensions ``(q, energy)``.
    """
    return iqce.mean("chi")


def _chi_selector(chi: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """Return a boolean mask selecting an azimuthal wedge, handling wraparound.

    The selection reproduces PyHyperScattering's ``slice_chi`` behaviour (so
    output initially matches the wider ALS tooling) but is kept local so the
    definition can evolve independently. ``chi`` is assumed to be in degrees.

    Args:
        chi: The chi coordinate values in degrees.
        center: Wedge center angle in degrees.
        half_width: Half-width of the wedge in degrees (each side of center).

    Returns:
        A boolean mask over ``chi`` selecting the wedge.
    """
    chi_min = float(np.min(chi))
    chi_max = float(np.max(chi))
    begin = center - half_width
    end = center + half_width

    # Translate a wedge that lies entirely below/above the coordinate range
    # back into range (PyHyper cases 3 and 5).
    if begin < chi_min and end < chi_min:
        nshift = math.floor((chi_min - end) / 360) + 1
        begin += 360 * nshift
        end += 360 * nshift
    elif begin > chi_max and end > chi_max:
        nshift = math.floor((begin - chi_max) / 360) + 1
        begin -= 360 * nshift
        end -= 360 * nshift

    if begin < chi_min and end > chi_max:
        # Wedge spans the whole range (PyHyper case 1).
        logger.warning(
            f"Chi wedge {center} +/- {half_width} deg spans the full chi range "
            f"[{chi_min}, {chi_max}]; averaging over all chi."
        )
        return np.ones_like(chi, dtype=bool)
    if begin < chi_min and end < chi_max:
        # Wraps under chi_min (PyHyper case 2).
        return ((chi >= chi_min) & (chi <= end)) | (
            (chi <= chi_max) & (chi >= (chi_max - (chi_min - begin) + 1))
        )
    if end > chi_max and begin > chi_min:
        # Wraps over chi_max (PyHyper case 4).
        return ((chi <= chi_max) & (chi >= begin)) | (
            (chi >= chi_min) & (chi <= (chi_min + (end - chi_max) - 1))
        )
    # Simple in-range wedge (PyHyper case 6).
    return (chi >= begin) & (chi <= end)


def slice_chi(iqce: xr.DataArray, center: float, half_width: float) -> xr.DataArray:
    """Average I(q, chi, E) over an azimuthal wedge centered at ``center``.

    If the wedge is narrower than the chi grid it selects zero or one bins; a
    notice is logged and the reduction continues using the nearest single bin.

    Args:
        iqce: Chi-resolved intensity with a ``chi`` dimension (degrees).
        center: Wedge center angle in degrees.
        half_width: Half-width of the wedge in degrees (each side of center).

    Returns:
        Intensity averaged over the wedge, dimensions ``(q, energy)``.
    """
    chi = np.asarray(iqce["chi"].values, dtype=float)
    mask = _chi_selector(chi, center, half_width)
    count = int(mask.sum())

    if count == 0:
        nearest = int(np.argmin(np.abs((chi - center + 180.0) % 360.0 - 180.0)))
        mask = np.zeros_like(chi, dtype=bool)
        mask[nearest] = True
        logger.warning(
            f"Chi wedge {center} +/- {half_width} deg is narrower than one chi "
            f"step; continuing with the nearest single bin at chi = {chi[nearest]:g} deg."
        )
    elif count == 1:
        logger.warning(
            f"Chi wedge {center} +/- {half_width} deg spans only a single chi "
            f"step; continuing with that one bin."
        )

    return iqce.isel(chi=mask).mean("chi")


def assemble_iq_table(
    chi_avg: xr.DataArray,
    slice_items: Sequence[tuple[float, xr.DataArray]] = (),
) -> pd.DataFrame:
    """Build the wide I(q, E) table from a precomputed chi-average and slices.

    Kept separate from ``iqce_to_table`` so batched processing can assemble the
    table from per-batch pieces that were concatenated along energy, without
    holding the full chi-resolved array.

    Args:
        chi_avg: Chi-average intensity with dimensions ``(q, energy)``.
        slice_items: Sequence of ``(angle_deg, wedge)`` pairs, each ``wedge`` a
            ``(q, energy)`` array appended as ``R_{energy}_chi{angle}`` columns.

    Returns:
        DataFrame with a leading ``q`` column, the chi-average energy columns,
        and one block of columns per chi slice.
    """
    table = iqe_to_table(chi_avg)
    for angle, wedge in slice_items:
        wedge_table = iqe_to_table(wedge)
        suffix = f"_chi{float(angle):g}"
        renamed = wedge_table.drop(columns="q").rename(
            columns=lambda name, s=suffix: f"{name}{s}"
        )
        table = pd.concat([table, renamed], axis=1)
    return table


def iqce_to_table(
    iqce: xr.DataArray,
    chi_slices: Sequence[float] = (),
    chi_width: float = 5.0,
) -> pd.DataFrame:
    """Flatten chi-resolved intensity into a wide table.

    The chi-average is always included as ``R_{energy}`` columns. For each
    requested chi angle, the corresponding wedge is appended as
    ``R_{energy}_chi{angle}`` columns, sharing the same q grid.

    Args:
        iqce: Chi-resolved intensity with dimensions including ``(q, chi, energy)``.
        chi_slices: Center angles (deg) for additional wedge columns.
        chi_width: Half-width of each wedge in degrees.

    Returns:
        DataFrame with a leading ``q`` column, the chi-average energy columns,
        and one block of columns per requested chi slice.
    """
    slice_items = [
        (float(angle), slice_chi(iqce, float(angle), chi_width))
        for angle in chi_slices
    ]
    return assemble_iq_table(chi_average(iqce), slice_items)


def compute_anisotropy(
    iqce: xr.DataArray,
    chi_width: float = 5.0,
    para_angle: float = 0.0,
    perp_angle: float = -90.0,
) -> xr.DataArray:
    """Compute the azimuthal anisotropy A(q, E).

    ``A = (I_para - I_perp) / (I_para + I_perp)`` where ``I_para`` and
    ``I_perp`` are chi wedges centered at ``para_angle`` and ``perp_angle``.
    This mirrors PyHyperScattering's ``AR`` definition (para at 0 deg, perp at
    -90 deg) but is computed here so it can be tuned independently. Values are
    bounded to ``[-1, 1]``.

    Args:
        iqce: Chi-resolved intensity with a ``chi`` dimension (degrees).
        chi_width: Half-width of the para/perp wedges in degrees.
        para_angle: Parallel wedge center in degrees.
        perp_angle: Perpendicular wedge center in degrees.

    Returns:
        Anisotropy with dimensions ``(q, energy)``.
    """
    para = slice_chi(iqce, para_angle, chi_width)
    perp = slice_chi(iqce, perp_angle, chi_width)
    anisotropy = (para - perp) / (para + perp)
    return anisotropy.clip(-1.0, 1.0)


def integrate_anisotropy(
    anisotropy: xr.DataArray,
    q_min: float | None = None,
    q_max: float | None = None,
) -> pd.DataFrame:
    """Integrate the anisotropy over q at each energy.

    For each energy, NaN points are dropped (negatives are kept, since the
    anisotropy is signed) and the remainder is integrated over q:

        int_A(E) = integral of A(q, E) dq

    Args:
        anisotropy: Anisotropy with dimensions ``(q, energy)``.
        q_min: Lower q bound, or None for no lower bound.
        q_max: Upper q bound, or None for no upper bound.

    Returns:
        DataFrame with columns ``energy`` and ``int_A``; energies with fewer
        than two valid points yield NaN.
    """
    q = np.asarray(anisotropy["q"].values, dtype=float)
    energies = np.asarray(anisotropy["energy"].values, dtype=float)

    q_range = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        q_range &= q >= q_min
    if q_max is not None:
        q_range &= q <= q_max

    int_values: list[float] = []
    for energy in energies:
        a_e = np.asarray(anisotropy.sel(energy=energy).values, dtype=float)
        q_e = q[q_range]
        a_e = a_e[q_range]

        valid = ~np.isnan(a_e)
        q_e = q_e[valid]
        a_e = a_e[valid]

        if q_e.size < 2:
            logger.warning(
                f"Energy {energy:.2f} eV: fewer than 2 valid anisotropy points "
                f"after trimming; integrated anisotropy set to NaN."
            )
            int_values.append(float("nan"))
            continue

        order = np.argsort(q_e)
        int_values.append(float(trapezoid(a_e[order], q_e[order])))

    return pd.DataFrame({"energy": energies, "int_A": int_values})


def iqchi_to_table(iqce: xr.DataArray, energy: float) -> pd.DataFrame:
    """Flatten the full I(q, chi) map at one energy into a 2D table.

    Args:
        iqce: Chi-resolved intensity with dimensions including ``(q, chi, energy)``.
        energy: The energy (eV) to extract (nearest match).

    Returns:
        DataFrame with a leading ``q`` column followed by one ``chi_{angle}``
        column per azimuthal bin.
    """
    frame = iqce.sel(energy=energy, method="nearest").transpose("q", "chi")
    table = frame.to_pandas()
    table.index.name = "q"
    table.columns = [f"chi_{float(angle):g}" for angle in table.columns]
    return table.reset_index()
