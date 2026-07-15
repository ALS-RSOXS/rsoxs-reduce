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


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Return weights ``w`` such that ``trapezoid(y, x) == sum(w * y)``.

    Used to propagate uncertainty through trapezoidal integration: for a linear
    combination ``sum(w * y)`` of independent terms, the variance is
    ``sum((w * dy)**2)``.

    Args:
        x: Strictly increasing sample positions.

    Returns:
        Per-sample trapezoid weights, the same shape as ``x``.
    """
    weights = np.zeros_like(x, dtype=float)
    if x.size < 2:
        return weights
    dx = np.diff(x)
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    if x.size > 2:
        weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights


def iqe_to_table(
    iqe: xr.DataArray, prefix: str = "R", sigma: xr.DataArray | None = None
) -> pd.DataFrame:
    """Flatten a ``(q, energy)`` array into a wide table with q as the first column.

    The full, shared q-grid is preserved and NaNs are left in place so every
    energy column shares one set of q-values. Energy columns are labelled
    ``{prefix}_{energy}`` with the energy rounded to the nearest tenth of an eV.
    When ``sigma`` is given, a paired ``d{prefix}_{energy}`` uncertainty column
    follows each value column.

    Args:
        iqe: Reduced quantity with dimensions ``(q, energy)`` (e.g. intensity
            or anisotropy).
        prefix: Column-name prefix for the per-energy columns.
        sigma: Optional uncertainty with the same dimensions as ``iqe``.

    Returns:
        DataFrame with a leading ``q`` column followed by one
        ``{prefix}_{energy}`` (and optional ``d{prefix}_{energy}``) column per
        energy.
    """
    values = iqe.transpose("q", "energy").to_pandas()
    values.index.name = "q"
    out = pd.DataFrame({"q": values.index.to_numpy()})
    sig = sigma.transpose("q", "energy").to_pandas() if sigma is not None else None
    for energy in values.columns:
        out[f"{prefix}_{float(energy):.1f}"] = values[energy].to_numpy()
        if sig is not None:
            out[f"d{prefix}_{float(energy):.1f}"] = sig[energy].to_numpy()
    return out


def compute_isi(
    iqe: xr.DataArray,
    q_min: float | None = None,
    q_max: float | None = None,
    sigma: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Compute the integrated scattering intensity per energy.

    For each energy, points that are NaN or negative are dropped (they indicate
    reduction artifacts and are excluded from analysis), the surviving intensity
    is scaled by ``q**2``, and the result is integrated over q:

        ISI(E) = integral of I(q, E) * q**2 dq

    When ``sigma`` is given, the uncertainty is propagated through the (linear)
    trapezoidal integral as ``dISI = sqrt(sum((w_k * q_k**2 * dI_k)**2))`` over
    the same surviving points, with ``w_k`` the trapezoid weights.

    Args:
        iqe: Reduced intensity with dimensions ``(q, energy)``.
        q_min: Lower q bound applied before trimming, or None for no lower bound.
        q_max: Upper q bound applied before trimming, or None for no upper bound.
        sigma: Optional per-bin uncertainty with the same dimensions as ``iqe``.

    Returns:
        DataFrame with columns ``energy`` and ``ISI`` (plus ``dISI`` when
        ``sigma`` is given); energies with fewer than two valid points yield NaN.
    """
    q = np.asarray(iqe["q"].values, dtype=float)
    energies = np.asarray(iqe["energy"].values, dtype=float)

    q_range = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        q_range &= q >= q_min
    if q_max is not None:
        q_range &= q <= q_max

    isi_values: list[float] = []
    disi_values: list[float] = []
    for energy in energies:
        intensity = np.asarray(iqe.sel(energy=energy).values, dtype=float)
        q_e = q[q_range]
        i_e = intensity[q_range]

        valid = ~np.isnan(i_e) & (i_e >= 0.0)
        q_e = q_e[valid]
        i_e = i_e[valid]
        if sigma is not None:
            s_e = np.asarray(sigma.sel(energy=energy).values, dtype=float)
            s_e = s_e[q_range][valid]

        if q_e.size < 2:
            logger.warning(
                f"Energy {energy:.2f} eV: fewer than 2 valid points after "
                f"trimming; ISI set to NaN."
            )
            isi_values.append(float("nan"))
            disi_values.append(float("nan"))
            continue

        order = np.argsort(q_e)
        q_e = q_e[order]
        i_e = i_e[order]
        integrand = i_e * q_e**2
        isi_values.append(float(trapezoid(integrand, q_e)))
        if sigma is not None:
            weights = _trapezoid_weights(q_e)
            disi_values.append(
                float(np.sqrt(np.sum((weights * q_e**2 * s_e[order]) ** 2)))
            )

    columns = {"energy": energies, "ISI": isi_values}
    if sigma is not None:
        columns["dISI"] = disi_values
    return pd.DataFrame(columns)


def chi_average(
    iqce: xr.DataArray, sigma: xr.DataArray | None = None
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Average the chi-resolved intensity over all chi to get isotropic I(q, E).

    When ``sigma`` is given, the uncertainty of the mean over the ``N`` valid
    chi bins is propagated as ``dI = sqrt(sum(dI_i**2)) / N`` (independent bins).

    Args:
        iqce: Chi-resolved intensity with a ``chi`` dimension.
        sigma: Optional per-bin uncertainty with the same dimensions as ``iqce``.

    Returns:
        Intensity with ``chi`` removed, dimensions ``(q, energy)``; or a tuple
        of that intensity and its propagated uncertainty when ``sigma`` is given.
    """
    average = iqce.mean("chi")
    if sigma is None:
        return average
    valid = iqce.notnull()
    count = valid.sum("chi")
    variance = (sigma.where(valid) ** 2).sum("chi")
    return average, np.sqrt(variance) / count


def slice_chi(
    iqce: xr.DataArray,
    center: float,
    half_width: float,
    sigma: xr.DataArray | None = None,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Average I(q, chi, E) over an azimuthal wedge centered at ``center``.

    If the wedge is narrower than the chi grid it selects zero or one bins; a
    notice is logged and the reduction continues using the nearest single bin.
    When ``sigma`` is given, the uncertainty of the wedge mean is propagated as
    ``dI = sqrt(sum(dI_i**2)) / N`` over the selected valid bins.

    Args:
        iqce: Chi-resolved intensity with a ``chi`` dimension (degrees).
        center: Wedge center angle in degrees.
        half_width: Half-width of the wedge in degrees (each side of center).
        sigma: Optional per-bin uncertainty with the same dimensions as ``iqce``.

    Returns:
        Intensity averaged over the wedge, dimensions ``(q, energy)``; or a
        tuple of that intensity and its propagated uncertainty when ``sigma``
        is given.
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

    selected = iqce.isel(chi=mask)
    average = selected.mean("chi")
    if sigma is None:
        return average
    valid = selected.notnull()
    n_valid = valid.sum("chi")
    variance = (sigma.isel(chi=mask).where(valid) ** 2).sum("chi")
    return average, np.sqrt(variance) / n_valid


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


def assemble_iq_table(
    chi_avg: xr.DataArray,
    slice_items: Sequence[tuple[float, xr.DataArray]] = (),
    chi_avg_sigma: xr.DataArray | None = None,
    slice_sigma_items: Sequence[tuple[float, xr.DataArray]] = (),
) -> pd.DataFrame:
    """Build the wide I(q, E) table from a precomputed chi-average and slices.

    Kept separate from ``iqce_to_table`` so batched processing can assemble the
    table from per-batch pieces that were concatenated along energy, without
    holding the full chi-resolved array. When the ``*_sigma`` arguments are
    given, paired ``dR_...`` uncertainty columns follow each value column.

    Args:
        chi_avg: Chi-average intensity with dimensions ``(q, energy)``.
        slice_items: Sequence of ``(angle_deg, wedge)`` pairs, each ``wedge`` a
            ``(q, energy)`` array appended as ``R_{energy}_chi{angle}`` columns.
        chi_avg_sigma: Optional uncertainty for ``chi_avg``.
        slice_sigma_items: Optional ``(angle_deg, wedge_sigma)`` pairs matching
            ``slice_items``.

    Returns:
        DataFrame with a leading ``q`` column, the chi-average energy columns,
        and one block of columns per chi slice.
    """
    table = iqe_to_table(chi_avg, sigma=chi_avg_sigma)
    sigma_map = {float(angle): wedge for angle, wedge in slice_sigma_items}
    for angle, wedge in slice_items:
        wedge_table = iqe_to_table(wedge, sigma=sigma_map.get(float(angle)))
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
    sigma: xr.DataArray | None = None,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Compute the azimuthal anisotropy A(q, E).

    ``A = (I_para - I_perp) / (I_para + I_perp)`` where ``I_para`` and
    ``I_perp`` are chi wedges centered at ``para_angle`` and ``perp_angle``.
    This mirrors PyHyperScattering's ``AR`` definition (para at 0 deg, perp at
    -90 deg) but is computed here so it can be tuned independently. Values are
    bounded to ``[-1, 1]``.

    When ``sigma`` is given, the uncertainty is propagated through the ratio as
    ``dA = 2/(P+Q)**2 * sqrt((Q*dP)**2 + (P*dQ)**2)`` from the wedge means
    ``P``/``Q`` and their uncertainties ``dP``/``dQ``.

    Args:
        iqce: Chi-resolved intensity with a ``chi`` dimension (degrees).
        chi_width: Half-width of the para/perp wedges in degrees.
        para_angle: Parallel wedge center in degrees.
        perp_angle: Perpendicular wedge center in degrees.
        sigma: Optional per-bin uncertainty with the same dimensions as ``iqce``.

    Returns:
        Anisotropy with dimensions ``(q, energy)``; or a tuple of that
        anisotropy and its propagated uncertainty when ``sigma`` is given.
    """
    if sigma is None:
        para = slice_chi(iqce, para_angle, chi_width)
        perp = slice_chi(iqce, perp_angle, chi_width)
        return ((para - perp) / (para + perp)).clip(-1.0, 1.0)

    para, para_sigma = slice_chi(iqce, para_angle, chi_width, sigma=sigma)
    perp, perp_sigma = slice_chi(iqce, perp_angle, chi_width, sigma=sigma)
    denom = para + perp
    anisotropy = ((para - perp) / denom).clip(-1.0, 1.0)
    anisotropy_sigma = (2.0 / denom**2) * np.sqrt(
        (perp * para_sigma) ** 2 + (para * perp_sigma) ** 2
    )
    return anisotropy, anisotropy_sigma


def integrate_anisotropy(
    anisotropy: xr.DataArray,
    q_min: float | None = None,
    q_max: float | None = None,
    sigma: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Integrate the anisotropy over q at each energy.

    For each energy, NaN points are dropped (negatives are kept, since the
    anisotropy is signed) and the remainder is integrated over q:

        int_A(E) = integral of A(q, E) dq

    When ``sigma`` is given, its uncertainty is propagated through the
    trapezoidal integral as ``d = sqrt(sum((w_k * dA_k)**2))``.

    Args:
        anisotropy: Anisotropy with dimensions ``(q, energy)``.
        q_min: Lower q bound, or None for no lower bound.
        q_max: Upper q bound, or None for no upper bound.
        sigma: Optional anisotropy uncertainty with the same dimensions.

    Returns:
        DataFrame with columns ``energy`` and ``int_A`` (plus ``dint_A`` when
        ``sigma`` is given); energies with fewer than two valid points yield NaN.
    """
    q = np.asarray(anisotropy["q"].values, dtype=float)
    energies = np.asarray(anisotropy["energy"].values, dtype=float)

    q_range = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        q_range &= q >= q_min
    if q_max is not None:
        q_range &= q <= q_max

    int_values: list[float] = []
    dint_values: list[float] = []
    for energy in energies:
        a_e = np.asarray(anisotropy.sel(energy=energy).values, dtype=float)
        q_e = q[q_range]
        a_e = a_e[q_range]

        valid = ~np.isnan(a_e)
        q_e = q_e[valid]
        a_e = a_e[valid]
        if sigma is not None:
            da_e = np.asarray(sigma.sel(energy=energy).values, dtype=float)
            da_e = da_e[q_range][valid]

        if q_e.size < 2:
            logger.warning(
                f"Energy {energy:.2f} eV: fewer than 2 valid anisotropy points "
                f"after trimming; integrated anisotropy set to NaN."
            )
            int_values.append(float("nan"))
            dint_values.append(float("nan"))
            continue

        order = np.argsort(q_e)
        q_sorted = q_e[order]
        int_values.append(float(trapezoid(a_e[order], q_sorted)))
        if sigma is not None:
            weights = _trapezoid_weights(q_sorted)
            dint_values.append(float(np.sqrt(np.sum((weights * da_e[order]) ** 2))))

    columns = {"energy": energies, "int_A": int_values}
    if sigma is not None:
        columns["dint_A"] = dint_values
    return pd.DataFrame(columns)


def iqchi_to_table(
    iqce: xr.DataArray, energy: float, sigma: xr.DataArray | None = None
) -> pd.DataFrame:
    """Flatten the full I(q, chi) map at one energy into a 2D table.

    When ``sigma`` is given, a paired ``dchi_{angle}`` uncertainty column
    follows each ``chi_{angle}`` value column.

    Args:
        iqce: Chi-resolved intensity with dimensions including ``(q, chi, energy)``.
        energy: The energy (eV) to extract (nearest match).
        sigma: Optional per-bin uncertainty with the same dimensions as ``iqce``.

    Returns:
        DataFrame with a leading ``q`` column followed by one ``chi_{angle}``
        (and optional ``dchi_{angle}``) column per azimuthal bin.
    """
    frame = iqce.sel(energy=energy, method="nearest").transpose("q", "chi")
    values = frame.to_pandas()
    values.index.name = "q"
    out = pd.DataFrame({"q": values.index.to_numpy()})
    sig = None
    if sigma is not None:
        sig = (
            sigma.sel(energy=energy, method="nearest").transpose("q", "chi").to_pandas()
        )
    for chi in values.columns:
        out[f"chi_{float(chi):g}"] = values[chi].to_numpy()
        if sig is not None:
            out[f"dchi_{float(chi):g}"] = sig[chi].to_numpy()
    return out
