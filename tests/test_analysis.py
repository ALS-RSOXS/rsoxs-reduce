"""Unit tests for the pure reduction math in rsoxs_reduce.analysis."""

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

from rsoxs_reduce.analysis import compute_isi, iqe_to_table, select_energies


def _make_iqe(values: np.ndarray, q: np.ndarray, energy: np.ndarray) -> xr.DataArray:
    """Build an I(q, E) DataArray for testing."""
    return xr.DataArray(
        values,
        dims=("q", "energy"),
        coords={"q": q, "energy": energy},
    )


def test_iqe_to_table_has_q_first_and_r_energy_columns():
    q = np.array([0.1, 0.2, 0.3])
    energy = np.array([285.24, 286.0])
    iqe = _make_iqe(np.ones((3, 2)), q, energy)

    table = iqe_to_table(iqe)

    assert list(table.columns)[0] == "q"
    assert np.allclose(table["q"].to_numpy(), q)
    # Energy columns are labelled R_{energy} rounded to the nearest tenth.
    assert list(table.columns)[1:] == ["R_285.2", "R_286.0"]


def test_compute_isi_matches_trapezoid_of_i_q_squared():
    q = np.array([1.0, 2.0, 3.0])
    intensity = np.array([1.0, 1.0, 1.0])
    iqe = _make_iqe(intensity.reshape(3, 1), q, np.array([285.0]))

    result = compute_isi(iqe)

    expected = trapezoid(intensity * q**2, q)
    assert result["ISI"].iloc[0] == pytest.approx(expected)


def test_compute_isi_drops_nan_and_negative_points():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    intensity = np.array([1.0, np.nan, -5.0, 2.0])
    iqe = _make_iqe(intensity.reshape(4, 1), q, np.array([285.0]))

    result = compute_isi(iqe)

    # Only q=1 and q=4 survive; integrate over those two points.
    q_valid = np.array([1.0, 4.0])
    i_valid = np.array([1.0, 2.0])
    expected = trapezoid(i_valid * q_valid**2, q_valid)
    assert result["ISI"].iloc[0] == pytest.approx(expected)


def test_compute_isi_respects_q_bounds():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    intensity = np.ones(4)
    iqe = _make_iqe(intensity.reshape(4, 1), q, np.array([285.0]))

    result = compute_isi(iqe, q_min=2.0, q_max=3.0)

    q_valid = np.array([2.0, 3.0])
    expected = trapezoid(np.ones(2) * q_valid**2, q_valid)
    assert result["ISI"].iloc[0] == pytest.approx(expected)


def test_compute_isi_returns_nan_with_too_few_points():
    q = np.array([1.0, 2.0])
    intensity = np.array([np.nan, -1.0])
    iqe = _make_iqe(intensity.reshape(2, 1), q, np.array([285.0]))

    result = compute_isi(iqe)

    assert np.isnan(result["ISI"].iloc[0])


def test_select_energies_matches_to_nearest_tenth():
    q = np.array([0.1, 0.2])
    energy = np.array([285.24, 286.01, 287.0])
    data = _make_iqe(np.arange(6.0).reshape(2, 3), q, energy)

    out = select_energies(data, [285.2, 287.0])

    # 285.24 -> 285.2 and 287.0 match; 286.01 -> 286.0 is excluded.
    assert np.allclose(out["energy"].to_numpy(), [285.24, 287.0])


def test_select_energies_raises_when_no_match():
    q = np.array([0.1, 0.2])
    energy = np.array([285.0, 286.0])
    data = _make_iqe(np.ones((2, 2)), q, energy)

    with pytest.raises(ValueError):
        select_energies(data, [999.0])
