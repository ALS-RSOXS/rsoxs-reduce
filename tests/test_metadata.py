"""Tests for the reproducibility header."""

from pathlib import Path

from rsoxs_reduce.config import ReductionConfig
from rsoxs_reduce.metadata import build_header


def test_build_header_contains_run_provenance_and_config():
    cfg = ReductionConfig(samples={89019: "S11_354K"})
    lines = build_header(
        cfg, 89019, "S11_354K", [285.2, 286.0], Path("reduction_config.toml")
    )
    text = "\n".join(lines)

    assert "[run]" in text
    assert "scan_number = 89019" in text
    assert "sample = S11_354K" in text
    assert "energies = 285.2, 286" in text
    assert "[provenance]" in text
    assert "PyHyperScattering =" in text
    assert "python =" in text
    # The full config is embedded as TOML.
    assert "[config]" in text
    assert "[samples]" in text


def test_build_header_reports_all_energies_when_none_selected():
    cfg = ReductionConfig()
    lines = build_header(cfg, 88994, "S11", [], Path("reduction_config.toml"))
    assert any("energies = all" in line for line in lines)
