"""Tests for TOML config load/serialize round-tripping."""

from pathlib import Path

import pytest

from rsoxs_reduce.config import ReductionConfig, config_to_toml_str, load_config


def test_config_roundtrips_through_toml(tmp_path):
    cfg = ReductionConfig(
        data_path=Path("data/beamtime"),
        i0_path=Path("data/i0.txt"),
        polarization="90",
        q_min=0.002,
        q_max=0.04,
        waterfall_factor=5.0,
        colormap="magma",
        samples={89019: "S11_354K", 88994: "S11"},
        md_filter={"CCD Camera Shutter Inhibit": 0},
    )
    path = tmp_path / "reduction_config.toml"
    path.write_text(config_to_toml_str(cfg), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.data_path == Path("data/beamtime")
    assert loaded.i0_path == Path("data/i0.txt")
    assert loaded.polarization == "90"
    assert loaded.q_min == pytest.approx(0.002)
    assert loaded.q_max == pytest.approx(0.04)
    assert loaded.waterfall_factor == pytest.approx(5.0)
    assert loaded.colormap == "magma"
    assert loaded.samples == {89019: "S11_354K", 88994: "S11"}


def test_sample_name_uses_map_with_fallback():
    cfg = ReductionConfig(samples={89019: "S11_354K"})
    assert cfg.sample_name(89019) == "S11_354K"
    assert cfg.sample_name(99999) == "scan_99999"


def test_none_bounds_are_omitted_and_reload_as_none(tmp_path):
    text = config_to_toml_str(ReductionConfig())
    assert "q_min" not in text
    assert "q_max" not in text

    path = tmp_path / "reduction_config.toml"
    path.write_text(text, encoding="utf-8")
    loaded = load_config(path)

    assert loaded.q_min is None
    assert loaded.q_max is None


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.toml")
