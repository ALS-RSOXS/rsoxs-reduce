"""Reduction configuration: the dataclass schema plus TOML load/serialize.

All beamtime-specific values (data paths, detector geometry, the scan-number to
sample-name map, plotting limits) live in an editable ``reduction_config.toml``.
The dataclass here defines the schema and generic fallback defaults; a real run
is fully described by the TOML, which is also embedded verbatim into the output
``.dat`` headers for reproducibility.
"""

import logging
import tomllib
from dataclasses import dataclass, field, fields
from importlib.resources import files
from pathlib import Path

import tomli_w

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "reduction_config.toml"
BUNDLED_CONFIG = "default_config.toml"


@dataclass
class ReductionConfig:
    """Static configuration for a reduction run.

    Attributes:
        data_path: Root directory holding the beamtime data.
        fits_subdir: Sub-directory (under ``data_path``) containing the FITS series.
        dark_subdir: Sub-directory (under ``data_path``) containing the dark frames.
        i0_path: Path to the I0 normalization file.
        i1_path: Path to the I1 normalization file.
        mask_path: Path to the nika-format detector mask.
        results_root: Root directory into which ``{sample}/`` folders are written.
        corr_mode: Loader correction mode.
        exposure_offset: Loader exposure-time offset in seconds.
        data_collected_after_mar2021: Loader flag for the post-March-2021 headers.
        dark_subtract: Whether the loader subtracts darks.
        dark_pedestal: Pedestal added back after dark subtraction.
        ni_distance: Sample-detector distance (nika geometry).
        ni_bcx: Beam-center x in pixels (nika geometry).
        ni_bcy: Beam-center y in pixels (nika geometry).
        ni_pixsize_x: Pixel size in x (mm).
        ni_pixsize_y: Pixel size in y (mm).
        integration_method: pyFAI integration method (``csr`` CPU, ``csr_ocl`` OpenCL).
        polarization: Polarization coordinate value to select after integration.
        stack_dims: Dimensions passed to ``loadFileSeries``.
        energy_match_decimals: Decimal places used when matching ``--energy`` values.
        q_min: Lower q bound for the ISI integral, or None for no lower bound.
        q_max: Upper q bound for the ISI integral, or None for no upper bound.
        detector_vmin: Lower bound of the 2D detector log color scale.
        detector_vmax: Upper bound of the 2D detector log color scale.
        detector_dpi: Output resolution for 2D detector frames.
        plot_dpi: Output resolution for line plots.
        waterfall_factor: Multiplicative offset between successive waterfall curves.
        colormap: Matplotlib colormap used to color curves by energy.
        md_filter: Metadata filter passed to ``loadFileSeries``.
        samples: Scan-number to sample-name map (beamtime specific).
    """

    # Paths (generic placeholders; real values come from the TOML).
    data_path: Path = Path("data")
    fits_subdir: str = "fits"
    dark_subdir: str = "fits"
    i0_path: Path = Path("i0.txt")
    i1_path: Path = Path("i1.txt")
    mask_path: Path = Path("mask.hdf")
    results_root: Path = Path("results")

    # Loader settings (ALS11012RSoXSLoader).
    corr_mode: str = "als"
    exposure_offset: float = 0.0
    data_collected_after_mar2021: bool = True
    dark_subtract: bool = True
    dark_pedestal: int = 200

    # Integrator geometry (PFEnergySeriesIntegrator, nika convention).
    ni_distance: float = 74.7688
    ni_bcx: float = 1831.35
    ni_bcy: float = 2101.49
    ni_pixsize_x: float = 0.0096
    ni_pixsize_y: float = 0.0096

    # Reduction / selection.
    integration_method: str = "csr"
    polarization: str = "0"
    energy_match_decimals: int = 1
    q_min: float | None = None
    q_max: float | None = None

    # Plotting.
    detector_vmin: float = 1e-3
    detector_vmax: float = 1e5
    detector_dpi: int = 72
    plot_dpi: int = 150
    waterfall_factor: float = 10.0
    colormap: str = "viridis"

    # Collections.
    stack_dims: list[str] = field(default_factory=lambda: ["energy", "polarization"])
    md_filter: dict[str, int] = field(
        default_factory=lambda: {"CCD Camera Shutter Inhibit": 0}
    )
    samples: dict[int, str] = field(default_factory=dict)

    @property
    def fits_path(self) -> Path:
        """Full path to the FITS series directory."""
        return self.data_path / self.fits_subdir

    @property
    def dark_path(self) -> Path:
        """Full path to the dark-frame directory."""
        return self.data_path / self.dark_subdir

    def sample_name(self, file_filter: int) -> str:
        """Return the sample name for a scan number.

        Args:
            file_filter: The scan number used as the file filter.

        Returns:
            The mapped sample name, or ``scan_{file_filter}`` if unknown.
        """
        return self.samples.get(file_filter, f"scan_{file_filter}")


# Grouping of dataclass fields into TOML sections for load/serialize.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "paths": (
        "data_path",
        "fits_subdir",
        "dark_subdir",
        "i0_path",
        "i1_path",
        "mask_path",
        "results_root",
    ),
    "loader": (
        "corr_mode",
        "exposure_offset",
        "data_collected_after_mar2021",
        "dark_subtract",
        "dark_pedestal",
    ),
    "geometry": (
        "ni_distance",
        "ni_bcx",
        "ni_bcy",
        "ni_pixsize_x",
        "ni_pixsize_y",
    ),
    "reduction": (
        "integration_method",
        "polarization",
        "energy_match_decimals",
        "q_min",
        "q_max",
    ),
    "plotting": (
        "detector_vmin",
        "detector_vmax",
        "detector_dpi",
        "plot_dpi",
        "waterfall_factor",
        "colormap",
    ),
}
_PATH_FIELDS: frozenset[str] = frozenset(
    {"data_path", "i0_path", "i1_path", "mask_path", "results_root"}
)


def config_to_dict(cfg: ReductionConfig) -> dict[str, object]:
    """Serialize a config to a nested, TOML-ready dict.

    Path fields become strings, ``None`` values are omitted (TOML has no null),
    and sample-map keys become strings.

    Args:
        cfg: The configuration to serialize.

    Returns:
        A nested dict keyed by TOML section.
    """
    out: dict[str, object] = {}
    for section, names in _SECTIONS.items():
        table: dict[str, object] = {}
        for name in names:
            value = getattr(cfg, name)
            if value is None:
                continue
            table[name] = str(value) if name in _PATH_FIELDS else value
        out[section] = table
    out["md_filter"] = dict(cfg.md_filter)
    out["samples"] = {str(key): value for key, value in cfg.samples.items()}
    return out


def config_to_toml_str(cfg: ReductionConfig) -> str:
    """Return the configuration as a TOML string (used for the .dat headers)."""
    return tomli_w.dumps(config_to_dict(cfg))


def load_config(path: Path) -> ReductionConfig:
    """Load a reduction configuration from a TOML file.

    Args:
        path: Path to the TOML file.

    Returns:
        A configuration with the file's values applied over the defaults.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {name for names in _SECTIONS.values() for name in names}
    kwargs: dict[str, object] = {}

    for section, names in _SECTIONS.items():
        table = raw.get(section, {})
        for key, value in table.items():
            if key not in names:
                logger.warning(f"Unknown key '{section}.{key}' in {path}; ignoring.")
                continue
            kwargs[key] = Path(value) if key in _PATH_FIELDS else value

    for top_level in ("md_filter",):
        if top_level in raw:
            kwargs[top_level] = dict(raw[top_level])
    if "samples" in raw:
        kwargs["samples"] = {int(key): value for key, value in raw["samples"].items()}

    # Warn on stray top-level tables that are not recognized.
    recognized_tables = set(_SECTIONS) | {"md_filter", "samples"}
    for table_name in raw:
        if table_name not in recognized_tables:
            logger.warning(f"Unknown table '[{table_name}]' in {path}; ignoring.")

    valid = {f.name for f in fields(ReductionConfig)} | {"md_filter", "samples"}
    assert known <= valid  # guard against section/schema drift
    return ReductionConfig(**kwargs)


def resolve_config_path(explicit: Path | None) -> Path:
    """Determine which config file to use.

    Resolution order: an explicit ``--config`` path, then ``reduction_config.toml``
    in the current working directory, then the bundled default.

    Args:
        explicit: The path passed via ``--config``, or None.

    Returns:
        The path to the config file to load.

    Raises:
        FileNotFoundError: If an explicit path was given but does not exist.
    """
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(f"--config file not found: {explicit}")
        return explicit

    cwd_config = Path.cwd() / CONFIG_FILENAME
    if cwd_config.exists():
        return cwd_config

    return Path(str(files("rsoxs_reduce").joinpath(BUNDLED_CONFIG)))
