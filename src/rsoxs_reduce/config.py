"""Reduction configuration: the dataclass schema plus TOML load/serialize.

All beamtime-specific values (data paths, detector geometry, the scan-number to
sample-name map, plotting limits) live in an editable ``reduction_config.toml``.
The dataclass here defines the schema and generic fallback defaults; a real run
is fully described by the TOML, which is also embedded verbatim into the output
``.dat`` headers for reproducibility.
"""

import logging
import math
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
        i1_path: Path to the I1 normalization file, or None to reduce without
            the I1 correction (omit the config key, or set it to an empty
            string or ``"none"``).
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
        ni_tiltx: Detector tilt x in the unit given by ``tilt_units``.
        ni_tilty: Detector tilt y in the unit given by ``tilt_units``.
        tilt_units: Unit of ``ni_tiltx``/``ni_tilty`` in the config, either
            ``"degrees"`` (default) or ``"radians"``; converted to the degrees
            that PyHyperScattering's nika geometry expects.
        ni_pixsize_x: Pixel size in x (mm).
        ni_pixsize_y: Pixel size in y (mm).
        mask_rot90: Number of 90-degree rotations (numpy ``rot90`` k) applied to
            the loaded mask.
        mask_flipud: Whether to flip the mask vertically (after rotation).
        mask_fliplr: Whether to flip the mask horizontally (after rotation).
        mask_invert: Whether to invert the mask (Nika ``M_ROIMask`` uses 1=keep,
            so invert to the pyFAI convention where nonzero=excluded).
        integration_method: pyFAI integration method (``csr`` CPU, ``csr_ocl`` OpenCL).
        polarization: Polarization coordinate value to select after integration.
        stack_dims: Dimensions passed to ``loadFileSeries``.
        energy_match_decimals: Decimal places used when matching energy values.
        energies: Energies (eV) to reduce; empty means all. Overridden by the
            ``--energy`` CLI option when that is given.
        q_bins: Number of radial q points in the output (pyFAI ``npts``); fewer
            bins bundle more pixels per point (coarser q resolution).
        q_min: Lower q bound for the ISI integral, or None for no lower bound.
        q_max: Upper q bound for the ISI integral, or None for no upper bound.
        return_sigma: Whether to propagate pyFAI's per-bin uncertainty through
            the reduction and include it in the output files and plots.
        sample_thickness: Sample thickness for intensity normalization (in
            ``thickness_units``), or None to skip thickness normalization.
        sample_thickness_uncertainty: Uncertainty on ``sample_thickness`` (same
            units); folded into the propagated uncertainty when ``return_sigma``
            is set.
        thickness_units: Unit of the thickness values, converted to cm before
            normalization (``"nm"`` default; also ``um``/``mm``/``cm``/``m``).
        detector_vmin: Lower bound of the 2D detector log color scale.
        detector_vmax: Upper bound of the 2D detector log color scale.
        detector_dpi: Output resolution for 2D detector frames.
        plot_dpi: Output resolution for line plots.
        waterfall_factor: Multiplicative offset between successive waterfall curves.
        colormap: Matplotlib colormap used to color curves by energy.
        batch_size: Energies to load and reduce per batch to bound memory; 0
            loads and reduces the whole scan at once (the default).
        chi_width: Azimuthal half-width in degrees for chi slices and anisotropy
            (a slice at angle theta averages ``chi in [theta - chi_width,
            theta + chi_width]``).
        anisotropy: Whether to write the A(q, E) anisotropy ``.dat``.
        anisotropy_plot: Whether to plot A(q, E) (all energies overlaid).
        integrated_anisotropy: Whether to write and plot the q-integrated
            anisotropy versus energy.
        iqchi_dat: Whether to write the full I(q, chi) ``.dat`` per energy.
        iqchi_plot: Whether to plot the full I(q, chi) map per energy.
        chi_slices: Center angles in degrees for additional chi-slice output
            columns/curves; empty means the chi-average only.
        md_filter: Metadata filter passed to ``loadFileSeries``.
        samples: Scan-number to sample-name map (beamtime specific).
    """

    # Paths (generic placeholders; real values come from the TOML).
    data_path: Path = Path("data")
    fits_subdir: str = "fits"
    dark_subdir: str = "fits"
    i0_path: Path = Path("i0.txt")
    i1_path: Path | None = None
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
    ni_tiltx: float = 0.0
    ni_tilty: float = 0.0
    tilt_units: str = "degrees"
    ni_pixsize_x: float = 0.0096
    ni_pixsize_y: float = 0.0096

    # Mask orientation (applied to the loaded mask before integration). The
    # defaults reproduce PyHyperScattering's intended nika transform.
    mask_rot90: int = 1
    mask_flipud: bool = True
    mask_fliplr: bool = False
    mask_invert: bool = True

    # Reduction / selection.
    integration_method: str = "csr"
    polarization: str = "0"
    energy_match_decimals: int = 1
    q_bins: int = 500
    q_min: float | None = None
    q_max: float | None = None
    return_sigma: bool = False
    sample_thickness: float | None = None
    sample_thickness_uncertainty: float | None = None
    thickness_units: str = "nm"

    # Plotting.
    detector_vmin: float = 1e-3
    detector_vmax: float = 1e5
    detector_dpi: int = 72
    plot_dpi: int = 150
    waterfall_factor: float = 10.0
    colormap: str = "viridis"

    # Processing ([processing] section).
    batch_size: int = 0

    # Output extras ([output] section). The chi-average I(q, E) is always
    # produced; everything here is additive and off by default.
    chi_width: float = 5.0
    anisotropy: bool = False
    anisotropy_plot: bool = False
    integrated_anisotropy: bool = False
    iqchi_dat: bool = False
    iqchi_plot: bool = False

    # Collections.
    energies: list[float] = field(default_factory=list)
    chi_slices: list[float] = field(default_factory=list)
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

    def _tilt_in_degrees(self, value: float) -> float:
        """Convert a tilt from the configured ``tilt_units`` to degrees.

        Args:
            value: Tilt value in the unit given by ``tilt_units``.

        Returns:
            The tilt in degrees (the unit PyHyperScattering's nika geometry
            expects).

        Raises:
            ValueError: If ``tilt_units`` is not ``"degrees"`` or ``"radians"``.
        """
        units = self.tilt_units.strip().lower()
        if units in ("deg", "degree", "degrees"):
            return value
        if units in ("rad", "radian", "radians"):
            return math.degrees(value)
        raise ValueError(
            f"tilt_units must be 'degrees' or 'radians', got {self.tilt_units!r}."
        )

    def _thickness_in_cm(self, value: float) -> float:
        """Convert a length from the configured ``thickness_units`` to cm.

        Args:
            value: Length value in ``thickness_units``.

        Returns:
            The length in centimeters.

        Raises:
            ValueError: If ``thickness_units`` is not a recognized length unit.
        """
        factors = {
            "nm": 1e-7,
            "um": 1e-4,
            "µm": 1e-4,
            "micron": 1e-4,
            "microns": 1e-4,
            "mm": 1e-1,
            "cm": 1.0,
            "m": 1e2,
        }
        units = self.thickness_units.strip().lower()
        if units not in factors:
            raise ValueError(
                f"thickness_units must be one of {sorted(factors)}, "
                f"got {self.thickness_units!r}."
            )
        return value * factors[units]

    @property
    def sample_thickness_cm(self) -> float | None:
        """Sample thickness in cm, or None if not set."""
        if self.sample_thickness is None:
            return None
        return self._thickness_in_cm(self.sample_thickness)

    @property
    def sample_thickness_uncertainty_cm(self) -> float | None:
        """Sample-thickness uncertainty in cm, or None if not set."""
        if self.sample_thickness_uncertainty is None:
            return None
        return self._thickness_in_cm(self.sample_thickness_uncertainty)

    @property
    def ni_tiltx_deg(self) -> float:
        """Detector x tilt in degrees, converted from the configured unit."""
        return self._tilt_in_degrees(self.ni_tiltx)

    @property
    def ni_tilty_deg(self) -> float:
        """Detector y tilt in degrees, converted from the configured unit."""
        return self._tilt_in_degrees(self.ni_tilty)

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
        "ni_tiltx",
        "ni_tilty",
        "tilt_units",
        "ni_pixsize_x",
        "ni_pixsize_y",
        "mask_rot90",
        "mask_flipud",
        "mask_fliplr",
        "mask_invert",
    ),
    "reduction": (
        "integration_method",
        "polarization",
        "energy_match_decimals",
        "energies",
        "q_bins",
        "q_min",
        "q_max",
        "return_sigma",
        "sample_thickness",
        "sample_thickness_uncertainty",
        "thickness_units",
    ),
    "plotting": (
        "detector_vmin",
        "detector_vmax",
        "detector_dpi",
        "plot_dpi",
        "waterfall_factor",
        "colormap",
    ),
    "processing": ("batch_size",),
    "output": (
        "chi_slices",
        "chi_width",
        "anisotropy",
        "anisotropy_plot",
        "integrated_anisotropy",
        "iqchi_dat",
        "iqchi_plot",
    ),
}
_PATH_FIELDS: frozenset[str] = frozenset(
    {"data_path", "i0_path", "i1_path", "mask_path", "results_root"}
)
# Path fields that may be left empty in the config, meaning "not set" (None).
_OPTIONAL_PATH_FIELDS: frozenset[str] = frozenset({"i1_path"})
# Config string values that mean "not set" for an optional path field.
_EMPTY_PATH_VALUES: frozenset[str] = frozenset({"", "none", "null"})


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
            if key in _PATH_FIELDS:
                if (
                    key in _OPTIONAL_PATH_FIELDS
                    and str(value).strip().lower() in _EMPTY_PATH_VALUES
                ):
                    kwargs[key] = None
                else:
                    kwargs[key] = Path(value)
            else:
                kwargs[key] = value

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


def validate_inputs(cfg: ReductionConfig) -> list[str]:
    """Check that every input path referenced by the config exists.

    Verifies the data directory and its FITS and dark sub-directories are
    present, that the I0 and mask paths exist and are files, and that at least
    one ``.fits`` file is present in the FITS sub-directory. The I1 path is
    checked only when set (it is optional). Only input paths are checked;
    ``results_root`` is an output and is created on demand.

    Args:
        cfg: The reduction configuration to validate.

    Returns:
        A list of human-readable problem descriptions; empty if all inputs
        are present.
    """
    problems: list[str] = []

    if not cfg.data_path.is_dir():
        problems.append(f"data_path is not a directory: {cfg.data_path}")
    if not cfg.fits_path.is_dir():
        problems.append(f"fits sub-directory is not a directory: {cfg.fits_path}")
    if not cfg.dark_path.is_dir():
        problems.append(f"dark sub-directory is not a directory: {cfg.dark_path}")

    files_to_check = [("i0_path", cfg.i0_path), ("mask_path", cfg.mask_path)]
    if cfg.i1_path is None:
        logger.info("i1_path is not set; reducing without the I1 correction.")
    else:
        files_to_check.append(("i1_path", cfg.i1_path))
    for label, path in files_to_check:
        if not path.is_file():
            problems.append(f"{label} is not a file: {path}")

    # Only worth scanning for .fits content if the directory itself exists.
    if cfg.fits_path.is_dir():
        try:
            has_fits = any(
                entry.is_file() and entry.suffix.lower() == ".fits"
                for entry in cfg.fits_path.iterdir()
            )
        except OSError as exc:
            problems.append(f"could not scan {cfg.fits_path} for .fits files: {exc}")
        else:
            if not has_fits:
                problems.append(f"no .fits files found in {cfg.fits_path}")

    return problems


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
