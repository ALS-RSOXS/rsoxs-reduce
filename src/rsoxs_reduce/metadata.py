"""Build the reproducibility header prepended to every output ``.dat`` file.

The header records when the reduction ran, the versions of the key packages
involved, the exact command line, and the full resolved configuration embedded
as TOML so a run can be reconstructed from the data file alone.
"""

import platform
import sys
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rsoxs_reduce import __version__
from rsoxs_reduce.config import ReductionConfig, config_to_toml_str

_RULE = "=" * 60
_SUBRULE = "-" * 60
_PROVENANCE_PACKAGES = ("PyHyperScattering", "pyFAI", "numpy", "scipy", "xarray", "pandas")


def _package_version(name: str) -> str:
    """Return an installed package version, or 'unknown' if unavailable."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def build_header(
    cfg: ReductionConfig,
    file_filter: int,
    sample: str,
    energies: Sequence[float],
    config_path: Path,
) -> list[str]:
    """Assemble the reproducibility header lines (without the leading '# ').

    Args:
        cfg: The fully resolved configuration used for the run.
        file_filter: The scan number that was reduced.
        sample: The resolved sample name.
        energies: The energies requested via ``--energy`` (empty means all).
        config_path: The config file that was loaded.

    Returns:
        A list of plain-text lines; the writer is responsible for comment prefixes.
    """
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    command = "rsoxs-reduce " + " ".join(sys.argv[1:])
    energy_summary = (
        ", ".join(f"{value:g}" for value in sorted(energies)) if energies else "all"
    )

    lines: list[str] = [
        _RULE,
        f"rsoxs-reduction v{__version__}",
        f"Generated: {generated}",
        _SUBRULE,
        "[run]",
        f"scan_number = {file_filter}",
        f"sample = {sample}",
        f"energies = {energy_summary}",
        f"config_file = {config_path}",
        f"command = {command}",
        _SUBRULE,
        "[provenance]",
    ]
    for package in _PROVENANCE_PACKAGES:
        lines.append(f"{package} = {_package_version(package)}")
    lines.append(f"python = {platform.python_version()}")
    lines.append(_SUBRULE)
    lines.append("[config]  # full resolved configuration, valid TOML")
    lines.extend(config_to_toml_str(cfg).splitlines())
    lines.append(_RULE)
    return lines
