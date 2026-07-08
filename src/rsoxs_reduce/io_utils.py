"""Filesystem helpers for results directories and tab-delimited .dat output."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_results_dir(results_root: Path, sample: str, dry_run: bool = False) -> Path:
    """Return (and, unless dry-run, create) the per-sample results directory.

    Args:
        results_root: Root directory for all sample output folders.
        sample: Sample name used as the sub-directory.
        dry_run: If True, do not create the directory; only log the intent.

    Returns:
        The path to the sample's results directory.
    """
    out_dir = Path(results_root) / sample
    if dry_run:
        logger.info(f"[dry-run] Would create results directory: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_dat(
    df: pd.DataFrame,
    path: Path,
    header: Sequence[str] | None = None,
    dry_run: bool = False,
    float_format: str = "%.8e",
) -> None:
    """Write a DataFrame to a tab-delimited ``.dat`` file with a comment header.

    Args:
        df: Data to write.
        path: Destination file path.
        header: Optional lines written as ``# ...`` comments before the table.
        dry_run: If True, do not write; only log the intent.
        float_format: Format string for floating-point values.
    """
    path = Path(path)
    if dry_run:
        logger.info(f"[dry-run] Would write {len(df)} rows to {path}")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in header or []:
            handle.write(f"# {line}\n")
        df.to_csv(
            handle,
            sep="\t",
            index=False,
            na_rep="nan",
            float_format=float_format,
        )
    logger.info(f"Wrote {path}")
