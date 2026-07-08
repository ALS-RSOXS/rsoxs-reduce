"""Typer CLI: reduce a single ALS 11.0.1.2 RSoXS energy scan by scan number."""

import logging
from pathlib import Path

import typer

from rsoxs_reduce.analysis import compute_isi, iqe_to_table, select_energies
from rsoxs_reduce.config import ReductionConfig, load_config, resolve_config_path
from rsoxs_reduce.io_utils import prepare_results_dir, write_dat
from rsoxs_reduce.metadata import build_header

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Reduce an ALS 11.0.1.2 RSoXS energy scan to I(q,E), ISI(E), and plots.",
)


def _planned_outputs(
    out_dir: Path, sample: str, plots: bool, detector_2d: bool
) -> list[Path]:
    """Return the list of paths a full run would produce (for --dry-run)."""
    outputs = [
        out_dir / f"{sample}_Ivsq.dat",
        out_dir / f"{sample}_ISIvsE.dat",
    ]
    if plots:
        outputs += [
            out_dir / f"{sample}_ISIvsE.png",
            out_dir / f"{sample}_Ivsq_overlay_raw.png",
            out_dir / f"{sample}_Ivsq_overlay_q2.png",
            out_dir / f"{sample}_Ivsq_waterfall_raw.png",
            out_dir / f"{sample}_Ivsq_waterfall_q2.png",
        ]
    if detector_2d:
        outputs.append(out_dir / "detector_2d")
    return outputs


def _apply_overrides(
    cfg: ReductionConfig,
    pol: str | None,
    q_min: float | None,
    q_max: float | None,
    integration_method: str | None,
    results_root: Path | None,
) -> None:
    """Apply explicitly-provided CLI options over the loaded config, in place."""
    if pol is not None:
        cfg.polarization = pol
    if q_min is not None:
        cfg.q_min = q_min
    if q_max is not None:
        cfg.q_max = q_max
    if integration_method is not None:
        cfg.integration_method = integration_method
    if results_root is not None:
        cfg.results_root = results_root


@app.command()
def main(
    file_filter: int = typer.Argument(..., help="Scan number, e.g. 89019."),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config TOML (default: ./reduction_config.toml, then bundled default).",
    ),
    pol: str | None = typer.Option(
        None, "--pol", help="Override the config's polarization selection."
    ),
    q_min: float | None = typer.Option(
        None, "--q-min", help="Override the ISI integral lower q bound."
    ),
    q_max: float | None = typer.Option(
        None, "--q-max", help="Override the ISI integral upper q bound."
    ),
    energy: list[float] = typer.Option(
        [],
        "--energy",
        "-e",
        help="Energy (eV) to include; repeat for several, e.g. -e 285.2 -e 286.0. "
        "Matched to the nearest tenth. Default: all energies.",
    ),
    integration_method: str | None = typer.Option(
        None, "--integration-method", help="Override the pyFAI integration method."
    ),
    results_root: Path | None = typer.Option(
        None, "--results-root", help="Override the results root directory."
    ),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate line plots."),
    detector_2d: bool = typer.Option(
        False,
        "--detector-2d/--no-detector-2d",
        help="Save low-res 2D detector frames (off by default).",
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing outputs without confirmation."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print planned outputs without loading data or writing."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Reduce one scan and write .dat files and plots under results/{sample}/."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = resolve_config_path(config)
    logger.info(f"Using config: {config_path}")
    cfg = load_config(config_path)
    _apply_overrides(cfg, pol, q_min, q_max, integration_method, results_root)

    sample = cfg.sample_name(file_filter)
    out_dir = cfg.results_root / sample
    logger.info(f"Sample {sample!r} (scan {file_filter}); results -> {out_dir}")

    planned = _planned_outputs(out_dir, sample, plots, detector_2d)

    if dry_run:
        logger.info("[dry-run] Planned outputs:")
        for path in planned:
            logger.info(f"  {path}")
        raise typer.Exit()

    dat_files = [p for p in planned if p.suffix == ".dat"]
    existing = [p for p in dat_files if p.exists()]
    if existing and not force:
        typer.confirm(
            f"{len(existing)} output file(s) already exist in {out_dir}. Overwrite?",
            abort=True,
        )

    # Heavy imports deferred so --help and --dry-run stay fast.
    from rsoxs_reduce import plotting
    from rsoxs_reduce.pipeline import (
        build_integrator,
        build_loader,
        load_stack,
        reduce_to_iqe,
    )

    loader = build_loader(cfg)
    data = load_stack(loader, file_filter, cfg)
    if energy:
        data = select_energies(data, energy, decimals=cfg.energy_match_decimals)
        logger.info(
            f"Selected {data.sizes['energy']} energies matching {sorted(energy)}."
        )
    integrator = build_integrator(cfg)
    iqe = reduce_to_iqe(integrator, data, cfg)

    results_dir = prepare_results_dir(cfg.results_root, sample)
    header = build_header(cfg, file_filter, sample, energy, config_path)

    table = iqe_to_table(iqe)
    write_dat(
        table,
        results_dir / f"{sample}_Ivsq.dat",
        header=[*header, "", "Column 1: q; remaining columns R_{energy}: I (eV, to 0.1)."],
    )

    isi = compute_isi(iqe, q_min=cfg.q_min, q_max=cfg.q_max)
    write_dat(
        isi,
        results_dir / f"{sample}_ISIvsE.dat",
        header=[*header, "", "ISI = integral of I(q,E) * q^2 dq; NaN/negative dropped."],
    )

    if plots:
        plotting.plot_isi(isi, results_dir / f"{sample}_ISIvsE.png", dpi=cfg.plot_dpi)
        for scale in ("raw", "q2"):
            plotting.plot_iq_curves(
                iqe,
                results_dir / f"{sample}_Ivsq_overlay_{scale}.png",
                scale=scale,
                mode="overlay",
                waterfall_factor=cfg.waterfall_factor,
                colormap=cfg.colormap,
                dpi=cfg.plot_dpi,
            )
            plotting.plot_iq_curves(
                iqe,
                results_dir / f"{sample}_Ivsq_waterfall_{scale}.png",
                scale=scale,
                mode="waterfall",
                waterfall_factor=cfg.waterfall_factor,
                colormap=cfg.colormap,
                dpi=cfg.plot_dpi,
            )

    if detector_2d:
        plotting.save_detector_frames(
            data,
            results_dir / "detector_2d",
            polarization=cfg.polarization,
            vmin=cfg.detector_vmin,
            vmax=cfg.detector_vmax,
            dpi=cfg.detector_dpi,
        )

    logger.info("Reduction complete.")


def cli() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    app()
