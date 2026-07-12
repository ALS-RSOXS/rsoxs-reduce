"""Typer CLI: reduce a single ALS 11.0.1.2 RSoXS energy scan by scan number."""

import logging
from pathlib import Path

import typer
import xarray as xr

from rsoxs_reduce.analysis import (
    assemble_iq_table,
    chi_average,
    compute_anisotropy,
    compute_isi,
    integrate_anisotropy,
    iqchi_to_table,
    iqe_to_table,
    select_energies,
    slice_chi,
)
from rsoxs_reduce.config import (
    ReductionConfig,
    load_config,
    resolve_config_path,
    validate_inputs,
)
from rsoxs_reduce.io_utils import prepare_results_dir, write_dat
from rsoxs_reduce.metadata import build_header

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Reduce an ALS 11.0.1.2 RSoXS energy scan to I(q,E), ISI(E), and plots.",
)


def _planned_outputs(
    cfg: ReductionConfig, out_dir: Path, sample: str, plots: bool, detector_2d: bool
) -> list[Path]:
    """Return the list of paths a full run would produce (for --dry-run).

    Per-energy I(q, chi) files are not enumerated (their energies are unknown
    until the data is loaded); the containing ``iqchi/`` directory is listed
    instead.
    """
    outputs = [
        out_dir / f"{sample}_Ivsq.dat",
        out_dir / f"{sample}_ISIvsE.dat",
    ]
    if cfg.anisotropy:
        outputs.append(out_dir / f"{sample}_Avsq.dat")
    if cfg.integrated_anisotropy:
        outputs.append(out_dir / f"{sample}_intAvsE.dat")
    if cfg.iqchi_dat or cfg.iqchi_plot:
        outputs.append(out_dir / "iqchi")
    if plots:
        outputs += [
            out_dir / f"{sample}_ISIvsE.png",
            out_dir / f"{sample}_Ivsq_overlay_raw.png",
            out_dir / f"{sample}_Ivsq_overlay_q2.png",
            out_dir / f"{sample}_Ivsq_waterfall_raw.png",
            out_dir / f"{sample}_Ivsq_waterfall_q2.png",
        ]
        if cfg.anisotropy_plot:
            outputs.append(out_dir / f"{sample}_Avsq.png")
        if cfg.integrated_anisotropy:
            outputs.append(out_dir / f"{sample}_intAvsE.png")
    if detector_2d:
        outputs.append(out_dir / "detector_2d")
    return outputs


def _validate_or_abort(cfg: ReductionConfig) -> None:
    """Abort with a non-zero exit code if any configured input path is missing.

    Args:
        cfg: The reduction configuration to validate.

    Raises:
        typer.Exit: With code 1 if one or more input paths are invalid.
    """
    problems = validate_inputs(cfg)
    if problems:
        logger.error(
            f"Config validation failed ({len(problems)} problem(s)); "
            "nothing was loaded or written:"
        )
        for problem in problems:
            logger.error(f"  - {problem}")
        logger.error("Check the [paths] section of your config and try again.")
        raise typer.Exit(code=1)


def _apply_overrides(
    cfg: ReductionConfig,
    pol: str | None,
    q_min: float | None,
    q_max: float | None,
    integration_method: str | None,
    results_root: Path | None,
    batch_size: int | None,
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
    if batch_size is not None:
        cfg.batch_size = batch_size


def _ensure_energy_dim(arr: xr.DataArray) -> xr.DataArray:
    """Promote a scalar ``energy`` coord to a length-1 dimension for concat."""
    return arr if "energy" in arr.dims else arr.expand_dims("energy")


def _reduce_stack(
    integrator: object,
    stack: xr.DataArray,
    cfg: ReductionConfig,
    sample: str,
    results_dir: Path,
    header: list[str],
    iqchi_dir: Path,
    make_plots: bool,
    detector_2d: bool,
) -> tuple[xr.DataArray, dict[float, xr.DataArray], xr.DataArray | None]:
    """Reduce one image stack and stream its per-energy outputs to disk.

    Handles either the whole scan or a single batch. Per-energy files (I(q,chi)
    tables/maps and 2D detector frames) are written here so nothing large is
    accumulated; the small chi-average, chi-slice, and anisotropy arrays are
    returned for the caller to concatenate along energy.

    Args:
        integrator: A configured energy-series integrator.
        stack: The unstacked image stack for this scan or batch.
        cfg: Reduction configuration.
        sample: Resolved sample name.
        results_dir: Per-sample results directory.
        header: Reproducibility header lines.
        iqchi_dir: Directory for per-energy I(q,chi) output.
        make_plots: Whether plotting is enabled.
        detector_2d: Whether to save 2D detector frames.

    Returns:
        The chi-average I(q, E), a map of chi-slice angle to I(q, E), and the
        anisotropy A(q, E) (or None if no anisotropy output was requested).
    """
    from rsoxs_reduce import plotting
    from rsoxs_reduce.pipeline import reduce_to_iqce

    iqce = reduce_to_iqce(integrator, stack, cfg)
    iqe = _ensure_energy_dim(chi_average(iqce))
    slice_map = {
        float(angle): _ensure_energy_dim(slice_chi(iqce, float(angle), cfg.chi_width))
        for angle in cfg.chi_slices
    }
    anisotropy: xr.DataArray | None = None
    if cfg.anisotropy or cfg.anisotropy_plot or cfg.integrated_anisotropy:
        anisotropy = _ensure_energy_dim(compute_anisotropy(iqce, chi_width=cfg.chi_width))

    energies = sorted({float(value) for value in iqce["energy"].values.ravel()})
    if cfg.iqchi_dat:
        for value in energies:
            write_dat(
                iqchi_to_table(iqce, value),
                iqchi_dir / f"{sample}_Iqchi_E{value:.1f}eV.dat",
                header=[*header, "", f"I(q, chi) at E = {value:.1f} eV; columns chi_{{angle}} in deg."],
            )
    if make_plots and cfg.iqchi_plot:
        plotting.plot_iqchi_maps(
            iqce,
            iqchi_dir,
            vmin=cfg.detector_vmin,
            vmax=cfg.detector_vmax,
            colormap=cfg.colormap,
            dpi=cfg.plot_dpi,
        )
    if detector_2d:
        plotting.save_detector_frames(
            stack,
            results_dir / "detector_2d",
            polarization=cfg.polarization,
            vmin=cfg.detector_vmin,
            vmax=cfg.detector_vmax,
            dpi=cfg.detector_dpi,
        )
    return iqe, slice_map, anisotropy


def _write_scan_outputs(
    cfg: ReductionConfig,
    sample: str,
    results_dir: Path,
    header: list[str],
    make_plots: bool,
    iqe: xr.DataArray,
    slice_full: dict[float, xr.DataArray],
    anisotropy: xr.DataArray | None,
) -> None:
    """Write the whole-scan outputs from the accumulated reduced arrays.

    These are the outputs that span all energies (wide tables and the
    overlay/waterfall/ISI/anisotropy plots) and so are produced once, after
    every batch has been reduced.

    Args:
        cfg: Reduction configuration.
        sample: Resolved sample name.
        results_dir: Per-sample results directory.
        header: Reproducibility header lines.
        make_plots: Whether plotting is enabled.
        iqe: Chi-average I(q, E) for the full scan.
        slice_full: Map of chi-slice angle to I(q, E) for the full scan.
        anisotropy: Anisotropy A(q, E) for the full scan, or None.
    """
    from rsoxs_reduce import plotting

    slice_items = [(angle, slice_full[angle]) for angle in sorted(slice_full)]
    slice_note = (
        f" chi slices {cfg.chi_slices} (+/-{cfg.chi_width} deg) appended as "
        "R_{energy}_chi{angle}." if cfg.chi_slices else ""
    )
    write_dat(
        assemble_iq_table(iqe, slice_items),
        results_dir / f"{sample}_Ivsq.dat",
        header=[*header, "", f"Column 1: q; R_{{energy}}: chi-averaged I (eV, to 0.1).{slice_note}"],
    )

    isi = compute_isi(iqe, q_min=cfg.q_min, q_max=cfg.q_max)
    write_dat(
        isi,
        results_dir / f"{sample}_ISIvsE.dat",
        header=[*header, "", "ISI = integral of I(q,E) * q^2 dq; NaN/negative dropped."],
    )

    integrated = None
    if cfg.anisotropy:
        write_dat(
            iqe_to_table(anisotropy, prefix="A"),
            results_dir / f"{sample}_Avsq.dat",
            header=[*header, "", "A = (I_para - I_perp)/(I_para + I_perp); para=0 deg, perp=-90 deg."],
        )
    if cfg.integrated_anisotropy:
        integrated = integrate_anisotropy(anisotropy, q_min=cfg.q_min, q_max=cfg.q_max)
        write_dat(
            integrated,
            results_dir / f"{sample}_intAvsE.dat",
            header=[*header, "", "int_A = integral of A(q,E) dq over [q_min, q_max]."],
        )

    if not make_plots:
        return

    slice_arrays = {
        f"chi {float(angle):g}": slice_full[angle] for angle in sorted(slice_full)
    }
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
            slices=slice_arrays,
        )
        plotting.plot_iq_curves(
            iqe,
            results_dir / f"{sample}_Ivsq_waterfall_{scale}.png",
            scale=scale,
            mode="waterfall",
            waterfall_factor=cfg.waterfall_factor,
            colormap=cfg.colormap,
            dpi=cfg.plot_dpi,
            slices=slice_arrays,
        )
    if cfg.anisotropy_plot:
        plotting.plot_anisotropy(
            anisotropy,
            results_dir / f"{sample}_Avsq.png",
            colormap=cfg.colormap,
            dpi=cfg.plot_dpi,
        )
    if cfg.integrated_anisotropy:
        plotting.plot_integrated_anisotropy(
            integrated,
            results_dir / f"{sample}_intAvsE.png",
            dpi=cfg.plot_dpi,
        )


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
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Energies to load and reduce per batch to bound memory (0 = whole "
        "scan at once). Overrides [processing].batch_size.",
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
    _apply_overrides(
        cfg, pol, q_min, q_max, integration_method, results_root, batch_size
    )

    # Anchor a relative results_root to the current working directory and make
    # the destination absolute, so the log shows exactly where output lands.
    cfg.results_root = (Path.cwd() / cfg.results_root).resolve()

    sample = cfg.sample_name(file_filter)
    out_dir = cfg.results_root / sample
    logger.info(f"Sample {sample!r} (scan {file_filter}); results -> {out_dir}")

    planned = _planned_outputs(cfg, out_dir, sample, plots, detector_2d)

    if dry_run:
        logger.info("[dry-run] Planned outputs:")
        for path in planned:
            logger.info(f"  {path}")
        raise typer.Exit()

    _validate_or_abort(cfg)

    dat_files = [p for p in planned if p.suffix == ".dat"]
    existing = [p for p in dat_files if p.exists()]
    if existing and not force:
        typer.confirm(
            f"{len(existing)} output file(s) already exist in {out_dir}. Overwrite?",
            abort=True,
        )

    # Heavy imports deferred so --help and --dry-run stay fast.
    from rsoxs_reduce.pipeline import (
        build_integrator,
        build_loader,
        enumerate_scan,
        filter_scan_energies,
        iter_file_batches,
        load_file_batch,
        load_stack,
        prepare_loader,
        setup_common_qgrid,
    )

    loader = build_loader(cfg)
    integrator = build_integrator(cfg)
    results_dir = prepare_results_dir(cfg.results_root, sample)
    header = build_header(cfg, file_filter, sample, energy, config_path)

    iqchi_dir = results_dir / "iqchi"
    if cfg.iqchi_dat or cfg.iqchi_plot:
        iqchi_dir.mkdir(parents=True, exist_ok=True)

    # Build the sequence of raw stacks: one per energy-batch, or a single
    # whole-scan stack when batching is off.
    if cfg.batch_size > 0:
        prepare_loader(loader, file_filter, cfg)
        entries, energies = enumerate_scan(loader, file_filter, cfg)
        entries, energies = filter_scan_energies(
            entries, energies, energy, cfg.energy_match_decimals
        )
        if not energies:
            logger.error("No energies to process after filtering; nothing written.")
            raise typer.Exit(code=1)
        setup_common_qgrid(integrator, energies)
        logger.info(
            f"Batch processing {len(energies)} energies in batches of "
            f"{cfg.batch_size} (scan {file_filter})."
        )

        def _stacks() -> "Iterator[xr.DataArray]":
            for batch_energies, batch_files in iter_file_batches(
                entries, energies, cfg.batch_size
            ):
                logger.info(
                    f"Loading batch: {len(batch_energies)} energies "
                    f"{batch_energies[0]:.1f}-{batch_energies[-1]:.1f} eV, "
                    f"{len(batch_files)} files."
                )
                yield load_file_batch(loader, batch_files, cfg.stack_dims)

        stacks: "Iterator[xr.DataArray]" = _stacks()
    else:
        data = load_stack(loader, file_filter, cfg)
        if energy:
            data = select_energies(data, energy, decimals=cfg.energy_match_decimals)
            logger.info(
                f"Selected {data.sizes['energy']} energies matching {sorted(energy)}."
            )
        stacks = iter([data])

    # Reduce each stack, streaming per-energy outputs and accumulating the small
    # reduced arrays. Each raw stack is released before the next is loaded.
    iqe_parts: list[xr.DataArray] = []
    slice_parts: dict[float, list[xr.DataArray]] = {
        float(angle): [] for angle in cfg.chi_slices
    }
    aniso_parts: list[xr.DataArray] = []
    for stack in stacks:
        iqe_b, slice_map_b, aniso_b = _reduce_stack(
            integrator, stack, cfg, sample, results_dir, header, iqchi_dir,
            plots, detector_2d,
        )
        iqe_parts.append(iqe_b)
        for angle in cfg.chi_slices:
            slice_parts[float(angle)].append(slice_map_b[float(angle)])
        if aniso_b is not None:
            aniso_parts.append(aniso_b)
        del stack

    # Concatenate along energy (a no-op for a single whole-scan stack).
    iqe = xr.concat(iqe_parts, dim="energy").sortby("energy")
    slice_full = {
        float(angle): xr.concat(slice_parts[float(angle)], dim="energy").sortby("energy")
        for angle in cfg.chi_slices
    }
    anisotropy = (
        xr.concat(aniso_parts, dim="energy").sortby("energy") if aniso_parts else None
    )

    _write_scan_outputs(
        cfg, sample, results_dir, header, plots, iqe, slice_full, anisotropy
    )

    logger.info("Reduction complete.")


def cli() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    app()
