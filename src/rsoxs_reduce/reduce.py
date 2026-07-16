"""Typer CLI: reduce a single ALS 11.0.1.2 RSoXS energy scan by scan number."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import typer
import xarray as xr

from rsoxs_reduce.analysis import (
    apply_thickness,
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
            out_dir / f"{sample}_mask_check.png",
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


def _thickness_params(cfg: ReductionConfig) -> tuple[float | None, float]:
    """Return ``(thickness_cm, relative_uncertainty)`` for the configured thickness.

    ``thickness_cm`` is None when no thickness is set (no normalization); the
    relative uncertainty is 0 unless a thickness uncertainty is also set.
    """
    thickness_cm = cfg.sample_thickness_cm
    if thickness_cm is None:
        return None, 0.0
    unc_cm = cfg.sample_thickness_uncertainty_cm
    return thickness_cm, (unc_cm / thickness_cm if unc_cm is not None else 0.0)


@dataclass
class _StackResult:
    """Per-stack reduced arrays (and uncertainties) accumulated across batches."""

    iqe: xr.DataArray
    iqe_sigma: xr.DataArray | None
    slices: dict[float, xr.DataArray]
    slice_sigmas: dict[float, xr.DataArray | None]
    anisotropy: xr.DataArray | None
    anisotropy_sigma: xr.DataArray | None


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
) -> "_StackResult":
    """Reduce one image stack and stream its per-energy outputs to disk.

    Handles either the whole scan or a single batch. Per-energy files (I(q,chi)
    tables/maps and 2D detector frames) are written here so nothing large is
    accumulated; the small chi-average, chi-slice, and anisotropy arrays (and
    their uncertainties when ``return_sigma`` is set) are returned for the
    caller to concatenate along energy.

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
        A ``_StackResult`` with the chi-average, chi-slice, and anisotropy
        arrays and their uncertainties (uncertainties are None unless
        ``return_sigma`` is set).
    """
    from rsoxs_reduce import plotting
    from rsoxs_reduce.pipeline import reduce_to_iqce

    iqce, iqce_sigma = reduce_to_iqce(integrator, stack, cfg)
    with_sigma = iqce_sigma is not None

    if with_sigma:
        iqe_value, iqe_sigma_value = chi_average(iqce, sigma=iqce_sigma)
        iqe = _ensure_energy_dim(iqe_value)
        iqe_sigma = _ensure_energy_dim(iqe_sigma_value)
    else:
        iqe = _ensure_energy_dim(chi_average(iqce))
        iqe_sigma = None

    slices: dict[float, xr.DataArray] = {}
    slice_sigmas: dict[float, xr.DataArray | None] = {}
    for angle in cfg.chi_slices:
        key = float(angle)
        if with_sigma:
            value, sigma = slice_chi(iqce, key, cfg.chi_width, sigma=iqce_sigma)
            slices[key] = _ensure_energy_dim(value)
            slice_sigmas[key] = _ensure_energy_dim(sigma)
        else:
            slices[key] = _ensure_energy_dim(slice_chi(iqce, key, cfg.chi_width))
            slice_sigmas[key] = None

    anisotropy: xr.DataArray | None = None
    anisotropy_sigma: xr.DataArray | None = None
    if cfg.anisotropy or cfg.anisotropy_plot or cfg.integrated_anisotropy:
        if with_sigma:
            value, sigma = compute_anisotropy(iqce, chi_width=cfg.chi_width, sigma=iqce_sigma)
            anisotropy = _ensure_energy_dim(value)
            anisotropy_sigma = _ensure_energy_dim(sigma)
        else:
            anisotropy = _ensure_energy_dim(compute_anisotropy(iqce, chi_width=cfg.chi_width))

    energies = sorted({float(value) for value in iqce["energy"].values.ravel()})
    if cfg.iqchi_dat:
        # The 2D .dat carries the thickness-normalized intensity; the heatmap
        # plot below stays on the raw detector scale for display.
        thickness_cm, rel_t = _thickness_params(cfg)
        if thickness_cm is None:
            iqce_out, iqce_sigma_out = iqce, iqce_sigma
        else:
            iqce_out, iqce_sigma_out = apply_thickness(
                iqce, iqce_sigma, thickness_cm, rel_t
            )
        for value in energies:
            write_dat(
                iqchi_to_table(iqce_out, value, sigma=iqce_sigma_out),
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
    return _StackResult(
        iqe=iqe,
        iqe_sigma=iqe_sigma,
        slices=slices,
        slice_sigmas=slice_sigmas,
        anisotropy=anisotropy,
        anisotropy_sigma=anisotropy_sigma,
    )


def _write_scan_outputs(
    cfg: ReductionConfig,
    sample: str,
    results_dir: Path,
    header: list[str],
    make_plots: bool,
    iqe: xr.DataArray,
    iqe_sigma: xr.DataArray | None,
    slice_full: dict[float, xr.DataArray],
    slice_sigma_full: dict[float, xr.DataArray | None],
    anisotropy: xr.DataArray | None,
    anisotropy_sigma: xr.DataArray | None,
) -> None:
    """Write the whole-scan outputs from the accumulated reduced arrays.

    These are the outputs that span all energies (wide tables and the
    overlay/waterfall/ISI/anisotropy plots) and so are produced once, after
    every batch has been reduced. When uncertainties are present they are
    written as paired columns and drawn on the plots.

    Args:
        cfg: Reduction configuration.
        sample: Resolved sample name.
        results_dir: Per-sample results directory.
        header: Reproducibility header lines.
        make_plots: Whether plotting is enabled.
        iqe: Chi-average I(q, E) for the full scan.
        iqe_sigma: Uncertainty of ``iqe``, or None.
        slice_full: Map of chi-slice angle to I(q, E) for the full scan.
        slice_sigma_full: Map of chi-slice angle to its uncertainty, or None.
        anisotropy: Anisotropy A(q, E) for the full scan, or None.
        anisotropy_sigma: Uncertainty of ``anisotropy``, or None.
    """
    from rsoxs_reduce import plotting

    thickness_cm, rel_t = _thickness_params(cfg)
    thickness_note = (
        f" Intensity normalized by sample thickness t = {thickness_cm:g} cm."
        if thickness_cm is not None else ""
    )

    # ISI from the (raw) chi-average; compute_isi converts q to cm^-1 for the
    # q^2 dq factor. Thickness normalization (a global scale, so the thickness
    # uncertainty is a correlated relative term) is applied to the final ISI.
    isi = compute_isi(iqe, q_min=cfg.q_min, q_max=cfg.q_max, sigma=iqe_sigma)
    if thickness_cm is not None:
        isi["ISI"] = isi["ISI"] / thickness_cm
        if "dISI" in isi.columns:
            isi["dISI"] = (
                (isi["dISI"] / thickness_cm) ** 2 + (isi["ISI"] * rel_t) ** 2
            ) ** 0.5
    isi_units = "cm^-4" if thickness_cm is not None else "(intensity units) * cm^-3"
    write_dat(
        isi,
        results_dir / f"{sample}_ISIvsE.dat",
        header=[*header, "", f"ISI = integral of I(q,E) * q^2 dq (q in cm^-1), units {isi_units}; NaN/negative dropped.{thickness_note}"],
    )

    # Thickness-normalize the chi-average and slices for the wide table + plots.
    if thickness_cm is not None:
        iqe, iqe_sigma = apply_thickness(iqe, iqe_sigma, thickness_cm, rel_t)
        for angle in list(slice_full):
            slice_full[angle], slice_sigma_full[angle] = apply_thickness(
                slice_full[angle], slice_sigma_full.get(angle), thickness_cm, rel_t
            )

    slice_items = [(angle, slice_full[angle]) for angle in sorted(slice_full)]
    slice_sigma_items = [
        (angle, slice_sigma_full[angle])
        for angle in sorted(slice_full)
        if slice_sigma_full.get(angle) is not None
    ]
    slice_note = (
        f" chi slices {cfg.chi_slices} (+/-{cfg.chi_width} deg) appended as "
        "R_{energy}_chi{angle}." if cfg.chi_slices else ""
    )
    write_dat(
        assemble_iq_table(
            iqe,
            slice_items,
            chi_avg_sigma=iqe_sigma,
            slice_sigma_items=slice_sigma_items,
        ),
        results_dir / f"{sample}_Ivsq.dat",
        header=[*header, "", f"Column 1: q; R_{{energy}}: chi-averaged I (eV, to 0.1).{slice_note}{thickness_note}"],
    )

    # Anisotropy is a ratio, so it is independent of sample thickness (t cancels).
    integrated = None
    if cfg.anisotropy:
        write_dat(
            iqe_to_table(anisotropy, prefix="A", sigma=anisotropy_sigma),
            results_dir / f"{sample}_Avsq.dat",
            header=[*header, "", "A = (I_para - I_perp)/(I_para + I_perp); para=0 deg, perp=-90 deg; thickness-independent."],
        )
    if cfg.integrated_anisotropy:
        integrated = integrate_anisotropy(
            anisotropy, q_min=cfg.q_min, q_max=cfg.q_max, sigma=anisotropy_sigma
        )
        write_dat(
            integrated,
            results_dir / f"{sample}_intAvsE.dat",
            header=[*header, "", "int_A = integral of A(q,E) dq over [q_min, q_max] (q in A^-1); thickness-independent."],
        )

    if not make_plots:
        return

    slice_arrays = {
        f"chi {float(angle):g}": slice_full[angle] for angle in sorted(slice_full)
    }
    slice_sigma_arrays = {
        f"chi {float(angle):g}": slice_sigma_full[angle]
        for angle in sorted(slice_full)
        if slice_sigma_full.get(angle) is not None
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
            sigma=iqe_sigma,
            slice_sigmas=slice_sigma_arrays,
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
            sigma=iqe_sigma,
            slice_sigmas=slice_sigma_arrays,
        )
    if cfg.anisotropy_plot:
        plotting.plot_anisotropy(
            anisotropy,
            results_dir / f"{sample}_Avsq.png",
            colormap=cfg.colormap,
            dpi=cfg.plot_dpi,
            sigma=anisotropy_sigma,
        )
    if cfg.integrated_anisotropy:
        plotting.plot_integrated_anisotropy(
            integrated,
            results_dir / f"{sample}_intAvsE.png",
            dpi=cfg.plot_dpi,
        )


def _first_image(stack: xr.DataArray) -> "object":
    """Return the first 2D detector frame from a raw stack (for the mask preview)."""
    extra = {dim: 0 for dim in stack.dims if dim not in ("pix_x", "pix_y")}
    return stack.isel(extra).values


def _run_mask_check(
    cfg: ReductionConfig, file_filter: int, sample: str, out_dir: Path
) -> None:
    """Write a mask-over-image preview for the scan and nothing else.

    Loads the first detector frame and the oriented mask, saves the overlay so
    the user can confirm (and adjust) the mask orientation, then returns.

    Args:
        cfg: Reduction configuration.
        file_filter: Scan number.
        sample: Resolved sample name.
        out_dir: Directory to write the preview into.
    """
    from rsoxs_reduce import plotting
    from rsoxs_reduce.pipeline import (
        build_loader,
        enumerate_scan,
        load_mask,
        prepare_loader,
    )

    _validate_or_abort(cfg)
    loader = build_loader(cfg)
    prepare_loader(loader, file_filter, cfg)
    entries, _ = enumerate_scan(loader, file_filter, cfg)
    if not entries:
        logger.error("No FITS files found for the scan; cannot check the mask.")
        raise typer.Exit(code=1)

    image = loader.loadSingleImage(str(entries[0][0]), coords={})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample}_mask_check.png"
    plotting.plot_mask_overlay(
        image.values,
        load_mask(cfg),
        path,
        vmin=cfg.detector_vmin,
        vmax=cfg.detector_vmax,
        dpi=cfg.plot_dpi,
    )
    logger.info(f"Mask-check image written to {path}")


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
        "Matched to the nearest tenth. Overrides [reduction].energies in the "
        "config. Default: the config list, or all energies if it is empty.",
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
    check_mask: bool = typer.Option(
        False,
        "--check-mask",
        help="Only write a mask-over-first-image preview to verify mask "
        "orientation, then exit (no reduction).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Reduce one scan and write .dat files and plots under results/{sample}/."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start_time = time.monotonic()

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

    if check_mask:
        _run_mask_check(cfg, file_filter, sample, out_dir)
        raise typer.Exit()

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

    # Energy selection: the --energy CLI option overrides the config's
    # [reduction].energies list; empty means all energies.
    requested_energies = list(energy) if energy else list(cfg.energies)

    loader = build_loader(cfg)
    integrator = build_integrator(cfg)
    results_dir = prepare_results_dir(cfg.results_root, sample)
    header = build_header(cfg, file_filter, sample, requested_energies, config_path)

    iqchi_dir = results_dir / "iqchi"
    if cfg.iqchi_dat or cfg.iqchi_plot:
        iqchi_dir.mkdir(parents=True, exist_ok=True)

    # Build the sequence of raw stacks: one per energy-batch, or a single
    # whole-scan stack when batching is off.
    if cfg.batch_size > 0:
        prepare_loader(loader, file_filter, cfg)
        entries, energies = enumerate_scan(loader, file_filter, cfg)
        entries, energies = filter_scan_energies(
            entries, energies, requested_energies, cfg.energy_match_decimals
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
        if requested_energies:
            data = select_energies(
                data, requested_energies, decimals=cfg.energy_match_decimals
            )
            logger.info(
                f"Selected {data.sizes['energy']} energies matching "
                f"{sorted(requested_energies)}."
            )
        stacks = iter([data])

    # Reduce each stack, streaming per-energy outputs and accumulating the small
    # reduced arrays. Each raw stack is released before the next is loaded.
    results: list[_StackResult] = []
    first_image = None
    for stack in stacks:
        if first_image is None and plots:
            first_image = _first_image(stack)
        results.append(
            _reduce_stack(
                integrator, stack, cfg, sample, results_dir, header, iqchi_dir,
                plots, detector_2d,
            )
        )
        del stack

    def _concat(parts: list[xr.DataArray]) -> xr.DataArray:
        """Concatenate per-stack arrays along energy (no-op for a single stack)."""
        return xr.concat(parts, dim="energy").sortby("energy")

    with_sigma = bool(results) and results[0].iqe_sigma is not None

    iqe = _concat([r.iqe for r in results])
    iqe_sigma = _concat([r.iqe_sigma for r in results]) if with_sigma else None

    slice_full: dict[float, xr.DataArray] = {}
    slice_sigma_full: dict[float, xr.DataArray | None] = {}
    for angle in cfg.chi_slices:
        key = float(angle)
        slice_full[key] = _concat([r.slices[key] for r in results])
        slice_sigma_full[key] = (
            _concat([r.slice_sigmas[key] for r in results]) if with_sigma else None
        )

    aniso_parts = [r.anisotropy for r in results if r.anisotropy is not None]
    anisotropy = _concat(aniso_parts) if aniso_parts else None
    aniso_sigma_parts = [
        r.anisotropy_sigma for r in results if r.anisotropy_sigma is not None
    ]
    anisotropy_sigma = _concat(aniso_sigma_parts) if aniso_sigma_parts else None

    # Rebuild the header with the total processing time for the summary files.
    final_header = build_header(
        cfg,
        file_filter,
        sample,
        requested_energies,
        config_path,
        elapsed_seconds=time.monotonic() - start_time,
    )
    _write_scan_outputs(
        cfg, sample, results_dir, final_header, plots, iqe, iqe_sigma, slice_full,
        slice_sigma_full, anisotropy, anisotropy_sigma,
    )

    # Mask-over-image preview, so the mask orientation is always verifiable.
    if plots and first_image is not None:
        from rsoxs_reduce import plotting

        plotting.plot_mask_overlay(
            first_image,
            integrator.mask,
            results_dir / f"{sample}_mask_check.png",
            vmin=cfg.detector_vmin,
            vmax=cfg.detector_vmax,
            dpi=cfg.plot_dpi,
        )

    logger.info(f"Reduction complete in {time.monotonic() - start_time:.1f} s.")


def cli() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    app()
