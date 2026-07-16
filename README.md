# rsoxs-reduce

Command-line reduction for ALS 11.0.1.2 RSoXS energy scans, built on
[PyHyperScattering](https://github.com/usnistgov/pyhyperscattering).

Given a scan number, it loads the FITS series, applies darks and I0/I1
normalization, azimuthally integrates each energy to `I(q, χ)`, and writes
tab-delimited `.dat` files and plots. Beyond the basic chi-averaged `I(q, E)`
and integrated scattering intensity, it can also produce chi slices, azimuthal
anisotropy, full 2D `I(q, χ)` maps, and propagated uncertainties — all driven
by an editable TOML config.

## Install / sync

Self-contained `uv` project. PyHyperScattering is pulled from the
[ALS-RSOXS fork](https://github.com/ALS-RSOXS/PyHyperScattering) and pinned in
`uv.lock`.

```bash
git clone https://github.com/ALS-RSOXS/rsoxs-reduce.git
cd rsoxs-reduce
uv sync
```

rsoxs-reduce is best used as a `uv tool` that can be called from any directory,
keeping your individual projects and datasets self-contained. Output is written
to a `./results` folder in whatever directory you run it from.

```bash
cd rsoxs-reduce
uv tool install .
```

Re-run `uv tool install --reinstall .` after pulling changes to update the
installed command.

## Configuration

All beamtime-specific settings live in an editable `reduction_config.toml`
(paths, detector geometry, mask orientation, reduction options, plotting limits,
the metadata filter, and the scan-number → sample-name `[samples]` map). The CLI
selects one, in order:

1. the path given to `--config`,
2. `reduction_config.toml` in the current working directory,
3. the bundled `default_config.toml` (placeholder template).

Copy `src/rsoxs_reduce/default_config.toml` to `reduction_config.toml` next to
your data and edit it. On Windows, use single-quoted TOML literal strings for
paths (e.g. `'C:\data\scan'`). Because the config is read from the working
directory, you can keep a different one per beamtime folder and run the tool
from anywhere.

The **full resolved configuration is embedded (as valid TOML) into the header of
every `.dat` file**, along with a timestamp, the command line, and the versions
of PyHyperScattering, pyFAI, numpy, scipy, xarray, and pandas — so any output can
be traced back to an exactly reproducible run.

A full reference of every config key, its default, and what it does is in
[Config reference](#config-reference) at the end.

## Run

Pass the scan number (the `file_filter` value from the notebook):

```bash
uv run rsoxs-reduce 89019          # or just `rsoxs-reduce 89019` once installed
```

Before validating and loading anything, the tool checks that every configured
input path exists (data directory, FITS/dark sub-directories, I0, mask, and at
least one `.fits` file), and aborts with a clear list if any are missing.

## Reduction model (`als` correction)

With `corr_mode = "als"` and all normalization inputs supplied, the loader
converts raw detector counts to an absolute differential scattering cross
section per unit solid angle:

$$\frac{d\sigma}{d\Omega}\;\left[\tfrac{1}{\mathrm{cm}\cdot\mathrm{sr}}\right] = R_{r}\;\left[\tfrac{\mathrm{nA}}{\mathrm{ADU}}\right]\;\cdot\;\frac{I_{\mathrm{det}}(q,E)\;[\mathrm{ADU}\cdot\mathrm{s}]}{I_{0}(E)\;[\mathrm{nA}]\;\cdot\;\tau\;[\mathrm{s}]}\;\cdot\;\Big(T_{s}(E)\cdot t\,[\mathrm{cm}]\cdot\Omega\,[\mathrm{sr}]\Big)^{-1}$$

which is equivalently the single fraction

$$\frac{d\sigma}{d\Omega} = \frac{R_{r}\,I_{\mathrm{det}}(q,E)}{I_{0}(E)\,\tau\,T_{s}(E)\,t\,\Omega}.$$

The responsivity $R_{r}$ makes the units consistent: $R_{r}\,I_{\mathrm{det}}/(I_{0}\tau)$
is dimensionless (${\mathrm{nA}}/{\mathrm{ADU}}\cdot{\mathrm{ADU}\cdot\mathrm{s}}/({\mathrm{nA}\cdot\mathrm{s}})$),
leaving $\mathrm{cm^{-1}\,sr^{-1}}$ from $(t\,\Omega)^{-1}$.

| Symbol | Meaning | Units |
|---|---|---|
| $d\sigma/d\Omega$ | Differential scattering cross section | $\mathrm{cm^{-1}\,sr^{-1}}$ |
| $R_{r}$ | Relative responsivity (see below) | $\mathrm{nA}/\mathrm{ADU}$ |
| $I_{\mathrm{det}}(q,E)$ | Counts from the area detector | $\mathrm{ADU}\cdot\mathrm{s}$ |
| $I_{0}(E)$ | Double-normalized incident intensity (see below) | $\mathrm{nA}$ |
| $\tau$ | Exposure time | $\mathrm{s}$ |
| $T_{s}(E)$ | Transmission through the sample (see below) | dimensionless |
| $t$ | Sample thickness | $\mathrm{cm}$ |
| $\Omega$ | Solid angle | $\mathrm{sr}$ |

### Double-normalized incident intensity $I_{0}(E)$

$$I_{0}(E)\;[\mathrm{nA}] = I_{0}^{\mathrm{ai3,samp}}(E)\;\cdot\;\frac{I_{0}^{\mathrm{PD,ref}}(E)}{I_{0}^{\mathrm{ai3,ref}}(E)}$$

(all three currents in $\mathrm{nA}$), where:

| Symbol | Meaning |
|---|---|
| $I_{0}^{\mathrm{ai3,samp}}(E)$ | Drain current on the upstream gold-mesh monitor during 2D data collection. |
| $I_{0}^{\mathrm{ai3,ref}}(E)$ | Drain current on the upstream gold-mesh monitor during the I0 flux measurement. |
| $I_{0}^{\mathrm{PD,ref}}(E)$ | Current on the downstream calibrated photodiode during the I0 flux measurement. |

The `ref` values come from the I0 file supplied via `[paths].i0_path`.

### Sample transmission $T_{s}(E)$

$$T_{s}(E) = \frac{I_{1}^{\mathrm{PD,samp}}(E)}{I_{1}^{\mathrm{ai3,samp}}(E)}\;\cdot\;\frac{I_{0}^{\mathrm{ai3,ref}}(E)}{I_{0}^{\mathrm{PD,ref}}(E)}$$

(dimensionless; a ratio of $\mathrm{nA}$ currents), where:

| Symbol | Meaning |
|---|---|
| $I_{1}^{\mathrm{PD,samp}}(E)$ | Current on the downstream calibrated photodiode during the I1 transmission measurement through the sample. |
| $I_{1}^{\mathrm{ai3,samp}}(E)$ | Drain current on the upstream gold-mesh monitor during the I1 transmission measurement through the sample. |
| $I_{0}^{\mathrm{ai3,ref}}(E)$ | Drain current on the upstream gold-mesh monitor during the I0 flux measurement. |
| $I_{0}^{\mathrm{PD,ref}}(E)$ | Current on the downstream calibrated photodiode during the I0 flux measurement. |

The I1 values come from the I1 file supplied via `[paths].i1_path`; the I0
reference values come from the I0 file. If `i1_path` is omitted, $T_{s}$ is not
applied (no transmission correction).

### Relative responsivity $R_{r}$

$R_{r}$ is a scale that accommodates the quantum efficiency of the 2D area
detector and of the calibrated photodiode, both of which are ill-defined at
photon energies across the carbon K-edge where these measurements are typically
made. **At present $R_{r} \equiv 1$** (it is not yet calibrated); it is kept
explicit in the equation only to mark where an absolute-intensity calibration
would enter. Future work is targeted at better calibrating the signal on an
absolute intensity scale.

### Sample thickness $t$ and ISI units

Sample thickness normalization is applied in post-processing (not in PyHyper):
the intensity-like outputs — `I(q,E)`, chi slices, `I(q,χ)`, and ISI — are
divided by the thickness `t`, converted to cm from `[reduction].thickness_units`
(default **nm**). Set `sample_thickness` to enable it; omit it to skip. The
anisotropy is a ratio, so `t` cancels and `A(q,E)`/`∫A` are unaffected.

`sample_thickness_uncertainty` is a **global, fully-correlated** relative term
$\delta t/t$: it is added in quadrature to each final reported quantity's
uncertainty (not propagated through the chi-average or q-integration, which
would wrongly shrink it), and only when `return_sigma = true`.

The ISI integral converts $q$ from $\mathrm{\AA^{-1}}$ to $\mathrm{cm^{-1}}$ for
the $q^{2}\,dq$ factor, so its $q$-contribution is in $\mathrm{cm^{-3}}$. With a
thickness-normalized intensity in $\mathrm{cm^{-1}}$, the fully reduced ISI is in
$\mathrm{cm^{-4}}$ (assuming $R_{r}$ is correct, which is still being calibrated).
The `q` coordinate written in the output files stays in $\mathrm{\AA^{-1}}$.

## Features

Everything below the basic chi-averaged `I(q, E)` + ISI output is opt-in via the
config; the defaults reproduce a plain chi-averaged reduction.

### Chi slices
Set `[output].chi_slices` to a list of azimuthal center angles (deg) to add
wedge-averaged curves alongside the chi-average. Each slice averages
`χ ± chi_width`. Slices appear as extra `R_{energy}_chi{angle}` columns in
`Ivsq.dat` and as additional linestyles on the I-vs-q plots. The chi-average is
always produced regardless.

### Anisotropy
`[output].anisotropy` writes `A(q, E) = (I_para − I_perp)/(I_para + I_perp)`
(para = χ 0°, perp = χ −90°, each a `chi_width` wedge) to `{sample}_Avsq.dat`.
`anisotropy_plot` adds an overlay plot bounded to [−1, 1], and
`integrated_anisotropy` writes and plots `∫A(q,E) dq` versus energy (using the
`q_min`/`q_max` bounds).

### Full 2D I(q, χ)
`[output].iqchi_dat` writes a 2D `q × χ` table per energy under `iqchi/`, and
`iqchi_plot` writes a per-energy heatmap. These are independent toggles.

### Uncertainty propagation
`[reduction].return_sigma = true` propagates pyFAI's per-bin uncertainty through
the whole reduction (standard first-order Gaussian propagation). Every `.dat`
gains a paired uncertainty column (`dR_…`, `dISI`, `dA_…`, `dint_A`, `dchi_…`),
the I-vs-q and anisotropy plots gain shaded ±1σ bands, and the ISI-vs-E and
∫A-vs-E plots gain error bars.

### Batch (memory-bounded) processing
`[processing].batch_size` (or `--batch-size N`) loads and reduces `N` energies at
a time, freeing each batch's raw images before the next. Results are identical
to a whole-scan run (`batch_size = 0`); batching only bounds peak memory for
large scans.

### Mask handling and verification
The mask is loaded and oriented in this package (not via PyHyper's nika path,
whose rotation is inoperative), with `[geometry]` controls `mask_rot90`,
`mask_flipud`, `mask_fliplr`, and `mask_invert`. Every run writes
`{sample}_mask_check.png` — the first detector frame (log scale) with masked
pixels shaded red in the orientation actually used. Use `--check-mask` to write
only that preview and exit, so you can iterate on the orientation flags quickly:

```bash
rsoxs-reduce 89019 --check-mask
```

### q-binning
`[reduction].q_bins` sets the number of radial q points (pyFAI `npts`). Fewer
points bundle more detector pixels per bin (coarser, linear-in-q resolution).

### Energy selection
`[reduction].energies` restricts the reduction to a list of energies (matched to
`energy_match_decimals`); empty means all. `--energy/-e` overrides it on the
command line.

## CLI options

`FILE_FILTER` (the scan number) is the one required argument. CLI flags override
the corresponding config values when provided.

| Flag | Purpose |
|---|---|
| `--config`, `-c` | Config TOML to use (default: `./reduction_config.toml`, then the bundled default). |
| `--energy`, `-e` | Reduce only these energies (eV); repeat, e.g. `-e 285.2 -e 286.0`. Matched to the nearest tenth. Overrides `[reduction].energies`. Default: the config list, or all. |
| `--pol` | Polarization coordinate to select. Overrides `[reduction].polarization`. |
| `--q-min`, `--q-max` | Restrict the q-range of the ISI (and ∫A) integral. Overrides `[reduction]`. |
| `--integration-method` | pyFAI method (e.g. `csr`, or `csr_ocl` if pyopencl is present). Overrides `[reduction]`. |
| `--results-root` | Override the output root directory. |
| `--batch-size` | Energies to load/reduce per batch to bound memory (`0` = whole scan). Overrides `[processing].batch_size`. |
| `--plots` / `--no-plots` | Generate line plots (on by default). |
| `--detector-2d` / `--no-detector-2d` | Save low-res 2D detector frames (off by default). |
| `--check-mask` | Write only a mask-over-first-image preview to verify orientation, then exit. |
| `--dry-run` | Print the planned outputs without loading data or writing files. |
| `--force` | Overwrite existing outputs without confirmation. |
| `--verbose`, `-v` | Enable debug logging. |

## Outputs

Written to `results/{sample}/` (the `results` root is anchored to the current
working directory and made absolute; it is logged at the start of each run). The
sample name comes from the `[samples]` map, falling back to `scan_{number}`.

**Always produced:**

- `{sample}_Ivsq.dat` — `q` column + one chi-averaged intensity column per energy.
- `{sample}_ISIvsE.dat` — `ISI = ∫ I(q,E)·q² dq` per energy (NaN/negative points dropped).
- With plots (default): `{sample}_mask_check.png`, `{sample}_ISIvsE.png`,
  `{sample}_Ivsq_overlay_{raw,q2}.png`, `{sample}_Ivsq_waterfall_{raw,q2}.png`.

**Conditional (config/flag):**

| Output | Enabled by |
|---|---|
| Extra `R_{energy}_chi{angle}` columns + slice curves | `[output].chi_slices` |
| `{sample}_Avsq.dat` (+ `.png` plot) | `anisotropy` (+ `anisotropy_plot`) |
| `{sample}_intAvsE.dat` + `.png` | `integrated_anisotropy` |
| `iqchi/{sample}_Iqchi_E{energy}eV.dat` | `iqchi_dat` |
| `iqchi/Iqchi_E{energy}eV.png` | `iqchi_plot` |
| Paired `d…` uncertainty columns + bands/error bars | `[reduction].return_sigma` |
| `detector_2d/` LogNorm frames per energy | `--detector-2d` |

## Config reference

Defaults are the values used when a key is omitted (the schema defaults in
`src/rsoxs_reduce/config.py`). The bundled `default_config.toml` ships
placeholder paths and zeroed geometry that you must replace for your beamtime.

### `[paths]`

| Key | Default | Description |
|---|---|---|
| `data_path` | `"data"` | Root directory holding the beamtime data. |
| `fits_subdir` | `"fits"` | Sub-directory (under `data_path`) with the FITS series. |
| `dark_subdir` | `"fits"` | Sub-directory (under `data_path`) with the dark frames. |
| `i0_path` | `"i0.txt"` | I0 normalization file (required for `als` correction mode). |
| `i1_path` | *(none)* | I1 normalization file. Omit, or set `""`/`"none"`, to reduce without the I1 correction. |
| `mask_path` | `"mask.hdf"` | Nika-format detector mask (HDF5 `M_ROIMask`, or TIFF). |
| `results_root` | `"results"` | Output root; relative paths are anchored to the current directory. |

### `[loader]` (ALS11012RSoXSLoader)

| Key | Default | Description |
|---|---|---|
| `corr_mode` | `"als"` | Intensity correction mode. |
| `exposure_offset` | `0.0` | Exposure-time offset (s) added by the loader. |
| `data_collected_after_mar2021` | `true` | Use the post-March-2021 shutter-inhibit header. |
| `dark_subtract` | `true` | Subtract dark frames. |
| `dark_pedestal` | `200` | Pedestal added back after dark subtraction. |

### `[geometry]` (nika convention)

| Key | Default | Description |
|---|---|---|
| `ni_distance` | `74.7688` | Sample–detector distance (mm). |
| `ni_bcx` | `1831.35` | Beam-center x (pixels). |
| `ni_bcy` | `2101.49` | Beam-center y (pixels). |
| `ni_tiltx` | `0.0` | Detector x tilt, in `tilt_units`. |
| `ni_tilty` | `0.0` | Detector y tilt, in `tilt_units`. |
| `tilt_units` | `"degrees"` | Unit of `ni_tiltx`/`ni_tilty`: `"degrees"` or `"radians"`. |
| `ni_pixsize_x` | `0.0096` | Pixel size in x (mm). |
| `ni_pixsize_y` | `0.0096` | Pixel size in y (mm). |
| `mask_rot90` | `1` | 90° rotations applied to the mask (numpy `rot90` k). |
| `mask_flipud` | `true` | Flip the mask vertically (after rotation). |
| `mask_fliplr` | `false` | Flip the mask horizontally (after rotation). |
| `mask_invert` | `true` | Invert the mask (Nika `M_ROIMask` 1=keep → pyFAI 1=exclude). |

### `[reduction]`

| Key | Default | Description |
|---|---|---|
| `integration_method` | `"csr"` | pyFAI integration method (`csr` CPU, `csr_ocl` OpenCL). |
| `polarization` | `"0"` | Polarization coordinate selected after integration. |
| `energy_match_decimals` | `1` | Decimal places used when matching `energies`/`--energy`. |
| `q_bins` | `500` | Number of radial q points (pyFAI `npts`); fewer = coarser q resolution. |
| `energies` | `[]` | Energies (eV) to reduce; empty = all. Overridden by `--energy`. |
| `q_min` | *(none)* | Lower q bound for the ISI/∫A integral. |
| `q_max` | *(none)* | Upper q bound for the ISI/∫A integral. |
| `return_sigma` | `false` | Propagate per-bin uncertainty into the outputs and plots. |
| `sample_thickness` | *(none)* | Sample thickness for intensity normalization (in `thickness_units`); omit to skip. |
| `sample_thickness_uncertainty` | *(none)* | Uncertainty on the thickness; folded into the propagated error only when `return_sigma` is on. |
| `thickness_units` | `"nm"` | Unit of the thickness values (`nm`/`um`/`mm`/`cm`/`m`), converted to cm before normalization. |

### `[plotting]`

| Key | Default | Description |
|---|---|---|
| `detector_vmin` | `1e-3` | Lower bound of the 2D detector log color scale. |
| `detector_vmax` | `1e5` | Upper bound of the 2D detector log color scale. |
| `detector_dpi` | `72` | Resolution for 2D detector frames. |
| `plot_dpi` | `150` | Resolution for line plots. |
| `waterfall_factor` | `10.0` | Multiplicative offset between successive waterfall curves. |
| `colormap` | `"viridis"` | Matplotlib colormap used to color curves by energy. |

### `[processing]`

| Key | Default | Description |
|---|---|---|
| `batch_size` | `0` | Energies loaded/reduced per batch to bound memory; `0` = whole scan at once. |

### `[output]`

The chi-averaged `I(q, E)` `.dat` and plots are always produced; everything here
is additive and off by default.

| Key | Default | Description |
|---|---|---|
| `chi_slices` | `[]` | Center angles (deg) for extra chi-slice columns/curves. |
| `chi_width` | `5.0` | Half-width (deg) of each chi slice and of the anisotropy wedges. |
| `anisotropy` | `false` | Write `{sample}_Avsq.dat` (A vs q per energy). |
| `anisotropy_plot` | `false` | Plot `A(q, E)`, all energies overlaid, y in [−1, 1]. |
| `integrated_anisotropy` | `false` | Write and plot `∫A(q,E) dq` versus energy. |
| `iqchi_dat` | `false` | Write the full 2D `q × χ` `.dat` per energy under `iqchi/`. |
| `iqchi_plot` | `false` | Write the full `I(q, χ)` heatmap per energy under `iqchi/`. |

### `[md_filter]`

Metadata key/value pairs a frame must match to be loaded (passed to
`loadFileSeries`). Default: `"CCD Camera Shutter Inhibit" = 0`.

### `[samples]`

Maps scan number → sample name, e.g. `12345 = "SampleA_ESCAN"`. Scans not listed
fall back to `scan_{number}`.
