# rsoxs-reduction

Command-line reduction for ALS 11.0.1.2 RSoXS energy scans, built on
[PyHyperScattering](https://github.com/usnistgov/pyhyperscattering).

It mirrors the interactive `S11_rsoxs.ipynb` workflow: load a scan (with its
sample-specific darks, I0, I1), integrate the image stack, average over `chi`,
and emit reduced data + plots for a single scan number.

## Install / sync

Self-contained `uv` project. PyHyperScattering is pulled from the
[ALS-RSOXS fork](https://github.com/ALS-RSOXS/PyHyperScattering) and pinned in
`uv.lock`.

```bash
git clone https://github.com/ALS-RSOXS/rsoxs-reduce.git
cd rsoxs-reduce
uv sync
```

## Configuration

All beamtime-specific settings live in an editable `reduction_config.toml`
(paths, detector geometry, plotting limits, the metadata filter, and the
scan-number to sample-name `[samples]` map). The CLI selects one, in order:

1. the path given to `--config`,
2. `reduction_config.toml` in the current working directory,
3. the bundled `default_config.toml` (placeholder template).

Copy `src/rsoxs_reduce/default_config.toml` to `reduction_config.toml` next
to your data and edit it. On Windows, use single-quoted TOML literal strings for
paths (e.g. `'C:\data\scan'`). Because the config is read from the working
directory, you can keep a different one per beamtime folder and run the tool
from anywhere.

The **full resolved configuration is embedded (as valid TOML) into the header of
every `.dat` file**, along with a timestamp, the command line, and the versions
of PyHyperScattering, pyFAI, numpy, scipy, xarray, and pandas — so any output can
be traced back to an exactly reproducible run.

## Run

Pass the scan number (the `file_filter` value from the notebook):

```bash
uv run rsoxs-reduce 89019
```

CLI flags (`--pol`, `--q-min`, `--q-max`, `--integration-method`,
`--results-root`) override the config file when provided.

Useful options:

| Flag | Purpose |
|---|---|
| `--energy`, `-e` | Reduce only these energies (eV); repeat the flag, e.g. `-e 285.2 -e 286.0`. Matched to the nearest tenth. Default: all. |
| `--q-min`, `--q-max` | Restrict the q-range of the ISI integral (default: full range). |
| `--pol` | Polarization coordinate to select (default `0`). |
| `--integration-method` | pyFAI method (default `csr`; use `csr_ocl` if pyopencl is present). |
| `--results-root` | Override the output root (default `Xu_2026/results`). |
| `--no-plots` | Skip line plots. |
| `--detector-2d` | Save low-res 2D detector frames (off by default). |
| `--dry-run` | Print the planned outputs without loading data or writing files. |
| `--force` | Overwrite existing outputs without confirmation. |

## Outputs

Written to `results/{sample}/`:

- `{sample}_Ivsq.dat` — q column + one intensity column per energy.
- `{sample}_ISIvsE.dat` — `ISI = ∫ I(q,E)·q² dq` per energy (NaN/negative points dropped).
- `{sample}_ISIvsE.png`, `{sample}_Ivsq_overlay_{raw,q2}.png`, `{sample}_Ivsq_waterfall_{raw,q2}.png`
- `detector_2d/` — low-resolution `LogNorm` detector frames per energy (only with `--detector-2d`).

Fixed acquisition parameters (data paths, I0/I1 paths, mask, detector geometry,
sample map) live in `reduction_config.toml`; `src/rsoxs_reduce/config.py` defines
the schema and defaults.
