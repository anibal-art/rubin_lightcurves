# Rubin Microlensing Light Curves

Tools to simulate **Rubin Observatory** microlensing light curves, designed to mirror the setup used in the Roman+Rubin synergy work. This repo is intended to be integrated into Karen’s pipeline (with image-based simulations from DP0/DP1) soon; meanwhile you can use it standalone.

> The code uses **rubin_sim** for cadence & photometry, **TRILEGAL** (via Astro Data Lab) for stellar populations, and **pyLIMA** for microlensing light-curve generation. The focus is on producing realistic light curves, not on population-level inference; e.g., we use uniform mass distributions and associate TRILEGAL source proper motions with foreground lenses along the same line of sight to obtain \( \mu_{\rm rel} \).

---

## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Run](#run)  
- [Models](#models)  
- [How It Works](#how-it-works)  
- [Parallel Execution](#parallel-execution)  
- [Troubleshooting](#troubleshooting)  
- [References](#references)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)

---

## Features

- 🔭 **Rubin cadence** from the latest baseline via `rubin_sim.maf`
- 📈 **Photometric errors** using the LSST model (Ivezić et al. 2019; DOI: [10.3847/1538-4357/ab042c](https://doi.org/10.3847/1538-4357/ab042c))
- 🌌 **TRILEGAL** star selection via **Astro Data Lab** (account required; P. Dal Tio et al. 2022; DOI: [10.3847/1538-4365/ac7be6](https://doi.org/10.3847/1538-4365/ac7be6))
- ✨ **Microlensing models** with **pyLIMA**:
  - **FSPL** (finite-source point lens) — free-floating planets (FFP)
  - **PSPL** (point-source point lens) — stellar lenses (e.g., BH)
  - **USBL** (uniform-source binary lens) — planetary systems (finite-source + parallax)
- 🧭 **Parallax** included by default
- ⚙️ **Parallel runner** (process pool) that scales: each child loads the TRILEGAL table once (Parquet), avoiding per-task pickling

---

## Installation

> Python 3.9+ recommended.

```bash
# Create & activate an environment (example with conda)
conda create -n rubin-ml python=3.10
conda activate rubin-ml

# Core deps
pip install rubin-sim pyLIMA astropy numpy pandas matplotlib

# Astro Data Lab client (provides the `dl` module)
pip install noirlab-auth-client noirlab-datalab

# For fast Parquet I/O used by the parallel runner
pip install pyarrow
```

**Rubin throughputs/baseline**  
The code expects Rubin throughputs under:
```
~/rubin_sim_data/throughputs/baseline/total_{u,g,r,i,z,y}.dat
```
and uses the latest Opsim baseline via `rubin_sim.data.get_baseline()`.

---

## Configuration

Edit `config_file.json`:

```json
{
  "model": "PSPL",
  "system_type": "BH",
  "path_save": "/home/USER/light_curve_rubin_test/",
  "ra": 266,
  "dec": -29.0,
  "radius": 0.1,
  "N": 50,
  "description": "test",
  "t0_range": [2460413.013828608, 2463335.01383],
  "Ds_max": 8000,
  "run_parallel": true,
  "N_tr": 8
}
```

- `model` / `system_type` are paired (see [Models](#models)).
- `t0_range` (JD) sets the window to draw event peaks \(t_0\).
- `radius` (deg) is the cone search around `(ra, dec)` for TRILEGAL sources.
- `Ds_max` (pc) filters by distance modulus.
- `run_parallel`: `true` to use the process pool runner.
- `N_tr`: number of worker processes (optional; can also be detected from Slurm).

> **Note:** The simulator expects band keys `u,g,r,i,z,Y` (uppercase `Y`). The loader maps TRILEGAL `umag→u`, …, `ymag→Y`.

---

## Run

```bash
python main.py
```

On first use you’ll be prompted for your **Astro Data Lab** credentials (the login runs only in the parent process; workers never prompt).

---

## Models

- **Free-floating planets (FFP)**  
  - Config: `"model": "FSPL", "system_type": "FFP"`  
  - Finite-source effects included (small Einstein radii)

- **Stellar lenses (e.g., BH)**  
  - Config: `"model": "PSPL", "system_type": "BH"`

- **Planetary systems (star+planet)**  
  - Config: `"model": "USBL", "system_type": "Planet_system"`  
  - Includes finite source and parallax

Parallax is included by default for these models.

---

## How It Works

1. **Query TRILEGAL** via Astro Data Lab within `(ra, dec, radius)`, limited by `Ds_max`. Rubin-band mags (u, g, r, i, z, y) are retrieved.
2. **Build source–lens pairs**: associate distant sources with nearer lenses along the line of sight and draw a relative angle to obtain \( \mu_{\rm rel} \).
3. **Rubin cadence**: `rubin_sim.maf` supplies visit times and 5σ depths using the latest baseline.
4. **Photometric errors**: computed with `calc_mag_error_m5` (Ivezić+ 2019 model).
5. **Simulate microlensing** using **pyLIMA** (FSPL/PSPL/USBL + parallax).
6. **Quality cuts**: ensure sufficient coverage and variability; events passing cuts are saved to `path_save`.

---

## Parallel Execution

The **Approach B** runner (`sim_parallel.py`) uses the `spawn` method with a **process pool**. To minimize IPC:

- The TRILEGAL DataFrame is written **once** to **Parquet** and loaded **once per worker**.
- Thread oversubscription is avoided by setting `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc., in each child.
- Control concurrency with `N_tr` (or leave it for Slurm to provide via `SLURM_CPUS_PER_TASK`).

---

## Troubleshooting

- **Repeated login prompts in parallel mode**  
  Ensure all interactive code (e.g., `ac.login(...)`) is guarded:
  ```python
  if __name__ == "__main__":
      cli()
  ```
  Workers must not perform logins.

- **Missing `pyarrow` / Parquet errors**  
  `pip install pyarrow`

- **KeyError for magnitudes**  
  Ensure mapping to `u,g,r,i,z,Y` is applied before simulation.

- **Throughput files not found**  
  Confirm `~/rubin_sim_data/throughputs/baseline/total_{band}.dat` exists.

- **Permissions**  
  Ensure `path_save` exists and is writable; the code attempts to create it.

---

## References

- **TRILEGAL via Astro Data Lab**: Piero Dal Tio et al., *ApJS*, DOI: [10.3847/1538-4365/ac7be6](https://doi.org/10.3847/1538-4365/ac7be6)  
- **Rubin photometric error model**: Ivezić et al. 2019, *ApJ*, DOI: [10.3847/1538-4357/ab042c](https://doi.org/10.3847/1538-4357/ab042c)  
- **pyLIMA**: microlensing modeling & simulation toolkit

---

## License

Specify a license in `LICENSE` (e.g., MIT). Example:  
```
MIT License — see LICENSE for details.
```

---

## Acknowledgements

Thanks to the Roman+Rubin synergy community and contributors to **rubin_sim**, **pyLIMA**, **Astro Data Lab**, and **TRILEGAL**.

---

<sub>Repository: https://github.com/anibal-art/rubin_lightcurves</sub>
