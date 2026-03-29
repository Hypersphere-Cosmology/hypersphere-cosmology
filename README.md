# Hypersphere Cosmology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19322656.svg)](https://doi.org/10.5281/zenodo.19322656)

**A geometric cosmological model derived from rₛ = 2R**

Sean P. Myers | Independent Researcher | ORCID: 0009-0000-3132-5383

## Overview

This repository contains the code for the paper:

> "Hypersphere Cosmology: The Universe as the Three-Dimensional Hyperspherical Surface of a Four-Dimensional Hyperball"
> Myers, S.P. (2026). Submitted to JCAP. **v14.5**

The model derives a comprehensive cosmology from a single postulate: the Schwarzschild radius equals twice the cosmic radius at all scales (rₛ = 2R).

### Core Physics

| Relation | Formula |
|---|---|
| Hubble parameter | H(z) = H₀(1+z) |
| Comoving distance | D_C(z) = (c/H₀) · ln(1+z) |
| Hubble distance | D_H(z) = c / H(z) = c / (H₀(1+z)) |
| Luminosity distance | D_L(z) = (1+z) · D_C(z) |
| Time dilation | γ(z) = √(2z+1) |
| H₀ (CC fit) | 62.3 ± 3.2 km/s/Mpc |

Note: `D_C(z) = (c/H₀)·ln(1+z)` is the **unique comoving distance consistent with H(z) = H₀(1+z)**, derived from ∫c/H(z')dz'. An earlier version of the code used a phenomenological `DM(z) = L·zc·log(1+z/zc)` form that was inconsistent with the stated Hubble relation; this has been corrected.

## Key Results

- **H₀ = 62.3 ± 3.2 km/s/Mpc** (cosmic chronometers)
- **Hubble tension resolved** geometrically via γ(z) = √(2z+1)
- **No dark matter or dark energy** — replaced by gravitational field energy
- **CMB acoustic scale** consistent with Planck 2018 θ_MC (compressed likelihood)
- **Low-ℓ CMB suppression** predicted from boundary conditions on S³

## Repository Structure

```
├── run_hypersphere_fit.py       # CC + SNe Ia + BAO fitter (MAP + MCMC)
├── boundary_cmb.py              # Boundary Boltzmann CMB likelihood
├── next_suite_cmb.py            # Extended multi-model comparison suite
├── data/
│   ├── firas_monopole_spec_v1.txt   # COBE/FIRAS CMB spectrum
│   └── bao_model_and_data.csv       # Compiled BAO measurements
└── requirements.txt
```

## Scripts

### `run_hypersphere_fit.py` — Main distance fitter

**What it does:**
- Fits H₀ (and BAO sound horizon r_d) to cosmic chronometers, Pantheon+ SNe Ia, and BAO
- Uses `D_C(z) = (c/H₀)·ln(1+z)` — consistent with H(z) = H₀(1+z)
- Optimizer: scipy `differential_evolution` (global MAP) + Metropolis-Hastings MCMC for posteriors
- Reports χ²/DOF for CC, SN, BAO separately and combined

**What it does NOT do:**
- Does not use random search (older versions did — now replaced with real MCMC)
- Does not fit CMB power spectra (use `boundary_cmb.py` for CMB)

```bash
python run_hypersphere_fit.py --data-root ./data [--mcmc-steps 3000]
```

Outputs: `hypersphere_fit_results.json`, `hypersphere_mcmc_chain.npy`, `hypersphere_mcmc_chain.csv`

### `boundary_cmb.py` — Boundary Boltzmann CMB

**What it does:**
- Implements the boundary radiation interpretation of the CMB
- Interprets the last-scattering surface as the S³/B⁴ boundary
- Computes the sound horizon r_d with γ(z) = √(2z+1) time dilation correction
- Fits to Planck 2018 compressed likelihood (θ_MC, ω_b, ω_cdm, A_s, n_s, τ)
- Computes TT power spectrum at low-ℓ (ℓ < 30) where boundary interpretation is physically motivated
- Provides an approximate template for ℓ = 30–200

**Honest limitations (calibrated against CAMB):**
- ✓ ℓ < 30 (Sachs-Wolfe plateau): boundary suppression model is physically motivated
- ✓ ℓ = 30–200 (first peak): acoustic scale θ_MC correctly computed
- ✗ ℓ > 200: power is off by 5–10× without full Boltzmann treatment
- ✗ Polarization at ℓ > 50 not reliably computed
- The χ²/DOF ≈ 1.6 reported in the paper applies to the compressed likelihood + low-ℓ evaluation only

```bash
python boundary_cmb.py --data-root ./data [--H0 62.3] [--plot]
```

Outputs: `boundary_cmb_results.json`, optionally `boundary_cmb_spectrum.png`

### `next_suite_cmb.py` — Extended multi-model comparison

**What it does:**
- Compares ΛCDM against several phenomenological HEA distance models
- Uses compressed Planck likelihood, Pantheon+, DESI BAO
- Computes BAO residuals, distance duality tables, AIC/BIC
- Uses random search + optional Nelder-Mead refinement (not MCMC)

```bash
python next_suite_cmb.py --data-root ./data [--refine]
```

## Data Requirements

The fitting scripts require these data files (not included due to size):

| Script | Required files |
|---|---|
| `run_hypersphere_fit.py` | `data/pantheon_plus/Pantheon+SH0ES.dat`, `data/pantheon_plus/Pantheon+SH0ES_STAT+SYS.cov`, `data/desi_bao/desi_bao_summary.csv` |
| `boundary_cmb.py` | `data/planck_acoustic/planck_compressed.csv` (optional; built-in Planck 2018 values used if absent) |

Cosmic chronometer data is built in (Moresco 2023 compilation, 33 measurements).

**Data sources:**
- Pantheon+: Brout et al. 2022, https://github.com/PantheonPlusSH0ES/DataRelease
- DESI BAO DR1: DESI Collaboration 2024, https://data.desi.lbl.gov
- Planck 2018: Planck Collaboration 2020, A&A 641 A6

## Quick Start (CC-only, no external data needed)

```bash
git clone https://github.com/Hypersphere-Cosmology/hypersphere-cosmology
cd hypersphere-cosmology
pip install -r requirements.txt
mkdir -p data

# Run CC-only fit (built-in data, no downloads needed)
python run_hypersphere_fit.py --data-root ./data --no-mcmc

# Run CMB compressed likelihood (built-in Planck 2018 values)
python boundary_cmb.py --data-root ./data
```

## Falsifiable Predictions

| Prediction | Test | Timeline |
|---|---|---|
| H(z) = H₀(1+z) | DESI DR3 Hubble diagram | ~2027 |
| r = 0 (no primordial B-modes) | CMB-S4 | ~2030 |
| Low-ℓ TT suppression (ℓ=2,3) | Future full-sky CMB | ~2028+ |
| 22% directional H₀ variation | DESI + Euclid | ~2028 |
| 3.3% smaller GW distances at z~1 | LIGO O5 / Einstein Telescope | ~2027–2035 |

## Known Issues and Reviewer Notes

Reviewer Canvas123 (2026) raised three valid concerns addressed in this version:

1. **DM(z)/H(z) inconsistency** [FIXED]: `run_hypersphere_fit.py` now uses
   `D_C(z) = (c/H₀)·ln(1+z)`, the analytic form consistent with H(z) = H₀(1+z).

2. **Fake MCMC** [FIXED]: `run_hypersphere_fit.py` now implements genuine
   Metropolis-Hastings MCMC with adaptive step sizes. MAP is found via
   `scipy.differential_evolution`.

3. **Missing CMB code** [ADDED]: `boundary_cmb.py` is the boundary Boltzmann
   implementation. It covers the compressed likelihood and low-ℓ TT spectrum.
   Full ℓ > 200 Boltzmann computation requires CAMB modifications (see limitations).

## Citation

```bibtex
@article{Myers2026Hypersphere,
  title={Hypersphere Cosmology: The Universe as the Three-Dimensional 
         Hyperspherical Surface of a Four-Dimensional Hyperball},
  author={Myers, Sean P.},
  year={2026},
  doi={10.5281/zenodo.19322656},
  note={Submitted to JCAP},
  url={https://github.com/Hypersphere-Cosmology/hypersphere-cosmology}
}
```

## License

MIT License.

## Contact

Sean P. Myers | seanpmyers1975@gmail.com | ORCID: 0009-0000-3132-5383
