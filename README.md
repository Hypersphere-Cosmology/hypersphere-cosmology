# Hypersphere Cosmology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19296662.svg)](https://doi.org/10.5281/zenodo.19296662)

**A geometric cosmological model derived from rₛ = 2R**

Sean P. Myers | Independent Researcher | ORCID: 0009-0000-3132-5383

## Overview

This repository contains all code and data for the paper:

> "Hypersphere Cosmology: The Universe as the Three-Dimensional Hyperspherical Surface of a Four-Dimensional Hyperball"
> Myers, S.P. (2026). Submitted to JCAP.

The model derives a comprehensive cosmology from a single postulate: the Schwarzschild radius equals twice the cosmic radius at all scales (rₛ = 2R).

## Key Results

- **H₀ = 62.3 ± 3.2 km/s/Mpc** (cosmic chronometers, χ²/DOF = 0.54)
- **Hubble tension resolved** geometrically via time dilation γ(z) = √(2z+1)
- **No dark matter or dark energy** — replaced by gravitational field energy (86.4%)
- **CMB as boundary radiation** from S³/B⁴ interface (TT, TE, EE within 0.5% of Planck)
- **SPARC rotation curves** fit without dark matter (ΔAIC = -5.1 vs ΛCDM+NFW)

## Repository Structure

```
├── run_hypersphere_fit.py       # Main fitting routine (CC, SNe Ia, BAO)
├── next_suite_cmb.py            # CMB power spectrum calculations
├── figure_25_bb_spectrum.png    # Figure 25: FIRAS CMB spectrum
├── figure_36_lss.png            # Figure 36: Large-scale structure
├── notebooks/
│   └── hypersphere_quick_start.ipynb  # Reproduces main results in <10 min
├── data/
│   ├── firas_spectrum/          # COBE/FIRAS monopole spectrum
│   ├── desi_bao/                # DESI DR1 + DR2 BAO data
│   └── bao_model_and_data.csv  # Compiled BAO measurements
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/Hypersphere-Cosmology/hypersphere-cosmology
cd hypersphere-cosmology
pip install -r requirements.txt
python run_hypersphere_fit.py
```

## Falsifiable Predictions

| Prediction | Test | Timeline |
|---|---|---|
| H(z) = H₀(1+z) | DESI DR3 | ~2027 |
| r = 0 (no primordial B-modes) | CMB-S4 | ~2030 |
| 22% directional H₀ variation | DESI + Euclid | ~2028 |
| 3.3% smaller GW distances at z~1 | LIGO O5 / Einstein Telescope | ~2027-2035 |

## Citation

```bibtex
@article{Myers2026Hypersphere,
  title={Hypersphere Cosmology: The Universe as the Three-Dimensional 
         Hyperspherical Surface of a Four-Dimensional Hyperball},
  author={Myers, Sean P.},
  journal={JCAP},
  year={2026},
  note={Submitted},
  url={https://github.com/Hypersphere-Cosmology/hypersphere-cosmology}
}
```

## License

MIT License. See LICENSE file.

## Contact

Sean P. Myers | seanpmyers1975@gmail.com | ORCID: 0009-0000-3132-5383
