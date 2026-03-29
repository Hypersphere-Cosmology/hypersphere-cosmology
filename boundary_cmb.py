# -*- coding: utf-8 -*-
"""
boundary_cmb.py — Boundary Boltzmann CMB code for Hypersphere Cosmology
=======================================================================
Implements a semi-analytic CMB likelihood that interprets the last-scattering
surface as the S³/B⁴ boundary of the Hypersphere model.

PHYSICAL INTERPRETATION:
  Standard cosmology: CMB photons last scatter at z* ≈ 1090 on a thin shell
  Hypersphere model: That shell IS the S³/B⁴ boundary — the 3-sphere surface
    of the 4-ball. The boundary modifies large-scale modes (ℓ=2,3 suppression)
    and affects the acoustic scale through D_A.

  Key insight: H(z) = H₀(1+z) is the LATE-UNIVERSE Hubble relation in this model.
  The early universe (z >> 100) still operates with radiation+baryon physics
  that determines the sound horizon r_d. The Hypersphere modification enters
  observationally through the ANGULAR DIAMETER DISTANCE:
    θ_MC = r_d / D_A(z*)
  where D_A uses the Hypersphere formula D_A(z) = D_C(z)/(1+z)
  and D_C(z) = (c/H₀)·ln(1+z).

  The sound horizon r_d is treated as a free parameter. For comparison, a
  standard approximation formula (Eisenstein & Hu 1998) is also provided.
  The γ(z) = √(2z+1) correction modifies the effective drag epoch timing.

HONEST LIMITATIONS (calibrated against CAMB):
  ✓ Compressed Planck likelihood (θ_MC, ω_b, ω_cdm, A_s, n_s, τ)
  ✓ ℓ < 30 (Sachs-Wolfe plateau): boundary suppression model is motivated
  ✗ ℓ > 200: 5–10× off vs data without full Boltzmann treatment
  ✗ The χ²/DOF ≈ 1.6 in the paper covers compressed + low-ℓ ONLY
  ✗ Full power spectrum requires modified CAMB/CLASS

USAGE:
  python boundary_cmb.py --data-root ./data [--H0 62.3] [--plot]

REQUIREMENTS: numpy, scipy
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd

# ─── Physical constants ──────────────────────────────────────────────────────
c_kms   = 299792.458  # km/s
T_cmb   = 2.7255      # K, Planck 2018

# ─── Hypersphere distances ────────────────────────────────────────────────────

def DC_hypersphere(z, H0):
    """
    Comoving distance: D_C(z) = (c/H₀)·ln(1+z) [Mpc].
    Derived from H(z) = H₀(1+z) via D_C = ∫c/H(z')dz'.
    """
    return (c_kms / H0) * np.log1p(np.asarray(z, float))

def DA_hypersphere(z, H0):
    """
    Angular diameter distance: D_A = D_C / (1+z) [Mpc].
    """
    z = np.asarray(z, float)
    return DC_hypersphere(z, H0) / (1.0 + z)

def DL_hypersphere(z, H0):
    """Luminosity distance D_L = (1+z)·D_C [Mpc]."""
    z = np.asarray(z, float)
    return (1.0 + z) * DC_hypersphere(z, H0)

def gamma_factor(z):
    """Time dilation factor γ(z) = √(2z+1)."""
    return np.sqrt(2.0 * np.asarray(z, float) + 1.0)

# ─── Sound horizon ────────────────────────────────────────────────────────────

def sound_horizon_EH98(omega_b, omega_m):
    """
    Eisenstein & Hu (1998) fitting formula for the sound horizon at drag epoch.
    This uses standard ΛCDM early-universe physics and is independent of H(z)
    at late times. H₀ enters via omega_m = Ω_m h².

    Returns r_d in Mpc (physical, not h⁻¹ Mpc).

    NOTE: H(z) = H₀(1+z) is a late-universe effective relation in the
    Hypersphere model. The early-universe sound horizon is still set by
    radiation-baryon fluid physics. r_d is therefore treated as a free
    parameter in the fit, with this formula as a prior reference.
    """
    h_eff = np.sqrt(omega_m / 0.3)  # rough scaling
    # EH98 Eq 6: z_d (drag epoch)
    b1 = 0.313 * omega_m**(-0.419) * (1.0 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_d = 1291.0 * omega_m**0.251 / (1.0 + 0.659 * omega_m**0.828) * \
          (1.0 + b1 * omega_b**b2)

    # EH98 Eq 4: sound horizon
    # r_s = 2/(3 k_eq) * sqrt(6/R_eq) * ln[(sqrt(1+Rd) + sqrt(Rd+Req))/(1+sqrt(Req))]
    # k_eq, R_eq, R_d defined below
    omega_r = 4.31e-5  # photon density (T=2.7255 K)
    R_eq = 31.5e3 * omega_b / omega_r  # at matter-radiation equality
    k_eq = 7.46e-2 * omega_m           # h/Mpc, but we need Mpc⁻¹
    # Convert to Mpc⁻¹: k_eq_Mpc = k_eq * h where h = sqrt(omega_m/0.3) ≈ rough
    # Use the direct EH formula that gives Mpc (not h⁻¹ Mpc)
    # r_s [Mpc/h] = 44.5 ln(9.83/omega_m) / sqrt(1 + 10 omega_b^0.75)
    r_s_over_h = 44.5 * np.log(9.83 / omega_m) / np.sqrt(1.0 + 10.0 * omega_b**0.75)
    h = np.sqrt(omega_m / 0.3) * np.sqrt(0.3 / omega_m) * 1.0  # need H₀ to convert
    # Actually EH gives r_s in Mpc/h — to get Mpc we need h = H₀/100
    # Without knowing H₀ separately, return in Mpc/h; caller multiplies by h
    return r_s_over_h  # in Mpc/h (× (H₀/100) to get Mpc... see wrapper below)

def sound_horizon_approx(H0, omega_b, omega_cdm):
    """
    Approximate sound horizon r_d in Mpc using E&H fitting formula.
    H₀ is needed to convert from Mpc/h to Mpc.
    omega_m = omega_b + omega_cdm (physical matter density Ω_m h²).

    Returns (r_d_Mpc, z_drag_approx)
    """
    h = H0 / 100.0
    omega_m = omega_b + omega_cdm
    r_s_h = sound_horizon_EH98(omega_b, omega_m)  # Mpc/h
    r_d = r_s_h / h  # Mpc
    # Approximate drag redshift (Hu & Sugiyama 1996)
    b1 = 0.313 * omega_m**(-0.419) * (1.0 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_d = 1291.0 * omega_m**0.251 / (1.0 + 0.659 * omega_m**0.828) * \
          (1.0 + b1 * omega_b**b2)
    return float(r_d), float(z_d)

def sound_horizon_with_gamma(H0, omega_b, omega_cdm):
    """
    Sound horizon with Hypersphere γ(z) = √(2z+1) correction.

    Interpretation: At the drag epoch z_d, the time dilation factor γ(z_d)
    effectively compresses the proper time available for acoustic oscillations.
    The corrected sound horizon is approximately:
        r_d_hyp ≈ r_d_standard / γ(z_d)

    This is a phenomenological correction. A rigorous calculation would require
    solving the Boltzmann equations on S³ geometry.

    Returns (r_d_hypersphere, r_d_standard, z_drag)
    """
    r_d_std, z_d = sound_horizon_approx(H0, omega_b, omega_cdm)
    g = float(gamma_factor(z_d))
    r_d_hyp = r_d_std / g
    return r_d_hyp, r_d_std, z_d

# ─── Angular acoustic scale θ_MC ─────────────────────────────────────────────

def theta_MC_hypersphere(H0, r_d, z_star=1089.8):
    """
    Angular acoustic scale: θ_MC = r_d / D_A(z*)
    D_A is the Hypersphere angular diameter distance.

    This is the key quantity: Planck measures θ_MC = 0.010409 ± 0.000031.
    In the Hypersphere model, D_A(z*) is SMALLER than ΛCDM D_A(z*) for
    H₀ ≈ 62 km/s/Mpc (because the ln(1+z) growth is slower than ΛCDM at z*),
    so a SMALLER r_d is needed to match θ_MC.

    Parameters:
      H0    : Hubble constant [km/s/Mpc]
      r_d   : sound horizon [Mpc] — FREE PARAMETER in the fit
      z_star: last-scattering redshift (default: 1089.8)

    Returns:
      theta_MC (dimensionless, ~0.0104 for Planck)
    """
    DA = float(DA_hypersphere(z_star, H0))
    if DA <= 0:
        return np.nan
    return r_d / DA

# ─── Planck 2018 compressed likelihood ───────────────────────────────────────

# Planck 2018 TT+TE+EE+lowE compressed parameters
# Source: Planck Collaboration 2020, A&A 641, A6, Table 1
PLANCK_2018_COMPRESSED = {
    "theta_MC": 1.04092e-2,   # best fit; σ = 0.00031×10⁻² → σ(θ_MC) = 3.1e-5
    "omega_b":  0.02237,      # σ = 0.00015
    "omega_cdm":0.1200,       # σ = 0.0012
    "A_s":      2.100e-9,     # σ = 0.030e-9
    "n_s":      0.9649,       # σ = 0.0042
    "tau":      0.0544,       # σ = 0.0073
}

PLANCK_2018_ERRORS = {
    "theta_MC":  3.1e-5,
    "omega_b":   0.00015,
    "omega_cdm": 0.0012,
    "A_s":       0.030e-9,
    "n_s":       0.0042,
    "tau":       0.0073,
}

def load_cmb_compressed(data_root):
    """
    Load Planck compressed likelihood from file, or use built-in values.
    File format: CSV with columns 'param', 'value', 'sigma'
    """
    p = os.path.join(data_root, "planck_acoustic", "planck_compressed.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        means = {}; errors = {}
        for _, row in df.iterrows():
            key = str(row['param']).lower()
            means[key]  = float(row['value'])
            errors[key] = float(row['sigma']) if 'sigma' in row.index else \
                          PLANCK_2018_ERRORS.get(key, 1.0)
        # Fill in any missing keys from built-ins
        for k in PLANCK_2018_COMPRESSED:
            if k not in means:
                means[k]  = PLANCK_2018_COMPRESSED[k]
                errors[k] = PLANCK_2018_ERRORS[k]
        return means, errors
    else:
        print("Note: Using built-in Planck 2018 compressed values (no file found)")
        return dict(PLANCK_2018_COMPRESSED), dict(PLANCK_2018_ERRORS)

# ─── Compressed CMB chi-squared ──────────────────────────────────────────────

def chi2_compressed_cmb(H0, r_d, omega_b, omega_cdm, A_s, n_s, tau,
                         planck_means, planck_errors, z_star=1089.8):
    """
    Chi-squared vs Planck 2018 compressed likelihood.

    Free parameters: θ_MC (from H₀, r_d), ω_b, ω_cdm, A_s, n_s, τ

    The Hypersphere modification enters through θ_MC = r_d / D_A(z*)
    where D_A uses D_C(z) = (c/H₀)ln(1+z) instead of ΛCDM D_C.

    r_d is a free parameter. It can be predicted from H₀ and ω_b via
    sound_horizon_with_gamma() but is fit freely here.

    Returns:
      chi2       : total chi-squared (6 terms)
      theta_mc   : computed θ_MC
      breakdown  : per-parameter chi-squared contributions
    """
    theta_mc = theta_MC_hypersphere(H0, r_d, z_star)
    if not np.isfinite(theta_mc):
        return np.inf, np.nan, {}

    params = {
        "theta_MC":  theta_mc,
        "omega_b":   omega_b,
        "omega_cdm": omega_cdm,
        "A_s":       A_s,
        "n_s":       n_s,
        "tau":       tau,
    }

    breakdown = {}
    chi2 = 0.0
    for key, val in params.items():
        if key in planck_means and key in planck_errors:
            pull = (val - planck_means[key]) / planck_errors[key]
            breakdown[key] = float(pull**2)
            chi2 += pull**2

    return float(chi2), theta_mc, breakdown

# ─── Low-ℓ power spectrum ─────────────────────────────────────────────────────

def sachs_wolfe_spectrum(ell_arr, A_s, n_s):
    """
    Approximate TT power spectrum D_ℓ in the Sachs-Wolfe plateau (ℓ < 30).

    Standard formula:
      C_ℓ^SW ∝ A_s · Γ(4 - n_s/2) · Γ(ℓ + (n_s-1)/2) / [Γ((5-n_s)/2) · Γ(ℓ + (5-n_s)/2)]
    Approximated as:
      D_ℓ ∝ A_s · (ℓ/ℓ_piv)^(n_s-1) × amplitude

    Boundary condition modification (S³ geometry):
      At the S³/B⁴ boundary, certain modes cannot propagate freely.
      The ℓ=2 (quadrupole) and ℓ=3 (octopole) modes are suppressed by
      the matching condition at the boundary. This is modeled as:
        S(ℓ) = 1 - α · exp(-ℓ/ℓ_boundary)
      where α~0.5, ℓ_boundary~4 (phenomenological).

    Calibration: D_ℓ ≈ 1000-2000 μK² in the plateau (ℓ~5-20), matching
    Planck approximate values.

    Returns: D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in μK² [APPROXIMATE]
    """
    ell = np.asarray(ell_arr, float)
    T_cmb_muK = T_cmb * 1e6

    # Sachs-Wolfe plateau amplitude
    # Factor 1/9 from adiabatic SW; T_prim / T0 = -Φ/3
    A_sw = (A_s / 2.2e-9) * 1000.0  # μK², calibrated to Planck plateau ~1000-2000 μK²

    # Tilt
    ell_pivot = 10.0
    tilt = (ell / ell_pivot)**(n_s - 1.0)

    # Boundary suppression of lowest modes (S³ geometry)
    alpha = 0.5          # suppression amplitude
    ell_boundary = 3.5   # scale below which suppression is significant
    suppression = 1.0 - alpha * np.exp(-ell / ell_boundary)

    # D_ℓ (the SW plateau is approximately flat in D_ℓ space)
    D_ell = A_sw * tilt * suppression

    return D_ell

def get_lowl_planck_data(data_root):
    """
    Return Planck 2018 approximate low-ℓ TT D_ℓ values.
    Loads from file if available, else uses hard-coded approximate values.

    These are approximate: exact values should be read from the Planck Legacy
    Archive (PLA) low-ℓ TT likelihood.
    """
    p = os.path.join(data_root, "planck_acoustic", "planck_lowl_TT.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return dict(zip(df['ell'].astype(int),
                        zip(df['Dl'].astype(float), df['sigma'].astype(float))))

    # Approximate Planck 2018 low-ℓ values from published figures
    # (D_ℓ in μK², σ dominated by cosmic variance σ_CV ≈ √(2/(2ℓ+1)) × D_ℓ)
    planck_approx = {
        2:  (252,  490),   # Large cosmic variance: σ_CV ≈ D_ℓ × √(2/5) ≈ 160
        3:  (750,  340),
        4:  (590,  230),
        5:  (430,  170),
        6:  (340,  130),
        7:  (185,  100),
        8:  (300,   95),
        9:  (230,   82),
        10: (270,   76),
        12: (250,   65),
        15: (290,   58),
        20: (315,   52),
        25: (345,   50),
        29: (370,   50),
    }
    return planck_approx

def chi2_lowl_TT(A_s, n_s, data_root, ell_max=29):
    """
    Chi-squared vs Planck 2018 low-ℓ TT power spectrum.
    Tests the boundary-induced quadrupole/octopole suppression prediction.

    Returns (chi2, n_ell)
    """
    planck_lowl = get_lowl_planck_data(data_root)
    ells = sorted([ell for ell in planck_lowl if 2 <= ell <= ell_max])

    model_D = sachs_wolfe_spectrum(np.array(ells), A_s, n_s)

    chi2 = 0.0
    n_ell = 0
    for i, ell in enumerate(ells):
        D_planck, sigma = planck_lowl[ell]
        chi2 += ((model_D[i] - D_planck) / sigma)**2
        n_ell += 1

    return float(chi2), n_ell

# ─── Full boundary CMB likelihood ────────────────────────────────────────────

def boundary_cmb_likelihood(H0, r_d, omega_b, omega_cdm, A_s, n_s, tau,
                              planck_means, planck_errors, data_root,
                              include_lowl=True, z_star=1089.8):
    """
    Combined CMB likelihood for Hypersphere boundary interpretation.

    Components:
      1. Compressed Planck likelihood: χ²(θ_MC, ω_b, ω_cdm, A_s, n_s, τ)
         θ_MC computed via Hypersphere D_A
      2. Low-ℓ TT: χ²(ℓ=2..29) testing boundary suppression

    Parameters:
      H0, r_d    : Hypersphere parameters (H₀ km/s/Mpc, r_d Mpc)
      omega_b    : Ω_b h²
      omega_cdm  : Ω_cdm h²
      A_s        : primordial amplitude
      n_s        : spectral index
      tau        : reionization optical depth
    """
    # Component 1: Compressed likelihood (6 Planck params)
    chi2_comp, theta_mc, breakdown = chi2_compressed_cmb(
        H0, r_d, omega_b, omega_cdm, A_s, n_s, tau,
        planck_means, planck_errors, z_star)

    chi2_total = chi2_comp
    n_data = 6  # number of Planck compressed params

    # Component 2: Low-ℓ TT spectrum
    chi2_ll = 0.0
    n_ll = 0
    if include_lowl:
        chi2_ll, n_ll = chi2_lowl_TT(A_s, n_s, data_root)
        chi2_total += chi2_ll
        n_data += n_ll

    n_free = 7   # H0, r_d, omega_b, omega_cdm, A_s, n_s, tau
    dof = max(1, n_data - n_free)

    # Sound horizon comparison (informational)
    r_d_hyp, r_d_std, z_d = sound_horizon_with_gamma(H0, omega_b, omega_cdm)

    return {
        "H0":               float(H0),
        "r_d":              float(r_d),
        "r_d_approx_hyp":   float(r_d_hyp),
        "r_d_approx_std":   float(r_d_std),
        "z_drag":           float(z_d),
        "omega_b":          float(omega_b),
        "omega_cdm":        float(omega_cdm),
        "A_s":              float(A_s),
        "n_s":              float(n_s),
        "tau":              float(tau),
        "theta_MC":         float(theta_mc),
        "chi2_compressed":  float(chi2_comp),
        "chi2_lowl_TT":     float(chi2_ll),
        "chi2_total":       float(chi2_total),
        "n_data":           int(n_data),
        "n_free_params":    int(n_free),
        "dof":              int(dof),
        "chi2_per_dof":     float(chi2_total / dof),
        "breakdown":        breakdown,
    }

# ─── Optimizer ───────────────────────────────────────────────────────────────

def fit_cmb(planck_means, planck_errors, data_root,
            H0_range=(50.0, 80.0), n_starts=300, seed=42):
    """
    Minimize χ² over 7 CMB parameters:
      H0, r_d, omega_b, omega_cdm, A_s, n_s, tau

    Uses scipy differential_evolution (global) if available.
    Falls back to random restart search.
    """
    # r_d bounds: Hypersphere D_A(z*=1090) is smaller than ΛCDM for H₀~62,
    # so r_d needed to match θ_MC≈0.0104 is also smaller.
    # D_A_hyp = (c/H₀)·ln(1091)/1090 ≈ (299792/62.3)·0.00642 ≈ 30.9 Mpc
    # r_d_needed = θ_MC × D_A ≈ 0.0104 × 30.9 ≈ 0.32 Mpc — far too small!
    #
    # WAIT: this reveals a tension. For H₀=62.3:
    #   D_A(z*=1090) ≈ 30.9 Mpc  → r_d = θ_MC × D_A ≈ 0.322 Mpc
    # Standard ΛCDM: D_A ≈ 13900 Mpc → r_d ≈ 145 Mpc
    # The difference is because D_C = (c/H₀)ln(1+z) grows much more slowly
    # than ΛCDM D_C at z=1090. This is a fundamental tension in the model
    # that cannot be resolved without modifying early-universe physics.
    #
    # We fit r_d freely and report θ_MC tension honestly.

    h0_lo, h0_hi = H0_range
    # r_d: let it range from ~0.1 to 200 Mpc to capture the model's prediction
    bounds = [
        (h0_lo, h0_hi),        # H0
        (0.1, 200.0),          # r_d [Mpc]
        (0.018, 0.026),        # omega_b
        (0.10,  0.14),         # omega_cdm
        (1.7e-9, 2.5e-9),      # A_s
        (0.92,  1.01),         # n_s
        (0.02,  0.12),         # tau
    ]

    def objective(p):
        H0, rd, wb, wc, As, ns, tau = p
        r = boundary_cmb_likelihood(
            H0, rd, wb, wc, As, ns, tau,
            planck_means, planck_errors, data_root)
        return r["chi2_total"]

    best_chi2 = np.inf
    best_p = None

    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(objective, bounds, seed=seed,
                                     maxiter=500, popsize=15, tol=1e-6)
        if res.fun < best_chi2:
            best_chi2 = res.fun
            best_p = list(res.x)
    except ImportError:
        rng = np.random.default_rng(seed)
        lows  = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        for _ in range(n_starts):
            p = list(rng.uniform(lows, highs))
            chi2 = objective(p)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_p = p[:]

    H0, rd, wb, wc, As, ns, tau = best_p
    return boundary_cmb_likelihood(
        H0, rd, wb, wc, As, ns, tau,
        planck_means, planck_errors, data_root)

# ─── Approximate TT template for plotting ─────────────────────────────────────

def tt_power_spectrum_approx(ell_arr, A_s, n_s, H0, r_d, omega_b, omega_cdm, tau):
    """
    Approximate TT power spectrum D_ℓ for ℓ=2..200.

    Combines:
      - SW plateau (ℓ < 30)
      - Gaussian acoustic peak template (30 ≤ ℓ ≤ 200)

    CAUTION: This is a rough template, NOT a Boltzmann solver.
    Uncertainty is ~10-30% at ℓ~100-200 and much worse at higher ℓ.
    Use CAMB/CLASS for any quantitative claims.
    """
    ell = np.asarray(ell_arr, float)

    # Acoustic scale
    theta_mc = theta_MC_hypersphere(H0, r_d)
    if not np.isfinite(theta_mc) or theta_mc <= 0:
        return np.full_like(ell, np.nan)
    ell_A = np.pi / theta_mc  # first acoustic peak location

    # SW plateau (ℓ < 30)
    D_sw = sachs_wolfe_spectrum(ell, A_s, n_s)

    # Acoustic peak template (all ℓ, with peak structure)
    tau_factor = np.exp(-2.0 * tau)
    A_peak = 6000.0 * (A_s / 2.2e-9) * tau_factor
    pw = 0.38 * ell_A  # peak width

    peak1 = np.exp(-0.5 * ((ell - ell_A)       / pw)**2)
    peak2 = 0.33 * np.exp(-0.5 * ((ell - 2*ell_A) / (0.85*pw))**2)

    D_acoustic = A_peak * (peak1 + peak2 + 0.03)

    # Blend: at ℓ < 30 use SW, at ℓ > 60 use acoustic, interpolate between
    w = np.clip((ell - 30.0) / 30.0, 0.0, 1.0)
    D_total = (1.0 - w) * D_sw + w * D_acoustic

    return D_total

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Boundary Boltzmann CMB — Hypersphere Cosmology")
    ap.add_argument("--data-root", default="./data",
                    help="Data directory (default: ./data)")
    ap.add_argument("--H0", type=float, default=None,
                    help="Fix H₀ and use Planck central values for other params")
    ap.add_argument("--rd", type=float, default=None,
                    help="Fix r_d [Mpc] (default: optimize)")
    ap.add_argument("--plot", action="store_true",
                    help="Generate CMB power spectrum comparison plot")
    ap.add_argument("--no-lowl", action="store_true",
                    help="Skip low-ℓ TT chi-squared")
    args = ap.parse_args()

    print("=" * 65)
    print("Boundary Boltzmann CMB — Hypersphere Cosmology")
    print("Last-scattering surface = S³/B⁴ boundary")
    print("=" * 65)
    print()
    print("IMPORTANT CAVEATS:")
    print("  ✓ Compressed Planck likelihood (θ_MC, ω_b, ω_cdm, A_s, n_s, τ)")
    print("  ✓ Low-ℓ TT (ℓ<30): boundary suppression at quadrupole/octopole")
    print("  ✗ ℓ > 200: NOT computed — off by 5-10× per CAMB cross-check")
    print("  ✗ χ²/DOF~1.6 in paper = compressed+low-ℓ ONLY, not full spectrum")
    print()

    # Load data
    planck_means, planck_errors = load_cmb_compressed(args.data_root)

    print("Planck 2018 compressed reference values:")
    for k in ["theta_MC", "omega_b", "omega_cdm", "A_s", "n_s", "tau"]:
        if k in planck_means:
            print(f"  {k:12s}: {planck_means[k]:.6g} ± {planck_errors[k]:.4g}")
    print()

    H0_eval = args.H0 if args.H0 else 62.3

    # Compute D_A at last scattering for Hypersphere model
    z_star = 1089.8
    DA_hyp = float(DA_hypersphere(z_star, H0_eval))
    DA_lcdm_approx = 13500.0  # Mpc, rough ΛCDM value

    print(f"Angular diameter distance to z*={z_star} for H₀={H0_eval} km/s/Mpc:")
    print(f"  D_A (Hypersphere, D_C=c/H₀·ln(1+z)): {DA_hyp:.2f} Mpc")
    print(f"  D_A (ΛCDM reference, approx):          {DA_lcdm_approx:.0f} Mpc")
    print(f"  Ratio: {DA_hyp/DA_lcdm_approx:.4f}  (Hypersphere D_A is MUCH smaller)")
    print()

    # Implied r_d to match Planck θ_MC
    theta_planck = planck_means.get("theta_MC", 1.04092e-2)
    r_d_implied = theta_planck * DA_hyp
    print(f"To match Planck θ_MC={theta_planck:.6f} with Hypersphere D_A:")
    print(f"  r_d_needed = θ_MC × D_A = {theta_planck:.6f} × {DA_hyp:.2f} = {r_d_implied:.3f} Mpc")
    print(f"  (Standard ΛCDM has r_d ≈ 147 Mpc)")
    print()

    r_d_hyp_approx, r_d_std, z_d = sound_horizon_with_gamma(
        H0_eval, planck_means.get("omega_b", 0.02237),
        planck_means.get("omega_cdm", 0.1200))
    print(f"Sound horizon from E&H98 approximation (H₀={H0_eval}):")
    print(f"  r_d (standard E&H98):              {r_d_std:.2f} Mpc/h")
    print(f"  r_d (÷ γ(z_d)={gamma_factor(z_d):.2f}, γ-correction): {r_d_hyp_approx:.2f} Mpc/h")
    print(f"  z_drag ≈ {z_d:.0f}")
    print()
    print("  NOTE: The fundamental tension here is that D_A(z*) in the")
    print("  Hypersphere model is ~450× smaller than ΛCDM D_A(z*),")
    print("  requiring r_d ~0.32 Mpc to match θ_MC. This is much smaller")
    print("  than the standard sound horizon (~147 Mpc). This tension")
    print("  represents a challenge for the model at the CMB scale.")
    print()

    # Fit or evaluate
    if args.H0 is not None and args.rd is not None:
        # Single evaluation at fixed H0 and r_d
        print(f"Evaluating at fixed H₀={args.H0}, r_d={args.rd} Mpc")
        wb   = planck_means.get("omega_b",   0.02237)
        wc   = planck_means.get("omega_cdm", 0.1200)
        As   = planck_means.get("A_s",       2.100e-9)
        ns   = planck_means.get("n_s",       0.9649)
        tau  = planck_means.get("tau",       0.0544)
        result = boundary_cmb_likelihood(
            args.H0, args.rd, wb, wc, As, ns, tau,
            planck_means, planck_errors, args.data_root,
            include_lowl=not args.no_lowl)
    elif args.H0 is not None:
        # Fix H0, optimize r_d and nuisance params
        print(f"Optimizing with fixed H₀={args.H0}...")
        result = fit_cmb(planck_means, planck_errors, args.data_root,
                         H0_range=(args.H0 * 0.9999, args.H0 * 1.0001))
    else:
        # Full optimization
        print("Optimizing all 7 CMB parameters...")
        result = fit_cmb(planck_means, planck_errors, args.data_root)

    # Print results
    print("── CMB Fit Results ─────────────────────────────────────────")
    print(f"  H₀          = {result['H0']:.3f} km/s/Mpc")
    print(f"  r_d (fit)   = {result['r_d']:.4f} Mpc")
    print(f"  θ_MC (model)= {result['theta_MC']:.6f}  "
          f"(Planck: {planck_means.get('theta_MC', 1.04092e-2):.6f})")
    pull_theta = (result['theta_MC'] - planck_means.get('theta_MC', 1.04092e-2)) / \
                  planck_errors.get('theta_MC', 3.1e-5)
    print(f"  θ_MC pull   = {pull_theta:.1f}σ")
    print(f"  ω_b         = {result['omega_b']:.5f}")
    print(f"  ω_cdm       = {result['omega_cdm']:.4f}")
    print(f"  A_s         = {result['A_s']:.4e}")
    print(f"  n_s         = {result['n_s']:.4f}")
    print(f"  τ           = {result['tau']:.4f}")
    print()
    print(f"  χ²(compressed) = {result['chi2_compressed']:.2f}  [6 Planck params]")
    if result['chi2_lowl_TT'] > 0:
        print(f"  χ²(low-ℓ TT)   = {result['chi2_lowl_TT']:.2f}  "
              f"[{result['n_data'] - 6} multipoles, ℓ<30]")
    print(f"  χ²_total       = {result['chi2_total']:.2f}")
    print(f"  N_data         = {result['n_data']}  (compressed + low-ℓ)")
    print(f"  N_params       = {result['n_free_params']}")
    print(f"  DOF            = {result['dof']}")
    print(f"  χ²/DOF         = {result['chi2_per_dof']:.3f}")
    print()
    print("  Per-parameter pulls (from Planck 2018 compressed values):")
    for k, v in result.get('breakdown', {}).items():
        sigma = v**0.5
        print(f"    {k:12s}: {sigma:+.2f}σ  (χ² = {v:.3f})")

    print()
    print("── Physical Interpretation ─────────────────────────────────")
    print("  The θ_MC constraint is the tightest CMB test for this model.")
    print("  With D_A(z*) ≈ 31 Mpc (vs ΛCDM ~13500 Mpc), the Hypersphere")
    print("  model requires r_d ≈ 0.32 Mpc. Whether this is physically")
    print("  achievable requires a full boundary Boltzmann treatment on S³.")
    print()
    print("  The low-ℓ TT spectrum (ℓ<30) where the boundary suppression")
    print("  of quadrupole/octopole is predicted to be observable is the")
    print("  most distinctive testable prediction of this code.")

    # Save
    with open("boundary_cmb_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print()
    print("Saved: boundary_cmb_results.json")

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ell_arr = np.arange(2, 201)
            D_model = tt_power_spectrum_approx(
                ell_arr,
                result["A_s"], result["n_s"], result["H0"], result["r_d"],
                result["omega_b"], result["omega_cdm"], result["tau"])

            # Low-ell Planck approximate values
            planck_ll = get_lowl_planck_data(args.data_root)
            ell_data = np.array(sorted(planck_ll.keys()))
            D_data   = np.array([planck_ll[e][0] for e in ell_data])
            D_err    = np.array([planck_ll[e][1] for e in ell_data])

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            ax = axes[0]
            ax.plot(ell_arr, D_model, 'b-', lw=2,
                    label=f'Hypersphere model (H₀={result["H0"]:.1f})')
            ax.errorbar(ell_data, D_data, yerr=D_err, fmt='k.', capsize=3,
                        label='Planck 2018 (approx, low-ℓ)')
            ax.axvspan(2, 5, alpha=0.1, color='green', label='Boundary suppression zone')
            ax.set_xlabel('Multipole ℓ')
            ax.set_ylabel(r'$D_\ell$ [μK²]')
            ax.set_title(f'Hypersphere CMB: full ℓ=2-200\n'
                         f'WARNING: ℓ>30 is a rough template — not a Boltzmann computation')
            ax.legend(fontsize=8)
            ax.set_xlim(2, 200)

            ax2 = axes[1]
            ax2.plot(ell_arr[:28], D_model[:28], 'b-', lw=2, label='Model (SW plateau)')
            ax2.errorbar(ell_data[ell_data <= 29], D_data[ell_data <= 29],
                         yerr=D_err[ell_data <= 29], fmt='k.', capsize=3,
                         label='Planck 2018 (approx)')
            ax2.axvspan(2, 5, alpha=0.15, color='green')
            ax2.set_xlabel('Multipole ℓ')
            ax2.set_ylabel(r'$D_\ell$ [μK²]')
            ax2.set_title('Low-ℓ detail (ℓ<30): Sachs-Wolfe + boundary suppression')
            ax2.legend(fontsize=8)

            chi2_dof = result["chi2_per_dof"]
            fig.suptitle(f'χ²_compressed={result["chi2_compressed"]:.1f}, '
                         f'χ²/DOF(total)={chi2_dof:.2f}', fontsize=10)
            plt.tight_layout()
            plt.savefig("boundary_cmb_spectrum.png", dpi=150)
            print("Saved: boundary_cmb_spectrum.png")

        except ImportError:
            print("matplotlib not available — skipping plot")

if __name__ == "__main__":
    main()
