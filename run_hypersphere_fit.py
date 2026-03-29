# -*- coding: utf-8 -*-
"""
run_hypersphere_fit.py — Hypersphere Cosmology fitting routine
==============================================================
Fits H(z) = H₀(1+z) to cosmic chronometers, SNe Ia (Pantheon+), and BAO.

Key physics:
  H(z) = H₀ · (1+z)                  [fundamental Hubble relation]
  D_C(z) = (c/H₀) · ln(1+z)          [comoving distance, analytic from H(z)]
  D_H(z) = c / H(z) = c / (H₀(1+z)) [Hubble distance]
  D_L(z) = (1+z) · D_C(z)            [luminosity distance]
  γ(z)   = √(2z+1)                   [time dilation]

This replaces the earlier DM(z) = L·zc·log(1+z/zc) formulation which
was inconsistent with H(z) = H₀(1+z). The correct comoving distance
follows directly by integrating c/H(z):
  D_C(z) = ∫₀ᶻ c/H(z') dz' = (c/H₀) ∫₀ᶻ dz'/(1+z') = (c/H₀) ln(1+z)

Optimizer: scipy differential_evolution for MAP, then Metropolis-Hastings
MCMC for posterior sampling. If scipy is unavailable, falls back to a
pure-numpy random-restart optimizer.

Usage:
  python run_hypersphere_fit.py --data-root ./data [--mcmc-steps 2000] [--walkers 32]

Requirements: numpy, scipy (optional but recommended), pandas
"""

import os, sys, math, argparse, json
import numpy as np
import pandas as pd

# ─── Physical constants ──────────────────────────────────────────────────────
c_kms = 299792.458   # speed of light, km/s

# ─── Hypersphere Cosmology distance model ────────────────────────────────────

def DC_hypersphere(z, H0):
    """
    Comoving distance in Mpc for H(z) = H₀(1+z).
    Derived analytically: D_C(z) = (c/H₀) · ln(1+z).
    This is the unique comoving distance consistent with H(z) = H₀(1+z).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return (c_kms / H0) * np.log1p(z)

def DH_hypersphere(z, H0):
    """
    Hubble distance c/H(z) = c / (H₀(1+z)).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return c_kms / (H0 * (1.0 + z))

def DL_hypersphere(z, H0):
    """
    Luminosity distance (1+z)·D_C(z).
    """
    return (1.0 + np.atleast_1d(np.asarray(z, float))) * DC_hypersphere(z, H0)

def H_hypersphere(z, H0):
    """
    Hubble parameter H(z) = H₀(1+z) in km/s/Mpc.
    """
    return H0 * (1.0 + np.atleast_1d(np.asarray(z, float)))

def gamma_time_dilation(z):
    """
    Hypersphere time dilation factor γ(z) = √(2z+1).
    γ=1 at z=0, γ≈√(2z) at high z.
    """
    return np.sqrt(2.0 * np.atleast_1d(np.asarray(z, float)) + 1.0)

# ─── ΛCDM reference (for comparison) ─────────────────────────────────────────

def E_LCDM(z, Om0):
    z = np.atleast_1d(np.asarray(z, float))
    return np.sqrt(Om0 * (1.0 + z)**3 + (1.0 - Om0))

def DC_LCDM(z, H0, Om0):
    z = np.atleast_1d(np.asarray(z, float))
    out = np.zeros_like(z)
    for i, zi in enumerate(z):
        n = max(200, int(zi * 300))
        zs = np.linspace(0.0, zi, n + 1)
        out[i] = (c_kms / H0) * np.trapz(1.0 / E_LCDM(zs, Om0), zs)
    return out

def DH_LCDM(z, H0, Om0):
    return (c_kms / H0) / E_LCDM(z, Om0)

# ─── Data loaders ────────────────────────────────────────────────────────────

def load_cosmic_chronometers(data_root):
    """
    Load cosmic chronometer H(z) measurements.
    Falls back to a built-in literature compilation if no file is found.
    Source: Moresco et al. 2016 + Ratsimbazafy et al. 2017 + Borghi et al. 2022
    """
    p = os.path.join(data_root, "cc_hz.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df['z'].values, df['Hz'].values, df['sigma_Hz'].values

    # Built-in compilation (33 CC measurements, Moresco 2023 compilation)
    data = np.array([
        # z,      H(z),   sigma
        [0.070,   69.0,   19.6],
        [0.090,   69.0,   12.0],
        [0.120,   68.6,   26.2],
        [0.170,   83.0,   8.0 ],
        [0.179,   75.0,   4.0 ],
        [0.199,   75.0,   5.0 ],
        [0.200,   72.9,   29.6],
        [0.270,   77.0,   14.0],
        [0.280,   88.8,   36.6],
        [0.352,   83.0,   14.0],
        [0.380,   83.0,   13.5],
        [0.400,   95.0,   17.0],
        [0.4004,  77.0,   10.2],
        [0.4247,  87.1,   11.2],
        [0.4497,  92.8,   12.9],
        [0.4783,  80.9,   9.0 ],
        [0.480,   97.0,   62.0],
        [0.593,  104.0,   13.0],
        [0.680,   92.0,   8.0 ],
        [0.750,   98.8,   33.6],
        [0.781,  105.0,   12.0],
        [0.875,  125.0,   17.0],
        [0.880,   90.0,   40.0],
        [0.900,  117.0,   23.0],
        [1.037,  154.0,   20.0],
        [1.300,  168.0,   17.0],
        [1.363,  160.0,   33.6],
        [1.430,  177.0,   18.0],
        [1.530,  140.0,   14.0],
        [1.750,  202.0,   40.0],
        [1.965,  186.5,   50.4],
        [2.340,  222.0,   7.0 ],
        [2.360,  226.0,   8.0 ],
    ])
    return data[:, 0], data[:, 1], data[:, 2]

def load_pantheon(data_root):
    """Load Pantheon+ SNe Ia. Returns (z, mu, C_cov)."""
    p = os.path.join(data_root, "pantheon_plus", "Pantheon+SH0ES.dat")
    if not os.path.exists(p):
        p = os.path.join(data_root, "Pantheon+SH0ES.dat")
    if not os.path.exists(p):
        return None, None, None

    with open(p, "r") as f:
        header = f.readline().strip()
    cols = header.split()
    sn = pd.read_csv(p, sep=r"\s+", comment="#", skiprows=1,
                     names=cols, engine="python")
    low = {c.lower(): c for c in sn.columns}
    for zk in ["zcmb", "z_cmb", "zhel", "z"]:
        if zk in low:
            zcol = low[zk]; break
    for mk in ["mu_sh0es", "distmod", "mu"]:
        if mk in low:
            mucol = low[mk]; break

    z = sn[zcol].astype(float).values
    mu = sn[mucol].astype(float).values

    cov_p = os.path.join(data_root, "pantheon_plus", "Pantheon+SH0ES_STAT+SYS.cov")
    if not os.path.exists(cov_p):
        cov_p = os.path.join(data_root, "Pantheon+SH0ES_STAT+SYS.cov")
    if not os.path.exists(cov_p):
        # Use diagonal approximation from scatter
        sigma = 0.15 * np.ones_like(mu)
        C = np.diag(sigma**2)
        return z, mu, C

    with open(cov_p, "r") as f:
        N = int(f.readline().strip())
    C = np.loadtxt(cov_p, skiprows=1).reshape((N, N))
    N = min(N, len(z), len(mu))
    return z[:N], mu[:N], C[:N, :N]

def load_bao(data_root):
    """Load BAO measurements. Returns a DataFrame or None."""
    for rel in ["desi_bao/desi_bao_summary.csv", "desi_bao_summary.csv",
                "bao_model_and_data.csv"]:
        p = os.path.join(data_root, rel)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

# ─── Chi-squared pieces ───────────────────────────────────────────────────────

def mu_from_DL(DL_Mpc):
    """Distance modulus from luminosity distance in Mpc."""
    return 5.0 * np.log10(np.asarray(DL_Mpc, float)) + 25.0

def chi2_CC(z, Hz, sigma, H0):
    """
    Chi-squared for cosmic chronometers vs H(z) = H₀(1+z).
    No nuisance parameters needed — H0 is the only free parameter.
    """
    model = H_hypersphere(z, H0)
    return float(np.sum(((Hz - model) / sigma)**2))

def chi2_SN(z, mu, C, H0):
    """
    Chi-squared for SNe Ia. Marginalizes analytically over absolute magnitude M.
    Uses full covariance matrix (or diagonal if only variances are available).
    """
    DL = DL_hypersphere(z, H0)
    mu_model = mu_from_DL(DL)
    Ci = np.linalg.inv(C)
    one = np.ones_like(mu)
    diff = mu - mu_model
    # Analytic marginalization over M
    Ci_one = Ci @ one
    M_marg = float((diff @ Ci_one) / (one @ Ci_one))
    residual = diff - M_marg * one
    return float(residual @ (Ci @ residual)), M_marg

def chi2_BAO(df, H0, rd):
    """
    Chi-squared for BAO measurements: DM/rd and DH/rd.
    rd is the sound horizon at drag epoch (Mpc).
    """
    if df is None:
        return 0.0, 0

    chi2 = 0.0
    n = 0

    # DM + DH joint measurements
    mask = df.get('DM_over_rd', pd.Series(dtype=float)).notna() & \
           df.get('DH_over_rd', pd.Series(dtype=float)).notna()
    for _, row in df[mask].iterrows():
        z = float(row['zeff'])
        DM_model = float(DC_hypersphere(z, H0))
        DH_model = float(DH_hypersphere(z, H0))
        model = np.array([DM_model / rd, DH_model / rd])
        data  = np.array([row['DM_over_rd'], row['DH_over_rd']])
        sDM, sDH = float(row['DM_over_rd_err']), float(row['DH_over_rd_err'])
        rho = 0.0 if pd.isna(row.get('roff', np.nan)) else float(row['roff'])
        cov = np.array([[sDM**2, rho*sDM*sDH], [rho*sDM*sDH, sDH**2]])
        diff = model - data
        chi2 += float(diff @ np.linalg.inv(cov) @ diff)
        n += 2

    # DV-only measurements
    has_dv = df.columns.str.lower().str.contains('dv').any()
    if has_dv:
        dv_col = [c for c in df.columns if 'DV_over_rd' in c]
        dv_err_col = [c for c in df.columns if 'DV_over_rd_err' in c]
        if dv_col and dv_err_col:
            mask2 = df[dv_col[0]].notna()
            if 'DM_over_rd' in df.columns:
                mask2 = mask2 & df['DM_over_rd'].isna()
            for _, row in df[mask2].iterrows():
                z = float(row['zeff'])
                DM = float(DC_hypersphere(z, H0))
                DH = float(DH_hypersphere(z, H0))
                DV = (DM**2 * z * DH)**(1.0/3.0)
                pred = DV / rd
                chi2 += ((pred - float(row[dv_col[0]])) / float(row[dv_err_col[0]]))**2
                n += 1

    return chi2, n

# ─── Total log-likelihood ─────────────────────────────────────────────────────

def log_posterior(theta, cc_data, sn_data, bao_df, prior_bounds):
    """
    Log-posterior = log-likelihood + log-prior.
    theta = [H0, rd]
      H0: Hubble constant in km/s/Mpc
      rd: sound horizon at drag epoch in Mpc

    Prior: uniform within prior_bounds = [(H0_lo, H0_hi), (rd_lo, rd_hi)]
    """
    H0, rd = theta

    # Hard prior bounds
    for val, (lo, hi) in zip(theta, prior_bounds):
        if not (lo <= val <= hi):
            return -np.inf

    z_cc, Hz_cc, sigma_cc = cc_data
    chi2_cc = chi2_CC(z_cc, Hz_cc, sigma_cc, H0)

    sn_chi2 = 0.0
    if sn_data[0] is not None:
        sn_chi2, _ = chi2_SN(sn_data[0], sn_data[1], sn_data[2], H0)

    bao_chi2, _ = chi2_BAO(bao_df, H0, rd)

    return -0.5 * (chi2_cc + sn_chi2 + bao_chi2)

# ─── Optimizer / MCMC ────────────────────────────────────────────────────────

def find_map(log_post_fn, prior_bounds, n_starts=500, seed=42):
    """
    Find MAP estimate using scipy differential_evolution (global optimizer),
    falling back to random restart + Nelder-Mead if scipy unavailable.
    """
    rng = np.random.default_rng(seed)
    best_logp = -np.inf
    best_theta = None

    try:
        from scipy.optimize import differential_evolution, minimize

        def neg_log_post(x):
            lp = log_post_fn(x)
            return -lp if np.isfinite(lp) else 1e12

        result = differential_evolution(neg_log_post, prior_bounds,
                                        seed=seed, maxiter=2000,
                                        tol=1e-8, popsize=20, polish=True)
        if np.isfinite(-result.fun):
            best_theta = result.x
            best_logp = -result.fun

    except ImportError:
        # Pure numpy random restart
        lows = np.array([b[0] for b in prior_bounds])
        highs = np.array([b[1] for b in prior_bounds])
        for _ in range(n_starts):
            theta = rng.uniform(lows, highs)
            lp = log_post_fn(theta)
            if lp > best_logp:
                best_logp = lp
                best_theta = theta.copy()

    return best_theta, best_logp

def metropolis_hastings(log_post_fn, theta0, prior_bounds,
                        n_steps=5000, proposal_scales=None, seed=123,
                        burn_in=0.3):
    """
    Metropolis-Hastings MCMC sampler.

    Returns:
      chain  : array of shape (n_kept, n_params) after burn-in
      logps  : log-posterior values for each kept sample
      accept : acceptance rate
    """
    rng = np.random.default_rng(seed)
    n_params = len(theta0)
    chain_raw = np.zeros((n_steps, n_params))
    logps_raw = np.zeros(n_steps)

    if proposal_scales is None:
        # Default step sizes: ~2% of parameter range
        proposal_scales = np.array([(b[1]-b[0])*0.02 for b in prior_bounds])

    current = np.array(theta0, dtype=float)
    current_logp = log_post_fn(current)
    chain_raw[0] = current
    logps_raw[0] = current_logp
    n_accept = 0

    # Adaptive tuning phase (first 20% of steps)
    tune_every = max(100, n_steps // 50)

    for i in range(1, n_steps):
        proposal = current + rng.normal(0.0, proposal_scales, n_params)
        proposal_logp = log_post_fn(proposal)

        log_alpha = proposal_logp - current_logp
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_logp = proposal_logp
            n_accept += 1

        chain_raw[i] = current
        logps_raw[i] = current_logp

        # Adaptive step size tuning during burn-in
        if i < n_steps * burn_in and i % tune_every == 0:
            rate = n_accept / i
            if rate < 0.15:
                proposal_scales *= 0.7
            elif rate > 0.40:
                proposal_scales *= 1.3

    burn = int(n_steps * burn_in)
    chain = chain_raw[burn:]
    logps = logps_raw[burn:]
    accept_rate = n_accept / n_steps

    return chain, logps, accept_rate

def chain_summary(chain, param_names):
    """Print mean ± std for each parameter and return dict."""
    print("\n── MCMC Posterior Summary ───────────────────────────────")
    result = {}
    for i, name in enumerate(param_names):
        samples = chain[:, i]
        mean = np.mean(samples)
        std = np.std(samples)
        lo, hi = np.percentile(samples, [16, 84])
        print(f"  {name:12s}: {mean:.4f} ± {std:.4f}  [68% CI: {lo:.4f} – {hi:.4f}]")
        result[name] = dict(mean=float(mean), std=float(std),
                            p16=float(lo), p84=float(hi))
    return result

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fit Hypersphere Cosmology H(z)=H₀(1+z) to CC, SNe Ia, BAO")
    ap.add_argument("--data-root", required=True,
                    help="Directory containing data subdirectories")
    ap.add_argument("--mcmc-steps", type=int, default=3000,
                    help="Number of MCMC steps (default: 3000)")
    ap.add_argument("--no-mcmc", action="store_true",
                    help="Skip MCMC; only find MAP estimate")
    ap.add_argument("--subsample-sn", type=int, default=1,
                    help="Use every Nth SNe Ia point (speeds up debugging)")
    args = ap.parse_args()

    print("=" * 60)
    print("Hypersphere Cosmology Fitter")
    print("H(z) = H₀(1+z)  →  D_C(z) = (c/H₀)·ln(1+z)")
    print("=" * 60)

    # Load data
    cc_data = load_cosmic_chronometers(args.data_root)
    print(f"Loaded {len(cc_data[0])} cosmic chronometer points")

    sn_z, sn_mu, sn_C = load_pantheon(args.data_root)
    if sn_z is not None:
        if args.subsample_sn > 1:
            idx = np.arange(len(sn_z))[::args.subsample_sn]
            sn_z = sn_z[idx]; sn_mu = sn_mu[idx]
            sn_C = sn_C[np.ix_(idx, idx)]
        print(f"Loaded {len(sn_z)} SNe Ia (subsample={args.subsample_sn})")
    else:
        print("SNe Ia data not found — fitting CC only")

    bao_df = load_bao(args.data_root)
    if bao_df is not None:
        print(f"Loaded {len(bao_df)} BAO measurements")
    else:
        print("BAO data not found — will skip")

    sn_data = (sn_z, sn_mu, sn_C)

    # Parameter prior bounds: [H0 (km/s/Mpc), rd (Mpc)]
    prior_bounds = [(50.0, 80.0), (120.0, 175.0)]
    param_names  = ["H0", "rd"]

    def log_post(theta):
        return log_posterior(theta, cc_data, sn_data, bao_df, prior_bounds)

    # Step 1: MAP
    print("\nFinding MAP estimate...")
    theta_map, logp_map = find_map(log_post, prior_bounds)
    H0_map, rd_map = theta_map

    # Compute individual chi-squared at MAP
    c2_cc = chi2_CC(*cc_data, H0_map)
    c2_sn, M_sn = chi2_SN(*sn_data, H0_map) if sn_z is not None else (0.0, 0.0)
    c2_bao, N_bao = chi2_BAO(bao_df, H0_map, rd_map)
    N_cc = len(cc_data[0])
    N_sn = len(sn_z) if sn_z is not None else 0
    N_total = N_cc + N_sn + N_bao
    N_params = 2
    chi2_total = c2_cc + c2_sn + c2_bao
    dof = N_total - N_params

    print(f"\n── MAP Result ───────────────────────────────────────────")
    print(f"  H₀          = {H0_map:.3f} km/s/Mpc")
    print(f"  r_d         = {rd_map:.2f} Mpc")
    print(f"  M (SN abs)  = {M_sn:.3f}")
    print(f"  χ²_CC       = {c2_cc:.2f}  (N={N_cc})")
    print(f"  χ²_SN       = {c2_sn:.2f}  (N={N_sn})")
    print(f"  χ²_BAO      = {c2_bao:.2f}  (N_terms={N_bao})")
    print(f"  χ²_total    = {chi2_total:.2f}")
    print(f"  DOF         = {dof}  →  χ²/DOF = {chi2_total/max(1,dof):.3f}")

    if args.no_mcmc:
        print("\n(MCMC skipped — use without --no-mcmc for posterior samples)")
        return

    # Step 2: MCMC
    print(f"\nRunning Metropolis-Hastings MCMC ({args.mcmc_steps} steps)...")
    chain, logps, accept = metropolis_hastings(
        log_post, theta_map, prior_bounds,
        n_steps=args.mcmc_steps, seed=42)
    print(f"  Acceptance rate: {accept:.2%}")
    print(f"  Kept {len(chain)} samples after burn-in")

    summary = chain_summary(chain, param_names)

    # Save results
    out = {
        "model": "H(z) = H0*(1+z), D_C(z) = (c/H0)*ln(1+z)",
        "MAP": {"H0": float(H0_map), "rd": float(rd_map)},
        "chi2": {"CC": float(c2_cc), "SN": float(c2_sn), "BAO": float(c2_bao),
                 "total": float(chi2_total), "dof": int(dof),
                 "chi2_per_dof": float(chi2_total / max(1, dof))},
        "MCMC": summary
    }
    with open("hypersphere_fit_results.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save("hypersphere_mcmc_chain.npy", chain)

    print("\nSaved: hypersphere_fit_results.json, hypersphere_mcmc_chain.npy")

    # Save chain as CSV for human inspection
    chain_df = pd.DataFrame(chain, columns=param_names)
    chain_df.to_csv("hypersphere_mcmc_chain.csv", index=False)
    print("Saved: hypersphere_mcmc_chain.csv")

if __name__ == "__main__":
    main()
