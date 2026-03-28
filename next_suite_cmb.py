# next_suite_cmb.py — robust compressed-CMB + SN + BAO fits, duality slope, BAO residuals
import os, argparse, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c_kms = 299792.458

def mu_from_DL_Mpc(DL_Mpc): return 5.0*np.log10(np.asarray(DL_Mpc,float)) + 25.0
def trapz_int(x,y): return np.trapz(y,x)
def arr(x): return np.atleast_1d(np.asarray(x,float))

# ---------- LCDM ----------
def E_LCDM(z, Om): z=arr(z); return np.sqrt(Om*(1+z)**3 + (1-Om))
def DM_LCDM(z, H0, Om):
    z=arr(z); out=np.zeros_like(z)
    for i, zi in enumerate(z):
        n=max(140,int(zi*240)); zs=np.linspace(0.0,zi,n+1)
        out[i]=(c_kms/H0)*trapz_int(zs,1.0/E_LCDM(zs,Om))
    return out
def DH_LCDM(z, H0, Om): return (c_kms/H0)/E_LCDM(arr(z),Om)

# ---------- HEA (time-sigmoid + curvature projector) ----------
def DM_HEA_timesig(z, L, zc, n):
    z=arr(z); zc=float(zc); n=float(n)
    if n<=0: n=1.0
    x=z/zc
    if abs(n-1.0)<1e-12:  return L*zc*np.log1p(x)
    if abs(n-2.0)<1e-12:  return L*zc*np.arctan(x)
    out=np.zeros_like(z)
    for i, zi in enumerate(z):
        N=max(220,int(120*zi/zc)+120)
        u=np.linspace(0.0,zi,N+1)
        w=1.0/(1.0+(u/zc)**n)
        out[i]=L*np.trapz(w,u)
    return out
def DH_HEA_timesig(z, L, zc, n):
    z=arr(z); zc=float(zc); n=float(n)
    return L/(1.0+(z/zc)**n)
def DM_project_curvature(chi, kappa):
    chi=arr(chi); k=float(kappa)
    if abs(k)<1e-16: return chi
    if k>0: r=1.0/np.sqrt(k); return r*np.sin(np.sqrt(k)*chi)
    r=1.0/np.sqrt(-k); return r*np.sinh(np.sqrt(-k)*chi)
def DM_HEA_timesig_kappa(z,L,zc,n,kappa): return DM_project_curvature(DM_HEA_timesig(z,L,zc,n),kappa)

# ---------- loaders ----------
def load_sn(root):
    p=os.path.join(root,"pantheon_plus","Pantheon+SH0ES.dat")
    if not os.path.exists(p): p=os.path.join(root,"Pantheon+SH0ES.dat")
    with open(p,"r") as f:
        header=""
        for line in f:
            s=line.strip()
            if s: header=s; break
    cols=header.split()
    sn=pd.read_csv(p, sep=r"\s+", comment="#", skiprows=1, names=cols, engine="python")
    L={c.lower():c for c in sn.columns}
    for zc in ["zcmb","z_cmb","zhel","z_hel","z"]:
        if zc in L: zcol=L[zc]; break
    else: raise RuntimeError("SN redshift column not found")
    for mc in ["mu_sh0es","distmod","mu"]:
        if mc in L: mucol=L[mc]; break
    else: raise RuntimeError("SN mu column not found")
    z=sn[zcol].astype(float).values
    mu=sn[mucol].astype(float).values
    cov=os.path.join(root,"pantheon_plus","Pantheon+SH0ES_STAT+SYS.cov")
    if not os.path.exists(cov): cov=os.path.join(root,"Pantheon+SH0ES_STAT+SYS.cov")
    with open(cov,"r") as f: N=int(f.readline().strip())
    C=np.loadtxt(cov,skiprows=1).reshape((N,N))
    N=min(N,len(z),len(mu))
    return z[:N],mu[:N],C[:N,:N]

def load_bao(root):
    p=os.path.join(root,"desi_bao","desi_bao_summary.csv")
    if not os.path.exists(p): p=os.path.join(root,"desi_bao_summary.csv")
    return pd.read_csv(p)

def _nearest_spd(C):
    # symmetrize
    C = 0.5*(C + C.T)
    # eigen clean-up
    w, V = np.linalg.eigh(C)
    floor = max(1e-18, 1e-12*np.max(w))
    w_clipped = np.clip(w, floor, None)
    C_spd = (V * w_clipped) @ V.T
    return C_spd

def load_cmb_compressed(root):
    base=os.path.join(root,"planck_acoustic")
    mean_csv=os.path.join(base,"planck_compressed.csv")
    cov_csv =os.path.join(base,"planck_compressed_cov.csv")
    m=pd.read_csv(mean_csv)
    vals={}
    for key in ["theta","omegabh2","omegach2","ns"]:
        row=m.loc[m['param'].str.lower()==key]
        if row.empty: raise RuntimeError(f"CMB compressed: missing {key}")
        vals[key]=float(row['value'].iloc[0])
    C=np.loadtxt(cov_csv, delimiter=",")
    if C.shape!=(4,4): raise RuntimeError("CMB compressed cov must be 4x4")
    C = _nearest_spd(C)
    Ci = np.linalg.inv(C)
    return vals, Ci

# ---------- likelihood pieces ----------
def chi2_sn(z,mu,C,DMf,pars):
    Ci=np.linalg.inv(C); one=np.ones_like(mu)
    DM=DMf(z,*pars); DL=(1.0+z)*DM; mu_mod=mu_from_DL_Mpc(DL)
    d=mu-mu_mod
    Ci1=Ci@one; iC1=float(one@Ci1); M=float((d@Ci1)/iC1)
    r=d-M*one
    return float(r@(Ci@r)), M, len(mu)

def chi2_bao(df,DMf,DHf,pars,rd):
    chi2=0.0; n=0
    m=df['DM_over_rd'].notna() & df['DH_over_rd'].notna()
    for _,r in df[m].iterrows():
        z=float(r['zeff']); DM=float(DMf(z,*pars)); DH=float(DHf(z,*pars))
        model=np.array([DM/rd, DH/rd], float)
        data =np.array([r['DM_over_rd'], r['DH_over_rd']], float)
        sDM,sDH=float(r['DM_over_rd_err']),float(r['DH_over_rd_err'])
        rho=0.0 if pd.isna(r['roff']) else float(r['roff'])
        cov=np.array([[sDM**2, rho*sDM*sDH],[rho*sDM*sDH, sDH**2]], float)
        inv=np.linalg.inv(cov); diff=model-data
        chi2+=float(diff@inv@diff); n+=2
    m2=df['DV_over_rd'].notna() & df['DM_over_rd'].isna()
    for _,r in df[m2].iterrows():
        z=float(r['zeff']); DM=float(DMf(z,*pars)); DH=float(DHf(z,*pars))
        DV=(DM**2*z*DH)**(1.0/3.0); pred=DV/rd
        chi2+=((pred-float(r['DV_over_rd']))/float(r['DV_over_rd_err']))**2; n+=1
    return chi2,n

def chi2_cmb_compressed(theta100_model, wb, wc, ns, cmb_mean, Ci):
    v = np.array([
        theta100_model - cmb_mean["theta"],
        wb             - cmb_mean["omegabh2"],
        wc             - cmb_mean["omegach2"],
        ns             - cmb_mean["ns"]
    ], float)
    return float(v @ (Ci @ v))

# ---------- growth (optional; LCDM only) ----------
def Om_z_LCDM(z, Om0):
    Ez = E_LCDM(z, Om0)
    return (Om0*(1.0+z)**3)/(Ez**2)

def chi2_growth_LCDM(df, H0, Om0):
    gamma=0.55
    z=df['z'].values
    shape = Om_z_LCDM(z, Om0)**gamma
    w = 1.0/(df['sigma'].values**2)
    s8_0 = np.sum(df['fs8'].values*shape*w)/np.sum(shape**2*w)
    res = df['fs8'].values - s8_0*shape
    return float(np.sum((res**2)*w)), s8_0

def maybe_load_growth(root):
    for p in [os.path.join(root,"growth","growth_fs8.csv"),
              os.path.join(root,"growth_fs8.csv")]:
        if os.path.exists(p):
            df=pd.read_csv(p)
            need={"z","fs8","sigma"}
            if not need.issubset(df.columns): raise RuntimeError("growth csv needs z,fs8,sigma")
            return df,p
    return None,None

# ---------- optimizer ----------
def random_search(obj,bounds,trials=3000,seed=123):
    rng=np.random.default_rng(seed); best=(None,np.inf,None)
    los=np.array([b[0] for b in bounds]); his=np.array([b[1] for b in bounds])
    eps=1e-9; los=los+eps*(his-los); his=his-eps*(his-los)
    for _ in range(trials):
        p=[rng.uniform(lo,hi) for lo,hi in bounds]
        val,info=obj(p)
        if val<best[1]: best=(p,val,info)
    return best

def maybe_refine(objective, start, bounds=None):
    def clamp(x,bnds):
        if bnds is None: return np.asarray(x,float)
        y=[]
        for xi,(lo,hi) in zip(x,bnds):
            xi=min(max(xi,lo),hi); y.append(xi)
        return np.array(y,float)
    try:
        from scipy.optimize import minimize
        x0=clamp(start,bounds)
        def f(x): return objective(list(clamp(x,bounds)))[0]
        res=minimize(f,x0,method="Nelder-Mead",options=dict(maxiter=8000))
        xb=clamp(res.x,bounds); val,info=objective(list(xb))
        return list(xb),val,info,("OK" if res.success else "NM")
    except Exception as e:
        val,info=objective(start); return start,val,info,f"no scipy ({e})"

# ---------- duality helper ----------
def sn_bin_DL(zsn, musn, z0, dz=0.03, minN=20, maxdz=0.12):
    w=dz
    for _ in range(10):
        m=np.abs(zsn - z0) <= w
        if m.sum()>=minN or w>=maxdz: break
        w += 0.01
    if m.sum()==0: return np.nan, np.nan, 0, w
    DL = 10.0**((musn[m]-25.0)/5.0)
    med=np.median(DL)
    mad=np.median(np.abs(DL-med))*1.4826
    sig = mad/max(1,np.sqrt(m.sum()))
    if not np.isfinite(sig) or sig==0: sig=np.std(DL)/max(1,np.sqrt(m.sum()))
    return med, sig, int(m.sum()), w

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--trials",type=int,default=6000)
    ap.add_argument("--refine",action="store_true")
    ap.add_argument("--rd-min",type=float,default=100.0)
    ap.add_argument("--rd-max",type=float,default=220.0)
    args=ap.parse_args()

    zsn, musn, Csn = load_sn(args.data_root)
    bao            = load_bao(args.data_root)
    cmb_mean, Ci   = load_cmb_compressed(args.data_root)
    growth_df, growth_path = maybe_load_growth(args.data_root)

    zstar = 1090.0  # effective last-scattering redshift for θ*

    # ---- parameter bounds ----
    rd_bounds=(min(args.rd_min,args.rd_max), max(args.rd_min,args.rd_max))
    # LCDM params: [H0, Om, rd, omegabh2, omegach2, ns]
    bLCDM=[(60,85),(0.20,0.40), rd_bounds, (0.018,0.026),(0.09,0.15),(0.94,0.99)]
    # HEA params: [L, zc, n, kappa, rd, omegabh2, omegach2, ns]
    bHEA=[(1.5e3,3.0e4),(0.5,3.0),(0.6,3.0),(-1e-7,1e-7), rd_bounds, (0.018,0.026),(0.09,0.15),(0.94,0.99)]

    def safe_theta100(rd, DMz):
        if not np.isfinite(rd) or not np.isfinite(DMz): return None
        if DMz <= 0: return None
        theta100 = 100.0*rd/DMz
        # sanity range: Planck ~1.04; allow generous [0.2, 5]
        if theta100<=0.2 or theta100>=5.0: return None
        return theta100

    # ---- objectives ----
    def obj_LCDM(p):
        H0,Om,rd,wb,wc,ns = p
        # SN
        c2sn,M,Nsn = chi2_sn(zsn,musn,Csn,DM_LCDM,(H0,Om))
        # BAO
        c2b,Nb = chi2_bao(bao,DM_LCDM,DH_LCDM,(H0,Om),rd)
        # CMB compressed
        DMz = float(DM_LCDM(zstar,H0,Om))
        th = safe_theta100(rd, DMz)
        if th is None: return 1e12, dict(reason="bad_theta100_LCDM")
        c2cmb = chi2_cmb_compressed(th, wb, wc, ns, cmb_mean, Ci)
        chi = c2sn + c2b + c2cmb
        info=dict(Nsn=Nsn,Nbao=Nb,M=M,theta100=th,c2cmb=c2cmb)
        if growth_df is not None:
            c2g,s8 = chi2_growth_LCDM(growth_df, H0, Om)
            chi += c2g; info.update(chi2_growth=c2g, s8_0=s8)
        return chi, info

    def obj_HEA(p):
        L,zc,n,kappa,rd,wb,wc,ns = p
        DMf=lambda z,*pp: DM_HEA_timesig_kappa(z,L,zc,n,kappa)
        DHf=lambda z,*pp: DH_HEA_timesig(z,L,zc,n)
        # SN
        c2sn,M,Nsn = chi2_sn(zsn,musn,Csn,DMf,())
        # BAO
        c2b,Nb = chi2_bao(bao,DMf,DHf,(),rd)
        # CMB compressed
        DMz=float(DMf(zstar))
        th = safe_theta100(rd, DMz)
        if th is None: return 1e12, dict(reason="bad_theta100_HEA")
        c2cmb = chi2_cmb_compressed(th, wb, wc, ns, cmb_mean, Ci)
        chi = c2sn + c2b + c2cmb
        info=dict(Nsn=Nsn,Nbao=Nb,M=M,theta100=th,c2cmb=c2cmb)
        return chi, info

    # ---- fit ----
    bestL = random_search(obj_LCDM, bLCDM, trials=args.trials, seed=1)
    bestH = random_search(obj_HEA , bHEA , trials=args.trials, seed=2)
    if args.refine:
        pr,val,inf,note = maybe_refine(obj_LCDM, bestL[0], bLCDM); print("[refine] LCDM:",note); bestL=(pr,val,inf)
        pr,val,inf,note = maybe_refine(obj_HEA , bestH[0], bHEA ); print("[refine] HEA-timesig+κ:",note); bestH=(pr,val,inf)

    # ---- report ----
    def show_model(name, pack):
        pars, chi2, info = pack
        print(f"== {name}")
        print("params:", np.array(pars,dtype=float))
        print("chi2_total:", float(chi2))
        print(f"  SN: N = {info.get('Nsn','?')}  | M ~ {info.get('M','?')}")
        print(f"  BAO terms ≈ {info.get('Nbao','?')}")
        print(f"  CMB mode: compressed  | 100*theta* model = {info.get('theta100','?')}  | chi2_CMB = {info.get('c2cmb','?')}")
        # AIC/BIC bookkeeping
        k = len(pars)
        N_eff = info.get('Nsn',0) + info.get('Nbao',0) + 4  # 4 CMB compressed params
        AIC = chi2 + 2*k
        BIC = chi2 + k*np.log(max(1,N_eff))
        print(f"  AIC: {AIC:.3f}  BIC: {BIC:.3f}\n")
    show_model("LCDM [H0, Om, r_d, ω_bh², ω_ch², n_s]", bestL)
    show_model("HEA-timesig+κ [L, z_c, n, κ, r_d, ω_bh², ω_ch², n_s]", bestH)

    # ---- BAO residuals + tables (using the better chi2) ----
    use_HEA = bestH[1] <= bestL[1]
    if use_HEA:
        L,zc,n,kappa,rd,wb,wc,ns = bestH[0]
        DMf=lambda z,*pp: DM_HEA_timesig_kappa(z,L,zc,n,kappa)
        DHf=lambda z,*pp: DH_HEA_timesig(z,L,zc,n)
        tag="HEA-timesig+κ"
    else:
        H0,Om,rd,wb,wc,ns = bestL[0]
        DMf=lambda z,*pp: DM_LCDM(z,H0,Om)
        DHf=lambda z,*pp: DH_LCDM(z,H0,Om)
        tag="LCDM"

    rows=[]
    for _,r in load_bao(args.data_root).iterrows():
        z=float(r['zeff']); DM=float(DMf(z)); DH=float(DHf(z))
        row={'z':z,'tracer':r.get('tracer','?')}
        if pd.notna(r.get('DM_over_rd')) and pd.notna(r.get('DH_over_rd')):
            row.update(
                DM_over_rd=r['DM_over_rd'], DM_over_rd_err=r['DM_over_rd_err'],
                DH_over_rd=r['DH_over_rd'], DH_over_rd_err=r['DH_over_rd_err'],
                DM_over_rd_model=DM/rd, DH_over_rd_model=DH/rd
            )
        if pd.notna(r.get('DV_over_rd')) and pd.isna(r.get('DM_over_rd')):
            DV=(DM**2*z*DH)**(1.0/3.0)
            row.update(
                DV_over_rd=r['DV_over_rd'], DV_over_rd_err=r['DV_over_rd_err'],
                DV_over_rd_model=DV/rd
            )
        rows.append(row)
    preds=pd.DataFrame(rows)
    preds.to_csv("bao_model_and_data.csv", index=False)

    res_rows=[]
    for _,row in preds.iterrows():
        z=row['z']
        if not pd.isna(row.get('DM_over_rd')) and not pd.isna(row.get('DM_over_rd_model')):
            pull=(row['DM_over_rd']-row['DM_over_rd_model'])/row['DM_over_rd_err']
            res_rows.append(dict(z=z,which="DM/rd",pull=pull))
        if not pd.isna(row.get('DH_over_rd')) and not pd.isna(row.get('DH_over_rd_model')):
            pull=(row['DH_over_rd']-row['DH_over_rd_model'])/row['DH_over_rd_err']
            res_rows.append(dict(z=z,which="DH/rd",pull=pull))
        if not pd.isna(row.get('DV_over_rd')) and not pd.isna(row.get('DV_over_rd_model')):
            pull=(row['DV_over_rd']-row['DV_over_rd_model'])/row['DV_over_rd_err']
            res_rows.append(dict(z=z,which="DV/rd",pull=pull))
    pd.DataFrame(res_rows).to_csv("bao_residuals_table.csv", index=False)

    fig,ax=plt.subplots(figsize=(8,5))
    m = preds[['DM_over_rd','DM_over_rd_model','DM_over_rd_err']].notna().all(axis=1)
    ax.errorbar(preds.loc[m,'z'], (preds.loc[m,'DM_over_rd']-preds.loc[m,'DM_over_rd_model'])/preds.loc[m,'DM_over_rd_err'],
                fmt='o', label='DM/rd')
    m = preds[['DH_over_rd','DH_over_rd_model','DH_over_rd_err']].notna().all(axis=1)
    ax.errorbar(preds.loc[m,'z'], (preds.loc[m,'DH_over_rd']-preds.loc[m,'DH_over_rd_model'])/preds.loc[m,'DH_over_rd_err'],
                fmt='s', label='DH/rd')
    m = preds[['DV_over_rd','DV_over_rd_model','DV_over_rd_err']].notna().all(axis=1)
    ax.errorbar(preds.loc[m,'z'], (preds.loc[m,'DV_over_rd']-preds.loc[m,'DV_over_rd_model'])/preds.loc[m,'DV_over_rd_err'],
                fmt='^', label='DV/rd')
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('z'); ax.set_ylabel('Residual (data-model)/σ')
    ax.set_title(f'BAO residuals vs {tag}'); ax.legend()
    plt.tight_layout(); plt.savefig("bao_residuals.png", dpi=150)

    # ---- distance–duality slope (model-dependent; SN vs model DA) ----
    def duality_table(DMf):
        rows=[]
        for _,r in load_bao(args.data_root).iterrows():
            z=float(r['zeff'])
            DL_SN, DLsig, Nsel, used_dz = sn_bin_DL(zsn, musn, z, dz=0.03)
            if not np.isfinite(DL_SN): continue
            DM=float(DMf(z)); DA=DM/(1.0+z)
            eta = DL_SN/(((1.0+z)**2)*DA)
            sigma=DLsig/(((1.0+z)**2)*DA)
            rows.append(dict(z=z, N_SN=Nsel, eta=eta, sigma=sigma))
        tab=pd.DataFrame(rows).sort_values("z").reset_index(drop=True)
        e=tab['eta'].values; s=tab['sigma'].values; w=1.0/np.clip(s**2,1e-30,np.inf)
        mean=np.sum(w*e)/np.sum(w); scal=1.0/mean; ecal=scal*e
        X=np.vstack([np.ones_like(tab.z.values), tab.z.values]).T
        W=np.diag(w)
        beta=np.linalg.inv(X.T@W@X) @ (X.T@W@ecal)
        covb=np.linalg.inv(X.T@W@X)
        a0,a1=beta; sa1=np.sqrt(covb[1,1])
        tab['eta_cal']=ecal; tab['pull']=(ecal-1.0)/s
        return tab, scal, a1, sa1

    if bestH[1] <= bestL[1]:
        L,zc,n,kappa,rd,wb,wc,ns = bestH[0]
        tab, s, a1, sa1 = duality_table(DMf=lambda z: DM_HEA_timesig_kappa(z,L,zc,n,kappa))
        tag_du="HEA"
    else:
        H0,Om,rd,wb,wc,ns = bestL[0]
        tab, s, a1, sa1 = duality_table(DMf=lambda z: DM_LCDM(z,H0,Om))
        tag_du="LCDM"
    tab.to_csv(f"duality_table_{tag_du}.csv", index=False)
    print(f"Distance–duality ({tag_du}): s={s:.4f},  a1={a1:+.4e} ± {sa1:.4e}  ({a1/sa1:+.2f}σ)")

    out = {
        "LCDM": {"params": list(map(float,bestL[0])), "chi2": float(bestL[1])},
        "HEA_timesig_kappa": {"params": list(map(float,bestH[0])), "chi2": float(bestH[1])},
        "used_model_for_plots": ("HEA" if bestH[1] <= bestL[1] else "LCDM"),
        "cmb_mean_used": cmb_mean,
        "files": {
            "bao_residuals": "bao_residuals.png",
            "bao_model_and_data": "bao_model_and_data.csv",
            "bao_residuals_table": "bao_residuals_table.csv",
            "duality_table": f"duality_table_{tag_du}.csv"
        }
    }
    with open("next_suite_summary.json","w") as f: json.dump(out, f, indent=2)
    print("\nSaved: bao_residuals.png, bao_model_and_data.csv, bao_residuals_table.csv, next_suite_summary.json, duality_table_*.csv")

if __name__=="__main__":
    main()
