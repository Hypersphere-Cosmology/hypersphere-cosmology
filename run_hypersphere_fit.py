# -*- coding: utf-8 -*-
import os, sys, time, math, argparse
import numpy as np
import pandas as pd

c_kms = 299792.458

def mu_from_DL_Mpc(DL_Mpc): return 5.0*np.log10(DL_Mpc) + 25.0
def trapz_int(x, y): return np.trapz(y, x)

def E_of_z_LCDM(z, Om): return np.sqrt(Om*(1+z)**3 + (1-Om))
def DM_LCDM(z, H0, Om):
    z = np.atleast_1d(z).astype(float)
    out = np.zeros_like(z)
    for i, zi in enumerate(z):
        n = max(120, int(zi*220))
        zs = np.linspace(0.0, zi, n+1)
        out[i] = (c_kms/H0)*trapz_int(zs, 1.0/E_of_z_LCDM(zs, Om))
    return out
def DH_LCDM(z, H0, Om): return (c_kms/H0)/E_of_z_LCDM(np.asarray(z,float), Om)

def DM_HEA_log(z,L): return L*np.log1p(np.atleast_1d(z).astype(float))
def DH_HEA_log(z,L): return L/(1.0+np.atleast_1d(z).astype(float))

def DM_HEA_sat_alpha(z,L,a): return L*np.log1p((1.0+np.atleast_1d(z).astype(float))**a)/a
def DH_HEA_sat_alpha(z,L,a): z=np.atleast_1d(z).astype(float);return L*((1+z)**(a-1.0))/(1+(1+z)**a)

def DM_HEA_sat_exp(z,L,zc): z=np.atleast_1d(z).astype(float);return L*np.log1p(z)*(1.0-np.exp(-z/zc))
def DH_HEA_sat_exp(z,L,zc): z=np.atleast_1d(z).astype(float);return L*((1-np.exp(-z/zc))/(1+z)+(np.log1p(z)/zc)*np.exp(-z/zc))

def load_pantheon(data_root):
    p=os.path.join(data_root,"pantheon_plus","Pantheon+SH0ES.dat")
    if not os.path.exists(p): p=os.path.join(data_root,"Pantheon+SH0ES.dat")
    with open(p,"r") as f: header=f.readline().strip()
    cols=header.split()
    sn=pd.read_csv(p,delim_whitespace=True,comment="#",skiprows=1,names=cols,engine="python")
    zcol="zCMB" if "zCMB" in sn.columns else ("zcmb" if "zcmb" in sn.columns else "z")
    mucol="MU_SH0ES" if "MU_SH0ES" in sn.columns else ("DISTMOD" if "DISTMOD" in sn.columns else "MU")
    z=sn[zcol].astype(float).values; mu=sn[mucol].astype(float).values
    cov=os.path.join(data_root,"pantheon_plus","Pantheon+SH0ES_STAT+SYS.cov")
    if not os.path.exists(cov): cov=os.path.join(data_root,"Pantheon+SH0ES_STAT+SYS.cov")
    with open(cov,"r") as f: N=int(f.readline().strip())
    arr=np.loadtxt(cov,skiprows=1); C=arr.reshape((N,N))
    N=min(N,len(z),len(mu)); return z[:N],mu[:N],C[:N,:N]

def load_bao(data_root):
    p=os.path.join(data_root,"desi_bao","desi_bao_summary.csv")
    if not os.path.exists(p): p=os.path.join(data_root,"desi_bao_summary.csv")
    return pd.read_csv(p)

def load_planck_theta(data_root):
    p=os.path.join(data_root,"planck_acoustic","planck_theta_star.csv")
    if not os.path.exists(p): p=os.path.join(data_root,"planck_theta_star.csv")
    df=pd.read_csv(p)
    row=df.loc[df['parameter'].str.contains('theta',case=False)].iloc[0]
    return float(row['value']),float(row['sigma'])

def chi2_sn(z,mu,C,DM_func,pars,subsample=1):
    if subsample>1: idx=np.arange(len(z))[::subsample];z,mu,C=z[idx],mu[idx],C[np.ix_(idx,idx)]
    Ci=np.linalg.inv(C);one=np.ones_like(mu)
    DM=DM_func(z,*pars);DL=(1+z)*DM;mu_model=mu_from_DL_Mpc(DL)
    d=mu-mu_model;Ci1=Ci@one;iC1=float(one@Ci1);M=float((d@Ci1)/iC1);r=d-M*one
    return float(r@(Ci@r)),M,len(mu)

def chi2_bao(df,DMf,DHf,pars,rd):
    chi2=0;n=0
    m=df['DM_over_rd'].notna() & df['DH_over_rd'].notna()
    for _,r in df[m].iterrows():
        z=float(r['zeff']);DM=float(DMf(z,*pars));DH=float(DHf(z,*pars))
        model=np.array([DM/rd,DH/rd]);data=np.array([r['DM_over_rd'],r['DH_over_rd']])
        sDM,sDH=float(r['DM_over_rd_err']),float(r['DH_over_rd_err'])
        rho=0.0 if pd.isna(r['roff']) else float(r['roff'])
        cov=np.array([[sDM**2,rho*sDM*sDH],[rho*sDM*sDH,sDH**2]]);inv=np.linalg.inv(cov)
        diff=model-data;chi2+=float(diff@inv@diff);n+=2
    m2=df['DV_over_rd'].notna() & df['DM_over_rd'].isna()
    for _,r in df[m2].iterrows():
        z=float(r['zeff']);DM=float(DMf(z,*pars));DH=float(DHf(z,*pars))
        DV=(DM**2*z*DH)**(1.0/3.0);pred=DV/rd
        chi2+=((pred-float(r['DV_over_rd']))/float(r['DV_over_rd_err']))**2;n+=1
    return chi2,n

def chi2_planck_theta(DMf,pars,rd,tm,ts):
    z=1090.0;DM=float(DMf(z,*pars));ht=100*rd/DM
    return ((ht-tm)/ts)**2,ht

def random_search(obj,bounds,trials=300,seed=123):
    rng=np.random.default_rng(seed);best=(None,np.inf,None)
    for _ in range(trials):
        pars=[rng.uniform(lo,hi) for (lo,hi) in bounds]
        val,extra=obj(pars)
        if val<best[1]: best=(pars,val,extra)
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--subsample",type=int,default=1)
    ap.add_argument("--random-trials",type=int,default=300)
    args=ap.parse_args()

    z,mu,C=load_pantheon(args.data_root);bao=load_bao(args.data_root)
    tmean,tsig=load_planck_theta(args.data_root)

    def total_LCDM(p):
        H0,Om,rd=p
        c2sn,M,N=chi2_sn(z,mu,C,DM_LCDM,(H0,Om),args.subsample)
        c2b,Nb=chi2_bao(bao,DM_LCDM,DH_LCDM,(H0,Om),rd)
        c2p,ht=chi2_planck_theta(DM_LCDM,(H0,Om),rd,tmean,tsig)
        return c2sn+c2b+c2p,dict(chi2_sn=c2sn,chi2_bao=c2b,chi2_planck=c2p,M=M,Nsn=N,Nbao=Nb,ht=ht)

    def total_HEA_log(p):
        L,rd=p
        c2sn,M,N=chi2_sn(z,mu,C,DM_HEA_log,(L,),args.subsample)
        c2b,Nb=chi2_bao(bao,DM_HEA_log,DH_HEA_log,(L,),rd)
        c2p,ht=chi2_planck_theta(DM_HEA_log,(L,),rd,tmean,tsig)
        return c2sn+c2b+c2p,dict(chi2_sn=c2sn,chi2_bao=c2b,chi2_planck=c2p,M=M,Nsn=N,Nbao=Nb,ht=ht)

    bLCDM=[(66,74),(0.2,0.35),(135,160)]
    bHEAlog=[(2000,15000),(135,170)]

    best_LCDM=random_search(total_LCDM,bLCDM,trials=args.random_trials)
    best_HEA=random_search(total_HEA_log,bHEAlog,trials=args.random_trials)

    for lbl,(pars,chi2,info) in [("LCDM",best_LCDM),("HEA-log",best_HEA)]:
        print("==",lbl);print("Params:",pars);print("χ²_total:",chi2)
        print(info)

if __name__=="__main__": main()
