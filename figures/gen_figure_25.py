import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.constants import h, c, k

def B_nu_MJy(nu, T):
    """Planck function in MJy/sr — verified against FIRAS data"""
    x = h * nu / (k * T)
    # W/m^2/Hz/sr * 1e26 Jy/sr * 1e-6 MJy/Jy = W/m^2/Hz/sr * 1e20 MJy/sr
    return (2 * h * nu**3 / c**2) / (np.exp(x) - 1) * 1e20

firas_file = "/Users/viralsatan/Documents/Hyperspherical Cosmology Model/archive/data/hypersphere-data/firas_spectrum/firas_monopole_spec_v1.txt"
data = np.loadtxt(firas_file, comments='#')
freq_cm1 = data[:, 0]
intensity_MJy = data[:, 1]    # MJy/sr
residuals_kJy = data[:, 2]    # kJy/sr
uncertainty_kJy = data[:, 3]  # kJy/sr

freq_Hz = freq_cm1 * c * 100  # cm^-1 → Hz
T_cmb = 2.725

nu_model = np.linspace(freq_Hz.min()*0.7, freq_Hz.max()*1.2, 500)
B_model = B_nu_MJy(nu_model, T_cmb)

# Verify alignment
B_at_firas = B_nu_MJy(freq_Hz, T_cmb)
print(f"Data[0]={intensity_MJy[0]:.1f}, Model[0]={B_at_firas[0]:.1f} MJy/sr")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

ax1 = axes[0]
ax1.plot(nu_model/1e11, B_model, 'b-', linewidth=2.5, zorder=3,
         label=f'Hypersphere boundary radiation (T = {T_cmb} K)')
ax1.errorbar(freq_Hz/1e11, intensity_MJy,
             yerr=uncertainty_kJy*0.001,  # kJy → MJy
             fmt='ro', markersize=5, linewidth=1, capsize=2,
             label='COBE/FIRAS data (Fixsen et al. 1996)', zorder=4, alpha=0.9)
ax1.set_xlabel('Frequency (10¹¹ Hz)', fontsize=12)
ax1.set_ylabel('Intensity (MJy sr⁻¹)', fontsize=12)
ax1.set_title('CMB Monopole Spectrum', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(0, max(freq_Hz/1e11)*1.1)
ax1.set_ylim(0, max(intensity_MJy)*1.15)
ax1.text(0.03, 0.55, 'Boundary thermalization (τ_b > 30)\nproduces exact Planck spectrum\nfrom internal consistency',
         transform=ax1.transAxes, fontsize=9, color='navy',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

ax2 = axes[1]
ax2.errorbar(freq_Hz/1e11, residuals_kJy, yerr=uncertainty_kJy,
             fmt='ro', markersize=4, linewidth=1, capsize=2, zorder=3,
             label='FIRAS residuals (data − 2.725K BB) ± 1σ')
ax2.axhline(0, color='b', linewidth=2, label='Hypersphere prediction', zorder=2)
ax2.fill_between(freq_Hz/1e11, -uncertainty_kJy, uncertainty_kJy,
                  alpha=0.15, color='red')
ax2.set_xlabel('Frequency (10¹¹ Hz)', fontsize=12)
ax2.set_ylabel('Residual (kJy sr⁻¹)', fontsize=12)
ax2.set_title('Residuals: FIRAS − T = 2.725 K Blackbody', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.text(0.05, 0.92, f'FIRAS: |Δρ/ρ| < 10⁻⁵\nConsistent with exact\nPlanck spectrum prediction',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('Figure 25: CMB Blackbody Spectrum\n'
             'S³/B⁴ boundary radiation with τ_b > 30 thermalization predicts exact Planck form',
             fontsize=11, fontweight='bold')
plt.tight_layout()
output = "/Users/viralsatan/Documents/Hyperspherical Cosmology Model/archive/data/hypersphere-data/figure_25_bb_spectrum.png"
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output}")
