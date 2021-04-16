#!/usr/bin/env python
""" fig_nicam_iwc_cat_profiles.py
    author: sami turbeville
    date modified: 21 Dec 2020
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utility import load, util, analysis_parameters as ap

MODEL = "NICAM"

# iwp_shl = load.get_iwp(MODEL, "SHL").values[11::12]
# t = load.get_temp(MODEL, "SHL")
# qv = load.get_qv(MODEL, "SHL")
# print("--------shl---------\n... calculating Tv...")
# Tv = (1 + 0.61*qv)*t
# del t, qv
# print("... calculating rho...")
# p = load.get_pres(MODEL, "SHL")
# rho = p / (287*Tv) # p[:,:,np.newaxis,np.newaxis] for sam
# del p
# print("... opening cloud ice kg/kg...")
# qi = xr.open_dataset(ap.SHL_NICAM_QI).ms_qi
# print("... calculating iwc kg/m3...")
# iwc = qi * rho
# del qi
# print("... get iwc categories...")
# iwc1_shl = iwc.where(iwp_shl>=1).mean(axis=(0,2,3))
# iwc2_shl = iwc.where((iwp_shl<1)&(iwp_shl>=1e-2)).mean(axis=(0,2,3))
# iwc3_shl = iwc.where((iwp_shl<1e-2)&(iwp_shl>=1e-4)).mean(axis=(0,2,3))
# del iwc
# print("... done. Getting swc...")
# qs = xr.open_dataset(ap.SHL_NICAM_QS).ms_qs
# swc = qs * rho
# del qs
# print("... get swc categories...")
# swc1_shl = swc.where(iwp_shl>=1).mean(axis=(0,2,3))
# swc2_shl = swc.where((iwp_shl<1)&(iwp_shl>=1e-2)).mean(axis=(0,2,3))
# swc3_shl = swc.where((iwp_shl<1e-2)&(iwp_shl>=1e-4)).mean(axis=(0,2,3))
# del swc
# print("... done. Getting gwc...")
# qg = xr.open_dataset(ap.SHL_NICAM_QG).ms_qg
# gwc = qg * rho
# del qg
# gwc1_shl = gwc.where(iwp_shl>=1).mean(axis=(0,2,3))
# gwc2_shl = gwc.where((iwp_shl<1)&(iwp_shl>=1e-2)).mean(axis=(0,2,3))
# gwc3_shl = gwc.where((iwp_shl<1e-2)&(iwp_shl>=1e-4)).mean(axis=(0,2,3))
# del gwc
# print("... categorize total water content ...")
# twc = xr.open_dataarray(ap.NICAM+"NICAM_twc_SHL.nc")[16:]
# twc1_shl = twc.where(iwp_shl>=1).mean(axis=(0,2,3))
# twc2_shl = twc.where((iwp_shl<1)&(iwp_shl>=1e-2)).mean(axis=(0,2,3))
# twc3_shl = twc.where((iwp_shl<1e-2)&(iwp_shl>=1e-4)).mean(axis=(0,2,3))
# del twc


# iwp_nau = load.get_iwp(MODEL, "NAU").values[11::12]
# t = load.get_temp(MODEL, "NAU")
# qv = load.get_qv(MODEL, "NAU")
# print("--------nau---------\n... calculating Tv...")
# Tv = (1 + 0.61*qv)*t
# del t, qv
# print("... calculating rho...")
# p = load.get_pres(MODEL, "NAU")
# rho = p / (287*Tv) # p[:,:,np.newaxis,np.newaxis] for sam
# del p
# print("... opening cloud ice kg/kg...")
# qi = xr.open_dataset(ap.NAU_NICAM_QI).ms_qi
# print("... calculating iwc kg/m3...")
# iwc = qi * rho
# del qi
# print("... get iwc categories...")
# iwc1_nau = iwc.where(iwp_nau>=1).mean(axis=(0,2,3))
# iwc2_nau = iwc.where((iwp_nau<1)&(iwp_nau>=1e-2)).mean(axis=(0,2,3))
# iwc3_nau = iwc.where((iwp_nau<1e-2)&(iwp_nau>=1e-4)).mean(axis=(0,2,3))
# del iwc
# print("... done. Getting swc...")
# qs = xr.open_dataset(ap.NAU_NICAM_QS).ms_qs
# swc = qs * rho
# del qs
# print("... get swc categories...")
# swc1_nau = swc.where(iwp_nau>=1).mean(axis=(0,2,3))
# swc2_nau = swc.where((iwp_nau<1)&(iwp_nau>=1e-2)).mean(axis=(0,2,3))
# swc3_nau = swc.where((iwp_nau<1e-2)&(iwp_nau>=1e-4)).mean(axis=(0,2,3))
# del swc
# print("... done. Getting gwc...")
# qg = xr.open_dataset(ap.NAU_NICAM_QG).ms_qg
# gwc = qg * rho
# del qg
# gwc1_nau = gwc.where(iwp_nau>=1).mean(axis=(0,2,3))
# gwc2_nau = gwc.where((iwp_nau<1)&(iwp_nau>=1e-2)).mean(axis=(0,2,3))
# gwc3_nau = gwc.where((iwp_nau<1e-2)&(iwp_nau>=1e-4)).mean(axis=(0,2,3))
# del gwc
# print("... categorize total water content ...")
# twc = xr.open_dataarray(ap.NICAM+"NICAM_twc_NAU.nc")[16:]
# twc1_nau = twc.where(iwp_nau>=1).mean(axis=(0,2,3))
# twc2_nau = twc.where((iwp_nau<1)&(iwp_nau>=1e-2)).mean(axis=(0,2,3))
# twc3_nau = twc.where((iwp_nau<1e-2)&(iwp_nau>=1e-4)).mean(axis=(0,2,3))
# del twc


iwp = load.get_iwp(MODEL, "TWP").values[11::12]
# load one at a time to get category mean
print("... categorize ice ...")
iwc = xr.open_dataarray(ap.NICAM+"NICAM_iwc_TWP.nc")[16:]
print(iwp.shape, iwc.shape)
iwc1 = iwc.where(iwp>=1).mean(axis=(0,2,3))
iwc2 = iwc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
iwc3 = iwc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del iwc
print("... categorize snow ...")
swc = xr.open_dataarray(ap.NICAM+"NICAM_swc_TWP.nc")[16:]
swc1 = swc.where(iwp>=1).mean(axis=(0,2,3))
swc2 = swc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
swc3 = swc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del swc
print("... categorize graupel ...")
gwc = xr.open_dataarray(ap.NICAM+"NICAM_gwc_TWP.nc")[16:]
gwc1 = gwc.where(iwp>=1).mean(axis=(0,2,3))
gwc2 = gwc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
gwc3 = gwc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del gwc
print("... categorize total water content ...")
twc = xr.open_dataarray(ap.NICAM+"NICAM_twc_TWP.nc")[16:]
twc1 = twc.where(iwp>=1).mean(axis=(0,2,3))
twc2 = twc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
twc3 = twc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del twc
print("... done.")

z = load.get_levels(MODEL, "TWP")
# z_shl = load.get_levels(MODEL, "SHL")
# z_nau = load.get_levels(MODEL, "NAU")

print("... getting dardar...")
dd_ds = xr.open_dataset(ap.DARDAR_TWP)
dd_iwc = dd_ds.iwc
dd_iwp = dd_ds.iwp
dd_z = dd_ds.height

dd_ind5 = np.argmin(abs(dd_z.values-5000))
print(dd_ind5)
del dd_ds

print("... categorize dardar...")
dd_iwc1 = dd_iwc.where(dd_iwp>=1000).mean(axis=0)
dd_iwc2 = dd_iwc.where((dd_iwp>=10)&(dd_iwp<1000)).mean(axis=0)
dd_iwc3 = dd_iwc.where((dd_iwp>=0.1)&(dd_iwp<10)).mean(axis=0)
print("... done.\nStarting figure...")

fig, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(15,8), sharey=True)
f=1000
fs = 20
a = 0.4
lw = 3
ax1.plot(dd_iwc1[:dd_ind5], (dd_z/f)[:dd_ind5], 'blueviolet', lw=lw)
ax2.plot(dd_iwc2[:dd_ind5], (dd_z/f)[:dd_ind5], 'blueviolet', lw=lw)
ax3.plot(dd_iwc3[:dd_ind5], (dd_z/f)[:dd_ind5], 'blueviolet', lw=lw)
ax1.plot(f*iwc1, z/f, 'g', lw=lw)
ax2.plot(f*iwc2, z/f, 'r', lw=lw)
ax3.plot(f*iwc3, z/f, 'b', lw=lw)
ax1.plot(f*swc1, z/f, 'g--', lw=lw)
ax2.plot(f*swc2, z/f, 'r--', lw=lw)
ax3.plot(f*swc3, z/f, 'b--', lw=lw)
ax1.plot(f*gwc1, z/f, 'g-.', lw=lw)
ax2.plot(f*gwc2, z/f, 'r-.', lw=lw)
ax3.plot(f*gwc3, z/f, 'b-.', lw=lw)
ax1.plot(f*twc1, z/f, 'k', lw=lw)
ax2.plot(f*twc2, z/f, 'k', lw=lw)
ax3.plot(f*twc3, z/f, 'k', lw=lw)
# ax1.plot(f*iwc1_shl, z_shl/f, 'g', alpha=a)
# ax2.plot(f*iwc2_shl, z_shl/f, 'r', alpha=a)
# ax3.plot(f*iwc3_shl, z_shl/f, 'b', alpha=a)
# ax1.plot(f*swc1_shl, z_shl/f, 'g--', alpha=a)
# ax2.plot(f*swc2_shl, z_shl/f, 'r--', alpha=a)
# ax3.plot(f*swc3_shl, z_shl/f, 'b--', alpha=a)
# ax1.plot(f*gwc1_shl, z_shl/f, 'g-.', alpha=a)
# ax2.plot(f*gwc2_shl, z_shl/f, 'r-.', alpha=a)
# ax3.plot(f*gwc3_shl, z_shl/f, 'b-.', alpha=a)
# ax1.plot(f*twc1_shl, z_shl/f, 'k', alpha=a)
# ax2.plot(f*twc2_shl, z_shl/f, 'k', alpha=a)
# ax3.plot(f*twc3_shl, z_shl/f, 'k', alpha=a)
# ax1.plot(f*iwc1_nau, z_nau/f, 'g', alpha=a, lw=lw-1)
# ax2.plot(f*iwc2_nau, z_nau/f, 'r', alpha=a, lw=lw-1)
# ax3.plot(f*iwc3_nau, z_nau/f, 'b', alpha=a, lw=lw-1)
# ax1.plot(f*swc1_nau, z_nau/f, 'g--', alpha=a, lw=lw-1)
# ax2.plot(f*swc2_nau, z_nau/f, 'r--', alpha=a, lw=lw-1)
# ax3.plot(f*swc3_nau, z_nau/f, 'b--', alpha=a, lw=lw-1)
# ax1.plot(f*gwc1_nau, z_nau/f, 'g-.', alpha=a, lw=lw-1)
# ax2.plot(f*gwc2_nau, z_nau/f, 'r-.', alpha=a, lw=lw-1)
# ax3.plot(f*gwc3_nau, z_nau/f, 'b-.', alpha=a, lw=lw-1)
# ax1.plot(f*twc1_nau, z_nau/f, 'k', alpha=a, lw=lw-1)
# ax2.plot(f*twc2_nau, z_nau/f, 'k', alpha=a, lw=lw-1)
# ax3.plot(f*twc3_nau, z_nau/f, 'k', alpha=a, lw=lw-1)
# ax1.plot([0,0], [0,0], 'g', label="Ice")
# ax1.plot([0,0], [0,0], 'g--', label="Snow")
# ax1.plot([0,0], [0,0], 'g-.', label="Graupel")
# ax1.plot([0,0], [0,0], 'k', label="Total water")
# ax2.plot([0,0], [0,0], 'r', label="Ice")
# ax2.plot([0,0], [0,0], 'r--', label="Snow")
# ax2.plot([0,0], [0,0], 'r-.', label="Graupel")
# ax2.plot([0,0], [0,0], 'k', label="Total water")
ax3.plot([0,0], [0,0], color='k', alpha=0.7, lw=lw, label="Ice")
ax3.plot([0,0], [0,0], 'k--', alpha=0.7, lw=lw, label="Snow")
ax3.plot([0,0], [0,0], 'k-.', alpha=0.7, lw=lw, label="Graupel")
ax3.plot([0,0], [0,0], 'k', lw=lw, label="Total water")
# ax3.plot([0,0], [0,0], 'k', lw=lw, label="TWP")
# ax3.plot([0,0], [0,0], 'k', lw=lw-1, alpha=a, label="NAU")
# ax3.plot([0,0], [0,0], 'k', alpha=a, label="SHL")
ax3.plot([0,0], [0,0], 'blueviolet', lw=lw, label="DARDAR")
ax1.set_xlabel("Water content (g/m$^3$)", fontsize=fs-2)
ax2.set_xlabel("Water content (g/m$^3$)", fontsize=fs-2)
ax3.set_xlabel("Water content (g/m$^3$)", fontsize=fs-2)
ax1.set_ylabel("Height (km)", fontsize=fs-2)
ax1.tick_params(labelsize=fs-4)
ax2.tick_params(labelsize=fs-4)
ax3.tick_params(labelsize=fs-4)
ax3.set_xticks([0,0.005,0.01])
ax1.set_ylim([0,20])
ax2.set_ylim([0,20])
ax3.set_ylim([0,20])
# ax1.legend(loc=1, fontsize=fs-4)
# ax2.legend(loc=1, fontsize=fs-4)
ax3.legend(loc=7, fontsize=fs-4,)
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax1.set_title("Deep Convection\nFWP $\geq$ 1000 g/m$^2$", fontsize=fs)
ax2.set_title("Anvils\n10 $\leq$ FWP < 1000 g/m$^2$", fontsize=fs)
ax3.set_title("Thin Cirrus\n0.1 $\leq$ FWP < 10 g/m$^2$", fontsize=fs)

ax1.annotate("(a)", xy=(0.05,0.95), xycoords="axes fraction", fontsize=fs-2, weight="bold")
ax2.annotate("(b)", xy=(0.05,0.95), xycoords="axes fraction", fontsize=fs-2, weight="bold")
ax3.annotate("(c)", xy=(0.05,0.95), xycoords="axes fraction", fontsize=fs-2, weight="bold")

plt.savefig("../plots/fig02_nicam_cat_iwc_twp_dardar.png", dpi=150, bbox_inches="tight")
print("... saved as ../plots/fig02_nicam_cat_iwc_twp_dardar.png")
plt.show()
