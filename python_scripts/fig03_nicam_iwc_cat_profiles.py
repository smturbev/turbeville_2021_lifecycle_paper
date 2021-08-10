#!/usr/bin/env python
""" fig03_nicam_iwc_cat_profiles.py
    author: sami turbeville
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utility import load, util, analysis_parameters as ap

MODEL = "NICAM"

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
print("... done.\nStarting figure...")

fig, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(15,8), sharey=True)
f=1000
fs = 20
a = 0.4
lw = 3

ax1.plot(f*iwc1, z/f, 'g', lw=lw)
ax2.plot(f*iwc2, z/f, 'r', lw=lw)
ax3.plot(f*iwc3, z/f, 'b', lw=lw)
ax1.plot(f*swc1, z/f, 'g--', lw=lw)
ax2.plot(f*swc2, z/f, 'r--', lw=lw)
ax3.plot(f*swc3, z/f, 'b--', lw=lw)
ax1.plot(f*gwc1, z/f, 'g-.', lw=lw)
ax2.plot(f*gwc2, z/f, 'r-.', lw=lw)
ax3.plot(f*gwc3, z/f, 'b-.', lw=lw)
ax1.plot(f*twc1, z/f, 'k', lw=lw-2)
ax2.plot(f*twc2, z/f, 'k', lw=lw-2)
ax3.plot(f*twc3, z/f, 'k', lw=lw-2)
ax3.plot([0,0], [0,0], color='k', alpha=0.5, lw=lw, label="Ice")
ax3.plot([0,0], [0,0], 'k--', alpha=0.5, lw=lw, label="Snow")
ax3.plot([0,0], [0,0], 'k-.', alpha=0.5, lw=lw, label="Graupel")
ax3.plot([0,0], [0,0], 'k', lw=lw-2, label="Total water")

ax1.set_xlabel("Water content (g m$^{-3}$)", fontsize=fs-2)
ax2.set_xlabel("Water content (g m$^{-3}$)", fontsize=fs-2)
ax3.set_xlabel("Water content (g m$^{-3}$)", fontsize=fs-2)
ax1.set_ylabel("Height (km)", fontsize=fs-2)
ax1.tick_params(labelsize=fs-4)
ax2.tick_params(labelsize=fs-4)
ax3.tick_params(labelsize=fs-4)
ax3.set_xticks([0,0.005,0.01])
ax1.set_yticks(list(np.arange(2,23,4)), minor=False)
ax1.set_ylim([0,20])
ax2.set_ylim([0,20])
ax3.set_ylim([0,20])
ax3.legend(loc=7, fontsize=fs-4,)
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax1.set_title("Deep Convection\nFWP $\geq$ 1000 g m$^{-2}$", fontsize=fs)
ax2.set_title("Thick Cirrus\n10 $\leq$ FWP < 1000 g m$^{-2}$", fontsize=fs)
ax3.set_title("Thin Cirrus\n0.1 $\leq$ FWP < 10 g m$^{-2}$", fontsize=fs)

ax1.annotate("(a)", xy=(-0.1,1.1), xycoords="axes fraction", fontsize=fs, weight="bold")
ax2.annotate("(b)", xy=(-0.1,1.1), xycoords="axes fraction", fontsize=fs, weight="bold")
ax3.annotate("(c)", xy=(-0.1,1.1), xycoords="axes fraction", fontsize=fs, weight="bold")

plt.savefig("../plots/fig03_nicam_cat_iwc_twp_dardar.png", dpi=150, bbox_inches="tight")
print("... saved as ../plots/fig03_nicam_cat_iwc_twp_dardar.png")
plt.show()
