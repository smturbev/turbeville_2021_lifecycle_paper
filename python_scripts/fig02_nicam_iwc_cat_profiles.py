#!/usr/bin/env python
""" fig_nicam_iwc_cat_profiles.py
    author: sami turbeville
    date modified: 21 Dec 2020
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utility import load, util, analysis_parameters as ap


REGION = "TWP"
MODEL = "NICAM"

iwp = load.get_iwp(MODEL, REGION).values[11::12]
# load one at a time to get category mean
print("... categorize ice ...")
iwc = xr.open_dataarray(ap.NICAM+"NICAM_iwc_TWP.nc")
print(iwp.shape, iwc.shape)
iwc1 = iwc.where(iwp>=1).mean(axis=(0,2,3))
iwc2 = iwc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
iwc3 = iwc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del iwc
print("... categorize snow ...")
swc = xr.open_dataarray(ap.NICAM+"NICAM_swc_TWP.nc") 
swc1 = swc.where(iwp>=1).mean(axis=(0,2,3))
swc2 = swc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
swc3 = swc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del swc
print("... categorize graupel ...")
gwc = xr.open_dataarray(ap.NICAM+"NICAM_gwc_TWP.nc") 
gwc1 = gwc.where(iwp>=1).mean(axis=(0,2,3))
gwc2 = gwc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
gwc3 = gwc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del gwc
print("... categorize total water content ...")
twc = xr.open_dataarray(ap.NICAM+"NICAM_twc_TWP.nc") 
twc1 = twc.where(iwp>=1).mean(axis=(0,2,3))
twc2 = twc.where((iwp<1)&(iwp>=1e-2)).mean(axis=(0,2,3))
twc3 = twc.where((iwp<1e-2)&(iwp>=1e-4)).mean(axis=(0,2,3))
del twc
print("... done.")

z = load.get_levels(MODEL, REGION)

fig, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(15,8), sharey=True)
f=1000
fs = 20
ax1.plot(f*iwc1, z/f, 'g')
ax2.plot(f*iwc2, z/f, 'r')
ax3.plot(f*iwc3, z/f, 'b')
ax1.plot(f*swc1, z/f, 'g--')
ax2.plot(f*swc2, z/f, 'r--')
ax3.plot(f*swc3, z/f, 'b--')
ax1.plot(f*gwc1, z/f, 'g-.')
ax2.plot(f*gwc2, z/f, 'r-.')
ax3.plot(f*gwc3, z/f, 'b-.')
ax1.plot(f*twc1, z/f, 'k')
ax2.plot(f*twc2, z/f, 'k')
ax3.plot(f*twc3, z/f, 'k')
ax1.plot([0,0], [0,0], 'g', label="Ice")
ax1.plot([0,0], [0,0], 'g--', label="Snow")
ax1.plot([0,0], [0,0], 'g-.', label="Graupel")
ax1.plot([0,0], [0,0], 'k', label="Total water")
ax2.plot([0,0], [0,0], 'r', label="Ice")
ax2.plot([0,0], [0,0], 'r--', label="Snow")
ax2.plot([0,0], [0,0], 'r-.', label="Graupel")
ax2.plot([0,0], [0,0], 'k', label="Total water")
ax3.plot([0,0], [0,0], 'b', label="Ice")
ax3.plot([0,0], [0,0], 'b--', label="Snow")
ax3.plot([0,0], [0,0], 'b-.', label="Graupel")
ax3.plot([0,0], [0,0], 'k', label="Total water")
ax1.set_xlabel("IWC (g/m$^3$)", fontsize=fs-2)
ax2.set_xlabel("IWC (g/m$^3$)", fontsize=fs-2)
ax3.set_xlabel("IWC (g/m$^3$)", fontsize=fs-2)
ax1.set_ylabel("Height (km)", fontsize=fs-2)
ax1.set_ylim([0,20])
ax2.set_ylim([0,20])
ax3.set_ylim([0,20])
ax1.legend(loc=1, fontsize=fs-4)
ax2.legend(loc=1, fontsize=fs-4)
ax3.legend(loc=1, fontsize=fs-4)
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
ax1.set_title("Deep Convection\nIWP > 1000 g/m$2$", fontsize=fs)
ax2.set_title("Anvils\n10 < IWP < 1000 g/m$2$", fontsize=fs)
ax3.set_title("Thin Cirrus\n0.1 < IWP < 10 g/m$2$", fontsize=fs)

ax1.annotate("(a)", xy=(0.05,0.95), xycoords="axes fraction", fontsize=fs-2)
ax2.annotate("(b)", xy=(0.05,0.95), xycoords="axes fraction", fontsize=fs-2)
ax3.annotate("(c)", xy=(0.05,0.95), xycoords="axes fraction", fontsize=fs-2)

plt.savefig("../plots/fig02_nicam_cat_iwc_{}.png".format(REGION.lower()), dpi=150, bbox_inches="tight")
plt.show()