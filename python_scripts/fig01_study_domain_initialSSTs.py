#!/usr/bin/env python
""" fig01_study_domain_initialSSTs.py
    author: Sami Turbeville
    date modified: 16 Feb 2021
    
    Script for Figure 5 in Turbeville et al 2021
    Plot of domains of DYAMOND output in tropics
    with initial SST from DYAMOND runs (from ECMWF reanalysis)
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs

var = xr.open_dataset('/home/disk/eos15/smturbev/dyamond/latlon_sst_sic.nc').var34[0,:,:]

fig = plt.figure(figsize=(14,3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()
cs = var.plot.contourf(x='lon',levels=np.arange(290,305,1), extend='both',cmap='Blues_r', add_colorbar=False)
ax.plot([-3,7,7,-3,-3],[9,9,19,19,9], 'k--', lw=2)
ax.plot([143,153,153,143,143],[5,5,-5,-5,5], 'k', lw=2)
ax.plot([163,173,173,163,163],[5,5,-5,-5,5], 'k--', lw=2)
fs=16
ax.annotate("(a)", xy=(-2.8,15), xycoords="data", fontsize=fs)
ax.annotate("(b)", xy=(143.2,1), xycoords="data", fontsize=fs)
ax.annotate("(c)", xy=(163.2,1), xycoords="data", fontsize=fs)

ax.set_ylim([-20,20])
# plt.xlim([100,180])
ax.set_xlim([-5,180])
# plt.xticks(np.arange(100,181,10))
ax.set_xticks(ticks=np.arange(-10,181,10), minor=False)
ax.set_yticks(ticks=np.arange(-20,21,5), minor=False)
ax.set_xlabel('Longitude', fontsize=fs-2)
ax.set_ylabel('Latitude', fontsize=fs-2)
cbar = plt.colorbar(cs, shrink=0.8, ax=ax, pad=0.01)
cbar.set_label('Temperature (K)', fontsize=fs-2)
cbar.ax.tick_params(labelsize=fs-2)
plt.tick_params(labelsize=fs-2)
ax.set_title("")
plt.savefig('../plots/fig01_study_domains_mono.png',dpi=200, bbox_inches="tight")
print("saved as ../plots/fig01_study_domains_mono.png")
plt.close()
