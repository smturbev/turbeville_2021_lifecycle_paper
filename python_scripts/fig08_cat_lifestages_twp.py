#!/usr/bin/env python
""" fig08_lifestages_twp.py
    Author: Sami Turbeville
    Updated: 16 Aug 2020
    
    This script plots the median of each category on the 
    joint albedo-olr histogram as a proxy for the life 
    cycle from deep convection to anvils to thin cirrus.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import util, analysis_parameters as ap

c = ap.COLORS

ctwp = pd.read_csv("../tables/mean_CCCM_TWP.csv", index_col=0)
ntwp = pd.read_csv("../tables/mean_NICAM_TWP.csv", index_col=0)
ftwp = pd.read_csv("../tables/mean_FV3_TWP.csv", index_col=0)
itwp = pd.read_csv("../tables/mean_ICON_TWP.csv", index_col=0)
stwp = pd.read_csv("../tables/mean_SAM_TWP.csv", index_col=0)

c_olr = ctwp.olr.values[:-1] #np.array([105,195,263])
c_alb = ctwp.alb.values[:-1] #np.array([0.63,0.28,0.09])
n_olr = ntwp.olr.values[:-1]
n_alb = ntwp.alb.values[:-1]
f_olr = ftwp.olr.values[:-1]
f_alb = ftwp.alb.values[:-1]
i_olr = itwp.olr.values[:-1]
i_alb = itwp.alb.values[:-1]
s_olr = stwp.olr.values[:-1]
s_alb = stwp.alb.values[:-1]

lw = 5
ms = 20
a = 1
# Plot median
fig, ax = plt.subplots(1,1,figsize=(4.9,5), constrained_layout=True)
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax)
ax.plot(c_olr, c_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax.plot(n_olr, n_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax.plot(f_olr, f_alb, lw=lw, marker='.', ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax.plot(i_olr, i_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax.plot(s_olr, s_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
ax.legend()
ax.grid(True)

fs = 18
ax.set_ylim([0.05,0.8])
ax.set_xlim([80,310])
ax.set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax.set_ylabel('Albedo', size=fs)
ax.set_title("") # 'Category Mean Albedo and OLR\nTWP', fontsize=fs)
ax.tick_params(labelsize=fs-4)

plt.savefig('../plots/fig08_cat_lifestages_twp_mean.png',dpi=150,bbox_inches='tight')
print('    saved to ../plots/fig08_cat_lifestages_twp_mean.png')
plt.close()
