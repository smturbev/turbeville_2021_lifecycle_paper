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
ttl = False

ctwp = pd.read_csv("../tables/CCCM_TWP.csv", index_col=0)
ntwp = pd.read_csv("../tables/NICAM_TWP.csv", index_col=0)
ftwp = pd.read_csv("../tables/FV3_TWP.csv", index_col=0)
itwp = pd.read_csv("../tables/ICON_TWP.csv", index_col=0)
stwp = pd.read_csv("../tables/SAM_TWP.csv", index_col=0)

c_med_olr = ctwp.OLR.values[:-2] #np.array([105,195,263])
c_med_alb = ctwp.ALB.values[:-2] #np.array([0.63,0.28,0.09])
n_med_olr = ntwp.OLR.values[:-2]
n_med_alb = ntwp.ALB.values[:-2]
f_med_olr = ftwp.OLR.values[:-2]
f_med_alb = ftwp.ALB.values[:-2]
i_med_olr = itwp.OLR.values[:-2]
i_med_alb = itwp.ALB.values[:-2]
s_med_olr = stwp.OLR.values[:-2]
s_med_alb = stwp.ALB.values[:-2]

c_tmed_olr = ctwp.OLR_TTL.values[:-2] #np.array([104,157,240])
c_tmed_alb = ctwp.ALB_TTL.values[:-2] #np.array([0.6598,0.3444,0.1049661])
n_tmed_olr = ntwp.OLR_TTL.values[:-2]
n_tmed_alb = ntwp.ALB_TTL.values[:-2]
f_tmed_olr = ftwp.OLR_TTL.values[:-2]
f_tmed_alb = ftwp.ALB_TTL.values[:-2]
i_tmed_olr = itwp.OLR_TTL.values[:-2]
i_tmed_alb = itwp.ALB_TTL.values[:-2]
s_tmed_olr = stwp.OLR_TTL.values[:-2]
s_tmed_alb = stwp.ALB_TTL.values[:-2]

lw = 5
ms = 20
a = 1
# Plot median
fig, ax = plt.subplots(1,1,figsize=(4.7,5), constrained_layout=True)
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax)
ax.plot(c_med_olr, c_med_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax.plot(n_med_olr, n_med_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax.plot(f_med_olr, f_med_alb, lw=lw, marker='.', ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax.plot(i_med_olr, i_med_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax.plot(s_med_olr, s_med_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
if ttl:
    ax.plot(c_tmed_olr, c_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax.plot(n_tmed_olr, n_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax.plot(f_tmed_olr, f_tmed_alb, lw=lw-2, marker='.', ms=ms+6, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax.plot(i_tmed_olr, i_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax.plot(s_tmed_olr, s_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax.plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax.plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
    h,l = ax.get_legend_handles_labels()
    leg1 = ax.legend(h[1:-2],l[1:-2], loc=1)
    leg2 = ax.legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
    ax.add_artist(leg1)
else:
    ax.legend()
ax.grid(True)

fs = 18
ax.set_ylim([0.05,0.8])
ax.set_xlim([80,310])
ax.set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax.set_ylabel('Albedo', size=fs)
ax.set_title('Category Median Albedo and OLR\nTWP', fontsize=fs)
ax.tick_params(labelsize=fs-4)

plt.savefig('../plots/fig08_cat_lifestages_twp.png',dpi=150,bbox_inches='tight')
print('    saved to ../plots/fig08_cat_lifestages_twp.png')
plt.close()
