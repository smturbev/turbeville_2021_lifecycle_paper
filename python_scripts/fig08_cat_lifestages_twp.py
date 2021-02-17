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
ttl = True

ntwp = pd.read_csv("../tables/NICAM_TWP_cat_alb_olr_ttl_cs.csv")
ftwp = pd.read_csv("../tables/FV3_TWP_cat_alb_olr_ttl_cs.csv")
itwp = pd.read_csv("../tables/ICON_TWP_cat_alb_olr_ttl_cs.csv")
stwp = pd.read_csv("../tables/SAM_TWP_cat_alb_olr_ttl_cs.csv")
cre_twp = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_TWP.csv")

c_med_olr = np.array([105,195,263])
c_med_alb = np.array([0.63,0.28,0.09])
n_med_olr = ntwp.olr_med.values[:-1]
n_med_alb = ntwp.alb_med.values[:-1]
f_med_olr = ftwp.olr_med.values[:-1]
f_med_alb = ftwp.alb_med.values[:-1]
i_med_olr = itwp.olr_med.values[:-1]
i_med_alb = itwp.alb_med.values[:-1]
s_med_olr = stwp.olr_med.values[:-1]
s_med_alb = stwp.alb_med.values[:-1]

c_tmed_olr = np.array([104,157,240])
c_tmed_alb = np.array([0.6598,0.3444,0.1049661])
n_tmed_olr = ntwp.olr_ttl.values
n_tmed_alb = ntwp.alb_ttl.values
f_tmed_olr = ftwp.olr_ttl.values
f_tmed_alb = ftwp.alb_ttl.values
i_tmed_olr = itwp.olr_ttl.values
i_tmed_alb = itwp.alb_ttl.values
s_tmed_olr = stwp.olr_ttl.values
s_tmed_alb = stwp.alb_ttl.values

lw = 5
ms = 20
a = 1
# Plot median
fig, [ax, ax1] = plt.subplots(1,2,figsize=(11,5), constrained_layout=True)
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax)
ax.plot(c_med_olr, c_med_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax.plot(n_med_olr, n_med_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax.plot(f_med_olr, f_med_alb, lw=lw, marker='.', ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax.plot(i_med_olr, i_med_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax.plot(s_med_olr, s_med_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
if ttl:
    ax.plot(c_tmed_olr, c_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax.plot(n_tmed_olr, n_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax.plot(f_tmed_olr, f_tmed_alb, lw=lw-2, marker='.', ms=ms+4, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax.plot(i_tmed_olr, i_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax.plot(s_tmed_olr, s_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax.plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax.plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
h,l = ax.get_legend_handles_labels()
leg1 = ax.legend(h[1:-2],l[1:-2], loc=1)
leg2 = ax.legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
ax.add_artist(leg1)
ax.grid(True)

fs = 18
ax.set_ylim([0.05,0.8])
ax.set_xlim([80,310])
ax.set_xlabel('OLR(W m$^{-2}$)', size=fs)
ax.set_ylabel('Albedo', size=fs)
ax.set_title('Category Median Albedo and OLR\nTWP', fontsize=fs)
ax.tick_params(labelsize=fs-4)

models=["CCCM","NICAM","FV3","ICON","SAM"]
lwcre = (cre_twp.olrcs.values[1:] - cre_twp.olr_med.values[1:])
swcre = (cre_twp.albcs.values[1:] - cre_twp.alb_med.values[1:])*413.2335274
isocre = lwcre + swcre
for j,m in enumerate(models):
        if m=="CCCM":
            m = "OBS"
            lab = "CCCM"
        else:
            lab = m
        ax1.scatter([2.3+0.1*j],[isocre[j]], c=c[m], 
                   marker='o', s=fs*4, label=lab) #markerfacecolor='none'
        ax1.scatter([0.3+0.1*j],[lwcre[j]], c=c[m], 
               marker='o', s=fs*4) #markerfacecolor='none'
        ax1.scatter([1.3+0.1*j],[swcre[j]], c=c[m],
               marker='o', s=fs*4)
ax1.set_xlim([0,3])
ax1.set_xticks([0,1,2,3], minor=True)
ax1.set_xticks([0.5,1.5,2.5])
ax1.set_xticklabels(['LW', 'SW', 'Net'])
ax1.grid(which='minor')

ax1.set_title('Isolated TTL Cirrus CRE\nTWP', fontsize=fs)
ax1.set_ylabel('CRE [W/m$^2$]', fontsize=fs)
ax1.tick_params(axis='y', labelsize=fs-4)
ax1.tick_params(axis='x', labelsize=fs)
ax1.set_ylim([-75,75])
ax1.axhline(y=0, color='gray', alpha=0.6)
ax1.legend()

ax.annotate("(a)", xy=(0.001,0.95), xycoords="axes fraction", fontsize=fs-2)
ax1.annotate("(b)", xy=(0.001,0.95), xycoords="axes fraction", fontsize=fs-2)

plt.subplots_adjust(wspace=0.3)
plt.savefig('../plots/fig08_cat_lifestages_twp.png',dpi=150,bbox_inches='tight')
print('    saved to ../plots/fig08_cat_lifestages_twp.png')
plt.close()
