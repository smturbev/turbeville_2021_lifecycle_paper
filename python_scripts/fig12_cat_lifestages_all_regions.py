#!/usr/bin/env/python
""" med_cat_on_joint_histogram.py
    Author: Sami Turbeville
    Updated: 16 Aug 2020
    
    This script plots the median of each category on the 
    joint albedo-olr histogram.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import util, analysis_parameters as ap

c = ap.COLORS
ttl = False
regions = ["SHL","TWP","NAU"]
models = ["CCCM","NICAM","FV3","ICON","SAM"]
net=False

# dar  = pd.read_csv("../tables/DARDAR_cat_alb_olr_ttl_cs.csv")
ctwp = pd.read_csv("../tables/CCCM_TWP.csv", index_col=0)
ntwp = pd.read_csv("../tables/NICAM_TWP.csv", index_col=0)
ftwp = pd.read_csv("../tables/FV3_TWP.csv", index_col=0)
itwp = pd.read_csv("../tables/ICON_TWP.csv", index_col=0)
stwp = pd.read_csv("../tables/SAM_TWP.csv", index_col=0)
cshl = pd.read_csv("../tables/CCCM_SHL.csv", index_col=0)
nshl = pd.read_csv("../tables/NICAM_SHL.csv", index_col=0)
fshl = pd.read_csv("../tables/FV3_SHL.csv", index_col=0)
ishl = pd.read_csv("../tables/ICON_SHL.csv", index_col=0)
sshl = pd.read_csv("../tables/SAM_SHL.csv", index_col=0)
cnau = pd.read_csv("../tables/CCCM_NAU.csv", index_col=0)
nnau = pd.read_csv("../tables/NICAM_NAU.csv", index_col=0)
fnau = pd.read_csv("../tables/FV3_NAU.csv", index_col=0)
inau = pd.read_csv("../tables/ICON_NAU.csv", index_col=0)
snau = pd.read_csv("../tables/SAM_NAU.csv", index_col=0)

# cre_twp = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_TWP.csv")
# cre_shl = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_SHL.csv")
# cre_nau = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_NAU.csv")

tc_med_olr = ctwp.OLR.values[:-2] #np.array([105,195,263])
tc_med_alb = ctwp.ALB.values[:-2] #np.array([0.63,0.28,0.09])
tn_med_olr = ntwp.OLR.values[:-2]
tn_med_alb = ntwp.ALB.values[:-2]
tf_med_olr = ftwp.OLR.values[:-2]
tf_med_alb = ftwp.ALB.values[:-2]
ti_med_olr = itwp.OLR.values[:-2]
ti_med_alb = itwp.ALB.values[:-2]
ts_med_olr = stwp.OLR.values[:-2]
ts_med_alb = stwp.ALB.values[:-2]

tc_tmed_olr = ctwp.OLR_TTL.values[:-2] #np.array([104,157,240])
tc_tmed_alb = ctwp.ALB_TTL.values[:-2] #np.array([0.6598,0.3444,0.1049661])
tn_tmed_olr = ntwp.OLR_TTL.values[:-2]
tn_tmed_alb = ntwp.ALB_TTL.values[:-2]
tf_tmed_olr = ftwp.OLR_TTL.values[:-2]
tf_tmed_alb = ftwp.ALB_TTL.values[:-2]
ti_tmed_olr = itwp.OLR_TTL.values[:-2]
ti_tmed_alb = itwp.ALB_TTL.values[:-2]
ts_tmed_olr = stwp.OLR_TTL.values[:-2]
ts_tmed_alb = stwp.ALB_TTL.values[:-2]

sc_med_olr = cshl.OLR.values[:-2] #np.array([105,195,263])
sc_med_alb = cshl.ALB.values[:-2] #np.array([0.63,0.28,0.09])
sn_med_olr = nshl.OLR.values[:-2]
sn_med_alb = nshl.ALB.values[:-2]
sf_med_olr = fshl.OLR.values[:-2]
sf_med_alb = fshl.ALB.values[:-2]
si_med_olr = ishl.OLR.values[:-2]
si_med_alb = ishl.ALB.values[:-2]
ss_med_olr = sshl.OLR.values[:-2]
ss_med_alb = sshl.ALB.values[:-2]

sc_tmed_olr = cshl.OLR_TTL.values[:-2] #np.array([104,157,240])
sc_tmed_alb = cshl.ALB_TTL.values[:-2] #np.array([0.6598,0.3444,0.1049661])
sn_tmed_olr = nshl.OLR_TTL.values[:-2]
sn_tmed_alb = nshl.ALB_TTL.values[:-2]
sf_tmed_olr = fshl.OLR_TTL.values[:-2]
sf_tmed_alb = fshl.ALB_TTL.values[:-2]
si_tmed_olr = ishl.OLR_TTL.values[:-2]
si_tmed_alb = ishl.ALB_TTL.values[:-2]
ss_tmed_olr = sshl.OLR_TTL.values[:-2]
ss_tmed_alb = sshl.ALB_TTL.values[:-2]

nc_med_olr = cnau.OLR.values[:-2] #np.array([105,195,263])
nc_med_alb = cnau.ALB.values[:-2] #np.array([0.63,0.28,0.09])
nn_med_olr = nnau.OLR.values[:-2]
nn_med_alb = nnau.ALB.values[:-2]
nf_med_olr = fnau.OLR.values[:-2]
nf_med_alb = fnau.ALB.values[:-2]
ni_med_olr = inau.OLR.values[:-2]
ni_med_alb = inau.ALB.values[:-2]
ns_med_olr = snau.OLR.values[:-2]
ns_med_alb = snau.ALB.values[:-2]

nc_tmed_olr = cnau.OLR_TTL.values[:-2] #np.array([104,157,240])
nc_tmed_alb = cnau.ALB_TTL.values[:-2] #np.array([0.6598,0.3444,0.1049661])
nn_tmed_olr = nnau.OLR_TTL.values[:-2]
nn_tmed_alb = nnau.ALB_TTL.values[:-2]
nf_tmed_olr = fnau.OLR_TTL.values[:-2]
nf_tmed_alb = fnau.ALB_TTL.values[:-2]
ni_tmed_olr = inau.OLR_TTL.values[:-2]
ni_tmed_alb = inau.ALB_TTL.values[:-2]
ns_tmed_olr = snau.OLR_TTL.values[:-2]
ns_tmed_alb = snau.ALB_TTL.values[:-2]

lw = 5
ms = 20
a = 1

# Plot median
fig, ax = plt.subplots(1,3,figsize=(15,5.5), constrained_layout=True, sharey=False)

util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="SHL",\
                 colorbar_on=False, ax=ax[0])
ax[0].plot(sc_med_olr, sc_med_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax[0].plot(sn_med_olr, sn_med_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax[0].plot(sf_med_olr, sf_med_alb, lw=lw, marker='.', ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax[0].plot(si_med_olr, si_med_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax[0].plot(ss_med_olr, ss_med_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
if ttl:
    ax[0].plot(sc_tmed_olr, sc_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax[0].plot(sn_tmed_olr, sn_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax[0].plot(sf_tmed_olr, sf_tmed_alb, lw=lw-2, marker='.', ms=ms+6, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax[0].plot(si_tmed_olr, si_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax[0].plot(ss_tmed_olr, ss_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax[0].plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax[0].plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
    h,l = ax[0].get_legend_handles_labels()
    leg1 = ax[0].legend(h[1:-2],l[1:-2], loc=1)
    leg2 = ax[0].legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
    ax[0].add_artist(leg1)
else:
    ax[0].legend()
ax[0].grid(True)

fs = 18
ax[0].set_ylim([0.05,0.8])
ax[0].set_xlim([80,310])
ax[0].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[0].set_ylabel('Albedo', size=fs)
ax[0].set_title('Category Median Albedo and OLR\nSHL', fontsize=fs)
ax[0].tick_params(labelsize=fs-4)
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax[1])
ax[1].plot(tc_med_olr, tc_med_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax[1].plot(tn_med_olr, tn_med_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax[1].plot(tf_med_olr, tf_med_alb, lw=lw, marker='.', ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax[1].plot(ti_med_olr, ti_med_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax[1].plot(ts_med_olr, ts_med_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
if ttl:
    ax[1].plot(tc_tmed_olr, tc_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax[1].plot(tn_tmed_olr, tn_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax[1].plot(tf_tmed_olr, tf_tmed_alb, lw=lw-2, marker='.', ms=ms+6, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax[1].plot(ti_tmed_olr, ti_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax[1].plot(ts_tmed_olr, ts_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax[1].plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax[1].plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
    h,l = ax[1].get_legend_handles_labels()
    leg1 = ax[1].legend(h[1:-2],l[1:-2], loc=1)
    leg2 = ax[1].legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
    ax[1].add_artist(leg1)
else:
    ax[1].legend()
ax[1].grid(True)

util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="NAU",\
                 colorbar_on=False, ax=ax[2])
ax[2].plot(nc_med_olr, nc_med_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax[2].plot(nn_med_olr, nn_med_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax[2].plot(nf_med_olr, nf_med_alb, lw=lw, marker='.',ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax[2].plot(ni_med_olr, ni_med_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax[2].plot(ns_med_olr, ns_med_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
if ttl:
    ax[2].plot(nc_tmed_olr, nc_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax[2].plot(nn_tmed_olr, nn_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax[2].plot(nf_tmed_olr, nf_tmed_alb, lw=lw-2, marker='.', ms=ms+6, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax[2].plot(ni_tmed_olr, ni_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax[2].plot(ns_tmed_olr, ns_tmed_alb, lw=lw-2, marker='.',ms=ms+6, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax[2].plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax[2].plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
    h,l = ax[2].get_legend_handles_labels()
    leg1 = ax[2].legend(h[1:-2],l[1:-2], loc=1)
    leg2 = ax[2].legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
    ax[2].add_artist(leg1)
else: 
    ax[2].legend()

ax[2].grid(True)

fs = 18
ax[2].set_ylim([0.05,0.8])
ax[2].set_xlim([80,310])
ax[2].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[2].set_ylabel('Albedo', size=fs)
ax[2].set_title('Category Median Albedo and OLR\nNAU', fontsize=fs)
ax[2].tick_params(labelsize=fs-4)

fs = 18
ax[1].set_ylim([0.05,0.8])
ax[1].set_xlim([80,310])
ax[1].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[1].set_ylabel('Albedo', size=fs)
ax[1].set_title('Category Median Albedo and OLR\nTWP', fontsize=fs)
ax[1].tick_params(labelsize=fs-4)


ax[0].set_ylim([0.01,0.85])
ax[1].set_ylim([0.01,0.85])
ax[2].set_ylim([0.01,0.85])

ax[1].set_ylabel(None)
ax[2].set_ylabel(None)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])

ax[0].annotate("(a)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs+4, weight="bold")
ax[1].annotate("(b)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs+4, weight="bold")
ax[2].annotate("(c)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs+4, weight="bold")

plt.savefig('../plots/fig12_cat_lifestages_all_regions.png',
            dpi=150,bbox_inches='tight')
print('    saved to ../plots/fig12_cat_lifestages_all_regions.png')
plt.close()
