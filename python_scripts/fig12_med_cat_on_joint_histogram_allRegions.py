#!/usr/bin/env python
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
ttl = True
regions = ["SHL","TWP","NAU"]
models = ["CCCM","NICAM","FV3","ICON","SAM"]
net=False

dar  = pd.read_csv("../plots/DARDAR_cat_alb_olr_ttl_cs.csv")
ntwp = pd.read_csv("../plots/NICAM_TWP_cat_alb_olr_ttl_cs.csv")
ftwp = pd.read_csv("../plots/FV3_TWP_cat_alb_olr_ttl_cs.csv")
itwp = pd.read_csv("../plots/ICON_TWP_cat_alb_olr_ttl_cs.csv")
stwp = pd.read_csv("../plots/SAM_TWP_cat_alb_olr_ttl_cs.csv")
nshl = pd.read_csv("../plots/NICAM_SHL_cat_alb_olr_ttl_cs.csv")
fshl = pd.read_csv("../plots/FV3_SHL_cat_alb_olr_ttl_cs.csv")
ishl = pd.read_csv("../plots/ICON_SHL_cat_alb_olr_ttl_cs.csv")
sshl = pd.read_csv("../plots/SAM_SHL_cat_alb_olr_ttl_cs.csv")
nnau = pd.read_csv("../plots/NICAM_NAU_cat_alb_olr_ttl_cs.csv")
fnau = pd.read_csv("../plots/FV3_NAU_cat_alb_olr_ttl_cs.csv")
inau = pd.read_csv("../plots/ICON_NAU_cat_alb_olr_ttl_cs.csv")
snau = pd.read_csv("../plots/SAM_NAU_cat_alb_olr_ttl_cs.csv")
cre_twp = pd.read_csv("../plots/isolated_ttl_cirrus_iceonly_noliq_percentiles_TWP.csv")
cre_shl = pd.read_csv("../plots/isolated_ttl_cirrus_iceonly_noliq_percentiles_SHL.csv")
cre_nau = pd.read_csv("../plots/isolated_ttl_cirrus_iceonly_noliq_percentiles_NAU.csv")

tc_med_olr = dar.twp_olr_med.values[:-1]
tc_med_alb = dar.twp_alb_med.values[:-1]
tn_med_olr = ntwp.olr_med.values[:-1]
tn_med_alb = ntwp.alb_med.values[:-1]
tf_med_olr = ftwp.olr_med.values[:-1]
tf_med_alb = ftwp.alb_med.values[:-1]
ti_med_olr = itwp.olr_med.values[:-1]
ti_med_alb = itwp.alb_med.values[:-1]
ts_med_olr = stwp.olr_med.values[:-1]
ts_med_alb = stwp.alb_med.values[:-1]

tc_tmed_olr = dar.twp_olr_med.values[1:]
tc_tmed_alb = dar.twp_alb_med.values[1:]
tn_tmed_olr = ntwp.olr_ttl.values
tn_tmed_alb = ntwp.alb_ttl.values
tf_tmed_olr = ftwp.olr_ttl.values
tf_tmed_alb = ftwp.alb_ttl.values
ti_tmed_olr = itwp.olr_ttl.values
ti_tmed_alb = itwp.alb_ttl.values
ts_tmed_olr = stwp.olr_ttl.values
ts_tmed_alb = stwp.alb_ttl.values

sc_med_olr = dar.shl_olr_med.values[:-1]
sc_med_alb = dar.shl_alb_med.values[:-1]
sn_med_olr = nshl.olr_med.values[:-1]
sn_med_alb = nshl.alb_med.values[:-1]
sf_med_olr = fshl.olr_med.values[:-1]
sf_med_alb = fshl.alb_med.values[:-1]
si_med_olr = ishl.olr_med.values[:-1]
si_med_alb = ishl.alb_med.values[:-1]
ss_med_olr = sshl.olr_med.values[:-1]
ss_med_alb = sshl.alb_med.values[:-1]

sc_tmed_olr = dar.shl_olr_med.values[1:]
sc_tmed_alb = dar.shl_alb_med.values[1:]
sn_tmed_olr = nshl.olr_ttl.values
sn_tmed_alb = nshl.alb_ttl.values
sf_tmed_olr = fshl.olr_ttl.values
sf_tmed_alb = fshl.alb_ttl.values
si_tmed_olr = ishl.olr_ttl.values
si_tmed_alb = ishl.alb_ttl.values
ss_tmed_olr = sshl.olr_ttl.values
ss_tmed_alb = sshl.alb_ttl.values

nc_med_olr = dar.nau_olr_med.values[:-1]
nc_med_alb = dar.nau_alb_med.values[:-1]
nn_med_olr = nnau.olr_med.values[:-1]
nn_med_alb = nnau.alb_med.values[:-1]
nf_med_olr = fnau.olr_med.values[:-1]
nf_med_alb = fnau.alb_med.values[:-1]
ni_med_olr = inau.olr_med.values[:-1]
ni_med_alb = inau.alb_med.values[:-1]
ns_med_olr = snau.olr_med.values[:-1]
ns_med_alb = snau.alb_med.values[:-1]

nc_tmed_olr = dar.nau_olr_med.values[1:]
nc_tmed_alb = dar.nau_alb_med.values[1:]
nn_tmed_olr = nnau.olr_ttl.values
nn_tmed_alb = nnau.alb_ttl.values
nf_tmed_olr = fnau.olr_ttl.values
nf_tmed_alb = fnau.alb_ttl.values
ni_tmed_olr = inau.olr_ttl.values
ni_tmed_alb = inau.alb_ttl.values
ns_tmed_olr = snau.olr_ttl.values
ns_tmed_alb = snau.alb_ttl.values

lw = 5
ms = 20
a = 1

# Plot median
fig, ax = plt.subplots(1,4,figsize=(20,5.5), constrained_layout=True, sharey=False)

util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="SHL",\
                 colorbar_on=False, ax=ax[0])
ax[0].plot(sc_med_olr, sc_med_alb, lw=lw, marker='.',ms=ms, label="CCCM", alpha=a, zorder=10, color=c["OBS"])
ax[0].plot(sn_med_olr, sn_med_alb, lw=lw, marker='.',ms=ms, label="NICAM", alpha=a, zorder=10, color=c["NICAM"])
ax[0].plot(sf_med_olr, sf_med_alb, lw=lw, marker='.', ms=ms, label="FV3", alpha=a, zorder=12, color=c["FV3"])
ax[0].plot(si_med_olr, si_med_alb, lw=lw, marker='.',ms=ms, label="ICON", alpha=a, zorder=13, color=c["ICON"])
ax[0].plot(ss_med_olr, ss_med_alb, lw=lw, marker='.',ms=ms, label="SAM", alpha=a, zorder=14, color=c["SAM"])
if ttl:
    ax[0].plot(sc_tmed_olr, sc_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax[0].plot(sn_tmed_olr, sn_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax[0].plot(sf_tmed_olr, sf_tmed_alb, lw=lw-2, marker='.', ms=ms+4, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax[0].plot(si_tmed_olr, si_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax[0].plot(ss_tmed_olr, ss_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax[0].plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax[0].plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
h,l = ax[0].get_legend_handles_labels()
leg1 = ax[0].legend(h[1:-2],l[1:-2], loc=1)
leg2 = ax[0].legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
ax[0].add_artist(leg1)
ax[0].grid(True)

fs = 18
ax[0].set_ylim([0.05,0.8])
ax[0].set_xlim([80,310])
ax[0].set_xlabel('OLR(W m$^{-2}$)', size=fs)
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
    ax[1].plot(tc_tmed_olr, tc_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax[1].plot(tn_tmed_olr, tn_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax[1].plot(tf_tmed_olr, tf_tmed_alb, lw=lw-2, marker='.', ms=ms+4, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax[1].plot(ti_tmed_olr, ti_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax[1].plot(ts_tmed_olr, ts_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax[1].plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax[1].plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
h,l = ax[1].get_legend_handles_labels()
leg1 = ax[1].legend(h[1:-2],l[1:-2], loc=1)
leg2 = ax[1].legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
ax[1].add_artist(leg1)
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
    ax[2].plot(nc_tmed_olr, nc_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["OBS"], linestyle='--', fillstyle='none')
    ax[2].plot(nn_tmed_olr, nn_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=10, 
            color=c["NICAM"], linestyle='--', fillstyle='none')
    ax[2].plot(nf_tmed_olr, nf_tmed_alb, lw=lw-2, marker='.', ms=ms+4, alpha=a, zorder=12, 
            color=c["FV3"], linestyle='--', fillstyle='none')
    ax[2].plot(ni_tmed_olr, ni_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=13, 
            color=c["ICON"], linestyle='--', fillstyle='none')
    ax[2].plot(ns_tmed_olr, ns_tmed_alb, lw=lw-2, marker='.',ms=ms+4, alpha=a, zorder=14, 
            color=c["SAM"], linestyle='--', fillstyle='none')
    ax[2].plot(0,0,color='gray',marker='.',ms=ms,lw=lw,label='All-Sky')
    ax[2].plot(0,0,color='gray',marker='.',ms=ms+4,lw=lw-2,linestyle='--',fillstyle='none',label='TTL-Ci')
h,l = ax[2].get_legend_handles_labels()
leg1 = ax[2].legend(h[1:-2],l[1:-2], loc=1)
leg2 = ax[2].legend([h[0]]+h[-2:],[l[0]]+l[-2:],loc=9)
ax[2].add_artist(leg1)
ax[2].grid(True)

fs = 18
ax[2].set_ylim([0.05,0.8])
ax[2].set_xlim([80,310])
ax[2].set_xlabel('OLR(W m$^{-2}$)', size=fs)
ax[2].set_ylabel('Albedo', size=fs)
ax[2].set_title('Category Median Albedo and OLR\nNAU', fontsize=fs)
ax[2].tick_params(labelsize=fs-4)

fs = 18
ax[1].set_ylim([0.05,0.8])
ax[1].set_xlim([80,310])
ax[1].set_xlabel('OLR(W m$^{-2}$)', size=fs)
ax[1].set_ylabel('Albedo', size=fs)
ax[1].set_title('Category Median Albedo and OLR\nTWP', fontsize=fs)
ax[1].tick_params(labelsize=fs-4)

ax[3].set_xticks([-0.3,0.7,1.7,2.7], minor=True)
ax[3].set_xticks([0.25,1.25,2.25])
ax[3].plot([-0.3,2.7],[0,0],'gray',lw=0.5)
for i,r in enumerate(regions):
    print(i,r)
    if r=="TWP":
        lwcre = (cre_twp.olrcs.values[1:] - cre_twp.olr_med.values[1:])
        swcre = (cre_twp.albcs.values[1:] - cre_twp.alb_med.values[1:])*413.2335274
        netcre = lwcre + swcre
    elif r=="NAU":
        lwcre = (cre_nau.olrcs.values[1:] - cre_nau.olr_med.values[1:]) 
        swcre = (cre_nau.albcs.values[1:] - cre_nau.alb_med.values[1:])*413.2335274
        netcre = lwcre + swcre
    else:
        lwcre = (cre_shl.olrcs.values[1:] - cre_shl.olr_med.values[1:])
        swcre = (cre_shl.albcs.values[1:] - cre_shl.alb_med.values[1:])*435.2760211
        netcre = lwcre + swcre
    for j,m in enumerate(models):
        if m=="CCCM":
            m = "OBS"
            print("OBS", [netcre])
        if net:
            ax[3].plot([i+0.1*j],[netcre[j]], color=c[m], 
                   marker='s', ms=fs/2) #markerfacecolor='none'
        else:
            if (m=="OBS") & (r=="TWP"):
                lablw = "LW CRE"
                labsw = "SW CRE"
                labnet = "Net CRE"
            else:
                lablw, labsw, labnet = None, None, None
            if j==0:
                ax[3].scatter([i+0.1*j],[lwcre[j]/6], c=c[m], 
                       marker='s', s=fs*4, label=lablw) #markerfacecolor='none'
                ax[3].scatter([i+0.1*j],[swcre[j]/6], edgecolor=c[m], label=labsw,
                       marker='s', c='none', s=fs*4)
                ax[3].scatter([i+0.1*j],[netcre[j]/6], c=c[m], 
                       marker='o', s=fs, label=labnet)
            else:
                ax[3].scatter([i+0.1*j],[lwcre[j]], c=c[m], 
                       marker='s', s=fs*4, label=lablw) #markerfacecolor='none'
                ax[3].scatter([i+0.1*j],[swcre[j]], edgecolor=c[m], label=labsw,
                       marker='s', c='none', s=fs*4)
                ax[3].scatter([i+0.1*j],[netcre[j]], c=c[m], 
                       marker='o', s=fs, label=labnet)
ax[3].set_xlim([-0.3,2.7])
ax[3].set_xticklabels(['SHL', 'TWP', 'NAU'])
ax[3].grid(which='minor')
if net:
    ax[3].set_title('Isolated TTL Cirrus Net CRE', fontsize=fs)
else:
    ax[3].set_title('Isolated TTL Cirrus CRE', fontsize=fs)
ax[3].set_ylabel('CRE [W/m$^2$]', fontsize=fs)
if not(net):
    ax[3].legend()
ax[3].tick_params(axis='y', labelsize=fs-4)
ax[3].tick_params(axis='x', labelsize=fs)

ax[0].set_ylim([0.01,0.85])
ax[1].set_ylim([0.01,0.85])
ax[2].set_ylim([0.01,0.85])

ax[1].set_ylabel(None)
ax[2].set_ylabel(None)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])

ax[0].annotate("(a)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs)
ax[1].annotate("(b)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs)
ax[2].annotate("(c)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs)
ax[3].annotate("(d)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs)

plt.savefig('../plots/fig11_median_category_alb_olr_allmodels_stn.png',
            dpi=150,bbox_inches='tight')
print('    saved to ../plots/fig11_median_category_alb_olr_allmodels_stn.png')
plt.close()
