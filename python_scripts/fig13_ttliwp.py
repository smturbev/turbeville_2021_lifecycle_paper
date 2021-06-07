#!/usr/bin/env python
""" fig07_iwp_twp.py
    Author: Sami Turbeville
    Updated: 04 Aug 2020
    
    This script reads in the frozen water path for FV3,
    SAM, GEOS, ICON and FV3 in the TWP region. It then 
    plots the histograms compared to DARDAR. 
"""

import numpy as np
import xarray as xr
from utility import load, util
import utility.analysis_parameters as ap
import matplotlib.pyplot as plt

def get_cat_ttl_iwp(model, region):
    """ Returns tuple of (ttliwp1, ttliwp2, ttliwp3), 
        the frozen water path in the 14-18km layer for 
        each category."""
    if model=="DARDAR":
        if region=="TWP":
            diwc = xr.open_dataset(ap.DARDAR_TWP)['iwc']
            iwp = xr.open_dataset(ap.DARDAR_TWP)['iwp']
        elif region=="NAU":
            diwc = xr.open_dataset(ap.DARDAR_NAU)['iwc']
            iwp = xr.open_dataset(ap.DARDAR_NAU)['iwp']
        else:
            diwc = xr.open_dataset(ap.DARDAR_SHL)['iwc']
            iwp = xr.open_dataset(ap.DARDAR_SHL)['iwp']
        ttliwp = util.int_wrt_alt(diwc.values[:,np.argmin(abs(diwc.height.values-18000)):np.argmin(abs(diwc.height.values-14000))],
                               diwc.height.values[np.argmin(abs(diwc.height.values-18000)):np.argmin(abs(diwc.height.values-14000))])
        del diwc
        print(iwp.shape, ttliwp.shape)
        ttliwp1 = np.where((iwp>=1000),ttliwp,np.nan)
        ttliwp2 = np.where((iwp>=10),ttliwp,np.nan)
        ttliwp3 = np.where((iwp>=0.1)&(iwp<10),ttliwp,np.nan)
        del iwp
    else:
        ttliwp = load.get_ttliwp(model, region)
        iwp = load.get_iwp(model, region, ice_only=False)
        print(ttliwp.shape, iwp.shape)
        if model.lower()=="nicam":
            iwp = iwp[1::12,0,:,:]
        if model.lower()=="icon":
            iwp = iwp[1::12,:]
        elif model.lower()=="fv3":
            iwp = iwp[1::12,:,:]
        print(iwp.shape, ttliwp.shape)
        ttliwp1 = np.where((iwp>=1.0), ttliwp, np.nan)
        ttliwp2 = np.where((iwp>=1e-2)&(iwp<1.0), ttliwp, np.nan)
        ttliwp3 = np.where((iwp>=1e-4)&(iwp<1e-2), ttliwp, np.nan)
        del iwp
    print("returning ttliwp per category for ", model, region)
    return (ttliwp1, ttliwp2, ttliwp3)

def ax_ttl_cat_hist(ttliwp1, ttliwp2, ttliwp3, model, region, ax=None, fs=16):
    """Returns axis with subplot for model and region specified 
        using ttliwp (kg/m2) for each category"""
    if ax==None:
        _, ax = plt.subplots(1,1)
    if model=="DARDAR":
        hist1, xedges = np.histogram(np.log10((ttliwp1).flatten()), bins=np.arange(-3,4,0.1))
        hist2, _ = np.histogram(np.log10((ttliwp2).flatten()), bins=np.arange(-3,4,0.1))
        hist3, _ = np.histogram(np.log10((ttliwp3).flatten()), bins=np.arange(-3,4,0.1))
    else:
        hist1, xedges = np.histogram(np.log10((ttliwp1*1000).flatten()), bins=np.arange(-3,4,0.1))
        hist2, _ = np.histogram(np.log10((ttliwp2*1000).flatten()), bins=np.arange(-3,4,0.1))
        hist3, _ = np.histogram(np.log10((ttliwp3*1000).flatten()), bins=np.arange(-3,4,0.1))
    xbins = (xedges[:-1]+xedges[1:])/2
    n = len(ttliwp1.flatten())

    ax.bar(xbins[:np.argmin(abs(xbins+1))], hist1[:np.argmin(abs(xbins+1))]/n, width=0.1, color='#a6b2c5')
    ax.bar(xbins[np.argmin(abs(xbins+1)):], hist1[np.argmin(abs(xbins+1)):]/n, width=0.1, color='tab:green',
           label="1 ({}%)".format(str(np.round(np.nansum(hist1[np.argmin(abs(xbins+1)):])/n*100,0)).split(".")[0]))
    ax.bar(xbins[:np.argmin(abs(xbins+1))], hist2[:np.argmin(abs(xbins+1))]/n, width=0.1, 
           bottom=hist1[:np.argmin(abs(xbins+1))]/n, color='#baaeae')
    ax.bar(xbins[np.argmin(abs(xbins+1)):], hist2[np.argmin(abs(xbins+1)):]/n, width=0.1, 
           bottom=hist1[np.argmin(abs(xbins+1)):]/n, color='#C82F2F',
            label="2 ({}%)".format(str(np.round(np.nansum(hist2[np.argmin(abs(xbins+1)):])/n*100,0)).split(".")[0]))
    ax.bar(xbins[:np.argmin(abs(xbins+1))], hist3[:np.argmin(abs(xbins+1))]/n, width=0.1, 
           bottom=(hist2+hist1)[:np.argmin(abs(xbins+1))]/n, color='#b0bcb3')
    ax.bar(xbins[np.argmin(abs(xbins+1)):], hist3[np.argmin(abs(xbins+1)):]/n, width=0.1, 
           bottom=(hist2+hist1)[np.argmin(abs(xbins+1)):]/n, color='C0',
           label="3 ({}%)".format(str(np.round(np.nansum(hist3[np.argmin(abs(xbins+1)):])/n*100,0)).split(".")[0]))
    ax.step(xedges[1:], (hist1+hist2+hist3)/n, color='k')
    ax.tick_params(labelsize=fs-3)
    ax.set_xlabel('log$_{10}$TTL IWP (g/m$^2$)', fontsize=fs) 
    loc=1
    fs_leg=fs-4
    ax.legend(fontsize=fs_leg, frameon=False, loc=loc)
    return ax

def plot_ttliwp(region="TWP", fs=14):
    n_ttliwp1, n_ttliwp2, n_ttliwp3 = get_cat_ttl_iwp("NICAM", region)
    f_ttliwp1, f_ttliwp2, f_ttliwp3 = get_cat_ttl_iwp("FV3", region)
    i_ttliwp1, i_ttliwp2, i_ttliwp3 = get_cat_ttl_iwp("ICON", region)
    s_ttliwp1, s_ttliwp2, s_ttliwp3 = get_cat_ttl_iwp("SAM", region)
    d_ttliwp1, d_ttliwp2, d_ttliwp3 = get_cat_ttl_iwp("DARDAR", region)

    print("creating figure...")
    # plot it
    fig = plt.figure(figsize=(4,11), constrained_layout=True)
    gs = fig.add_gridspec(5,1,hspace=0.05)

    ntax= fig.add_subplot(gs[1])
    ftax= fig.add_subplot(gs[2])
    itax= fig.add_subplot(gs[3])
    stax= fig.add_subplot(gs[4])
    dtax= fig.add_subplot(gs[0])

    ntax= ax_ttl_cat_hist(n_ttliwp1, n_ttliwp2, n_ttliwp3, "NICAM", region, ax=ntax, fs=fs)
    ftax= ax_ttl_cat_hist(f_ttliwp1, f_ttliwp2, f_ttliwp3, "FV3", region, ax=ftax, fs=fs)
    itax= ax_ttl_cat_hist(i_ttliwp1, i_ttliwp2, i_ttliwp3, "ICON", region, ax=itax, fs=fs)
    stax= ax_ttl_cat_hist(s_ttliwp1, s_ttliwp2, s_ttliwp3, "SAM", region, ax=stax, fs=fs)
    dtax= ax_ttl_cat_hist(d_ttliwp1, d_ttliwp2, d_ttliwp3, "DARDAR", region, ax=dtax, fs=fs)

    ntax.set_ylim([0,0.075])
    ftax.set_ylim([0,0.075])
    itax.set_ylim([0,0.075])
    stax.set_ylim([0,0.075])
    dtax.set_ylim([0,0.075])

    ntax.set_xticklabels([])
    ftax.set_xticklabels([])
    itax.set_xticklabels([])
    dtax.set_xticklabels([])
#     ntax.set_yticklabels([])
#     ftax.set_yticklabels([])
#     itax.set_yticklabels([])
#     stax.set_yticklabels([])
#     dtax.set_yticklabels([])

    ntax.set_xlabel(None)
    ftax.set_xlabel(None)
    itax.set_xlabel(None)
    dtax.set_xlabel(None)

    ticks = np.arange(-3,4,1)
    dtax.set_xticks(ticks)
    ntax.set_xticks(ticks)
    ftax.set_xticks(ticks)
    itax.set_xticks(ticks)
    stax.set_xticks(ticks)

    dtax.set_ylabel("DARDAR\nFraction of profiles", fontsize=fs)
    ntax.set_ylabel("NICAM\nFraction of profiles", fontsize=fs)
    ftax.set_ylabel("FV3\nFraction of profiles", fontsize=fs)
    itax.set_ylabel("ICON\nFraction of profiles", fontsize=fs)
    stax.set_ylabel("SAM\nFraction of profiles", fontsize=fs)
    
    dtax.annotate("(i)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs, weight="bold")
    ntax.annotate("(ii)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs, weight="bold")
    ftax.annotate("(iii)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs, weight="bold")
    itax.annotate("(iv)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs, weight="bold")
    stax.annotate("(v)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs, weight="bold")
    dtax.set_title(region, fontsize=fs+4)

    plt.savefig("../plots/fig13_ttliwp_hist_cat_%s.png"%(region.lower()), dpi=160)
    print("saved to ../plots/fig13_ttliwp_hist_cat_%s.png"%(region.lower()))
    plt.close()

    print("Done!")

    return

if __name__=="__main__":
    plot_ttliwp(region="TWP")
    plot_ttliwp(region="SHL")
    plot_ttliwp(region="NAU")