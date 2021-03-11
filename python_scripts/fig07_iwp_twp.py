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

# load all iwp 
REGION = "TWP"
fs = 14

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
        fig, ax = plt.subplots(1,1)
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
           label="CAT 1 ({}%)".format(str(np.round(np.nansum(hist1[np.argmin(abs(xbins+1)):])/n*100,0)).split(".")[0]))
    ax.bar(xbins[:np.argmin(abs(xbins+1))], hist2[:np.argmin(abs(xbins+1))]/n, width=0.1, 
           bottom=hist1[:np.argmin(abs(xbins+1))]/n, color='#baaeae')
    ax.bar(xbins[np.argmin(abs(xbins+1)):], hist2[np.argmin(abs(xbins+1)):]/n, width=0.1, 
           bottom=hist1[np.argmin(abs(xbins+1)):]/n, color='#C82F2F',
            label="CAT 2 ({}%)".format(str(np.round(np.nansum(hist2[np.argmin(abs(xbins+1)):])/n*100,0)).split(".")[0]))
    ax.bar(xbins[:np.argmin(abs(xbins+1))], hist3[:np.argmin(abs(xbins+1))]/n, width=0.1, 
           bottom=(hist2+hist1)[:np.argmin(abs(xbins+1))]/n, color='#b0bcb3')
    ax.bar(xbins[np.argmin(abs(xbins+1)):], hist3[np.argmin(abs(xbins+1)):]/n, width=0.1, 
           bottom=(hist2+hist1)[np.argmin(abs(xbins+1)):]/n, color='C0',
           label="CAT 3 ({}%)".format(str(np.round(np.nansum(hist3[np.argmin(abs(xbins+1)):])/n*100,0)).split(".")[0]))
    ax.step(xedges[1:], (hist1+hist2+hist3)/n, color='k')
    ax.tick_params(labelsize=fs-3)
#     ax.set_title(model+" TTL IWP, "+region+"\n"+str(np.nansum(hist3[:np.argmin(abs(xbins+1))]))+" profiles", size=16)
    ax.set_xlabel('log10(TTL IWP) [g/m2]', fontsize=fs)
#     ax.set_ylabel(model+'\nFraction of Profiles')
    ax.legend(fontsize=fs-6, frameon=False, loc=1)
    return ax

def main():
    n_ttliwp1, n_ttliwp2, n_ttliwp3 = get_cat_ttl_iwp("NICAM", REGION)
    f_ttliwp1, f_ttliwp2, f_ttliwp3 = get_cat_ttl_iwp("FV3", REGION)
    i_ttliwp1, i_ttliwp2, i_ttliwp3 = get_cat_ttl_iwp("ICON", REGION)
    s_ttliwp1, s_ttliwp2, s_ttliwp3 = get_cat_ttl_iwp("SAM", REGION)
    d_ttliwp1, d_ttliwp2, d_ttliwp3 = get_cat_ttl_iwp("DARDAR", REGION)

    print("Getting %s for frozen hydrometeors"%(REGION))
    # giwp = util.iwp_wrt_pres("GEOS", REGION, hydro_type=hydro_type, geos_graupel=True)
    print("geos zeros until we get new data")
    iiwp = load.get_iwp("ICON", REGION, ice_only=False).values
    print("loaded icon")
    fiwp = load.get_iwp("FV3", REGION, ice_only=False).values
    print("loaded fv3")
    niwp = load.get_iwp("NICAM", REGION, ice_only=False).values
    print("loaded nicam")
    siwp = load.get_iwp("SAM", REGION, ice_only=False)
    sno = siwp.count().values
    siwp = siwp.values
    print("loaded sam... convert to g/m2")
    # convert to g/m2
    niwp = niwp * 1000
    fiwp = fiwp * 1000
    iiwp = iiwp * 1000
    siwp = siwp * 1000
    print("    done... get obs")
    # get observational data
    # ciwp = xr.open_dataset(ap.CERES_TWP)["iwp MODIS"]
    if REGION=="TWP":
        ciwp = xr.open_dataset(ap.DARDAR_TWP)["iwp"]
    elif REGION=="SHL":
        ciwp = xr.open_dataset(ap.DARDAR_SHL)["iwp"]
    else:
        ciwp = xr.open_dataset(ap.DARDAR_NAU)["iwp"]
    print("    done... bin by iwp")
    # calculate histograms
    bins = np.arange(-3,4,0.1)
    chist, _ = np.histogram(np.log10(ciwp), bins=bins)
    ihist, xedges = np.histogram(np.log10(iiwp.flatten()), bins=bins)
    fhist, _ = np.histogram(np.log10(fiwp.flatten()), bins=bins)
    shist, _ = np.histogram(np.log10(siwp.flatten()), bins=bins)
    nhist, _ = np.histogram(np.log10(niwp.flatten()), bins=bins)
    print("get num of prof")
    # get number of profiles
    cno = (len(ciwp.values.flatten()))
    ino = (len(iiwp.flatten()))
    sno = (len(siwp.flatten()))
    fno = (len(fiwp.flatten()))
    nno = (len(niwp.flatten()))

    print("counting % in each category...")
    # color the bins by category
    # clear sky threshold = 0.01 g/m2
    thres=0.1
    cs_bin_ind = np.argmin(abs(np.log10(thres)-bins))
    # thin cirrus to anvil threshold = 10 g/m2
    lower_bin_ind = np.argmin(abs(1-bins))
    # convective threshold = 1000 g/m2
    upper_bin_ind = np.argmin(abs(3-bins))
    # sum count in each category and divide by total
    ccat3 = np.nansum(chist[cs_bin_ind:lower_bin_ind])/cno *100
    ccat2 = np.nansum(chist[lower_bin_ind:upper_bin_ind])/cno *100
    ccat1 = np.nansum(chist[upper_bin_ind:])/cno *100
    ccat0 = 100. - (ccat1+ccat2+ccat3)
    ncat3 = np.nansum(nhist[cs_bin_ind:lower_bin_ind])/nno *100
    ncat2 = np.nansum(nhist[lower_bin_ind:upper_bin_ind])/nno *100
    ncat1 = np.nansum(nhist[upper_bin_ind:])/nno *100
    ncat0 = 100. - (ncat1+ncat2+ncat3)
    fcat3 = np.nansum(fhist[cs_bin_ind:lower_bin_ind])/fno *100
    fcat2 = np.nansum(fhist[lower_bin_ind:upper_bin_ind])/fno *100
    fcat1 = np.nansum(fhist[upper_bin_ind:])/fno *100
    fcat0 = 100. - (fcat1+fcat2+fcat3)
    icat3 = np.nansum(ihist[cs_bin_ind:lower_bin_ind])/ino *100
    icat2 = np.nansum(ihist[lower_bin_ind:upper_bin_ind])/ino *100
    icat1 = np.nansum(ihist[upper_bin_ind:])/ino *100
    icat0 = 100. - (icat1+icat2+icat3)
    scat3 = np.nansum(shist[cs_bin_ind:lower_bin_ind])/sno *100
    scat2 = np.nansum(shist[lower_bin_ind:upper_bin_ind])/sno *100
    scat1 = np.nansum(shist[upper_bin_ind:])/sno *100
    scat0 = 100. - (scat1+scat2+scat3)

    print("normalize:")
    # normalize
    chist = chist/cno
    ihist = ihist/ino
    shist = shist/sno
    fhist = fhist/fno
    nhist = nhist/nno


    print("creating figure...")
    # plot it
    fig = plt.figure(figsize=(8,12), constrained_layout=True)
    gs = fig.add_gridspec(5,2,hspace=0.05)
    xmid = (xedges[1:]+xedges[:-1])/2

    ntax= fig.add_subplot(gs[1,1])
    ftax= fig.add_subplot(gs[2,1])
    itax= fig.add_subplot(gs[3,1])
    stax= fig.add_subplot(gs[4,1])
    dtax= fig.add_subplot(gs[0,1])

    ntax= ax_ttl_cat_hist(n_ttliwp1, n_ttliwp2, n_ttliwp3, "NICAM", REGION, ax=ntax, fs=fs)
    ftax= ax_ttl_cat_hist(f_ttliwp1, f_ttliwp2, f_ttliwp3, "FV3", REGION, ax=ftax, fs=fs)
    itax= ax_ttl_cat_hist(i_ttliwp1, i_ttliwp2, i_ttliwp3, "ICON", REGION, ax=itax, fs=fs)
    stax= ax_ttl_cat_hist(s_ttliwp1, s_ttliwp2, s_ttliwp3, "SAM", REGION, ax=stax, fs=fs)
    dtax= ax_ttl_cat_hist(d_ttliwp1, d_ttliwp2, d_ttliwp3, "DARDAR", REGION, ax=dtax, fs=fs)

    ntax.set_ylim([0,0.075])
    ftax.set_ylim([0,0.075])
    itax.set_ylim([0,0.075])
    stax.set_ylim([0,0.075])
    dtax.set_ylim([0,0.075])

    ntax.set_xticklabels([])
    ftax.set_xticklabels([])
    itax.set_xticklabels([])
    dtax.set_xticklabels([])
    ntax.set_yticklabels([])
    ftax.set_yticklabels([])
    itax.set_yticklabels([])
    stax.set_yticklabels([])
    dtax.set_yticklabels([])

    ntax.set_xlabel(None)
    ftax.set_xlabel(None)
    itax.set_xlabel(None)
    dtax.set_xlabel(None)

    cax = fig.add_subplot(gs[0,0])
    fax = fig.add_subplot(gs[2,0])
    iax = fig.add_subplot(gs[3,0])
    sax = fig.add_subplot(gs[4,0])
    nax = fig.add_subplot(gs[1,0])

    cax.bar(xmid[:cs_bin_ind], chist[:cs_bin_ind], color='lightgray', width=0.1)
    cax.bar(xmid[cs_bin_ind:lower_bin_ind], chist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    cax.bar(xmid[lower_bin_ind:upper_bin_ind], chist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    cax.bar(xmid[upper_bin_ind:], chist[upper_bin_ind:], color='tab:green', width=0.1)
    # cax.step(xedges[1:], chist, color='k')
    cax.set_ylim([0,0.075])
    cax.annotate(str(np.round(ccat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate(str(np.round(ccat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate(str(np.round(ccat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate(str(np.round(ccat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("CAT 1", xy=(3.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("CAT 2", xy=(1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("CAT 3", xy=(-0.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("Cirrus-free", xy=(-2.4,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.set_title("Total Column FWP, {}".format(REGION), size=fs+4)
    cax.set_xticklabels([])
    cax.set_ylabel("DARDAR\nFraction of profiles", fontsize=fs)

    fax.bar(xmid[:cs_bin_ind], fhist[:cs_bin_ind], color='lightgray', width=0.1)
    fax.bar(xmid[cs_bin_ind:lower_bin_ind], fhist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    fax.bar(xmid[lower_bin_ind:upper_bin_ind], fhist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    fax.bar(xmid[upper_bin_ind:], fhist[upper_bin_ind:], color='tab:green', width=0.1)
    # fax.step(xedges[1:], chist, color='k')
    fax.set_ylim([0,0.075])
    fax.annotate(str(np.round(fcat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate(str(np.round(fcat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate(str(np.round(fcat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate(str(np.round(fcat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("CAT 1", xy=(3.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("CAT 2", xy=(1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("CAT 3", xy=(-0.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("Cirrus-free", xy=(-2.4,0.062), xycoords="data", color="k", fontsize=fs-2)
    # fax.set_title("FV3 FWP, "+REGION+"\n {:} profiles".format(fno), size=16)
    fax.set_xticklabels([])
    fax.set_ylabel("FV3\nFraction of profiles", fontsize=fs)

    iax.bar(xmid[:cs_bin_ind], ihist[:cs_bin_ind], color='lightgray', width=0.1)
    iax.bar(xmid[cs_bin_ind:lower_bin_ind], ihist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    iax.bar(xmid[lower_bin_ind:upper_bin_ind], ihist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    iax.bar(xmid[upper_bin_ind:], ihist[upper_bin_ind:], color='tab:green', width=0.1)
    # iax.step(xedges[1:], chist, color='k')
    iax.set_ylim([0,0.075])
    iax.annotate(str(np.round(icat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate(str(np.round(icat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate(str(np.round(icat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate(str(np.round(icat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("CAT 1", xy=(3.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("CAT 2", xy=(1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("CAT 3", xy=(-0.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("Cirrus-free", xy=(-2.4,0.062), xycoords="data", color="k", fontsize=fs-2)
    # iax.set_title("ICON FWP, "+REGION+"\n {:} profiles".format(ino), size=16)
    iax.set_xticklabels([])
    iax.set_ylabel("ICON\nFraction of profiles", fontsize=fs)

    sax.bar(xmid[:cs_bin_ind], shist[:cs_bin_ind], color='lightgray', width=0.1)
    sax.bar(xmid[cs_bin_ind:lower_bin_ind], shist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    sax.bar(xmid[lower_bin_ind:upper_bin_ind], shist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    sax.bar(xmid[upper_bin_ind:], shist[upper_bin_ind:], color='tab:green', width=0.1)
    # sax.step(xedges[1:], chist, color='k')
    sax.set_ylim([0,0.075])
    sax.annotate(str(np.round(scat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate(str(np.round(scat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate(str(np.round(scat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate(str(np.round(scat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("CAT 1", xy=(3.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("CAT 2", xy=(1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("CAT 3", xy=(-0.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("Cirrus-free", xy=(-2.4,0.062), xycoords="data", color="k", fontsize=fs-2)
    # sax.set_title("SAM FWP, "+REGION+"\n {:} profiles".format(sno), size=16)
    sax.set_xlabel("log10(FWP) [g/m$^2$]", fontsize=fs)
    sax.set_ylabel("SAM\nFraction of profiles", fontsize=fs)

    nax.bar(xmid[:cs_bin_ind], nhist[:cs_bin_ind], color='lightgray', width=0.1)
    nax.bar(xmid[cs_bin_ind:lower_bin_ind], nhist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    nax.bar(xmid[lower_bin_ind:upper_bin_ind], nhist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    nax.bar(xmid[upper_bin_ind:], nhist[upper_bin_ind:], color='tab:green', width=0.1)
    # nax.step(xedges[1:], chist, color='k')
    nax.set_ylim([0,0.075])
    nax.annotate(str(np.round(ncat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate(str(np.round(ncat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate(str(np.round(ncat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate(str(np.round(ncat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("CAT 1", xy=(3.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("CAT 2", xy=(1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("CAT 3", xy=(-0.1,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("Cirrus-free", xy=(-2.4,0.062), xycoords="data", color="k", fontsize=fs-2)
    # nax.set_title("NICAM FWP, "+REGION+"\n {:} profiles".format(nno), size=16)
    nax.set_xticklabels([])
    nax.set_ylabel("NICAM\nFraction of profiles", fontsize=fs)

    ticks = np.arange(-3,4,1)
    cax.set_xticks(ticks)
    nax.set_xticks(ticks)
    fax.set_xticks(ticks)
    iax.set_xticks(ticks)
    sax.set_xticks(ticks)
    dtax.set_xticks(ticks)
    ntax.set_xticks(ticks)
    ftax.set_xticks(ticks)
    itax.set_xticks(ticks)
    stax.set_xticks(ticks)
    cax.tick_params(labelsize=fs-3)
    nax.tick_params(labelsize=fs-3)
    fax.tick_params(labelsize=fs-3)
    iax.tick_params(labelsize=fs-3)
    sax.tick_params(labelsize=fs-3)

    cax.annotate("(a)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    nax.annotate("(b)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    fax.annotate("(c)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    iax.annotate("(d)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    sax.annotate("(e)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    dtax.annotate("(f)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    ntax.annotate("(g)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    ftax.annotate("(h)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    itax.annotate("(i)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    stax.annotate("(j)", xy=(-3.1,0.067), xycoords="data", color="k", fontsize=fs)
    dtax.set_title("TTL IWP, TWP", fontsize=fs+4)
    plt.savefig("../plots/fig07_fwp_ttliwp_hist_cat_%s.png"%(REGION.lower()), dpi=160)
    print("saved to ../plots/fig07_fwp_ttliwp_hist_cat_%s.png"%(REGION.lower()))
    plt.close()

    print("Done!")

    return

if __name__=="__main__":
    main()