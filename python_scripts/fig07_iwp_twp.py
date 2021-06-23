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

def moving_average(a, n=3) :
    # ret = np.cumsum(a, dtype=float)
    # ret[n:] = ret[n:] - ret[:-n]
    # return ret[n - 1:] / n
    if len(a)%3==1:
        a = a[:-1]
    elif len(a)%3==2:
        a = a[:-2]
    print(len(a)%3)
    avg = (a[0::3]+a[1::3]+a[2::3])/3
    return avg

def main(fs = 14):
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
    ciwp = moving_average(ciwp.values, n=3)
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
    cno = (len(ciwp.flatten()))
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
    fig = plt.figure(figsize=(4,11), constrained_layout=True)
    gs = fig.add_gridspec(5,1,hspace=0.05)
    xmid = (xedges[1:]+xedges[:-1])/2

    cax = fig.add_subplot(gs[0])
    nax = fig.add_subplot(gs[1])
    fax = fig.add_subplot(gs[2])
    iax = fig.add_subplot(gs[3])
    sax = fig.add_subplot(gs[4])

    cax.bar(xmid[:cs_bin_ind], chist[:cs_bin_ind], color='lightgray', width=0.1)
    cax.bar(xmid[cs_bin_ind:lower_bin_ind], chist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    cax.bar(xmid[lower_bin_ind:upper_bin_ind], chist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    cax.bar(xmid[upper_bin_ind:], chist[upper_bin_ind:], color='tab:green', width=0.1)
    cax.set_ylim([0,0.075])
    cax.annotate(str(np.round(ccat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate(str(np.round(ccat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate(str(np.round(ccat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate(str(np.round(ccat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("CAT 1", xy=(3.,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("CAT 2", xy=(1.6,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("CAT 3", xy=(-0.2,0.062), xycoords="data", color="k", fontsize=fs-2)
    cax.annotate("None", xy=(-1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
#     cax.set_title(REGION, size=fs)
    cax.set_xticklabels([])
    cax.set_ylabel("DARDAR\nFraction of profiles", fontsize=fs)

    fax.bar(xmid[:cs_bin_ind], fhist[:cs_bin_ind], color='lightgray', width=0.1)
    fax.bar(xmid[cs_bin_ind:lower_bin_ind], fhist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    fax.bar(xmid[lower_bin_ind:upper_bin_ind], fhist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    fax.bar(xmid[upper_bin_ind:], fhist[upper_bin_ind:], color='tab:green', width=0.1)
    fax.set_ylim([0,0.075])
    fax.annotate(str(np.round(fcat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate(str(np.round(fcat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate(str(np.round(fcat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate(str(np.round(fcat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("CAT 1", xy=(3.,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("CAT 2", xy=(1.6,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("CAT 3", xy=(-0.2,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.annotate("None", xy=(-1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    fax.set_xticklabels([])
    fax.set_ylabel("FV3\nFraction of profiles", fontsize=fs)

    iax.bar(xmid[:cs_bin_ind], ihist[:cs_bin_ind], color='lightgray', width=0.1)
    iax.bar(xmid[cs_bin_ind:lower_bin_ind], ihist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    iax.bar(xmid[lower_bin_ind:upper_bin_ind], ihist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    iax.bar(xmid[upper_bin_ind:], ihist[upper_bin_ind:], color='tab:green', width=0.1)
    iax.set_ylim([0,0.075])
    iax.annotate(str(np.round(icat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate(str(np.round(icat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate(str(np.round(icat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate(str(np.round(icat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("CAT 1", xy=(3.,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("CAT 2", xy=(1.6,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("CAT 3", xy=(-0.2,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.annotate("None", xy=(-1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    iax.set_xticklabels([])
    iax.set_ylabel("ICON\nFraction of profiles", fontsize=fs)

    sax.bar(xmid[:cs_bin_ind], shist[:cs_bin_ind], color='lightgray', width=0.1)
    sax.bar(xmid[cs_bin_ind:lower_bin_ind], shist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    sax.bar(xmid[lower_bin_ind:upper_bin_ind], shist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    sax.bar(xmid[upper_bin_ind:], shist[upper_bin_ind:], color='tab:green', width=0.1)
    sax.set_ylim([0,0.075])
    sax.annotate(str(np.round(scat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate(str(np.round(scat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate(str(np.round(scat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate(str(np.round(scat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("CAT 1", xy=(3.,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("CAT 2", xy=(1.6,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("CAT 3", xy=(-0.2,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.annotate("None", xy=(-1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    sax.set_xlabel("log$_{10}$FWP (g/m$^2$)", fontsize=fs)
    sax.set_ylabel("SAM\nFraction of profiles", fontsize=fs)

    nax.bar(xmid[:cs_bin_ind], nhist[:cs_bin_ind], color='lightgray', width=0.1)
    nax.bar(xmid[cs_bin_ind:lower_bin_ind], nhist[cs_bin_ind:lower_bin_ind], color='C0', width=0.1)
    nax.bar(xmid[lower_bin_ind:upper_bin_ind], nhist[lower_bin_ind:upper_bin_ind], color='#C82F2F', width=0.1)
    nax.bar(xmid[upper_bin_ind:], nhist[upper_bin_ind:], color='tab:green', width=0.1)
    nax.set_ylim([0,0.075])
    nax.annotate(str(np.round(ncat1,0)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate(str(np.round(ncat2,0)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate(str(np.round(ncat3,0)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate(str(np.round(ncat0,0)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("CAT 1", xy=(3.,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("CAT 2", xy=(1.6,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("CAT 3", xy=(-0.2,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.annotate("None", xy=(-1.7,0.062), xycoords="data", color="k", fontsize=fs-2)
    nax.set_xticklabels([])
    nax.set_ylabel("NICAM\nFraction of profiles", fontsize=fs)

    ticks = np.arange(-3,4,1)
    cax.set_xticks(ticks)
    nax.set_xticks(ticks)
    fax.set_xticks(ticks)
    iax.set_xticks(ticks)
    sax.set_xticks(ticks)
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
    plt.savefig("../plots/fig07_fwp_hist_cat_%s_coarsened.png"%(REGION.lower()), dpi=160)
    print("saved to ../plots/fig07_fwp_hist_cat_%s_coarsened.png"%(REGION.lower()))
    plt.close()

    print("Done!")

    return

if __name__=="__main__":
    main(fs=13)