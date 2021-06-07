#!/usr/bin/env dyamond
#%%
""" vert_cld_frac.py
    Author: Sami Turbeville
    Updated: 16 Oct 2020
    
    This script reads in the frozen water content for
    NICAM, FV3, ICON, SAM, DARDAR, and CloudSat-CALIPSO.
    It then plots the cloud occurrence at each vertical
    level for the region.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from . import load, util
from . import analysis_parameters as ap

c = ap.COLORS
THRES = 5e-7 #kg/m3 = 5e-4 g/m3
TWP_CF = [13, 13, 0.4, 25, 27, 18]
SHL_CF = [34, 25, 8,   30, 46, 29]
NAU_CF = [19, 42, 2,   40, 42, 33]
MODELS = ["DARDAR", "CCCM", "NICAM", "FV3", "ICON","SAM"]
TWP_CF_DICT = dict(zip(MODELS,TWP_CF))
SHL_CF_DICT = dict(zip(MODELS,SHL_CF))
NAU_CF_DICT = dict(zip(MODELS,NAU_CF))


def get_ilwc(model, region):
    """ Returns the total water content for specified region and model
    Paramters: 
        - model (str)  : dyamond model (eg. NICAM, FV3, ICON, etc.)
        - region (str) : TWP, NAU, or SHL
    Returns: 
        - np.array     : total water content (kg/m3)
                         numpy array of dimensions (time, z, xdim, ydim)
                         for ICON dimensions are (time, z, xydim)
    """
    qi = load.load_frozen(model, region, ice_only=True).astype('float16')
    ql = load.load_liq(model, region).values.astype('float16')
    qil = (qi + ql).astype('float16')
    del qi, ql
    print('... added qi + ql ...\n... del qi, ql...')
    ilwc = load.q_to_iwc(qil, model, region)
    del qil
    return ilwc

def get_fwc(model, region):
    """Returns the frozen water content for specified region and model in kg/m3.
    Paramters: 
        - model (str)  : dyamond model (eg. NICAM, FV3, ICON, etc.)
        - region (str) : TWP, NAU, or SHL
    Returns: 
        - np.array     : total water content (kg/m3)
                         numpy array of dimensions (time, z, xdim, ydim)
                         for ICON dimensions are (time, z, xydim)
    """
    if region!="TWP":
        if region=="NAU":
            q = xr.open_dataarray("/home/disk/eos15/smturbev/dyamond/temp_%s_tot_q_%s.nc"%(model,region))
        else:
            q = xr.open_dataarray("../../dyamond/temp_%s_tot_q_%s.nc"%(model,region))
        iwc = load.q_to_iwc(q,model,region)
        del q
    else:
        iwc = load.get_twc(model, region)
    return iwc

def get_cld_frac(model, region, ice_only=True, q=None):
    """Returns the vertical cloud fraction and altitude for each model and region as a tuple.
    Paramters: 
        - model (str)     : dyamond model (eg. NICAM, FV3, ICON, etc.)
        - region (str)    : TWP, NAU, or SHL
        - ice_only (bool) : if False, uses 2D -> 3D froz hydrometeor estimation
        - q (np.array)    : Mixing ratio - None or np.array
    """
    if model=="OBS":
        # CloudSat-CALIPSO (cc) & DARDAR (dd)
        if region=="TWP":
            cc_frac = xr.open_dataset(ap.CERES_TWP)["Cloud fraction (CALIPSO-CloudSat)"]
            dd_iwc = xr.open_dataset(ap.DARDAR_TWP)['iwc']
        elif region=="SHL":
            cc_frac = xr.open_dataset(ap.CERES_SHL)["Cloud fraction (CALIPSO-CloudSat)"]
            dd_iwc = xr.open_dataset(ap.DARDAR_NAU)['iwc']
        else:
            cc_frac = xr.open_dataset(ap.CERES_NAU)["Cloud fraction (CALIPSO-CloudSat)"]
            dd_iwc = xr.open_dataset(ap.DARDAR_SHL)['iwc']

        cc_frac = cc_frac.mean(dim=["time"])  
        dd_frac = np.where(dd_iwc>(THRES*1000),1,0)
        dno = dd_frac.shape[0]
        dd_frac = np.nansum(dd_frac,axis=(0))/dno

        dz = dd_iwc.height
        del dd_iwc
        print("\treturned cloudsat-calipso as xarray with alt as dimension")
        print("\treturned dardar as tuple of numpy array of cld frac and altitude")
        return (cc_frac), (dd_frac, dz)
    else:
        if q is not None:
            print("\tq is not None")
            iwc = load.q_to_iwc(q, model, region)
        else:
            if ice_only:
                print("\tusing qi and ql only")
                iwc = get_ilwc(model, region)
            elif not(ice_only):
                print("\tusing total frozen hydrometeors")
                iwc = get_fwc(model, region)
        if (model=="SAM") & (region=="TWP"):
            iwc = iwc*1000
        print("\tgot ilwc for",model,region)
        model_frac = np.where(iwc>THRES,1,0)
        del iwc
        if model=="ICON":
            model_no = len(model_frac[:,0,:].flatten())
            model_frac = np.nansum(model_frac,axis=(0,2))/model_no
        else:
            model_no = len(model_frac[:,0,:,:].flatten())
            model_frac = np.nansum(model_frac,axis=(0,2,3))/model_no
        del model_no
        model_z = load.get_levels(model, region)
        print("\treturning model cld frac and z for",model, region)
        return (model_frac, model_z)
    return

def plot_vert_cld_frac(region, ax=None, ice_only=True, savename=None, fs=18):
    """Produces the figure of vertical cloud fraction for the region specified.
    Parameters:
        - region (str)   : TWP, NAU, or SHL
        - model (str)    : NICAM, FV3, ICON, SAM, etc.
        - ax (axis obj)  : pyplot.axis object for using subplots; if None, saves 
                           plot in savename location
        - ice_only (bool): if False, uses 2D -> 3D froz hydrometeor estimation
        - plot_ttl (bool): if True, uses pyplot.fillbetween to show the 14-18km layer
                           defined as the TTL
        - savename (str) : relative path where to save figure (e.g. "../example.png")
    Returns: 
        - pyplot.axis    : axis with plot of specified region cloud fraction profiles
    """
    
    print("Beginning...")
    i_frac, iz = get_cld_frac("ICON",region,ice_only=ice_only)
    n_frac, nz = get_cld_frac("NICAM",region,ice_only=ice_only)
    s_frac, sz = get_cld_frac("SAM",region,ice_only=ice_only)
    f_frac, fz = get_cld_frac("FV3",region,ice_only=ice_only)
    cc_frac, (dd_frac, dz) = get_cld_frac("OBS",region)
    print("Testing mean cld frac...")
    print(i_frac.mean(), n_frac.mean(), s_frac.mean(), f_frac.mean())
    print("Plotting...")
    # plot cld frac for fthres
    if ax is None:
        fig = plt.figure(figsize=(5,4))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
    ind0 = np.argmin(abs(dz.values-5000))
    if region.lower()=="twp":
        cf_list = TWP_CF
    elif region.lower()=="shl":
        cf_list = SHL_CF
    elif region.lower()=="nau":
        cf_list = NAU_CF
    else:
        raise Exception("use region=twp, shl or nau")
    ax.plot(dd_frac[:ind0], dz[:ind0]/1000, color=c['OBS'], lw=5, label="DARDAR ({}%)".format(cf_list[0])) #cut-off DARDAR at 5km
    ax.plot(cc_frac[:ind0], cc_frac.alt[:ind0], color=c['OBS'], lw=4, label="CCCM ({}%)".format(cf_list[1]), linestyle="dashed")
    if not(ice_only):
        alpha=0.5
    else:
        alpha=1
    ax.plot(n_frac, nz/1000, color=c['NICAM'], lw=3, label="NICAM ({}%)".format(cf_list[2]), alpha=alpha)
    ax.plot(f_frac, fz/1000, color=c['FV3'], lw=3, label="FV3 ({}%)".format(cf_list[3]), alpha=alpha)
    ax.plot(i_frac, iz/1000, color=c['ICON'], lw=3, label="ICON ({}%)".format(cf_list[4]), alpha=alpha)
    ax.plot(s_frac, sz/1000, color=c['SAM'], lw=3, label="SAM ({}%)".format(cf_list[5]), alpha=alpha)
    ax.fill_between([-0.03,0.83],14,18, color='black', alpha=0.2, label="TTL")
    ax.set_ylim([0,20])
    ax.set_xlim([-0.03,0.83]) #83
    ax.set_ylabel("Height (km)",fontsize=fs)
    ax.set_title(region, fontsize=fs)
    ax.tick_params(labelsize=fs-2)
    ax.grid()
    if ax is None:
        if savename is None:
            plt.savefig("../plots/frozen_cloud_occurrence_%s.png"%(region), bbox_inches="tight", dpi=200)
            print("saved as ../plots/frozen_cloud_occurrence_%s.png"%(region))
        else:
            plt.savefig((savename), bbox_inches="tight", dpi=200)
            print("saved as %s.png"%(savename))
        plt.close()
    return ax

def plot_shl_twp_nau_cld_frac(fs=22, savename=None):
    """Plots and saves regional comparison of vertical cloud occurrence. Used for Figure 9 in paper.
    Parameters: 
        - fs (int)      : fontsize for annotation of (a), (b), and (c) for subplot labels
        - savename (str): path relative location and file name with extension for figure
    Returns:
        - None
    """
    fig, [axs, axt, axn] = plt.subplots(1,3,figsize=(18,7),sharey=True,sharex=True)
    axs = plot_vert_cld_frac("SHL", ax=axs)
    axt = plot_vert_cld_frac("TWP", ax=axt)
    axn = plot_vert_cld_frac("NAU", ax=axn)
    hs, ls = axs.get_legend_handles_labels()
    _, lt = axt.get_legend_handles_labels()
    _, ln = axn.get_legend_handles_labels()
    axs.legend().remove()
    axt.legend().remove()
    axn.legend().remove()
    new_label = []
    new_ln = []
    new_ls = []
    new_lt = []
    fig.subplots_adjust(right=0.95)
    for i,l in enumerate(ls):
        new_label.append(l.split(" ")[0]) # l.split(")")[0][:-1] + ", " + lt[i].split("(")[-1][:-2] + ", " + ln[i].split("(")[-1][:-2]+")"
        new_ls.append(ls[i].split("(")[-1][:-1])
        new_lt.append(lt[i].split("(")[-1][:-1])
        new_ln.append(ln[i].split("(")[-1][:-1])
    new_label[-1]="TTL"
    fig_legend = axt.legend(hs, new_label, loc=9, bbox_to_anchor=(0.5,-0.1), fontsize=fs-4, ncol=7)
    axs.legend = axs.legend(hs[:-1], new_ls[:-1], loc="lower right")
    axt.legend = axt.legend(hs[:-1], new_lt[:-1], loc="lower right")
    axn.legend = axn.legend(hs[:-1], new_ln[:-1], loc="lower right")
    axt.add_artist(fig_legend)

    axt.set_ylabel("")
    axn.set_ylabel("")
    axs.annotate("(a)", xy=(0.7, 18.32), xycoords="data", fontsize=fs, weight="bold")
    axt.annotate("(b)", xy=(0.7, 18.32), xycoords="data", fontsize=fs, weight="bold")
    axn.annotate("(c)", xy=(0.7, 18.32), xycoords="data", fontsize=fs, weight="bold")
    if savename is None: 
        plt.savefig("../plots/fig09_vert_cld_frac.png", dpi=200, bbox_inches="tight", pad_inches=1)
        print("saved as ../plots/fig09_vert_cld_frac.png")
    else: 
        plt.savefig(savename, dpi=200, bbox_inches="tight", pad_inches=1)
        print("saved as %s"%(savename))
    plt.close()

def plot_vert_cld_frac_model(model, region, ax=None, ice_only=True, plot_ttl=True, savename=None):
    """Produces the figure of vertical cloud fraction for the region and model specified.
    Parameters:
        - region (str)   : TWP, NAU, or SHL
        - model (str)    : NICAM, FV3, ICON, SAM, etc.
        - ax (axis obj)  : pyplot.axis object for using subplots
        - ice_only (bool): if False, uses 2D -> 3D froz hydrometeor estimation
        - plot_ttl (bool): if True, uses pyplot.fillbetween to show the 14-18km layer
                           defined as the TTL
        - savename (str) : relative path where to save figure (e.g. "../example.png")
    Returns: 
        - pyplot.axis    : axis with plot of specified region cloud fraction profiles
    
    """
    print("Beginning...")
    if model.lower()=="icon":
        print("icon")
        frac, z = get_cld_frac("ICON",region,ice_only=ice_only)
    elif model.lower()=="nicam":
        frac, z = get_cld_frac("NICAM",region,ice_only=ice_only)
    elif model.lower()=="sam":
        frac, z = get_cld_frac("SAM",region,ice_only=ice_only)
    elif model.lower()=="fv3":
        frac, z = get_cld_frac("FV3",region,ice_only=ice_only)
    else:
        cc_frac, (dd_frac, dz) = get_cld_frac("OBS",region)

    # plot cld frac for fthres
    if ax is None:
        fig = plt.figure(figsize=(5,4))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
    if region.lower()=="twp":
        cf_list = TWP_CF_DICT
    elif region.lower()=="shl":
        cf_list = SHL_CF_DICT
    elif region.lower()=="nau":
        cf_list = NAU_CF_DICT
    if not(ice_only):
        alpha=0.5
    else:
        alpha=1
    if model=="OBS":
        ind0 = np.argmin(abs(dz.values-5000))
        ax.plot(dd_frac[:ind0], dz[:ind0]/1000, color=c['OBS'], lw=5, label="DARDAR ({}%)".format(cf_list["DARDAR"])) #cut-off DARDAR at 5km
        ax.plot(cc_frac[:ind0], cc_frac.alt[:ind0], color=c['OBS'], lw=4, 
                label="CloudSat-\nCALIPSO ({}%)".format(cf_list["CCCM"]), linestyle="dashed")
    else:
        ax.plot(frac, z/1000, color=c[model], lw=3, label="{} ({}%)".format(model, cf_list[model]), alpha=alpha)
    if plot_ttl:
        ax.fill_between([-0.03,0.83],14,18, color='black', alpha=0.2, label="TTL")
    ax.set_ylim([0,20])
    ax.set_xlim([-0.03,0.83]) #83
    ax.legend(loc=4)
    ax.set_ylabel("Height (km)")
    ax.set_title('Cloud Occurrence, %s'%(region), fontsize=16)
    ax.grid()
    if ax is None:
        if savename is None:
            plt.savefig("../plots/frozen_cloud_occurrence_%s.png"%(region), bbox_inches="tight", dpi=200)
            print("saved as ../plots/frozen_cloud_occurrence_%s.png"%(region))
        else:
            plt.savefig((savename), bbox_inches="tight", dpi=200)
            print("saved as %s.png"%(savename))
             
        plt.close()
    return ax
