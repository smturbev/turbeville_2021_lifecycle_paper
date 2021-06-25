#! usr/bin/env python

#%%
import numpy as np
import pandas as pd
from webob.byterange import _is_content_range_valid
from utility import load, load01deg

# %%
def get_cat_lw_sw(model, region, save=True, mean=False):
    if model.lower()=="cccm":
        ds = load01deg.get_cccm(region)
        olr = ds["Outgoing LW radiation at TOA"]
        swu = ds["Outgoing SW radiation at TOA"]
        swd = ds["Incoming SW radiation at TOA"]
        alb = swu/swd.values
        olrcs = ds["Clear-sky outgoing LW radiation at TOA"]
        swucs = ds["Clear-sky outgoing SW radiation at TOA"]
        albcs = swucs/swd.values
        fwp = ds["iwp MODIS"]
        lwp = ds["lwp MODIS"]
    else:
        olr, alb = load.get_olr_alb(model, region)
        fwp = load.get_iwp(model, region, ice_only=False)*1000
        lwp = load.get_lwp(model, region, rain=False)*1000
        if model.lower()!="sam":
            fwp = fwp[11::12]
            lwp = lwp[11::12]
        print(olr.shape, fwp.shape, lwp.shape, alb.shape)
        olrcs = load.get_clearskyolr(model, region, fwp, lwp)
        albcs = load.get_clearskyalb(model, region, fwp, lwp)
        if mean:
            olrcs = np.nanmean(olrcs, axis=0)
            albcs = np.nanmean(albcs, axis=0)
        else:
            olrcs = np.nanmedian(olrcs, axis=0)
            albcs = np.nanmedian(albcs, axis=0)
    print(olr.shape, olrcs.shape, fwp.shape, lwp.shape, alb.shape, albcs.shape)
    # cat mean values (olr, lwcre, alb, swcre)
    thres1 = 1000
    thres2 = 10
    thres3 = 0.1
    if mean:
        # cat 1
        olr1 = np.nanmean(np.where(fwp>thres1, olr, np.nan))
        lwcre1 = np.nanmean(olrcs-(np.where(fwp>thres1, olr, np.nan)))
        alb1 = np.nanmean(np.where(fwp>thres1, alb, np.nan))
        swcre1 = np.nanmean(albcs-(np.where(fwp>thres1, alb, np.nan)))
        # cat 2
        olr2 = np.nanmean(np.where((fwp>thres2)&(fwp<=thres1), olr, np.nan))
        lwcre2= np.nanmean(olrcs-(np.where((fwp>thres2)&(fwp<=thres1), olr, np.nan)))
        alb2 = np.nanmean(np.where((fwp>thres2)&(fwp<=thres1), alb, np.nan))
        swcre2 = np.nanmean(albcs-(np.where((fwp>thres2)&(fwp<=thres1), alb, np.nan)))
        # cat 3
        olr3 = np.nanmean(np.where((fwp>thres3)&(fwp<=thres2), olr, np.nan))
        lwcre3= np.nanmean(olrcs-(np.where((fwp>thres3)&(fwp<=thres2), olr, np.nan)))
        alb3 = np.nanmean(np.where((fwp>thres3)&(fwp<=thres2), alb, np.nan))
        swcre3 = np.nanmean(albcs-(np.where((fwp>thres3)&(fwp<=thres2), alb, np.nan)))
        # clear sky
        olr4 = np.nanmean(olrcs)
        alb4 = np.nanmean(albcs)
    else:
                # cat 1
        olr1 = np.nanmedian(np.where(fwp>thres1, olr, np.nan))
        lwcre1 = np.nanmedian(olrcs-(np.where(fwp>thres1, olr, np.nan)))
        alb1 = np.nanmedian(np.where(fwp>thres1, alb, np.nan))
        swcre1 = np.nanmedian(albcs-(np.where(fwp>thres1, alb, np.nan)))
        # cat 2
        olr2 = np.nanmedian(np.where((fwp>thres2)&(fwp<=thres1), olr, np.nan))
        lwcre2= np.nanmedian(olrcs-(np.where((fwp>thres2)&(fwp<=thres1), olr, np.nan)))
        alb2 = np.nanmedian(np.where((fwp>thres2)&(fwp<=thres1), alb, np.nan))
        swcre2 = np.nanmedian(albcs-(np.where((fwp>thres2)&(fwp<=thres1), alb, np.nan)))
        # cat 3
        olr3 = np.nanmedian(np.where((fwp>thres3)&(fwp<=thres2), olr, np.nan))
        lwcre3= np.nanmedian(olrcs-(np.where((fwp>thres3)&(fwp<=thres2), olr, np.nan)))
        alb3 = np.nanmedian(np.where((fwp>thres3)&(fwp<=thres2), alb, np.nan))
        swcre3 = np.nanmedian(albcs-(np.where((fwp>thres3)&(fwp<=thres2), alb, np.nan)))
        # clear sky
        olr4 = np.nanmedian(olrcs)
        alb4 = np.nanmedian(albcs)

    if region.lower()=="shl":
        solar_const = 435.2760211
    else:
        solar_const = 413.2335274

    olr_list = np.array([olr1, olr2, olr3, olr4])
    alb_list = np.array([alb1, alb2, alb3, alb4])
    lw_list = np.array([lwcre1, lwcre2, lwcre3, 0])
    sw_list = np.array([swcre1, swcre2, swcre3, 0])*solar_const

    df = pd.DataFrame(np.array([olr_list, alb_list, lw_list, sw_list]).T, columns=["olr","alb","lwcre","swcre"], index=["CAT1","CAT2","CAT3","CS"])
    print(df)
    if save:
        df.to_csv("../tables/mean_{}_{}.csv".format(model, region))
        print("saved.")
    return df

def get_isottlci(model, region, thres=0.1):
    """
    Saves the clear sky olr, alb, lw and sw cre to a csv file for
    both the mean and median values of isolated TTL cirrus columns.

    Parameters:
        -thres (f)  : clear-sky threshold in g/m2
        -model (str): model acronym
        -region(str): region acronym
    """
    thres = thres # to kg/m2
    olr, alb = load.get_olr_alb(model, region)
    time = np.arange(3,alb.shape[0]*3+3,3)%24
    if model.lower()=="nicam":
        time = time[:, np.newaxis, np.newaxis, np.newaxis]
    elif model.lower()=="icon":
        time = time[:, np.newaxis]
    else:
        time = np.array(time)[:, np.newaxis, np.newaxis]
    if region.lower()=="twp":
        alb = alb.where(time>=20)
        olr = olr.where(time>=20)
    elif region.lower()=="nau":
        alb = alb.where((time>=22)|(time<=2))
        olr = olr.where((time>=22)|(time<=2))
    else:
        alb = alb.where((time>=11)&(time<=15))
        olr = olr.where((time>=11)&(time<=15))
    fwp = load.get_iwp(model, region, ice_only=False)*1000
    ttliwp = load.get_ttliwp(model, region)*1000
    lwp = load.get_lwp(model, region, rain=False)*1000
    if model.lower()!="sam":
        fwp = fwp[11::12]
        lwp = lwp[11::12]
    print(olr.shape, fwp.shape, lwp.shape, alb.shape)
    olrcs = load.get_clearskyolr(model, region, fwp, lwp)
    albcs = load.get_clearskyalb(model, region, fwp, lwp)
    print(olr.shape, olrcs.shape, fwp.shape, lwp.shape, alb.shape, albcs.shape)
    if model.lower()=="nicam":
        fwp = fwp[:,0]
        lwp = lwp[:,0]
        olr = olr[:,0]
        alb = alb[:,0]
        albcs = albcs[:,0]
        olrcs = olrcs[:,0]
    # get isolated ttl ci
    print("fwp and ttliwp shape", fwp.shape, ttliwp.shape)
    wp_below14km = (fwp.values - ttliwp.values) + lwp.values
    del fwp, lwp
    print("got wp below 14km")
    if region.lower()=="shl":
        solar_const = 435.2760211
    else:
        solar_const = 413.2335274
    isottlci_mask = np.where((wp_below14km<thres)&(ttliwp>=thres), True, False)
    olr_iso = np.where(isottlci_mask, olr, np.nan)
    del olr
    alb_iso = np.where(isottlci_mask, alb, np.nan)
    del alb
    freq = np.sum(isottlci_mask)/len(isottlci_mask.flatten())
    print("freq", model, region, freq, np.sum(isottlci_mask), len(isottlci_mask.flatten()))
    lwcre_mean = np.nanmean(olrcs-olr_iso)
    swcre_mean = np.nanmean(albcs-alb_iso)*solar_const
    df_mean = pd.read_csv("../tables/iso_ttl_ci_all_mean.csv", index_col=0)
    print("mean",[np.nanmean(olr_iso), np.nanmean(alb_iso), lwcre_mean, swcre_mean])
    df_mean[model+"_"+region] = [freq, np.nanmean(olr_iso), np.nanmean(alb_iso), lwcre_mean, swcre_mean]
    df_mean.to_csv("../tables/iso_ttl_ci_all_mean.csv")
    lwcre_med = np.nanmedian(olrcs-olr_iso)
    swcre_med = np.nanmedian(albcs-alb_iso)*solar_const
    df_med = pd.read_csv("../tables/iso_ttl_ci_all_median.csv", index_col=0)
    print("median",[np.nanmedian(olr_iso), np.nanmedian(alb_iso), lwcre_med, swcre_med])
    df_med[model+"_"+region] = [freq, np.nanmedian(olr_iso), np.nanmedian(alb_iso), lwcre_med, swcre_med]
    df_med.to_csv("../tables/iso_ttl_ci_all_median.csv")
    print("...done.")
    return

# %%
models = ["NICAM","FV3","SAM","ICON"]
regions= ["TWP","SHL","NAU"]
for m in models:
    for r in regions:
        # get_cat_lw_sw(m,r)
        get_isottlci(m,r)
