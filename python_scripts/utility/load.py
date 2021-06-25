#%% 
""" load.py
    author: sami turbeville @smturbev
    date modified: 1 March 2021
    
    Loads various variables from FV3, ICON, SAM and NICAM for cleaner scripts.
        - get_iwp(model, region, ice_only=True, sam_noise=True, is3d=True)
        - get_lwp(model, region, rain=False, sam_noise=True, is3d=True)
        - get_ttliwp(model, region)
        - get_levels(model, region)
        - get_pr(model, region)
        - get_pres(model, region)
        - get_temp(model, region)
        - get_qv(model, region)
        - get_swd(model, region)
        - get_olr_alb(model, region)
        - get_clearskyolr(modek, region)
        - load_tot_hydro(model, region)
        - load_frozen(model, region, ice_only=False)
"""

import xarray as xr
import numpy as np
import time
import sys
import pandas as pd
from . import analysis_parameters as ap
from . import util, reshape
np.warnings.filterwarnings("ignore")

INCLUDE_SHOCK = False # True uses full time period, False cuts out the first two days

### ------------------ get methods -------------------------------- ###
### ------ 2D ----- ###
def get_iwp(model, region, ice_only=True, sam_noise=True, is3d=True):
    """ Return the 2D iwp on native grid. """
    if ice_only:
        print("... Getting iwp for %s in the %s region ..."%(model, region))
    else:
        print("... Getting fwp for %s in the %s region ..."%(model, region))
    if INCLUDE_SHOCK:
        ind0=0
    else:
        ind0 = 96*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            return(xr.open_dataset(ap.TWP_NICAM_IWP).sa_cldi[ind0:])
        elif region.lower()=="shl":
            return(xr.open_dataset(ap.SHL_NICAM_IWP).sa_cldi[ind0:])
        elif region.lower()=="nau":
            return(xr.open_dataset(ap.NAU_NICAM_IWP).sa_cldi[ind0:])
        print("NICAM only has frozen water path (2d) output... not iwp, swp, gwp separately")
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.TWP_FV3_IWP).intqi
            if not(ice_only):
                swp = xr.open_dataset(ap.TWP_FV3_SWP).intqs
                gwp = xr.open_dataset(ap.TWP_FV3_GWP).intqg
                fwp = iwp + swp + gwp
                return fwp[ind0:]
            return iwp[ind0:]
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.SHL_FV3_IWP).intqi
            if not(ice_only):
                swp = xr.open_dataset(ap.SHL_FV3_SWP).intqs
                gwp = xr.open_dataset(ap.SHL_FV3_GWP).intqg
                fwp = iwp + swp + gwp
                return fwp[ind0:]
            return iwp[ind0:]
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.NAU_FV3_IWP).intqi
            if not(ice_only):
                swp = xr.open_dataset(ap.NAU_FV3_SWP).intqs
                gwp = xr.open_dataset(ap.NAU_FV3_GWP).intqg
                fwp = iwp + swp + gwp
                return fwp[ind0:]
            return iwp[ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            ds = xr.open_dataset(ap.TWP_ICON_IWP)
        elif region.lower()=="shl":
            ds = xr.open_dataset(ap.SHL_ICON_IWP)
        elif region.lower()=="nau":
            ds = xr.open_dataset(ap.NAU_ICON_IWP)
        else: assert Exception("region not valid, try SHL, NAU, or TWP")
        iwp = reshape.reshape("TQI_DIA", ds, dim=2)
        print(iwp.shape)
        if not(ice_only):
            swp = reshape.reshape("TQS", ds, dim=2)
            print(swp.shape)
            gwp = reshape.reshape("TQG", ds, dim=2)
            print(gwp.shape)
            fwp = iwp + swp + gwp
            return fwp[ind0:]
        return iwp[ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.TWP_SAM_WP_NOISE).IWP if is3d else  xr.open_dataset(ap.TWP_SAM_IWP).IWP
            print("SAM,", iwp.shape)
            if not(ice_only):
                if sam_noise:
                    swp = xr.open_dataset(ap.TWP_SAM_WP_NOISE).SWP
                    gwp = xr.open_dataset(ap.TWP_SAM_WP_NOISE).GWP
                    fwp = iwp + swp + gwp
                    print(swp.shape, "returning fwp as type",type(fwp))
                    return fwp[16:]
                else:
                    swp = xr.open_dataset(ap.TWP_SAM_SWP).SWP
                    gwp = xr.open_dataset(ap.TWP_SAM_GWP).GWP
                    fwp = iwp + swp + gwp
                    print("fwp = iwp + swp + gwp")
                    return fwp[ind0//12:] # three hourly
            else: return iwp[ind0//12:]
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.SHL_SAM_WP_NOISE).IWP if is3d else  xr.open_dataset(ap.SHL_SAM_IWP).IWP
            print("SAM,", iwp.shape)
            if not(ice_only):
                if sam_noise:
                    swp = xr.open_dataset(ap.SHL_SAM_WP_NOISE).SWP
                    gwp = xr.open_dataset(ap.SHL_SAM_WP_NOISE).GWP
                    fwp = iwp + swp + gwp
                    return fwp[16:]
                else:
                    swp = xr.open_dataset(ap.SHL_SAM_SWP).SWP
                    gwp = xr.open_dataset(ap.SHL_SAM_GWP).GWP
                    fwp = iwp + swp + gwp
                    print("fwp = iwp + swp + gwp")
                    return fwp[ind0//12:] # three hourly
            else: return iwp[ind0//12:]
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.NAU_SAM_WP_NOISE).IWP if is3d else  xr.open_dataset(ap.NAU_SAM_IWP).IWP
            print("SAM,", iwp.shape)
            if not(ice_only):
                if sam_noise:
                    swp = xr.open_dataset(ap.NAU_SAM_WP_NOISE).SWP
                    gwp = xr.open_dataset(ap.NAU_SAM_WP_NOISE).GWP
                    fwp = iwp + swp + gwp
                    return fwp[16:]
                else:
                    swp = xr.open_dataset(ap.NAU_SAM_SWP).SWP
                    gwp = xr.open_dataset(ap.NAU_SAM_GWP).GWP
                    fwp = iwp + swp + gwp
                    print("fwp = iwp + swp + gwp")
                    return fwp[ind0//12:] # three hourly
            else: return iwp[ind0//12:]
        else:
            raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    return

def get_lwp(model, region, rain=False, sam_noise=True, is3d=True):
    """ Return the 2D iwp on native grid. """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 96*2 # exclude first two days
    if model.lower()=="nicam":
        print("... returning frozen water path for NICAM.")
        if region.lower()=="twp":
            return(xr.open_dataset(ap.TWP_NICAM_LWP).sa_cldw[ind0:])
        elif region.lower()=="shl":
            return(xr.open_dataset(ap.SHL_NICAM_LWP).sa_cldw[ind0:])
        elif region.lower()=="nau":
            return(xr.open_dataset(ap.NAU_NICAM_LWP).sa_cldw[ind0:])
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            cwp = xr.open_dataset(ap.TWP_FV3_LWP).intql
            if rain:
                rwp = xr.open_dataset(ap.TWP_FV3_RWP).intqr
                lwp = cwp + rwp
                return lwp[ind0:]
            return cwp[ind0:]
        elif region.lower()=="shl":
            cwp = xr.open_dataset(ap.SHL_FV3_LWP).intql
            if rain:
                rwp = xr.open_dataset(ap.SHL_FV3_RWP).intqr
                lwp = cwp + rwp
                return lwp[ind0:]
            return cwp[ind0:]
        elif region.lower()=="nau":
            cwp = xr.open_dataset(ap.NAU_FV3_LWP).intql
            if rain:
                rwp = xr.open_dataset(ap.NAU_FV3_RWP).intqr
                lwp = cwp + rwp
                return lwp[ind0:]
            return cwp[ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            ds = xr.open_dataset(ap.TWP_ICON_IWP)
        elif region.lower()=="shl":
            ds = xr.open_dataset(ap.SHL_ICON_IWP)
        elif region.lower()=="nau":
            ds = xr.open_dataset(ap.NAU_ICON_IWP)
        else: assert Exception("region not valid, try SHL, NAU, or TWP")
        cwp = reshape.reshape("TQC_DIA", ds, dim=2)
        print(cwp.shape)
        return cwp[ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            if is3d:
                lwp = xr.open_dataset(ap.TWP_SAM_LWP_NOISE).CWP
                if rain:
                    rwp = xr.open_dataset(ap.TWP_SAM_LWP_NOISE).RWP
                    return (lwp + rwp)[ind0//12:] # three hourly
            else:
                lwp = xr.open_dataset(ap.TWP_SAM_LWP).CWP
                if rain:
                    rwp = xr.open_dataset(ap.TWP_SAM_RWP).RWP
                    return (lwp + rwp)[ind0:]
            return lwp[ind0//12:]
        elif region.lower()=="shl":
            if is3d:
                lwp = xr.open_dataset(ap.SHL_SAM_LWP_NOISE).CWP
                if rain:
                    rwp = xr.open_dataset(ap.SHL_SAM_LWP_NOISE).RWP
                    return (lwp + rwp)[ind0//12:] # three hourly
            else:
                lwp = xr.open_dataset(ap.SHL_SAM_LWP).CWP
                if rain:
                    rwp = xr.open_dataset(ap.SHL_SAM_RWP).RWP
                    return (lwp + rwp)[ind0:]
            return lwp[ind0//12:]
        elif region.lower()=="nau":
            if is3d:
                lwp = xr.open_dataset(ap.NAU_SAM_LWP_NOISE).CWP
                if rain:
                    rwp = xr.open_dataset(ap.NAU_SAM_LWP_NOISE).RWP
                    return (lwp + rwp)[ind0//12:] # three hourly
            else:
                lwp = xr.open_dataset(ap.NAU_SAM_LWP).CWP
                if rain:
                    rwp = xr.open_dataset(ap.NAU_SAM_RWP).RWP
                    return (lwp + rwp)[ind0:]
            return lwp[ind0//12:]
        else:
            raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    return
    
def get_ttliwp(model, region):
    """ Returns the integrated frozen water path in the 14-18km layer
        for specificed model and region. """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        print("... returning frozen water path for NICAM.")
        if region.lower()=="twp":
            return(xr.open_dataarray(ap.TWP_NICAM_TTLIWP)[ind0:])
        elif region.lower()=="shl":
            return(xr.open_dataarray(ap.SHL_NICAM_TTLIWP)[ind0:])
        elif region.lower()=="nau":
            return(xr.open_dataarray(ap.NAU_NICAM_TTLIWP)[ind0:])
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            return xr.open_dataarray(ap.TWP_FV3_TTLIWP)[ind0:]
        elif region.lower()=="shl":
            return xr.open_dataarray(ap.SHL_FV3_TTLIWP)[ind0:]
        elif region.lower()=="nau":
            return xr.open_dataarray(ap.NAU_FV3_TTLIWP)[ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            return xr.open_dataarray(ap.TWP_ICON_TTLIWP)[ind0:]
        elif region.lower()=="shl":
            return xr.open_dataarray(ap.SHL_ICON_TTLIWP)[ind0:]
        elif region.lower()=="nau":
            return xr.open_dataarray(ap.NAU_ICON_TTLIWP)[ind0:]
        else: assert Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            return xr.open_dataarray(ap.TWP_SAM_TTLIWP)[ind0:]
        elif region.lower()=="shl":
            return xr.open_dataarray(ap.SHL_SAM_TTLIWP)[ind0:]
        elif region.lower()=="nau":
            return xr.open_dataarray(ap.NAU_SAM_TTLIWP)[ind0:]
        else:
            raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    return

def get_iwv(model, region):
    """ Returns the total column integrated water vapor for model and region.
    
        
        iwv = -1/g * integral(qv * dp)
        
        model = string of which dyamond model to use
        region = string of "TWP", "SHL" or "NAU"
    """
    p = get_pres(model, region)
    qv = get_qv(model, region)
    print(p.shape, qv.shape)
    if model.lower()=="sam":
        print("    newaxes added")
        p = p.values[:,:,np.newaxis,np.newaxis]
    else:
        p = p.values
    print(qv.mean(), p.mean())
    if model.lower()=="nicam": 
        is_xy, is_td, is_const_p = True, False, False
    elif model.lower()=="icon": 
        is_xy, is_td, is_const_p = False, True, False
    elif model.lower()=="sam": 
        is_xy, is_td, is_const_p = True, False, True
    elif model.lower()=="fv3":
        iwv = util.int_wrt_pres_f(p, qv)
        return iwv
    else: raise Exception("model ({}) or region ({}) not defined".format(model, region))
    iwv = util.int_wrt_pres(p, qv, xy=is_xy, td=is_td, const_p=is_const_p)
    return iwv

def iwp_wrt_pres(model, region, hydro_type="ice"):
    if ((region=="TWP")&(model=="ICON")):
        iwp = xr.open_dataset(ap.TWP_ICON_IWP)["iwp"].values
        return iwp
    else:
        p = get_pres(model,region)
        if hydro_type=="ice":
            q = load_frozen(model, region, ice_only=True).values
        elif hydro_type=="frozen":
            q = load_frozen(model, region, ice_only=False).values
        else:
            q = load_tot_hydro(model, region).values
        p = np.where(np.isnan(p),0,p)
        q = np.where(np.isnan(q),0,q)
        if model.lower()=="nicam":
            vint = util.int_wrt_pres(p,q,xy=True,const_p=False)
        elif model.lower()=="fv3":
            vint = util.int_wrt_pres_f(p,q)
        elif model.lower()=="icon":
            vint = util.int_wrt_pres(p,q,xy=False,td=True,const_p=False)
        elif model.lower()=="sam":
            vint = util.int_wrt_pres(p,q,xy=True,td=False,const_p=True)
        else:
            raise Exception("Model or region not defined. Try FV3, ICON, SAM, NICAM in the TWP, SHL, or NAU.")
        return vint

def q_to_iwc(q, model, region):
    """Converts mixing ratio of q (kg/kg) to ice water content (kg/m3)
        input = model name (string) and q = mixing ratio as xarray or numpyarray.
        Only works for time and space averaged data (aka data has one dimension-height)
        
        returns xarray or numpy array with iwc as kg/m3
    """
    if model.lower() == "fv3":
        t = get_temp(model, region)
        qv = get_qv(model, region)
        p = get_pres(model, region)
        rho = p / \
              (287*(1 + 0.61*(qv))*(np.nanmean(t, axis=(2))[:,:,np.newaxis,np.newaxis]))
        iwc = q.values * rho
        print("Warning: FV3 uses the spatially averaged density b/c \
        specific humidity and temperature are on different grids")
    elif model.lower() =="sam":
        t = get_temp(model, region).values
        qv = get_qv(model, region).values
        p = get_pres(model, region).values
        rho = p[:,:,np.newaxis,np.newaxis] / \
              (287*(1 + 0.61*qv)*t)
        iwc = q.values * rho
    else:
        if model.lower() == "icon":
            t = get_temp(model, region).values.astype('float32')
            qv = get_qv(model, region).values.astype('float16')
            Tv = (1 + 0.61*qv)*t
            print("... Tv ...")
            del qv, t
            p = get_pres(model, region).values.astype('float32')
        else:
            t = get_temp(model, region).values
            qv = get_qv(model, region).values
            p = get_pres(model, region).values
            Tv = (1 + 0.61*qv)*t
            print("... Tv ...")
            del qv, t
        rho = p / (287*Tv) 
        print("... rho ...")
        del p, Tv
        iwc = q * rho  # kg/m2
        print("... iwc ...")
        del rho
    print("Returning ice water content (kg/m3) for %s as %s xarray\n\n"%(model, iwc.shape))
    iwcxr = xr.DataArray(iwc, dims=list(q.dims), coords=q.coords, 
                     attrs={'standard_name':'iwc','long_name':'ice_water_content','units':'kg/m3'})
    return iwcxr

def get_pr(model, region):
    """ Returns preciptiation rate in mm/s for given model and region
            Region must be "twp", "nau" or "shl".
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 96*2 # exclude first two days
    if model.lower()=="nicam":
        print("... returning frozen water path for NICAM.")
        if region.lower()=="twp":
            return(xr.open_dataset(ap.TWP_NICAM_PR)["ss_tppn"][ind0:,0,:,:])
        elif region.lower()=="shl":
            return(xr.open_dataset(ap.SHL_NICAM_PR)["ss_tppn"][ind0:,0,:,:])
        elif region.lower()=="nau":
            return(xr.open_dataset(ap.NAU_NICAM_PR)["ss_tppn"][ind0:,0,:,:])
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            return xr.open_dataset(ap.TWP_FV3_PR)["pr"][ind0:]
        elif region.lower()=="shl":
            return xr.open_dataset(ap.SHL_FV3_PR)["pr"][ind0:]
        elif region.lower()=="nau":
            return xr.open_dataset(ap.NAU_FV3_PR)["pr"][ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            pr = xr.open_dataset(ap.TWP_ICON_PR)["tp"]
        elif region.lower()=="shl":
            pr = xr.open_dataset(ap.SHL_ICON_PR)["tp"]
        elif region.lower()=="nau":
            pr = xr.open_dataset(ap.NAU_ICON_PR)["tp"]
        else: assert Exception("region not valid, try SHL, NAU, or TWP")
        pr = util.precip(pr,dt=15*60,returnPr=True)
        pr = pr.where(pr>=0)
        return pr[ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            p = xr.open_dataset(ap.TWP_SAM_PR)["Precac"]
        elif region.lower()=="shl":
            p = xr.open_dataset(ap.SHL_SAM_PR)["Precac"]
        elif region.lower()=="nau":
            p = xr.open_dataset(ap.NAU_SAM_PR)["Precac"]
        else: raise Exception("try valid region (SHL, NAU, TWP)")
        p = util.precip(p,dt=30*60,returnPr=True)
        p = p.where(p>=0)
        p[490,:,:] = np.nan
        return p[ind0//2:]
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    return

def get_swd(model, region):
    """ Return swd for models in region.
    
        For models that don't output swd we will use the zonal mean
            to estimate swd for closest latitude.
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 96*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.TWP_NICAM_SWD)['ss_swd_toa'][ind0*2:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.NAU_NICAM_SWD)['ss_swd_toa'][ind0*2:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.SHL_NICAM_SWD)['ss_swd_toa'][ind0*2:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.TWP_FV3_SWD)['fsdt'][ind0:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.NAU_FV3_SWD)['fsdt'][ind0:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.SHL_FV3_SWD)['fsdt'][ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.TWP_ICON_SWN)['ASOB_T'][ind0:]
            swu = xr.open_dataset(ap.TWP_ICON_SWU)['ASOU_T'][ind0:]
        else:
            if region.lower()=="shl":
                rad = xr.open_dataset(ap.SHL_ICON_RAD)
            elif region.lower()=="nau":
                rad = xr.open_dataset(ap.NAU_ICON_RAD)
            else: raise Exception("invalid region: try SHL, TWP, or NAU")
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            swu_un = util.undomean(swu, xy=False)
            swn_un = util.undomean(swn, xy=False)
            print(swu)
            swu = xr.DataArray(swu_un, dims=["time","cell"], \
                               coords={"time":swu.t.values,\
                                       "cell":swu.cell})
            swn = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":swu.time.values,\
                                       "cell":swu.cell})
        swd = swn + swu
        swd = swd[ind0:]
        print("\n\t return calculated sw downward at toa for icon")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.TWP_SAM_SWN)['SWNTA']
            swd_n = xr.open_dataset(ap.TWP_NICAM_SWD)#['ss_swd_toa']
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.NAU_SAM_SWN)['SWNTA']
            swd_n = xr.open_dataset(ap.NAU_NICAM_SWD)#['ss_swd_toa']
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.SHL_SAM_SWN)['SWNTA']
            swd_n = xr.open_dataset(ap.SHL_NICAM_SWD)
#         print(swn.shape, swd_n[:1920*2:2,0,:,:].shape)
        swd = swd_n.reindex(indexers={"lat":swn.lat.values,"lon":swn.lon.values}, 
                            method="nearest")
        swd = swd.ss_swd_toa[:1920*2:2,0]
        if swd.shape!=swn.shape:
            raise Exception("Didn't work {},{},{}".format(swd.shape, swn.shape, swd_n['ss_swd_toa'].shape))
        print("SWD", swd.shape)
        swd = swd[ind0//2:]
        print("Excluding initial shock", not(INCLUDE_SHOCK), swd.shape)
#         swd = np.zeros(swn.shape)
#         print('starting loop...')
#         for la in range(len(swn.lat)):
#             for lo in range(len(swn.lon)):
#                 ind_la = np.argmin(abs(swn.lat.values[la]-swd_n.lat.values))
#                 ind_lo = np.argmin(abs(swn.lon.values[lo]-swd_n.lon.values))
#                 swd[:,la,lo] = swd_n[:1920*2:2,0,ind_la,ind_lo]
#         print('...looping done\n\t return calculated sw downward at toa for SAM')
    return swd

def get_asr(model, region):
    """ Return asr (absorbed sw radiation) for models in region."""
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 96*2 # exclude first two days
    if model.lower()=="nicam":
            if region.lower()=="twp":
                swd = xr.open_dataset(ap.TWP_NICAM_SWD)['ss_swd_toa']
                swu = xr.open_dataset(ap.TWP_NICAM_SWU)['ss_swu_toa']
            elif region.lower()=="nau":
                swd = xr.open_dataset(ap.NAU_NICAM_SWD)['ss_swd_toa']
                swu = xr.open_dataset(ap.NAU_NICAM_SWU)['ss_swu_toa']
            elif region.lower()=="shl":
                swd = xr.open_dataset(ap.SHL_NICAM_SWD)['ss_swd_toa']
                swu = xr.open_dataset(ap.SHL_NICAM_SWU)['ss_swu_toa']
            asr = swd - swu
            del swd, swu
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.TWP_FV3_SWD)['fsdt']
            swu = xr.open_dataset(ap.TWP_FV3_SWU)['fsut']
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.NAU_FV3_SWD)['fsdt']
            swu = xr.open_dataset(ap.NAU_FV3_SWU)['fsut']
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.SHL_FV3_SWD)['fsdt']
            swu = xr.open_dataset(ap.NAU_FV3_SWU)['fsut']
        asr = swd - swu
        del swd, swu
    elif model.lower()=="icon":
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.TWP_ICON_SWN)['ASOB_T']
        else:
            if region.lower()=="nau":
                rad = xr.open_dataset(ap.NAU_ICON_RAD)
            elif region.lower()=="shl":
                rad = xr.open_dataset(ap.SHL_ICON_RAD)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            swn_un = util.undomean(swn, xy=False)
            asr = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":swn.t.values, "cell":swn.cell})
            del swn, swn_un, rad
    elif model.lower()=="sam":
        print('sam')
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.TWP_SAM_SWN)['SWNTA']
        elif region.lower()=="nau":
            asr = xr.open_dataset(ap.NAU_SAM_SWN)['SWNTA']
        elif region.lower()=="shl":
            asr = xr.open_dataset(ap.SHL_SAM_SWN)['SWNTA']
        print(asr.shape)
    else: raise Exception("Invalid Model %s; try NICAM, FV3, ICON, SAM.")
    return asr[ind0:]


def get_swu(model, region):
    """Returns the (olr, alb) of each model and region as a tuple of xarrays"""
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_NICAM_SWU)['ss_swu_toa']
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            swu = xr.open_dataset(ap.NAU_NICAM_SWU)['ss_swu_toa']
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.SHL_NICAM_SWU)['ss_swu_toa']
        else: raise Exception("Region not supported (try TWP, NAU, SHL)")
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_FV3_SWU)["fsut"][11::12]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.NAU_FV3_SWU)["fsut"][11::12]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.SHL_FV3_SWU)["fsut"][11::12]
        else: raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
    elif model.lower()=="icon":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_ICON_SWU)["ASOU_T"][11::12]
        else:
            if region.lower()=="shl":
                rad = xr.open_dataset(ap.SHL_ICON_RAD)
            elif region.lower()=="nau":
                rad = xr.open_dataset(ap.NAU_ICON_RAD)
            else: raise Exception("invalid region: try shl, nau, or twp")
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swu_un = util.undomean(swu, xy=False)
            swu = xr.DataArray(swu_un[11::12], dims=["time","cell"], \
                               coords={"time":swu.t.values,\
                                       "cell":swu.cell})
    elif model.lower()=="sam":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.TWP_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "TWP")[5::6,:,:]
            swu = swd - swn
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.SHL_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "SHL")[5::6,:,:]
            swu = swd - swn
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.NAU_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "NAU")[5::6,:,:]
            swu = swd - swn
        else: raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
    else: raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'ICON', 'SAM')")
    print("returned swu 3 hrly")
    return swu[ind0:]

def get_olr_alb(model, region):
    """Returns 3hrly xarray of (olr, alb) of each model and region as a tuple of xarrays"""
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            print("Getting olr and albedo for NICAM TWP:")
            st= time.time()
            olr = xr.open_dataset(ap.TWP_NICAM_OLR)['sa_lwu_toa'][11::12,:,:,:]
            swu = xr.open_dataset(ap.TWP_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
            swd = xr.open_dataset(ap.TWP_NICAM_SWD)['ss_swd_toa'][11::12,:,:,:]
            print("... calculating albedo for shape",olr.shape,swu.shape,swd.shape)
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            print("Getting olr and albedo for NICAM NAURU:")
            st= time.time()
            olr = xr.open_dataset(ap.NAU_NICAM_OLR)['sa_lwu_toa'][11::12,:,:,:]
            swu = xr.open_dataset(ap.NAU_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
            swd = xr.open_dataset(ap.NAU_NICAM_SWD)['ss_swd_toa'][11::12,:,:,:]
            print("... calculating albedo for shape",olr.shape,swu.shape,swd.shape)
        elif region.lower()=="shl":
            print("Getting olr and albedo for NICAM SAHEL:")
            st= time.time()
            olr = xr.open_dataset(ap.SHL_NICAM_OLR)['sa_lwu_toa'][11::12,:,:,:]
            swu = xr.open_dataset(ap.SHL_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
            swd = xr.open_dataset(ap.SHL_NICAM_SWD)['ss_swd_toa'][11::12,:,:,:]
            print("... calculating albedo for shape",olr.shape,swu.shape,swd.shape)
        else: print("Region not supported (try TWP, NAU, SHL)")
        alb = swu/swd
        alb = alb[ind0:]
        olr = olr[ind0:]
        del swu, swd
        print("... calculated albedo and opened olr (%s seconds elapsed)..."%str(time.time()-st))
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            print("Getting olr and albedo for FV3 TWP:")
            olr = xr.open_dataset(ap.TWP_FV3_OLR)["flut"][11::12,:,:]
            swu = xr.open_dataset(ap.TWP_FV3_SWU)["fsut"][11::12,:,:]
            swu = swu[ind0:]
            swd = get_swd("FV3", "TWP")[11::12,:,:]
            alb = swu.values/swd
            print(olr.shape, alb.shape)
        elif region.lower()=="nau":
            print("Getting olr and albedo for FV3 NAU:")
            olr = xr.open_dataset(ap.NAU_FV3_OLR)["flut"][11::12,:,:]
            swu = xr.open_dataset(ap.NAU_FV3_SWU)["fsut"][11::12,:,:]
            swu = swu[ind0:]
            swd = get_swd("FV3", "NAU")[11::12,:,:]
            alb = swu.values/swd
            print(olr.shape, alb.shape)
        elif region.lower()=="shl":
            print("Getting olr and albedo for FV3 SHL:")
            olr = xr.open_dataset(ap.SHL_FV3_OLR)["flut"][11::12,:,:]
            swu = xr.open_dataset(ap.SHL_FV3_SWU)["fsut"][11::12,:,:]
            swu = swu[ind0:]
            swd = get_swd("FV3", "SHL")[11::12,:,:]
            alb = swu.values/swd
            print(olr.shape, alb.shape)
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
        alb = alb
        olr = olr[ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            print("Getting olr and albedo for ICON TWP:")
            olr = xr.open_dataset(ap.TWP_ICON_OLR)["ATHB_T"]
            swu = xr.open_dataset(ap.TWP_ICON_SWU)["ASOU_T"]
            swn = xr.open_dataset(ap.TWP_ICON_SWN)["ASOB_T"]
            swd = swn + swu.values
            del swn
            alb = swu/swd.values
            alb = alb.where((alb.values>0)&(swd.values>0)&(alb.values<1))
        elif region.lower()=="nau":
            print("Getting olr and albedo for ICON NAU:")
            rad = xr.open_dataset(ap.NAU_ICON_RAD)
            olr = reshape.reshape("ATHB_T", rad, dim=2)
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            del rad
            olr_un = util.undomean(olr, xy=False)
            swu_un = util.undomean(swu, xy=False)
            del swu
            swn_un = util.undomean(swn, xy=False)
            del swn
            olr = xr.DataArray(olr_un, dims=["time","cell"], \
                               coords={"time":olr.t.values,\
                                       "cell":olr.cell})
            swu = xr.DataArray(swu_un, dims=["time","cell"], \
                               coords={"time":olr.time.values,\
                                       "cell":olr.cell})
            swn = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":olr.time.values,\
                                       "cell":olr.cell})
            swd = swn + swu
            del swn
            alb = swu/swd
        elif region.lower()=="shl":
            print("Getting olr and albedo for ICON SHL:")
            rad = xr.open_dataset(ap.SHL_ICON_RAD)
            olr = reshape.reshape("ATHB_T", rad, dim=2)
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            olr_un = util.undomean(olr, xy=False)
            swu_un = util.undomean(swu, xy=False)
            swn_un = util.undomean(swn, xy=False)
            olr = xr.DataArray(olr_un, dims=["time","cell"], \
                               coords={"time":olr.t.values,\
                                       "cell":olr.cell})
            swu = xr.DataArray(swu_un, dims=["time","cell"], \
                               coords={"time":olr.time.values,\
                                       "cell":olr.cell})
            swn = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":olr.time.values,\
                                       "cell":olr.cell})
            swd = swn + swu
            del swn
            alb = swu/swd
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
        alb = alb.where((alb<1)&(alb>0))
        alb = alb[11::12]
        olr = olr[11::12]
        alb = alb[ind0:]
        olr = olr[ind0:]
        print(olr.shape, alb.shape)
    elif model.lower()=="sam":
        if region.lower()=="twp":
            print("Getting olr and albedo for SAM TWP:")
            olr = xr.open_dataset(ap.TWP_SAM_OLR)["LWNTA"][5::6,:,:]
            swn = xr.open_dataset(ap.TWP_SAM_SWN)["SWNTA"][5::6,:,:]
            olr = olr[ind0:]
            swn = swn[ind0:]
            swd = get_swd("SAM", "TWP")[5::6,:,:]
        elif region.lower()=="nau":
            print("Getting olr and albedo for SAM NAU:")
            olr = xr.open_dataset(ap.NAU_SAM_OLR)["LWNTA"][5::6,:,:]
            swn = xr.open_dataset(ap.NAU_SAM_SWN)["SWNTA"][5::6,:,:]
            olr = olr[ind0:]
            swn = swn[ind0:]
            swd = get_swd("SAM", "NAU")[5::6,:,:]
        elif region.lower()=="shl":
            print("Getting olr and albedo for SAM SHL:")
            olr = xr.open_dataset(ap.SHL_SAM_OLR)["LWNTA"][5::6,:,:]
            swn = xr.open_dataset(ap.SHL_SAM_SWN)["SWNTA"][5::6,:,:]
            olr = olr[ind0:]
            swn = swn[ind0:]
            swd = get_swd("SAM", "SHL")[5::6,:,:]
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
        print(swd.shape, swn.shape)
        swu = swd.values - swn.values
        print("... subtracted...")
        alb = swu/swd.values
        print("... calculated alb...")
        alb = xr.DataArray(alb, dims=olr.dims, coords=olr.coords, attrs={'long_name':'albedo at TOA (aver)',
                                                            'units':'None'})
        print("... made xarray...")
        alb = alb.where((alb.values>0)&(swd.values>0))
        print("... made sure alb values are valid...")
        print("... calculated mean", alb.mean().values, "...")
        print("... returning olr and albedo", olr.shape, alb.shape, "...")
    else: raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'ICON', 'SAM')")
    return olr, alb

def get_clearskyolr(model, region, fwp=None, lwp=None, thres=0.1):
    """returns clear sky albedo for each grid point.
    
        thres = threshold for clear sky fwp in g/m2
        fwp = frozen water path in g/m2"""
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 96*2 # exclude first two days
    if fwp is None:
        fwp = get_iwp(model, region, ice_only=False)
        fwp = fwp[11::12]
    if lwp is None: 
        lwp = get_lwp(model, region, rain=False)
        lwp = lwp[11::12]
    if model.lower()=="nicam":
        if region.lower()=="twp":
            olrcs = xr.open_dataset(ap.TWP_NICAM_OLR_CS)["ss_lwu_toa_c"][ind0:].values[11::12]
        elif region.lower()=="shl":
            olr, _ = get_olr_alb(model, region)
            olrcs = np.where((fwp<thres)&(lwp<thres), olr, np.nan)
            # check if there are nan values and replace with latitute avg
            # olrcs_lat = np.nanmean(np.where((fwp<thres)&(lwp<thres), olr, np.nan), axis=(0,3))
            # olrcs = np.where(np.isnan(olrcs), olrcs_lat[:,:,np.newaxis], olrcs)
        elif region.lower()=="nau":
            olrcs = xr.open_dataset(ap.NAU_NICAM_OLR_CS)["ss_lwu_toa_c"][ind0:].values[11::12]
        else: raise Exception("region not defined", region)
    else:
        olr, _ = get_olr_alb(model, region)
        olrcs = np.where((fwp+lwp<thres), olr, np.nan)
        # check if there are nan values and replace with latitute avg
        # if model.lower()!="icon": # icon doesn't have lat-lon info
        #     olrcs_lat = np.nanmean(np.where((fwp+lwp<thres), olr, np.nan), axis=(0,2))
        #     olrcs = np.where(np.isnan(olrcs), olrcs_lat[:,np.newaxis], olrcs)
    return olrcs

def get_clearskyalb(model, region, fwp=None, lwp=None, thres=0.1):
    """Returns clear sky albedo for each grid point.
    
        thres = threshold for clear sky fwp in g/m2
        fwp = frozen water path in g/m2"""
    if fwp is None:
        fwp = get_iwp(model, region, ice_only=False)
        fwp = fwp[11::12]
    if lwp is None: 
        lwp = get_lwp(model, region, rain=False)
        lwp = lwp[11::12]
    _, alb = get_olr_alb(model, region)
    time = np.arange(3,alb.shape[0]*3+3,3)%24
    if model.lower()=="nicam":
        time = time[:, np.newaxis, np.newaxis, np.newaxis]
    elif model.lower()=="icon":
        time = time[:, np.newaxis]
    else:
        time = np.array(time)[:, np.newaxis, np.newaxis]
    if region.lower()=="twp":
        alb = alb.where(time>=20)
    elif region.lower()=="nau":
        alb = alb.where((time>=22)|(time<=2))
    else:
        alb = alb.where((time>=11)&(time<=15))
    cs = np.where((fwp+lwp<thres), alb, np.nan)
    # check if there are nan values and replace with latitute avg
    # if model.lower()!="icon": # icon doesn't have lat-lon info
    #     cs_lat = np.nanmean(np.where((fwp+lwp<thres), alb, np.nan), axis=(0,-1))
    #     print(cs_lat.shape, cs.shape)
    #     if model.lower()=="nicam":
    #         cs = np.where(np.isnan(cs), cs_lat[:,:,np.newaxis], cs)
    #     else:
    #         cs = np.where(np.isnan(cs), cs_lat[:,np.newaxis], cs)
    return cs

### ------ 3D ----- ###
def get_levels(model, region="TWP"):
    """Returns numpy array of vertical levels for given model and region."""
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        print("... returning frozen water path for NICAM.")
        if region.lower()=="twp":
            z = xr.open_dataset(ap.TWP_NICAM_QI).lev.values 
        elif region.lower()=="shl":
            z = xr.open_dataset(ap.SHL_NICAM_QI).lev.values 
        elif region.lower()=="nau":
            z = xr.open_dataset(ap.NAU_NICAM_QI).lev.values 
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            z = xr.open_dataset(ap.TWP_FV3_Z).altitude.values
        elif region.lower()=="shl":
            z = xr.open_dataset(ap.SHL_FV3_Z).altitude.values
        elif region.lower()=="nau":
            z = xr.open_dataset(ap.NAU_FV3_Z).altitude.values
        z = np.nanmean(z, axis=0)
    elif model.lower()=="icon":
        if region.lower()=="twp":
            z = xr.open_dataset(ap.SHL_ICON_Z).HHL.values[0]
        elif region.lower()=="shl":
            z = xr.open_dataset(ap.SHL_ICON_Z).HHL.values[0]
        elif region.lower()=="nau":
            z = xr.open_dataset(ap.SHL_ICON_Z).HHL.values[0]
        else: assert Exception("region not valid, try SHL, NAU, or TWP")
        z = np.nanmean(z, axis=1)[14:]
        print(z.shape, ind0>0, "shape of z, if true removed first day of model output")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            z = xr.open_dataset(ap.TWP_SAM_QI).z.values
        elif region.lower()=="shl":
            z = xr.open_dataset(ap.TWP_SAM_QI).z.values
        elif region.lower()=="nau":
            z = xr.open_dataset(ap.TWP_SAM_QI).z.values
        else:
            raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    print("\t returned height with shape", z.shape)
    return z

def get_pres(model, region):
    """Returns pressure in Pascals for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'SAM', or 'NICAM' (five of the DYAMOND models
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            p = xr.open_dataset(ap.TWP_NICAM_P)["ms_pres"][ind0:]
        elif region.lower()=="shl":
            p = xr.open_dataset(ap.SHL_NICAM_P)["ms_pres"][ind0:]
        elif region.lower()=="nau":
            p = xr.open_dataset(ap.NAU_NICAM_P)["ms_pres"][ind0:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            p = xr.open_dataset(ap.TWP_FV3_P)["pres"][ind0:]
        elif region.lower()=="shl":
            p = xr.open_dataset(ap.SHL_FV3_P)["pres"][ind0:]
        elif region.lower()=="nau":
            p = xr.open_dataset(ap.NAU_FV3_P)["pres"][ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            p = xr.open_dataset(ap.TWP_ICON_P)["NEW"][ind0:]
        elif region.lower()=="shl":
            p = xr.open_dataset(ap.SHL_ICON_P)["NEW"][ind0:]
        elif region.lower()=="nau":
            p = xr.open_dataset(ap.NAU_ICON_P)["P"][ind0:]
        else: assert Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            p = ((xr.open_dataset(ap.TWP_SAM_P)["p"])*100)[ind0:,:]
        elif region.lower()=="shl":
            p = ((xr.open_dataset(ap.SHL_SAM_P)["p"])*100)[ind0:,:]
        elif region.lower()=="nau":
            p = ((xr.open_dataset(ap.NAU_SAM_P)["p"])*100)[ind0:,:]
        else:
            raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    print("\t returned pressure with shape", p.shape)
    return p

def get_temp(model, region):
    """Returns temperature in Kelvin for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON',  'SAM', or 'NICAM' (five of the DYAMOND models
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            t = xr.open_dataset(ap.TWP_NICAM_T)["ms_tem"][ind0:]
        elif region.lower()=="shl":
            t = xr.open_dataset(ap.SHL_NICAM_T)["ms_tem"][ind0:]
        elif region.lower()=="nau":
            t = xr.open_dataset(ap.NAU_NICAM_T)["ms_tem"][ind0:]
        else: raise Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            t = xr.open_dataset(ap.TWP_FV3_T)["temp"][ind0:]
        elif region.lower()=="shl":
            t = xr.open_dataset(ap.TWP_FV3_T)["temp"][ind0:]
        elif region.lower()=="nau":
            t = xr.open_dataset(ap.TWP_FV3_T)["temp"][ind0:]
        else: raise Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="icon":
        if region.lower()=="twp":
            t = xr.open_dataset(ap.TWP_ICON_T)["NEW"][ind0:]
        elif region.lower()=="shl":
            t = xr.open_dataset(ap.SHL_ICON_T) #K
            t = reshape.reshape("T", t, dim=3)[ind0:]
        elif region.lower()=="nau":
            t = xr.open_dataset(ap.NAU_ICON_T)["T"][ind0:]
        else: raise Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            t = xr.open_dataset(ap.TWP_SAM_T)["TABS"][ind0:,:]
        elif region.lower()=="shl":
            t = xr.open_dataset(ap.SHL_SAM_T)["TABS"][ind0:,:]
        elif region.lower()=="nau":
            t = xr.open_dataset(ap.NAU_SAM_T)["TABS"][ind0:,:]
        else: raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    print("\t returned temperature with shape", t.shape)
    return t

def get_qv(model, region):
    """Returns mixing ration in kg/kg for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            qv = xr.open_dataset(ap.TWP_NICAM_QV)["ms_qv"][ind0:]
        elif region.lower()=="shl":
            qv = xr.open_dataset(ap.SHL_NICAM_QV)["ms_qv"][ind0:]
        elif region.lower()=="nau":
            qv = xr.open_dataset(ap.NAU_NICAM_QV)["ms_qv"][ind0:]
        else: raise Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            qv = xr.open_dataset(ap.TWP_FV3_QV)["qv"][ind0:]
        elif region.lower()=="shl":
            qv = xr.open_dataset(ap.SHL_FV3_QV)["qv"][ind0:]
        elif region.lower()=="nau":
            qv = xr.open_dataset(ap.NAU_FV3_QV)["qv"][ind0:]
        else: raise Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="icon":
        if region.lower()=="twp":
            qv = xr.open_dataset(ap.TWP_ICON_QV)["NEW"][ind0:]
        elif region.lower()=="shl":
            qv = xr.open_dataset(ap.SHL_ICON_QV) #kg/kg
            qv = reshape.reshape("QV", qv, dim=3)[ind0:]
        elif region.lower()=="nau":
            qv = xr.open_dataset(ap.NAU_ICON_QV)["QV"][ind0:]
        else: raise Exception("region not valid, try SHL, NAU, or TWP")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            qv = (xr.open_dataset(ap.TWP_SAM_QV)["QV"]/1000)[ind0:,:]
        elif region.lower()=="shl":
            qv = (xr.open_dataset(ap.SHL_SAM_QV)["QV"]/1000)[ind0:,:]
        elif region.lower()=="nau":
            qv = (xr.open_dataset(ap.NAU_SAM_QV)["QV"]/1000)[ind0:,:]
        else: raise Exception("try valid region (SHL, NAU, TWP)")
    else: raise Exception("invalide model: model = SAM, ICON, FV3, NICAM")
    print("\t returned water vapor mixing ratio with shape", qv.shape)
    return qv

def get_twc(model, region):
    """Returns total water content in kg/m3 for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if region.lower()=="twp":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            q = xr.open_dataset(ap.TWP_ICON_TWC)["iwc"] #kg/kg
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            q = xr.open_dataset(ap.TWP_FV3_TWC)["iwc"] #kg/kg
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            q = xr.open_dataset(ap.TWP_SAM_TWC)["iwc"]/1000 #g/kg to kg/kg
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            q = xr.open_dataset(ap.TWP_NICAM_TWC)["twc"]
        else: raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, ICON or SAM for model")
    else: raise Exception("Region ("+region+") is invalid. Try TWP for region.")
    print("Returned total water content (kg/m3) for "+model+" in "+region+" with shape", q.shape)
    return q[ind0:]


### ------------------ load methods -------------------------------- ###
def load_tot_hydro(model, region, ice_only=True):
    """ Returns xarray of the total hydrometeors IWP + LWP for the model and region.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            qi = xr.open_dataset(ap.TWP_NICAM_QI)['ms_qi']
            ql = xr.open_dataset(ap.TWP_NICAM_QL)['ms_qc'].values
            if ice_only:
                return (qi + ql)[ind0:]
            qs = xr.open_dataset(ap.TWP_NICAM_QS)['ms_qs']
            qg = xr.open_dataset(ap.TWP_NICAM_QG)['ms_qg'].values
            qr = xr.open_dataset(ap.TWP_NICAM_QR)['ms_qr'].values
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            qi = xr.open_dataset(ap.NAU_NICAM_QI)['ms_qi']
            ql = xr.open_dataset(ap.NAU_NICAM_QL)['ms_qc'].values
            if ice_only:
                return (qi + ql)[ind0:]
            qs = xr.open_dataset(ap.NAU_NICAM_QS)['ms_qs']
            qg = xr.open_dataset(ap.NAU_NICAM_QG)['ms_qg'].values
            qr = xr.open_dataset(ap.NAU_NICAM_QR)['ms_qr'].values
        elif region.lower()=="shl":
            qi = xr.open_dataset(ap.SHL_NICAM_QI)['ms_qi']
            ql = xr.open_dataset(ap.SHL_NICAM_QL)['ms_qc']
            if ice_only:
                return (qi + ql)[ind0:]
            qs = xr.open_dataset(ap.SHL_NICAM_QS)['ms_qs']
            qg = xr.open_dataset(ap.SHL_NICAM_QG)['ms_qg'].values
            qr = xr.open_dataset(ap.SHL_NICAM_QR)['ms_qr'].values
        else: raise Exception("Region not supported (try TWP, NAU, SHL)")
        if qi.shape!=ql.shape or qi.shape!=qs.shape:
            raise Exception("shapes don't match with shapes", qi.shape, ql.shape, qs.shape, qg.shape, qr.shape)
        q = qi.values + ql + qs + qg + qr
        q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                            coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
        return q_xr[ind0:]
    elif model.lower()=="fv3":
        qi = load_frozen(model, region, ice_only=ice_only)
        ql = load_liq(model, region, rain=ice_only)
        q = qi.values + ql
        qxr = xr.DataArray(q, dims=['time','pfull','grid_yt','grid_xt'], 
                            coords={'time':qi.time, 'pfull':qi.pfull, 'grid_yt':qi.grid_yt, 'grid_xt':qi.grid_xt})
        return qxr
    elif model.lower()=="icon":
        qi = load_frozen(model, region, ice_only=ice_only)
        ql = load_liq(model, region, rain=ice_only)
        q = qi.values + ql
        qxr = xr.DataArray(q, dims=['time','lev','cell'], 
                            coords={'time':qi.t.values, 'lev':qi.lev, 
                                    'cell':qi.cell})
        return qxr
    elif model.lower()=="sam":
        qi = load_frozen(model, region, ice_only=ice_only)
        ql = load_liq(model, region, rain=ice_only)
        q = qi.values + ql
        qxr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                            coords={'time':qi.time, 'lev':qi.z.values, 
                                    'lat':qi.lat, 'lon':qi.lon}, 
                           attrs={'units':'kg/kg'})
        return qxr
    else: raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'ICON', 'SAM')")

def load_tot_hydro1x1(model, region, return_ind=False, iceliq_only=True, exclude_shock=True):
    """ Returns xarray of the total hydrometeors IWP + LWP for the model and region.
    
        region = string of 'TWP' for Tropical Western Pacific - Manus and 
                 returns data for 1x1 deg region over ARM site.
        model  = string of 'FV3', 'ICON', SAM', or 'NICAM' (five of the DYAMOND models)
    """
    st = time.time()
    if exclude_shock: 
        ind0 = 8*2 # exclude first two days
    else:
        ind0 = 0 
    if region.lower()=="twp":
        lat0, lat1, lon0, lon1 = -1,0,147,148
    elif region.lower()=="nau":
        lat0, lat1, lon0, lon1 = -1,0,166,167
    else:
        lat0, lat1, lon0, lon1 = 13,14,2,3
    if model.lower()=="nicam": #NICAM
        print("Getting hydrometeors for NICAM %s"%(region))
        qi = load_frozen(model, region, ice_only=False)
        lat0 = np.argmin(abs(qi.lat.values-lat0))
        lat1 = np.argmin(abs(qi.lat.values-lat1))
        lon0 = np.argmin(abs(qi.lon.values-lon0))
        lon1 = np.argmin(abs(qi.lon.values-lon1))
        print("   time, lev, lat, lon = dims: ",(qi.dims))
        qi = qi[:,:,lat0:lat1,lon0:lon1]
        if region.lower()=="twp":
            ql = xr.open_dataset(ap.TWP_NICAM_QL)['ms_qc'][:,:,lat0:lat1,lon0:lon1]
            qs = xr.open_dataset(ap.TWP_NICAM_QS)['ms_qs'][:,:,lat0:lat1,lon0:lon1]
            qg = xr.open_dataset(ap.TWP_NICAM_QG)['ms_qg'][:,:,lat0:lat1,lon0:lon1]
            qr = xr.open_dataset(ap.TWP_NICAM_QR)['ms_qr'][:,:,lat0:lat1,lon0:lon1]
        elif region.lower()=="shl":
            qi = qi[:,:,lat0:lat1,lon0:lon1]
            ql = xr.open_dataset(ap.SHL_NICAM_QL)['ms_qc'][:,:,lat0:lat1,lon0:lon1]
            qs = xr.open_dataset(ap.SHL_NICAM_QS)['ms_qs'][:,:,lat0:lat1,lon0:lon1]
            qg = xr.open_dataset(ap.SHL_NICAM_QG)['ms_qg'][:,:,lat0:lat1,lon0:lon1]
            qr = xr.open_dataset(ap.SHL_NICAM_QR)['ms_qr'][:,:,lat0:lat1,lon0:lon1]
        elif region.lower()=="nau":
            ql = xr.open_dataset(ap.NAU_NICAM_QL)['ms_qc'][:,:,lat0:lat1,lon0:lon1]
            qs = xr.open_dataset(ap.NAU_NICAM_QS)['ms_qs'][:,:,lat0:lat1,lon0:lon1]
            qg = xr.open_dataset(ap.NAU_NICAM_QG)['ms_qg'][:,:,lat0:lat1,lon0:lon1]
            qr = xr.open_dataset(ap.NAU_NICAM_QR)['ms_qr'][:,:,lat0:lat1,lon0:lon1]
        else: raise Exception("Region not supported (try TWP, NAU or SHL)")
        if qi.shape!=ql.shape or qi.shape!=qs.shape:
            raise Exception("shapes don't match", qi.shape, ql.shape, qs.shape, qg.shape, qr.shape)
        print("    done (%s seconds elapsed)...\n... adding qi, qs, qg, ql, qr..."%str(time.time()-st))
        if iceliq_only:
            q = qi.values + ql
        else:
            q = qi.values + ql + qs + qg + qr
        print("... creating xarray...")
        q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                            coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
        print("    done (%s seconds elapsed)"%str(time.time()-st))
        if return_ind:
            return q_xr[ind0:], (lat0,lat1,lon0,lon1)
        else:
            return q_xr[ind0:]
    elif model.lower()=="fv3": #FV3
        qi = load_frozen(model, region, ice_only=True)
        lat0 = np.argmin(abs(qi.grid_yt.values-lat0))
        lat1 = np.argmin(abs(qi.grid_yt.values-lat1))
        lon0 = np.argmin(abs(qi.grid_xt.values-lon0))
        lon1 = np.argmin(abs(qi.grid_xt.values-lon1))
        print("   time, lev, lat, lon = dims: ",(qi.dims))
        qi = qi[ind0:,:,lat0:lat1,lon0:lon1]
        print("Getting all hydrometeors for FV3 TWP:")
        ql = load_liq(model, region, rain=False).values[:,:,lat0:lat1,lon0:lon1]
        print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
        q = qi.values + ql
        z = get_levels(model, region)
        print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
        print(q.shape, z.shape, qi.time.shape)
        qxr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                           coords={'time':qi.time, 'lev':z, 'lat':qi.grid_yt.values, 'lon':qi.grid_xt.values})
        if return_ind:
            return qxr, (lat0,lat1,lon0,lon1)
        else:
            return qxr
    elif model.lower()=="icon": #ICON
        print("Getting ice...")
        qi = load_frozen(model, region, ice_only=True)
        if region.lower()=="twp":
            print("Getting liquid...")
            ql = xr.open_dataset(ap.TWP_ICON_QL)['NEW']
            coords = pd.read_csv("/home/disk/eos15/jnug/ICON/native/ICON_TWP_10x10_coords.csv", 
                                 names=['lat','lon'])
        elif region.lower()=="shl":
            print("Getting liquid...")
            ql = xr.open_dataset(ap.SHL_ICON_QL)['NEW']
            coords = pd.read_csv("/home/disk/eos15/jnug/ICON/native/ICON_SHL_10x10_coords.csv", 
                                 names=['lat','lon'])
        elif region.lower()=="nau":
            print("Getting liquid...")
            ql = xr.open_dataset(ap.NAU_ICON_QL)['QC_DIA']
            coords = pd.read_csv("/home/disk/eos15/jnug/ICON/native/ICON_NAU_10x10_coords.csv", 
                                 names=['lat','lon'])
        else: raise Exception("region not valid: try SHL, NAU, TWP")
        ind = np.array([False] * len(coords.lat))
        for i in range(len(coords.lat)):
            if coords.lat[i]<lat1 and coords.lat[i]>=lat0:
                if coords.lon[i]>=lon0 and coords.lon[i]<lon1:
                    ind[i] = True
        indrepeat = np.repeat(np.repeat(ind[np.newaxis,np.newaxis,:], qi.shape[0], axis=0), qi.shape[1], axis=1)
        print(indrepeat.shape, qi.shape, np.sum(indrepeat))
        qi = np.where(indrepeat, qi, np.nan)
        ql = np.where(indrepeat, ql, np.nan)
        print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
        q = qi + ql
        print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
        del qi, ql
        if region.lower()=="twp":
            q_da = xr.open_dataset(ap.TWP_ICON_QL)['NEW']
        elif region.lower()=="nau":
            q_da = load_liq(model, region, rain=False)
        elif region.lower()=="shl":
            q_da = xr.open_dataset(ap.SHL_ICON_QL)['NEW']
        qnew = np.zeros((q_da.shape[0], q_da.shape[1], q_da.cell[ind].shape[0]))
        n=int(0)
        for i in range(len(q_da.cell)):
            if ind[i]:
                qnew[:,:,n] = q[:,:,i]
                n += 1
        qxr = xr.DataArray(qnew, dims=['time','lev','cell'],
                           coords={'time':q_da.t.values,'lev':q_da.lev.values, 
                                   'cell':q_da.cell.values[ind]})
        if return_ind:
            return qxr[ind0:], ind
        else:
            return qxr[ind0:]
    elif model.lower()=="sam":
        print("Getting all hydrometeors for SAM %s:"%(region))
        qi = load_frozen(model, region, ice_only=True)
        lat0 = np.argmin(abs(qi.lat.values-lat0))
        lat1 = np.argmin(abs(qi.lat.values-lat1))
        lon0 = np.argmin(abs(qi.lon.values-lon0))
        lon1 = np.argmin(abs(qi.lon.values-lon1))
        print("   time, lev, lat, lon = dims: ",(qi.dims))
        qi = qi[:,:,lat0:lat1,lon0:lon1]
        if region.lower()=="twp":
            ql = xr.open_dataset(ap.TWP_SAM_QL)['QC'][:,:,lat0:lat1,lon0:lon1].values/1000
        elif region.lower()=="shl":
            ql = xr.open_dataset(ap.SHL_SAM_QL)['QC'][:,:,lat0:lat1,lon0:lon1].values/1000
        elif region.lower()=="nau":
            ql = xr.open_dataset(ap.NAU_SAM_QL)['QC'][:,:,lat0:lat1,lon0:lon1].values/1000
        else:
            raise Exception("Region not supported (try 'TWP')")
        print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
        q = qi.values + ql
        print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
        qxr = xr.DataArray(q.astype("float64"), dims=['time','lev','lat','lon'], 
                            coords={'time':qi.time, 'lev':qi.z.values, 
                                    'lat':qi.lat, 'lon':qi.lon}, 
                           attrs={'units':'kg/kg'})
        print("... returned qi + ql as xarray with units of kg/kg")
        if return_ind:
            return qxr[ind0:], (lat0,lat1,lon0,lon1)
        else:
            return qxr[ind0:]
    else: raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'ICON', 'SAM')")


def load_frozen(model, region, ice_only=True):
    """ Returns xarray of frozen hydrometeors.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    st= time.time()
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam":
        if region.lower()=="twp":
            qi = xr.open_dataset(ap.TWP_NICAM_QI)['ms_qi']
            if not(ice_only):
                qs = xr.open_dataset(ap.TWP_NICAM_QS)['ms_qs']
                qg = xr.open_dataset(ap.TWP_NICAM_QG)['ms_qg']
            else: print("    returned only qi (%s seconds elapsed)"%str(time.time()-st))
        elif region.lower()=="shl":
            qi = xr.open_dataset(ap.SHL_NICAM_QI)['ms_qi']
            if not(ice_only):
                qs = xr.open_dataset(ap.SHL_NICAM_QS)['ms_qs']
                qg = xr.open_dataset(ap.SHL_NICAM_QG)['ms_qg']
            else: print("    returned qi only (%s seconds elapsed)"%str(time.time()-st))
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            qi = xr.open_dataset(ap.NAU_NICAM_QI)['ms_qi']
            if not(ice_only):
                qs = xr.open_dataset(ap.NAU_NICAM_QS)['ms_qs']
                qg = xr.open_dataset(ap.NAU_NICAM_QG)['ms_qg']
            else: print("    returned qi only (%s seconds elapsed)"%str(time.time()-st))
        else: raise Exception("Region not supported (try TWP, NAU, SHL)")
        if (ice_only):
            return qi[ind0:]
        else:
            q = qi.values + qs + qg
            del qs,qg
            q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
            print("    returned qi+qs+qg (%s seconds elapsed) for %s %s"%(str(time.time()-st),model, region))
            return q_xr[ind0:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            qi = xr.open_dataset(ap.TWP_FV3_QI)['qi']
        elif region.lower()=="shl":
            qi = xr.open_dataset(ap.SHL_FV3_QI)['qi']
        elif region.lower()=="nau":
            qi = xr.open_dataset(ap.NAU_FV3_QI)['qi']
        else: raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        if ice_only:
            return qi[ind0:]
        else:
            # qi_tot = ice_to_total("FV3", region, qi)
            print("cannot return estimated total frozen hydrometeors, returned qi only")
            return qi[ind0:]
    elif model.lower()=="icon":
        if region.lower()=="twp":
            print("Getting frozen hydrometeors for ICON TWP:")
            qi = xr.open_dataset(ap.TWP_ICON_QI)['NEW'].astype('float32')
        elif region.lower()=="shl":
            print("Getting frozen hydrometeors for ICON SHL:")
            qi = xr.open_dataset(ap.SHL_ICON_QI)["NEW"].astype('float32')
        elif region.lower()=="nau":
            print("Getting frozen hydrometeors for ICON NAU:")
            qi = xr.open_dataset(ap.NAU_ICON_QI)['QI_DIA'].astype('float32')
        else: 
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        if not(ice_only):
            # qi_tot = ice_to_total(model, region, qi)
            print("cannot return estimated total frozen hydrometeors, returned qi only")
            return qi[ind0:]
        else:
            return qi[ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            print("Getting frozen hydrometeors for SAM TWP:")
            qi = ((xr.open_dataset(ap.TWP_SAM_QI)['QI']).astype('float64'))/1000
        elif region.lower()=="nau":
            print("Getting frozen hydrometeors for SAM NAURU:")
            qi = xr.open_dataset(ap.NAU_SAM_QI)['QI']/1000
        elif region.lower()=="shl":
            print("Getting frozen hydrometeors for SAM SAHEL:")
            qi = xr.open_dataset(ap.SHL_SAM_QI)['QI']/1000
        else: 
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        if ice_only:
            return qi[ind0:]
        else:
            # qi_tot = ice_to_total(model, region, qi)
            print("cannot return estimated total frozen hydrometeors, returned qi only")
            return qi[ind0:]
    else: raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'ICON', 'SAM')")
    return


def load_liq(model, region, rain=False):
    """ Returns xarray of the total hydrometeors IWP + LWP for the model and region.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    if INCLUDE_SHOCK: 
        ind0=0
    else:
        ind0 = 8*2 # exclude first two days
    if model.lower()=="nicam": #NICAM
        if region.lower()=="twp":
            print("Getting hydrometeors for NICAM TWP:")
            ql = xr.open_dataset(ap.TWP_NICAM_QL)['ms_qc']
            if rain:
                qr = xr.open_dataset(ap.TWP_NICAM_QR)['ms_qr']
                print("    ... adding rain too ...")
                q = ql.values + qr
                print("... creating xarray...")
                q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':ql.time, 'lev':ql.lev, 'lat':ql.lat, 'lon':ql.lon},
                               attrs={'name':'liquid+rain_water_content','units':'kg/kg'})
                ql = q_xr    
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            print("Getting liquid hydrometeors for NICAM NAURU:")
            ql = xr.open_dataset(ap.NAU_NICAM_QL)['ms_qc']
            if rain:
                qr = xr.open_dataset(ap.NAU_NICAM_QR)['ms_qr']
                print("    ... adding rain too ...")
                q = ql.values + qr
                print("... creating xarray...")
                q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':ql.time, 'lev':ql.lev, 'lat':ql.lat, 'lon':ql.lon},
                               attrs={'name':'liquid+rain_water_content','units':'kg/kg'})
                ql = q_xr
        elif region.lower()=="shl":
            print("Getting liquid hydrometeors for NICAM SAHEL:")
            ql = xr.open_dataset(ap.SHL_NICAM_QL)['ms_qc']
            if rain:
                qr = xr.open_dataset(ap.SHL_NICAM_QR)['ms_qr']
                print("    ... adding rain too ...")
                q = ql.values + qr
                print("... creating xarray...")
                q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':ql.time, 'lev':ql.lev, 'lat':ql.lat, 'lon':ql.lon},
                               attrs={'name':'liquid+rain_water_content','units':'kg/kg'})
                ql = q_xr
        else: raise Exception("Region not supported (try TWP, NAU, SHL)")
        return ql[ind0:]
    elif model.lower()=="fv3": #FV3
        if region.lower()=="twp":
            return (xr.open_dataset(ap.TWP_FV3_QL)['ql'][ind0:])
        elif region.lower()=="nau":
            return (xr.open_dataset(ap.NAU_FV3_QL)['ql'][ind0:])
        elif region.lower()=="shl":
            return (xr.open_dataset(ap.SHL_FV3_QL)['ql'][ind0:])
        else: raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    elif model.lower()=="icon": #ICON
        if region.lower()=="twp":
            return (xr.open_dataset(ap.TWP_ICON_QL)['NEW'][ind0:])
        elif region.lower()=="nau":
            return (xr.open_dataset(ap.NAU_ICON_QL)['QC_DIA'][ind0:])
        elif region.lower()=="shl":
            return (xr.open_dataset(ap.SHL_ICON_QL)['NEW'][ind0:])
        else: raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            return(xr.open_dataset(ap.TWP_SAM_QL)['QC'][ind0:])
        elif region.lower()=="nau":
            return(xr.open_dataset(ap.NAU_SAM_QL)['QC'][ind0:])
        elif region.lower()=="shl":
            return(xr.open_dataset(ap.SHL_SAM_QL)['QC'][ind0:])
        else: raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    else: raise Exception("Model not supported at this time (try 'NICAM', 'FV3',  'ICON', 'SAM')")
    return
