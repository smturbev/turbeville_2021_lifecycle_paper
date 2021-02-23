""" load.py
    author: sami turbeville @smturbev
    date created: 29 june 2020
    
    Loads various variables from FV3, ICON, GEOS, SAM and NICAM for cleaner scripts.
        - get_cre(model, region)
        - get_levels(model, region)
        - get_pr(model, region)
        - get_pres(model, region)
        - get_temp(model, region)
        - get_qv(model, region)
        - get_swd(model, region)
        - get_olr_alb(model, region)
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

NICAM_INCLUDE_SHOCK = False # True uses full time period, False cuts out the first two days

### ------------------ get methods -------------------------------- ###
def get_iwp(model, region, ice_only=True, sam_noise=True, is3d=True):
    """ Return the 2D iwp on native grid. """
    if ice_only:
        print("... Getting iwp for %s in the %s region ..."%(model, region))
    else:
        print("... Getting fwp for %s in the %s region ..."%(model, region))
    if region.lower()=="twp":
        if model.lower()=="nicam":
            print("... returning frozen water path for NICAM.")
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.TWP_NICAM_IWP).sa_cldi)
            else:
                return(xr.open_dataset(ap.TWP_NICAM_IWP).sa_cldi[192:])
        elif model.lower()=="fv3":
            iwp = xr.open_dataset(ap.TWP_FV3_IWP).intqi
            if not(ice_only):
                swp = xr.open_dataset(ap.TWP_FV3_SWP).intqs
                gwp = xr.open_dataset(ap.TWP_FV3_GWP).intqg
                fwp = iwp + swp + gwp
                return fwp
            return iwp
        elif model.lower()=="icon":
            ds = xr.open_dataset(ap.TWP_ICON_IWP)
            iwp = reshape.reshape("TQI_DIA", ds, dim=2)
            print(iwp.shape)
            if not(ice_only):
                swp = reshape.reshape("TQS", ds, dim=2)
                print(swp.shape)
                gwp = reshape.reshape("TQG", ds, dim=2)
                print(gwp.shape)
                fwp = iwp + swp + gwp
                return fwp
            return iwp
        elif model.lower()=="sam":
            if is3d:
                iwp = xr.open_dataset(ap.TWP_SAM_WP_NOISE).IWP
            else:
                iwp = xr.open_dataset(ap.TWP_SAM_IWP).IWP
            print("SAM,", iwp.shape)
            if not(ice_only):
                if sam_noise:
                    swp = xr.open_dataset(ap.TWP_SAM_WP_NOISE).SWP
                    gwp = xr.open_dataset(ap.TWP_SAM_WP_NOISE).GWP
                    fwp = iwp + swp + gwp
                else:
                    swp = xr.open_dataset(ap.TWP_SAM_SWP).SWP
                    gwp = xr.open_dataset(ap.TWP_SAM_GWP).GWP
                    fwp = iwp + swp + gwp
                return fwp
                print("fwp = iwp + swp + gwp")
            return iwp
        return
    elif region.lower()=="shl":
        if model.lower()=="nicam":
            print("... returning frozen water path for NICAM.")
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.SHL_NICAM_IWP).sa_cldi)
            else:
                return(xr.open_dataset(ap.SHL_NICAM_IWP).sa_cldi[192:])
        elif model.lower()=="fv3":
            iwp = xr.open_dataset(ap.SHL_FV3_IWP).intqi
            if not(ice_only):
                swp = xr.open_dataset(ap.SHL_FV3_SWP).intqs
                gwp = xr.open_dataset(ap.SHL_FV3_GWP).intqg
                fwp = iwp + swp + gwp
                return fwp
            return iwp
        elif model.lower()=="icon":
            ds = xr.open_dataset(ap.SHL_ICON_IWP)
            iwp = reshape.reshape("TQI_DIA", ds, dim=2)
            print(iwp.shape)
            if not(ice_only):
                swp = reshape.reshape("TQS", ds, dim=2)
                print(swp.shape)
                gwp = reshape.reshape("TQG", ds, dim=2)
                print(gwp.shape)
                fwp = iwp + swp + gwp
                return fwp
            return iwp
        elif model.lower()=="sam":
            if is3d:
                iwp = xr.open_dataset(ap.SHL_SAM_WP_NOISE).IWP
            else:
                iwp = xr.open_dataset(ap.SHL_SAM_IWP).IWP
            print("SAM,", iwp.shape)
            if not(ice_only):
                if sam_noise:
                    swp = xr.open_dataset(ap.SHL_SAM_WP_NOISE).SWP
                    gwp = xr.open_dataset(ap.SHL_SAM_WP_NOISE).GWP
                    fwp = iwp + swp + gwp
                else:
                    swp = xr.open_dataset(ap.SHL_SAM_SWP).SWP
                    gwp = xr.open_dataset(ap.SHL_SAM_GWP).GWP
                    fwp = iwp + swp + gwp
                return fwp
                print("fwp = iwp + swp + gwp")
            return iwp
        return
    elif region.lower()=="nau":
        if model.lower()=="nicam":
            print("... returning frozen water path - NICAM.")
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.NAU_NICAM_IWP).sa_cldi)
            else:
                return(xr.open_dataset(ap.NAU_NICAM_IWP).sa_cldi[192:])
        elif model.lower()=="fv3":
            iwp = xr.open_dataset(ap.NAU_FV3_IWP).intqi
            if not(ice_only):
                swp = xr.open_dataset(ap.NAU_FV3_SWP).intqs
                gwp = xr.open_dataset(ap.NAU_FV3_GWP).intqg
                fwp = iwp + swp + gwp
                return fwp
            return iwp
        elif model.lower()=="icon":
            ds = xr.open_dataset(ap.NAU_ICON_IWP)
            iwp = reshape.reshape("TQI_DIA", ds, dim=2)
            print(iwp.shape)
            if not(ice_only):
                swp = reshape.reshape("TQS", ds, dim=2)
                print(swp.shape)
                gwp = reshape.reshape("TQG", ds, dim=2)
                print(gwp.shape)
                fwp = iwp + swp + gwp
                return fwp
            return iwp
        elif model.lower()=="sam":
            if is3d:
                iwp = xr.open_dataset(ap.NAU_SAM_WP_NOISE).IWP
            else:
                iwp = xr.open_dataset(ap.NAU_SAM_IWP).IWP
            print("SAM,", iwp.shape)
            if not(ice_only):
                if sam_noise:
                    swp = xr.open_dataset(ap.NAU_SAM_WP_NOISE).SWP
                    gwp = xr.open_dataset(ap.NAU_SAM_WP_NOISE).GWP
                    fwp = iwp + swp + gwp
                else:
                    swp = xr.open_dataset(ap.NAU_SAM_SWP).SWP
                    gwp = xr.open_dataset(ap.NAU_SAM_GWP).GWP
                    fwp = iwp + swp + gwp
                return fwp
                print("fwp = iwp + swp + gwp")
            return iwp
        return
    else:
        raise Exception("models=SAM,ICON,FV3,NICAM; region=TWP,SHL")
    return

def get_lwp(model, region, rain=False, sam_noise=True, is3d=True):
    """ Return the 2D iwp on native grid. """
    if region.lower()=="twp":
        if model.lower()=="nicam":
            print("frozen water path")
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.TWP_NICAM_LWP).sa_cldw)
            else:
                return(xr.open_dataset(ap.TWP_NICAM_LWP).sa_cldw[192:])
        elif model.lower()=="fv3":
            lwp = xr.open_dataset(ap.TWP_FV3_LWP).intql
            if rain:
                rwp = xr.open_dataset(ap.TWP_FV3_RWP).intqr
                fwp = lwp + rwp
                return fwp
            return lwp
        elif model.lower()=="icon":
            ds = xr.open_dataset(ap.TWP_ICON_IWP)
            lwp = reshape.reshape("TQC_DIA", ds, dim=2)
            return lwp
        elif model.lower()=="sam":
            
            if is3d:
                lwp = xr.open_dataset(ap.TWP_SAM_LWP_NOISE).CWP
                if rain:
                    rwp = xr.open_dataset(ap.TWP_SAM_LWP_NOISE).RWP
                    return (lwp + rwp)
            else:
                lwp = xr.open_dataset(ap.TWP_SAM_LWP).CWP
                if rain:
                    rwp = xr.open_dataset(ap.TWP_SAM_RWP).RWP
                    return (lwp + rwp)
            return lwp
        return
    elif region.lower()=="shl":
        if model.lower()=="nicam":
            print("frozen water path")
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.SHL_NICAM_LWP).sa_cldw)
            else:
                return(xr.open_dataset(ap.SHL_NICAM_LWP).sa_cldw[192:])
        elif model.lower()=="fv3":
            lwp = xr.open_dataset(ap.SHL_FV3_LWP).intql
            if rain:
                rwp = xr.open_dataset(ap.SHL_FV3_RWP).intqr
                fwp = lwp + rwp
                return fwp
            return lwp
        elif model.lower()=="icon":
            ds = xr.open_dataset(ap.SHL_ICON_IWP)
            lwp = reshape.reshape("TQC_DIA", ds, dim=2)
            print(lwp.shape)
            return lwp
        elif model.lower()=="sam":
            lwp = xr.open_dataset(ap.SHL_SAM_LWP).CWP
            if sam_ok:
                lwpmax = xr.open_dataset(ap.SAM_CWP_MAX).CWP.values
                lwp_ok = lwp.where((lwp>lwpmax/64000)|(lwp==0))
                lwp = lwp_ok
            if rain:
                rwp = xr.open_dataset(ap.SHL_SAM_RWP).RWP
                if sam_ok:
                    rwpmax = xr.open_dataset(ap.SAM_RWP_MAX).RWP.values
                    rwp_ok = rwp.where((rwp>rwpmax/64000)|(rwp==0))
                    fwp = lwp_ok + rwp_ok
                else:
                    fwp = lwp + rwp
                return fwp
            return lwp
        return
    elif region.lower()=="nau":
        if model.lower()=="nicam":
            print("frozen water path")
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.NAU_NICAM_LWP).sa_cldw)
            else:
                return(xr.open_dataset(ap.NAU_NICAM_LWP).sa_cldw[192:])
        elif model.lower()=="fv3":
            lwp = xr.open_dataset(ap.NAU_FV3_LWP).intql
            if rain:
                rwp = xr.open_dataset(ap.NAU_FV3_RWP).intqr
                fwp = lwp + rwp
                return fwp
            return lwp
        elif model.lower()=="icon":
            ds = xr.open_dataset(ap.NAU_ICON_IWP)
            lwp = reshape.reshape("TQC_DIA", ds, dim=2)
            print(lwp.shape)
            return lwp
        elif model.lower()=="sam":
            lwp = xr.open_dataset(ap.NAU_SAM_LWP).CWP
            if sam_ok:
                lwpmax = xr.open_dataset(ap.SAM_CWP_MAX).CWP.values
                lwp_ok = lwp.where((lwp>lwpmax/64000)|(lwp==0))
                lwp = lwp_ok
            if rain:
                rwp = xr.open_dataset(ap.NAU_SAM_RWP).RWP
                if sam_ok:
                    rwpmax = xr.open_dataset(ap.SAM_RWP_MAX).RWP.values
                    rwp_ok = rwp.where((rwp>rwpmax/64000)|(rwp==0))
                    fwp = lwp_ok + rwp_ok
                else:
                    fwp = lwp + rwp
                return fwp
            return lwp
        return
    else:
        raise Exception("models=SAM,ICON,FV3,NICAM; region=TWP,SHL")
    return

def get_ttliwp(model, region):
    """ Returns the integrated frozen water path in the 14-18km layer
        for specificed model and region. """
    if region.lower()=="twp":
        if model.lower()=="nicam":
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataarray(ap.TWP_NICAM_TTLIWP))
            else:
                return(xr.open_dataarray(ap.TWP_NICAM_TTLIWP)[16:])
        elif model.lower()=="sam":
            return xr.open_dataarray(ap.TWP_SAM_TTLIWP)
        elif model.lower()=="icon":
            return xr.open_dataarray(ap.TWP_ICON_TTLIWP)
        elif model.lower()=="fv3":
            return xr.open_dataarray(ap.TWP_FV3_TTLIWP)
        else:
            raise Exception("Model, %s, not defined. Try NICAM, FV3, ICON, GEOS, or SAM."%model)
    elif region.lower() == "shl":
        if model.lower()=="nicam":
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataarray(ap.SHL_NICAM_TTLIWP))
            else:
                return(xr.open_dataarray(ap.SHL_NICAM_TTLIWP)[16:])
        elif model.lower()=="sam":
            return xr.open_dataarray(ap.SHL_SAM_TTLIWP)
        elif model.lower()=="icon":
            return xr.open_dataarray(ap.SHL_ICON_TTLIWP)
        elif model.lower()=="fv3":
            return xr.open_dataarray(ap.SHL_FV3_TTLIWP)
        else:
            raise Exception("Model, %s, not defined. Try NICAM, FV3, ICON, GEOS, or SAM."%model)
    elif region.lower() == "nau":
        if model.lower()=="nicam":
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataarray(ap.NAU_NICAM_TTLIWP))
            else:
                return(xr.open_dataarray(ap.NAU_NICAM_TTLIWP)[16:])    
        elif model.lower()=="sam":
            return xr.open_dataarray(ap.NAU_SAM_TTLIWP)
        elif model.lower()=="icon":
            return xr.open_dataarray(ap.NAU_ICON_TTLIWP)
        elif model.lower()=="fv3":
            return xr.open_dataarray(ap.NAU_FV3_TTLIWP)
        else:
            raise Exception("Model, %s, not defined. Try NICAM, FV3, ICON, GEOS, or SAM."%model)
    else:
        raise Exception("Region not defined... try NAU, TWP or SHL.")

def get_levels(model, region):
    """Returns numpy array of vertical levels for given model and region."""
    if region.lower()=="twp":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            print("returning all levels - levels for variables are 14-90")
            z = xr.open_dataset(ap.TWP_ICON_Z).HHL.values
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            z = xr.open_dataset(ap.TWP_FV3_Z).altitude.values
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            z = xr.open_dataset(ap.TWP_SAM_QI).z.values
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            z = xr.open_dataset(ap.TWP_NICAM_QI).lev.values 
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="shl":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            z = xr.open_dataset(ap.SHL_ICON_Z).HHL.values 
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            z = xr.open_dataset(ap.SHL_FV3_Z).altitude.values
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            z = xr.open_dataset(ap.SHL_SAM_QI).z.values
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            z = xr.open_dataset(ap.SHL_NICAM_QI).lev.values 
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="nau":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            z = xr.open_dataset(ap.NAU_ICON_Z).HHL.values 
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            z = xr.open_dataset(ap.NAU_FV3_Z).altitude.values
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            z = xr.open_dataset(ap.NAU_SAM_QI).z.values
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            z = xr.open_dataset(ap.NAU_NICAM_QI).lev.values
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    else:
        raise Exception("Region ("+region+") is invalid. Try TWP, NAU or SHL for region.")
    print("Returned Height levels (m) for "+model+" in "+region+" with shape", z.shape)
    return z

def get_pr(model, region):
    """ Returns preciptiation rate in mm/s for given model and region
            Region must be "twp", "nau" or "shl".
    """
    if region.lower()=="twp":
        if model.lower()=="nicam":
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.TWP_NICAM_PR)["ss_tppn"][:,0,:,:])
            else:
                return(xr.open_dataset(ap.TWP_NICAM_PR)["ss_tppn"][192:,0,:,:])
        elif model.lower()=="fv3":
            return xr.open_dataset(ap.TWP_FV3_PR)["pr"]
        elif model.lower()=="icon":
            p = xr.open_dataset(ap.SHL_ICON_PR)["tp"]
            p = util.precip(p,dt=15*60,returnPr=True)
            p = p.where(p>=0)
            return p
        elif model.lower()=="sam":
            p = xr.open_dataset(ap.TWP_SAM_PR)["Precac"]
            p = precip(p,dt=30*60,returnPr=True)
            p = p.where(p>=0)
            p[490,:,:] = np.nan
            return p
        else: 
            raise Exception("Model should be one of: NICAM, FV3, GEOS, ICON, SAM")
    elif region.lower()=="shl":
        if model.lower()=="nicam":
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.SHL_NICAM_PR)["ss_tppn"][:,0,:,:])
            else:
                return(xr.open_dataset(ap.SHL_NICAM_PR)["ss_tppn"][192:,0,:,:])
        elif model.lower()=="fv3":
            return xr.open_dataset(ap.SHL_FV3_PR)["pr"]
        elif model.lower()=="icon":
            p = xr.open_dataset(ap.SHL_ICON_PR)["tp"]
            p = util.precip(p,dt=15*60,returnPr=True)
            p = p.where(p>=0)
            return p
        elif model.lower()=="sam":
            p = xr.open_dataset(ap.SHL_SAM_PR)["Precac"]
            p = precip(p,dt=30*60,returnPr=True)
            p = p.where(p>=0)
            p[1632,:,:] = np.nan
            return p
        else: 
            raise Exception("Model should be one of: NICAM, FV3, GEOS, ICON, SAM")
    elif region.lower()=="nau":
        if model.lower()=="nicam":
            if NICAM_INCLUDE_SHOCK:
                return(xr.open_dataset(ap.NAU_NICAM_PR)["ss_tppn"][:,0,:,:])
            else:
                return(xr.open_dataset(ap.NAU_NICAM_PR)["ss_tppn"][192:,0,:,:])
        elif model.lower()=="fv3":
            return xr.open_dataset(ap.NAU_FV3_PR)["pr"]
        elif model.lower()=="icon":
            print("Need to get all days/timesteps..")
            return #xr.open_dataset(ap.NAU_ICON_PR)[
        elif model.lower()=="sam":
            p = xr.open_dataset(ap.NAU_SAM_PR)["Precac"]
            p = precip(p,dt=30*60,returnPr=True)
            p = p.where(p>=0)
            return p
        else: 
            raise Exception("Model should be one of: NICAM, FV3, GEOS, ICON, SAM")
    return

def get_pres(model, region):
    """Returns pressure in Pascals for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models
    """
    if region.lower()=="twp":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            p = xr.open_dataset(ap.TWP_ICON_P)["NEW"].values #Pa
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            p = xr.open_dataset(ap.TWP_FV3_P)["pres"].values #Pa
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            p = xr.open_dataset(ap.TWP_GEOS_P)["P"].values #Pa
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            p = ((xr.open_dataset(ap.TWP_SAM_P)["p"].values)*100)[:,:] #mb to Pa
            print("shape of SAM p: ", p.shape)
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                p = xr.open_dataset(ap.TWP_NICAM_P)["ms_pres"].values # Pa
            else:
                p = xr.open_dataset(ap.TWP_NICAM_P)["ms_pres"].values[16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="shl":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            p = xr.open_dataset(ap.SHL_ICON_P)["NEW"].values #Pa
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            p = xr.open_dataset(ap.SHL_FV3_P)["pres"].values #Pa
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            p = xr.open_dataset(ap.SHL_GEOS_P)["P"].values #Pa
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            p = (xr.open_dataset(ap.SHL_SAM_P)["p"].values*100)[:,:] #mb to Pa
            print("shape of SAM p:", p.shape)
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                p = xr.open_dataset(ap.SHL_NICAM_P)["ms_pres"].values # Pa
            else:
                p = xr.open_dataset(ap.SHL_NICAM_P)["ms_pres"].values[16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="nau":
        if model.lower()=="icon" or model.lower()=="icon-2.5km":
            p = xr.open_dataset(ap.NAU_ICON_P)["P"].values #Pa
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            p = xr.open_dataset(ap.NAU_FV3_P)["pres"].values #Pa
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            p = xr.open_dataset(ap.NAU_GEOS_P)["P"].values #Pa
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            p = (xr.open_dataset(ap.NAU_SAM_P)["p"].values*100)[:,:] #mb to Pa
            print("shape of SAM p: ", p.shape)
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                p = xr.open_dataset(ap.NAU_NICAM_P)["ms_pres"].values # Pa
            else:
                p = xr.open_dataset(ap.NAU_NICAM_P)["ms_pres"].values[16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    else:
        raise Exception("Region ("+region+") is invalid. Try TWP, NAU or SHL for region.")
    print("Returned Pressure (Pa) for "+model+" in "+region+" with shape", p.shape)
    return p
        
def get_temp(model, region):
    """Returns temperature in Kelvin for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models
    """
    if region.lower()=="twp":
        if model.lower()=="icon" or model.lower()=="icon-2.5km":
            t = xr.open_dataset(ap.TWP_ICON_T)["NEW"].values.astype('float32') #K
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            t = xr.open_dataset(ap.TWP_FV3_T)["temp"].values #K 
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            t = xr.open_dataset(ap.TWP_SAM_T)["TABS"].values # K
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                t = xr.open_dataset(ap.TWP_NICAM_T)["ms_tem"].values # K
            else:
                t = xr.open_dataset(ap.TWP_NICAM_T)["ms_tem"].values[16:]
                
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="shl":
        if model.lower()=="icon" or model.lower()=="icon-2.5km":
            t = xr.open_dataset(ap.SHL_ICON_T) #K
            t = reshape.reshape("T", t, dim=3).values
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            t = xr.open_dataset(ap.SHL_FV3_T)["temp"].values #K 
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            t = xr.open_dataset(ap.SHL_SAM_T)["TABS"].values # K 
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                t = xr.open_dataset(ap.SHL_NICAM_T)["ms_tem"].values # K
            else:
                t = xr.open_dataset(ap.SHL_NICAM_T)["ms_tem"].values[16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="nau":
        if model.lower()=="icon" or model.lower()=="icon-2.5km":
            t = xr.open_dataset(ap.NAU_ICON_T)["T"].values #K
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            t = xr.open_dataset(ap.NAU_FV3_T)["temp"].values #K 
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            t = xr.open_dataset(ap.NAU_SAM_T)["TABS"].values # K 
            print("shape of SAM t: ", t.shape)
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                t = xr.open_dataset(ap.NAU_NICAM_T)["ms_tem"].values # K
            else:
                t = xr.open_dataset(ap.NAU_NICAM_T)["ms_tem"].values[16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    else:
        raise Exception("Region ("+region+") is invalid. Try TWP, NAU or SHL for region.")
    print("Returned Temperature (K) for "+model+" in "+region+" with shape", t.shape)
    return t

def get_qv(model, region):
    """Returns mixing ration in kg/kg for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    if region.lower()=="twp":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            qv = xr.open_dataset(ap.TWP_ICON_QV)["NEW"] #kg/kg
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            qv = xr.open_dataset(ap.TWP_FV3_QV)["qv"] #kg/kg
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            qv = xr.open_dataset(ap.TWP_GEOS_QV)["QV"] #kg/kg
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            qv = xr.open_dataset(ap.TWP_SAM_QV)["QV"]/1000 #g/kg to kg/kg
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                qv = xr.open_dataset(ap.TWP_NICAM_QV)["ms_qv"]
            else:
                qv = xr.open_dataset(ap.TWP_NICAM_QV)["ms_qv"][16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="shl":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            qv = xr.open_dataset(ap.SHL_ICON_QV) #kg/kg
            qv = reshape.reshape("QV", qv, dim=3)
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            qv = xr.open_dataset(ap.SHL_FV3_QV)["qv"] #kg/kg
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            qv = xr.open_dataset(ap.SHL_GEOS_QV)["QV"] #kg/kg
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            qv = xr.open_dataset(ap.SHL_SAM_QV)["QV"]/1000 #g/kg to kg/kg
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                qv = xr.open_dataset(ap.SHL_NICAM_QV)["ms_qv"]
            else:
                qv = xr.open_dataset(ap.SHL_NICAM_QV)["ms_qv"][16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    elif region.lower()=="nau":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            qv = xr.open_dataset(ap.NAU_ICON_QV)["QV"] #kg/kg
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            qv = xr.open_dataset(ap.NAU_FV3_QV)["qv"] #kg/kg
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            qv = xr.open_dataset(ap.NAU_GEOS_QV)["QV"] #kg/kg
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            qv = xr.open_dataset(ap.NAU_SAM_QV)["QV"]/1000 #SAM is saved as g/kg - convert to kg/kg to match
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                qv = xr.open_dataset(ap.NAU_NICAM_QV)["ms_qv"]
            else:
                qv = xr.open_dataset(ap.NAU_NICAM_QV)["ms_qv"][16:]
        else: 
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    else:
        raise Exception("Region ("+region+") is invalid. Try TWP, NAU or SHL for region.")
    print("Returned Mixing Ratio of water vapor (kg/kg) for "+model+" in "+region+" with shape", qv.shape)
    return qv

def get_twc(model, region):
    """Returns total water content in kg/m3 for model and region given.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    if region.lower()=="twp":
        if model.lower()=="icon" or model.lower()=="icon-3.5km":
            q = xr.open_dataset(ap.TWP_ICON_TWC)["iwc"] #kg/kg
        elif model.lower()=="fv3" or model.lower()=="fv3-3.35km":
            q = xr.open_dataset(ap.TWP_FV3_TWC)["iwc"] #kg/kg
        elif model.lower()=="geos" or model.lower()=="geos-3km":
            q = xr.open_dataset(ap.TWP_GEOS_TWC)["twc"] #kg/kg
        elif model.lower()=="sam" or model.lower()=="sam-4km":
            q = xr.open_dataset(ap.TWP_SAM_TWC)["iwc"]/1000 #g/kg to kg/kg
        elif model.lower()=="nicam" or model.lower()=="nicam-3.5km":
            if NICAM_INCLUDE_SHOCK:
                q = xr.open_dataset(ap.TWP_NICAM_TWC)["twc"]
            else:
                q = xr.open_dataset(ap.TWP_NICAM_TWC)["twc"][16:]
        else:
            raise Exception("Model ("+model+") is invalid. Try NICAM, FV3, GEOS, ICON or SAM for model")
    else:
        raise Exception("Region ("+region+") is invalid. Try TWP for region.")
    print("Returned Mixing Ratio of water vapor (kg/m3) for "+model+" in "+region+" with shape", q.shape)
    return q


def get_swd(model, region):
    """ Return swd for models in region.
    
        For models that don't output swd we will use the zonal mean
            to estimate swd for closest latitude.
    """
    # icon has net and upward
    if model.lower()=="icon":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.TWP_ICON_SWN)['ASOB_T']
            swu = xr.open_dataset(ap.TWP_ICON_SWU)['ASOU_T']
        elif region.lower()=="nau":
            rad = xr.open_dataset(ap.NAU_ICON_RAD)
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            swu_un = util.undomean(swu, xy=False)
            swn_un = util.undomean(swn, xy=False)
            swu = xr.DataArray(swu_un, dims=["time","cell"], \
                               coords={"time":s.time.values,\
                                       "cell":swu.cell})
            swn = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":s.time.values,\
                                       "cell":swu.cell})
        elif region.lower()=="shl":
            rad = xr.open_dataset(ap.SHL_ICON_RAD)
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
    elif model.lower()=="nicam":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.TWP_NICAM_SWD)['ss_swd_toa']
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.NAU_NICAM_SWD)['ss_swd_toa']
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.SHL_NICAM_SWD)['ss_swd_toa']
        if not(NICAM_INCLUDE_SHOCK):
            swd = swd[192:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.TWP_FV3_SWD)['fsdt']
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.NAU_FV3_SWD)['fsdt']
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.SHL_FV3_SWD)['fsdt']
    elif model.lower()=="sam":
        print('sam')
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.TWP_SAM_SWN)['SWNTA']
            swd_n = xr.open_dataset(ap.TWP_NICAM_SWD)['ss_swd_toa']
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.NAU_SAM_SWN)['SWNTA']
            swd_n = xr.open_dataset(ap.NAU_NICAM_SWD)['ss_swd_toa']
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.SHL_SAM_SWN)['SWNTA']
            swd_n = xr.open_dataset(ap.SHL_NICAM_SWD)['ss_swd_toa']
        print(swn.shape, swd_n[:1920*2:2,0,:,:].shape)
        swd = np.zeros(swn.shape)
        print('starting loop...')
        for la in range(len(swn.lat)):
            ind = np.argmin(abs(swn.lat.values[la]-swd_n.lat.values))
            swd[:,la,:] = np.repeat((np.nanmean(swd_n[:1920*2:2,0,ind,:],axis=1))[:,np.newaxis], swn.shape[-1], axis=1)
        print('...looping done')
    elif model.lower()=="geos":
        if region.lower()=="twp":
#             swu = xr.open_dataset(ap.TWP_GEOS_SW)["OSR"]
            swd = xr.open_dataset(ap.TWP_GEOS_SW)["RADSWT"]
        elif region.lower()=="nau":
#             swu = xr.open_dataset(ap.NAU_GEOS_SW)["OSR"]
            swd = xr.open_dataset(ap.NAU_GEOS_SW)["RADSWT"]
        elif region.lower()=="shl":
#             swu = xr.open_dataset(ap.SHL_GEOS_SW)["OSR"]
            swd = xr.open_dataset(ap.SHL_GEOS_SW)["RADSWT"]
    return swd

def get_asr(model, region):
    """ Return swd for models in region.
    
        For models that don't output swd we will use the zonal mean
            to estimate swd for closest latitude.
    """
    # icon has net and upward
    if model.lower()=="icon":
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.TWP_ICON_SWN)['ASOB_T']
        elif region.lower()=="nau":
            rad = xr.open_dataset(ap.NAU_ICON_RAD)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            swn_un = util.undomean(swn, xy=False)
            asr = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":swn.t.values,\
                                       "cell":swn.cell})
            del swn, swn_un, rad
        elif region.lower()=="shl":
            rad = xr.open_dataset(ap.SHL_ICON_RAD)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            swn_un = util.undomean(swn, xy=False)
            asr = xr.DataArray(swn_un, dims=["time","cell"], \
                               coords={"time":swn.t.values,\
                                       "cell":swn.cell})
            del swn, swn_un, rad
    elif model.lower()=="nicam":
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
        if not(NICAM_INCLUDE_SHOCK):
            asr = asr[192:]
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
    elif model.lower()=="sam":
        print('sam')
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.TWP_SAM_SWN)['SWNTA']
        elif region.lower()=="nau":
            asr = xr.open_dataset(ap.NAU_SAM_SWN)['SWNTA']
        elif region.lower()=="shl":
            asr = xr.open_dataset(ap.SHL_SAM_SWN)['SWNTA']
        print(swn.shape, swd_n[:1920*2:2,0,:,:].shape)
    elif model.lower()=="geos":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_GEOS_SW)["OSR"]
            swd = xr.open_dataset(ap.TWP_GEOS_SW)["RADSWT"]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.NAU_GEOS_SW)["OSR"]
            swd = xr.open_dataset(ap.NAU_GEOS_SW)["RADSWT"]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.NAU_GEOS_SW)["OSR"]
            swd = xr.open_dataset(ap.NAU_GEOS_SW)["RADSWT"]
        asr = swd - swu
    print(asr.shape)
    return asr

def get_olr_alb(model, region):
    """Returns the (olr, alb) of each model and region as a tuple of xarrays"""
    if model.lower()=="nicam":
        if region.lower()=="twp":
            print("Getting olr and albedo for NICAM TWP:")
            st= time.time()
            olr = xr.open_dataset(ap.TWP_NICAM_OLR)['sa_lwu_toa'][11::12,:,:,:]
            swu = xr.open_dataset(ap.TWP_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
            swd = xr.open_dataset(ap.TWP_NICAM_SWD)['ss_swd_toa'][11::12,:,:,:]
            print("... calculating albedo for shape",olr.shape,swu.shape,swd.shape)
            alb = swu/swd
            del swu, swd
            print("... calculated albedo and opened olr (%s seconds elapsed)..."%str(time.time()-st))
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            print("Getting olr and albedo for NICAM NAURU:")
            st= time.time()
            olr = xr.open_dataset(ap.NAU_NICAM_OLR)['sa_lwu_toa'][11::12,:,:,:]
            swu = xr.open_dataset(ap.NAU_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
            swd = xr.open_dataset(ap.NAU_NICAM_SWD)['ss_swd_toa'][11::12,:,:,:]
            print("... calculating albedo for shape",olr.shape,swu.shape,swd.shape)
            alb = swu/swd
            del swu, swd
            print("... calculated albedo and opened olr (%s seconds elapsed)..."%str(time.time()-st))
        elif region.lower()=="shl":
            print("Getting olr and albedo for NICAM SAHEL:")
            st= time.time()
            olr = xr.open_dataset(ap.SHL_NICAM_OLR)['sa_lwu_toa'][11::12,:,:,:]
            swu = xr.open_dataset(ap.SHL_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
            swd = xr.open_dataset(ap.SHL_NICAM_SWD)['ss_swd_toa'][11::12,:,:,:]
            print("... calculating albedo for shape",olr.shape,swu.shape,swd.shape)
            alb = swu/swd
            del swu, swd
            print("... calculated albedo and opened olr (%s seconds elapsed)..."%str(time.time()-st))
        else: print("Region not supported (try TWP, NAU, SHL)")
        if not(NICAM_INCLUDE_SHOCK):
            olr = olr[192:]
            alb = alb[192:]
        return olr, alb
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            print("Getting olr and albedo for FV3 TWP:")
            olr = xr.open_dataset(ap.TWP_FV3_OLR)["flut"][11::12,:,:]
            swu = xr.open_dataset(ap.TWP_FV3_SWU)["fsut"][11::12,:,:]
            swd = get_swd("FV3", "TWP")[11::12,:,:]
            alb = swu.values/swd
            print(olr.shape, alb.shape)
        elif region.lower()=="nau":
            print("Getting olr and albedo for FV3 NAU:")
            olr = xr.open_dataset(ap.NAU_FV3_OLR)["flut"][11::12,:,:]
            swu = xr.open_dataset(ap.NAU_FV3_SWU)["fsut"][11::12,:,:]
            swd = get_swd("FV3", "NAU")[11::12,:,:]
            alb = swu.values/swd
            print(olr.shape, alb.shape)
        elif region.lower()=="shl":
            print("Getting olr and albedo for FV3 SHL:")
            olr = xr.open_dataset(ap.SHL_FV3_OLR)["flut"][11::12,:,:]
            swu = xr.open_dataset(ap.SHL_FV3_SWU)["fsut"][11::12,:,:]
            swd = get_swd("FV3", "SHL")[11::12,:,:]
            alb = swu.values/swd
            print(olr.shape, alb.shape)
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
        return olr, alb
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
        elif region.lower()=="shl":
            print("Getting olr and albedo for ICON SHL:")
            rad = xr.open_dataset(ap.SHL_ICON_RAD)
            olr = reshape.reshape("ATHB_T", rad, dim=2)
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swn = reshape.reshape("ASOB_T", rad, dim=2)
            dims = olr.dims
            coords = olr.coords
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
        return olr, alb
    elif model.lower()=="sam":
        if region.lower()=="twp":
            print("Getting olr and albedo for SAM TWP:")
            olr = xr.open_dataset(ap.TWP_SAM_OLR)["LWNTA"][5::6,:,:]
            swn = xr.open_dataset(ap.TWP_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "TWP")[5::6,:,:]
            swu = swd - swn
            alb = swu.values/swd
            alb = xr.DataArray(alb, dims=olr.dims, coords=olr.coords, attrs={'long_name':'albedo at TOA (aver)',
                                                                'units':'None'})
            alb = alb.where((alb.values>0)&(swd>0))
            print("mean", alb.mean())
            print(olr.shape, alb.shape)
        elif region.lower()=="nau":
            print("Getting olr and albedo for SAM NAU:")
            olr = xr.open_dataset(ap.NAU_SAM_OLR)["LWNTA"][5::6,:,:]
            swn = xr.open_dataset(ap.NAU_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "NAU")[5::6,:,:]
            swu = swd - swn
            alb = swu.values/swd
            alb = xr.DataArray(alb, dims=olr.dims, coords=olr.coords, attrs={'long_name':'albedo at TOA (aver)',
                                                                'units':'None'})
            print(olr.shape, alb.shape)
        elif region.lower()=="shl":
            print("Getting olr and albedo for SAM SHL:")
            olr = xr.open_dataset(ap.SHL_SAM_OLR)["LWNTA"][5::6,:,:]
            swn = xr.open_dataset(ap.SHL_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "SHL")[5::6,:,:]
            swu = swd - swn
            alb = swu.values/swd
            alb = xr.DataArray(alb, dims=olr.dims, coords=olr.coords, attrs={'long_name':'albedo at TOA (aver)',
                                                                'units':'None'})
            print(olr.shape, alb.shape)
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
        return olr, alb
    else:
        raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'GEOS'/'GEOS5', 'ICON', 'SAM')")
    return

def get_swu(model, region):
    """Returns the (olr, alb) of each model and region as a tuple of xarrays"""
    if model.lower()=="nicam":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            swu = xr.open_dataset(ap.NAU_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.SHL_NICAM_SWU)['ss_swu_toa'][11::12,:,:,:]
        else: print("Region not supported (try TWP, NAU, SHL)")
        if not(NICAM_INCLUDE_SHOCK):
            swu = swu[192:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_FV3_SWU)["fsut"][11::12,:,:]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.NAU_FV3_SWU)["fsut"][11::12,:,:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.SHL_FV3_SWU)["fsut"][11::12,:,:]
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
    elif model.lower()=="icon":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.TWP_ICON_SWU)["ASOU_T"]
        elif region.lower()=="nau":
            rad = xr.open_dataset(ap.NAU_ICON_RAD)
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swu_un = util.undomean(swu, xy=False)
            swu = xr.DataArray(swu_un, dims=["time","cell"], \
                               coords={"time":swu.t.values,\
                                       "cell":swu.cell})
        elif region.lower()=="shl":
            rad = xr.open_dataset(ap.SHL_ICON_RAD)
            swu = reshape.reshape("ASOU_T", rad, dim=2)
            swu_un = util.undomean(swu, xy=False)
            swu = xr.DataArray(swu_un, dims=["time","cell"], \
                               coords={"time":swu.t.values,\
                                       "cell":swu.cell})
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.TWP_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "TWP")[5::6,:,:]
            swu = swd - swn
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.NAU_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "NAU")[5::6,:,:]
            swu = swd - swn
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.SHL_SAM_SWN)["SWNTA"][5::6,:,:]
            swd = get_swd("SAM", "SHL")[5::6,:,:]
            swu = swd - swn
        else: 
            raise Exception("Region not supported. Try 'TWP', 'NAU', 'SHL'.")
    else:
        raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'GEOS'/'GEOS5', 'ICON', 'SAM')")
    return swu


### ------------------ load methods -------------------------------- ###
def load_tot_hydro(model, region, ice_only=True):
    """ Returns xarray of the total hydrometeors IWP + LWP for the model and region.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    st = time.time()
    if model.lower()=="nicam": #NICAM
        if region.lower()=="twp":
            print("Getting hydrometeors for TWP:")
            print("... opening all hydrometeors for NICAM...")
            qi = xr.open_dataset(ap.TWP_NICAM_QI)['ms_qi']
            ql = xr.open_dataset(ap.TWP_NICAM_QL)['ms_qc'].values
            if ice_only:
                return (qi + ql)
            qs = xr.open_dataset(ap.TWP_NICAM_QS)['ms_qs']
            qg = xr.open_dataset(ap.TWP_NICAM_QG)['ms_qg'].values
            qr = xr.open_dataset(ap.TWP_NICAM_QR)['ms_qr'].values
            if qi.shape!=ql.shape or qi.shape!=qs.shape:
                print(qi.shape, ql.shape, qs.shape, qg.shape, qr.shape)
            print("    done (%s seconds elapsed)...\n... adding qi, qs, qg, ql, qr..."%str(time.time()-st))
            q = qi.values + ql + qs + qg + qr
            print("... creating xarray...")
            q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
            print("    done (%s seconds elapsed)"%str(time.time()-st))
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            print("Getting hydrometeors for NAURU:")
            print("... opening all hydrometeors for NICAM...")
            qi = xr.open_dataset(ap.NAU_NICAM_QI)['ms_qi']
            ql = xr.open_dataset(ap.NAU_NICAM_QL)['ms_qc'].values
            if ice_only:
                return (qi + ql)
            qs = xr.open_dataset(ap.NAU_NICAM_QS)['ms_qs']
            qg = xr.open_dataset(ap.NAU_NICAM_QG)['ms_qg'].values
            qr = xr.open_dataset(ap.NAU_NICAM_QR)['ms_qr'].values
            if qi.shape!=ql.shape or qi.shape!=qs.shape:
                print(qi.shape, ql.shape, qs.shape, qg.shape, qr.shape)
                return
            print("    done (%s seconds elapsed)..."%str(time.time()-st))
            q = qi.values + ql + qs + qg + qr
            print("... creating xarray...")
            q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
            print("    done (%s seconds elapsed)"%str(time.time()-st))
        elif region.lower()=="shl":
            print("Getting hydrometeors for SAHEL:")
            print("... opening all hydrometeors for NICAM...")
            qi = xr.open_dataset(ap.SHL_NICAM_QI)['ms_qi']
            ql = xr.open_dataset(ap.SHL_NICAM_QL)['ms_qc']
            if ice_only:
                return (qi + ql)
            qs = xr.open_dataset(ap.SHL_NICAM_QS)['ms_qs']
            qg = xr.open_dataset(ap.SHL_NICAM_QG)['ms_qg']
            qr = xr.open_dataset(ap.SHL_NICAM_QR)['ms_qr']
            print("    done (%s seconds elapsed)..."%str(time.time()-st))
            if qi.shape!=ql.shape or qi.shape!=qs.shape:
                print(qi.shape, ql.shape, qs.shape, qg.shape, qr.shape)
            q = qi.values + ql + qs + qg + qr
            print("... creating xarray...")
            q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
            print("    done (%s seconds elapsed)"%str(time.time()-st))
        else: print("Region not supported (try TWP, NAU, SHL)")
        if not(NICAM_INCLUDE_SHOCK):
            q_xr = q_xr[16:]
        return q_xr
    elif model.lower()=="fv3": #FV3
        if region.lower()=="twp":
            print("Getting all hydrometeors for FV3 TWP:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.TWP_FV3_QL)['ql']
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','pfull','grid_yt','grid_xt'], 
                                coords={'time':qi.time, 'pfull':qi.pfull, 'grid_yt':qi.grid_yt, 'grid_xt':qi.grid_xt})
            return qxr
        elif region.lower()=="nau":
            print("Getting all hydrometeors for FV3 NAU:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.NAU_FV3_QL)['ql']
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','pfull','grid_yt','grid_xt'], 
                                coords={'time':qi.time, 'pfull':qi.pfull, 'grid_yt':qi.grid_yt, 'grid_xt':qi.grid_xt})
            return qxr
        elif region.lower()=="shl":
            print("Getting all hydrometeors for FV3 SHL:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.SHL_FV3_QL)['ql']
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','pfull','grid_yt','grid_xt'], 
                                coords={'time':qi.time, 'pfull':qi.pfull, 'grid_yt':qi.grid_yt, 'grid_xt':qi.grid_xt})
            return qxr
        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    elif model.lower()=="icon": #ICON
        if region.lower()=="twp":
            print("Getting all hydrometeors for ICON TWP:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.TWP_ICON_QL)['NEW']
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','lev','cell'], 
                                coords={'time':qi.t.values, 'lev':qi.lev, 
                                        'cell':qi.cell})
            return qxr
        elif region.lower()=="nau":
            print("Getting all hydrometeors for ICON NAU:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.NAU_ICON_QL)['QC_DIA']
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','lev','cell'], 
                                coords={'time':qi.t.values, 'lev':qi.lev, 
                                        'cell':qi.cell})
            return qxr
        elif region.lower()=="shl":
            print("Getting all hydrometeors for ICON SHL:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.SHL_ICON_QL)["TQC_DIA"]
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','lev','cell'], 
                                coords={'time':qi.t.values, 'lev':qi.lev.values, 
                                        'cell':qi.cell})
            return qxr
        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            print("Getting all hydrometeors for SAM TWP:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.TWP_SAM_QL)['QC'].values.astype("float64")/1000
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.z.values, 
                                        'lat':qi.lat, 'lon':qi.lon}, 
                               attrs={'units':'kg/kg'})
            print("... returned qi + ql as xarray with units of kg/kg")
            return qxr
        elif region.lower()=="nau":
            print("Getting all hydrometeors for SAM NAURU:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.NAU_SAM_QL)['QC'].values.astype("float64")/1000
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.z.values, 
                                        'lat':qi.lat, 'lon':qi.lon}, 
                               attrs={'units':'kg/kg'})
            print("... returned qi + ql as xarray with units of kg/kg")
            return qxr
        elif region.lower()=="shl":
            print("Getting all hydrometeors for SAM SAHEL:")
            qi = load_frozen(model, region, ice_only=ice_only)
            ql = xr.open_dataset(ap.SHL_SAM_QL)['QC'].values.astype("float64")/1000
            print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
            q = qi.values + ql
            print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
            qxr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                coords={'time':qi.time, 'lev':qi.z.values, 
                                        'lat':qi.lat, 'lon':qi.lon}, 
                               attrs={'units':'kg/kg'})
            print("... returned qi + ql as xarray with units of kg/kg")
            return qxr
        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    else:
        raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'GEOS'/'GEOS5', 'ICON', 'SAM')")

def load_frozen(model, region, ice_only=True):
    """ Returns xarray of frozen hydrometeors.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    st= time.time()
    if model.lower()=="nicam":
        if region.lower()=="twp":
            print("Getting frozen hydrometeors for TWP:")
            print("... opening all hydrometeors for NICAM...")
            qi = xr.open_dataset(ap.TWP_NICAM_QI)['ms_qi']
            if not(ice_only):
                qs = xr.open_dataset(ap.TWP_NICAM_QS)['ms_qs']
                qg = xr.open_dataset(ap.TWP_NICAM_QG)['ms_qg']
                print("    done (%s seconds elapsed)..."%str(time.time()-st))
                q = qi.values + qs + qg
                del qs,qg
                print("... creating xarray...")
                q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                    coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
                print("    returned qi+qs+qg (%s seconds elapsed)"%str(time.time()-st))
                qi = q_xr
                del q_xr
            else:
                print("    returned only qi (%s seconds elapsed)"%str(time.time()-st))
        elif (region.lower()=="nau") or (region.lower()=="nauru"):
            print("Getting frozen hydrometeors for NAURU:")
            print("... opening all hydrometeors for nicam...")
            qi = xr.open_dataset(ap.NAU_NICAM_QI)['ms_qi']
            if not(ice_only):
                qs = xr.open_dataset(ap.NAU_NICAM_QS)['ms_qs']
                qg = xr.open_dataset(ap.NAU_NICAM_QG)['ms_qg']
                print("    done (%s seconds elapsed)..."%str(time.time()-st))
                q = qi.values + qs + qg
                del qs,qg
                print("... creating xarray...")
                q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                    coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
                print("    returned qi+qs+qg (%s seconds elapsed)"%str(time.time()-st))
                qi = q_xr
                del q_xr
            else:
                print("    returned qi only (%s seconds elapsed)"%str(time.time()-st))
        elif region.lower()=="shl":
            print("Getting frozen hydrometeors for SAHEL:")
            print("... opening all hydrometeors for NICAM...")
            qi = xr.open_dataset(ap.SHL_NICAM_QI)['ms_qi']
            if not(ice_only):
                qs = xr.open_dataset(ap.SHL_NICAM_QS)['ms_qs']
                qg = xr.open_dataset(ap.SHL_NICAM_QG)['ms_qg']
                print("    done  with qs+qg+qi (%s seconds elapsed)..."%str(time.time()-st))
                q = qi.values + qs + qg
                del qs,qg
                print("... creating xarray...")
                q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                                    coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
                print("    returned qi+qs+qg (%s seconds elapsed)"%str(time.time()-st))
                qi = q_xr
                del q_xr
            else:
                print("    returned qi only (%s seconds elapsed)"%str(time.time()-st))
        else:
            raise Exception("Region not supported (try TWP, NAU, SHL)")
        if not(NICAM_INCLUDE_SHOCK):
            qi = qi[16:]
        return qi
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            print("Getting frozen hydrometeors for TWP:")
            print("... opening all hydrometeors for FV3...")
            qi = xr.open_dataset(ap.TWP_FV3_QI)['qi']
        elif region.lower()=="nau":
            print("Getting frozen hydrometeors for NAU:")
            print("... opening all hydrometeors for FV3...")
            qi = xr.open_dataset(ap.NAU_FV3_QI)['qi']
        elif region.lower()=="shl":
            print("Getting frozen hydrometeors for SHL:")
            print("... opening all hydrometeors for FV3...")
            qi = xr.open_dataset(ap.SHL_FV3_QI)['qi']
        else: 
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        if not(ice_only):
            qi_tot = ice_to_total("FV3", region, qi)
            print("returned estimated total frozen hydrometeors")
            return qi_tot
        else:
            return qi
    elif model.lower()=="icon":
        if region.lower()=="twp":
            print("Getting frozen hydrometeors for TWP:")
            print("... opening all hydrometeors for ICON...")
            qi = xr.open_dataset(ap.TWP_ICON_QI)['NEW'].astype('float32')
        elif region.lower()=="nau":
            print("Getting frozen hydrometeors for NAU:")
            print("... opening all hydrometeors for ICON...")
            qi = xr.open_dataset(ap.NAU_ICON_QI)['QI_DIA'].astype('float32')
        elif region.lower()=="shl":
            print("Getting frozen hydrometeors for SHL:")
            print("... opening all hydrometeors for ICON...")
            qi = xr.open_dataset(ap.SHL_ICON_QI)["NEW"].astype('float32')
        else: 
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        if not(ice_only):
            qi_tot = ice_to_total(model, region, qi)
            print("returned estimated total frozen hydrometeors")
            return qi_tot
        else:
            return qi
    elif model.lower()=="sam":
        if region.lower()=="twp":
            print("Getting frozen hydrometeors for TWP:")
            print("... opening all hydrometeors for SAM...")
            qi = ((xr.open_dataset(ap.TWP_SAM_QI)['QI']).astype('float64'))/1000
        elif region.lower()=="nau":
            print("Getting frozen hydrometeors for NAURU:")
            print("... opening all hydrometeors for SAM...")
            qi = xr.open_dataset(ap.NAU_SAM_QI)['QI']/1000
        elif region.lower()=="shl":
            print("Getting frozen hydrometeors for SAHEL:")
            print("... opening all hydrometeors for SAM...")
            qi = xr.open_dataset(ap.SHL_SAM_QI)['QI']/1000
        else: 
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        if not(ice_only):
            qi_tot = ice_to_total(model, region, qi)
            print("returned estimated total frozen hydrometeors")
            return qi_tot
        else:
            return qi
    else:
        raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'GEOS'/'GEOS5', 'ICON', 'SAM')")
    return

def load_tot_hydro1x1(model, region, return_ind=False, iceliq_only=True):
    """ Returns xarray of the total hydrometeors IWP + LWP for the model and region.
    
        region = string of 'TWP' for Tropical Western Pacific - Manus and 
                 returns data for 1x1 deg region over ARM site.
        model  = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    st = time.time()
    if region.lower()=="twp":
        lat0, lat1, lon0, lon1 = -1,0,147,148
    elif region.lower()=="nau":
        lat0, lat1, lon0, lon1 = -1,0,166,167
    else:
        lat0, lat1, lon0, lon1 = 13,14,2,3
    if model.lower()=="nicam": #NICAM
        print("Getting hydrometeors for %s:"%(region))
        print("... opening all hydrometeors for NICAM...")
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
        elif region.lower()=="nau":
            ql = xr.open_dataset(ap.NAU_NICAM_QL)['ms_qc'][:,:,lat0:lat1,lon0:lon1]
            qs = xr.open_dataset(ap.NAU_NICAM_QS)['ms_qs'][:,:,lat0:lat1,lon0:lon1]
            qg = xr.open_dataset(ap.NAU_NICAM_QG)['ms_qg'][:,:,lat0:lat1,lon0:lon1]
            qr = xr.open_dataset(ap.NAU_NICAM_QR)['ms_qr'][:,:,lat0:lat1,lon0:lon1]
        elif region.lower()=="shl":
            qi = qi[:,:,lat0:lat1,lon0:lon1]
            ql = xr.open_dataset(ap.SHL_NICAM_QL)['ms_qc'][:,:,lat0:lat1,lon0:lon1]
            qs = xr.open_dataset(ap.SHL_NICAM_QS)['ms_qs'][:,:,lat0:lat1,lon0:lon1]
            qg = xr.open_dataset(ap.SHL_NICAM_QG)['ms_qg'][:,:,lat0:lat1,lon0:lon1]
            qr = xr.open_dataset(ap.SHL_NICAM_QR)['ms_qr'][:,:,lat0:lat1,lon0:lon1]
        else: raise Exception("Region not supported (try TWP)")
        if qi.shape!=ql.shape or qi.shape!=qs.shape:
            print(qi.shape, ql.shape, qs.shape, qg.shape, qr.shape)
        print("    done (%s seconds elapsed)...\n... adding qi, qs, qg, ql, qr..."%str(time.time()-st))
        if iceliq_only:
            q = qi.values+ql
        else:
            q = qi.values + ql + qs + qg + qr
        print("... creating xarray...")
        q_xr = xr.DataArray(q, dims=['time','lev','lat','lon'], 
                            coords={'time':qi.time, 'lev':qi.lev, 'lat':qi.lat, 'lon':qi.lon})
        print("    done (%s seconds elapsed)"%str(time.time()-st))
        if not(NICAM_INCLUDE_SHOCK):
            q_xr = q_xr[16:]
        if return_ind:
            return q_xr, (lat0,lat1,lon0,lon1)
        else:
            return q_xr
    elif model.lower()=="fv3": #FV3
        qi = load_frozen(model, region, ice_only=True)
        lat0 = np.argmin(abs(qi.grid_yt.values-lat0))
        lat1 = np.argmin(abs(qi.grid_yt.values-lat1))
        lon0 = np.argmin(abs(qi.grid_xt.values-lon0))
        lon1 = np.argmin(abs(qi.grid_xt.values-lon1))
        print("   time, lev, lat, lon = dims: ",(qi.dims))
        qi = qi[:,:,lat0:lat1,lon0:lon1]
        print("Getting all hydrometeors for FV3 TWP:")
        if region.lower()=="twp":
            ql = xr.open_dataset(ap.TWP_FV3_QL)['ql'].values[:,:,lat0:lat1,lon0:lon1]
        elif region.lower()=="nau":
            ql = xr.open_dataset(ap.NAU_FV3_QL)['ql'].values[:,:,lat0:lat1,lon0:lon1]
        elif region.lower()=="shl":
            ql = xr.open_dataset(ap.SHL_FV3_QLf)['ql'].values[:,:,lat0:lat1,lon0:lon1]
        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
        print("... opened qi and ql (%s s elapsed)..."%(time.time()-st))
        q = qi.values + ql
        z = np.nanmean(get_levels(model, region), axis=0)
        print("... added qi + ql (%s s elapsed)..."%(time.time()-st))
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
        elif region.lower()=="nau":
            print("Getting liquid...")
            ql = load_liq(model, region, rain=False)
            coords = pd.read_csv("/home/disk/eos15/jnug/ICON/native/ICON_NAU_10x10_coords.csv", 
                                 names=['lat','lon'])
        else:
            print("Getting liquid...")
            ql = xr.open_dataset(ap.SHL_ICON_QL)['NEW']
            coords = pd.read_csv("/home/disk/eos15/jnug/ICON/native/ICON_SHL_10x10_coords.csv", 
                                 names=['lat','lon'])
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
        n=0
        for i in range(len(q_da.cell)):
            if ind[i]:
                qnew[:,:,n] = q[:,:,i]
                n += 1
        qxr = xr.DataArray(qnew, dims=['time','lev','cell'],
                           coords={'time':q_da.t.values,'lev':q_da.lev.values, 
                                   'cell':q_da.cell.values[ind]})
        if return_ind:
            return qxr, ind
        else:
            return qxr
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
            return qxr, (lat0,lat1,lon0,lon1)
        else:
            return qxr
    else:
        raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'GEOS'/'GEOS5', 'ICON', 'SAM')")


def load_liq(model, region, rain=False):
    """ Returns xarray of the total hydrometeors IWP + LWP for the model and region.
    
        model = string of 'TWP', 'SHL' or 'NAU' for Tropical Western Pacific - Manus,
            Sahel - Niamey or Nauru
        region = string of 'FV3', 'ICON', 'GEOS', 'SAM', or 'NICAM' (five of the DYAMOND models)
    """
    st = time.time()
    if model.lower()=="nicam": #NICAM
        if region.lower()=="twp":
            print("Getting hydrometeors for TWP:")
            print("... opening liquid hydrometeors for nicam...")
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
            print("Getting liquid hydrometeors for NAURU:")
            print("... opening all hydrometeors for NICAM...")
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
            print("Getting liquid hydrometeors for SAHEL:")
            print("... opening all hydrometeors for NICAM...")
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
        else: print("Region not supported (try TWP, NAU, SHL)")
        if not(NICAM_INCLUDE_SHOCK):
            ql = ql[16:]
        return ql
    elif model.lower()=="fv3": #FV3
        if region.lower()=="twp":
            return (xr.open_dataset(ap.TWP_FV3_QL)['ql'][:])
        elif region.lower()=="nau":
            return (xr.open_dataset(ap.NAU_FV3_QL)['ql'][:])
        elif region.lower()=="shl":
            return (xr.open_dataset(ap.SHL_FV3_QL)['ql'][:])
        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    elif model.lower()=="icon": #ICON
        if region.lower()=="twp":
            return (xr.open_dataset(ap.TWP_ICON_QL)['NEW'][:])
        elif region.lower()=="nau":
            return (xr.open_dataset(ap.NAU_ICON_QL)['QC_DIA'][:])
        elif region.lower()=="shl":
            return (xr.open_dataset(ap.SHL_ICON_QL)['NEW'][:])
        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    elif model.lower()=="sam":
        if region.lower()=="twp":
            return(xr.open_dataset(ap.TWP_SAM_QL)['QC'])
        elif region.lower()=="nau":
            return(xr.open_dataset(ap.NAU_SAM_QL)['QC'])
        elif region.lower()=="shl":
            return(xr.open_dataset(ap.SHL_SAM_QL)['QC'])

        else:
            raise Exception("Region not supported (try 'TWP', 'NAU', 'SHL')")
    else:
        raise Exception("Model not supported at this time (try 'NICAM', 'FV3', 'GEOS'/'GEOS5', 'ICON', 'SAM')")