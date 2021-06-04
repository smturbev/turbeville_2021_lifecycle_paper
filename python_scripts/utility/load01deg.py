#%%
""" load01dege.py
    author: sami turbeville @smturbev
    date created: 22 July 2020
    
    Loads various variables from FV3, ICON, GEOS, SAM and NICAM for cleaner scripts.
        - get_asr(model, region)
        - get_swu(model, region)
        - get_swd(model, region)
        - get_olr(model, region)
        - get_iwp(model, region, ice_only=True)

        
"""

import xarray as xr
import numpy as np
import sys
from . import analysis_parameters as ap
from . import util

INCLUDE_SHOCK = False # True uses full time period, False cuts out the first two days

def get_cccm(region):
    """Returns CCCM dataset as xarray for specified region."""
    if region.lower()=="twp":
        return xr.open_dataset(ap.CERES_TWP)
    elif region.lower()=="shl":
        return xr.open_dataset(ap.CERES_SHL)
    else:
        return xr.open_dataset(ap.CERES_NAU)

def get_dardar(region):
    """Returns xarray dataset for specified region of DARDAR"""
    if region.lower()=="twp":
        return xr.open_dataset(ap.DARDAR_TWP)
    elif region.lower()=="shl":
        return xr.open_dataset(ap.DARDAR_SHL)
    else:
        return xr.open_dataset(ap.DARDAR_NAU)

def get_asr(model, region):
    """ Return swd for models in region.
    
        For models that don't output swd we will use the zonal mean
            to estimate swd for closest latitude.
    """
    ind0=0 if INCLUDE_SHOCK else 96*2 # exclude first two days
    if model.lower()=="icon":
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.ALL_TWP_ICON_SWN)['ASOB_T'][:]
        elif region.lower()=="nau":
            asr = xr.open_dataset(ap.ALL_NAU_ICON_SWN)['ASOB_T'][:]
        elif region.lower()=="shl":
            asr = xr.open_dataset(ap.ALL_SHL_ICON_SWN)['ASOB_T'][:]
        asr = util.undomean(asr)[ind0:]
    elif model.lower()=="nicam":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_NICAM_SWD)['ss_swd_toa'][ind0:]
            swu = xr.open_dataset(ap.ALL_TWP_NICAM_SWU)['ss_swu_toa'][ind0:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_NICAM_SWD)['ss_swd_toa'][ind0:]
            swu = xr.open_dataset(ap.ALL_NAU_NICAM_SWU)['ss_swu_toa'][ind0:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_NICAM_SWD)['ss_swd_toa'][ind0:]
            swu = xr.open_dataset(ap.ALL_SHL_NICAM_SWU)['ss_swu_toa'][ind0:]
        asr = swd[:,0,:,:] - swu[:,0,:,:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][:]
            swu = xr.open_dataset(ap.ALL_TWP_FV3_SWU)['fsut'][:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][:]
            swu = xr.open_dataset(ap.ALL_NAU_FV3_SWU)['fsut'][:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][:]
            swu = xr.open_dataset(ap.ALL_SHL_FV3_SWU)['fsut'][:]
        asr = (swd - swu)[ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.ALL_TWP_SAM_SWN)['SWNTA'][ind0//2:]
        elif region.lower()=="nau":
            asr = xr.open_dataset(ap.ALL_NAU_SAM_SWN)['SWNTA'][ind0//2:]
        elif region.lower()=="shl":
            asr = xr.open_dataset(ap.ALL_SHL_SAM_SWN)['SWNTA'][ind0//2:]
    elif model.lower()=="mpas":
        if region.lower()=="twp":
            cur = xr.open_dataset(ap.ALL_TWP_MPAS_SWN)['acswnett'][:]
        elif region.lower()=="nau":
            cur = xr.open_dataset(ap.ALL_NAU_MPAS_SWN)['acswnett'][:]
        elif region.lower()=="shl":
            cur = xr.open_dataset(ap.ALL_SHL_MPAS_SWN)['acswnett'][:]
        asr = np.zeros(cur.shape)
        for t in range(len(cur.xtime)-1):
            asr[t] = (cur[t+1,:,:].values - cur[t,:,:].values)/900
        asr = xr.DataArray(asr, dims=cur.dims, coords=cur.coords)
        asr[0] = np.nan
        asr = asr[ind0:]
    elif model.lower()=="arp":
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.ALL_TWP_ARP_SWN)['nswrf'][ind0:,:,:]/900
        elif region.lower()=="nau":
            asr = xr.open_dataset(ap.ALL_NAU_ARP_SWN)['nswrf'][ind0:,:,:]/900
        elif region.lower()=="shl":
            asr = xr.open_dataset(ap.ALL_SHL_ARP_SWN)['nswrf'][ind0:,:,:]/900
    elif model.lower()=="um":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.ALL_TWP_UM_SWU)['toa_outgoing_shortwave_flux'][:]
            swd = xr.open_dataset(ap.ALL_TWP_UM_SWD)['toa_incoming_shortwave_flux'][:]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.ALL_NAU_UM_SWU)['toa_outgoing_shortwave_flux'][:]
            swd = xr.open_dataset(ap.ALL_NAU_UM_SWD)['toa_incoming_shortwave_flux'][:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.ALL_SHL_UM_SWU)['toa_outgoing_shortwave_flux'][:]
            swd = xr.open_dataset(ap.ALL_SHL_UM_SWD)['toa_incoming_shortwave_flux'][:]
        asr = (swd - swu)[ind0:]
    elif model.lower()=="ecmwf":
        if region.lower()=="twp":
            asr = xr.open_dataset(ap.ALL_TWP_ECMWF_SWN)["tsr"][ind0//4:,:,:]/3600
        elif region.lower()=="nau":
            asr = xr.open_dataset(ap.ALL_NAU_ECMWF_SWN)['tsr'][ind0//4:,:,:]/3600
        elif region.lower()=="shl":
            asr = xr.open_dataset(ap.ALL_SHL_ECMWF_SWN)['tsr'][ind0//4:,:,:]/3600
    print("Returned ASR for "+model+" ("+region+") with shape:", asr.shape)
    return asr

def get_swu(model, region):
    """ Return swu for models in region.
    
        For models that don't output swd we will use the zonal mean
            to estimate swd for closest latitude.
    """
    ind0=0 if INCLUDE_SHOCK else 96*2 # exclude first two days
    if model.lower()=="icon":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.ALL_TWP_ICON_SWU)['ASOU_T'][:]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.ALL_NAU_ICON_SWU)['ASOU_T'][:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.ALL_SHL_ICON_SWU)['ASOU_T'][:]
        swu_undone = util.undomean(swu)
        swu = xr.DataArray(swu_undone, dims=swu.dims, coords=swu.coords, attrs={'name':'SWU_mean_undone_%s'%region})
        swu = swu[ind0:]
    elif model.lower()=="nicam":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.ALL_TWP_NICAM_SWU)['ss_swu_toa'][:]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.ALL_NAU_NICAM_SWU)['ss_swu_toa'][:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.ALL_SHL_NICAM_SWU)['ss_swu_toa'][:]
        swu = swu[ind0:,0,:,:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.ALL_TWP_FV3_SWU)['fsut'][ind0:]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.ALL_NAU_FV3_SWU)['fsut'][ind0:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.ALL_SHL_FV3_SWU)['fsut'][ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.ALL_TWP_SAM_SWN)['SWNTA'][:]
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][1:1921*2:2].values
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.ALL_NAU_SAM_SWN)['SWNTA'][:]
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][1:1921*2:2].values
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.ALL_SHL_SAM_SWN)['SWNTA'][:]
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][1:1921*2:2].values
        swu = swd - swn
        swu = swu.where(swd>0,0)
        swu = swu.where(swu>0,0)
        swu = swu[ind0//2:]
    elif model.lower()=="mpas":
        if region.lower()=="twp":
            cur = xr.open_dataset(ap.ALL_TWP_MPAS_SWN)['acswnett'][:3840]
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][:].values
        elif region.lower()=="nau":
            cur = xr.open_dataset(ap.ALL_NAU_MPAS_SWN)['acswnett'][:3840]
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][:3840].values
        elif region.lower()=="shl":
            cur = xr.open_dataset(ap.ALL_SHL_MPAS_SWN)['acswnett'][:3840]
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][:3840].values
        swn = np.zeros(cur.shape)
        for t in range(len(cur.xtime)-2):
            swn[t+1] = (cur[t+2,:,:].values - cur[t+1,:,:].values)/900
        swn = xr.DataArray(swn, dims=cur.dims, coords=cur.coords)
        print(swn.shape, swd.shape)
        swu = swd - swn
        swu[0,:,:] = np.nan
        swu[-1,:,:]= np.nan
        swu = swu.where(swd>0,0)
        swu = swu.where(swu>0,0)
        swu = swu[ind0:]
    elif model.lower()=="arp":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.ALL_TWP_ARP_SWN)['nswrf'][:,:,:]/900
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][:3840].values
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.ALL_NAU_ARP_SWN)['nswrf'][:,:,:]/900
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][:3840].values
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.ALL_SHL_ARP_SWN)['nswrf'][:,:,:]/900
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][:3840].values
        swu = swd - swn
        swu = swu.where(swd>0,0)
        swu = swu.where(swu>0,0)
        swu = swu[ind0:]
    elif model.lower()=="um":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.ALL_TWP_UM_SWU)['toa_outgoing_shortwave_flux'][ind0:]
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.ALL_NAU_UM_SWU)['toa_outgoing_shortwave_flux'][ind0:]
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.ALL_SHL_UM_SWU)['toa_outgoing_shortwave_flux'][ind0:]
    elif model.lower()=="ecmwf":
        if region.lower()=="twp":
            swn = xr.open_dataset(ap.ALL_TWP_ECMWF_SWN)["tsr"][:,:,:]/3600
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][3:960*4:4].values
        elif region.lower()=="nau":
            swn = xr.open_dataset(ap.ALL_NAU_ECMWF_SWN)['tsr'][:,:,:]/3600
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][3:960*4:4].values
        elif region.lower()=="shl":
            swn = xr.open_dataset(ap.ALL_SHL_ECMWF_SWN)['tsr'][:,:,:]/3600
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][3:960*4:4].values
        swu = swd - swn
        swu = swu.where(swd>0,0)
        swu = swu.where(swu>0,0)
        swu = swu[ind0//4:]
    else: 
        raise Exception("region ({}) or model ({}) input error".format(region, model))
    print("Returned SWU for "+model+" ("+region+") with shape:", swu.shape)
    return swu

def get_swd(model, region):
    """ Return swd for models in region.
    
        For models that don't output swd we will use the zonal mean
            to estimate swd for closest latitude.
    """
    ind0=0 if INCLUDE_SHOCK else 96*2 # exclude first two days
    if model.lower()=="icon":
        if region.lower()=="twp":
            swu = xr.open_dataset(ap.ALL_TWP_ICON_SWU)['ASOU_T'][:]
            swn = xr.open_dataset(ap.ALL_TWP_ICON_SWN)['ASOB_T'][:]
            swd = swn+swu
        elif region.lower()=="nau":
            swu = xr.open_dataset(ap.ALL_NAU_ICON_SWU)['ASOU_T'][:]
            swn = xr.open_dataset(ap.ALL_NAU_ICON_SWN)['ASOB_T'][:]
            swd = swn+swu
        elif region.lower()=="shl":
            swu = xr.open_dataset(ap.ALL_SHL_ICON_SWU)['ASOU_T'][:]
            swn = xr.open_dataset(ap.ALL_SHL_ICON_SWN)['ASOB_T'][:]
            swd = swn+swu
        swd_undone = util.undomean(swd)
        swd = xr.DataArray(swd_undone, dims=swd.dims, coords=swd.coords, attrs={'name':'SWD_mean_undone_%s'%region})
        swd = swd[ind0:]
    elif model.lower()=="nicam":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_NICAM_SWD)['ss_swd_toa'][:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_NICAM_SWD)['ss_swd_toa'][:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_NICAM_SWD)['ss_swd_toa'][:]
        swd = swd[ind0:,0,:,:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][:]
        swd = swd[ind0:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][1:1921*2:2].values
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][1:1921*2:2].values
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][1:1921*2:2].values
        print("    used FV3 SWD...")
        swd = swd[ind0//2:]
    elif model.lower()=="geos":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_NICAM_SWD)['ss_swd_toa'][ind0:3925,0,:,:].values
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_NICAM_SWD)['ss_swd_toa'][ind0:3925,0,:,:].values
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_NICAM_SWD)['ss_swd_toa'][ind0:3925,0,:,:].values
        print("    used NICAM SWD...")
    elif model.lower()=="mpas":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][ind0:3840].values
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][ind0:3840].values
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][ind0:3840].values
        print("    used FV3 SWD...")
    elif model.lower()=="arp":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][ind0:3840].values
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][ind0:3840].values
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][ind0:3840].values
        print("    used FV3 SWD...")
    elif model.lower()=="um":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_UM_SWD)['toa_incoming_shortwave_flux'][ind0:]
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_UM_SWD)['toa_incoming_shortwave_flux'][ind0:]
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_UM_SWD)['toa_incoming_shortwave_flux'][ind0:]
    elif model.lower()=="ecmwf":
        if region.lower()=="twp":
            swd = xr.open_dataset(ap.ALL_TWP_FV3_SWD)['fsdt'][3:960*4:4].values
        elif region.lower()=="nau":
            swd = xr.open_dataset(ap.ALL_NAU_FV3_SWD)['fsdt'][3:960*4:4].values
        elif region.lower()=="shl":
            swd = xr.open_dataset(ap.ALL_SHL_FV3_SWD)['fsdt'][3:960*4:4].values
        print("    used FV3 SWD...")
        swd = swd[ind0//4:]
    print("Returned SWD for "+model+" ("+region+") with shape:", swd.shape)
    return swd

def get_olr(model, region):
    """ Return olr for models in region.
    
        For models that don't output olr we will use the zonal mean
            to estimate olr for closest latitude.
    """
    ind0=0 if INCLUDE_SHOCK else 96*2 # exclude first two days
    if model.lower()=="icon":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_ICON_OLR)['ATHB_T'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_ICON_OLR)['ATHB_T'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_ICON_OLR)['ATHB_T'][:]
        olr_undone = -1*(util.undomean(olr))
        olr_undone[0,:,:] = np.nan
        olr_undone[193,:,:] = np.nan
        olr = xr.DataArray(olr_undone, dims=olr.dims, coords=olr.coords, attrs={'name':'OLR_mean_undone_%s'%region})
    elif model.lower()=="nicam":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_NICAM_OLR)['sa_lwu_toa'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_NICAM_OLR)['sa_lwu_toa'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_NICAM_OLR)['sa_lwu_toa'][:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_FV3_OLR)['flut'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_FV3_OLR)['flut'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_FV3_OLR)['flut'][:]
    elif model.lower()=="sam":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_SAM_OLR)['LWNTA'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_SAM_OLR)['LWNTA'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_SAM_OLR)['LWNTA'][:]
    elif model.lower()=="geos":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_GEOS_OLR)['OLR'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_GEOS_OLR)['OLR'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_GEOS_OLR)['OLR'][:]
    elif model.lower()=="mpas":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_MPAS_OLR)['olrtoa'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_MPAS_OLR)['olrtoa'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_MPAS_OLR)['olrtoa'][:]
    elif model.lower()=="arp":
        if region.lower()=="twp":
            olr = -xr.open_dataset(ap.ALL_TWP_ARP_OLR)['ttr'][:,:,:]/900
        elif region.lower()=="nau":
            olr = -xr.open_dataset(ap.ALL_NAU_ARP_OLR)['ttr'][:,:,:]/900
        elif region.lower()=="shl":
            olr = -xr.open_dataset(ap.ALL_SHL_ARP_OLR)['ttr'][:,:,:]/900
    elif model.lower()=="um":
        if region.lower()=="twp":
            olr = xr.open_dataset(ap.ALL_TWP_UM_OLR)['toa_outgoing_longwave_flux'][:]
        elif region.lower()=="nau":
            olr = xr.open_dataset(ap.ALL_NAU_UM_OLR)['toa_outgoing_longwave_flux'][:]
        elif region.lower()=="shl":
            olr = xr.open_dataset(ap.ALL_SHL_UM_OLR)['toa_outgoing_longwave_flux'][:]
    elif model.lower()=="ecmwf":
        if region.lower()=="twp":
            olr = -xr.open_dataset(ap.ALL_TWP_ECMWF_OLR)["ttr"][:,:,:]/3600
        elif region.lower()=="nau":
            olr = -xr.open_dataset(ap.ALL_NAU_ECMWF_OLR)['ttr'][:,:,:]/3600
        elif region.lower()=="shl":
            olr = -xr.open_dataset(ap.ALL_SHL_ECMWF_OLR)['ttr'][:,:,:]/3600
    else: raise Exception("model not valid.")
    if model.lower()=="sam":
        olr = olr[ind0//2:]
    elif model.lower()=="ecmwf":
        olr = olr[ind0//4:]
    else:
        olr = olr[ind0:]
    print("Returned olr for "+model+" ("+region+") with shape:", olr.shape)
    return olr

def get_iwp(model, region, ice_only=True):
    """ Return iwp for models in region as xarray.
            If ice_only=False, returns frozen water path,
            otherwise returns ice only.
    """
    ind0=0 if INCLUDE_SHOCK else 96*2 # exclude first two days
    if model.lower()=="icon":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_ICON_IWP)['TQI_DIA'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_TWP_ICON_SWP)['TQS'][:]
                gwp = xr.open_dataset(ap.ALL_TWP_ICON_GWP)['TQG'][:]
                fwp = iwp + swp.values + gwp.values
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_ICON_IWP)['TQI_DIA'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_NAU_ICON_SWP)['TQS'][:]
                gwp = xr.open_dataset(ap.ALL_NAU_ICON_GWP)['TQG'][:]
                fwp = iwp + swp.values + gwp.values
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_ICON_IWP)['TQI_DIA'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_SHL_ICON_SWP)['TQS'][:]
                gwp = xr.open_dataset(ap.ALL_SHL_ICON_GWP)['TQG'][:]
                fwp = iwp + swp.values + gwp.values
    elif model.lower()=="nicam":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_NICAM_IWP)['sa_cldi'][:]
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_NICAM_IWP)['sa_cldi'][:]
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_NICAM_IWP)['sa_cldi'][:]
        if not(ice_only):
            fwp = iwp[:,0,:,:]
        else:
            iwp = iwp[:,0,:,:]
    elif model.lower()=="fv3":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_FV3_IWP)['intqi'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_TWP_FV3_SWP)['intqs'][:]
                gwp = xr.open_dataset(ap.ALL_TWP_FV3_GWP)['intqg'][:]
                fwp = iwp + swp.values + gwp.values
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_FV3_IWP)['intqi'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_NAU_FV3_SWP)['intqs'][:]
                gwp = xr.open_dataset(ap.ALL_NAU_FV3_GWP)['intqg'][:]
                fwp = iwp + swp.values + gwp.values
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_FV3_IWP)['intqi'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_SHL_FV3_SWP)['intqs'][:]
                gwp = xr.open_dataset(ap.ALL_SHL_FV3_GWP)['intqg'][:]
                fwp = iwp + swp.values + gwp.values
    elif model.lower()=="sam":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_SAM_IWP)['IWP'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_TWP_SAM_SWP)['SWP'][:]
                fwp = iwp + swp.values
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_SAM_IWP)['IWP'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_NAU_SAM_SWP)['SWP'][:]
                fwp = iwp + swp.values
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_SAM_IWP)['IWP'][:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_SHL_SAM_SWP)['SWP'][:]
                fwp = iwp + swp.values
    elif model.lower()=="geos":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_GEOS_IWP)['TQI']
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_GEOS_IWP)['TQI']
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_GEOS_IWP)['TQI']
        if not(ice_only):
            fwp = iwp
    elif model.lower()=="mpas":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_MPAS_IWP)['vert_int_qi'][:3838]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_TWP_MPAS_SWP)['vert_int_qs'][:3838]
                gwp = xr.open_dataset(ap.ALL_TWP_MPAS_GWP)['vert_int_qg'][:3838]
                fwp = iwp + swp.values + gwp.values
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_MPAS_IWP)['vert_int_qi'][:3838]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_NAU_MPAS_SWP)['vert_int_qs'][:3838]
                gwp = xr.open_dataset(ap.ALL_NAU_MPAS_GWP)['vert_int_qg'][:3838]
                fwp = iwp + swp.values + gwp.values
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_MPAS_IWP)['vert_int_qi'][:3838]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_SHL_MPAS_SWP)['vert_int_qs'][:3838]
                gwp = xr.open_dataset(ap.ALL_SHL_MPAS_GWP)['vert_int_qg'][:3838]
                fwp = iwp + swp.values + gwp.values
    elif model.lower()=="arp":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_ARP_IWP)['var255'][:,:,:]
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_ARP_IWP)['var255'][:,:,:]
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_ARP_IWP)['var255'][:,:,:]
        if not(ice_only):
            fwp = iwp
    elif model.lower()=="um":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_UM_IWP)['atmosphere_mass_content_of_cloud_ice'][:]
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_UM_IWP)['atmosphere_mass_content_of_cloud_ice'][:]
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_UM_IWP)['atmosphere_mass_content_of_cloud_ice'][:]
        if not(ice_only):
            fwp = iwp
    elif model.lower()=="ecmwf":
        if region.lower()=="twp":
            iwp = xr.open_dataset(ap.ALL_TWP_ECMWF_IWP)["tciw"][:,:,:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_TWP_ECMWF_SWP)["tcsw"][:,:,:]
                fwp = iwp + swp.values
        elif region.lower()=="nau":
            iwp = xr.open_dataset(ap.ALL_NAU_ECMWF_IWP)['tciw'][:,:,:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_NAU_ECMWF_SWP)["tcsw"][:,:,:]
                fwp = iwp + swp.values
        elif region.lower()=="shl":
            iwp = xr.open_dataset(ap.ALL_SHL_ECMWF_IWP)['tciw'][:,:,:]
            if not(ice_only):
                swp = xr.open_dataset(ap.ALL_SHL_ECMWF_SWP)["tcsw"][:,:,:]
                fwp = iwp + swp.values
    else: raise Exception("Invalid model")
    if ice_only:
        if model.lower()=="sam":
            iwp = iwp[ind0//2:]
        elif model.lower()=="ecmwf":
            iwp = iwp[ind0//4:]
        else:
            iwp = iwp[ind0:]
        print("Returned iwp for "+model+" ("+region+") with shape:", iwp.shape)
        return iwp
    else:
        if model.lower()=="sam":
            fwp = fwp[ind0//2:]
        elif model.lower()=="ecmwf":
            fwp = fwp[ind0//4:]
        else:
            fwp = fwp[ind0:]
        print("Returned fwp for "+model+" ("+region+") with shape:", iwp.shape)
        return fwp
    return
