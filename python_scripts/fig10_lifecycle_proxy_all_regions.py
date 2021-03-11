#!/usr/bin/env python
""" fig10_lifecycle_proxy_all_regions.py
    Author: Sami Turbeville
    Updated: 08 Nov 2020
    
    This script makes joint albedo-OLR histograms for FV3,
    SAM, ICON, NICAM, MPAS, ECMWF, ARPNH, and UM 
    averaged spatially to match CERES CCCM footprint for
    all regions: shl, twp, and nau.
"""

import numpy as np
import xarray as xr
from utility import load01deg, util
import utility.analysis_parameters as ap
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

#----------------- TWP -----------------------------------
#load albedo and olr for all models
# gswu = load01deg.get_swu("GEOS", "TWP")
fswu = load01deg.get_swu("FV3", "TWP")
iswu = load01deg.get_swu("ICON", "TWP")
sswu = load01deg.get_swu("SAM", "TWP")
nswu = load01deg.get_swu("NICAM", "TWP")

# gswd = load01deg.get_swd("GEOS", "TWP")
fswd = load01deg.get_swd("FV3", "TWP")
iswd = load01deg.get_swd("ICON", "TWP")
sswd = load01deg.get_swd("SAM", "TWP")
nswd = load01deg.get_swd("NICAM", "TWP")

# galb = gswu/gswd
falb = fswu/fswd
ialb = iswu/iswd
salb = sswu/sswd
nalb = nswu/nswd
salb = xr.DataArray(salb, dims=sswu.dims, coords=sswu.coords)
ialb = ialb.where((ialb>0)&(ialb<1))

del fswu, fswd, iswu, iswd, sswu, sswd, nswu, nswd

# golr = load01deg.get_olr("GEOS", "TWP")
folr = load01deg.get_olr("FV3", "TWP")
iolr = load01deg.get_olr("ICON", "TWP")
solr = load01deg.get_olr("SAM", "TWP")
nolr = load01deg.get_olr("NICAM", "TWP")[:,0,:,:]

# load observational data
# calipso-ceres-cloudsat-modis (CCCM)
cswu = xr.open_dataset(ap.CERES_TWP)["CERES SW TOA flux - upwards"]
cswd = xr.open_dataset(ap.CERES_TWP)["CERES SW TOA flux - downwards"]
colr = xr.open_dataset(ap.CERES_TWP)["Outgoing LW radiation at TOA"]
calb = cswu/cswd
del cswu, cswd

# use between 10am-2pm LT or <4UTC
# galb = galb.where(galb.time.dt.hour<4)
# golr = golr.where(golr.time.dt.hour<4)
falb = falb.where(falb.time.dt.hour<4)
folr = folr.where(folr.time.dt.hour<4)
ialb = ialb.where(ialb.time.dt.hour<4)
iolr = iolr.where(iolr.time.dt.hour<4)
salb = salb.where((salb.time.dt.hour<6)&(salb.time.dt.hour>2))
solr = solr.where(solr.time.dt.hour<4)
nalb = nalb.where(nalb.time.dt.hour<4)
nolr = nolr.where(nolr.time.dt.hour<4)
print("len zeros in cccm alb", len(calb.values[calb.values<=0]), end="\n\n")
calb = calb.where(calb>0)

# print(galb.shape, golr.shape)
print(falb.shape, folr.shape)
print(ialb.shape, iolr.shape)
print(salb.shape, solr.shape)
print(nalb.shape, nolr.shape)
print(calb.shape, colr.shape)


# average over ~30km (0.3deg) to match CCCM footprint
# Choose the daytime values only (smaller dataset to average spatially)

i_alb_llavg = np.zeros((ialb.shape[0],30,30))
f_alb_llavg = np.zeros((falb.shape[0],30,30))
s_alb_llavg = np.zeros((salb.shape[0],30,30))
n_alb_llavg = np.zeros((nalb.shape[0],30,30))

i_olr_llavg = np.zeros((iolr.shape[0],30,30))
f_olr_llavg = np.zeros((folr.shape[0],30,30))
s_olr_llavg = np.zeros((solr.shape[0],30,30))
n_olr_llavg = np.zeros((nolr.shape[0],30,30))

for n in range(30):
    for m in range(30):
        i_alb_llavg[:,n,m] = np.mean(ialb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_alb_llavg[:,n,m] = np.mean(falb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_alb_llavg[:,n,m] = np.mean(salb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_alb_llavg[:,n,m] = np.mean(nalb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        
        i_olr_llavg[:,n,m] = np.mean(iolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_olr_llavg[:,n,m] = np.mean(folr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_olr_llavg[:,n,m] = np.mean(solr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_olr_llavg[:,n,m] = np.mean(nolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
print('done')

# flatten arrays for histogram
folr_flat = f_olr_llavg.flatten()
falb_flat = f_alb_llavg.flatten()
iolr_flat = i_olr_llavg.flatten()
ialb_flat = i_alb_llavg.flatten()
# golr_flat = g_olr_llavg.flatten()
# galb_flat = g_alb_llavg.flatten()
solr_flat = s_olr_llavg.flatten()
salb_flat = s_alb_llavg.flatten()
nolr_flat = n_olr_llavg.flatten()
nalb_flat = n_alb_llavg.flatten()

#-------------- SAHEL -------------------------------------------------
#load albedo and olr_shl for all models
# gswu_shl= load01deg.get_swu("GEOS", "SHL")
fswu_shl= load01deg.get_swu("FV3", "SHL")
iswu_shl= load01deg.get_swu("ICON", "SHL")
sswu_shl= load01deg.get_swu("SAM", "SHL")
nswu_shl= load01deg.get_swu("NICAM", "SHL")

# gswd_shl= load01deg.get_swd("GEOS", "SHL")
fswd_shl= load01deg.get_swd("FV3", "SHL")
iswd_shl= load01deg.get_swd("ICON", "SHL")
sswd_shl= load01deg.get_swd("SAM", "SHL")
nswd_shl= load01deg.get_swd("NICAM", "SHL")

# galb_shl = gswu/gswd
falb_shl = fswu_shl/fswd_shl
ialb_shl = iswu_shl.where(iswu_shl>0)/iswd_shl.where(iswd_shl>0)
salb_shl = np.where(sswu_shl>0,sswu_shl,np.nan)/np.where(sswd_shl>0,sswd_shl,np.nan)
salb_shl = xr.DataArray(salb_shl, dims=sswu_shl.dims, coords=sswu_shl.coords)
nalb_shl = nswu_shl/nswd_shl
ialb_shl = ialb_shl.where((ialb_shl>0)&(ialb_shl<1))

del fswu_shl, fswd_shl, iswu_shl, iswd_shl, sswu_shl, sswd_shl, nswu_shl, nswd_shl

# golr_shl = load01deg.get_olr("GEOS", "SHL")
folr_shl = load01deg.get_olr("FV3", "SHL")
iolr_shl = load01deg.get_olr("ICON", "SHL")
solr_shl = load01deg.get_olr("SAM", "SHL")
nolr_shl = load01deg.get_olr("NICAM", "SHL")[:,0,:,:]

# load observational data
# calipso-ceres-cloudsat-modis (CCCM)
cswu_shl= xr.open_dataset(ap.CERES_SHL)["CERES SW TOA flux - upwards"]
cswd_shl= xr.open_dataset(ap.CERES_SHL)["CERES SW TOA flux - downwards"]
colr_shl = xr.open_dataset(ap.CERES_SHL)["Outgoing LW radiation at TOA"]
calb_shl = cswu_shl/cswd_shl
del cswu_shl, cswd_shl

# use between 10am-2pm LT or 9UTC-13UTC
falb_shl = falb_shl.where((falb_shl.time.dt.hour<13) & (falb_shl.time.dt.hour>=9))
folr_shl = folr_shl.where((folr_shl.time.dt.hour<13) & (folr_shl.time.dt.hour>=9))
ialb_shl = ialb_shl.where((ialb_shl.time.dt.hour<13) & (ialb_shl.time.dt.hour>=9))
iolr_shl = iolr_shl.where((iolr_shl.time.dt.hour<13) & (iolr_shl.time.dt.hour>=9))
salb_shl = salb_shl.where((salb_shl.time.dt.hour<13) & (salb_shl.time.dt.hour>=9))
solr_shl = solr_shl.where((solr_shl.time.dt.hour<13) & (solr_shl.time.dt.hour>=9))
nalb_shl = nalb_shl.where((nalb_shl.time.dt.hour<13) & (nalb_shl.time.dt.hour>=9))
nolr_shl = nolr_shl.where((nolr_shl.time.dt.hour<13) & (nolr_shl.time.dt.hour>=9))

print("len zeros in cccm alb", len(calb_shl.values[calb_shl.values<=0]), end="\n\n")
calb_shl = calb_shl.where(calb_shl>0)

# print(galb_shl.shape, golr_shl.shape)
print(falb_shl.shape, folr_shl.shape)
print(ialb_shl.shape, iolr_shl.shape)
print(salb_shl.shape, solr_shl.shape)
print(nalb_shl.shape, nolr_shl.shape)
print(calb_shl.shape, colr_shl.shape)


# average over ~30km (0.3deg) to match CCCM footprint
# Choose the daytime values only (smaller dataset to average spatially)

i_alb_shl_llavg = np.zeros((ialb_shl.shape[0],30,30))
f_alb_shl_llavg = np.zeros((falb_shl.shape[0],30,30))
s_alb_shl_llavg = np.zeros((salb_shl.shape[0],30,30))
n_alb_shl_llavg = np.zeros((nalb_shl.shape[0],30,30))

i_olr_shl_llavg = np.zeros((iolr_shl.shape[0],30,30))
f_olr_shl_llavg = np.zeros((folr_shl.shape[0],30,30))
s_olr_shl_llavg = np.zeros((solr_shl.shape[0],30,30))
n_olr_shl_llavg = np.zeros((nolr_shl.shape[0],30,30))

for n in range(30):
    for m in range(30):
        i_alb_shl_llavg[:,n,m] = np.mean(ialb_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_alb_shl_llavg[:,n,m] = np.mean(falb_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_alb_shl_llavg[:,n,m] = np.mean(salb_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_alb_shl_llavg[:,n,m] = np.mean(nalb_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        
        i_olr_shl_llavg[:,n,m] = np.mean(iolr_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_olr_shl_llavg[:,n,m] = np.mean(folr_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_olr_shl_llavg[:,n,m] = np.mean(solr_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_olr_shl_llavg[:,n,m] = np.mean(nolr_shl[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
print('done')

# flatten arrays for histogram
folr_shl_flat = f_olr_shl_llavg.flatten()
falb_shl_flat = f_alb_shl_llavg.flatten()
iolr_shl_flat = i_olr_shl_llavg.flatten()
ialb_shl_flat = i_alb_shl_llavg.flatten()
# golr_shl_flat = g_olr_shl_llavg.flatten()
# galb_shl_flat = g_alb_shl_llavg.flatten()
solr_shl_flat = s_olr_shl_llavg.flatten()
salb_shl_flat = s_alb_shl_llavg.flatten()
nolr_shl_flat = n_olr_shl_llavg.flatten()
nalb_shl_flat = n_alb_shl_llavg.flatten()

#----------------------- NAURU -------------------------------------
#load albedo and olr_nau for all models
# gswu_nau= load01deg.get_swu("GEOS", "NAU")
fswu_nau= load01deg.get_swu("FV3", "NAU")
iswu_nau= load01deg.get_swu("ICON", "NAU")
sswu_nau= load01deg.get_swu("SAM", "NAU")
nswu_nau= load01deg.get_swu("NICAM", "NAU")

# gswd_nau= load01deg.get_swd("GEOS", "NAU")
fswd_nau= load01deg.get_swd("FV3", "NAU")
iswd_nau= load01deg.get_swd("ICON", "NAU")
sswd_nau= load01deg.get_swd("SAM", "NAU")
nswd_nau= load01deg.get_swd("NICAM", "NAU")

# galb_nau = gswu/gswd
falb_nau = fswu_nau/fswd_nau
ialb_nau = iswu_nau.where(iswu_nau>0)/iswd_nau.where(iswd_nau>0)
salb_nau = np.where(sswu_nau>0,sswu_nau,np.nan)/np.where(sswd_nau>0,sswd_nau,np.nan)
nalb_nau = nswu_nau/nswd_nau
ialb_nau = ialb_nau.where((ialb_nau>0)&(ialb_nau<1))
salb_nau = xr.DataArray(salb_nau, dims=sswu_nau.dims, coords=sswu_nau.coords)

del fswu_nau, fswd_nau, iswu_nau, iswd_nau, sswu_nau, sswd_nau, nswu_nau, nswd_nau

# golr_nau = load01deg.get_olr("GEOS", "NAU")
folr_nau = load01deg.get_olr("FV3", "NAU")
iolr_nau = load01deg.get_olr("ICON", "NAU")
solr_nau = load01deg.get_olr("SAM", "NAU")
nolr_nau = load01deg.get_olr("NICAM", "NAU")[:,0,:,:]

# load observational data
# calipso-ceres-cloudsat-modis (CCCM)
cswu_nau = xr.open_dataset(ap.CERES_NAU)["CERES SW TOA flux - upwards"]
cswd_nau = xr.open_dataset(ap.CERES_NAU)["CERES SW TOA flux - downwards"]
colr_nau = xr.open_dataset(ap.CERES_NAU)["Outgoing LW radiation at TOA"]
calb_nau = cswu_nau/cswd_nau
del cswu_nau, cswd_nau

# use between 10am-2pm LT or 22UTC-2UTC
# galb_nau = galb_nau.where(galb_nau.time.dt.hour<13 & galb_nau.time.dt.hour>=9)
# golr_nau = golr_nau.where(golr_nau.time.dt.hour<13 & golr_nau.time.dt.hour>=9)
falb_nau = falb_nau.where((falb_nau.time.dt.hour<2) | (falb_nau.time.dt.hour>=22))
folr_nau = folr_nau.where((folr_nau.time.dt.hour<2) | (folr_nau.time.dt.hour>=22))
ialb_nau = ialb_nau.where((ialb_nau.time.dt.hour<2) | (ialb_nau.time.dt.hour>=22))
iolr_nau = iolr_nau.where((iolr_nau.time.dt.hour<2) | (iolr_nau.time.dt.hour>=22))
salb_nau = salb_nau.where((salb.time.dt.hour<8)&(salb.time.dt.hour>=4))
solr_nau = solr_nau.where((solr.time.dt.hour<8)&(solr.time.dt.hour>=4))
nalb_nau = nalb_nau.where((nalb_nau.time.dt.hour<2) | (nalb_nau.time.dt.hour>=22))
nolr_nau = nolr_nau.where((nolr_nau.time.dt.hour<2) | (nolr_nau.time.dt.hour>=22))

print("len zeros in cccm alb", len(calb_nau.values[calb_nau.values<=0]), end="\n\n")
calb_nau = calb_nau.where(calb_nau>0)

# print(galb_nau.shape, golr_nau.shape)
print(falb_nau.shape, folr_nau.shape)
print(ialb_nau.shape, iolr_nau.shape)
print(salb_nau.shape, solr_nau.shape)
print(nalb_nau.shape, nolr_nau.shape)
print(calb_nau.shape, colr_nau.shape)


# average over ~30km (0.3deg) to match CCCM footprint
# Choose the daytime values only (smaller dataset to average spatially)

i_alb_nau_llavg = np.zeros((ialb_nau.shape[0],30,30))
f_alb_nau_llavg = np.zeros((falb_nau.shape[0],30,30))
s_alb_nau_llavg = np.zeros((salb_nau.shape[0],30,30))
n_alb_nau_llavg = np.zeros((nalb_nau.shape[0],30,30))

i_olr_nau_llavg = np.zeros((iolr_nau.shape[0],30,30))
f_olr_nau_llavg = np.zeros((folr_nau.shape[0],30,30))
s_olr_nau_llavg = np.zeros((solr_nau.shape[0],30,30))
n_olr_nau_llavg = np.zeros((nolr_nau.shape[0],30,30))

for n in range(30):
    for m in range(30):
        i_alb_nau_llavg[:,n,m] = np.mean(ialb_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_alb_nau_llavg[:,n,m] = np.mean(falb_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_alb_nau_llavg[:,n,m] = np.mean(salb_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_alb_nau_llavg[:,n,m] = np.mean(nalb_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        
        i_olr_nau_llavg[:,n,m] = np.mean(iolr_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_olr_nau_llavg[:,n,m] = np.mean(folr_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_olr_nau_llavg[:,n,m] = np.mean(solr_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_olr_nau_llavg[:,n,m] = np.mean(nolr_nau[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
print('done')

# flatten arrays for histogram
folr_nau_flat = f_olr_nau_llavg.flatten()
falb_nau_flat = f_alb_nau_llavg.flatten()
iolr_nau_flat = i_olr_nau_llavg.flatten()
ialb_nau_flat = i_alb_nau_llavg.flatten()
# golr_nau_flat = g_olr_nau_llavg.flatten()
# galb_nau_flat = g_alb_nau_llavg.flatten()
solr_nau_flat = s_olr_nau_llavg.flatten()
salb_nau_flat = s_alb_nau_llavg.flatten()
nolr_nau_flat = n_olr_nau_llavg.flatten()
nalb_nau_flat = n_alb_nau_llavg.flatten()

# create figure
fig = plt.figure(figsize=(6*5+1,3*6), constrained_layout=True)
gs = fig.add_gridspec(3,16*5+1)
cmap = cm.gist_earth_r
ms = 30
lw = 2
fs = 24
c = 'b'
levs = np.arange(-3.2, -1, 0.2)
xbins = np.linspace(70,320,26)
xbinmid = (xbins[1:]+xbins[:-1])/2
cax = fig.add_subplot(gs[1,:16])
colr = colr[~np.isnan(calb)]
calb = calb[~np.isnan(calb)]
util.dennisplot("density", colr, calb, levels=levs, 
                model="CERES CCCM (All year, 2007-10)", region="TWP", 
                var_name="",units="", cmap=cmap, ax=cax, colorbar_on=False, fs=fs)
cax.plot([np.nanmean(colr)],[np.nanmean(calb)], 'r.', ms=ms)
cymean, _, _ = stats.binned_statistic(colr.values.flatten(), calb.values.flatten(), bins=xbins)
# cax.plot(xbinmid, cymean, c, lw=lw)
print("1/15: made ceres syn1 axis...")
nax = fig.add_subplot(gs[1,16:16*2])
nolr_flat = nolr_flat[~np.isnan(nalb_flat)]
nalb_flat = nalb_flat[~np.isnan(nalb_flat)]
util.dennisplot("density", nolr_flat, nalb_flat, levels=levs, model="NICAM", region="TWP", 
           var_name="",units="", cmap=cmap, ax=nax, colorbar_on=False, fs=fs)
nax.plot([np.nanmean(nolr_flat)],[np.nanmean(nalb_flat)], 'r.', ms=ms)
nymean, _, _ = stats.binned_statistic(nolr_flat, nalb_flat, bins=xbins)
# nax.plot(xbinmid, nymean, c, lw=lw)
print("2/15: made nicam axis...")
fax = fig.add_subplot(gs[1,16*2:16*3])
folr_flat = folr_flat[~np.isnan(falb_flat)]
falb_flat = falb_flat[~np.isnan(falb_flat)]
fax, cs = util.dennisplot("density", folr_flat, falb_flat, levels=levs, model="FV3", region="TWP", 
           var_name="",units="", cmap=cmap, ax=fax, colorbar_on=False, fs=fs)
fax.plot([np.nanmean(folr_flat)],[np.nanmean(falb_flat)], 'r.', ms=ms)
fymean, _, _ = stats.binned_statistic(folr_flat, falb_flat, bins=xbins)
# fax.plot(xbinmid, fymean, c, lw=lw)
print("3/15: made fv3 axis...")
iax = fig.add_subplot(gs[1,16*3:16*4])
iolr_flat = iolr_flat[~np.isnan(ialb_flat)]
ialb_flat = ialb_flat[~np.isnan(ialb_flat)]
util.dennisplot("density", iolr_flat, ialb_flat, levels=levs, model="ICON", region="TWP", 
           var_name="",units="", cmap=cmap, ax=iax, colorbar_on=False, fs=fs)
iax.plot([np.nanmean(iolr_flat)],[np.nanmean(ialb_flat[~np.isnan(ialb_flat)])], 'r.', ms=ms)
iymean, _, _ = stats.binned_statistic(iolr_flat, ialb_flat, bins=xbins)
# iax.plot(xbinmid, iymean, c, lw=lw)
print("4/15: made icon axis...")
sax = fig.add_subplot(gs[1,16*4:16*5])
solr_flat = solr_flat[~np.isnan(salb_flat)]
salb_flat = salb_flat[~np.isnan(salb_flat)]
util.dennisplot("density", solr_flat, salb_flat, levels=levs, model="SAM", region="TWP", 
           var_name="",units="", cmap=cmap, ax=sax, colorbar_on=False, fs=fs)
sax.plot([np.nanmean(solr_flat)],[np.nanmean(salb_flat)], 'r.', ms=ms)
symean, _, _ = stats.binned_statistic(solr_flat, salb_flat, bins=xbins)
# sax.plot(xbinmid, symean, c, lw=lw)
print("5/15: made sam axis...")
cax.tick_params(labelsize=fs-4)
nax.tick_params(labelsize=fs-4)
fax.tick_params(labelsize=fs-4)
iax.tick_params(labelsize=fs-4)
sax.tick_params(labelsize=fs-4)

cax = fig.add_subplot(gs[0,:16])
colr_shl = colr_shl[~np.isnan(calb_shl)]
calb_shl = calb_shl[~np.isnan(calb_shl)]
util.dennisplot("density", colr_shl, calb_shl, levels=levs, 
                model="CERES CCCM (JAS, 2007-10)", region="SHL", 
                var_name="",units="", cmap=cmap, ax=cax, colorbar_on=False, fs=fs)
cax.plot([np.nanmean(colr_shl)],[np.nanmean(calb_shl)], 'r.', ms=ms)
cymean, _, _ = stats.binned_statistic(colr_shl.values.flatten(), calb_shl.values.flatten(), bins=xbins)
# cax.plot(xbinmid, cymean, c, lw=lw)
print("6/15: made ceres syn1 axis...")
nax = fig.add_subplot(gs[0,16:16*2])
nolr_shl_flat = nolr_shl_flat[~np.isnan(nalb_shl_flat)]
nalb_shl_flat = nalb_shl_flat[~np.isnan(nalb_shl_flat)]
util.dennisplot("density", nolr_shl_flat, nalb_shl_flat, levels=levs, model="NICAM", region="SHL", 
           var_name="",units="", cmap=cmap, ax=nax, colorbar_on=False, fs=fs)
nax.plot([np.nanmean(nolr_shl_flat)],[np.nanmean(nalb_shl_flat)], 'r.', ms=ms)
nymean, _, _ = stats.binned_statistic(nolr_shl_flat, nalb_shl_flat, bins=xbins)
# nax.plot(xbinmid, nymean, c, lw=lw)
print("7/15: made nicam axis...")
fax = fig.add_subplot(gs[0,16*2:16*3])
folr_shl_flat = folr_shl_flat[~np.isnan(falb_shl_flat)]
falb_shl_flat = falb_shl_flat[~np.isnan(falb_shl_flat)]
fax, cs = util.dennisplot("density", folr_shl_flat, falb_shl_flat, levels=levs, model="FV3", region="SHL", 
           var_name="",units="", cmap=cmap, ax=fax, colorbar_on=False, fs=fs)
fax.plot([np.nanmean(folr_shl_flat)],[np.nanmean(falb_shl_flat)], 'r.', ms=ms)
fymean, _, _ = stats.binned_statistic(folr_shl_flat, falb_shl_flat, bins=xbins)
# fax.plot(xbinmid, fymean, c, lw=lw)
print("8/15: made fv3 axis...")
iax = fig.add_subplot(gs[0,16*3:16*4])
iolr_shl_flat = iolr_shl_flat[~np.isnan(ialb_shl_flat)]
ialb_shl_flat = ialb_shl_flat[~np.isnan(ialb_shl_flat)]
util.dennisplot("density", iolr_shl_flat, ialb_shl_flat, levels=levs, model="ICON", region="SHL", 
           var_name="",units="", cmap=cmap, ax=iax, colorbar_on=False, fs=fs)
iax.plot([np.nanmean(iolr_shl_flat)],[np.nanmean(ialb_shl_flat[~np.isnan(ialb_shl_flat)])], 'r.', ms=ms)
iymean, _, _ = stats.binned_statistic(iolr_shl_flat, ialb_shl_flat, bins=xbins)
# iax.plot(xbinmid, iymean, c, lw=lw)
print("9/15: made icon axis...")
sax = fig.add_subplot(gs[0,16*4:16*5])
solr_shl_flat = solr_shl_flat[~np.isnan(salb_shl_flat)]
salb_shl_flat = salb_shl_flat[~np.isnan(salb_shl_flat)]
util.dennisplot("density", solr_shl_flat, salb_shl_flat, levels=levs, model="SAM", region="SHL", 
           var_name="",units="", cmap=cmap, ax=sax, colorbar_on=False, fs=fs)
sax.plot([np.nanmean(solr_shl_flat)],[np.nanmean(salb_shl_flat)], 'r.', ms=ms)
symean, _, _ = stats.binned_statistic(solr_shl_flat, salb_shl_flat, bins=xbins)
# sax.plot(xbinmid, symean, c, lw=lw)
print("10/15: made sam axis...")
# annotate the figure labels (a-e)
cax.tick_params(labelsize=fs-4)
nax.tick_params(labelsize=fs-4)
fax.tick_params(labelsize=fs-4)
iax.tick_params(labelsize=fs-4)
sax.tick_params(labelsize=fs-4)

cax.annotate("(a)", xy=(0.42,1.25), xycoords="axes fraction", fontsize=fs+5, weight="bold")
nax.annotate("(b)", xy=(0.42,1.25), xycoords="axes fraction", fontsize=fs+5, weight="bold")
fax.annotate("(c)", xy=(0.42,1.25), xycoords="axes fraction", fontsize=fs+5, weight="bold")
iax.annotate("(d)", xy=(0.42,1.25), xycoords="axes fraction", fontsize=fs+5, weight="bold")
sax.annotate("(e)", xy=(0.42,1.25), xycoords="axes fraction", fontsize=fs+5, weight="bold")

cax = fig.add_subplot(gs[2,:16])
colr_nau = colr_nau[~np.isnan(calb_nau)]
calb_nau = calb_nau[~np.isnan(calb_nau)]
util.dennisplot("density", colr_nau, calb_nau, levels=levs, 
                model="CERES CCCM (JAS, 2007-10)", region="NAU", 
                var_name="",units="", cmap=cmap, ax=cax, colorbar_on=False, fs=fs)
cax.plot([np.nanmean(colr_nau)],[np.nanmean(calb_nau)], 'r.', ms=ms)
cymean, _, _ = stats.binned_statistic(colr_nau.values.flatten(), calb_nau.values.flatten(), bins=xbins)
# cax.plot(xbinmid, cymean, c, lw=lw)
print("11/15: made ceres syn1 axis...")
nax = fig.add_subplot(gs[2,16:16*2])
nolr_nau_flat = nolr_nau_flat[~np.isnan(nalb_nau_flat)]
nalb_nau_flat = nalb_nau_flat[~np.isnan(nalb_nau_flat)]
util.dennisplot("density", nolr_nau_flat, nalb_nau_flat, levels=levs, model="NICAM", region="NAU", 
           var_name="",units="", cmap=cmap, ax=nax, colorbar_on=False, fs=fs)
nax.plot([np.nanmean(nolr_nau_flat)],[np.nanmean(nalb_nau_flat)], 'r.', ms=ms)
nymean, _, _ = stats.binned_statistic(nolr_nau_flat, nalb_nau_flat, bins=xbins)
# nax.plot(xbinmid, nymean, c, lw=lw)
print("12/15: made nicam axis...")
fax = fig.add_subplot(gs[2,16*2:16*3])
folr_nau_flat = folr_nau_flat[~np.isnan(falb_nau_flat)]
falb_nau_flat = falb_nau_flat[~np.isnan(falb_nau_flat)]
fax, cs = util.dennisplot("density", folr_nau_flat, falb_nau_flat, levels=levs, model="FV3", region="NAU", 
           var_name="",units="", cmap=cmap, ax=fax, colorbar_on=False, fs=fs)
fax.plot([np.nanmean(folr_nau_flat)],[np.nanmean(falb_nau_flat)], 'r.', ms=ms)
fymean, _, _ = stats.binned_statistic(folr_nau_flat, falb_nau_flat, bins=xbins)
# fax.plot(xbinmid, fymean, c, lw=lw)
print("13/15: made fv3 axis...")
iax = fig.add_subplot(gs[2,16*3:16*4])
iolr_nau_flat = iolr_nau_flat[~np.isnan(ialb_nau_flat)]
ialb_nau_flat = ialb_nau_flat[~np.isnan(ialb_nau_flat)]
util.dennisplot("density", iolr_nau_flat, ialb_nau_flat, levels=levs, model="ICON", region="NAU", 
           var_name="",units="", cmap=cmap, ax=iax, colorbar_on=False, fs=fs)
iax.plot([np.nanmean(iolr_nau_flat)],[np.nanmean(ialb_nau_flat[~np.isnan(ialb_nau_flat)])], 'r.', ms=ms)
iymean, _, _ = stats.binned_statistic(iolr_nau_flat, ialb_nau_flat, bins=xbins)
# iax.plot(xbinmid, iymean, c, lw=lw)
print("14/15: made icon axis...")
sax = fig.add_subplot(gs[2,16*4:16*5])
solr_nau_flat = solr_nau_flat[~np.isnan(salb_nau_flat)]
salb_nau_flat = salb_nau_flat[~np.isnan(salb_nau_flat)]
util.dennisplot("density", solr_nau_flat, salb_nau_flat, levels=levs, model="SAM", region="NAU", 
           var_name="",units="", cmap=cmap, ax=sax, colorbar_on=False, fs=fs)
sax.plot([np.nanmean(solr_nau_flat)],[np.nanmean(salb_nau_flat)], 'r.', ms=ms)
symean, _, _ = stats.binned_statistic(solr_nau_flat, salb_nau_flat, bins=xbins)
# sax.plot(xbinmid, symean, c, lw=lw)
print("15/15: made sam axis...")

cax.tick_params(labelsize=fs-4)
nax.tick_params(labelsize=fs-4)
fax.tick_params(labelsize=fs-4)
iax.tick_params(labelsize=fs-4)
sax.tick_params(labelsize=fs-4)


#add obs and colorbar
cbax0 = fig.add_subplot(gs[0,-1])
cbax1 = fig.add_subplot(gs[1,-1])
cbax2 = fig.add_subplot(gs[2,-1])
cbar0 = plt.colorbar(cs, cax=cbax0)
cbar1 = plt.colorbar(cs, cax=cbax1)
cbar2 = plt.colorbar(cs, cax=cbax2)
cbar0.set_label("log$_10$(pdf)", fontsize=fs-4)
cbar1.set_label("log$_10$(pdf)", fontsize=fs-4)
cbar2.set_label("log$_10$(pdf)", fontsize=fs-4)
cbar0.ax.tick_params(labelsize=fs-4)
cbar1.ax.tick_params(labelsize=fs-4)
cbar2.ax.tick_params(labelsize=fs-4)
print("... made colorbars ...")
fig.suptitle(" ")
plt.savefig("../plots/fig10_lifecycle_proxy_all_regions.png")
print("... saved in ../plots/fig10_lifecycle_proxy_all_regions.png")
plt.close()

print("TWP")
print("C:", np.nanmedian(colr), np.nanmedian(calb))
print("N:", np.nanmedian(nolr_flat), np.nanmedian(nalb_flat))
print("F:", np.nanmedian(folr_flat), np.nanmedian(falb_flat))
print("I:", np.nanmedian(iolr_flat), np.nanmedian(ialb_flat))
print("S:", np.nanmedian(solr_flat), np.nanmedian(salb_flat))

print("SHL")
print("C:", np.nanmedian(colr_shl), np.nanmedian(calb_shl))
print("N:", np.nanmedian(nolr_shl_flat), np.nanmedian(nalb_shl_flat))
print("F:", np.nanmedian(folr_shl_flat), np.nanmedian(falb_shl_flat))
print("I:", np.nanmedian(iolr_shl_flat), np.nanmedian(ialb_shl_flat))
print("S:", np.nanmedian(solr_shl_flat), np.nanmedian(salb_shl_flat))

print("NAU")
print("C:", np.nanmedian(colr_nau), np.nanmedian(calb_nau))
print("N:", np.nanmedian(nolr_nau_flat), np.nanmedian(nalb_nau_flat))
print("F:", np.nanmedian(folr_nau_flat), np.nanmedian(falb_nau_flat))
print("I:", np.nanmedian(iolr_nau_flat), np.nanmedian(ialb_nau_flat))
print("S:", np.nanmedian(solr_nau_flat), np.nanmedian(salb_nau_flat))
