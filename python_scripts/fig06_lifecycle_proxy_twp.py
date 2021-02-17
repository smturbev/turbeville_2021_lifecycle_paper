#!/usr/bin/env python
""" fig06_lifecycle_proxy_twp.py
    Author: Sami Turbeville
    Updated: 05 Aug 2020
    
    This script makes joint albedo-OLR histograms for FV3,
    SAM, GEOS, ICON, NICAM, MPAS, ECMWF, ARPNH, and UM 
    averaged spatially to match CERES CCCM footprint in the TWP. 
"""

import numpy as np
import xarray as xr
from utility import load01deg, util
import utility.analysis_parameters as ap
import matplotlib.pyplot as plt
from matplotlib import cm

#load albedo and olr for all models
gswu = load01deg.get_swu("GEOS", "TWP")
fswu = load01deg.get_swu("FV3", "TWP")
iswu = load01deg.get_swu("ICON", "TWP")
sswu = load01deg.get_swu("SAM", "TWP")
nswu = load01deg.get_swu("NICAM", "TWP")
mswu = load01deg.get_swu("MPAS", "TWP")
eswu = load01deg.get_swu("ECMWF", "TWP")
aswu = load01deg.get_swu("ARP", "TWP")
uswu = load01deg.get_swu("UM", "TWP")

gswd = load01deg.get_swd("GEOS", "TWP")
fswd = load01deg.get_swd("FV3", "TWP")
iswd = load01deg.get_swd("ICON", "TWP")
sswd = load01deg.get_swd("SAM", "TWP")
nswd = load01deg.get_swd("NICAM", "TWP")
mswd = load01deg.get_swd("MPAS", "TWP")
eswd = load01deg.get_swd("ECMWF", "TWP")
aswd = load01deg.get_swd("ARP", "TWP")
uswd = load01deg.get_swd("UM", "TWP")

galb = gswu/gswd
falb = fswu/fswd
ialb = iswu/iswd
salb = sswu/sswd
nalb = nswu/nswd
malb = mswu/mswd
ealb = eswu/eswd
aalb = aswu/aswd
ualb = uswu/uswd
aalb = xr.DataArray(aalb, dims=fswu.dims, coords=fswu.coords)

ialb = ialb.where((ialb>0)&(ialb<1))
malb = np.where((malb<1)&(malb>0),malb,np.nan)

del gswd, gswu, fswu, fswd, iswu, iswd, sswu, sswd, nswu, nswd, mswu, mswd, eswu, eswd, aswu, aswd, uswu, uswd

golr = load01deg.get_olr("GEOS", "TWP")
folr = load01deg.get_olr("FV3", "TWP")
iolr = load01deg.get_olr("ICON", "TWP")
solr = load01deg.get_olr("SAM", "TWP")
nolr = load01deg.get_olr("NICAM", "TWP")[:,0,:,:]
molr = load01deg.get_olr("MPAS", "TWP")[:-1,:,:]
eolr = load01deg.get_olr("ECMWF", "TWP")
aolr = load01deg.get_olr("ARP", "TWP")
uolr = load01deg.get_olr("UM", "TWP")
aolr = xr.DataArray(aolr, dims=folr.dims, coords=folr.coords)
molr = np.where((molr<330)&(molr>60),molr, np.nan)

# load observational data
# calipso-ceres-cloudsat-modis (CCCM)
cswu = xr.open_dataset(ap.CERES_TWP)["CERES SW TOA flux - upwards"]
cswd = xr.open_dataset(ap.CERES_TWP)["CERES SW TOA flux - downwards"]
colr = xr.open_dataset(ap.CERES_TWP)["Outgoing LW radiation at TOA"]
calb = cswu/cswd
del cswu, cswd

# use between 10am-2pm LT or <4UTC
galb = galb.where(galb.time.dt.hour<4)
golr = golr.where(golr.time.dt.hour<4)
falb = falb.where(falb.time.dt.hour<4)
folr = folr.where(folr.time.dt.hour<4)
ialb = ialb.where(ialb.time.dt.hour<4)
iolr = iolr.where(iolr.time.dt.hour<4)
salb = salb.where((salb.time.dt.hour<6)&(salb.time.dt.hour>=2))
solr = solr.where((solr.time.dt.hour<6)&(solr.time.dt.hour>=2))
nalb = nalb.where(nalb.time.dt.hour<4)
nolr = nolr.where(nolr.time.dt.hour<4)
malb = np.where(nalb.time.dt.hour.values[:3840, np.newaxis, np.newaxis]<4, malb, np.nan)
molr = np.where(nolr.time.dt.hour.values[:3840, np.newaxis, np.newaxis]<4, molr, np.nan)
ealb = ealb.where(ealb.time.dt.hour<4)
eolr = eolr.where(eolr.time.dt.hour<4)
aalb = aalb.where(aalb.time.dt.hour<4)
aolr = aolr.where(aolr.time.dt.hour<4)
ualb = ualb.where(ualb.time.dt.hour<4)
uolr = uolr.where(uolr.time.dt.hour<4)
print("len zeros in cccm alb", len(calb.values[calb.values<=0]), end="\n\n")
calb = calb.where(calb>0)

print(galb.shape, golr.shape)
print(falb.shape, folr.shape)
print(ialb.shape, iolr.shape)
print(salb.shape, solr.shape)
print(nalb.shape, nolr.shape)
print(malb.shape, molr.shape)
print(ealb.shape, eolr.shape)
print(aalb.shape, aolr.shape)
print(ualb.shape, uolr.shape)
print(calb.shape, colr.shape)


# average over ~30km (0.3deg) to match CCCM footprint
# Choose the daytime values only (smaller dataset to average spatially)

i_alb_llavg = np.zeros((3937,30,30))
f_alb_llavg = np.zeros((3840,30,30))
g_alb_llavg = np.zeros((3925,30,30))
s_alb_llavg = np.zeros((1920,30,30))
m_alb_llavg = np.zeros((3840,30,30))
n_alb_llavg = np.zeros((3936,30,30))
e_alb_llavg = np.zeros((960,30,30))
a_alb_llavg = np.zeros((3840,30,30))
u_alb_llavg = np.zeros((960,30,30))

i_olr_llavg = np.zeros((3937,30,30))
f_olr_llavg = np.zeros((3840,30,30))
g_olr_llavg = np.zeros((3925,30,30))
s_olr_llavg = np.zeros((1920,30,30))
m_olr_llavg = np.zeros((3840,30,30))
n_olr_llavg = np.zeros((3936,30,30))
e_olr_llavg = np.zeros((960,30,30))
a_olr_llavg = np.zeros((3840,30,30))
u_olr_llavg = np.zeros((960,30,30))

for n in range(30):
    for m in range(30):
        i_alb_llavg[:,n,m] = np.mean(ialb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_alb_llavg[:,n,m] = np.mean(falb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        g_alb_llavg[:,n,m] = np.mean(galb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_alb_llavg[:,n,m] = np.mean(salb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        m_alb_llavg[:,n,m] = np.mean(malb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_alb_llavg[:,n,m] = np.mean(nalb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        e_alb_llavg[:,n,m] = np.mean(ealb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        a_alb_llavg[:,n,m] = np.mean(aalb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        u_alb_llavg[:,n,m] = np.mean(ualb[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        
        i_olr_llavg[:,n,m] = np.mean(iolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        f_olr_llavg[:,n,m] = np.mean(folr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        g_olr_llavg[:,n,m] = np.mean(golr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        s_olr_llavg[:,n,m] = np.mean(solr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        m_olr_llavg[:,n,m] = np.mean(molr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        n_olr_llavg[:,n,m] = np.mean(nolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        e_olr_llavg[:,n,m] = np.mean(eolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        a_olr_llavg[:,n,m] = np.mean(aolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
        u_olr_llavg[:,n,m] = np.mean(uolr[:,3*n:(3*n)+3,3*m:(3*m)+3], axis=(1,2))
print('done')

# flatten arrays for histogram
folr_flat = f_olr_llavg.flatten()
falb_flat = f_alb_llavg.flatten()
iolr_flat = i_olr_llavg.flatten()
ialb_flat = i_alb_llavg.flatten()
golr_flat = g_olr_llavg.flatten()
galb_flat = g_alb_llavg.flatten()
solr_flat = s_olr_llavg.flatten()
salb_flat = s_alb_llavg.flatten()
molr_flat = m_olr_llavg.flatten()
malb_flat = m_alb_llavg.flatten()
nolr_flat = n_olr_llavg.flatten()
nalb_flat = n_alb_llavg.flatten()
eolr_flat = e_olr_llavg.flatten()
ealb_flat = e_alb_llavg.flatten()
aolr_flat = a_olr_llavg.flatten()
aalb_flat = a_alb_llavg.flatten()
uolr_flat = u_olr_llavg.flatten()
ualb_flat = u_alb_llavg.flatten()

# create figure
fig = plt.figure(figsize=(34,14), constrained_layout=True)
gs = fig.add_gridspec(2,16*5+1)
cmap = cm.gist_earth_r
ms = 30
levs = np.arange(-3.2, -1, 0.2)
cax = fig.add_subplot(gs[0,:16])
util.dennisplot("density", colr.values.flatten(), calb.values.flatten(), levels=levs, 
                model="CERES CCCM (All year, 2007-10)", region="TWP", 
                var_name="",units="", cmap=cmap, ax=cax, colorbar_on=False)
cax.plot([np.nanmean(colr)],[np.nanmean(calb)], 'r.', ms=ms)
print("1/10: made ceres syn1 axis...")
nax = fig.add_subplot(gs[0,16:16*2])
util.dennisplot("density", nolr_flat, nalb_flat, levels=levs, model="NICAM", region="TWP", 
           var_name="",units="", cmap=cmap, ax=nax, colorbar_on=False)
nax.plot([np.nanmean(nolr_flat)],[np.nanmean(nalb_flat)], 'r.', ms=ms)
print("2/10: made nicam axis...")
fax = fig.add_subplot(gs[0,16*2:16*3])
fax, cs = util.dennisplot("density", folr_flat, falb_flat, levels=levs, model="FV3", region="TWP", 
           var_name="",units="", cmap=cmap, ax=fax, colorbar_on=False)
fax.plot([np.nanmean(folr_flat)],[np.nanmean(falb_flat)], 'r.', ms=ms)
print("3/10: made fv3 axis...")
iax = fig.add_subplot(gs[0,16*3:16*4])
util.dennisplot("density", iolr_flat, ialb_flat, levels=levs, model="ICON", region="TWP", 
           var_name="",units="", cmap=cmap, ax=iax, colorbar_on=False)
iax.plot([np.nanmean(iolr_flat)],[np.nanmean(ialb_flat[~np.isnan(ialb_flat)])], 'r.', ms=ms)
print("4/10: made icon axis...")
schem_ax = fig.add_subplot(gs[1,:16])
util.proxy_schematic(ax=schem_ax)
print("5/10: made schematic axis...")
sax = fig.add_subplot(gs[0,16*4:16*5])
util.dennisplot("density", solr_flat, salb_flat, levels=levs, model="SAM", region="TWP", 
           var_name="",units="", cmap=cmap, ax=sax, colorbar_on=False)
sax.plot([np.nanmean(solr_flat)],[np.nanmean(salb_flat)], 'r.', ms=ms)
print("6/10: made sam axis...")
mpasax = fig.add_subplot(gs[1,16*3:16*4])
util.dennisplot("density", molr_flat, malb_flat, levels=levs, model="MPAS", region="TWP", 
           var_name="",units="", cmap=cmap, ax=mpasax, colorbar_on=False)
mpasax.plot([np.nanmean(molr_flat)],[np.nanmean(malb_flat[~np.isnan(malb_flat)])], 'r.', ms=ms)
print("7/10: made mpas axis...")
eax = fig.add_subplot(gs[1,16*2:16*3])
util.dennisplot("density", eolr_flat, ealb_flat, levels=levs, model="IFS", region="TWP", 
           var_name="",units="", cmap=cmap, ax=eax, colorbar_on=False)
eax.plot([np.nanmean(eolr_flat)],[np.nanmean(ealb_flat)], 'r.', ms=ms)
print("8/10: made ecmwf axis...")
aax = fig.add_subplot(gs[1,16:16*2])
util.dennisplot("density", aolr_flat, aalb_flat, levels=levs, model="ARPNH", region="TWP", 
           var_name="",units="", cmap=cmap, ax=aax, colorbar_on=False)
aax.plot([np.nanmean(aolr_flat)],[np.nanmean(aalb_flat)], 'r.', ms=ms)
print("9/10: made arpnh axis...")
uax = fig.add_subplot(gs[1,16*4:16*5])
util.dennisplot("density", uolr_flat, ualb_flat, levels=levs, model="UM", region="TWP", 
           var_name="",units="", cmap=cmap, ax=uax, colorbar_on=False)
uax.plot([np.nanmean(uolr_flat)],[np.nanmean(ualb_flat)], 'r.', ms=ms)
print("10/10: made um axis...")
#add obs and colorbar
cbax0 = fig.add_subplot(gs[0,-1])
cbax1 = fig.add_subplot(gs[1,-1])
cbar0 = plt.colorbar(cs, cax=cbax0)
cbar1 = plt.colorbar(cs, cax=cbax1)
cbar0.set_label("log10(pdf)", fontsize=18)
cbar1.set_label("log10(pdf)", fontsize=18)
cbar0.ax.tick_params(labelsize=14)
cbar1.ax.tick_params(labelsize=14)
print("... made colorbars ...")
plt.savefig("../plots/fig06_lifecycle_proxy.png", bbox_inches="tight")
print("... saved in ../plots/fig06_lifecycle_proxy.png")
plt.close()
