#!/usr/bin/env python
""" fig11_iwp_all_regions.py
    Author: Sami Turbeville
    Updated: 18 Feb 2021
   
    This script reads in the frozen water path for FV3,
    SAM, GEOS, ICON and FV3. It then plots the histograms
    compared to DARDAR.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import utility.analysis_parameters as ap
from utility import load, util

fs = 18

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

# load all fwp 
obs = "DARDAR"

print("Getting frozen hydrometeors")
# giwp = util.iwp_wrt_pres("GEOS", REGION, hydro_type=hydro_type, geos_graupel=True)
print("geos zeros until we get new data")
iiwp_twp = load.get_iwp("ICON", "TWP", ice_only=False).values
iiwp_shl = load.get_iwp("ICON", "SHL", ice_only=False).values
iiwp_nau = load.get_iwp("ICON", "NAU", ice_only=False).values
print("loaded icon")
fiwp_twp = load.get_iwp("FV3", "TWP", ice_only=False).values
fiwp_shl = load.get_iwp("FV3", "SHL", ice_only=False).values
fiwp_nau = load.get_iwp("FV3", "NAU", ice_only=False).values
print("loaded fv3")
niwp_twp = load.get_iwp("NICAM", "TWP", ice_only=False).values
niwp_shl = load.get_iwp("NICAM", "SHL", ice_only=False).values
niwp_nau = load.get_iwp("NICAM", "NAU", ice_only=False).values
print("loaded nicam")
siwp_twp = load.get_iwp("SAM", "TWP", ice_only=False)
siwp_shl = load.get_iwp("SAM", "SHL", ice_only=False)
siwp_nau = load.get_iwp("SAM", "NAU", ice_only=False)
sno_twp = siwp_twp.count().values
sno_shl = siwp_shl.count().values
sno_nau = siwp_nau.count().values
siwp_twp = siwp_twp.values
siwp_shl = siwp_shl.values
siwp_nau = siwp_nau.values

print("loaded sam... convert to g/m2")
# convert to g/m2
niwp_twp = niwp_twp * 1000
fiwp_twp = fiwp_twp * 1000
iiwp_twp = iiwp_twp * 1000
siwp_twp = siwp_twp * 1000

niwp_shl = niwp_shl * 1000
fiwp_shl = fiwp_shl * 1000
iiwp_shl = iiwp_shl * 1000
siwp_shl = siwp_shl * 1000

niwp_nau = niwp_nau * 1000
fiwp_nau = fiwp_nau * 1000
iiwp_nau = iiwp_nau * 1000
siwp_nau = siwp_nau * 1000
print("    done... get obs")

# get observational data
if obs=="CCCM":
    ciwp_twp = xr.open_dataset(ap.CERES_TWP)["iwp MODIS"]
    ciwp_shl = xr.open_dataset(ap.CERES_SHL)["iwp MODIS"]
    ciwp_nau = xr.open_dataset(ap.CERES_NAU)["iwp MODIS"]
    c1t, c2t, c3t, c4t = 2, 48, 37, 13
    c1s, c2s, c3s, c4s = 1, 36, 38, 25
    c1n, c2n, c3n, c4n = 1, 29, 29, 41
else:
    ciwp_twp = xr.open_dataset(ap.DARDAR_TWP)["iwp"]
    ciwp_shl = xr.open_dataset(ap.DARDAR_SHL)["iwp"]
    ciwp_nau = xr.open_dataset(ap.DARDAR_NAU)["iwp"]
    c1t, c2t, c3t, c4t = 11, 48, 25, 15
    c1s, c2s, c3s, c4s = 4, 28, 31, 38
    c1n, c2n, c3n, c4n = 9, 42, 27, 22
    ciwp_twp = moving_average(ciwp_twp, n=3)
    ciwp_shl = moving_average(ciwp_shl, n=3)
    ciwp_nau = moving_average(ciwp_nau, n=3)
print("    done... bin by iwp")

# calculate histograms
bins = np.arange(-3,4,0.1)
chist_twp, _ = np.histogram(np.log10(ciwp_twp), bins=bins)
ihist_twp, xedges = np.histogram(np.log10(iiwp_twp.flatten()), bins=bins)
fhist_twp, _ = np.histogram(np.log10(fiwp_twp.flatten()), bins=bins)
shist_twp, _ = np.histogram(np.log10(siwp_twp.flatten()), bins=bins)
nhist_twp, _ = np.histogram(np.log10(niwp_twp.flatten()), bins=bins)

chist_shl, _ = np.histogram(np.log10(ciwp_shl), bins=bins)
ihist_shl, xedges = np.histogram(np.log10(iiwp_shl.flatten()), bins=bins)
fhist_shl, _ = np.histogram(np.log10(fiwp_shl.flatten()), bins=bins)
shist_shl, _ = np.histogram(np.log10(siwp_shl.flatten()), bins=bins)
nhist_shl, _ = np.histogram(np.log10(niwp_shl.flatten()), bins=bins)

chist_nau, _ = np.histogram(np.log10(ciwp_nau), bins=bins)
ihist_nau, xedges = np.histogram(np.log10(iiwp_nau.flatten()), bins=bins)
fhist_nau, _ = np.histogram(np.log10(fiwp_nau.flatten()), bins=bins)
shist_nau, _ = np.histogram(np.log10(siwp_nau.flatten()), bins=bins)
nhist_nau, _ = np.histogram(np.log10(niwp_nau.flatten()), bins=bins)

print("get num of prof")
# get number of profiles
cno_twp = (len(ciwp_twp.values.flatten()))
ino_twp = (len(iiwp_twp.flatten()))
# sno_twp = (len(siwp_twp.flatten()))
fno_twp = (len(fiwp_twp.flatten()))
nno_twp = (len(niwp_twp.flatten()))

cno_shl = (len(ciwp_shl.values.flatten()))
ino_shl = (len(iiwp_shl.flatten()))
# sno_shl = (len(siwp_shl.flatten()))
fno_shl = (len(fiwp_shl.flatten()))
nno_shl = (len(niwp_shl.flatten()))

cno_nau = (len(ciwp_nau.values.flatten()))
ino_nau = (len(iiwp_nau.flatten()))
# sno_nau = (len(siwp_nau.flatten()))
fno_nau = (len(fiwp_nau.flatten()))
nno_nau = (len(niwp_nau.flatten()))

print("counting percent in each category...")
# color the bins by category
# clear sky threshold = 0.01 g/m2
thres=0.1
cs_bin_ind = np.argmin(abs(np.log10(thres)-bins))
# thin cirrus to anvil threshold = 10 g/m2
lower_bin_ind = np.argmin(abs(1-bins))
# convective threshold = 1000 g/m2
upper_bin_ind = np.argmin(abs(3-bins))
# sum count in each category and divide by total

# ds = pd.read_csv('../tables/cat_freq_allRegions.csv',index_col=0)
ntwp = pd.read_csv('../tables/NICAM_TWP.csv',index_col=0)
ftwp = pd.read_csv('../tables/FV3_TWP.csv',index_col=0)
itwp = pd.read_csv('../tables/ICON_TWP.csv',index_col=0)
stwp = pd.read_csv('../tables/SAM_TWP.csv',index_col=0)
nshl = pd.read_csv('../tables/NICAM_SHL.csv',index_col=0)
fshl = pd.read_csv('../tables/FV3_SHL.csv',index_col=0)
ishl = pd.read_csv('../tables/ICON_SHL.csv',index_col=0)
sshl = pd.read_csv('../tables/SAM_SHL.csv',index_col=0)
nnau = pd.read_csv('../tables/NICAM_NAU.csv',index_col=0)
fnau = pd.read_csv('../tables/FV3_NAU.csv',index_col=0)
inau = pd.read_csv('../tables/ICON_NAU.csv',index_col=0)
snau = pd.read_csv('../tables/SAM_NAU.csv',index_col=0)

print("normalize:")
# normalize
chist_twp = chist_twp/cno_twp
ihist_twp = ihist_twp/ino_twp
shist_twp = shist_twp/sno_twp
fhist_twp = fhist_twp/fno_twp
nhist_twp = nhist_twp/nno_twp

chist_shl = chist_shl/cno_shl
ihist_shl = ihist_shl/ino_shl
shist_shl = shist_shl/sno_shl
fhist_shl = fhist_shl/fno_shl
nhist_shl = nhist_shl/nno_shl

chist_nau = chist_nau/cno_nau
ihist_nau = ihist_nau/ino_nau
shist_nau = shist_nau/sno_nau
fhist_nau = fhist_nau/fno_nau
nhist_nau = nhist_nau/nno_nau

print("creating figure...")
# plot it
# fig = plt.figure(figsize=(20,8), constrained_layout=True)
# gs = fig.add_gridspec(2,3,hspace=-0.1, wspace=-0.1)
# cax = fig.add_subplot(gs[0,0])
# nax = fig.add_subplot(gs[0,1])
# fax = fig.add_subplot(gs[0,2])
# iax = fig.add_subplot(gs[1,1])
# sax = fig.add_subplot(gs[1,2])

fig, [[cax, nax, fax],[lax, iax, sax]] = plt.subplots(2,3,
                                            sharey=True, sharex=True,
                                            figsize=(21,8))
lax.axis("off")

s = 'tab:orange'
n = 'tab:purple'
t = 'tab:green'
a = 0.9
xmid = (xedges[1:]+xedges[:-1])/2

cax.axvline(3, color="k", alpha=0.4, linestyle="dashed")
cax.axvline(1, color="k", alpha=0.4, linestyle="dashed")
cax.axvline(-1, color="k", alpha=0.4, linestyle="dashed")
nax.axvline(3, color="k", alpha=0.4, linestyle="dashed")
nax.axvline(1, color="k", alpha=0.4, linestyle="dashed")
nax.axvline(-1, color="k", alpha=0.4, linestyle="dashed")
fax.axvline(3, color="k", alpha=0.4, linestyle="dashed")
fax.axvline(1, color="k", alpha=0.4, linestyle="dashed")
fax.axvline(-1, color="k", alpha=0.4, linestyle="dashed")
iax.axvline(3, color="k", alpha=0.4, linestyle="dashed")
iax.axvline(1, color="k", alpha=0.4, linestyle="dashed")
iax.axvline(-1, color="k", alpha=0.4, linestyle="dashed")
sax.axvline(3, color="k", alpha=0.4, linestyle="dashed")
sax.axvline(1, color="k", alpha=0.4, linestyle="dashed")
sax.axvline(-1, color="k", alpha=0.4, linestyle="dashed")

cax.step(xedges[1:], chist_shl, s, label='SHL')
cax.step(xedges[1:], chist_twp, t, label='TWP')
cax.step(xedges[1:], chist_nau, n, label='NAU')

fax.step(xedges[1:], fhist_twp, color=t)
fax.step(xedges[1:], fhist_shl, color=s)
fax.step(xedges[1:], fhist_nau, color=n)

iax.step(xedges[1:], ihist_twp, color=t)
iax.step(xedges[1:], ihist_shl, color=s)
iax.step(xedges[1:], ihist_nau, color=n)

sax.step(xedges[1:], shist_twp, color=t)
sax.step(xedges[1:], shist_shl, color=s)
sax.step(xedges[1:], shist_nau, color=n)

nax.step(xedges[1:], nhist_twp, color=t)
nax.step(xedges[1:], nhist_shl, color=s)
nax.step(xedges[1:], nhist_nau, color=n)

cax.set_ylim([0,0.076])
nax.set_ylim([0,0.076])
fax.set_ylim([0,0.076])
iax.set_ylim([0,0.076])
sax.set_ylim([0,0.076])
cax.set_xlim([-3.2,4])
nax.set_xlim([-3.2,4])
fax.set_xlim([-3.2,4])
iax.set_xlim([-3.2,4])
sax.set_xlim([-3.2,4])

cax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
cax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
cax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
nax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
nax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
nax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
fax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
fax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
fax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
iax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
iax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
iax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
sax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
sax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
sax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")

color0, color1, color2, color3 = "k","k","k","k"

cax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
cax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
cax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
nax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
nax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
nax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
fax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
fax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
fax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
iax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
iax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
iax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")
sax.annotate("TWP: ", xy=(-2.95,0.053), xycoords="data", fontsize=fs, color="k")
sax.annotate("SHL: ", xy=(-2.95,0.061), xycoords="data", fontsize=fs, color="k")
sax.annotate("NAU: ", xy=(-2.95,0.045), xycoords="data", fontsize=fs, color="k")

cax.annotate("CAT 1", xy=(3.,0.068), xycoords="data", fontsize=fs, color=color1)
cax.annotate("CAT 2", xy=(1.6,0.068), xycoords="data", fontsize=fs, color=color2)
cax.annotate("CAT 3", xy=(-0.2,0.068), xycoords="data", fontsize=fs, color=color3)
# cax.annotate("CS", xy=(-1.8,0.068), xycoords="data", fontsize=fs, color=color0)
nax.annotate("CAT 1", xy=(3.,0.068), xycoords="data", fontsize=fs, color=color1)
nax.annotate("CAT 2", xy=(1.6,0.068), xycoords="data", fontsize=fs, color=color2)
nax.annotate("CAT 3", xy=(-0.2,0.068), xycoords="data", fontsize=fs, color=color3)
# nax.annotate("CS", xy=(-1.8,0.068), xycoords="data", fontsize=fs, color=color0)
fax.annotate("CAT 1", xy=(3.,0.068), xycoords="data", fontsize=fs, color=color1)
fax.annotate("CAT 2", xy=(1.6,0.068), xycoords="data", fontsize=fs, color=color2)
fax.annotate("CAT 3", xy=(-0.2,0.068), xycoords="data", fontsize=fs, color=color3)
# fax.annotate("CS", xy=(-1.8,0.068), xycoords="data", fontsize=fs, color=color0)
iax.annotate("CAT 1", xy=(3.,0.068), xycoords="data", fontsize=fs, color=color1)
iax.annotate("CAT 2", xy=(1.6,0.068), xycoords="data", fontsize=fs, color=color2)
iax.annotate("CAT 3", xy=(-0.2,0.068), xycoords="data", fontsize=fs, color=color3)
# iax.annotate("CS", xy=(-1.8,0.068), xycoords="data", fontsize=fs, color=color0)
sax.annotate("CAT 1", xy=(3.,0.068), xycoords="data", fontsize=fs, color=color1)
sax.annotate("CAT 2", xy=(1.6,0.068), xycoords="data", fontsize=fs, color=color2)
sax.annotate("CAT 3", xy=(-0.2,0.068), xycoords="data", fontsize=fs, color=color3)
# sax.annotate("CS", xy=(-1.8,0.068), xycoords="data", fontsize=fs, color=color0)


color0, color1, color2, color3 = "k","k","k","k"

cax.annotate(str(c1t), xy=(3.1,0.053), xycoords="data", fontsize=fs, color=color1)
cax.annotate(str(c2t), xy=(1.7,0.053), xycoords="data", fontsize=fs, color=color2)
cax.annotate(str(c3t), xy=(-0.1,0.053), xycoords="data", fontsize=fs, color=color3)
cax.annotate(str(c4t), xy=(-1.8,0.053), xycoords="data", fontsize=fs, color=color0)
nax.annotate(str(np.around(ntwp.freq.CAT1)).split(".")[0], xy=(3.3,0.053), xycoords="data", fontsize=fs, color=color1)
nax.annotate(str(np.around(ntwp.freq.CAT2)).split(".")[0], xy=(1.8,0.053), xycoords="data", fontsize=fs, color=color2)
nax.annotate(str(np.around(ntwp.freq.CAT3)).split(".")[0], xy=(-0.1,0.053), xycoords="data", fontsize=fs, color=color3)
nax.annotate(str(np.around(ntwp.freq.CS)).split(".")[0], xy=(-1.7,0.053), xycoords="data", fontsize=fs, color=color0)
fax.annotate(str(np.around(ftwp.freq.CAT1)).split(".")[0], xy=(3.3,0.053), xycoords="data", fontsize=fs, color=color1)
fax.annotate(str(np.around(ftwp.freq.CAT2)).split(".")[0], xy=(1.8,0.053), xycoords="data", fontsize=fs, color=color2)
fax.annotate(str(np.around(ftwp.freq.CAT3)).split(".")[0], xy=(-0.1,0.053), xycoords="data", fontsize=fs, color=color3)
fax.annotate(str(np.around(ftwp.freq.CS)).split(".")[0], xy=(-1.8,0.053), xycoords="data", fontsize=fs, color=color0)
iax.annotate(str(np.around(itwp.freq.CAT1)).split(".")[0], xy=(3.3,0.053), xycoords="data", fontsize=fs, color=color1)
iax.annotate(str(np.around(itwp.freq.CAT2)).split(".")[0], xy=(1.8,0.053), xycoords="data", fontsize=fs, color=color2)
iax.annotate(str(np.around(itwp.freq.CAT3)).split(".")[0], xy=(-0.1,0.053), xycoords="data", fontsize=fs, color=color3)
iax.annotate(str(np.around(itwp.freq.CS)).split(".")[0], xy=(-1.8,0.053), xycoords="data", fontsize=fs, color=color0)
sax.annotate(str(np.around(stwp.freq.CAT1)).split(".")[0], xy=(3.3,0.053), xycoords="data", fontsize=fs, color=color1)
sax.annotate(str(np.around(stwp.freq.CAT2)).split(".")[0], xy=(1.8,0.053), xycoords="data", fontsize=fs, color=color2)
sax.annotate(str(np.around(stwp.freq.CAT3)).split(".")[0], xy=(-0.1,0.053), xycoords="data", fontsize=fs, color=color3)
sax.annotate(str(np.around(stwp.freq.CS)).split(".")[0], xy=(-1.8,0.053), xycoords="data", fontsize=fs, color=color0)

cax.annotate(c1s, xy=(3.1,0.061), xycoords="data", fontsize=fs, color=color1)
cax.annotate(c2s, xy=(1.7,0.061), xycoords="data", fontsize=fs, color=color2)
cax.annotate(c3s, xy=(-0.1,0.061), xycoords="data", fontsize=fs, color=color3)
cax.annotate(c4s, xy=(-1.8,0.061), xycoords="data", fontsize=fs, color=color0)
nax.annotate(str(np.around(nshl.freq.CAT1)).split(".")[0], xy=(3.3,0.061), xycoords="data", fontsize=fs, color=color1)
nax.annotate(str(np.around(nshl.freq.CAT2)).split(".")[0], xy=(1.8,0.061), xycoords="data", fontsize=fs, color=color2)
nax.annotate(str(np.around(nshl.freq.CAT3)).split(".")[0], xy=(-0.1,0.061), xycoords="data", fontsize=fs, color=color3)
nax.annotate(str(np.around(nshl.freq.CS)).split(".")[0], xy=(-1.7,0.061), xycoords="data", fontsize=fs, color=color0)
fax.annotate(str(np.around(fshl.freq.CAT1)).split(".")[0], xy=(3.3,0.061), xycoords="data", fontsize=fs, color=color1)
fax.annotate(str(np.around(fshl.freq.CAT2)).split(".")[0], xy=(1.8,0.061), xycoords="data", fontsize=fs, color=color2)
fax.annotate(str(np.around(fshl.freq.CAT3)).split(".")[0], xy=(-0.1,0.061), xycoords="data", fontsize=fs, color=color3)
fax.annotate(str(np.around(fshl.freq.CS)).split(".")[0], xy=(-1.8,0.061), xycoords="data", fontsize=fs, color=color0)
iax.annotate(str(np.around(ishl.freq.CAT1)).split(".")[0], xy=(3.3,0.061), xycoords="data", fontsize=fs, color=color1)
iax.annotate(str(np.around(ishl.freq.CAT2)).split(".")[0], xy=(1.8,0.061), xycoords="data", fontsize=fs, color=color2)
iax.annotate(str(np.around(ishl.freq.CAT3)).split(".")[0], xy=(-0.1,0.061), xycoords="data", fontsize=fs, color=color3)
iax.annotate(str(np.around(ishl.freq.CS)).split(".")[0], xy=(-1.8,0.061), xycoords="data", fontsize=fs, color=color0)
sax.annotate(str(np.around(sshl.freq.CAT1)).split(".")[0], xy=(3.3,0.061), xycoords="data", fontsize=fs, color=color1)
sax.annotate(str(np.around(sshl.freq.CAT2)).split(".")[0], xy=(1.8,0.061), xycoords="data", fontsize=fs, color=color2)
sax.annotate(str(np.around(sshl.freq.CAT3)).split(".")[0], xy=(-0.1,0.061), xycoords="data", fontsize=fs, color=color3)
sax.annotate(str(np.around(sshl.freq.CS)).split(".")[0], xy=(-1.8,0.061), xycoords="data", fontsize=fs, color=color0)


cax.annotate(c1n, xy=(3.1,0.045), xycoords="data", fontsize=fs, color=color1)
cax.annotate(c2n, xy=(1.7,0.045), xycoords="data", fontsize=fs, color=color2)
cax.annotate(c3n, xy=(-0.1,0.045), xycoords="data", fontsize=fs, color=color3)
cax.annotate(c4n, xy=(-1.8,0.045), xycoords="data", fontsize=fs, color=color0)
nax.annotate(str(np.around(nnau.freq.CAT1)).split(".")[0], xy=(3.3,0.045), xycoords="data", fontsize=fs, color=color1)
nax.annotate(str(np.around(nnau.freq.CAT2)).split(".")[0], xy=(1.8,0.045), xycoords="data", fontsize=fs, color=color2)
nax.annotate(str(np.around(nnau.freq.CAT3)).split(".")[0], xy=(-0.1,0.045), xycoords="data", fontsize=fs, color=color3)
nax.annotate(str(np.around(nnau.freq.CS)).split(".")[0], xy=(-1.7,0.045), xycoords="data", fontsize=fs, color=color0)
fax.annotate(str(np.around(fnau.freq.CAT1)).split(".")[0], xy=(3.3,0.045), xycoords="data", fontsize=fs, color=color1)
fax.annotate(str(np.around(fnau.freq.CAT2)).split(".")[0], xy=(1.8,0.045), xycoords="data", fontsize=fs, color=color2)
fax.annotate(str(np.around(fnau.freq.CAT3)).split(".")[0], xy=(-0.1,0.045), xycoords="data", fontsize=fs, color=color3)
fax.annotate(str(np.around(fnau.freq.CS)).split(".")[0], xy=(-1.8,0.045), xycoords="data", fontsize=fs, color=color0)
iax.annotate(str(np.around(inau.freq.CAT1)).split(".")[0], xy=(3.3,0.045), xycoords="data", fontsize=fs, color=color1)
iax.annotate(str(np.around(inau.freq.CAT2)).split(".")[0], xy=(1.8,0.045), xycoords="data", fontsize=fs, color=color2)
iax.annotate(str(np.around(inau.freq.CAT3)).split(".")[0], xy=(-0.1,0.045), xycoords="data", fontsize=fs, color=color3)
iax.annotate(str(np.around(inau.freq.CS)).split(".")[0], xy=(-1.8,0.045), xycoords="data", fontsize=fs, color=color0)
sax.annotate(str(np.around(snau.freq.CAT1)).split(".")[0], xy=(3.3,0.045), xycoords="data", fontsize=fs, color=color1)
sax.annotate(str(np.around(snau.freq.CAT2)).split(".")[0], xy=(1.8,0.045), xycoords="data", fontsize=fs, color=color2)
sax.annotate(str(np.around(snau.freq.CAT3)).split(".")[0], xy=(-0.1,0.045), xycoords="data", fontsize=fs, color=color3)
sax.annotate(str(np.around(snau.freq.CS)).split(".")[0], xy=(-1.8,0.045), xycoords="data", fontsize=fs, color=color0)
          
cno = int(np.nanmean([cno_twp, cno_shl, cno_nau]))
nno = int(np.nanmean([nno_twp, nno_shl, nno_nau]))
fno = int(np.nanmean([fno_twp, fno_shl, fno_nau]))
ino = int(np.nanmean([ino_twp, ino_shl, ino_nau]))
sno = int(np.nanmean([sno_twp, sno_shl, sno_nau]))

# fig.suptitle("Frozen Water Path Histograms", size=17)
# cax.set_title("DARDAR \n {:} profiles".format(cno), size=17)
# nax.set_title("NICAM \n {:} profiles".format(nno), size=17)
# fax.set_title("FV3 \n {:} profiles".format(fno), size=17)
# iax.set_title("ICON\n {:} profiles".format(ino), size=17)
# sax.set_title("SAM \n {:} profiles".format(sno), size=17)

sax.set_xlabel("log$_{10}$FWP (g~m$^{-2}$)", fontsize=fs)
cax.set_xlabel("log$_{10}$FWP (g~m$^{-2}$)", fontsize=fs)
iax.set_xlabel("log$_{10}$FWP (g~m$^{-2}$)", fontsize=fs)

# cax.set_xticklabels([])
# iax.set_xticklabels([])
# fax.set_xticklabels([])
# nax.set_xticklabels([])
# nax.set_yticklabels([])
# fax.set_yticklabels([])
# sax.set_yticklabels([])
sax.set_title("SAM", fontsize=fs+2)
nax.set_title("NICAM", fontsize=fs+2)
fax.set_title("FV3", fontsize=fs+2)
iax.set_title("ICON", fontsize=fs+2)
cax.set_title(obs, fontsize=fs+2)
cax.set_ylabel("Fraction of profiles", fontsize=fs)
iax.set_ylabel("Fraction of profiles", fontsize=fs)
cax.tick_params(labelsize=fs-2)
iax.tick_params(labelsize=fs-2)
sax.tick_params(labelsize=fs-2)

cax.annotate("(a)", xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=fs+4, weight="bold")
nax.annotate("(b)", xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=fs+4, weight="bold")
fax.annotate("(c)", xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=fs+4, weight="bold")
iax.annotate("(d)", xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=fs+4, weight="bold")
sax.annotate("(e)", xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=fs+4, weight="bold")

cax.tick_params(labelsize=fs-4)

h,l = cax.get_legend_handles_labels()
lax.legend(h,l,loc="center", fontsize=fs, title="Regions", title_fontsize=fs+6)

ticks = np.arange(-3,4,1)
cax.set_xticks(ticks)
nax.set_xticks(ticks)
fax.set_xticks(ticks)
iax.set_xticks(ticks)
sax.set_xticks(ticks)

# cax.grid(True)
# nax.grid(True)
# fax.grid(True)
# iax.grid(True)
# sax.grid(True)
# cax.set_axisbelow(True)
# nax.set_axisbelow(True)
# fax.set_axisbelow(True)
# iax.set_axisbelow(True)
# sax.set_axisbelow(True)

plt.savefig("../plots/fig11_iwp_hist_cat_nfis_allRegions_{}_3x2_coarsened.png".format(obs), dpi=160)
print("saved to ../plots/fig11_iwp_hist_cat_nfis_allRegions_{}_3x2_coarsened.png".format(obs))
plt.close()

print("Done!")
