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
from utility import load, util
import pandas as pd
import utility.analysis_parameters as ap
import matplotlib.pyplot as plt

# load all iwp 
hydro_type = "frozen"

print("Getting %s for hydrometeors"%(hydro_type))
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
# ciwp = xr.open_dataset(ap.CERES_TWP)["iwp MODIS"]
ciwp_twp = xr.open_dataset(ap.DARDAR_TWP)["iwp"]
ciwp_shl = xr.open_dataset(ap.DARDAR_SHL)["iwp"]
ciwp_nau = xr.open_dataset(ap.DARDAR_NAU)["iwp"]
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

# ds = pd.read_csv('../tables/cat_freq_allRegions.csv',index_col=0)
ntwp = pd.read_csv('../tables/NICAM_TWP_cat_alb_olr_ttl_cs.csv',index_col=0)
ftwp = pd.read_csv('../tables/FV3_TWP_cat_alb_olr_ttl_cs.csv',index_col=0)
itwp = pd.read_csv('../tables/ICON_TWP_cat_alb_olr_ttl_cs.csv',index_col=0)
stwp = pd.read_csv('../tables/SAM_TWP_cat_alb_olr_ttl_cs.csv',index_col=0)
nshl = pd.read_csv('../tables/NICAM_SHL_cat_alb_olr_ttl_cs.csv',index_col=0)
fshl = pd.read_csv('../tables/FV3_SHL_cat_alb_olr_ttl_cs.csv',index_col=0)
ishl = pd.read_csv('../tables/ICON_SHL_cat_alb_olr_ttl_cs.csv',index_col=0)
sshl = pd.read_csv('../tables/SAM_SHL_cat_alb_olr_ttl_cs.csv',index_col=0)
nnau = pd.read_csv('../tables/NICAM_NAU_cat_alb_olr_ttl_cs.csv',index_col=0)
fnau = pd.read_csv('../tables/FV3_NAU_cat_alb_olr_ttl_cs.csv',index_col=0)
inau = pd.read_csv('../tables/ICON_NAU_cat_alb_olr_ttl_cs.csv',index_col=0)
snau = pd.read_csv('../tables/SAM_NAU_cat_alb_olr_ttl_cs.csv',index_col=0)

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
fig = plt.figure(figsize=(5.5,14), constrained_layout=True)
gs = fig.add_gridspec(5,1,hspace=0.05)
xmid = (xedges[1:]+xedges[:-1])/2

cax = fig.add_subplot(gs[0,0])
nax = fig.add_subplot(gs[1,0])
fax = fig.add_subplot(gs[2,0])
iax = fig.add_subplot(gs[3,0])
sax = fig.add_subplot(gs[4,0])

s = 'y'
n = 'c'
a = 0.9

cax.bar(xmid[:cs_bin_ind], chist_twp[:cs_bin_ind], color='lightgray', width=0.1, alpha=a)
cax.bar(xmid[cs_bin_ind:lower_bin_ind], chist_twp[cs_bin_ind:lower_bin_ind], color='C0', alpha=a, width=0.1)
cax.bar(xmid[lower_bin_ind:upper_bin_ind], chist_twp[lower_bin_ind:upper_bin_ind], color='#C82F2F', alpha=a, width=0.1)
cax.bar(xmid[upper_bin_ind:], chist_twp[upper_bin_ind:], color='tab:green', width=0.1, alpha=a)
cax.step(xedges[1:], chist_twp, 'k', label='TWP')
cax.step(xedges[1:], chist_shl, s, label='SHL')
cax.step(xedges[1:], chist_nau, n, label='NAU')
cax.legend(loc=6)

fax.bar(xmid[:cs_bin_ind], fhist_twp[:cs_bin_ind], color='lightgray', width=0.1, alpha=a)
fax.bar(xmid[cs_bin_ind:lower_bin_ind], fhist_twp[cs_bin_ind:lower_bin_ind], color='C0', alpha=a, width=0.1)
fax.bar(xmid[lower_bin_ind:upper_bin_ind], fhist_twp[lower_bin_ind:upper_bin_ind], color='#C82F2F', alpha=a, width=0.1)
fax.bar(xmid[upper_bin_ind:], fhist_twp[upper_bin_ind:], color='tab:green', width=0.1, alpha=a)
fax.step(xedges[1:], fhist_twp, color='k')
fax.step(xedges[1:], fhist_shl, s)
fax.step(xedges[1:], fhist_nau, n)

iax.bar(xmid[:cs_bin_ind], ihist_twp[:cs_bin_ind], color='lightgray', width=0.1, alpha=a)
iax.bar(xmid[cs_bin_ind:lower_bin_ind], ihist_twp[cs_bin_ind:lower_bin_ind], color='C0', alpha=a, width=0.1)
iax.bar(xmid[lower_bin_ind:upper_bin_ind], ihist_twp[lower_bin_ind:upper_bin_ind], color='#C82F2F', alpha=a, width=0.1)
iax.bar(xmid[upper_bin_ind:], ihist_twp[upper_bin_ind:], color='tab:green', width=0.1, alpha=a)
iax.step(xedges[1:], ihist_twp, color='k')
iax.step(xedges[1:], ihist_shl, s)
iax.step(xedges[1:], ihist_nau, n)

sax.bar(xmid[:cs_bin_ind], shist_twp[:cs_bin_ind], color='lightgray', width=0.1, alpha=a)
sax.bar(xmid[cs_bin_ind:lower_bin_ind], shist_twp[cs_bin_ind:lower_bin_ind], color='C0', alpha=a, width=0.1)
sax.bar(xmid[lower_bin_ind:upper_bin_ind], shist_twp[lower_bin_ind:upper_bin_ind], color='#C82F2F', alpha=a, width=0.1)
sax.bar(xmid[upper_bin_ind:], shist_twp[upper_bin_ind:], color='tab:green', width=0.1, alpha=a)
sax.step(xedges[1:], shist_twp, color='k')
sax.step(xedges[1:], shist_shl, s)
sax.step(xedges[1:], shist_nau, n)

nax.bar(xmid[:cs_bin_ind], nhist_twp[:cs_bin_ind], color='lightgray', width=0.1, alpha=a)
nax.bar(xmid[cs_bin_ind:lower_bin_ind], nhist_twp[cs_bin_ind:lower_bin_ind], color='C0', alpha=a, width=0.1)
nax.bar(xmid[lower_bin_ind:upper_bin_ind], nhist_twp[lower_bin_ind:upper_bin_ind], color='#C82F2F', alpha=a, width=0.1)
nax.bar(xmid[upper_bin_ind:], nhist_twp[upper_bin_ind:], color='tab:green', width=0.1, alpha=a)
nax.step(xedges[1:], nhist_twp, color='k')
nax.step(xedges[1:], nhist_shl, s)
nax.step(xedges[1:], nhist_nau, n)

cax.set_ylim([0,0.075])
nax.set_ylim([0,0.075])
fax.set_ylim([0,0.075])
iax.set_ylim([0,0.075])
sax.set_ylim([0,0.075])

cax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
cax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
cax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
nax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
nax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
nax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
fax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
fax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
fax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
iax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
iax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
iax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
sax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
sax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
sax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")

color0, color1, color2, color3 = "k","k","k","k"

cax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
cax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
cax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
nax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
nax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
nax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
fax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
fax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
fax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
iax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
iax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
iax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")
sax.annotate("TWP: ", xy=(-2.95,0.06), xycoords="data", fontsize=17, color="k")
sax.annotate("SHL: ", xy=(-2.95,0.068), xycoords="data", fontsize=17, color="k")
sax.annotate("NAU: ", xy=(-2.95,0.052), xycoords="data", fontsize=17, color="k")

color0, color1, color2, color3 = "k","k","k","k"

cax.annotate("9%", xy=(3.1,0.06), xycoords="data", fontsize=17, color=color1)
cax.annotate("38%", xy=(1.7,0.06), xycoords="data", fontsize=17, color=color2)
cax.annotate("18%", xy=(-0.1,0.06), xycoords="data", fontsize=17, color=color3)
cax.annotate("35%", xy=(-1.6,0.06), xycoords="data", fontsize=17, color=color0)
nax.annotate(str(np.around(ntwp.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.06), xycoords="data", fontsize=17, color=color1)
nax.annotate(str(np.around(ntwp.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.06), xycoords="data", fontsize=17, color=color2)
nax.annotate(str(np.around(ntwp.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.06), xycoords="data", fontsize=17, color=color3)
nax.annotate(str(np.around(ntwp.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.06), xycoords="data", fontsize=17, color=color0)
fax.annotate(str(np.around(ftwp.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.06), xycoords="data", fontsize=17, color=color1)
fax.annotate(str(np.around(ftwp.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.06), xycoords="data", fontsize=17, color=color2)
fax.annotate(str(np.around(ftwp.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.06), xycoords="data", fontsize=17, color=color3)
fax.annotate(str(np.around(ftwp.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.06), xycoords="data", fontsize=17, color=color0)
iax.annotate(str(np.around(itwp.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.06), xycoords="data", fontsize=17, color=color1)
iax.annotate(str(np.around(itwp.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.06), xycoords="data", fontsize=17, color=color2)
iax.annotate(str(np.around(itwp.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.06), xycoords="data", fontsize=17, color=color3)
iax.annotate(str(np.around(itwp.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.06), xycoords="data", fontsize=17, color=color0)
sax.annotate(str(np.around(stwp.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.06), xycoords="data", fontsize=17, color=color1)
sax.annotate(str(np.around(stwp.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.06), xycoords="data", fontsize=17, color=color2)
sax.annotate(str(np.around(stwp.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.06), xycoords="data", fontsize=17, color=color3)
sax.annotate(str(np.around(stwp.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.06), xycoords="data", fontsize=17, color=color0)

cax.annotate("3%", xy=(3.1,0.068), xycoords="data", fontsize=17, color=color1)
cax.annotate("21%", xy=(1.7,0.068), xycoords="data", fontsize=17, color=color2)
cax.annotate("20%", xy=(-0.1,0.068), xycoords="data", fontsize=17, color=color3)
cax.annotate("56%", xy=(-1.6,0.068), xycoords="data", fontsize=17, color=color0)
nax.annotate(str(np.around(nshl.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.068), xycoords="data", fontsize=17, color=color1)
nax.annotate(str(np.around(nshl.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.068), xycoords="data", fontsize=17, color=color2)
nax.annotate(str(np.around(nshl.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.068), xycoords="data", fontsize=17, color=color3)
nax.annotate(str(np.around(nshl.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.068), xycoords="data", fontsize=17, color=color0)
fax.annotate(str(np.around(fshl.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.068), xycoords="data", fontsize=17, color=color1)
fax.annotate(str(np.around(fshl.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.068), xycoords="data", fontsize=17, color=color2)
fax.annotate(str(np.around(fshl.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.068), xycoords="data", fontsize=17, color=color3)
fax.annotate(str(np.around(fshl.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.068), xycoords="data", fontsize=17, color=color0)
iax.annotate(str(np.around(ishl.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.068), xycoords="data", fontsize=17, color=color1)
iax.annotate(str(np.around(ishl.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.068), xycoords="data", fontsize=17, color=color2)
iax.annotate(str(np.around(ishl.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.068), xycoords="data", fontsize=17, color=color3)
iax.annotate(str(np.around(ishl.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.068), xycoords="data", fontsize=17, color=color0)
sax.annotate(str(np.around(sshl.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.068), xycoords="data", fontsize=17, color=color1)
sax.annotate(str(np.around(sshl.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.068), xycoords="data", fontsize=17, color=color2)
sax.annotate(str(np.around(sshl.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.068), xycoords="data", fontsize=17, color=color3)
sax.annotate(str(np.around(sshl.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.068), xycoords="data", fontsize=17, color=color0)
             
             
cax.annotate("9%", xy=(3.1,0.052), xycoords="data", fontsize=17, color=color1)
cax.annotate("42%", xy=(1.7,0.052), xycoords="data", fontsize=17, color=color2)
cax.annotate("27%", xy=(-0.1,0.052), xycoords="data", fontsize=17, color=color3)
cax.annotate("22%", xy=(-1.6,0.052), xycoords="data", fontsize=17, color=color0)
nax.annotate(str(np.around(nnau.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", fontsize=17, color=color1)
nax.annotate(str(np.around(nnau.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", fontsize=17, color=color2)
nax.annotate(str(np.around(nnau.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", fontsize=17, color=color3)
nax.annotate(str(np.around(nnau.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", fontsize=17, color=color0)
fax.annotate(str(np.around(fnau.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", fontsize=17, color=color1)
fax.annotate(str(np.around(fnau.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", fontsize=17, color=color2)
fax.annotate(str(np.around(fnau.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", fontsize=17, color=color3)
fax.annotate(str(np.around(fnau.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", fontsize=17, color=color0)
iax.annotate(str(np.around(inau.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", fontsize=17, color=color1)
iax.annotate(str(np.around(inau.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", fontsize=17, color=color2)
iax.annotate(str(np.around(inau.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", fontsize=17, color=color3)
iax.annotate(str(np.around(inau.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", fontsize=17, color=color0)
sax.annotate(str(np.around(snau.catfreq.CAT1)).split(".")[0]+"%", xy=(3.1,0.052), xycoords="data", fontsize=17, color=color1)
sax.annotate(str(np.around(snau.catfreq.CAT2)).split(".")[0]+"%", xy=(1.7,0.052), xycoords="data", fontsize=17, color=color2)
sax.annotate(str(np.around(snau.catfreq.CAT3)).split(".")[0]+"%", xy=(-0.1,0.052), xycoords="data", fontsize=17, color=color3)
sax.annotate(str(np.around(snau.catfreq.CS)).split(".")[0]+"%", xy=(-1.6,0.052), xycoords="data", fontsize=17, color=color0)
          
cno = int(np.nanmean([cno_twp, cno_shl, cno_nau]))
nno = int(np.nanmean([nno_twp, nno_shl, nno_nau]))
fno = int(np.nanmean([fno_twp, fno_shl, fno_nau]))
ino = int(np.nanmean([ino_twp, ino_shl, ino_nau]))
sno = int(np.nanmean([sno_twp, sno_shl, sno_nau]))

cax.set_title("Frozen Water Path Histograms", size=17)
# cax.set_title("DARDAR \n {:} profiles".format(cno), size=17)
# nax.set_title("NICAM \n {:} profiles".format(nno), size=17)
# fax.set_title("FV3 \n {:} profiles".format(fno), size=17)
# iax.set_title("ICON\n {:} profiles".format(ino), size=17)
# sax.set_title("SAM \n {:} profiles".format(sno), size=17)

sax.set_xlabel("log10(IWP) [g/m$^2$]", fontsize=17)

cax.set_xticklabels([])
iax.set_xticklabels([])
fax.set_xticklabels([])
nax.set_xticklabels([])
sax.set_ylabel("SAM\nFraction of profiles", fontsize=17)
nax.set_ylabel("NICAM\nFraction of profiles", fontsize=17)
fax.set_ylabel("FV3\nFraction of profiles", fontsize=17)
iax.set_ylabel("ICON\nFraction of profiles", fontsize=17)
cax.set_ylabel("DARDAR\nFraction of profiles", fontsize=17)

cax.annotate("(a)", xy=(0.03,0.1), xycoords="axes fraction", fontsize=17)
nax.annotate("(b)", xy=(0.03,0.1), xycoords="axes fraction", fontsize=17)
fax.annotate("(c)", xy=(0.03,0.1), xycoords="axes fraction", fontsize=17)
iax.annotate("(d)", xy=(0.03,0.1), xycoords="axes fraction", fontsize=17)
sax.annotate("(e)", xy=(0.03,0.1), xycoords="axes fraction", fontsize=17)


ticks = np.arange(-3,4,1)
cax.set_xticks(ticks)
nax.set_xticks(ticks)
fax.set_xticks(ticks)
iax.set_xticks(ticks)
sax.set_xticks(ticks)

plt.savefig("../plots/fig10_iwp_hist_cat_nfis_allRegions.png", dpi=160)
print("saved to ../plots/fig10_iwp_hist_cat_nfis_allRegions.png")
plt.close()

print("Done!")
