
#!usr/bin/env python
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import load, load01deg, analysis_parameters as ap

c = ap.COLORS

# %%
# load cccm
# ciwvt = load01deg.get_cccm("TWP")["Water vapor mixing ratio profile (GEOS5)"]
# ciwvs = load01deg.get_cccm("SHL")["Water vapor mixing ratio profile (GEOS5)"]
# ciwvn = load01deg.get_cccm("NAU")["Water vapor mixing ratio profile (GEOS5)"]

# cfwpt = load01deg.get_cccm("TWP")["iwp MODIS"]
# cfwps = load01deg.get_cccm("SHL")["iwp MODIS"]
# cfwpn = load01deg.get_cccm("NAU")["iwp MODIS"]
# # get category median values
# cat1 = 1000
# cat2 = 10
# cat3 = 0.1
# print("cccm...")
# c1t = (np.where(cfwpt>=cat1, ciwvt, np.nan))
# c2t = (np.where((cfwpt>=cat2)&(cfwpt<cat1), ciwvt, np.nan))
# c3t = (np.where((cfwpt>=cat3)&(cfwpt<cat2), ciwvt, np.nan))
# c4t = (np.where((cfwpt<cat3), ciwvt, np.nan))
# c1s = (np.where(cfwps>=cat1, ciwvs, np.nan))
# c2s = (np.where((cfwps>=cat2)&(cfwps<cat1), ciwvs, np.nan))
# c3s = (np.where((cfwps>=cat3)&(cfwps<cat2), ciwvs, np.nan))
# c4s = (np.where((cfwps<cat3), ciwvs, np.nan))
# c1n = (np.where(cfwpn>=cat1, ciwvn, np.nan))
# c2n = (np.where((cfwpn>=cat2)&(cfwpn<cat1), ciwvn, np.nan))
# c3n = (np.where((cfwpn>=cat3)&(cfwpn<cat2), ciwvn, np.nan))
# c4n = (np.where((cfwpn<cat3), ciwvn, np.nan))
# del ciwvt, ciwvs, ciwvn, cfwpt, cfwps, cfwpn
# print("\tCCCM TWP ({})\n\t1: {} {}%\n\t2: {} {}%\n\t3: {} {}\n\t4: {} {}".format(
#     len(c1t),
#     np.sum(~np.isnan(c1t)), np.sum(~np.isnan(c1t))*100/len(c1t),
#     np.sum(~np.isnan(c2t)), np.sum(~np.isnan(c2t))*100/len(c2t),
#     np.sum(~np.isnan(c3t)), np.sum(~np.isnan(c3t))*100/len(c3t),
#     np.sum(~np.isnan(c4t)), np.sum(~np.isnan(c4t))*100/len(c4t)
# ))
# print("\tCCCM SHL ({})\n\t1: {} {}%\n\t2: {} {}%\n\t3: {} {}\n\t4: {} {}".format(
#     len(c1s),
#     np.sum(~np.isnan(c1s)), np.sum(~np.isnan(c1s))*100/len(c1s),
#     np.sum(~np.isnan(c2s)), np.sum(~np.isnan(c2s))*100/len(c2s),
#     np.sum(~np.isnan(c3s)), np.sum(~np.isnan(c3s))*100/len(c3s),
#     np.sum(~np.isnan(c4s)), np.sum(~np.isnan(c4s))*100/len(c4s)
# ))
# print("\tCCCM NAU ({})\n\t1: {} {}%\n\t2: {} {}%\n\t3: {} {}\n\t4: {} {}".format(
#     len(c1n),
#     np.sum(~np.isnan(c1n)), np.sum(~np.isnan(c1n))*100/len(c1n),
#     np.sum(~np.isnan(c2n)), np.sum(~np.isnan(c2n))*100/len(c2n),
#     np.sum(~np.isnan(c3n)), np.sum(~np.isnan(c3n))*100/len(c3n),
#     np.sum(~np.isnan(c4n)), np.sum(~np.isnan(c4n))*100/len(c4n)
# ))
# c1t = np.nanmedian(c1t)
# c2t = np.nanmedian(c2t)
# c3t = np.nanmedian(c3t)
# c4t = np.nanmedian(c4t)
# c1s = np.nanmedian(c1s)
# c2s = np.nanmedian(c2s)
# c3s = np.nanmedian(c3s)
# c4s = np.nanmedian(c4s)
# c1n = np.nanmedian(c1n)
# c2n = np.nanmedian(c2n)
# c3n = np.nanmedian(c3n)
# c4n = np.nanmedian(c4n)

c1t = np.nan
c2t = np.nan
c3t = np.nan
c4t = np.nan
c1s = np.nan
c2s = np.nan
c3s = np.nan
c4s = np.nan
c1n = np.nan
c2n = np.nan
c3n = np.nan
c4n = np.nan
# print("... done")
print("skipping observed IWV until I can figure out how to integrate wv mixing ratio")

# %%
# define cat limits
cat1 = 1
cat2 = 1e-2
cat3 = 1e-4

# %%
# load sam
siwvt = load.get_iwv("SAM","TWP")
sfwpt = load.get_iwp("SAM","TWP", ice_only=False).values
s1t = np.nanmedian(np.where(sfwpt>=cat1, siwvt, np.nan))
s2t = np.nanmedian(np.where((sfwpt>=cat2)&(sfwpt<cat1), siwvt, np.nan))
s3t = np.nanmedian(np.where((sfwpt>=cat3)&(sfwpt<cat2), siwvt, np.nan))
s4t = np.nanmedian(np.where((sfwpt<cat3), siwvt, np.nan))
siwvs = load.get_iwv("SAM","SHL")
siwvn = load.get_iwv("SAM","NAU")
sfwps = load.get_iwp("SAM","SHL", ice_only=False).values
sfwpn = load.get_iwp("SAM","NAU", ice_only=False).values
print("sam...")
s1s = np.nanmedian(np.where(sfwps>=cat1, siwvs, np.nan))
s2s = np.nanmedian(np.where((sfwps>=cat2)&(sfwps<cat1), siwvs, np.nan))
s3s = np.nanmedian(np.where((sfwps>=cat3)&(sfwps<cat2), siwvs, np.nan))
s4s = np.nanmedian(np.where((sfwps<cat3), siwvs, np.nan))
s1n = np.nanmedian(np.where(sfwpn>=cat1, siwvn, np.nan))
s2n = np.nanmedian(np.where((sfwpn>=cat2)&(sfwpn<cat1), siwvn, np.nan))
s3n = np.nanmedian(np.where((sfwpn>=cat3)&(sfwpn<cat2), siwvn, np.nan))
s4n = np.nanmedian(np.where((sfwpn<cat3), siwvn, np.nan))
del siwvt, siwvs, siwvn, sfwpt, sfwps, sfwpn
print("... done.")

# %%
# load icon
iiwvt = load.get_iwv("ICON","TWP")
iiwvs = load.get_iwv("ICON","SHL")
iiwvn = load.get_iwv("ICON","NAU")
ifwpt = load.get_iwp("ICON","TWP", ice_only=False).values[::12]
ifwps = load.get_iwp("ICON","SHL", ice_only=False).values[::12]
ifwpn = load.get_iwp("ICON","NAU", ice_only=False).values[::12]
print("icon...")
i1t = np.nanmedian(np.where(ifwpt>=cat1, iiwvt, np.nan))
i2t = np.nanmedian(np.where((ifwpt>=cat2)&(ifwpt<cat1), iiwvt, np.nan))
i3t = np.nanmedian(np.where((ifwpt>=cat3)&(ifwpt<cat2), iiwvt, np.nan))
i4t = np.nanmedian(np.where((ifwpt<cat3), iiwvt, np.nan))
i1s = np.nanmedian(np.where(ifwps>=cat1, iiwvs, np.nan))
i2s = np.nanmedian(np.where((ifwps>=cat2)&(ifwps<cat1), iiwvs, np.nan))
i3s = np.nanmedian(np.where((ifwps>=cat3)&(ifwps<cat2), iiwvs, np.nan))
i4s = np.nanmedian(np.where((ifwps<cat3), iiwvs, np.nan))
i1n = np.nanmedian(np.where(ifwpn>=cat1, iiwvn, np.nan))
i2n = np.nanmedian(np.where((ifwpn>=cat2)&(ifwpn<cat1), iiwvn, np.nan))
i3n = np.nanmedian(np.where((ifwpn>=cat3)&(ifwpn<cat2), iiwvn, np.nan))
i4n = np.nanmedian(np.where((ifwpn<cat3), iiwvn, np.nan))
del iiwvt, iiwvs, iiwvn, ifwpt, ifwps, ifwpn
print("... done")

# %%
# load nicam
cat1 = 1
cat2 = 1e-2
cat3 = 1e-4
niwvt = load.get_iwv("NICAM","TWP")
nfwpt = load.get_iwp("NICAM","TWP", ice_only=False).values[::12,0]
n1t = np.nanmedian(np.where(nfwpt>=cat1, niwvt, np.nan))
n2t = np.nanmedian(np.where((nfwpt>=cat2)&(nfwpt<cat1), niwvt, np.nan))
n3t = np.nanmedian(np.where((nfwpt>=cat3)&(nfwpt<cat2), niwvt, np.nan))
n4t = np.nanmedian(np.where((nfwpt<cat3), niwvt, np.nan))
niwvs = load.get_iwv("NICAM","SHL")
niwvn = load.get_iwv("NICAM","NAU")
nfwps = load.get_iwp("NICAM","SHL", ice_only=False).values[::12,0]
nfwpn = load.get_iwp("NICAM","NAU", ice_only=False).values[::12,0]
print("nicam...")

n1s = np.nanmedian(np.where(nfwps>=cat1, niwvs, np.nan))
n2s = np.nanmedian(np.where((nfwps>=cat2)&(nfwps<cat1), niwvs, np.nan))
n3s = np.nanmedian(np.where((nfwps>=cat3)&(nfwps<cat2), niwvs, np.nan))
n4s = np.nanmedian(np.where((nfwps<cat3), niwvs, np.nan))
n1n = np.nanmedian(np.where(nfwpn>=cat1, niwvn, np.nan))
n2n = np.nanmedian(np.where((nfwpn>=cat2)&(nfwpn<cat1), niwvn, np.nan))
n3n = np.nanmedian(np.where((nfwpn>=cat3)&(nfwpn<cat2), niwvn, np.nan))
n4n = np.nanmedian(np.where((nfwpn<cat3), niwvn, np.nan))
del niwvt, niwvs, niwvn, nfwpt, nfwps, nfwpn
print("... done")
# %%
# load fv3 
fiwvt = load.get_iwv("FV3","TWP")
fiwvs = load.get_iwv("FV3","SHL")
fiwvn = load.get_iwv("FV3","NAU")
ffwpt = load.get_iwp("FV3","TWP", ice_only=False).values[::12]
ffwps = load.get_iwp("FV3","SHL", ice_only=False).values[::12]
ffwpn = load.get_iwp("FV3","NAU", ice_only=False).values[::12]
print("fv3...")
f1t = np.nanmedian(np.where(ffwpt>=cat1, fiwvt, np.nan))
f2t = np.nanmedian(np.where((ffwpt>=cat2)&(ffwpt<cat1), fiwvt, np.nan))
f3t = np.nanmedian(np.where((ffwpt>=cat3)&(ffwpt<cat2), fiwvt, np.nan))
f4t = np.nanmedian(np.where((ffwpt<cat3), fiwvt, np.nan))
f1s = np.nanmedian(np.where(ffwps>=cat1, fiwvs, np.nan))
f2s = np.nanmedian(np.where((ffwps>=cat2)&(ffwps<cat1), fiwvs, np.nan))
f3s = np.nanmedian(np.where((ffwps>=cat3)&(ffwps<cat2), fiwvs, np.nan))
f4s = np.nanmedian(np.where((ffwps<cat3), fiwvs, np.nan))
f1n = np.nanmedian(np.where(ffwpn>=cat1, fiwvn, np.nan))
f2n = np.nanmedian(np.where((ffwpn>=cat2)&(ffwpn<cat1), fiwvn, np.nan))
f3n = np.nanmedian(np.where((ffwpn>=cat3)&(ffwpn<cat2), fiwvn, np.nan))
f4n = np.nanmedian(np.where((ffwpn<cat3), fiwvn, np.nan))
del fiwvt, fiwvs, fiwvn, ffwpt, ffwps, ffwpn
print("... done")


# %%
# save as csv table
twp_fwp_cat_med = pd.DataFrame([[c1t/1000,n1t,f1t,i1t,s1t],[c2t/1000,n2t,f2t,i2t,s2t],
                                [c3t/1000,n3t,f3t,i3t,s3t],[c4t/1000,n4t,f4t,i4t,s4t]],
                                columns=["CCCM","NICAM","FV3","ICON","SAM"],
                                index=["CAT1","CAT2","CAT3","CS"]
                              )
shl_fwp_cat_med = pd.DataFrame([[c1s/1000,n1s,f1s,i1s,s1s],[c2s/1000,n2s,f2s,i2s,s2s],
                                [c3s/1000,n3s,f3s,i3s,s3s],[c4s/1000,n4s,f4s,i4s,s4s]],
                                columns=["CCCM","NICAM","FV3","ICON","SAM"],
                                index=["CAT1","CAT2","CAT3","CS"]
                              )
nau_fwp_cat_med = pd.DataFrame([[c1n/1000,n1n,f1n,i1n,s1n],[c2n/1000,n2n,f2n,i2n,s2n],
                                [c3n/1000,n3n,f3n,i3n,s3n],[c4n/1000,n4n,f4n,i4n,s4n]],
                                columns=["CCCM","NICAM","FV3","ICON","SAM"],
                                index=["CAT1","CAT2","CAT3","CS"]
                              )
twp_fwp_cat_med.to_csv("../tables/twp_iwv_cat_med_kgm-2.csv")
shl_fwp_cat_med.to_csv("../tables/shl_iwv_cat_med_kgm-2.csv")
nau_fwp_cat_med.to_csv("../tables/nau_iwv_cat_med_kgm-2.csv")


# %%
