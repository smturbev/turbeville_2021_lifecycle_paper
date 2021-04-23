
#!usr/bin/env python
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import load, load01deg, analysis_parameters as ap

c = ap.COLORS

# %%
# load cccm
clwpt = load01deg.get_cccm("TWP")["lwp MODIS"]
clwps = load01deg.get_cccm("SHL")["lwp MODIS"]
clwpn = load01deg.get_cccm("NAU")["lwp MODIS"]
cfwpt = load01deg.get_cccm("TWP")["iwp MODIS"]
cfwps = load01deg.get_cccm("SHL")["iwp MODIS"]
cfwpn = load01deg.get_cccm("NAU")["iwp MODIS"]
# get category median values
cat1 = 1000
cat2 = 10
cat3 = 0.1
print("cccm...")
c1t = (np.where(cfwpt>=cat1, clwpt, np.nan))
c2t = (np.where((cfwpt>=cat2)&(cfwpt<cat1), clwpt, np.nan))
c3t = (np.where((cfwpt>=cat3)&(cfwpt<cat2), clwpt, np.nan))
c4t = (np.where((cfwpt<cat3), clwpt, np.nan))
c1s = (np.where(cfwps>=cat1, clwps, np.nan))
c2s = (np.where((cfwps>=cat2)&(cfwps<cat1), clwps, np.nan))
c3s = (np.where((cfwps>=cat3)&(cfwps<cat2), clwps, np.nan))
c4s = (np.where((cfwps<cat3), clwps, np.nan))
c1n = (np.where(cfwpn>=cat1, clwpn, np.nan))
c2n = (np.where((cfwpn>=cat2)&(cfwpn<cat1), clwpn, np.nan))
c3n = (np.where((cfwpn>=cat3)&(cfwpn<cat2), clwpn, np.nan))
c4n = (np.where((cfwpn<cat3), clwpn, np.nan))
del clwpt, clwps, clwpn, cfwpt, cfwps, cfwpn
print("\tCCCM TWP ({})\n\t1: {} {}%\n\t2: {} {}%\n\t3: {} {}\n\t4: {} {}".format(
    len(c1t),
    np.sum(~np.isnan(c1t)), np.sum(~np.isnan(c1t))*100/len(c1t),
    np.sum(~np.isnan(c2t)), np.sum(~np.isnan(c2t))*100/len(c2t),
    np.sum(~np.isnan(c3t)), np.sum(~np.isnan(c3t))*100/len(c3t),
    np.sum(~np.isnan(c4t)), np.sum(~np.isnan(c4t))*100/len(c4t)
))
print("\tCCCM SHL ({})\n\t1: {} {}%\n\t2: {} {}%\n\t3: {} {}\n\t4: {} {}".format(
    len(c1s),
    np.sum(~np.isnan(c1s)), np.sum(~np.isnan(c1s))*100/len(c1s),
    np.sum(~np.isnan(c2s)), np.sum(~np.isnan(c2s))*100/len(c2s),
    np.sum(~np.isnan(c3s)), np.sum(~np.isnan(c3s))*100/len(c3s),
    np.sum(~np.isnan(c4s)), np.sum(~np.isnan(c4s))*100/len(c4s)
))
print("\tCCCM NAU ({})\n\t1: {} {}%\n\t2: {} {}%\n\t3: {} {}\n\t4: {} {}".format(
    len(c1n),
    np.sum(~np.isnan(c1n)), np.sum(~np.isnan(c1n))*100/len(c1n),
    np.sum(~np.isnan(c2n)), np.sum(~np.isnan(c2n))*100/len(c2n),
    np.sum(~np.isnan(c3n)), np.sum(~np.isnan(c3n))*100/len(c3n),
    np.sum(~np.isnan(c4n)), np.sum(~np.isnan(c4n))*100/len(c4n)
))
c1t = np.nanmedian(c1t)
c2t = np.nanmedian(c2t)
c3t = np.nanmedian(c3t)
c4t = np.nanmedian(c4t)
c1s = np.nanmedian(c1s)
c2s = np.nanmedian(c2s)
c3s = np.nanmedian(c3s)
c4s = np.nanmedian(c4s)
c1n = np.nanmedian(c1n)
c2n = np.nanmedian(c2n)
c3n = np.nanmedian(c3n)
c4n = np.nanmedian(c4n)

print("... done")
print("skipping observed lwp until I can figure out how to integrate wv mixing ratio")

# %%
# load sam
cat1 = 1
cat2 = 1e-2
cat3 = 1e-4
slwpt = load.get_lwp("SAM","TWP").values
sfwpt = load.get_iwp("SAM","TWP", ice_only=False).values
s1t = np.nanmedian(np.where(sfwpt>=cat1, slwpt, np.nan))
s2t = np.nanmedian(np.where((sfwpt>=cat2)&(sfwpt<cat1), slwpt, np.nan))
s3t = np.nanmedian(np.where((sfwpt>=cat3)&(sfwpt<cat2), slwpt, np.nan))
s4t = np.nanmedian(np.where((sfwpt<cat3), slwpt, np.nan))
slwps = load.get_lwp("SAM","SHL").values
slwpn = load.get_lwp("SAM","NAU").values
sfwps = load.get_iwp("SAM","SHL", ice_only=False).values
sfwpn = load.get_iwp("SAM","NAU", ice_only=False).values
print("sam...")
s1s = np.nanmedian(np.where(sfwps>=cat1, slwps, np.nan))
s2s = np.nanmedian(np.where((sfwps>=cat2)&(sfwps<cat1), slwps, np.nan))
s3s = np.nanmedian(np.where((sfwps>=cat3)&(sfwps<cat2), slwps, np.nan))
s4s = np.nanmedian(np.where((sfwps<cat3), slwps, np.nan))
s1n = np.nanmedian(np.where(sfwpn>=cat1, slwpn, np.nan))
s2n = np.nanmedian(np.where((sfwpn>=cat2)&(sfwpn<cat1), slwpn, np.nan))
s3n = np.nanmedian(np.where((sfwpn>=cat3)&(sfwpn<cat2), slwpn, np.nan))
s4n = np.nanmedian(np.where((sfwpn<cat3), slwpn, np.nan))
del slwpt, slwps, slwpn, sfwpt, sfwps, sfwpn
print("... done.")

# %%
# load nicam
cat1 = 1
cat2 = 1e-2
cat3 = 1e-4
nlwpt = load.get_lwp("NICAM","TWP").values[::12]
nfwpt = load.get_iwp("NICAM","TWP", ice_only=False).values[::12]
n1t = np.nanmedian(np.where(nfwpt>=cat1, nlwpt, np.nan))
n2t = np.nanmedian(np.where((nfwpt>=cat2)&(nfwpt<cat1), nlwpt, np.nan))
n3t = np.nanmedian(np.where((nfwpt>=cat3)&(nfwpt<cat2), nlwpt, np.nan))
n4t = np.nanmedian(np.where((nfwpt<cat3), nlwpt, np.nan))
nlwps = load.get_lwp("NICAM","SHL").values[::12]
nlwpn = load.get_lwp("NICAM","NAU").values[::12]
nfwps = load.get_iwp("NICAM","SHL", ice_only=False).values[::12]
nfwpn = load.get_iwp("NICAM","NAU", ice_only=False).values[::12]
print("nicam...")

n1s = np.nanmedian(np.where(nfwps>=cat1, nlwps, np.nan))
n2s = np.nanmedian(np.where((nfwps>=cat2)&(nfwps<cat1), nlwps, np.nan))
n3s = np.nanmedian(np.where((nfwps>=cat3)&(nfwps<cat2), nlwps, np.nan))
n4s = np.nanmedian(np.where((nfwps<cat3), nlwps, np.nan))
n1n = np.nanmedian(np.where(nfwpn>=cat1, nlwpn, np.nan))
n2n = np.nanmedian(np.where((nfwpn>=cat2)&(nfwpn<cat1), nlwpn, np.nan))
n3n = np.nanmedian(np.where((nfwpn>=cat3)&(nfwpn<cat2), nlwpn, np.nan))
n4n = np.nanmedian(np.where((nfwpn<cat3), nlwpn, np.nan))
del nlwpt, nlwps, nlwpn, nfwpt, nfwps, nfwpn
print("... done")
#%%
# load fv3 
flwpt = load.get_lwp("FV3","TWP").values[::12]
flwps = load.get_lwp("FV3","SHL").values[::12]
flwpn = load.get_lwp("FV3","NAU").values[::12]
ffwpt = load.get_iwp("FV3","TWP", ice_only=False).values[::12]
ffwps = load.get_iwp("FV3","SHL", ice_only=False).values[::12]
ffwpn = load.get_iwp("FV3","NAU", ice_only=False).values[::12]
print("fv3...")
f1t = np.nanmedian(np.where(ffwpt>=cat1, flwpt, np.nan))
f2t = np.nanmedian(np.where((ffwpt>=cat2)&(ffwpt<cat1), flwpt, np.nan))
f3t = np.nanmedian(np.where((ffwpt>=cat3)&(ffwpt<cat2), flwpt, np.nan))
f4t = np.nanmedian(np.where((ffwpt<cat3), flwpt, np.nan))
f1s = np.nanmedian(np.where(ffwps>=cat1, flwps, np.nan))
f2s = np.nanmedian(np.where((ffwps>=cat2)&(ffwps<cat1), flwps, np.nan))
f3s = np.nanmedian(np.where((ffwps>=cat3)&(ffwps<cat2), flwps, np.nan))
f4s = np.nanmedian(np.where((ffwps<cat3), flwps, np.nan))
f1n = np.nanmedian(np.where(ffwpn>=cat1, flwpn, np.nan))
f2n = np.nanmedian(np.where((ffwpn>=cat2)&(ffwpn<cat1), flwpn, np.nan))
f3n = np.nanmedian(np.where((ffwpn>=cat3)&(ffwpn<cat2), flwpn, np.nan))
f4n = np.nanmedian(np.where((ffwpn<cat3), flwpn, np.nan))
del flwpt, flwps, flwpn, ffwpt, ffwps, ffwpn
print("... done")
#%%
# load icon
ilwpt = load.get_lwp("ICON","TWP").values[::12]
ilwps = load.get_lwp("ICON","SHL").values[::12]
ilwpn = load.get_lwp("ICON","NAU").values[::12]
ifwpt = load.get_iwp("ICON","TWP", ice_only=False).values[::12]
ifwps = load.get_iwp("ICON","SHL", ice_only=False).values[::12]
ifwpn = load.get_iwp("ICON","NAU", ice_only=False).values[::12]
print("icon...")
i1t = np.nanmedian(np.where(ifwpt>=cat1, ilwpt, np.nan))
i2t = np.nanmedian(np.where((ifwpt>=cat2)&(ifwpt<cat1), ilwpt, np.nan))
i3t = np.nanmedian(np.where((ifwpt>=cat3)&(ifwpt<cat2), ilwpt, np.nan))
i4t = np.nanmedian(np.where((ifwpt<cat3), ilwpt, np.nan))
i1s = np.nanmedian(np.where(ifwps>=cat1, ilwps, np.nan))
i2s = np.nanmedian(np.where((ifwps>=cat2)&(ifwps<cat1), ilwps, np.nan))
i3s = np.nanmedian(np.where((ifwps>=cat3)&(ifwps<cat2), ilwps, np.nan))
i4s = np.nanmedian(np.where((ifwps<cat3), ilwps, np.nan))
i1n = np.nanmedian(np.where(ifwpn>=cat1, ilwpn, np.nan))
i2n = np.nanmedian(np.where((ifwpn>=cat2)&(ifwpn<cat1), ilwpn, np.nan))
i3n = np.nanmedian(np.where((ifwpn>=cat3)&(ifwpn<cat2), ilwpn, np.nan))
i4n = np.nanmedian(np.where((ifwpn<cat3), ilwpn, np.nan))
del ilwpt, ilwps, ilwpn, ifwpt, ifwps, ifwpn
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
twp_fwp_cat_med.to_csv("../tables/twp_lwp_cat_med_kgm-2.csv")
shl_fwp_cat_med.to_csv("../tables/shl_lwp_cat_med_kgm-2.csv")
nau_fwp_cat_med.to_csv("../tables/nau_lwp_cat_med_kgm-2.csv")


# %%
