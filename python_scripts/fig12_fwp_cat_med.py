
#!usr/bin/eng python
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import load, load01deg, analysis_parameters as ap

c = ap.COLORS

# %%
# load cccm
ciwpt = load01deg.get_cccm("TWP")["iwp MODIS"]
ciwps = load01deg.get_cccm("SHL")["iwp MODIS"]
ciwpn = load01deg.get_cccm("NAU")["iwp MODIS"]
# get category median values
cat1 = 1000
cat2 = 10
cat3 = 0.1
print("cccm...")
c1t = (np.where(ciwpt>=cat1, ciwpt, np.nan))
c2t = (np.where((ciwpt>=cat2)&(ciwpt<cat1), ciwpt, np.nan))
c3t = (np.where((ciwpt>=cat3)&(ciwpt<cat2), ciwpt, np.nan))
c4t = (np.where((ciwpt<cat3), ciwpt, np.nan))
c1s = (np.where(ciwps>=cat1, ciwps, np.nan))
c2s = (np.where((ciwps>=cat2)&(ciwps<cat1), ciwps, np.nan))
c3s = (np.where((ciwps>=cat3)&(ciwps<cat2), ciwps, np.nan))
c4s = (np.where((ciwps<cat3), ciwps, np.nan))
c1n = (np.where(ciwpn>=cat1, ciwpn, np.nan))
c2n = (np.where((ciwpn>=cat2)&(ciwpn<cat1), ciwpn, np.nan))
c3n = (np.where((ciwpn>=cat3)&(ciwpn<cat2), ciwpn, np.nan))
c4n = (np.where((ciwpn<cat3), ciwpn, np.nan))
del ciwpt, ciwps, ciwpn
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
# %%
# load sam
cat1 = 1
cat2 = 1e-2
cat3 = 1e-4
siwpt = load.get_iwp("SAM","TWP", ice_only=False)
siwps = load.get_iwp("SAM","SHL", ice_only=False)
siwpn = load.get_iwp("SAM","NAU", ice_only=False)
print("sam...")
s1t = np.nanmedian(np.where(siwpt>=cat1, siwpt, np.nan))
s2t = np.nanmedian(np.where((siwpt>=cat2)&(siwpt<cat1), siwpt, np.nan))
s3t = np.nanmedian(np.where((siwpt>=cat3)&(siwpt<cat2), siwpt, np.nan))
s4t = np.nanmedian(np.where((siwpt<cat3), siwpt, np.nan))
s1s = np.nanmedian(np.where(siwps>=cat1, siwps, np.nan))
s2s = np.nanmedian(np.where((siwps>=cat2)&(siwps<cat1), siwps, np.nan))
s3s = np.nanmedian(np.where((siwps>=cat3)&(siwps<cat2), siwps, np.nan))
s4s = np.nanmedian(np.where((siwps<cat3), siwps, np.nan))
s1n = np.nanmedian(np.where(siwpn>=cat1, siwpn, np.nan))
s2n = np.nanmedian(np.where((siwpn>=cat2)&(siwpn<cat1), siwpn, np.nan))
s3n = np.nanmedian(np.where((siwpn>=cat3)&(siwpn<cat2), siwpn, np.nan))
s4n = np.nanmedian(np.where((siwpn<cat3), siwpn, np.nan))
print("... done.")
#%%
# load nicam
cat1 = 1
cat2 = 1e-2
cat3 = 1e-4
niwpt = load.get_iwp("NICAM","TWP", ice_only=False)
niwps = load.get_iwp("NICAM","SHL", ice_only=False)
niwpn = load.get_iwp("NICAM","NAU", ice_only=False)
print("nicam...")
n1t = np.nanmedian(np.where(niwpt>=cat1, niwpt, np.nan))
n2t = np.nanmedian(np.where((niwpt>=cat2)&(niwpt<cat1), niwpt, np.nan))
n3t = np.nanmedian(np.where((niwpt>=cat3)&(niwpt<cat2), niwpt, np.nan))
n4t = np.nanmedian(np.where((niwpt<cat3), niwpt, np.nan))
n1s = np.nanmedian(np.where(niwps>=cat1, niwps, np.nan))
n2s = np.nanmedian(np.where((niwps>=cat2)&(niwps<cat1), niwps, np.nan))
n3s = np.nanmedian(np.where((niwps>=cat3)&(niwps<cat2), niwps, np.nan))
n4s = np.nanmedian(np.where((niwps<cat3), niwps, np.nan))
n1n = np.nanmedian(np.where(niwpn>=cat1, niwpn, np.nan))
n2n = np.nanmedian(np.where((niwpn>=cat2)&(niwpn<cat1), niwpn, np.nan))
n3n = np.nanmedian(np.where((niwpn>=cat3)&(niwpn<cat2), niwpn, np.nan))
n4n = np.nanmedian(np.where((niwpn<cat3), niwpn, np.nan))
print("... done")
#%%
# load fv3 
fiwpt = load.get_iwp("FV3","TWP", ice_only=False)
fiwps = load.get_iwp("FV3","SHL", ice_only=False)
fiwpn = load.get_iwp("FV3","NAU", ice_only=False)
print("fv3...")
f1t = np.nanmedian(np.where(fiwpt>=cat1, fiwpt, np.nan))
f2t = np.nanmedian(np.where((fiwpt>=cat2)&(fiwpt<cat1), fiwpt, np.nan))
f3t = np.nanmedian(np.where((fiwpt>=cat3)&(fiwpt<cat2), fiwpt, np.nan))
f4t = np.nanmedian(np.where((fiwpt<cat3), fiwpt, np.nan))
f1s = np.nanmedian(np.where(fiwps>=cat1, fiwps, np.nan))
f2s = np.nanmedian(np.where((fiwps>=cat2)&(fiwps<cat1), fiwps, np.nan))
f3s = np.nanmedian(np.where((fiwps>=cat3)&(fiwps<cat2), fiwps, np.nan))
f4s = np.nanmedian(np.where((fiwps<cat3), fiwps, np.nan))
f1n = np.nanmedian(np.where(fiwpn>=cat1, fiwpn, np.nan))
f2n = np.nanmedian(np.where((fiwpn>=cat2)&(fiwpn<cat1), fiwpn, np.nan))
f3n = np.nanmedian(np.where((fiwpn>=cat3)&(fiwpn<cat2), fiwpn, np.nan))
f4n = np.nanmedian(np.where((fiwpn<cat3), fiwpn, np.nan))
print("... done")
#%%
# load icon
iiwpt = load.get_iwp("ICON","TWP", ice_only=False).values
iiwps = load.get_iwp("ICON","SHL", ice_only=False).values
iiwpn = load.get_iwp("ICON","NAU", ice_only=False).values
print("icon...")
i1t = np.nanmedian(np.where(iiwpt>=cat1, iiwpt, np.nan))
i2t = np.nanmedian(np.where((iiwpt>=cat2)&(iiwpt<cat1), iiwpt, np.nan))
i3t = np.nanmedian(np.where((iiwpt>=cat3)&(iiwpt<cat2), iiwpt, np.nan))
i4t = np.nanmedian(np.where((iiwpt<cat3), iiwpt, np.nan))
i1s = np.nanmedian(np.where(iiwps>=cat1, iiwps, np.nan))
i2s = np.nanmedian(np.where((iiwps>=cat2)&(iiwps<cat1), iiwps, np.nan))
i3s = np.nanmedian(np.where((iiwps>=cat3)&(iiwps<cat2), iiwps, np.nan))
i4s = np.nanmedian(np.where((iiwps<cat3), iiwps, np.nan))
i1n = np.nanmedian(np.where(iiwpn>=cat1, iiwpn, np.nan))
i2n = np.nanmedian(np.where((iiwpn>=cat2)&(iiwpn<cat1), iiwpn, np.nan))
i3n = np.nanmedian(np.where((iiwpn>=cat3)&(iiwpn<cat2), iiwpn, np.nan))
i4n = np.nanmedian(np.where((iiwpn<cat3), iiwpn, np.nan))
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
twp_fwp_cat_med.to_csv("../tables/twp_fwp_cat_med_kgm-2.csv")
shl_fwp_cat_med.to_csv("../tables/shl_fwp_cat_med_kgm-2.csv")
nau_fwp_cat_med.to_csv("../tables/nau_fwp_cat_med_kgm-2.csv")


# %%
