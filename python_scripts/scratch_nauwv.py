#!/usr/bin/env python
# %%
import matplotlib.pyplot as plt
from utility import load, util, analysis_parameters as ap
import numpy as np
from matplotlib import cm

REGION="NAU"
colors = ap.COLORS

#%%
# wv load - icon
qv = load.get_qv("ICON",REGION).astype("float32")
t = load.get_temp("ICON",REGION).values.astype("float16")
p = load.get_pres("ICON",REGION).values.astype("float16")
wv = util.iwc(qv, t, qv, p, "ICON")
del qv, t, p
i_wv_mean = np.nanmean(wv, axis=(0,2))
del wv

#%%
# wv load - fv3
qv = load.get_qv("FV3",REGION)
t = load.get_temp("FV3",REGION).values
p = load.get_pres("FV3",REGION).values
print(p.shape, t.shape, qv.shape)
wv = util.iwc(qv, t, qv, p, "FV3")
del qv, t, p
f_wv_mean = np.nanmean(wv, axis=(0,2,3))
del wv

#%%
# wv load - nicam
qv = load.get_qv("NICAM",REGION)
t = load.get_temp("NICAM",REGION).values
p = load.get_pres("NICAM",REGION).values
wv = util.iwc(qv, t, qv, p, "NICAM")
del qv, t, p
n_wv_mean = np.nanmean(wv, axis=(0,2,3))
del wv

#%%
# wv load - sam
qv = load.get_qv("SAM",REGION)
t = load.get_temp("SAM",REGION).values
p = load.get_pres("SAM",REGION).values
print("shapes", p[:,:,np.newaxis, np.newaxis].shape, t.shape, qv.shape)
wv = util.iwc(qv, t, qv, p, "SAM")
del qv, t, p
s_wv_mean = np.nanmean(wv, axis=(0,2,3))
del wv
#%%
# get heights
nz = load.get_levels("NICAM", REGION)
fz = load.get_levels("FV3", REGION)
iz = load.get_levels("ICON", REGION)
sz = load.get_levels("SAM", REGION)

#%%
# PLOT MEANS
fig, [ax, axttl] = plt.subplots(1,2,figsize=(13,10))
ax.plot(np.log10(n_wv_mean/1000), nz/1000, color=colors["NICAM"], label="NICAM")
ax.plot(np.log10(f_wv_mean/1000), fz/1000, color=colors["FV3"], label="FV3")
ax.plot(np.log10(i_wv_mean/1000), iz/1000, color=colors["ICON"], label="ICON")
ax.plot(np.log10(s_wv_mean/1000), sz/1000, color=colors["SAM"], label="SAM")
ax.fill_between(14,18,[-7,1], color="gray", alpha=0.4, label="TTL")

ax.set_xlabel("water vapor (g/m3)")
ax.set_ylabel("Height (km)")

axttl.plot(np.log10(n_wv_mean/1000), nz/1000, color=colors["NICAM"], label="NICAM")
axttl.plot(np.log10(f_wv_mean/1000), fz/1000, color=colors["FV3"], label="FV3")
axttl.plot(np.log10(i_wv_mean/1000), iz/1000, color=colors["ICON"], label="ICON")
axttl.plot(np.log10(s_wv_mean/1000), sz/1000, color=colors["SAM"], label="SAM")
axttl.fill_between(14,18,[-7,1], color="gray", alpha=0.4, label="TTL")

axttl.set_xlabel("water vapor (g/m3)")
axttl.set_ylabel("Height (km)")
axttl.set_ylim([12,20])

plt.savefig("../plots/nau_mean_wv_profiles.png", dpi=150, bbox_inches="tight")
plt.show()