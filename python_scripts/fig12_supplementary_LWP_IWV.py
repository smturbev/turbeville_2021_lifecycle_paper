# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utility import analysis_parameters as ap

c = ap.COLORS

# %%
# load csv
lwp_twp = pd.read_csv("../tables/twp_lwp_cat_med_kgm-2.csv", index_col=0)
lwp_shl = pd.read_csv("../tables/shl_lwp_cat_med_kgm-2.csv", index_col=0)
lwp_nau = pd.read_csv("../tables/nau_lwp_cat_med_kgm-2.csv", index_col=0)

fwp_twp = pd.read_csv("../tables/twp_fwp_cat_med_kgm-2.csv", index_col=0)
fwp_shl = pd.read_csv("../tables/shl_fwp_cat_med_kgm-2.csv", index_col=0)
fwp_nau = pd.read_csv("../tables/nau_fwp_cat_med_kgm-2.csv", index_col=0)

# iwv_twp = pd.read_csv("../tables/twp_iwv_cat_med_kgm-2.csv", index_col=0)
# iwv_shl = pd.read_csv("../tables/shl_iwv_cat_med_kgm-2.csv", index_col=0)
# iwv_nau = pd.read_csv("../tables/nau_iwv_cat_med_kgm-2.csv", index_col=0)

# %%
# Plot median
fig, ax = plt.subplots(1,4,figsize=(10,3), constrained_layout=True)
ms_star = 50
ms = 20
lw = 1
ax[0].scatter(fwp_twp.loc[["CAT1"]], lwp_twp.loc[["CAT1"]], c=list(c.values())[:5], 
              marker="o", edgecolors="k", s=ms)
ax[0].scatter(fwp_shl.loc[["CAT1"]], lwp_shl.loc[["CAT1"]], c=list(c.values())[:5], 
              marker="s", edgecolors="k", s=ms)
ax[0].scatter(fwp_nau.loc[["CAT1"]], lwp_nau.loc[["CAT1"]], c=list(c.values())[:5], 
              marker="*", edgecolors="k", s=ms_star)
ax[1].scatter(fwp_twp.loc[["CAT2"]], lwp_twp.loc[["CAT2"]], c=list(c.values())[:5], 
              marker="o", edgecolors="k", s=ms)
ax[1].scatter(fwp_shl.loc[["CAT2"]], lwp_shl.loc[["CAT2"]], c=list(c.values())[:5],
              marker="s", edgecolors="k", s=ms)
ax[1].scatter(fwp_nau.loc[["CAT2"]], lwp_nau.loc[["CAT2"]], c=list(c.values())[:5],
              marker="*", edgecolors="k", s=ms_star)
ax[2].scatter(fwp_twp.loc[["CAT3"]], lwp_twp.loc[["CAT3"]], c=list(c.values())[:5], 
              marker="o", edgecolors="k", s=ms)
ax[2].scatter(fwp_shl.loc[["CAT3"]], lwp_shl.loc[["CAT3"]], c=list(c.values())[:5],
              marker="s", edgecolors="k", s=ms)
ax[2].scatter(fwp_nau.loc[["CAT3"]], lwp_nau.loc[["CAT3"]], c=list(c.values())[:5],
              marker="*", edgecolors="k", s=ms_star)
ax[3].scatter(fwp_twp.loc[["CS"]], lwp_twp.loc[["CS"]], c=list(c.values())[:5], 
              marker="o", edgecolors="k", s=ms)
ax[3].scatter(fwp_shl.loc[["CS"]], lwp_shl.loc[["CS"]], c=list(c.values())[:5],
              marker="s", edgecolors="k", s=ms)
ax[3].scatter(fwp_nau.loc[["CS"]], lwp_nau.loc[["CS"]], c=list(c.values())[:5],
              marker="*", edgecolors="k", s=ms_star)
ax[0].plot([fwp_twp.loc[["CAT1"]].CCCM, fwp_shl.loc[["CAT1"]].CCCM, fwp_nau.loc[["CAT1"]].CCCM], 
           [lwp_twp.loc[["CAT1"]].CCCM, lwp_shl.loc[["CAT1"]].CCCM, lwp_nau.loc[["CAT1"]].CCCM],
           color=c["OBS"], lw=lw)
ax[0].plot([fwp_twp.loc[["CAT1"]].NICAM, fwp_shl.loc[["CAT1"]].NICAM, fwp_nau.loc[["CAT1"]].NICAM], 
           [lwp_twp.loc[["CAT1"]].NICAM, lwp_shl.loc[["CAT1"]].NICAM, lwp_nau.loc[["CAT1"]].NICAM],
           color=c["NICAM"], lw=lw)
ax[0].plot([fwp_twp.loc[["CAT1"]].FV3, fwp_shl.loc[["CAT1"]].FV3, fwp_nau.loc[["CAT1"]].FV3], 
           [lwp_twp.loc[["CAT1"]].FV3, lwp_shl.loc[["CAT1"]].FV3, lwp_nau.loc[["CAT1"]].FV3],
           color=c["FV3"], lw=lw)
ax[0].plot([fwp_twp.loc[["CAT1"]].ICON, fwp_shl.loc[["CAT1"]].ICON, fwp_nau.loc[["CAT1"]].ICON], 
           [lwp_twp.loc[["CAT1"]].ICON, lwp_shl.loc[["CAT1"]].ICON, lwp_nau.loc[["CAT1"]].ICON],
           color=c["ICON"], lw=lw)
ax[0].plot([fwp_twp.loc[["CAT1"]].SAM, fwp_shl.loc[["CAT1"]].SAM, fwp_nau.loc[["CAT1"]].SAM], 
           [lwp_twp.loc[["CAT1"]].SAM, lwp_shl.loc[["CAT1"]].SAM, lwp_nau.loc[["CAT1"]].SAM],
           color=c["SAM"], lw=lw)

ax[1].plot([fwp_twp.loc[["CAT2"]].CCCM, fwp_shl.loc[["CAT2"]].CCCM, fwp_nau.loc[["CAT2"]].CCCM], 
           [lwp_twp.loc[["CAT2"]].CCCM, lwp_shl.loc[["CAT2"]].CCCM, lwp_nau.loc[["CAT2"]].CCCM],
           color=c["OBS"], lw=lw)
ax[1].plot([fwp_twp.loc[["CAT2"]].NICAM, fwp_shl.loc[["CAT2"]].NICAM, fwp_nau.loc[["CAT2"]].NICAM], 
           [lwp_twp.loc[["CAT2"]].NICAM, lwp_shl.loc[["CAT2"]].NICAM, lwp_nau.loc[["CAT2"]].NICAM],
           color=c["NICAM"], lw=lw)
ax[1].plot([fwp_twp.loc[["CAT2"]].FV3, fwp_shl.loc[["CAT2"]].FV3, fwp_nau.loc[["CAT2"]].FV3], 
           [lwp_twp.loc[["CAT2"]].FV3, lwp_shl.loc[["CAT2"]].FV3, lwp_nau.loc[["CAT2"]].FV3],
           color=c["FV3"], lw=lw)
ax[1].plot([fwp_twp.loc[["CAT2"]].ICON, fwp_shl.loc[["CAT2"]].ICON, fwp_nau.loc[["CAT2"]].ICON], 
           [lwp_twp.loc[["CAT2"]].ICON, lwp_shl.loc[["CAT2"]].ICON, lwp_nau.loc[["CAT2"]].ICON],
           color=c["ICON"], lw=lw)
ax[1].plot([fwp_twp.loc[["CAT2"]].SAM, fwp_shl.loc[["CAT2"]].SAM, fwp_nau.loc[["CAT2"]].SAM], 
           [lwp_twp.loc[["CAT2"]].SAM, lwp_shl.loc[["CAT2"]].SAM, lwp_nau.loc[["CAT2"]].SAM],
           color=c["SAM"], lw=lw)

ax[2].plot([fwp_twp.loc[["CAT3"]].CCCM, fwp_shl.loc[["CAT3"]].CCCM, fwp_nau.loc[["CAT3"]].CCCM], 
           [lwp_twp.loc[["CAT3"]].CCCM, lwp_shl.loc[["CAT3"]].CCCM, lwp_nau.loc[["CAT3"]].CCCM],
           color=c["OBS"], lw=lw)
ax[2].plot([fwp_twp.loc[["CAT3"]].NICAM, fwp_shl.loc[["CAT3"]].NICAM, fwp_nau.loc[["CAT3"]].NICAM], 
           [lwp_twp.loc[["CAT3"]].NICAM, lwp_shl.loc[["CAT3"]].NICAM, lwp_nau.loc[["CAT3"]].NICAM],
           color=c["NICAM"], lw=lw)
ax[2].plot([fwp_twp.loc[["CAT3"]].FV3, fwp_shl.loc[["CAT3"]].FV3, fwp_nau.loc[["CAT3"]].FV3], 
           [lwp_twp.loc[["CAT3"]].FV3, lwp_shl.loc[["CAT3"]].FV3, lwp_nau.loc[["CAT3"]].FV3],
           color=c["FV3"], lw=lw)
ax[2].plot([fwp_twp.loc[["CAT3"]].ICON, fwp_shl.loc[["CAT3"]].ICON, fwp_nau.loc[["CAT3"]].ICON], 
           [lwp_twp.loc[["CAT3"]].ICON, lwp_shl.loc[["CAT3"]].ICON, lwp_nau.loc[["CAT3"]].ICON],
           color=c["ICON"], lw=lw)
ax[2].plot([fwp_twp.loc[["CAT3"]].SAM, fwp_shl.loc[["CAT3"]].SAM, fwp_nau.loc[["CAT3"]].SAM], 
           [lwp_twp.loc[["CAT3"]].SAM, lwp_shl.loc[["CAT3"]].SAM, lwp_nau.loc[["CAT3"]].SAM],
           color=c["SAM"], lw=lw)

ax[3].plot([fwp_twp.loc[["CS"]].CCCM, fwp_shl.loc[["CS"]].CCCM, fwp_nau.loc[["CS"]].CCCM], 
           [lwp_twp.loc[["CS"]].CCCM, lwp_shl.loc[["CS"]].CCCM, lwp_nau.loc[["CS"]].CCCM],
           color=c["OBS"], lw=lw)
ax[3].plot([fwp_twp.loc[["CS"]].NICAM, fwp_shl.loc[["CS"]].NICAM, fwp_nau.loc[["CS"]].NICAM], 
           [lwp_twp.loc[["CS"]].NICAM, lwp_shl.loc[["CS"]].NICAM, lwp_nau.loc[["CS"]].NICAM],
           color=c["NICAM"], lw=lw)
ax[3].plot([fwp_twp.loc[["CS"]].FV3, fwp_shl.loc[["CS"]].FV3, fwp_nau.loc[["CS"]].FV3], 
           [lwp_twp.loc[["CS"]].FV3, lwp_shl.loc[["CS"]].FV3, lwp_nau.loc[["CS"]].FV3],
           color=c["FV3"], lw=lw)
ax[3].plot([fwp_twp.loc[["CS"]].ICON, fwp_shl.loc[["CS"]].ICON, fwp_nau.loc[["CS"]].ICON], 
           [lwp_twp.loc[["CS"]].ICON, lwp_shl.loc[["CS"]].ICON, lwp_nau.loc[["CS"]].ICON],
           color=c["ICON"], lw=lw)
ax[3].plot([fwp_twp.loc[["CS"]].SAM, fwp_shl.loc[["CS"]].SAM, fwp_nau.loc[["CS"]].SAM], 
           [lwp_twp.loc[["CS"]].SAM, lwp_shl.loc[["CS"]].SAM, lwp_nau.loc[["CS"]].SAM],
           color=c["SAM"], lw=lw)
for i in range(4):
    ax[i].set_xlabel("FWP (kg/m2)")
ax[0].set_ylabel("LWP (kg/m2)")
plt.show()
# %%
fwp_twp.loc[["CAT1"]]
# %%
fwp_twp.T.CAT1

# %%
df1 = pd.DataFrame([fwp_twp.T.CAT1, lwp_twp.T.CAT1], index=["FWP","LWP"]).T
df2 = pd.DataFrame([fwp_twp.T.CAT2, lwp_twp.T.CAT2], index=["FWP","LWP"]).T
df3 = pd.DataFrame([fwp_twp.T.CAT3, lwp_twp.T.CAT3], index=["FWP","LWP"]).T
df4 = pd.DataFrame([fwp_twp.T.CS, lwp_twp.T.CS], index=["FWP","LWP"]).T
y = list(df1.index)
df1
# %%
_, ax = plt.subplots(2,2, figsize=(15,15))
pd.plotting.scatter_matrix(df1, c=list(c.values())[:5], ax=ax, marker="o",
                hist_kwds={"bins":20}, s=80, alpha=0.8, cmap="tab10")
plt.savefig("../plots/pair_plot_cat_med.png")
plt.show()
# %%
