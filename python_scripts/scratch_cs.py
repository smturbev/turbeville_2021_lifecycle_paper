# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from utility import load, load01deg

REGION="NAU"
# %%
c_olrcs = load01deg.get_cccm(REGION)["Clear-sky outgoing LW radiation at TOA"]
c_swucs = load01deg.get_cccm(REGION)["Clear-sky outgoing SW radiation at TOA"]
c_swd   = load01deg.get_cccm(REGION)["Incoming SW radiation at TOA"]
c_albcs = c_swucs/c_swd
del c_swucs, c_swd

fwp = load.get_iwp("NICAM",REGION,ice_only=False)
lwp = load.get_lwp("NICAM",REGION,rain=False)
fwp = fwp[11::12]
lwp = lwp[11::12]
n_olrcs = load.get_clearskyolr("NICAM",REGION,fwp,lwp)
n_albcs = load.get_clearskyalb("NICAM",REGION,fwp,lwp)
n_olrcs = n_olrcs[:,0]
n_albcs = n_albcs[:,0]
fwp = load.get_iwp("FV3",REGION,ice_only=False)
lwp = load.get_lwp("FV3",REGION,rain=False)
fwp = fwp[11::12]
lwp = lwp[11::12]
f_olrcs = load.get_clearskyolr("FV3",REGION,fwp,lwp)
f_albcs = load.get_clearskyalb("FV3",REGION,fwp,lwp)
fwp = load.get_iwp("ICON",REGION,ice_only=False)
lwp = load.get_lwp("ICON",REGION,rain=False)
fwp = fwp[11::12]
lwp = lwp[11::12]
i_olrcs = load.get_clearskyolr("ICON",REGION,fwp,lwp)
i_albcs = load.get_clearskyalb("ICON",REGION,fwp,lwp)
fwp = load.get_iwp("SAM",REGION,ice_only=False)
lwp = load.get_lwp("SAM",REGION,rain=False)
# fwp = fwp[11::12]
# lwp = lwp[11::12]
s_olrcs = load.get_clearskyolr("SAM",REGION,fwp,lwp)
s_albcs = load.get_clearskyalb("SAM",REGION,fwp,lwp)
del fwp, lwp

print("OLR")
print("CCCM: mean, median, std\n",np.nanmean(c_olrcs),
        np.nanmedian(c_olrcs), np.nanstd(c_olrcs))
print("NICA: mean, median, std\n",np.nanmean(n_olrcs),
        np.nanmedian(n_olrcs), np.nanstd(n_olrcs))
print("FV3 : mean, median, std\n",np.nanmean(f_olrcs),
        np.nanmedian(f_olrcs), np.nanstd(f_olrcs))
print("ICON: mean, median, std\n",np.nanmean(i_olrcs),
        np.nanmedian(i_olrcs), np.nanstd(i_olrcs))
print("SAM : mean, median, std\n",np.nanmean(s_olrcs),
        np.nanmedian(s_olrcs), np.nanstd(s_olrcs))
print("\n\nALBEDO")
print("CCCM: mean, median, std\n",np.nanmean(c_albcs),
        np.nanmedian(c_albcs), np.nanstd(c_albcs))
print("NICA: mean, median, std\n",np.nanmean(n_albcs),
        np.nanmedian(n_albcs), np.nanstd(n_albcs))
print("FV3 : mean, median, std\n",np.nanmean(f_albcs),
        np.nanmedian(f_albcs), np.nanstd(f_albcs))
print("ICON: mean, median, std\n",np.nanmean(i_albcs),
        np.nanmedian(i_albcs), np.nanstd(i_albcs))
print("SAM : mean, median, std\n",np.nanmean(s_albcs),
        np.nanmedian(s_albcs), np.nanstd(s_albcs))

#%%
olrcs = [n_olrcs, f_olrcs, s_olrcs]
albcs = [n_albcs, f_albcs, s_albcs]

fig, ax = plt.subplots(3,3, figsize=(12,12), 
                        sharex=True, sharey=True)
print(ax.shape)
for i in range(3):
    pc0 = ax[i,0].pcolormesh(np.linspace(143,153,(olrcs[i].shape[-1])),
                       np.linspace(-5,5,(olrcs[i].shape[-2])),
                       np.nanmean(olrcs[i], axis=0)
                      )
    pc1 = ax[i,1].pcolormesh(np.linspace(143,153,(olrcs[i].shape[-1])),
                       np.linspace(-5,5,(olrcs[i].shape[-2])),
                       np.nanmedian(olrcs[i], axis=0)
                      )
    pc2 = ax[i,2].pcolormesh(np.linspace(143,153,(olrcs[i].shape[-1])),
                       np.linspace(-5,5,(olrcs[i].shape[-2])),
                       np.nanstd(olrcs[i], axis=0)
                      )
    plt.colorbar(pc0, ax=ax[i,0])
    plt.colorbar(pc1, ax=ax[i,1])
    plt.colorbar(pc2, ax=ax[i,2])
    ax[i,0].set_title("mean")
    ax[i,1].set_title("median")
    ax[i,2].set_title("std")

print("Saved.")
plt.savefig("../plots/clear-sky_{}.png".format(REGION))
plt.show()

# %%
