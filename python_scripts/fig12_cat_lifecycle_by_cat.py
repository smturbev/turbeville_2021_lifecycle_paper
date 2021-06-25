#/usr/bin/env python
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import util, analysis_parameters as ap

c = ap.COLORS

#%%
# means
ctwp = pd.read_csv("../tables/mean_CCCM_TWP.csv", index_col=0)
ntwp = pd.read_csv("../tables/mean_NICAM_TWP.csv", index_col=0)
ftwp = pd.read_csv("../tables/mean_FV3_TWP.csv", index_col=0)
itwp = pd.read_csv("../tables/mean_ICON_TWP.csv", index_col=0)
stwp = pd.read_csv("../tables/mean_SAM_TWP.csv", index_col=0)
cshl = pd.read_csv("../tables/mean_CCCM_SHL.csv", index_col=0)
nshl = pd.read_csv("../tables/mean_NICAM_SHL.csv", index_col=0)
fshl = pd.read_csv("../tables/mean_FV3_SHL.csv", index_col=0)
ishl = pd.read_csv("../tables/mean_ICON_SHL.csv", index_col=0)
sshl = pd.read_csv("../tables/mean_SAM_SHL.csv", index_col=0)
cnau = pd.read_csv("../tables/mean_CCCM_NAU.csv", index_col=0)
nnau = pd.read_csv("../tables/mean_NICAM_NAU.csv", index_col=0)
fnau = pd.read_csv("../tables/mean_FV3_NAU.csv", index_col=0)
inau = pd.read_csv("../tables/mean_ICON_NAU.csv", index_col=0)
snau = pd.read_csv("../tables/mean_SAM_NAU.csv", index_col=0)

#%%
tc_olr = ctwp.olr.values #np.array([105,195,263])
tc_alb = ctwp.alb.values #np.array([0.63,0.28,0.09])
tn_olr = ntwp.olr.values
tn_alb = ntwp.alb.values
tf_olr = ftwp.olr.values
tf_alb = ftwp.alb.values
ti_olr = itwp.olr.values
ti_alb = itwp.alb.values
ts_olr = stwp.olr.values
ts_alb = stwp.alb.values

sc_olr = cshl.olr.values #np.array([105,195,263])
sc_alb = cshl.alb.values #np.array([0.63,0.28,0.09])
sn_olr = nshl.olr.values
sn_alb = nshl.alb.values
sf_olr = fshl.olr.values
sf_alb = fshl.alb.values
si_olr = ishl.olr.values
si_alb = ishl.alb.values
ss_olr = sshl.olr.values
ss_alb = sshl.alb.values

nc_olr = cnau.olr.values #np.array([105,195,263])
nc_alb = cnau.alb.values #np.array([0.63,0.28,0.09])
nn_olr = nnau.olr.values
nn_alb = nnau.alb.values
nf_olr = fnau.olr.values
nf_alb = fnau.alb.values
ni_olr = inau.olr.values
ni_alb = inau.alb.values
ns_olr = snau.olr.values
ns_alb = snau.alb.values

lw = 3
ms = 20
a = 1
fs = 18
#%%
# Plot median
fig, ax = plt.subplots(2,2,figsize=(9,9), constrained_layout=True)
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax[0,0])
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax[0,1])
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax[1,0])
util.dennisplot("density",np.array([]),np.array([]),\
                 model="Category Median Albedo & OLR\n",region="TWP",\
                 colorbar_on=False, ax=ax[1,1])

ax[0,0].plot([sc_olr[0],tc_olr[0],nc_olr[0]], 
           [sc_alb[0],tc_alb[0],nc_alb[0]],
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[0,0].plot([sn_olr[0],tn_olr[0],nn_olr[0]], 
           [sn_alb[0],tn_alb[0],nn_alb[0]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[0,0].plot([sf_olr[0],tf_olr[0],nf_olr[0]], 
           [sf_alb[0],tf_alb[0],nf_alb[0]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[0,0].plot([si_olr[0],ti_olr[0],ni_olr[0]], 
           [si_alb[0],ti_alb[0],ni_alb[0]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[0,0].plot([ss_olr[0],ts_olr[0],ns_olr[0]], 
           [ss_alb[0],ts_alb[0],ns_alb[0]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ax[0,1].plot([sc_olr[1],tc_olr[1],nc_olr[1]], 
           [sc_alb[1],tc_alb[1],nc_alb[1]], 
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[0,1].plot([sn_olr[1],tn_olr[1],nn_olr[1]], 
           [sn_alb[1],tn_alb[1],nn_alb[1]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[0,1].plot([sf_olr[1],tf_olr[1],nf_olr[1]], 
           [sf_alb[1],tf_alb[1],nf_alb[1]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[0,1].plot([si_olr[1],ti_olr[1],ni_olr[1]], 
           [si_alb[1],ti_alb[1],ni_alb[1]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[0,1].plot([ss_olr[1],ts_olr[1],ns_olr[1]], 
           [ss_alb[1],ts_alb[1],ns_alb[1]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ax[1,0].plot([sc_olr[2],tc_olr[2],nc_olr[2]], 
           [sc_alb[2],tc_alb[2],nc_alb[2]], 
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[1,0].plot([sn_olr[2],tn_olr[2],nn_olr[2]], 
           [sn_alb[2],tn_alb[2],nn_alb[2]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[1,0].plot([sf_olr[2],tf_olr[2],nf_olr[2]], 
           [sf_alb[2],tf_alb[2],nf_alb[2]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[1,0].plot([si_olr[2],ti_olr[2],ni_olr[2]], 
           [si_alb[2],ti_alb[2],ni_alb[2]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[1,0].plot([ss_olr[2],ts_olr[2],ns_olr[2]], 
           [ss_alb[2],ts_alb[2],ns_alb[2]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ax[1,1].plot([sc_olr[3],tc_olr[3],nc_olr[3]], 
           [sc_alb[3],tc_alb[3],nc_alb[3]], 
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[1,1].plot([sn_olr[3],tn_olr[3],nn_olr[3]], 
           [sn_alb[3],tn_alb[3],nn_alb[3]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[1,1].plot([sf_olr[3],tf_olr[3],nf_olr[3]], 
           [sf_alb[3],tf_alb[3],nf_alb[3]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[1,1].plot([si_olr[3],ti_olr[3],ni_olr[3]], 
           [si_alb[3],ti_alb[3],ni_alb[3]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[1,1].plot([ss_olr[3],ts_olr[3],ns_olr[3]], 
           [ss_alb[3],ts_alb[3],ns_alb[3]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ms = 80
ax[0,0].scatter([sc_olr[0], sn_olr[0],sf_olr[0],si_olr[0],ss_olr[0]],
              [sc_alb[0], sn_alb[0],sf_alb[0],si_alb[0],ss_alb[0]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[0,1].scatter([sc_olr[1],sn_olr[1],sf_olr[1],si_olr[1],ss_olr[1]],
              [sc_alb[1],sn_alb[1],sf_alb[1],si_alb[1],ss_alb[1]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,0].scatter([sc_olr[2],sn_olr[2],sf_olr[2],si_olr[2],ss_olr[2]],
              [sc_alb[2],sn_alb[2],sf_alb[2],si_alb[2],ss_alb[2]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,1].scatter([sc_olr[3],sn_olr[3],sf_olr[3],si_olr[3],ss_olr[3]],
              [sc_alb[3],sn_alb[3],sf_alb[3],si_alb[3],ss_alb[3]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ms = 190
ax[0,0].scatter([tc_olr[0],tn_olr[0],tf_olr[0],ti_olr[0],ts_olr[0]],
              [tc_alb[0],tn_alb[0],tf_alb[0],ti_alb[0],ts_alb[0]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[0,1].scatter([tc_olr[1],tn_olr[1],tf_olr[1],ti_olr[1],ts_olr[1]],
              [tc_alb[1],tn_alb[1],tf_alb[1],ti_alb[1],ts_alb[1]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,0].scatter([tc_olr[2],tn_olr[2],tf_olr[2],ti_olr[2],ts_olr[2]],
              [tc_alb[2],tn_alb[2],tf_alb[2],ti_alb[2],ts_alb[2]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,1].scatter([tc_olr[3],tn_olr[3],tf_olr[3],ti_olr[3],ts_olr[3]],
              [tc_alb[3],tn_alb[3],tf_alb[3],ti_alb[3],ts_alb[3]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ms = 200
ax[0,0].scatter([nc_olr[0],nn_olr[0],nf_olr[0],ni_olr[0],ns_olr[0]],
              [nc_alb[0],nn_alb[0],nf_alb[0],ni_alb[0],ns_alb[0]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[0,1].scatter([nc_olr[1],nn_olr[1],nf_olr[1],ni_olr[1],ns_olr[1]],
              [nc_alb[1],nn_alb[1],nf_alb[1],ni_alb[1],ns_alb[1]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,0].scatter([nc_olr[2],nn_olr[2],nf_olr[2],ni_olr[2],ns_olr[2]],
              [nc_alb[2],nn_alb[2],nf_alb[2],ni_alb[2],ns_alb[2]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,1].scatter([nc_olr[3],nn_olr[3],nf_olr[3],ni_olr[3],ns_olr[3]],
              [nc_alb[3],nn_alb[3],nf_alb[3],ni_alb[3],ns_alb[3]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])

ax[0,0].set_ylim([0.3,0.8])
ax[0,0].set_xlim([110,160])
ax[0,0].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[0,0].set_ylabel('Albedo', size=fs)
ax[0,0].set_title('CAT 1', fontsize=fs)
ax[0,0].set_xticks(np.arange(95,155,10))
ax[0,0].tick_params(labelsize=fs-4)

ax[0,1].set_ylim([0.2,0.6])
ax[0,1].set_xlim([150,220])
ax[0,1].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[0,1].set_ylabel('Albedo', size=fs)
ax[0,1].set_title('CAT 2', fontsize=fs)
ax[0,1].set_xticks(np.arange(100,255,25))
ax[0,1].tick_params(labelsize=fs-4)

ax[1,0].set_ylim([0.05,0.4])
ax[1,0].set_xlim([245,285])
ax[1,0].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[1,0].set_ylabel('Albedo', size=fs)
ax[1,0].set_title('CAT 3', fontsize=fs)
ax[1,0].set_xticks(np.arange(250,285,10))
ax[1,0].tick_params(labelsize=fs-4)

ax[1,1].set_ylim([0.05,0.4])
ax[1,1].set_xlim([265,305])
ax[1,1].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[1,1].set_ylabel('Albedo', size=fs)
ax[1,1].set_title('Clear Sky', fontsize=fs)
ax[1,1].set_xticks(np.arange(270,305,10))
ax[1,1].tick_params(labelsize=fs-4)
ax[1,1].grid(True)

h, l = ax[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="none",bbox_to_anchor=(1.2,0.7), fontsize=fs-4)

plt.tight_layout()
plt.savefig('../plots/fig12_cat_lifecycle_stn_mean.png',dpi=150,bbox_inches='tight', pad_inches=0.1)
print('    saved to ../plots/fig12_cat_lifecycle_stn_mean.png')
plt.show()

# %%
