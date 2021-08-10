#/usr/bin/env python
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import util, analysis_parameters as ap

c = ap.COLORS

#%%
ctwp = pd.read_csv("../tables/CCCM_TWP.csv", index_col=0)
ntwp = pd.read_csv("../tables/NICAM_TWP.csv", index_col=0)
ftwp = pd.read_csv("../tables/FV3_TWP.csv", index_col=0)
itwp = pd.read_csv("../tables/ICON_TWP.csv", index_col=0)
stwp = pd.read_csv("../tables/SAM_TWP.csv", index_col=0)
cshl = pd.read_csv("../tables/CCCM_SHL.csv", index_col=0)
nshl = pd.read_csv("../tables/NICAM_SHL.csv", index_col=0)
fshl = pd.read_csv("../tables/FV3_SHL.csv", index_col=0)
ishl = pd.read_csv("../tables/ICON_SHL.csv", index_col=0)
sshl = pd.read_csv("../tables/SAM_SHL.csv", index_col=0)
cnau = pd.read_csv("../tables/CCCM_NAU.csv", index_col=0)
nnau = pd.read_csv("../tables/NICAM_NAU.csv", index_col=0)
fnau = pd.read_csv("../tables/FV3_NAU.csv", index_col=0)
inau = pd.read_csv("../tables/ICON_NAU.csv", index_col=0)
snau = pd.read_csv("../tables/SAM_NAU.csv", index_col=0)

# cre_twp = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_TWP.csv")
# cre_shl = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_SHL.csv")
# cre_nau = pd.read_csv("../tables/isolated_ttl_cirrus_iceonly_noliq_percentiles_NAU.csv")

tc_med_olr = ctwp.OLR.values[:-1] #np.array([105,195,263])
tc_med_alb = ctwp.ALB.values[:-1] #np.array([0.63,0.28,0.09])
tn_med_olr = ntwp.OLR.values[:-1]
tn_med_alb = ntwp.ALB.values[:-1]
tf_med_olr = ftwp.OLR.values[:-1]
tf_med_alb = ftwp.ALB.values[:-1]
ti_med_olr = itwp.OLR.values[:-1]
ti_med_alb = itwp.ALB.values[:-1]
ts_med_olr = stwp.OLR.values[:-1]
ts_med_alb = stwp.ALB.values[:-1]

sc_med_olr = cshl.OLR.values[:-1] #np.array([105,195,263])
sc_med_alb = cshl.ALB.values[:-1] #np.array([0.63,0.28,0.09])
sn_med_olr = nshl.OLR.values[:-1]
sn_med_alb = nshl.ALB.values[:-1]
sf_med_olr = fshl.OLR.values[:-1]
sf_med_alb = fshl.ALB.values[:-1]
si_med_olr = ishl.OLR.values[:-1]
si_med_alb = ishl.ALB.values[:-1]
ss_med_olr = sshl.OLR.values[:-1]
ss_med_alb = sshl.ALB.values[:-1]

nc_med_olr = cnau.OLR.values[:-1] #np.array([105,195,263])
nc_med_alb = cnau.ALB.values[:-1] #np.array([0.63,0.28,0.09])
nn_med_olr = nnau.OLR.values[:-1]
nn_med_alb = nnau.ALB.values[:-1]
nf_med_olr = fnau.OLR.values[:-1]
nf_med_alb = fnau.ALB.values[:-1]
ni_med_olr = inau.OLR.values[:-1]
ni_med_alb = inau.ALB.values[:-1]
ns_med_olr = snau.OLR.values[:-1]
ns_med_alb = snau.ALB.values[:-1]

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

ax[0,0].plot([sc_med_olr[0],tc_med_olr[0],nc_med_olr[0]], 
           [sc_med_alb[0],tc_med_alb[0],nc_med_alb[0]],
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[0,0].plot([sn_med_olr[0],tn_med_olr[0],nn_med_olr[0]], 
           [sn_med_alb[0],tn_med_alb[0],nn_med_alb[0]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[0,0].plot([sf_med_olr[0],tf_med_olr[0],nf_med_olr[0]], 
           [sf_med_alb[0],tf_med_alb[0],nf_med_alb[0]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[0,0].plot([si_med_olr[0],ti_med_olr[0],ni_med_olr[0]], 
           [si_med_alb[0],ti_med_alb[0],ni_med_alb[0]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[0,0].plot([ss_med_olr[0],ts_med_olr[0],ns_med_olr[0]], 
           [ss_med_alb[0],ts_med_alb[0],ns_med_alb[0]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ax[0,1].plot([sc_med_olr[1],tc_med_olr[1],nc_med_olr[1]], 
           [sc_med_alb[1],tc_med_alb[1],nc_med_alb[1]], 
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[0,1].plot([sn_med_olr[1],tn_med_olr[1],nn_med_olr[1]], 
           [sn_med_alb[1],tn_med_alb[1],nn_med_alb[1]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[0,1].plot([sf_med_olr[1],tf_med_olr[1],nf_med_olr[1]], 
           [sf_med_alb[1],tf_med_alb[1],nf_med_alb[1]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[0,1].plot([si_med_olr[1],ti_med_olr[1],ni_med_olr[1]], 
           [si_med_alb[1],ti_med_alb[1],ni_med_alb[1]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[0,1].plot([ss_med_olr[1],ts_med_olr[1],ns_med_olr[1]], 
           [ss_med_alb[1],ts_med_alb[1],ns_med_alb[1]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ax[1,0].plot([sc_med_olr[2],tc_med_olr[2],nc_med_olr[2]], 
           [sc_med_alb[2],tc_med_alb[2],nc_med_alb[2]], 
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[1,0].plot([sn_med_olr[2],tn_med_olr[2],nn_med_olr[2]], 
           [sn_med_alb[2],tn_med_alb[2],nn_med_alb[2]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[1,0].plot([sf_med_olr[2],tf_med_olr[2],nf_med_olr[2]], 
           [sf_med_alb[2],tf_med_alb[2],nf_med_alb[2]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[1,0].plot([si_med_olr[2],ti_med_olr[2],ni_med_olr[2]], 
           [si_med_alb[2],ti_med_alb[2],ni_med_alb[2]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[1,0].plot([ss_med_olr[2],ts_med_olr[2],ns_med_olr[2]], 
           [ss_med_alb[2],ts_med_alb[2],ns_med_alb[2]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ax[1,1].plot([sc_med_olr[3],tc_med_olr[3],nc_med_olr[3]], 
           [sc_med_alb[3],tc_med_alb[3],nc_med_alb[3]], 
           lw=lw, ms=ms, label="CCCM", alpha=a,  color=c["OBS"])
ax[1,1].plot([sn_med_olr[3],tn_med_olr[3],nn_med_olr[3]], 
           [sn_med_alb[3],tn_med_alb[3],nn_med_alb[3]], 
           lw=lw, ms=ms, label="NICAM", alpha=a,  color=c["NICAM"])
ax[1,1].plot([sf_med_olr[3],tf_med_olr[3],nf_med_olr[3]], 
           [sf_med_alb[3],tf_med_alb[3],nf_med_alb[3]], 
           lw=lw, ms=ms, label="FV3", alpha=a,  color=c["FV3"])
ax[1,1].plot([si_med_olr[3],ti_med_olr[3],ni_med_olr[3]], 
           [si_med_alb[3],ti_med_alb[3],ni_med_alb[3]], 
           lw=lw, ms=ms, label="ICON", alpha=a,  color=c["ICON"])
ax[1,1].plot([ss_med_olr[3],ts_med_olr[3],ns_med_olr[3]], 
           [ss_med_alb[3],ts_med_alb[3],ns_med_alb[3]], 
           lw=lw, ms=ms, label="SAM", alpha=a,  color=c["SAM"])

ms = 80
ax[0,0].scatter([sc_med_olr[0],sn_med_olr[0],sf_med_olr[0],si_med_olr[0],ss_med_olr[0]],
              [sc_med_alb[0],sn_med_alb[0],sf_med_alb[0],si_med_alb[0],ss_med_alb[0]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[0,1].scatter([sc_med_olr[1],sn_med_olr[1],sf_med_olr[1],si_med_olr[1],ss_med_olr[1]],
              [sc_med_alb[1],sn_med_alb[1],sf_med_alb[1],si_med_alb[1],ss_med_alb[1]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,0].scatter([sc_med_olr[2],sn_med_olr[2],sf_med_olr[2],si_med_olr[2],ss_med_olr[2]],
              [sc_med_alb[2],sn_med_alb[2],sf_med_alb[2],si_med_alb[2],ss_med_alb[2]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,1].scatter([sc_med_olr[3],sn_med_olr[3],sf_med_olr[3],si_med_olr[3],ss_med_olr[3]],
              [sc_med_alb[3],sn_med_alb[3],sf_med_alb[3],si_med_alb[3],ss_med_alb[3]],
             marker="s",s=ms, edgecolors="k",label="SHL", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ms = 170
ax[0,0].scatter([tc_med_olr[0],tn_med_olr[0],tf_med_olr[0],ti_med_olr[0],ts_med_olr[0]],
              [tc_med_alb[0],tn_med_alb[0],tf_med_alb[0],ti_med_alb[0],ts_med_alb[0]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[0,1].scatter([tc_med_olr[1],tn_med_olr[1],tf_med_olr[1],ti_med_olr[1],ts_med_olr[1]],
              [tc_med_alb[1],tn_med_alb[1],tf_med_alb[1],ti_med_alb[1],ts_med_alb[1]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,0].scatter([tc_med_olr[2],tn_med_olr[2],tf_med_olr[2],ti_med_olr[2],ts_med_olr[2]],
              [tc_med_alb[2],tn_med_alb[2],tf_med_alb[2],ti_med_alb[2],ts_med_alb[2]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,1].scatter([tc_med_olr[3],tn_med_olr[3],tf_med_olr[3],ti_med_olr[3],ts_med_olr[3]],
              [tc_med_alb[3],tn_med_alb[3],tf_med_alb[3],ti_med_alb[3],ts_med_alb[3]],
             marker=".",s=ms, edgecolors="k",label="TWP", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ms = 200
ax[0,0].scatter([nc_med_olr[0],nn_med_olr[0],nf_med_olr[0],ni_med_olr[0],ns_med_olr[0]],
              [nc_med_alb[0],nn_med_alb[0],nf_med_alb[0],ni_med_alb[0],ns_med_alb[0]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[0,1].scatter([nc_med_olr[1],nn_med_olr[1],nf_med_olr[1],ni_med_olr[1],ns_med_olr[1]],
              [nc_med_alb[1],nn_med_alb[1],nf_med_alb[1],ni_med_alb[1],ns_med_alb[1]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,0].scatter([nc_med_olr[2],nn_med_olr[2],nf_med_olr[2],ni_med_olr[2],ns_med_olr[2]],
              [nc_med_alb[2],nn_med_alb[2],nf_med_alb[2],ni_med_alb[2],ns_med_alb[2]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])
ax[1,1].scatter([nc_med_olr[3],nn_med_olr[3],nf_med_olr[3],ni_med_olr[3],ns_med_olr[3]],
              [nc_med_alb[3],nn_med_alb[3],nf_med_alb[3],ni_med_alb[3],ns_med_alb[3]],
             marker="*",s=ms, edgecolors="k",label="NAU", c="none", zorder=2.5) # , c="none"list(c.values())[:5])

ax[0,0].set_ylim([0.3,0.8])
ax[0,0].set_xlim([95,150])
ax[0,0].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[0,0].set_ylabel('Albedo', size=fs)
ax[0,0].set_title('CAT 1', fontsize=fs)
ax[0,0].set_xticks(np.arange(95,155,10))
ax[0,0].tick_params(labelsize=fs-4)

ax[0,1].set_ylim([0.05,0.5])
ax[0,1].set_xlim([100,250])
ax[0,1].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[0,1].set_ylabel('Albedo', size=fs)
ax[0,1].set_title('CAT 2', fontsize=fs)
ax[0,1].set_xticks(np.arange(100,255,25))
ax[0,1].tick_params(labelsize=fs-4)

ax[1,0].set_ylim([0.05,0.3])
ax[1,0].set_xlim([250,300])
ax[1,0].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[1,0].set_ylabel('Albedo', size=fs)
ax[1,0].set_title('CAT 3', fontsize=fs)
ax[1,0].set_xticks(np.arange(250,305,10))
ax[1,0].tick_params(labelsize=fs-4)

ax[1,1].set_ylim([0.05,0.3])
ax[1,1].set_xlim([265,315])
ax[1,1].set_xlabel('OLR (W m$^{-2}$)', size=fs)
ax[1,1].set_ylabel('Albedo', size=fs)
ax[1,1].set_title('Clear Sky', fontsize=fs)
ax[1,1].set_xticks(np.arange(270,315,10))
ax[1,1].tick_params(labelsize=fs-4)
ax[1,1].grid(True)

ax[0,0].annotate("(a)", xy=(-0.2, 1.1), xycoords="axes fraction", fontsize=fs, weight="bold")
ax[0,1].annotate("(b)", xy=(-0.2, 1.1), xycoords="axes fraction", fontsize=fs, weight="bold")
ax[1,0].annotate("(c)", xy=(-0.2, 1.1), xycoords="axes fraction", fontsize=fs, weight="bold")
ax[1,1].annotate("(d)", xy=(-0.2, 1.1), xycoords="axes fraction", fontsize=fs, weight="bold")


h, l = ax[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="none",bbox_to_anchor=(1.2,0.7), fontsize=fs-4)

plt.tight_layout()
plt.savefig('../plots/fig12_cat_lifecycle_stn.png',dpi=150,bbox_inches='tight', pad_inches=0.1)
print('    saved to ../plots/fig12_cat_lifecycle_stn.png')
plt.show()

# %%
