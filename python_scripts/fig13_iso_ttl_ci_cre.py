#!/usr/bin/env python
""" med_cat_on_joint_histogram.py
    Author: Sami Turbeville
    Updated: 16 Aug 2020
    
    This script plots the median of each category on the 
    joint albedo-olr histogram.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utility import util, analysis_parameters as ap

c = ap.COLORS
ttl = True
regions = ["SHL","TWP","NAU"]
models = ["CCCM","NICAM","FV3","ICON","SAM"]
net=False

# dar  = pd.read_csv("../tables/DARDAR_cat_alb_olr_ttl_cs.csv")
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
# %%
clw_twp = (ctwp.OLR.CS - ctwp.OLR.ISO_TTL)
nlw_twp = (ntwp.OLR.CS - ntwp.OLR.ISO_TTL)
flw_twp = (ftwp.OLR.CS - ftwp.OLR.ISO_TTL)
ilw_twp = (itwp.OLR.CS - itwp.OLR.ISO_TTL)
slw_twp = (stwp.OLR.CS - stwp.OLR.ISO_TTL)
clw_shl = (cshl.OLR.CS - cshl.OLR.ISO_TTL)
nlw_shl = (nshl.OLR.CS - nshl.OLR.ISO_TTL)
flw_shl = (fshl.OLR.CS - fshl.OLR.ISO_TTL)
ilw_shl = (ishl.OLR.CS - ishl.OLR.ISO_TTL)
slw_shl = (sshl.OLR.CS - sshl.OLR.ISO_TTL)
clw_nau = (cnau.OLR.CS - cnau.OLR.ISO_TTL)
nlw_nau = (nnau.OLR.CS - nnau.OLR.ISO_TTL)
flw_nau = (fnau.OLR.CS - fnau.OLR.ISO_TTL)
ilw_nau = (inau.OLR.CS - inau.OLR.ISO_TTL)
slw_nau = (snau.OLR.CS - snau.OLR.ISO_TTL)

csw_twp = ((ctwp.ALB.CS - ctwp.ALB.ISO_TTL)*413.2335274)
nsw_twp = ((ntwp.ALB.CS - ntwp.ALB.ISO_TTL)*413.2335274)
fsw_twp = ((ftwp.ALB.CS - ftwp.ALB.ISO_TTL)*413.2335274)
isw_twp = ((itwp.ALB.CS - itwp.ALB.ISO_TTL)*413.2335274)
ssw_twp = ((stwp.ALB.CS - stwp.ALB.ISO_TTL)*413.2335274)
csw_shl = ((cshl.ALB.CS - cshl.ALB.ISO_TTL)*435.2760211)
nsw_shl = ((nshl.ALB.CS - nshl.ALB.ISO_TTL)*435.2760211)
fsw_shl = ((fshl.ALB.CS - fshl.ALB.ISO_TTL)*435.2760211)
isw_shl = ((ishl.ALB.CS - ishl.ALB.ISO_TTL)*435.2760211)
ssw_shl = ((sshl.ALB.CS - sshl.ALB.ISO_TTL)*435.2760211)
csw_nau = ((cnau.ALB.CS - cnau.ALB.ISO_TTL)*413.2335274)
nsw_nau = ((nnau.ALB.CS - nnau.ALB.ISO_TTL)*413.2335274)
fsw_nau = ((fnau.ALB.CS - fnau.ALB.ISO_TTL)*413.2335274)
isw_nau = ((inau.ALB.CS - inau.ALB.ISO_TTL)*413.2335274)
ssw_nau = ((snau.ALB.CS - snau.ALB.ISO_TTL)*413.2335274)

lwcre = np.array([clw_shl/6, clw_twp/6, clw_nau/6, nlw_shl, nlw_twp, nlw_nau, \
        flw_shl, flw_twp, flw_nau, ilw_shl, ilw_twp, ilw_nau, \
        slw_shl, slw_twp, slw_nau])
swcre = np.array([csw_shl/6, csw_twp/6, csw_nau/6, nsw_shl, nsw_twp, nsw_nau, \
        fsw_shl, fsw_twp, fsw_nau, isw_shl, isw_twp, isw_nau, \
        ssw_shl, ssw_twp, ssw_nau])
netcre = lwcre + swcre


print(lwcre, swcre, netcre)
# %%
# bar plot
fig, ax = plt.subplots(1, 1, figsize=(7,4))
colors = ["darkgray"]*3+[c["NICAM"]]*3+[c["FV3"]]*3+[c["ICON"]]*3+[c["SAM"]]*3
# colors = ["tab:orange","tab:green","tab:purple"]
x_list = []
for i in range(60):
    if i%4!=0:
        x_list.append(i)
print(x_list)
bar_list = ax.bar(x_list, list(lwcre)+list(swcre)+list(netcre), width=1,
        align="center", color=colors, alpha=0.9, edgecolor="k")
for i in range(len(bar_list)):
    if i%3==0:
        hatch = "///"
    elif i%3==1:
        hatch = None
    else:
        hatch = "..."
    bar_list[i].set_hatch(hatch)
ax.scatter(-10,0, color="darkgray", label="CCCM")
ax.scatter(-10,0, color=list(c.values())[1], label="NICAM")
ax.scatter(-10,0, color=list(c.values())[2], label="FV3")
ax.scatter(-10,0, color=list(c.values())[3], label="ICON")
ax.scatter(-10,0, color=list(c.values())[4], label="SAM")
h, l = ax.get_legend_handles_labels()
axt = ax.twinx()
axt.axis("off")
axt.set_xlim([0,60])
axt.bar(-10,3,color="lightgray", hatch="///", label="SHL")
axt.bar(-10,3,color="lightgray", hatch=None, label="TWP")
axt.bar(-10,3,color="lightgray", hatch="...", label="NAU")
# fig.legend(loc=9, bbox_to_anchor=(0,0.05),ncol=3)
ax.set_xlim([0,60])
ax.set_ylim([-15.1,25])
ht, lt = axt.get_legend_handles_labels()
print(l, lt, type(l))
h_new, l_new = (h)+(ht), (l)+(lt)
ax.legend(loc=9, bbox_to_anchor=(0.5,0), ncol=5)
axt.legend(loc=9, bbox_to_anchor=(0.5,-0.1), ncol=3)
ax.axvline(20, color="gray")
ax.axvline(40, color="gray")
ax.axhline(0, color="gray")

ax.set_xticks(np.arange(2,60,4))
ax.set_xticklabels([""]*15) # ["C","N","F","I","S","","","","","","C","N","F","I","S"])
ax.spines["bottom"].set_position(("data",0))
ax.tick_params(bottom=False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
# ax1 = plt.gca().twiny()
# ax1.set_xticks(np.arange(2,60,4))
# ax1.set_xticklabels(["","","","","","C","N","F","I","S","","","","",""])
# ax1.set_xlim([-1,60])
# ax1.spines["top"].set_position("zero")
# ax1.spines["bottom"].set_position("zero")
# ax1.xaxis.set_ticks_position('top')
ax.grid(axis="y")
ax.set_axisbelow(True)

ax.annotate("LW CRE", xycoords="axes fraction", xy=(0.12,0.9))
ax.annotate("SW CRE", xycoords="axes fraction", xy=(0.45,0.9))
ax.annotate("Net CRE", xycoords="axes fraction", xy=(0.78,0.9))

ax.set_ylabel("CRE (W/m$^2$)")

plt.savefig("../plots/fig14_iso_ttl_ci_cre_bar.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()


# %%
# lw = 5
# ms = 20
# a = 1
# fs = 18

# # Plot median
# fig, ax = plt.subplots(1,1,figsize=(10,5), constrained_layout=True, sharey=False)

# ax.set_xticks([-0.3,0.7,1.7,2.7], minor=True)
# ax.set_xticks([0.23,1.23,2.23])
# ax.plot([-0.3,2.7],[0,0],'gray',lw=0.5)
# ax.axvline(0.06,color="gray", linestyle="--", alpha=0.6)
# ax.axvline(0.36,color="gray", linestyle="--", alpha=0.6)
# ax.axvline(1.06,color="gray", linestyle="--", alpha=0.6)
# ax.axvline(1.36,color="gray", linestyle="--", alpha=0.6)
# ax.axvline(2.06,color="gray", linestyle="--", alpha=0.6)
# ax.axvline(2.36,color="gray", linestyle="--", alpha=0.6)
# models=["CCCM","NICAM","FV3","ICON","SAM"]

# old double for loop
# for i,r in enumerate(regions):
#     print(i,r)
#     if r=="TWP":
#         olr_cs = np.array([ctwp.OLR.CS, ntwp.OLR.CS,
#                            ftwp.OLR.CS, itwp.OLR.CS, stwp.OLR.CS])
#         alb_cs = np.array([ctwp.ALB.CS, ntwp.ALB.CS,
#                            ftwp.ALB.CS, itwp.ALB.CS, stwp.ALB.CS])
#         olr_isottl = np.array([ctwp.OLR.ISO_TTL, ntwp.OLR.ISO_TTL,
#                                ftwp.OLR.ISO_TTL, itwp.OLR.ISO_TTL, stwp.OLR.ISO_TTL])
#         alb_isottl = np.array([ctwp.ALB.ISO_TTL, ntwp.ALB.ISO_TTL,
#                                ftwp.ALB.ISO_TTL, itwp.ALB.ISO_TTL, stwp.ALB.ISO_TTL])
#         const = 413.2335274
#     elif r=="NAU":
#         olr_cs = np.array([cnau.OLR.CS, nnau.OLR.CS,
#                            fnau.OLR.CS, inau.OLR.CS, snau.OLR.CS])
#         alb_cs = np.array([cnau.ALB.CS, nnau.ALB.CS,
#                            fnau.ALB.CS, inau.ALB.CS, snau.ALB.CS])
#         olr_isottl = np.array([cnau.OLR.ISO_TTL, nnau.OLR.ISO_TTL,
#                                fnau.OLR.ISO_TTL, inau.OLR.ISO_TTL, snau.OLR.ISO_TTL])
#         alb_isottl = np.array([cnau.ALB.ISO_TTL, nnau.ALB.ISO_TTL,
#                                fnau.ALB.ISO_TTL, inau.ALB.ISO_TTL, snau.ALB.ISO_TTL])
#         const = 413.2335274
#     else:
#         olr_cs = np.array([cshl.OLR.CS, nshl.OLR.CS,
#                            fshl.OLR.CS, ishl.OLR.CS, sshl.OLR.CS])
#         alb_cs = np.array([cshl.ALB.CS, nshl.ALB.CS,
#                            fshl.ALB.CS, ishl.ALB.CS, sshl.ALB.CS])
#         olr_isottl = np.array([cshl.OLR.ISO_TTL, nshl.OLR.ISO_TTL,
#                                fshl.OLR.ISO_TTL, ishl.OLR.ISO_TTL, sshl.OLR.ISO_TTL])
#         alb_isottl = np.array([cshl.ALB.ISO_TTL, nshl.ALB.ISO_TTL,
#                                fshl.ALB.ISO_TTL, ishl.ALB.ISO_TTL, sshl.ALB.ISO_TTL])
#         const = 435.2760211 # SHL
#     lwcre = (olr_cs - olr_isottl)
#     swcre = (alb_cs - alb_isottl)*const
#     netcre = lwcre, swcre
#     lwcre[0] = lwcre[0]/6
#     swcre[0] = swcre[0]/6
#     netcre[0] = netcre[0]/6
#     for j,m in enumerate(models):

#         if r=="TWP":
#             labnet = m
#         else:
#             lablw, labsw, labnet = None, None, None
#         if m=="CCCM":
#             m = "OBS"
#             print("OBS", lwcre[0], swcre[0], netcre[0])
#         ax.scatter([i-0.18+0.04*j],[lwcre[j]], c=c[m], 
#                   marker='s', s=fs*4, label=lablw) #markerfacecolor='none'
#         ax.scatter([i+0.12+0.04*j],[swcre[j]], edgecolor=c[m], label=labsw,
#                   marker='s', c='none', s=fs*4)
#         ax.scatter([i+0.42+0.04*j],[netcre[j]], c=c[m], 
#                    marker='o', s=fs*4, label=labnet)

# ax.set_xlim([-0.3,2.7])
# ax.set_xticklabels(['SHL', 'TWP', 'NAU'])
# ax.grid(which='minor',color="black")
# ax.set_title('Isolated TTL Cirrus CRE', fontsize=fs)
# ax.set_ylabel('CRE (W/m$^2$)', fontsize=fs)
# ax.legend(bbox_to_anchor=(1.01,.4), fontsize=fs-4)
# ax.tick_params(axis='y', labelsize=fs-4)
# ax.tick_params(axis='x', labelsize=fs)
# ax.set_ylim([-30,30])

# ax2 = ax.twiny()
# ax2.set_xlim([-0.3,2.7])
# new_tick_locations = np.arange(-0.1,2.7,0.33) #xbin_perc
# # Move twinned axis ticks and label from top to bottom
# ax2.xaxis.set_ticks_position("bottom")
# ax2.xaxis.set_label_position("bottom")
# ax2.set_xticks(new_tick_locations)
# ax2.set_xticklabels(["LW","SW","Net","LW","SW","Net","LW","SW","Net"], fontsize=fs)
# # Offset the twin axis below the host
# ax.spines["bottom"].set_position(("axes", -0.1))
# ax.spines['bottom'].set_visible(False)

# ax.annotate("(d)", xy=(0.01,0.94), xycoords="axes fraction", fontsize=fs)



plt.savefig('../plots/fig13_iso_ttl_ci_cre.png',
            dpi=150,bbox_inches='tight')
print('    saved to ../plots/fig13_iso_ttl_ci_cre.png')
plt.close()
