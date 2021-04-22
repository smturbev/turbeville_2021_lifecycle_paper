#!/usr/bin/env/python
"""
    fig01_3drendering_fv3.py
    author: Sami Turbeville
    date created: 21 Dec 2020
    
    Generates 3D rendering of cloud water content (g/m3) for FV3 in TWP
    at t = 182.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utility import load, util
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

t = 182-(16)
greys = cm.get_cmap("Greys", 30)
model="FV3"

transp_grays = greys(range(20))
transp = np.zeros(transp_grays.shape)
for i in range(21):
    transp[-i,:-1] = transp_grays[-i,:-1]
    transp[-i,-1] = 0.8-(0.9*i/30)

new_cmap = ListedColormap(transp)

qn = load.load_tot_hydro1x1(model, "TWP", iceliq_only=True, exclude_shock=False) # liquid and ice water content
qn = qn * 1000 # convert to g/m3
print(qn.shape, qn.time[t].values)

tstring = str(qn.time[t].values).split("T")[-1][:5]+" UTC "+str(qn.time[t].values).split("-")[2][:2]+" Aug 2016"

if model=="FV3":
    z = np.nanmean(load.get_levels(model, "TWP"), axis=0)
else:
    z = load.get_levels(model, "TWP")

x3d = np.repeat(np.repeat(qn.lat.values[np.newaxis,:,np.newaxis],\
                                    qn.shape[1], axis=0), qn.shape[3], axis=2)
y3d = np.repeat(np.repeat(qn.lon.values[np.newaxis,np.newaxis,:],\
                          qn.shape[1], axis=0), qn.shape[2], axis=1)
z3d = np.repeat(np.repeat(z[:,np.newaxis, np.newaxis],\
                          qn.shape[2], axis=1), qn.shape[3], axis=2)

print(qn[0,:,:,:].shape, x3d.shape, z3d.shape)

xskip, yskip, zskip = 1,1,1
in_cloud = (np.where(qn[t,::zskip,::xskip,::yskip] > 5e-5, True, False)).flatten()
plt.rcParams["image.cmap"] = "Greys"

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=20)
sc = ax.scatter((x3d[::zskip,::xskip,::yskip]).flatten()[in_cloud], \
                (y3d[::zskip,::xskip,::yskip]).flatten()[in_cloud], \
                (z3d[::zskip,::xskip,::yskip]).flatten()[in_cloud]/1000,\
                c=(qn[t,::zskip,::xskip,::yskip].values.flatten())[in_cloud], edgecolors='face',\
                vmax = 0.5, vmin=0.001,\
                s=4*xskip, cmap=new_cmap, depthshade=True) #vmax = np.max(qn[t,:,:,:])/1.5,

ax.set_zlim(0,20)
ax.set_ylabel("Longitude ($^\circ$E)")
ax.set_xlabel("Latitude ($^\circ$N)")
ax.set_zlabel("Height (km)")
ax.set_yticks(np.arange(147,148,0.2))
ax.set_yticklabels([147,None,147.4,None,147.8,None])
ax.set_xticks(np.arange(-1,0,0.2))
ax.set_xticklabels([-1,None,-0.6,None,-0.2,None])
ax.set_title(model + ", " + tstring)
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("water content (g/m$^3$)")
ax.w_xaxis.set_pane_color((.5,.58,1.0))
ax.w_yaxis.set_pane_color((.5,.58,1.0))
ax.w_zaxis.set_pane_color((.2,.2,1.0))
plt.savefig("../plots/fig01_cloud3d.png", transparent=False, dpi=200)
plt.close()
