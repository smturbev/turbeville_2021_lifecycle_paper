#!/usr/bin/env python
"""
    fig01_3drendering_fv3.py
    author: Sami Turbeville
    date created: 21 Dec 2020
    
    Generates 3D rendering of cloud water content (g/m3) for FV3 in TWP
    at t = 182.
"""
# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utility import load
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

is_one_by_one = True
t = 182-16
greys = cm.get_cmap("gist_yarg", 28)
model="FV3"

transp_grays = greys(range(20))
transp = np.zeros(transp_grays.shape)
for i in range(21):
    transp[-i,:-1] = transp_grays[-i,:-1]
    if is_one_by_one:
        transp[-i,-1] = 0.5-(np.sqrt(i)/18) # 0.6-(0.85*i/30)
    else:
        transp[-i,-1] = 0.3-(np.sqrt(i)/18) # 0.6-(0.85*i/30)
    print(transp[-i,-1])

new_cmap = ListedColormap(transp)
# %%
if is_one_by_one:
    qn = load.load_tot_hydro1x1(model, "TWP", iceliq_only=True, exclude_shock=False) # liquid and ice water content
else:
    qn = load.load_tot_hydro(model, "TWP", ice_only=True)
qn = qn[t] * 1000 # convert to g/m3
print(qn.shape, qn.time.values)

tstring = str(qn.time.values).split("T")[-1][:5]+" UTC "+str(qn.time.values).split("-")[2][:2]+" Aug 2016"

if model=="FV3":
    z = load.get_levels(model, "TWP")
else:
    z = load.get_levels(model, "TWP")
# %%
if is_one_by_one:
    lat = qn.lat.values
    lon = qn.lon.values
else:
    lat = qn.grid_yt.values
    lon = qn.grid_xt.values
x3d = np.repeat(np.repeat(lat[np.newaxis,:,np.newaxis],\
                          qn.shape[0], axis=0), qn.shape[2], axis=2)
y3d = np.repeat(np.repeat(lon[np.newaxis,np.newaxis,:],\
                          qn.shape[0], axis=0), qn.shape[1], axis=1)
z3d = np.repeat(np.repeat(z[:,np.newaxis, np.newaxis],\
                          qn.shape[1], axis=1), qn.shape[2], axis=2)

print(qn[:,:,:].shape, x3d.shape, z3d.shape)
# %%
if is_one_by_one:
    xskip, yskip, zskip = 1, 1, 1
else:
    xskip, yskip, zskip = 5, 5, 1
in_cloud = (np.where(qn[::zskip,::xskip,::yskip] > 1e-4, True, False)).flatten()
# plt.rcParams["image.cmap"] = new_cmap

# %%
fs = 14
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=20)
sc = ax.scatter((x3d[::zskip,::xskip,::yskip]).flatten()[in_cloud], \
                (y3d[::zskip,::xskip,::yskip]).flatten()[in_cloud], \
                (z3d[::zskip,::xskip,::yskip]).flatten()[in_cloud]/1000,\
                c=(qn[::zskip,::xskip,::yskip].values.flatten())[in_cloud], \
                edgecolors='face',\
                vmax = (0.5), vmin=0.1,\
                s=4*xskip, cmap=new_cmap, depthshade=True) #vmax = np.max(qn[:,:,:])/1.5,
if not(is_one_by_one):
    ax.plot([  0,  0,  -1,  -1,  0,  0, 0, -1, -1, 0],
            [147,148,148,147,147,147,148,148,147,147],
            [ 20 ,20, 20, 20, 20,  0, 0, 0, 0, 0], c='r')
    ax.plot([    0,  0,  0,  0,  -1,  -1,  -1,  -1,  -1],
            [147,147,148,148,148,148,148,147,147],
            [ 0 , 20, 20, 0,  0,  20, 20, 20, 0], c='r')
ax.set_zlim(0,20)
ax.set_ylabel("Longitude ($^\circ$E)", fontsize=fs)
ax.set_xlabel("Latitude ($^\circ$N)", fontsize=fs)
ax.set_zlabel("Height (km)", fontsize=fs)
if is_one_by_one:
    ax.set_yticks(np.arange(147,148,0.2))
    ax.set_yticklabels([147,None,147.4,None,147.8,None])
    ax.set_xticks(np.arange(-1,0,0.2))
    ax.set_xticklabels([-1,None,-0.6,None,-0.2,None])
    ax.annotate("(b)", xycoords="axes fraction", xy=(0.55,-0.1), fontsize=fs+4)
else:
    ax.annotate("(a)", xycoords="axes fraction", xy=(0.55,-0.1), fontsize=fs+4)
# ax.set_title(model + ", " + tstring)
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("water content (g/m$^3$)", fontsize=fs-2)
ax.w_xaxis.set_pane_color((.5,.58,1.0))
ax.w_yaxis.set_pane_color((.5,.58,1.0))
ax.w_zaxis.set_pane_color((.2,.2,1.0))
ax.tick_params(labelsize=fs-2)
cbar.ax.tick_params(labelsize=fs-2)
if is_one_by_one:
    plt.savefig("../plots/fig01_cloud3d.png", transparent=False, dpi=200, bbox_inches="tight", pad_inches=0.2)
else:
    ax.invert_xaxis()
    plt.savefig("../plots/fig01_cloud3d_10x10.png", transparent=False, dpi=200, bbox_inches="tight", pad_inches=0.2)
plt.close()

# %%
