# %%
import matplotlib.pyplot as plt
from utility import load01deg as load
import numpy as np

model, region = "FV3", "TWP"

# %%
# FV3 lat lon plot of albedo, OLR, IWP, LWP
olr = load.get_olr(model, region)
swu = load.get_swu(model, region)
swd = load.get_swd(model, region)
alb = swu/swd
del swu, swd 
iwp = load.get_iwp(model, region, ice_only=False)
lwp = load.get_lwp(model, region, rain=False)
iwp = iwp[11::12]
lwp = lwp[11::12]
olr = olr[11::12]
alb = alb[11::12]

olr30 = np.zeros((olr.shape[0],30,30))
alb30 = np.zeros((olr.shape[0],30,30))
iwp30 = np.zeros((olr.shape[0],30,30))
lwp30 = np.zeros((olr.shape[0],30,30))

print("start loop")
for j in range(30):
    for k in range(30):
        olr30[:,j,k] = np.nanmean((olr[:,j*3:(j+1)*3,k*3:(k+1)*3]))
        alb30[:,j,k] = np.nanmean(alb[:,j*3:(j+1)*3,k*3:(k+1)*3])
        iwp30[:,j,k] = np.nanmean(iwp[:,j*3:(j+1)*3,k*3:(k+1)*3])
        lwp30[:,j,k] = np.nanmean(lwp[:,j*3:(j+1)*3,k*3:(k+1)*3])
print("end loop")

plt.plot(np.nanmean(alb30, axis=(1,2)))
plt.savefig("scratch_temp_alb30_timeseries_twp.png")
plt.close()

# %%
alb_high = np.where((alb30>0.4)&(olr30>225), alb30, np.nan)
olr_high = np.where((alb30>0.4)&(olr30>225), olr30, np.nan)
iwp_lowclouds = np.where((alb30>0.4)&(olr30>225), iwp30, np.nan)
lwp_lowclouds = np.where((alb30>0.4)&(olr30>225), lwp30, np.nan)

print(np.argmax(np.nanmean(iwp_lowclouds, axis=(1,2))))

# %%
lat = iwp.lat.values
lon = iwp.lon.values
lat30 = (lat[:90:3]+lat[3:93:3])/2
lon30 = (lon[:90:3]+lon[3:93:3])/2

# %%
t = 0
for t in range(20,40):
    fig, ax = plt.subplots(2,4, figsize=(17,6), sharex=True, sharey=True)
    cb0 = ax[0,0].pcolormesh(lon30, lat30, alb_high[t], cmap="cividis", vmin=0, vmax=1)
    cb1 = ax[0,1].pcolormesh(lon30, lat30, olr_high[t], cmap="viridis_r", vmin=225, vmax=310)
    cb2 = ax[0,2].pcolormesh(lon30, lat30, np.log10(iwp_lowclouds[t]), cmap="Blues", vmin=-5, vmax=1)
    cb3 = ax[0,3].pcolormesh(lon30, lat30, np.log10(lwp_lowclouds[t]), cmap="Greens", vmin=-5, vmax=1)
    plt.colorbar(cb0, ax=ax[0,0])
    plt.colorbar(cb1, ax=ax[0,1])
    plt.colorbar(cb2, ax=ax[0,2])
    plt.colorbar(cb3, ax=ax[0,3])

    cb0 = ax[1,0].pcolormesh(lon, lat, alb[t].values, cmap="cividis", vmin=0, vmax=1)
    cb1 = ax[1,1].pcolormesh(lon, lat, olr[t].values, cmap="viridis_r", vmin=225, vmax=310)
    cb2 = ax[1,2].pcolormesh(lon, lat, np.log10(iwp[t].values), cmap="Blues", vmin=-5, vmax=1)
    cb3 = ax[1,3].pcolormesh(lon, lat, np.log10(lwp[t].values), cmap="Greens", vmin=-5, vmax=1)
    plt.colorbar(cb0, ax=ax[1,0])
    plt.colorbar(cb1, ax=ax[1,1])
    plt.colorbar(cb2, ax=ax[1,2])
    plt.colorbar(cb3, ax=ax[1,3])

    ax[0,0].set_title("albedo")
    ax[0,1].set_title("OLR (W/m$^2$)")
    ax[0,2].set_title("IWP (log$_{10}$[kg/m$^3$])")
    ax[0,3].set_title("LWP (log$_{10}$[kg/m$^3$])")
    fig.suptitle("t = "+str(t))
    print("done.", t)
    plt.savefig("../plots/scratch_fv3_highalb-higholr_%s.png"%t, bbox_inches="tight",pad_inches=0.5)
    plt.show()
# %%
