# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utility import load, analysis_parameters as ap, load01deg


STAT="median"
REGION="TWP"

# %%
def get_cat3_iwc(model, region, stat=STAT):
    """Returns the profiles of iwc only for CAT 3"""
    fwp = load.get_iwp(model, region, ice_only=False)
    if model.lower()!="sam":
        fwp = fwp[11::12]
    twc = load.get_twc(model, region)
    print(fwp.shape, twc.shape)
    if (fwp.shape[0]!=twc.shape[0]): raise Exception("fwp and twc time not aligned: shapes", fwp.shape, twc.shape)
    if len(fwp.shape)==len(twc.shape):
        twc3 = np.where((fwp>1e-4)&(fwp<1e-2), twc, np.nan)
    elif len(fwp.shape)<len(twc.shape):
        fwp = fwp[:,np.newaxis]
        twc3 = np.where((fwp>1e-4)&(fwp<1e-2), twc, np.nan)
    else:
        raise Exception("fwp and twc shapes don't match",fwp.shape, twc.shape)
    if stat.lower()=="mean":
        if len(twc3.shape)<4: # icon
            twc3 = np.nanmean(twc3, axis=(0,2))
        else:
            twc3 = np.nanmean(twc3, axis=(0,2,3))
    elif stat.lower()=="median":
        if len(twc3.shape)<4: # icon
            twc3 = np.nanmedian(twc3, axis=(0,2))
        else:
            twc3 = np.nanmedian(twc3, axis=(0,2,3))
    else: raise Exception("stat ({}) not defined. Try 'mean' or 'median'.".format(stat))
    return twc3

def plot_twc(twc, model, region, ax=None, c="k", stat=STAT):
    z = load.get_levels(model, region)
    if model.lower()=="icon":
        z = z[14:]
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if (twc.shape==z.shape):
        ax.plot(twc, z, color=c)
    else: raise Exception("shapes don't match twc and z", twc.shape, z.shape)
    return ax

def plot_twc_all_models(region, ax=None, savename="../plots/twc3_{}_twp.png".format(STAT)):
    """Returns an matplotlib.pyplot.axis with all model twc3 plotted"""
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5,10))
    ntwc3 = get_cat3_iwc("NICAM", region)
    ax = plot_twc(ntwc3, "NICAM", region, ax=ax, c=ap.COLORS["NICAM"])
    ftwc3 = get_cat3_iwc("FV3", region)
    ax = plot_twc(ftwc3, "FV3", region, ax=ax, c=ap.COLORS["FV3"])
    itwc3 = get_cat3_iwc("ICON", region)
    ax = plot_twc(itwc3, "ICON", region, ax=ax, c=ap.COLORS["ICON"])
    stwc3 = get_cat3_iwc("SAM", region)
    ax = plot_twc(stwc3, "SAAM", region, ax=ax, c=ap.COLORS["SAM"])
    if ax is None:
        plt.savefig(savename)
    return ax

def plot_all(savename):
    """Returns a figure with all models and regions plotted. EAch region is a subplot."""
    fig, ax = plt.subplots(3,1,figsize=(15,10))
    plot_twc_all_models("TWP", ax=ax[1])
    plot_twc_all_models("SHL", ax=ax[0])
    plot_twc_all_models("NAU", ax=ax[2])
    plt.savefig(savename)
    plt.close()

def print_cat3_cre(model, region):
    """prints the CAT3 LW CRE"""
    ds = pd.read_csv("../tables/{}_{}.csv".format(model, region), index_col=0)
    lw_cre = ds.OLR.CS-ds.OLR.CAT3
    solar_const = 413.2335274 if (region.lower()=="shl") else 435.2760211
    sw_cre = solar_const * (ds.ALB.CS-ds.ALB.CAT3)
    net_cre = lw_cre - sw_cre
    print("{} {}\n\tLW\tSW\tNet\n\t{}\t{}\t{}".format(model, region,
                            int(lw_cre), int(sw_cre), int(net_cre)))
    return [lw_cre, sw_cre, net_cre]

def cat3_profiles(model, region):
    if model.lower()=="nicam":
        ice_only = False
    else:
        ice_only = True
    if model.lower()=="cccm":
        ds = load01deg.get_cccm(region)
        iwc = ds["iwc used in CERES radiation, avg over cloudy part of CERES footprint"].values/1000
        fwp = ds["iwp MODIS"].values/1000
        z = ds["alt"].values*1000
        # print(model, "\n", iwc.mean())
    elif model.lower()=="dardar":
        ds = load01deg.get_dardar(region)
        iwc = ds["iwc"].values/1000
        fwp = ds["iwp"].values/1000
        z = ds["height"].values
        # print(model, "\n", iwc.mean())
    else:
        iwc = load.get_twc(model, region).values
        fwp = load.get_iwp(model, region, ice_only=False).values
        z = load.get_levels(model, region)
        # print(model,"\n\t", iwc.shape, fwp.shape, z.shape)
    if model.lower()=="fv3" or model.lower()=="icon":
        fwp = fwp[11::12,np.newaxis]
    elif model.lower()=="nicam":
        fwp = fwp[11::12]
    else:
        fwp = fwp[:,np.newaxis]
    # print(fwp.shape, iwc.shape, z.shape)
    iwc3 = np.where((fwp>=1e-4)&(fwp<1e-2), iwc, np.nan)
    # print(fwp.shape, iwc.shape, iwc3.shape, z.shape)
    return (iwc3, z)

def get_cld_frac(iwc, thres=5e-7):
    """Returns the cloud occurrence at each vertical level using the threshold in kg/m2"""
    return

def get_num_cloud_layers(iwc, z, thres=5e-7):
    """Returns the number of cloud layers in each column.
    
    z must be in meters
    """
    cld_array = np.where(iwc>=thres, 1, 0)
    print("cloud fraction: ", np.nansum(iwc))
    ind0, ind1 = np.argmin(abs(z-10000)), np.argmin(abs(z-20000))
    if ind1<ind0:
        ind0, ind1 = ind1, ind0
    # initialize array with 1s if a cloud is at 7.5km 0 otherwise
    cld_layers = np.where(cld_array[:,ind0]==1, 1, 0)
    print("initial num cldy points", np.sum(cld_array[:,ind0]), np.sum(cld_layers))
    # If there is a new cloud layer increment cld_layers
    print(ind0, ind1, cld_layers.shape)
    for i in range(ind0, ind1):
        cld_layers = np.where((cld_array[:,i]==0)&(cld_array[:,i+1]==1), 
                               cld_layers+1, cld_layers)
        print(z[i], "cld layers", cld_layers.shape, np.nansum(cld_layers))
    return cld_layers
    

# %%

models = ["NICAM", "FV3", "ICON", "SAM"]
regions= ["SHL", "TWP","NAU"]
# iwc3, z = cat3_profiles("SAM","TWP")
# print(iwc3.shape, z.shape)
# plt.plot(np.nanmean(iwc3, axis=(0,2,3)), z)
# plt.ylim([0,20000])
# plt.xlabel("mean iwc (kg/m3)")
# plt.ylabel("Height (m)")
# plt.savefig("temp_vert_profile_cat3.png")
# plt.show()
# cre_list = np.zeros((len(models),len(regions),3))
# for i,m in enumerate(models):
    # for j,r in enumerate(regions):               
#         cre_list[i,j,:] = print_cat3_cre(m,r)
m = "ICON"
iwc3, z = cat3_profiles(m, "TWP")
cld_layers = get_num_cloud_layers(iwc3, z, thres=1e-9)

# %%
del iwc3, z
print("shape cld_layers", cld_layers.shape)

# %%
cld_layers = np.where(cld_layers==0, np.nan, cld_layers)
print(m, "\n\tmean_num_cld_layers:   ", np.nanmean(cld_layers),
         "\n\tmedian_num_cld_layers: ", np.nanmedian(cld_layers),
         "\n\tmax_num_cld_layers:    ", np.nanmax(cld_layers))
# if __name__=="__main__":
#     plot_twc_all_models("TWP")


# %%
ind0, ind1 = np.argmin(abs(z-6000)), np.argmin(abs(z-20000))
if ind1<ind0:
    ind0, ind1 = ind1, ind0
for i in range(ind0,ind1):
    print(i, z[i], np.nanmax(iwc3[:,i]))
# %%
