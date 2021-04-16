""" util.py
    Sami Turbeville
    11/7/2019
    
    module for useful functions to keep code in python_scripts clean
"""
from datetime import datetime
import numpy as np
import xarray as xr
from scipy import stats
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
import matplotlib.transforms as trans
from . import analysis_parameters as ap
from . import load
np.warnings.filterwarnings("ignore")
#################### Hydrometeors ###############################

def ttl_iwp_wrt_pres(q, p, model, region):
    p_tz = False
    xy = True
    z = load.get_levels(model, region)
    if model.lower()=="nicam":
        ind0, ind1 = np.argmin(abs(z-14000)), np.argmin(abs(z-18000))+1
    elif model.lower()=="fv3":
        z= np.nanmean(z,axis=0)
        ind0, ind1 = (min([np.argmin(abs(z-18000)),np.argmin(abs(z-14000))]), \
                      max([np.argmin(abs(z-14000)), np.argmin(abs(z-18000))])+1)
    elif model.lower()=="icon":
        z = z[14:]
        ind0, ind1 = (min([np.argmin(abs(z-14000)), np.argmin(abs(z-18000))]), \
                      max([np.argmin(abs(z-14000)), np.argmin(abs(z-18000))])+1)
        xy=False
    elif model.lower()=="sam":
        ind0, ind1 = (min([np.argmin(abs(z-14000)), np.argmin(abs(z-18000))]), \
                      max([np.argmin(abs(z-14000)), np.argmin(abs(z-18000))])+1)
        p_tz = True
    else:
        raise Exception("Model or region not defined. Try FV3, ICON, SAM, NICAM in the TWP, SHL, or NAU.")
    print(ind0,ind1, z[ind0], z[ind1])
    if ((xy) & (~p_tz)):
        vint = iwp(q[:,ind0:ind1,:,:],p[:,ind0:ind1,:,:],model,region)
    elif not(p_tz): #xy=False = ICON
        vint = iwp(q[:,ind0:ind1,:],p[:,ind0:ind1,:],model,region)
    else: #xy=True and p_tz=True = SAM
        vint = iwp(q[:,ind0:ind1,:,:],p[:,ind0:ind1,np.newaxis,np.newaxis],model,region)
    return vint

def iwp(q, p, model, region):
    """ Inputs must be in kg/kg and Pa. """
    p = np.where(np.isnan(p),0,p)
    q = np.where(np.isnan(q),0,q)
    if model.lower()=="nicam":
        vint = int_wrt_pres(p,q,xy=True,const_p=False)
    elif model.lower()=="fv3":
        vint = int_wrt_pres_f(p,q)
    elif model.lower()=="icon":
        vint = int_wrt_pres(p,q,xy=False,td=True,const_p=False)
    elif model.lower()=="sam":
        vint = int_wrt_pres(p,q,xy=True,td=False,const_p=True)
    else:
        raise Exception("Model or region not defined. Try FV3, ICON, SAM, NICAM in the TWP, SHL, or NAU.")
    return vint

def iwp_wrt_pres(model, region, hydro_type="ice"):
    if ((region=="TWP")&(model=="ICON")):
        iwp = xr.open_dataset(ap.TWP_ICON_IWP)["iwp"].values
        return iwp
    else:
        p = load.get_pres(model,region)
        if hydro_type=="ice":
            q = load.load_frozen(model, region, ice_only=True).values
        elif hydro_type=="frozen":
            q = load.load_frozen(model, region, ice_only=False).values
        else:
            q = load.load_tot_hydro(model, region).values
        p = np.where(np.isnan(p),0,p)
        q = np.where(np.isnan(q),0,q)
        if model.lower()=="nicam":
            vint = int_wrt_pres(p,q,xy=True,const_p=False)
        elif model.lower()=="fv3":
            vint = int_wrt_pres_f(p,q)
        elif model.lower()=="icon":
            vint = int_wrt_pres(p,q,xy=False,td=True,const_p=False)
        elif model.lower()=="sam":
            vint = int_wrt_pres(p,q,xy=True,td=False,const_p=True)
        else:
            raise Exception("Model or region not defined. Try FV3, ICON, SAM, NICAM in the TWP, SHL, or NAU.")
        return vint
    
def int_wrt_pres(p, q, xy=True, td=False, const_p=False):
    """
    Integrate wrt pressure, where pressure varies in time.
    Assumes p and q are saved on the same vertical level.
    
    Args:
        p (numpy array): pressures in Pa
        q (numpy array): hydrometeor mixing ratio in kg/kg
        xy (boolean): true if horizontal dimension has 2 coordinates
        const_p (boolean): true if pressure data only varies in time
    Returns:
        vint (numpy array): vertically integrated hydrometor
                            in kg/m^2
    """
    if xy:
        nt, nh, nx, ny = q.shape
        vint = np.empty((nt, nx, ny))
        g = 9.8 #m/s^2
        for t in range(nt):
            vsum = np.zeros((nx, ny))
            for n in range(1, nh-1):
                if not const_p:
                    dp = 0.5*(p[t,n+1,:,:]-p[t,n-1,:,:])
                else:
                    dp = 0.5*(p[t,n+1]-p[t,n-1])
                if td:
                    calc = (q[t,n,:,:]*dp)/g
                else:
                    calc = -1*(q[t,n,:,:]*dp)/g 
                vsum = calc + vsum 
            vint[t, :, :] = vsum
    
    else: # ICON
        nt, nh, nc = q.shape
        vint = np.empty((nt, nc))
        g = 9.8 #m/s^2
        for t in range(nt):
            vsum = np.zeros(nc)
            for n in range(1, nh-1):
                if not const_p:
                    dp = 0.5*(p[t,n+1,:]-p[t,n-1,:])
                else:
                    dp = 0.5*(p[t,n+1]-p[t,n-1])
                if td:
                    calc = (q[t,n,:]*dp)/g
                else:
                    calc = -1*(q[t,n,:]*dp)/g
                vsum = calc + vsum 
            vint[t, :] = vsum
    
    return vint

def int_wrt_pres_f(p, q):
    """
    Integrate wrt pressure for FV3 - where variables
    have 2 horizontal coords (x, y), order is top-down, and
    pressure varies in time. Assumes p and q are saved on the
    same vertical level.
    
    Args:
        p (numpy array): pressures in Pa
        q (numpy array): hydrometeor mixing ratio in kg/kg
    Returns:
        vint (numpy array): vertically integrated hydrometor
                            in kg/m^2
    """
    nt, nhtot, nx, ny = q.shape
    vint = np.empty((nt, nx, ny))
    g = 9.8 #m/s^2
    
    for t in range(nt):
        qt = q[t, :, :, :]
        pt = p[t, :, :, :]
        nh = nhtot

        vsum = np.zeros((nx, ny))
        for n in range(1, nh-1):
            dp = 0.5*(pt[n+1,:,:]-pt[n-1,:,:])
            calc = (qt[n,:,:]*dp)/g
            vsum = calc + vsum

        vint[t, :, :] = vsum

    return vint

def int_wrt_alt(iwc, z):
    """ Returns the vertically integrated path from iwc
        (ice water content or liquid).
        
        Args:
            - iwc : (numpy array) ice water content in kg/m3 with
                    height is the 2nd dimension (e.g iwc[time,height])
            - z   : (numpy array) one dimensional array with height in meters
        
        Returns:
            - vint : (numpy array) has dimensions of iwc (without height dimension)
    """
    calc = np.zeros(iwc.shape)
    for i in range(1,z.shape[0]-2):
        dz = abs(z[i-1]-z[i+1])
        calc[:,i] = ((iwc[:,i])*dz)/9.81
    vint = np.nansum(calc, axis=1)
    return vint

def q_to_iwc(q, model, region):
    """Converts mixing ratio of q (kg/kg) to ice water content (kg/m3)
        input = model name (string) and q = mixing ratio as xarray or numpyarray.
        Only works for time and space averaged data (aka data has one dimension-height)
        
        returns xarray or numpy array with iwc as kg/m3
    """
    if model.lower() == "fv3":
        t = load.get_temp(model, region)
        qv = load.get_qv(model, region)
        p = load.get_pres(model, region)
        rho = p / \
              (287*(1 + 0.61*(qv))*(np.nanmean(t, axis=(2))[:,:,np.newaxis,np.newaxis]))
        iwc = q.values * rho
        print("Warning: FV3 uses the spatially averaged density b/c \
        specific humidity and temperature are on different grids")
    elif model.lower() =="sam":
        t = load.get_temp(model, region).values
        qv = load.get_qv(model, region).values
        p = load.get_pres(model, region).values
        rho = p[:,:,np.newaxis,np.newaxis] / \
              (287*(1 + 0.61*qv)*t)
        iwc = q.values * rho
    else:
        if model.lower() == "icon":
            t = load.get_temp(model, region).values.astype('float32')
            qv = load.get_qv(model, region).values.astype('float16')
            Tv = (1 + 0.61*qv)*t
            print("... Tv ...")
            del qv, t
            p = load.get_pres(model, region).values.astype('float32')
        else:
            t = load.get_temp(model, region).values
            qv = load.get_qv(model, region).values
            p = load.get_pres(model, region).values
            Tv = (1 + 0.61*qv)*t
            print("... Tv ...")
            del qv, t
        rho = p / (287*Tv) 
        print("... rho ...")
        del p, Tv
        iwc = q * rho  # kg/m2
        print("... iwc ...")
        del rho
    print("Returning ice water content (kg/m3) for %s as %s xarray\n\n"%(model, iwc.shape))
    iwcxr = xr.DataArray(iwc, dims=list(q.dims), coords=q.coords, 
                     attrs={'standard_name':'iwc','long_name':'ice_water_content','units':'kg/m3'})
    return iwcxr

def iwc(q, t, qv, p, model):
    """Converts mixing ratio of q (kg/kg) to ice water content (kg/m3)
        
        returns xarray or numpy array with iwc as kg/m3
    """
    print('shape qv, t',qv.shape, t.shape)
    if model.lower() == "fv3":
        rho = p / \
              (287*(1 + 0.61*(qv))*(np.nanmean(t, axis=(2))[:,:,np.newaxis,np.newaxis]))
        iwc = q.values * rho
        print("Warning: FV3 uses the spatially averaged density b/c \
        specific humidity and temperature are on different grids")
    elif model.lower() =="sam":
        rho = p[:,:,np.newaxis,np.newaxis] / \
              (287*(1 + 0.61*(qv))*(t))
        iwc = q.values * rho
    else:
        if model.lower() == "icon":
            t = t.astype('float32')
            p = p.astype('float32')
            qv = qv.astype('float16')
        rtv = (287*(1 + 0.61*qv)*t)
        del qv, t
        rho = p / rtv 
        del p, rtv
        iwc = q * rho # kg/m2
    print("Returning ice water content (kg/m2) for %s as %s xarray\n\n"%(model, iwc.shape))
    iwcxr = xr.DataArray(iwc, dims=list(q.dims), coords=q.coords, 
                     attrs={'standard_name':'iwc','long_name':'ice_water_content','units':'kg/m3'})
    return iwcxr

def iwv(model, region):
    """ Returns the total column integrated water vapor for model and region.
    
        
        iwv = -1/g * integral(qv * dp)
        
        model = string of which dyamond model to use
        region = string of "TWP", "SHL" or "NAU"
    """
    p = load.get_pres(model, region)
    qv = load.get_qv(model, region)
    print("shape of p and qv:",p.shape, qv.shape)
    cur = q_loop(model, region, qv, p)
    del qv, p
    print("water vapor content done...\n... Summing columns...")
    iwv = 1/9.8 * np.nansum(cur,axis=1) / 10 # g/cm2 (conversion: 10 kg/m2 = 1 g/cm2)
    del cur
    print("Returned IWV (g/cm2) for {} in {} with shape".format("nicam","twp"),iwv.shape)
    return iwv

def q_loop(model, region, q, p, levels=(1,None)):
    """ Returns array for vertical integration. """
    base, top = levels      
    if top==None:
        print("Total column integration:")
        top = q.shape[1]-1
        print(top, "levels")
    print("Starting Loop for IWP...")
    if model.lower()=="nicam" or model.lower()=="sam":
        cur = nsg_q_loop(q.values,p,base,top)
    elif model.lower()=="fv3":
        # on the pressure level 
        print("    ", np.nanmean(q.pfull[-1]), "hPa of lowest level")
        cur = f_q_loop(q.values,q.pfull.values,base,top)
    elif model.lower()=="icon":
        print("    icon")
        cur = q.values[:,base:top+1,:]*abs(p[:,base:,:]-p[:,:top,:])/2
    else:
        raise Exception("Model ({}) not supported. Try ICON, NICAM, FV3, or SAM.".format(model))
    print("Looping done...")
    return cur

def nsg_q_loop(q, p, base, top):
    cur = q[:,base:top+1,:,:]*abs((p[:,base:,:,:]-p[:,:top,:,:])/2)
    return cur

def f_q_loop(q,p,base,top):
    cur = q[:,base:top+1,:,:]*abs((p[base:]-p[:top])*100/2)[np.newaxis,:,np.newaxis,np.newaxis]
    return cur

def i_q_loop(cur,q,p,base,top):
    cur = q[:,base:top+1,:]*abs(p[:,base:,:]-p[:,:top,:])/2
    return cur
######################## Data Manipulation ###############################
def llavg(data, model="FV3", var="qi", dim=3, region="TWP", save=False):
    ''' input data as xarray.DataArray, model name, variable name,
        the number of dimensions (3d or 2d data), save is a boolean
        returns the averaged data
    '''
    if model=="FV3" or model=="fv3" or model=="Fv3":
        ntime = len(data.time)
        nlon  = len(data.grid_xt)
        nlat  = len(data.grid_yt)
        lon = data.grid_xt
        lat = data.grid_yt
        if dim==3:
            nz = len(data.pfull)
            data_llavg = np.zeros((ntime, nz, 31, 31))
        elif dim==2:
            data_llavg = np.zeros((ntime,31,31))
        new_lat = [None]*31
        new_lon = [None]*31
        attrs = data.attrs
        n=11
        v = "pfull"
        if dim==3:
            z = data.pfull
    elif model=="ICON" or model=="icon":
        ntime = len(data.t)
        if dim==3:
            nz = len(data.lev)
            data_llavg = np.zeros((ntime, nz, 31, 31))
        elif dim==2:
            data_llavg = np.zeros((ntime,31,31))
        new_lat = [None]*31
        new_lon = [None]*31
        attrs = data.attrs
        n=11
        v = "lev"
        if dim==3:
            z = data.lev
    elif model=="NICAM" or model=="nicam":
        n = 9
        ntime = len(data.time)
        lon = data.lon
        lat = data.lat
        nlon  = len(lon)
        nlat  = len(lat)
        if dim==3:
            nz = len(data.lev)
            data_llavg = np.zeros((ntime, nz, nlon//n, nlat//n))
        elif dim==2:
            data_llavg = np.zeros((ntime,nlon//n,nlat//n))
#             data = data[:,0,:,:]
        new_lat = [None]*(nlat//n)
        new_lon = [None]*(nlon//n)
        attrs = data.attrs
        v = "lev"
        if dim==3:
            z = data.lev
    else: 
        print("models other than FV3 not supported at this time... sorry")
        return
    stime = time.time()
    print("Averaging %s %s from shape of"%(model,var), data.shape, "to", data_llavg.shape)
    if dim==3:
        print("    Process Started (3D)...")
        for i in range(nlon//n):
            for j in range(nlat//n):
#                 print(i,j)
                data_llavg[:,:,j,i] = data[:,:,(j*n):(j+1)*n,(i*n):(i+1)*n].mean(axis=(2,3)).values
                new_lat[j] = lat[j*n:(j+1)*n].mean().values
                new_lon[i] = lon[i*n:(i+1)*n].mean().values
        print("    Converting to new xarray...")
        da = xr.DataArray(data_llavg[:,:,:,:], dims=["time", v, "lat", "lon"], 
                          coords={"time":data.time,v:z,"lon":new_lon[:],"lat":new_lat[:]}, attrs=attrs)
    elif dim==2:
        print("    Process Started (2D)...")
        for i in range(nlon//n):
            for j in range(nlat//n):
                data_llavg[:,j,i] = data[:,(j*n):(j+1)*n,(i*n):(i+1)*n].mean(axis=(1,2)).values
                new_lat[j] = lat[j*n:(j+1)*n].mean().values
                new_lon[i] = lon[i*n:(i+1)*n].mean().values
        print("   ...converting to new xarray")
        da = xr.DataArray(data_llavg[:,:,:], dims=["time", "lat", "lon"], 
                          coords={"time":data.time,"lon":new_lon[:],"lat":new_lat[:]}, attrs=attrs)
    ds = xr.Dataset({'%s_30km_avg'%(var):da})
    ds.attrs = {'long_name':'Native_%s_%s_averaged_over_0.3deg'%(model, var)}
    print("    Process Finished: new shape:", da.shape)

    if save:
        if model=="FV3":
            savename = ap.FV3 + "FV3_%s_0.3deg_%s.nc"%(var, region)
        elif model=="NICAM":
            savename = ap.NICAM + "NICAM_%s_0.3deg_%s.nc"%(var, region)
        ds.to_netcdf(savename)
        print("saved as "+savename)
    etime = time.time()
    print("    That took %f minutes"%((etime-stime)/60))
    return ds   
 
def rho(qv, p, t):
    """calculates density of air in kg/m3 for given input"""
    R =  287 # (Gas constant of air) J/(kg*K)
    Tv = (1 + 0.61*qv) * t # K
    rho = p / (R*Tv) # kg/m3
    return rho
    
def undomean(meanarray, xy=True):
    """pass in whole xarray with dimensions time, lat, lon
       for ICON radiation variables only
       
       returns new xarray (the running mean is undone), so
       it is only the raw data
    """
    # initialize
    data = np.empty(np.shape(meanarray))
    if xy:
        data[0,:,:] = meanarray[0,:,:] # they both will be zero
        data[1,:,:] = meanarray[1,:,:] # this is the actual first data point... 

        # loop forward through the rest of the indices in array to "undo" the "mean from model start"
        for i in np.arange(2,np.shape(meanarray)[0]):
            data[i,:,:] = (i+1)*meanarray[i,:,:] - i*meanarray[i-1,:,:]
        data = xr.DataArray(data, dims=('time', 'lon', 'lat'), 
                      coords={'time': meanarray.time,
                              'lat':meanarray.lat,
                              'lon':meanarray.lon})
    else:
        data[0,:] = meanarray[0,:] # they both will be zero
        data[1,:] = meanarray[1,:] # this is the actual first data point... 

        # loop forward through the rest of the indices in array to "undo" the "mean from model start"
        for i in np.arange(2,np.shape(meanarray)[0]):
            data[i,:] = (i+1)*meanarray[i,:] - i*meanarray[i-1,:]
        
    return data


def precip(data, dt=900, returnPr=False, returnAcc=False):
    """ Takes an xarray as data input and dt is set to 960 seconds (15 mins)
            "time" must be first index/dimension, accepts only arrays with 3 or 4 dims
    
        returns precip rate if returnPr is True, returns accumulated precip
            if returnAcc is True as an xarray.
    """
    # initialize new xarray to return with same dims and coords as input data
    new_data = xr.DataArray(np.zeros(data.shape), dims=data.dims, coords=data.coords)
    if len(data.shape)==3:
        if returnPr and returnAcc:
            print("Choose either returnPr or ReturnAcc, cannot choose both")
        elif returnPr:
            # input is accumulated precipitation (eg. ICON)
            new_data[1:,:,:] = (data[1:,:,:].values-data[:-1,:,:].values)/dt
            new_data.attrs = {'long_name':'Precipitation Rate', 'units':'kg m-2 s-1'}
            print("Returned Precipitation Rate")
        elif returnAcc:
            for i in range(len(data.time)):
                if i == 0:
                    new_data[i] = np.zeros(data.shape[1:])
                new_data[i+1] = new_data[i] + data[i] * dt
            print("Returned Accumulated Precipitation")
        return new_data
    else: 
        print("data shape is not supported, must have 3 dimensions")
        return
        
######################### Plotting #######################

def dennisplot(stat, olr, alb, var=None, xbins=None, ybins=None, 
               levels=None, model="model", region="TWP", var_name="var_name",units="units", 
               cmap=plt.get_cmap("ocean_r"), ax=None, save=False, colorbar_on=True, fs=20):
    ''' Returns axis with contourf of olr and albedo.
    
    Parameters:
        - stat (str)   : - 'difference' returns contourf of the difference between the first minus the second in the tuple
                         - 'density' returns density plot of olr-alb joint histogram (pdf), or
                         - statistic for scipy.stats.binned_statistic_2d
        - olr (array)  : 1-D array of OLR values (from 85-310 W/m2), 
        - alb (array)  : 1-D array of Albedo values (from 0-1),
        - var (array)  : 1-D array (var is optional if stat=density or difference)
        - colorbar_on (bool)
                       : returns a tuple of ax, mappable_countour if False
                       
    Returns: 
        - ax (plt.axis): axis with plot 
        - cs (mappable): returned value from plt.contourf, if colorbar_on = False
        
    Note: Values for mean sw downward flux at toa from 
              http://www.atmos.albany.edu/facstaff/brose/classes/ATM623_Spring2015/Notes/Lectures/Lecture11%20--%20Insolation.html. 
    '''
    if xbins is None:
        xbins = np.linspace(70,320,26)
    if ybins is None:
        ybins = np.linspace(0,0.8,33)
    if levels is None:
        if stat=="difference":
            levels = np.linspace(-1,1,100)
        else:
            levels = np.arange(-3,-1.2,0.1)
    if stat=="difference":
        print("difference")
        olr0, olr1 = olr
        alb0, alb1 = alb
        olr0 = olr0[~np.isnan(alb0)]
        alb0 = alb0[~np.isnan(alb0)]
        alb0 = alb0[~np.isnan(olr0)]
        olr0 = olr0[~np.isnan(olr0)]
        olr1 = olr1[~np.isnan(alb1)]
        alb1 = alb1[~np.isnan(alb1)]
        alb1 = alb1[~np.isnan(olr1)]
        olr1 = olr1[~np.isnan(olr1)]
        hist0, xedges, yedges = np.histogram2d(olr0,alb0,bins=(xbins,ybins))
        nan_len = np.sum(~np.isnan(alb0))
        hist0 = hist0/nan_len
        print(nan_len)
        hist1, xedges, yedges = np.histogram2d(olr1,alb1,bins=(xbins,ybins))
        nan_len = np.sum(~np.isnan(alb1))
        hist1 = hist1/nan_len
        print(nan_len)
        binned_stat = hist0-hist1
    else:
        if (olr.shape!=alb.shape) and (var is not None):
            raise Exception("shapes don't match: olr %s, alb %s, %s %s."%(olr.shape, alb.shape, var_name, var.shape))
        elif var is not None:
            if (olr.shape!=var.shape) or (alb.shape!=var.shape):
                raise Exception("shapes don't match: olr %s, alb %s, %s %s."%(olr.shape, alb.shape, var_name, var.shape))
        elif (olr.shape!=alb.shape):
            raise Exception("shapes of alb and olr don't match: %s != %s"%(alb.shape, olr.shape))
        olr = olr[~np.isnan(alb)]
        if stat!='density':
            var = var[~np.isnan(alb)]
        alb = alb[~np.isnan(alb)]
        alb = alb[~np.isnan(olr)]
        if stat!='density':
            var = var[~np.isnan(olr)]
        olr = olr[~np.isnan(olr)]
        if stat!='density':
            alb = alb[~np.isnan(var)]
            olr = olr[~np.isnan(var)]
            var = var[~np.isnan(var)]
        if stat=='density':
            # check for nans
            binned_stat, xedges, yedges = np.histogram2d(olr,alb,bins=(xbins,ybins))
            nan_len = xr.DataArray(alb).count().values
            binned_stat = binned_stat/(nan_len)
            print(nan_len)
        else: 
            var = var[~np.isnan(olr)]
            binned_stat, xedges, yedges, _ = stats.binned_statistic_2d(olr, alb, var, 
                                                                          bins=(xbins,ybins), statistic=stat)
    xbins2, ybins2 = (xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2
    if ax is None:
        ax = plt.gca()
    if stat=="difference":
        csn = ax.contourf(xbins2, ybins2, binned_stat.T*100, levels, cmap=cmap, extend='both')
    else:
        csn = ax.contourf(xbins2, ybins2, np.log10(binned_stat.T), levels, cmap=cmap, extend='both')
        ax.contour(csn, colors='k', linestyles='solid', linewidths=1)
    if region=="NAU":
        ax.plot([80,317],[0.57,0.],label="Neutral CRE", color='black') # calculated in line_neutral_cre.ipynb
    elif region=="TWP":
        ax.plot([80,309],[0.55,0.],label="Neutral CRE", color='black') # calculated in line_neutral_cre.ipynb
    else:
        ax.plot([80,320],[0.75,0.2],label="Neutral CRE", color='black') # calculated in line_neutral_cre.ipynb
    ax.grid()
    ax.set_xticks([100,150,200,250,300])
    ax.set_ylim([0.05,0.8])
    ax.set_xlim([80,310])
    ax.set_xlabel('OLR (W m$^{-2}$)', size=fs)
    ax.set_ylabel('Albedo', size=fs)
    if var!=None:
        ax.set_title('{m} {v} {n}\n{l} Total Profiles'.format(m=model, v=var_name, n=region, l=len(olr)), size=fs)
    else:
        ax.set_title('{m} {n}\n{l} Total Profiles'.format(m=model, n=region, l=len(olr)), size=fs)
    ax.tick_params(axis='both',labelsize=fs)

    # plot the colorbar
    if colorbar_on:
        cb = plt.colorbar(csn, ax=ax, orientation='vertical')#, ticks=levtick)
        cb.ax.tick_params(labelsize=fs)
        if stat=="density":
            cb.set_label('log$_10$PDF)', fontsize=fs)
        elif stat=="difference":
            cb.set_label('pdf % difference', fontsize=fs)
        else:
            cb.set_label('log$_10$%s (%s)'%(stat.upper(), units), fontsize=fs)
    if save:
        plt.savefig('../plots/olr_alb/native_%s_%s_%s_%s.png'%(var_name.lower().replace(" ","_"), 
                                                               stat, model, region[:3]), bbox_inches="tight")
        print('    saved as ../plots/olr_alb/native_%s_%s_%s_%s.png'%(var_name.lower().replace(" ","_"), 
                                                                   stat, model, region[:3]))
    if colorbar_on:
        ret = ax
    else:
        ret = ax, csn
    return ret

def proxy_schematic(ax=None, arrow=True, fs=24):
    """Returns an axis with the plot showing the schematic of the
    cloud populations and idealized lifecycle (if arrow=True).
    
    Parameters:
        ax (plt.axis)   = axis for plotting
        arrow (boolean) = Draws an arrow from deep convection 
                to thin cirrus if true
    """
    c = ['C0', 'teal', 'skyblue', 'darkslategray']
    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    c3 = c[3]
    if ax==None:
        fig = plt.figure(figsize=(8,7.7))
        ax = fig.add_subplot(111, aspect='auto')
    dc = mpat.Ellipse((110,0.6),85,0.3, alpha=0.9, color=c0)
    ci = mpat.Ellipse((112,0.42), 180, 0.25,alpha=0.9, color=c1)
    cu = mpat.Ellipse((240,0.5),90,0.42,alpha=0.9, color=c2)
    cs = mpat.Ellipse((260,0.2),80,0.3, alpha=0.9, color=c3)
    dennisplot("density",np.array([0]),np.array([0]),ax=ax, colorbar_on=False, region="TWP")
    
    ax.annotate("    Deep\nConvection", xy=(82,0.57),xycoords='data', fontsize=fs-2, color='w')
    ax.annotate("   Anvils\n       &\nThick Cirrus", xy=(145,0.19),xycoords='data', fontsize=fs, color='w')
    ax.annotate("  Low\nClouds", xy=(220,0.45),xycoords='data', fontsize=fs, color='w')
    ax.annotate(" Thin\nCirrus", xy=(242,0.16),xycoords='data',fontsize=fs, color='w')

    t_start = ax.transData
    t = trans.Affine2D().rotate_deg(-30)
    t_end = t_start + t

    ci.set_transform(t_end)

    arc = mpat.FancyArrowPatch((110, 0.56), (280, 0.1), connectionstyle="arc3,rad=.2", 
                               arrowstyle = '->', alpha=0.9, lw=6, linestyle='solid', color='k')#(0.9,(2,2)))
    arc.set_arrowstyle('->', head_length=15, head_width=12)
    ax.add_patch(ci)
    ax.add_patch(dc)
    ax.add_patch(cu)
    ax.add_patch(cs)
    if arrow:
        ax.add_patch(arc)
    ax.set_axisbelow(True)
    ax.set_title("Schematic of Cloud Types\n", fontsize=fs)
    return ax

def diurnal_lt(time, data, model, region, bi_diurnal=False):
    """
    Returns local time and data array in local time for given
    model and region. 
    
    time = numpy array of hour in day
    data = numpy array of same shape as time
    model = 'nicam', 'sam', 'fv3', or 'icon'
    region = 'twp', 'nau', 'shl'
    """
    if bi_diurnal:
        d = 2
    else:
        d = 1
    # adjust to local time
    if region=="TWP":
        lt = 10
    elif region=="NAU":
        lt = 12
    elif region=="SHL":
        lt = 1
    # local times
    is_ltime = (np.arange(3,24.1*d,3)+lt)%(24*d)
    ltime = [i for i,j in sorted(zip(is_ltime,data))]
    ldata = [j for i,j in sorted(zip(is_ltime,data))]
    print("Returned time array and data in local time starting at midnight for %s %s"%(region, model))
    return ltime, ldata
    


def tdatetime(t_ind):
    """
    Convert string to datetime objects
    
    Input: timestep (0.25 for example) from DYAMOND data
    Output: datetime object for that input
    """
    t = t_ind
    mo = 8
    yr = 2016
    day = int(t//24)+1
    if day > 31:
        mo = 9
        day = day-31
#     print(mo, day)
    # hrs = t%24 #hours
    timeh = int(t%24) # hour hand
    if timeh<10:
        timeh="0"+str(timeh)
    timemin = str(t).split(".")[-1]
    if timemin == "25":
        timem = "15"
    elif timemin =="5":
        timem = "30"
    elif timemin =="75":
        timem = "45"
    else: 
        timem= "00"
    #  year, month, day, hour, minute
    tstr = datetime(int(yr), int(mo), int(day), int(timeh), int(timem))

    return tstr

def tstring(t_ind):
    """returns t string form for animation title"""
    t = t_ind
    mo_yr = " August 2016 "
    day = int(t//24)+1
    # hrs = t%24 #hours
    timeh = int(t%24) # hour hand
    if timeh<10:
        timeh="0"+str(timeh)
    timemin = str(t).split(".")[-1]
    if timemin == "25":
        timem = "15"
    elif timemin =="5":
        timem = "30"
    elif timemin =="75":
        timem = "45"
    else: 
        timem= "00"
    tstr = str(timeh)+":"+(timem)+" UTC "+str(day)+mo_yr
    return tstr

    