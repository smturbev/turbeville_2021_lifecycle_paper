""" 
    reshape.py
    Author: Sami Turbeville
    Created: 3 April 2020
    
    Takes in concatenated form of ICON file and reshape to time, level, ncell
    Works as long as there are 8 timesteps per day, if your file is different, 
    update the variable ts_perday in reshape function. 
"""
import xarray as xr
import numpy as np
import pandas as pd
import time

#Example of file and var names 
icon_test_file = "/home/disk/eos15/smturbev/dyamond/ICON-2.5km/ICON-2.5km_allTime_tot_qi_dia__TWP.nc"
var = "QI_DIA"

def reshape(var, d, dim=3):
    """ Takes in file with variable name, var. Outputs a rearranged time
    """
    if dim==3:
        ts_perday = 8
        ds = d[var] #41,616,19442
        nd, nh, ncell = ds.shape
        lev = ds.height[::8].values
        cell = ds.cell.values
        units = ds.attrs
        dummy_time = pd.date_range('2016-08-01', periods=328, freq="3H")
        print("Starting to reshape variable, %s..."%(var))
        s_time = time.time()
        print("    Original shape was: %s"%str(ds.shape))
        ds_r1 = np.reshape(ds.values, (nd, ts_perday, nh//ts_perday, ncell), 'F')
        ds_r2 = np.reshape(ds_r1, (nd*ts_perday, nh//ts_perday, ncell))

        da = xr.DataArray(ds_r2, dims=['t', 'lev', 'cell'], 
                          coords={'t':dummy_time,'lev':lev,'cell':cell}, 
                          attrs=units)
        e_time = time.time()
    elif dim==2:
        ts_perday = 96
        ds = d[var] #41,616,19442
        nd, nh, ncell = ds.shape
        cell = ds.cell.values
        units = ds.attrs
        dummy_time = pd.date_range('2016-08-01', periods=(nd*nh), freq="0.25H")
        print("Starting to reshape variable, %s..."%(var))
        s_time = time.time()
        print("    Original shape was: %s"%str(ds.shape))
        ds_r1 = np.reshape(ds.values, (nd*nh, ncell))
        da = xr.DataArray(ds_r1, dims=['t', 'cell'], 
                          coords={'t':dummy_time,'cell':cell}, 
                          attrs=units)
        e_time = time.time()
    print("Successfully reshaped variable!\n    New shape is %s.\nIt took %f minutes."%(str(da.shape), (e_time-s_time)/60))
    return da

def assign_new(old_ds, new_da, file, save=True):
    s_time = time.time()
    dn = old_ds.assign(NEW=new_da)
    print("New data array to save: \n\n", dn, "\n\n")
    if save:
        print("Saving new DataArray as new Variable to old Dataset in file %s"%(file))
        dn.to_netcdf(file)
        print("\n\nFile saved successfully!")
    e_time = time.time()
    print("It took %s minutes"%((e_time-s_time)/60))
    return dn

