#! usr/bin/env/python
import numpy as np
import utility.load01deg as load

REGION="NAU"
ds = load.get_cccm(REGION)
iwp = ds["iwp MODIS"]
alb = ds["Outgoing SW radiation at TOA"]/ds["Incoming SW radiation at TOA"]
olr = ds["Outgoing LW radiation at TOA"]
alb_array = alb.values.flatten()
olr_array = olr.values.flatten()
iwp_array = iwp.values.flatten()
olr_array = np.where(~np.isnan(alb_array),olr_array,np.nan)
iwp_array = np.where(~np.isnan(alb_array),iwp_array,np.nan)

print("------- all --------")
print("olr, alb, iwp shape", olr_array.shape, alb_array.shape, iwp_array.shape)
print("olr, alb, iwp mean", np.nanmean(olr_array), np.nanmean(alb_array), np.nanmean(iwp_array))

print("std dev olr: {}\n\tn_samples: {}".format(np.nanstd(olr_array), len(olr_array)))
print("std dev alb: {}\n\tn_samples: {}".format(np.nanstd(alb_array), len(alb_array)))
print("\nstd err olr: {}\nstd err alb: {}".format(np.nanstd(olr_array)/np.sqrt(len(olr_array)), 
        np.nanstd(alb_array)/np.sqrt(len(alb_array))))
print("\n---------- cat 1 ----------")
olr1 = np.where(iwp_array>=1000,olr_array,np.nan)
olr2 = np.where((iwp_array>=10)&(iwp_array<1000),olr_array,np.nan)
olr3 = np.where((iwp_array>=0.1)&(iwp_array<10),olr_array,np.nan)
alb1 = np.where(iwp_array>=1000,alb_array,np.nan)
alb2 = np.where((iwp_array>=10)&(iwp_array<1000),alb_array,np.nan)
alb3 = np.where((iwp_array>=0.1)&(iwp_array<10),alb_array,np.nan)

print("olr1 mean, std",np.nanmean(olr1), np.nanstd(olr1))
print("n_samples: ",len(olr1[~np.isnan(olr1)]))
print("std err olr1: ",np.nanstd(olr1)/np.sqrt(len(olr1[~np.isnan(olr1)])))
print("alb1 mean, std",np.nanmean(alb1), np.nanstd(alb1))
print("n_samples: ",len(alb1[~np.isnan(alb1)]))
print("std err alb1: ",np.nanstd(alb1)/np.sqrt(len(alb1[~np.isnan(alb1)])))

print("\n\n--------- cat 2 ---------")
print("olr2 mean, std",np.nanmean(olr2), np.nanstd(olr2))
print("n_samples: ",len(olr2[~np.isnan(olr2)]))
print("std err olr2: ",np.nanstd(olr2)/np.sqrt(len(olr2[~np.isnan(olr2)])))
print("alb2 mean, std",np.nanmean(alb2), np.nanstd(alb2))
print("n_samples: ",len(alb2[~np.isnan(alb2)]))
print("std err alb2: ",np.nanstd(alb2)/np.sqrt(len(alb2[~np.isnan(alb2)])))

print("\n\n--------- cat 3 ---------")
print("olr3 mean, std",np.nanmean(olr3), np.nanstd(olr3))
print("n_samples: ",len(olr3[~np.isnan(olr3)]))
print("std err olr3: ",np.nanstd(olr3)/np.sqrt(len(olr3[~np.isnan(olr3)])))
print("alb3 mean, std",np.nanmean(alb3), np.nanstd(alb3))
print("n_samples: ",len(alb3[~np.isnan(alb3)]))
print("std err alb3: ",np.nanstd(alb3)/np.sqrt(len(alb3[~np.isnan(alb3)])))

print("---- summary ----")
print("std err olr1: ",np.nanstd(olr1)/np.sqrt(len(olr1[~np.isnan(olr1)])))
print("std err olr2: ",np.nanstd(olr2)/np.sqrt(len(olr2[~np.isnan(olr2)])))
print("std err olr3: ",np.nanstd(olr3)/np.sqrt(len(olr3[~np.isnan(olr3)])))

print("std err alb1: ",np.nanstd(alb1)/np.sqrt(len(alb1[~np.isnan(alb1)])))
print("std err alb2: ",np.nanstd(alb2)/np.sqrt(len(alb2[~np.isnan(alb2)])))
print("std err alb3: ",np.nanstd(alb3)/np.sqrt(len(alb3[~np.isnan(alb3)])))

print("olr1 mean:", np.nanmean(olr1))
print("olr2 mean:", np.nanmean(olr2))
print("olr3 mean:", np.nanmean(olr3))

print("alb1 mean:", np.nanmean(alb1))
print("alb2 mean:", np.nanmean(alb2))
print("alb3 mean:", np.nanmean(alb3))