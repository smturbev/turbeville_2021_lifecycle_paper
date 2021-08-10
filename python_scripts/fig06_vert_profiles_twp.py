#!/usr/bin/env python
""" fig06_vert_profiles_twp.py
    Author: Sami Turbeville    
    Plots time- and area- cloud fraction over 38-day
    period in DYAMOND, compared to DARDAR and CCCM.
"""

import numpy as np
import xarray as xr
from utility import util, load01deg
import utility.analysis_parameters as ap
import statistics as stats
import matplotlib.pyplot as plt
from utility import vert_cld_frac

c = ap.COLORS
REGION = "TWP"

# scatter plot of mean vs std
fig, axz = plt.subplots(1, 1, figsize=(6,7))
fs=14

axz = vert_cld_frac.plot_vert_cld_frac(REGION, ax=axz)
axz.set_title("")
axz.legend(fontsize=fs)

plt.savefig("../plots/fig06_vertical_%s.png"%REGION.lower(),dpi=150,
            bbox_inches="tight", pad_inches=2)
print("... saved to ../plots/fig06_vertical_%s.png"%REGION.lower())
plt.close()
