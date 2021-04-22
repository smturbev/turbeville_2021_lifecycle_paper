#!/usr/bin/env/python
""" fig03_schematic_joint_histogram.py
    author: Sami Turbeville
    date modified: 21 Dec 2020
    
    script to generate figure 3 in Turbeville et al 2021
"""
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
import matplotlib.transforms as trans
from utility import util, analysis_parameters as ap

c = ['C0', 'teal', 'skyblue', 'darkslategray', 'darkgoldenrod']
c0 = c[0]
c1 = c[1]
c2 = c[2]
c3 = c[3]
c4 = c[4]

fig = plt.figure(figsize=(8,7.7))
fs=24
ax = fig.add_subplot(111, aspect='auto')
util.dennisplot("density", np.zeros(0), np.zeros(0), colorbar_on=False, ax=ax)
dc = mpat.Ellipse((110,0.6),85,0.3, alpha=0.9, color=c0)
an = mpat.Ellipse((112,0.42), 180, 0.25,alpha=0.9, color=c1)
cu = mpat.Ellipse((240,0.5),90,0.42,alpha=0.9, color=c2)
ci = mpat.Ellipse((260,0.2),80,0.3, alpha=0.9, color=c3)
cs = mpat.Ellipse((270,0.1),33,0.1, alpha=0.6, ec=c4, fc=c4, fill=True, lw=3)
cs_outline = mpat.Ellipse((270,0.1),33,0.1, alpha=0.9, ec=c4, fc=None, fill=False, lw=3)


plt.annotate("    Deep\nConvection", xy=(82,0.57),xycoords='data', fontsize=fs-2, color='w')
plt.annotate("   Anvils\n       &\nThick Cirrus", xy=(145,0.19),xycoords='data', fontsize=fs, color='w')
plt.annotate("  Low\nClouds", xy=(220,0.45),xycoords='data', fontsize=fs, color='w')
plt.annotate(" Thin\nCirrus", xy=(242,0.18),xycoords='data',fontsize=fs, color='w')
plt.annotate("Clear\n  Sky", xy=(257,0.067),xycoords='data',fontsize=fs-5, color='w')

t_start = ax.transData
t = trans.Affine2D().rotate_deg(-30)
t_end = t_start + t

an.set_transform(t_end)

arc = mpat.FancyArrowPatch((110, 0.56), (257, 0.12), connectionstyle="arc3,rad=.21", 
                           arrowstyle = '->', alpha=0.9, lw=6, linestyle='solid', color='k')#(0.9,(2,2)))
arc.set_arrowstyle('->', head_length=15, head_width=12)
ax.add_patch(an)
ax.add_patch(dc)
ax.add_patch(cu)
ax.add_patch(ci)
ax.add_patch(arc)
ax.add_patch(cs)
ax.add_patch(cs_outline)
ax.set_axisbelow(True)
ax.set_title("Schematic of Cloud Types\n", fontsize=fs)
plt.savefig('../plots/fig03_schematic_joint_hist_arrow.png',dpi=200)
print('saved as ../plots/fig03_schematic_joint_hist_arrow.png')
plt.close()