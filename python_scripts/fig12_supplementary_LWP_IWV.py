# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utility import analysis_parameters as ap

c = ap.COLORS

# %%
# load csv
lwp_twp = pd.read_csv("../tables/twp_lwp_cat_med_kgm-2.csv", index_col=0)
lwp_shl = pd.read_csv("../tables/shl_lwp_cat_med_kgm-2.csv", index_col=0)
lwp_nau = pd.read_csv("../tables/nau_lwp_cat_med_kgm-2.csv", index_col=0)

fwp_twp = pd.read_csv("../tables/twp_fwp_cat_med_kgm-2.csv", index_col=0)
fwp_shl = pd.read_csv("../tables/shl_fwp_cat_med_kgm-2.csv", index_col=0)
fwp_nau = pd.read_csv("../tables/nau_fwp_cat_med_kgm-2.csv", index_col=0)

iwv_twp = pd.read_csv("../tables/twp_iwv_cat_med_kgm-2.csv", index_col=0)
iwv_shl = pd.read_csv("../tables/shl_iwv_cat_med_kgm-2.csv", index_col=0)
iwv_nau = pd.read_csv("../tables/nau_iwv_cat_med_kgm-2.csv", index_col=0)


# %%
df1 = pd.DataFrame(1000*(np.array([fwp_twp.T.CAT1, lwp_twp.T.CAT1, 
            iwv_twp.T.CAT1])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df2 = pd.DataFrame(1000*(np.array([fwp_twp.T.CAT2, lwp_twp.T.CAT2, 
            iwv_twp.T.CAT2])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df3 = pd.DataFrame(1000*(np.array([fwp_twp.T.CAT3, lwp_twp.T.CAT3, 
            iwv_twp.T.CAT3])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df4 = pd.DataFrame(1000*(np.array([fwp_twp.T.CS, lwp_twp.T.CS, 
            iwv_twp.T.CS])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df1.where(~np.isnan(df1), other=0, inplace=True)
df2.where(~np.isnan(df1), other=0, inplace=True)
df3.where(~np.isnan(df1), other=0, inplace=True)
df4.where(~np.isnan(df1), other=0, inplace=True)

df1s = pd.DataFrame(1000*(np.array([fwp_shl.T.CAT1, lwp_shl.T.CAT1, 
            iwv_shl.T.CAT1])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df2s = pd.DataFrame(1000*(np.array([fwp_shl.T.CAT2, lwp_shl.T.CAT2, 
            iwv_shl.T.CAT2])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df3s = pd.DataFrame(1000*(np.array([fwp_shl.T.CAT3, lwp_shl.T.CAT3, 
            iwv_shl.T.CAT3])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df4s = pd.DataFrame(1000*(np.array([fwp_shl.T.CS, lwp_shl.T.CS, 
            iwv_shl.T.CS])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df1s.where(~np.isnan(df1s), other=0, inplace=True)
df2s.where(~np.isnan(df1s), other=0, inplace=True)
df3s.where(~np.isnan(df1s), other=0, inplace=True)
df4s.where(~np.isnan(df1s), other=0, inplace=True)

df1n = pd.DataFrame(1000*(np.array([fwp_nau.T.CAT1, lwp_nau.T.CAT1, 
            iwv_nau.T.CAT1])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df2n = pd.DataFrame(1000*(np.array([fwp_nau.T.CAT2, lwp_nau.T.CAT2, 
            iwv_nau.T.CAT2])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df3n = pd.DataFrame(1000*(np.array([fwp_nau.T.CAT3, lwp_nau.T.CAT3, 
            iwv_nau.T.CAT3])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df4n = pd.DataFrame(1000*(np.array([fwp_nau.T.CS, lwp_nau.T.CS, 
            iwv_nau.T.CS])), index=["FWP","LWP","IWV"], 
            columns=["CCCM","NICAM","FV3","ICON","SAM"]).T
df1n.where(~np.isnan(df1n), other=0, inplace=True)
df2n.where(~np.isnan(df1n), other=0, inplace=True)
df3n.where(~np.isnan(df1n), other=0, inplace=True)
df4n.where(~np.isnan(df1n), other=0, inplace=True)
x = list(df1.T.keys())
x
# %%
fig = plt.figure(figsize=(15,9))
colors = ["cornflowerblue", "slateblue"]
regions = ["SHL","TWP","NAU"]
ylims = [3000, 170, 11, 6]
gs = fig.add_gridspec(3,4)

df = ([[df1s, df2s, df3s, df4s],
      [df1, df2, df3, df4],
      [df1n, df2n, df3n, df4n]])

dfd = np.zeros((3,4))
for i in range(3):
    for j in range(4):
        print(i,j)
        ax = fig.add_subplot(gs[i,j])
        dfd = df[i][j].drop(["IWV"], axis=1)
        dfd.plot.bar(ax=ax, color=colors)
        if i==0:
            ax.set_title("CAT {}".format(str(j+1)))
        if j==0:
            ax.set_ylabel(regions[i])
        ax.set_ylim([0,ylims[j]])
plt.subplots_adjust(hspace=0)
plt.savefig("../plots/cat_iwp_lwp_bar.png")
plt.show()

# %%
colors = ["tab:olive", "tab:blue", "tab:cyan"]

fig = plt.figure(figsize=(15,3))
gs = fig.add_gridspec(1,4)
cat1 = fig.add_subplot(gs[0])
cat2 = fig.add_subplot(gs[1])
cat3 = fig.add_subplot(gs[2])
cat4 = fig.add_subplot(gs[3])

df1sd = df1s.drop(["FWP", "LWP"], axis=1)
df1td = df1.drop(["FWP", "LWP"], axis=1)
df1nd = df1n.drop(["FWP", "LWP"], axis=1)
df2sd = df1s.drop(["FWP", "LWP"], axis=1)
df2td = df1.drop(["FWP", "LWP"], axis=1)
df2nd = df1n.drop(["FWP", "LWP"], axis=1)
df3sd = df1s.drop(["FWP", "LWP"], axis=1)
df3td = df1.drop(["FWP", "LWP"], axis=1)
df3nd = df1n.drop(["FWP", "LWP"], axis=1)
df4sd = df1s.drop(["FWP", "LWP"], axis=1)
df4td = df1.drop(["FWP", "LWP"], axis=1)
df4nd = df1n.drop(["FWP", "LWP"], axis=1)

df1_plot = pd.DataFrame(np.array([df1sd.values[:,0], df1td.values[:,0], df1nd.values[:,0]]).T/1000, 
    index=df1sd.index.values, columns=["SHL","TWP","NAU"])
df2_plot = pd.DataFrame(np.array([df2sd.values[:,0], df2td.values[:,0], df2nd.values[:,0]]).T/1000, 
    index=df1sd.index.values, columns=["SHL","TWP","NAU"])
df3_plot = pd.DataFrame(np.array([df3sd.values[:,0], df3td.values[:,0], df3nd.values[:,0]]).T/1000, 
    index=df1sd.index.values, columns=["SHL","TWP","NAU"])
df4_plot = pd.DataFrame(np.array([df4sd.values[:,0], df4td.values[:,0], df4nd.values[:,0]]).T/1000, 
    index=df1sd.index.values, columns=["SHL","TWP","NAU"])
df1_plot.plot.bar(ax=cat1, color=colors)
df2_plot.plot.bar(ax=cat2, color=colors)
df3_plot.plot.bar(ax=cat3, color=colors)
df4_plot.plot.bar(ax=cat4, color=colors)

cat1.set_title("CAT 1")
cat2.set_title("CAT 2")
cat3.set_title("CAT 3")
cat4.set_title("CS")
cat1.set_ylabel("IWV (kg/m2)")

plt.savefig("../plots/cat_iwv_stn.png", bbox_inches="tight", pad_inches=0.5)
plt.show()

# %%
ax = pd.plotting.scatter_matrix(df1, c=list(c.values())[:5], figsize=(8,8), 
                marker="o", hist_kwds={"bins":20}, s=100, alpha=0.8, 
                cmap="tab10", range_padding=0.5, grid=True)
print(ax)
plt.savefig("../plots/pair_plot_cat_med.png") 
plt.show()
# %%
# plot by category
a = 0.2
c_twp = "k"
c_shl = "m"
c_nau = "b"

fig, ax = plt.subplots(2, 2, sharex=True, constrained_layout=True, 
                       figsize=(8,7))

# CAT 1
## twp
ax[0,0].scatter(x, df1.FWP, marker="o", s=100, c=c_twp,
              alpha=a, edgecolors="k")
ax[0,0].scatter(x, df1.LWP, marker="s", s=100, c=c_twp,
              alpha=a, edgecolors="k")
axt = ax[0,0].twinx()
axt.set_title("CAT1")
axt.scatter(x, df1.IWV, marker=">", s=100, c=c_twp,
              alpha=a, edgecolors="k")
## shl
ax[0,0].scatter(x, df1s.FWP, marker="o", s=100, c=c_shl,
              alpha=a, edgecolors="k")
ax[0,0].scatter(x, df1s.LWP, marker="s", s=100, c=c_shl,
              alpha=a, edgecolors="k")
axt.scatter(x, df1s.IWV, marker=">", s=100, c=c_shl,
              alpha=a, edgecolors="k")
## nau
ax[0,0].scatter(x, df1n.FWP, marker="o", s=100, c=c_nau,
              alpha=a, edgecolors="k")
ax[0,0].scatter(x, df1n.LWP, marker="s", s=100, c=c_nau,
              alpha=a, edgecolors="k")
axt.scatter(x, df1n.IWV, marker=">", s=100, c=c_nau,
              alpha=a, edgecolors="k")
# CAT 2
## twp
ax[0,1].scatter(x, df2.FWP, marker="o", s=100, c=c_twp,
              alpha=a, edgecolors="k")
ax[0,1].scatter(x, df2.LWP, marker="s", s=100, c=c_twp,
              alpha=a, edgecolors="k")
axt = ax[0,1].twinx()
axt.set_title("CAT2")
axt.scatter(x, df2.IWV, marker=">", s=100, c=c_twp,
              alpha=a, edgecolors="k")
axt.set_ylabel("IWV (g/m$^2$)")
## shl
ax[0,1].scatter(x, df2s.FWP, marker="o", s=100, c=c_shl,
              alpha=a, edgecolors="k")
ax[0,1].scatter(x, df2s.LWP, marker="s", s=100, c=c_shl,
              alpha=a, edgecolors="k")
axt.scatter(x, df2s.IWV, marker=">", s=100, c=c_shl,
              alpha=a, edgecolors="k")
## nau
ax[0,1].scatter(x, df2n.FWP, marker="o", s=100, c=c_nau,
              alpha=a, edgecolors="k")
ax[0,1].scatter(x, df2n.LWP, marker="s", s=100, c=c_nau,
              alpha=a, edgecolors="k")
axt.scatter(x, df2n.IWV, marker=">", s=100, c=c_nau,
              alpha=a, edgecolors="k")
# CAT 3
## twp
ax[1,0].scatter(x, df3.FWP, marker="o", s=100, c=c_twp,
              alpha=a, edgecolors="k")
ax[1,0].scatter(x, df3.LWP, marker="s", s=100, c=c_twp,
              alpha=a, edgecolors="k")
axt = ax[1,0].twinx()
axt.set_title("CAT3")
axt.scatter(x, df3s.IWV, marker=">", s=100, c=c_twp,
              alpha=a, edgecolors="k")
## shl
ax[1,0].scatter(x, df3s.FWP, marker="o", s=100, c=c_shl,
              alpha=a, edgecolors="k")
ax[1,0].scatter(x, df3s.LWP, marker="s", s=100, c=c_shl,
              alpha=a, edgecolors="k")
axt.scatter(x, df3.IWV, marker=">", s=100, c=c_shl,
              alpha=a, edgecolors="k")
## nau
ax[1,0].scatter(x, df3n.FWP, marker="o", s=100, c=c_nau,
              alpha=a, edgecolors="k")
ax[1,0].scatter(x, df3n.LWP, marker="s", s=100, c=c_nau,
              alpha=a, edgecolors="k")
axt.scatter(x, df3n.IWV, marker=">", s=100, c=c_nau,
              alpha=a, edgecolors="k")
# CAT 4 (CS)
## twp
ax[1,1].scatter(x, df4.FWP, marker="o", s=100, c=c_twp,
              alpha=a, edgecolors="k")
ax[1,1].scatter(x, df4.LWP, marker="s", s=100, c=c_twp,
              alpha=a, edgecolors="k")
axt = ax[1,1].twinx()
axt.set_title("CAT4")
axt.scatter(x, df4.IWV, marker=">", s=100, c=c_twp,
              alpha=a, edgecolors="k")
axt.set_ylabel("IWV (g/m$^2$)")
## shl
ax[1,1].scatter(x, df4s.FWP, marker="o", s=100, c=c_shl,
              alpha=a, edgecolors="k")
ax[1,1].scatter(x, df4s.LWP, marker="s", s=100, c=c_shl,
              alpha=a, edgecolors="k")
axt.scatter(x, df4.IWV, marker=">", s=100, c=c_shl,
              alpha=a, edgecolors="k")
## nau
ax[1,1].scatter(x, df4n.FWP, marker="o", s=100, c=c_nau,
              alpha=a, edgecolors="k")
ax[1,1].scatter(x, df4n.LWP, marker="s", s=100, c=c_nau,
              alpha=a, edgecolors="k")
axt.scatter(x, df4n.IWV, marker=">", s=100, c=c_nau,
              alpha=a, edgecolors="k")

# fig
ax[0,0].set_ylabel("WP (g/m$^2$)")
ax[1,0].set_ylabel("WP (g/m$^2$)")

plt.savefig("../plots/cat_med_fwp_lwp_iwv_stn.png", dpi=150)
plt.show()
# %%
