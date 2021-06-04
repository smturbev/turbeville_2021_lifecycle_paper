# %%
import matplotlib.pyplot as plt

# %%
fig = plt.figure(figsize=(15,11))
fs=12
gs = fig.add_gridspec(2,2)

ax0 = fig.add_subplot(gs[1,1])
ax1 = fig.add_subplot(gs[0,1])
axz = fig.add_subplot(gs[:,0])

axz.plot([0,1,2,3],[0,1,2,3], label="example")
axz.plot([2,2,2,2],[0,1,2,3], label="NICAM")
axz.plot([3,2,1,2],[0,1,2,3], label="FV3")

h, l = axz.get_legend_handles_labels()
fig.legend(h,l, loc="upper center", bbox_to_anchor=(0.4, 0.05),
           ncol=3)
# plt.tight_layout()
# plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.2)
plt.show()
# %%
