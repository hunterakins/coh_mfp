import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.data_test import get_env
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

"""
Description:
Show the simulation environment used

Date:
3/10/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""



"""
axes[0] is the ssp and vla 
"""


fig_name = 'env.png'

proj_str = 's5_deep'
freq= 49

env = get_env(proj_str, freq)
env.zr = np.insert(env.zr, 21, env.zr[0] + 21*1.875)
env.zr = env.zr[1::3]

fig, axes = plt.subplots(2,1)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
#plt.ylabel('Depth (m)')
fig.text(0.01, 0.6, 'Depth (m)', va='center', rotation='vertical')

#plt.subplots_adjust(hspace=0.4)
fig.tight_layout()
ax1 = axes[0]
ax1_copy = ax1.twiny()
zr = env.zr
zs = env.zs
#x, y = get_sea_surface(env.cw)
#ax1.plot(x, y, color='b')
ax1.set_xlabel('Sound speed (m/s)')
ax1.invert_yaxis()
#ax1_copy.scatter(0, zr[0], color='k', alpha=1, s=10, label='SSP')
pt = ax1_copy.scatter([.75]*zr.size, zr, color='k', s=10, label='Receive array')
pt = ax1_copy.scatter([.8]*zr.size, zr, color='k', s=10, label='Receive array')
pt = ax1_copy.scatter([.85]*zr.size, zr, color='k', s=10, label='Receive array')
pt = ax1_copy.scatter([.9]*zr.size, zr, color='k', s=10, label='Receive array')
ax1_copy.set_xticks([])
ax1_copy.set_xlim([0, 1])
ax1_copy.set_yticks([0, 50, 100, 150, 200])
ax1_copy.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax1_copy.text(0.77, 105, 'VLA')
line, = ax1.plot(env.cw, env.z_ss, color='k', label='SSP')
ax1.set_ylim([np.max(zr),0])
#ax1.legend([line, pt], ['SSP', 'VLA'])

xc, yc = [0, 0, 1, 1], [0.66, 1, 1, 0.66]

ax2 = axes[1]
ax2.set_xticks([])
ax2.set_yticks([1, 0.66, 0.33])
ax2.set_ylim([0,1])
ax2.set_xlim([0,1])
ax2.yaxis.set_major_formatter(FormatStrFormatter(''))
ax2.fill(xc, yc, 'k', alpha=0.1, lw=0)


xc, yc = [0, 0, 1, 1], [.33, 0.66, .66, 0.33]
ax2.fill(xc, yc, color='k', alpha=.3, lw=0)

xc, yc = [0, 0, 1, 1], [0, 0.33, .33, 0.0]

ax2.fill(xc, yc, color='k', alpha=0.5, lw=0)


ax2.text(0.1, .66+.33/2, 'Density = 1.8 g/cm$^{3}$\nAttenuation = 0.3 dB/kmHz')
ax2.text(0.1, .66-.33/2, 'Density = 2.1 g/cm$^{3}$\nAttenuation = 0.09 dB/kmHz')
ax2.text(0.1, .33/2, 'Density = 2.66 g/cm$^{3}$\nAttenuation = 0.02 dB/kmHz')

dy = 0.06
dy1 = 0.04
x_pos = 0.7
ax2.text(x_pos, 1-dy, 'c$_{top}$=1572 m/s')
ax2.text(x_pos, .66+dy1, 'c$_{bot}$=1593 m/s')
ax2.text(x_pos, .66-dy, 'c$_{top}$=1881 m/s')
ax2.text(x_pos, .33+dy1, 'c$_{bot}$=3245.8 m/s')
ax2.text(x_pos, .33/2, 'c$_{hs}$ = 5200 m/s')

ax2.text(-0.083, 1, '216.5')
ax2.text(-0.063, .66, '240')
ax2.text(-0.076, .33, '1040')



fig.set_size_inches(6,6)
#plt.savefig('/home/hunter/research/coherent_matched_field/paper/pics/' + fig_name, dpi=500)


plt.show()
