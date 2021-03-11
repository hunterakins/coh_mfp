import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from coh_mfp.data_test import DRPRuns, DataRun

rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

"""
Description:
Figure for looking at velocity estimates from bartlett surfaces

Date:
2/5/2020

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

from vel_estimation import load_vel_arr, load_mfp_results

def load_ship_range_func():
    root_folder = '/home/hunter/research/code/swellex/ship/npys/'
    x = np.load(root_folder + 's5_r.npy')
    t = np.load(root_folder + 's5_r_tgrid.npy')
    t = t * 60
    dt = t[1:] - t[:-1]
    dr = x[1:] - x[:-1]
    dr = dr*1000
    v = dr/ dt
    t_v = (t[1:] + t[:-1]) /2
    return t_v, v

def load_ship_v_doppler():
    root_folder = '/home/hunter/research/code/swellex/ship/'
    v = np.load(root_folder + 's5_deep_v.npy')
    t = v[0,:].reshape(v.shape[1])
    v = v[1,:].reshape(v.shape[1])
    return t, v

dopp_t, dopp_v = load_ship_v_doppler()

t_v, v = load_ship_range_func()
corr_factor = 192.5 / 216.5

fig_name = 'vel_amb'


def restrict_v(t_v, v, dr):
    cov_t = dr.cov_t
    good_inds = [i for i in range(t_v.size) if t_v[i] > np.min(cov_t) and t_v[i] < np.max(cov_t)]
    return t_v[good_inds], v[good_inds]

fig, axes = plt.subplots(3,1, sharex=True, sharey=True)

proj_strs = ['s5_deep', 's5_quiet1', 's5_quiet2']
best_vs = []

num_snapshots = 4

plot_descriptors = ['Loud', 'Quiet 1', 'Quiet 2']
cmap = plt.cm.get_cmap('viridis')
for i in range(3):
    proj_str = proj_strs[i]
    drp = load_mfp_results(proj_str, num_snapshots)
    dr, vel_arr, tilt_arr = drp.get_param_arrs(15,10, 13)
    vel_arr = 10*np.log10(vel_arr)
    best_v = drp.vv[np.argmax(vel_arr, axis=0)]
    best_vs.append(best_v)
    print(np.min(vel_arr))
    db_max = 0
    db_min = -10
    levels = np.linspace(db_min, 0, 20)
    ax = axes[i].pcolormesh(dr.cov_t/60, drp.vv, vel_arr, vmin=db_min, vmax=0, cmap=cmap)

    t_v, v = restrict_v(t_v, v, dr)
    dopp_t, dopp_v = restrict_v(dopp_t, dopp_v, dr)
    axes[i].scatter(t_v/60, v, s=16, marker='x', color='k')
    axes[i].scatter(dopp_t/60, dopp_v, s=8, marker='+', color='k', alpha=0.6)
    axes[i].set_ylabel(plot_descriptors[i])
    axes[i].legend(['GPS', 'Doppler'], framealpha=1)
   
plt.tight_layout(pad=2)

cb = fig.colorbar(ax, ax=axes.ravel().tolist())
cb.set_label('   dB', rotation='horizontal')
fig.text(0.5, 0.01,  'Event Time (min)', ha='center')
fig.text(0.01, 0.5,  'Source range rate (m/s)', va='center', rotation='vertical')



plt.savefig('/home/hunter/research/coherent_matched_field/pics/' + fig_name, dpi=500, orientation='landscape')


plt.show()

#big_ax = fig.add_subplot(111, frameon=False)

plt.figure()
for i in range(3):
    plt.scatter(dr.cov_t/60, best_vs[i], color='k', s=10, alpha=.8)

plt.scatter(dopp_t/60, dopp_v, s=8, marker='+', color='r')



 
plt.show()
