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


proj_strs = ['s5_deep', 's5_quiet1', 's5_quiet2', 's5_quiet3', 's5_quiet4']
proj_strs = proj_strs[:-1]
fig, axes = plt.subplots(len(proj_strs),1, sharex=True, sharey=True)
best_vs = []

N_fft = 2048
num_snapshots = 36
num_synth_els = 5
fact = 8
num_snapshots = int(num_snapshots / fact)
N_fft = fact*N_fft

subfolder = str(N_fft)
plot_descriptors = ['TSL', 'TSQ1', 'TSQ2', 'TSQ3', 'TSQ4']
cmap = plt.cm.get_cmap('viridis')

letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

for i in range(len(proj_strs)):
    proj_str = proj_strs[i]
    drp = load_mfp_results(proj_str,subfolder, num_snapshots, num_synth_els)
    dr, vel_arr, tilt_arr = drp.get_param_arrs(num_snapshots,num_synth_els, 13)
    vel_arr = vel_arr[1:,:]
    vel_arr = 10*np.log10(vel_arr)
    best_v = drp.vv[np.argmax(vel_arr, axis=0)]
    best_vs.append(best_v)
    print(np.min(vel_arr))
    db_max = 0
    db_min = -10
    levels = np.linspace(db_min, 0, 20)
    ax = axes[i].pcolormesh(dr.cov_t/60, drp.vv[1:], vel_arr[1:,:], vmin=db_min, vmax=0, cmap=cmap)

    t_v, v = restrict_v(t_v, v, dr)
    dopp_t, dopp_v = restrict_v(dopp_t, dopp_v, dr)
    #axes[i].scatter(t_v/60, v, s=16, marker='x', color='k')
    if i == 0:
        axes[i].scatter(dopp_t/60, dopp_v, marker='.', color='k', linewidth=1,alpha=1)
        axes[i].scatter(dopp_t/60, dopp_v, marker='.', color='w', linewidth=.5,alpha=1)

    axes[i].scatter(dr.cov_t/60, drp.vv[1:][np.argmax(vel_arr[1:,:], axis=0)], color='mediumblue', marker='*', alpha=1,linewidth=1)
    axes[i].set_ylabel(plot_descriptors[i])
    if i == 0:
        bstar = axes[i].scatter([], [], color='mediumblue', marker='*', alpha=1, linewidth=2)
        white_dot = axes[i].scatter([], [], marker='.', color='k', linewidth=2,alpha=1)
        black_dot = axes[i].scatter([], [], marker='.', color='w', linewidth=1,alpha=1)
        axes[i].legend([(white_dot, black_dot), bstar], ['$\hat{v}$ from Doppler','$\hat{v}_{L}$'], framealpha=1)
    else:
        axes[i].legend(['$\hat{v}_{Q' + str(i) + '}$'], framealpha=1)
    axes[i].text(dopp_t[0]/60 + .5, -2.55, letters[i], color='w',fontsize=15)
   
plt.tight_layout(pad=2)

cb = fig.colorbar(ax, ax=axes.ravel().tolist())
cb.set_label('   dB', rotation='horizontal')
fig.text(0.5, 0.01,  'Event Time (min)', ha='center')
fig.text(0.01, 0.5,  'Source range rate (m/s)', va='center', rotation='vertical')
fig.set_size_inches(8,4)

plt.savefig('/home/hunter/research/coherent_matched_field/paper/pics/' + fig_name, dpi=500, orientation='landscape')


plt.show()

#big_ax = fig.add_subplot(111, frameon=False)

plt.figure()
for i in range(5):
    plt.scatter(dr.cov_t/60, best_vs[i], color='k', s=10, alpha=.8)

plt.scatter(dopp_t/60, dopp_v, s=8, marker='+', color='r')



 
plt.show()
