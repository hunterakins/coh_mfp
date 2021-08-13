import numpy as np
from matplotlib import pyplot as plt
from comparison_plot import get_tracking_spo
from wnc_test import get_cov_time, check_v_arr
from proc_out import SwellProcObj, load_spo
from vel_estimation import load_vel_arr
from scipy.interpolate import interp1d
from copy import deepcopy
from swellex.audio import make_snapshots as ms

vv = ms.get_vv()

"""
Description:
Make a figure that shows the range of maximum correlation
but compares num synth els used

Date:
3/8/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

def get_hit_prcnt(cov_times, range_vals):
    tgrid = np.load('/home/hunter/research/code/swellex/ship/npys/s5_r_tgrid.npy')
    r_km = np.load('/home/hunter/research/code/swellex/ship/npys/s5_r.npy')
    r = r_km*1000
    r_interp = interp1d(60*tgrid[:,0], r[:,0])
    rvals = r_interp(cov_times)
    #plt.figure()
    #plt.plot(rvals)
    #plt.plot(range_vals)
    #plt.show()

    diffs = abs(rvals - range_vals)
    sq_diffs = np.square(diffs)
    count = 0
    for x in diffs:
        if x < 200:
            count += 1
    prcnt = count / cov_times.size
    print('hit percent:' + str(prcnt * 100) + '\%')
    rmse = np.sqrt(np.mean(sq_diffs))
    print('rmse', rmse)
    return sq_diffs

if __name__ == '__main__':

    proj_str = 's5_quiet1'
    #proj_str = 's5_quiet3'

    N_fft = 2048
    num_snapshots = 36
    fact = 8

    N_fft = fact*N_fft
    num_snapshots = int(num_snapshots / fact)

    subfolder = str(N_fft)  + '_sec1'

    tilt_angle = -.5
    num_freqs = 13
    wn_gain = -0.5

    fig_name = proj_str + '_range_est_color.png'
        
    fig, axes = plt.subplots(2,2, sharex='col', sharey='row')
    db_min = -15
    nse_list = [5, 8]
    for i in range(len(nse_list)):
        num_synth_els = nse_list[i]
        cov_times=get_cov_time(proj_str, subfolder, num_snapshots, num_synth_els)
        v_arr = load_vel_arr(proj_str, subfolder, num_snapshots,num_synth_els)
        v_arr = check_v_arr(v_arr, cov_times)
        v_interp = interp1d(v_arr[0,:], v_arr[1,:])

        root_folder ='pickles/'

        cmap = plt.cm.get_cmap('viridis')
        wnc = True
        #wnc = False

        for cov_index in range(cov_times.size):
            cov_time = cov_times[cov_index]
            v_source = v_interp(cov_time)

            spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle, num_synth_els, num_freqs, v_source, cov_time, wn_gain)
            spo.get_bathy_corr()
            spo.wnc_out -= np.max(spo.wnc_out)
            spo.bart_out -= np.max(spo.bart_out)


            if cov_index == 0:
                range_vals = np.zeros((cov_times.size, spo.corr_grid_r.size))
                depth_vals = np.zeros((cov_times.size, spo.corr_grid_z.size))

            if wnc == True:
                range_vals[cov_index, :] = np.max(spo.wnc_out, axis=0)
                depth_vals[cov_index, :] = np.max(spo.wnc_out, axis=1)
            else:
                range_vals[cov_index, :] = np.max(spo.bart_out, axis=0)
                depth_vals[cov_index, :] = np.max(spo.bart_out, axis=1)

        cf = axes[1,i].pcolormesh(cov_times/60, spo.corr_grid_r,range_vals.T, vmin=db_min, vmax=0, cmap=cmap, shading='auto')
        cf = axes[0,i].pcolormesh(cov_times/60, spo.corr_grid_z,depth_vals.T, vmin=db_min, vmax=0, cmap=cmap, shading='auto')
    cb = fig.colorbar(cf, ax=axes.ravel().tolist())
    cb.set_label('dB', rotation='horizontal')

    cols = [str(x) + " synth els" for x in nse_list]

    axes[0,0].invert_yaxis()
    for i in range(len(nse_list)):
        axes[0,i].set_title(cols[i])


    fig.text(0.5, 0.02, 'Event Time (m)', ha='center')
    axes[0,0].set_ylabel('Depth (m)')
    axes[1,0].set_ylabel('Range (m)')

    letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    i = 0
    for ax in axes.ravel().tolist():
        if i < len(nse_list):
            ax.text(40.5, 175, letters[i], color='w', fontsize=15)
        else:
            ax.text(40.5, 1000, letters[i], color='w', fontsize=15)
        i += 1
    
    fig.set_size_inches(8, 4)
    #plt.savefig('/home/hunter/research/coherent_matched_field/paper/pics/' + fig_name, dpi=500, orientation='landscape')
    plt.show()
        
