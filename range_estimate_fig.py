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
Compares the various methods

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

    #proj_str = 's5_quiet2'
    #proj_str = 's5_deep'
    proj_str = 's5_quiet3'

    N_fft = 2048
    num_snapshots = 36

    fact = 8

    long_snap = False
    synth_N_fft = fact*N_fft
    synth_num_snapshots = int(num_snapshots / fact)
    fig_name = proj_str + '_range_est_color.png'
        
    fig, axes = plt.subplots(2,2, sharex='col', sharey='row')
    db_min = -15
    for sub_count in range(1):
        subfolder = str(N_fft) 
        synth_subfolder = str(synth_N_fft)
        tilt_angle = -1 
        if sub_count == 1:
            subfolder += '_sec2'
            synth_subfolder += '_sec2'
            tilt_angle = -.5
        elif sub_count == 2:
            subfolder = str(N_fft) + '_sec1'
            synth_subfolder = str(synth_N_fft) + '_sec1'
            num_synth_els = 8
            tilt_angle = -.5
        num_synth_els = 5
        num_freqs = 13
        synth_wn_gain = -0.5
        wn_gain = -2.0
        cov_times=get_cov_time(proj_str, subfolder, num_snapshots, num_synth_els)
        v_arr = load_vel_arr(proj_str, subfolder, num_snapshots,5)
        v_arr = check_v_arr(v_arr, cov_times)
        v_interp = interp1d(v_arr[0,:], v_arr[1,:])

    #fig_err = plt.figure()

        synth_cov_times = get_cov_time(proj_str, synth_subfolder, synth_num_snapshots, num_synth_els)
        synth_v_arr = load_vel_arr(proj_str, synth_subfolder, synth_num_snapshots,num_synth_els)
        synth_v_arr = check_v_arr(synth_v_arr, synth_cov_times)
        synth_v_interp = interp1d(synth_v_arr[0,:], synth_v_arr[1,:])

        root_folder ='pickles/'

        cmap = plt.cm.get_cmap('viridis')
        wnc = True
        #wnc = False


        for cov_index in range(cov_times.size):
            cov_time = cov_times[cov_index]
            v_source = v_interp(cov_time)

            spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle, 1, num_freqs, v_source, cov_time, wn_gain)
            spo.get_bathy_corr()
            spo.wnc_out -= np.max(spo.wnc_out)
            spo.bart_out -= np.max(spo.bart_out)


            if cov_index == 0:
                simple_range_vals = np.zeros((cov_times.size, spo.corr_grid_r.size))
                simple_depth_vals = np.zeros((cov_times.size, spo.corr_grid_z.size))

            if wnc == True:
                simple_range_vals[cov_index, :] = np.max(spo.wnc_out, axis=0)
                simple_depth_vals[cov_index, :] = np.max(spo.wnc_out, axis=1)
            else:
                simple_range_vals[cov_index, :] = np.max(spo.bart_out, axis=0)
                simple_depth_vals[cov_index, :] = np.max(spo.bart_out, axis=1)



        """ NOW DO SYNTHETIC """
        for cov_index in range(synth_cov_times.size):
            cov_time = synth_cov_times[cov_index]
            v_source = synth_v_interp(cov_time)
            v_source = vv[np.argmin([abs(v_source -x) for x in vv])]
            synth_spo = load_spo(root_folder, proj_str, synth_subfolder, synth_num_snapshots, tilt_angle, num_synth_els, num_freqs, v_source, cov_time, synth_wn_gain)
            synth_spo.get_bathy_corr()
            synth_spo.wnc_out -= np.max(synth_spo.wnc_out)
            synth_spo.bart_out -= np.max(synth_spo.bart_out)

            if cov_index == 0:
                synth_range_vals = np.zeros((synth_cov_times.size, synth_spo.corr_grid_r.size))
                synth_depth_vals = np.zeros((synth_cov_times.size, synth_spo.corr_grid_z.size))
                range_est = np.zeros((synth_cov_times.size))
            if wnc == True:
                synth_range_vals[cov_index, :] = np.max(synth_spo.wnc_out, axis=0)
                synth_depth_vals[cov_index, :] = np.max(synth_spo.wnc_out, axis=1)
                est = synth_spo.corr_grid_r[np.argmax(synth_spo.wnc_out) % synth_spo.corr_grid_r.size]
                range_est[cov_index] = est
            else:
                synth_range_vals[cov_index, :] = np.max(synth_spo.bart_out, axis=0)
                synth_depth_vals[cov_index, :] = np.max(synth_spo.bart_out, axis=1)
                est = synth_spo.corr_grid_r[np.argmax(synth_spo.bart_out) % synth_spo.corr_grid_r.size]
                range_est[cov_index] = est

        pcnt_corr = get_hit_prcnt(synth_cov_times, range_est)


        print(np.min(synth_spo.corr_grid_r), np.max(synth_spo.corr_grid_r))    
        print(np.min(spo.corr_grid_r), np.max(spo.corr_grid_r))    
        cf = axes[1,1].pcolormesh(synth_cov_times/60, synth_spo.corr_grid_r,synth_range_vals.T, vmin=db_min, vmax=0, cmap=cmap)
        cf = axes[1,0].pcolormesh(cov_times/60, spo.corr_grid_r,simple_range_vals.T, vmin=db_min, vmax=0, cmap=cmap)
        cf = axes[0,1].pcolormesh(synth_cov_times/60, synth_spo.corr_grid_z,synth_depth_vals.T, vmin=db_min, vmax=0, cmap=cmap)
        cf = axes[0,0].pcolormesh(cov_times/60, spo.corr_grid_z,simple_depth_vals.T, vmin=db_min, vmax=0, cmap=cmap)

    cb = fig.colorbar(cf, ax=axes.ravel().tolist())
    cb.set_label('dB', rotation='vertical', fontsize=12)

    cols = ['Traditional WNC', 'Range-coherent WNC']

    axes[0,0].invert_yaxis()
    for i in range(2):
        axes[0,i].set_title(cols[i], fontsize=15)


    fig.text(0.5, 0.02, 'Event Time (min)', ha='center', fontsize=15)
    axes[0,0].set_ylabel('Depth (m)', fontsize=15)
    axes[1,0].set_ylabel('Range (m)', fontsize=15)


    letters = ['e)', 'f)', 'g)', 'h)']

    if proj_str == 's5_deep':
        letters = ['a)', 'b)', 'c)', 'd)']
    i = 0
    for ax in axes.ravel().tolist():
        if i < 2:
            ax.text(40.5, 175, letters[i], color='w', fontsize=26)
        else:
            ax.text(40.5, 1000, letters[i], color='w', fontsize=26)
        i += 1
    
    #fig.set_size_inches(8, 4)

    if proj_str == 's5_deep':
        cb.remove()

    plt.savefig('/home/hunter/research/coherent_matched_field/paper/pics/' + fig_name, dpi=500, orientation='landscape',bbox_inches='tight')
    plt.show()
        
