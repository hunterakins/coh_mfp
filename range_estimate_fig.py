import numpy as np
from matplotlib import pyplot as plt
from comparison_plot import get_tracking_spo
from wnc_test import get_cov_time
from proc_out import SwellProcObj, load_spo
from vel_estimation import load_vel_arr
from scipy.interpolate import interp1d
from copy import deepcopy

"""
Description:
Make a figure that shows the range of maximum correlation
Compares the various methods

Date:
3/8/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

if __name__ == '__main__':

    #proj_str = 's5_deep'
    proj_str = 's5_quiet4'

    N_fft = 2048
    num_snapshots = 36
    fact = 16
    N_fft = fact*N_fft

    subfolder = str(N_fft)
    num_snapshots = int(num_snapshots / fact)
    num_synth_els = 5
    num_tracking_els = num_synth_els
    tilt_angle = -1 
    num_freqs = 13
    wn_gain = -.5
    cov_times=get_cov_time(proj_str, subfolder, num_snapshots, num_synth_els)
    v_arr = load_vel_arr(proj_str, subfolder, num_snapshots)
    if v_arr[0, -1] < cov_times[-1]:
        new_t = cov_times[-1]
        new_v = v_arr[1,-1]
        new_entry = np.array([new_t, new_v]).reshape(2,1)
        v_arr = np.concatenate((v_arr, new_entry), axis=1)

    v_interp = interp1d(v_arr[0,:], v_arr[1,:])

    root_folder ='pickles/'

    cmap = plt.cm.get_cmap('viridis')
    fig_name = proj_str + '_range_est_color.png'
    #wnc = True
    wnc = True


    for cov_index in range(cov_times.size-num_tracking_els):
        cov_time = cov_times[cov_index]
        v_source = v_interp(cov_time)
        tracking_spo = get_tracking_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle,num_freqs, num_tracking_els, v_interp, cov_index, wnc=True, wn_gain = wn_gain)
        tracking_spo.wnc_out -= np.max(tracking_spo.wnc_out)
        tracking_spo.bart_out -= np.max(tracking_spo.bart_out)
    
        synth_spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle, num_synth_els, num_freqs, v_source, cov_time, wn_gain)
        synth_spo.get_bathy_corr()
        synth_spo.wnc_out -= np.max(synth_spo.wnc_out)
        synth_spo.bart_out -= np.max(synth_spo.bart_out)
    
    
        spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle, 1, num_freqs, v_source, cov_time, wn_gain)
        spo.get_bathy_corr()
        spo.wnc_out -= np.max(spo.wnc_out)
        spo.bart_out -= np.max(spo.bart_out)

        if cov_index == 0:
            synth_range_vals = np.zeros((cov_times.size-num_tracking_els, synth_spo.corr_grid_r.size))
            tracking_range_vals = np.zeros((cov_times.size-num_tracking_els, synth_spo.corr_grid_r.size))
            simple_range_vals = np.zeros((cov_times.size-num_tracking_els, synth_spo.corr_grid_r.size))
            synth_depth_vals = np.zeros((cov_times.size-num_tracking_els, synth_spo.corr_grid_z.size))
            tracking_depth_vals = np.zeros((cov_times.size-num_tracking_els, synth_spo.corr_grid_z.size))
            simple_depth_vals = np.zeros((cov_times.size-num_tracking_els, synth_spo.corr_grid_z.size))

        if wnc == True:
            synth_range_vals[cov_index, :] = np.max(synth_spo.wnc_out, axis=0)
            tracking_range_vals[cov_index, :] = np.max(tracking_spo.wnc_out, axis=0)
            simple_range_vals[cov_index, :] = np.max(spo.wnc_out, axis=0)
            synth_depth_vals[cov_index, :] = np.max(synth_spo.wnc_out, axis=1)
            tracking_depth_vals[cov_index, :] = np.max(tracking_spo.wnc_out, axis=1)
            simple_depth_vals[cov_index, :] = np.max(spo.wnc_out, axis=1)
        else:
            synth_range_vals[cov_index, :] = np.max(synth_spo.bart_out, axis=0)
            tracking_range_vals[cov_index, :] = np.max(tracking_spo.bart_out, axis=0)
            simple_range_vals[cov_index, :] = np.max(spo.bart_out, axis=0)
            synth_depth_vals[cov_index, :] = np.max(synth_spo.bart_out, axis=1)
            tracking_depth_vals[cov_index, :] = np.max(tracking_spo.bart_out, axis=1)
            simple_depth_vals[cov_index, :] = np.max(spo.bart_out, axis=1)



    fig, axes = plt.subplots(2,3, sharex=True, sharey='row')
    cov_times = cov_times[:-num_tracking_els]
    db_min = -15
    
    cf = axes[1,2].pcolormesh(cov_times, spo.corr_grid_r,synth_range_vals.T, vmin=db_min, vmax=0, cmap=cmap)
    cf = axes[1,0].pcolormesh(cov_times, spo.corr_grid_r,simple_range_vals.T, vmin=db_min, vmax=0, cmap=cmap)
    cf = axes[1,1].pcolormesh(cov_times, spo.corr_grid_r,tracking_range_vals.T, vmin=db_min, vmax=0, cmap=cmap)
    cf = axes[0,2].pcolormesh(cov_times, spo.corr_grid_z,synth_depth_vals.T, vmin=db_min, vmax=0, cmap=cmap)
    cf = axes[0,0].pcolormesh(cov_times, spo.corr_grid_z,simple_depth_vals.T, vmin=db_min, vmax=0, cmap=cmap)
    cf = axes[0,1].pcolormesh(cov_times, spo.corr_grid_z,tracking_depth_vals.T, vmin=db_min, vmax=0, cmap=cmap)
    cb = fig.colorbar(cf, ax=axes.ravel().tolist())
    cb.set_label('dB', rotation='horizontal')

    cols = ['Traditional', 'MFT', 'Range-coherent']

    for i in range(3):
        axes[0,i].invert_yaxis()
        axes[0,i].set_title(cols[i])


    fig.text(0.5, 0.02, 'Time (s)', ha='center')
    axes[0,0].set_ylabel('Depth (m)')
    axes[1,0].set_ylabel('Range (m)')
    
    fig.set_size_inches(8, 4)
    #plt.savefig('/home/hunter/research/coherent_matched_field/pics/' + fig_name, dpi=500, orientation='landscape')
    plt.show()
    
