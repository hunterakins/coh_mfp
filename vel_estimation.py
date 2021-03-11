import numpy as np
from matplotlib import pyplot as plt

"""
Description:
Estimate the velocity of the s5 event using matched field inversion

Date:
2/17/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

from coh_mfp.data_test import DRPRuns, DataRun
from coh_mfp.quick_plots import get_drp_runs
from swellex.audio.config import get_proj_tones, get_proj_zs

def run_vel_mfp(proj_str, num_snapshots):
    incoh = False
    wnc = False
    freqs = get_proj_tones(proj_str)
    subfolder = '2048'
    vv = np.linspace(-1.8, -2.4, 20)
    vv = np.array([round(x, 4) for x in vv])
    num_snap_list = [num_snapshots]
    num_freq_list = [13]
    num_synth_el_list = [10]
    tilt_angles = np.array([-1])
    dr_runs = DRPRuns(proj_str, subfolder, num_snap_list, num_freq_list, num_synth_el_list, vv, tilt_angles, incoh, wnc)
    dr_runs.run_all()
    dr_runs.save('vel_est_' + str(num_snapshots) + '_' + proj_str)
   
def load_mfp_results(proj_str, num_snapshots):
    drp = get_drp_runs(proj_str, '2048', 'vel_est_' + str(num_snapshots) + '_' + proj_str)
    return drp

def load_vel_arr(proj_str, num_snapshots=4):
    x = np.load('npy_files/' + proj_str + '/vel_est_' + str(num_snapshots) + '.npy') 
    return x


if __name__ == '__main__':
    num_snapshots = 4
    for proj_str in ['s5_deep', 's5_quiet1', 's5_quiet2', 's5_quiet3', 's5_quiet4']:
        run_vel_mfp(proj_str, num_snapshots)
        drp = load_mfp_results(proj_str, num_snapshots)
        #drp.show_param_corrs()
        dr, vel_arr, tilt_arr = drp.get_param_arrs(num_snapshots, 10, 13)
        
        max_vel_corr_inds = np.argmax(vel_arr, axis=0)
        plt.figure()
        plt.plot(dr.cov_t, drp.vv[max_vel_corr_inds])

        best_vel = drp.vv[max_vel_corr_inds]
        xy_arr = np.zeros((2, best_vel.size))
        xy_arr[0,:] = dr.cov_t
        xy_arr[1,:] = best_vel[:]
        np.save('npy_files/' + proj_str + '/vel_est_' + str(num_snapshots) + '.npy', xy_arr)

        #tilt_db = 10*np.log10(tilt_arr)
        #plt.figure()
        #levels = np.linspace(-1, 0, 11)
        #plt.contourf(dr.cov_t,drp.tilt_angles, tilt_db, levels=levels)
        plt.show()
        

        
        
