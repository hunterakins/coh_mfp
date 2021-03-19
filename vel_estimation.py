import numpy as np
from matplotlib import pyplot as plt
import time

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
import swellex.audio.make_snapshots as ms

vv = ms.get_vv()
def run_vel_mfp(proj_str, subfolder, num_snapshots, num_synth_els):
    incoh = False
    wnc = False
    freqs = get_proj_tones(proj_str)
    num_snap_list = [num_snapshots]
    num_freq_list = [13]
    num_synth_el_list = [num_synth_els]
    tilt_angles = np.array([-1])
    dr_runs = DRPRuns(proj_str, subfolder, num_snap_list, num_freq_list, num_synth_el_list, vv, tilt_angles, incoh, wnc)
    dr_runs.run_all()
    dr_runs.save('vel_est_' + str(num_snapshots))
   
def load_mfp_results(proj_str, subfolder, num_snapshots):
    drp = get_drp_runs(proj_str, subfolder, 'vel_est_' + str(num_snapshots))
    return drp

def load_vel_arr(proj_str, subfolder, num_snapshots=4):
    x = np.load('npy_files/' + proj_str + '/' + subfolder + '/vel_est_' + str(num_snapshots) + '.npy') 
    return x


if __name__ == '__main__':
    N_fft = 2048
    num_snapshots = 36
    fact = 8
    N_fft = fact*N_fft
    num_synth_els = 5
    num_snapshots = int(num_snapshots/ fact)
    print(num_snapshots)
    num_freqs=13
    subfolder = str(N_fft)

    t0 = time.time()
    for proj_str in ['s5_deep', 's5_quiet1', 's5_quiet2', 's5_quiet3', 's5_quiet4']:
    #for proj_str in ['s5_quiet4']:#, 's5_quiet3', 's5_quiet4']:
    #for proj_str in ['s5_quiet4']:
        #run_vel_mfp(proj_str, subfolder, num_snapshots, num_synth_els)
        drp = load_mfp_results(proj_str, subfolder, num_snapshots)
        #drp.show_param_corrs()
        dr, vel_arr, tilt_arr = drp.get_param_arrs(num_snapshots, num_synth_els, num_freqs)
        
        max_vel_corr_inds = np.argmax(vel_arr, axis=0)

        best_vel = drp.vv[max_vel_corr_inds]
        xy_arr = np.zeros((2, best_vel.size))
        xy_arr[0,:] = dr.cov_t
        xy_arr[1,:] = best_vel[:]
        np.save('npy_files/' + proj_str + '/' + subfolder + '/vel_est_' + str(num_snapshots) + '.npy', xy_arr)

        plt.figure()
        plt.plot(xy_arr[0,:], xy_arr[1,:])
    plt.show()


    print('total runtime = ', time.time() - t0)
