import numpy as np
from matplotlib import pyplot as plt
from comparison_plot import get_tracking_spo
from wnc_test import get_cov_time, check_v_arr
from proc_out import SwellProcObj, load_spo
from vel_estimation import load_vel_arr
from scipy.interpolate import interp1d
from copy import deepcopy
from swellex.audio import make_snapshots as ms
import sys

vv = ms.get_vv()

"""
Description:
Make a figure that shows the bartlett rmse as a function of estimated snr


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

    diffs = abs(rvals - range_vals)/1000
    sq_diffs = np.square(diffs)
    count = 0
    for x in diffs:
        if x < 200/1000:
            count += 1
    prcnt = count / cov_times.size
    print('hit percent:' + str(prcnt * 100) + '\%')
    rmse = np.sqrt(np.mean(sq_diffs))
    print('rmse', rmse)
    return sq_diffs

def data_snr(proj_string):
    """
    Give a function of exp. time in seconds
    that gives the snr
    assume nfft is 16384

    """
    snr0 = 21.5
    if proj_string == 's5_deep':
        snr0 -= 0
    elif proj_string[:-1] == 's5_quiet':
        snr0 -= 26
        snr0 -= 4*(int(proj_string[-1])-1)

    v_avg = (8.65 - 1.35)*1000 / (52*60)#m /s
    print('v_avg', v_avg)
    r0 = 1350 #m
    
    r_t = lambda t: r0 + v_avg * (52*60 - t)
    snr_t = lambda t: snr0 - 10*np.log10(r_t(t) / r0)
    tgrid = np.linspace(0, 52*60, 100)
    return snr_t

if __name__ == '__main__':

    colors = ['r', 'b', 'g', 'k']
    fig, axis = plt.subplots(1,1)
    proj_count = 0
    for proj_str in ['s5_deep', 's5_quiet1', 's5_quiet2', 's5_quiet3']:

        snr_t = data_snr(proj_str)

        #proj_str = 's5_quiet3'

        N_fft = 2048
        num_snapshots = 36

        fact = 8

        N_fft = fact*N_fft
        num_snapshots = int(num_snapshots / fact)
            

        for sub_count in range(3):
            subfolder = str(N_fft) 
            subfolder = str(N_fft)
            if sub_count == 1:
                subfolder += '_sec2'
            elif sub_count == 2:
                subfolder = str(N_fft) + '_sec1'
            num_els = 5
            tilt_angle = -1 
            num_freqs = 13
            wn_gain = -0.5
            wn_gain = -.5

            cov_times = get_cov_time(proj_str, subfolder, num_snapshots, num_els)
            v_arr = load_vel_arr(proj_str, subfolder, num_snapshots,num_els)
            v_arr = check_v_arr(v_arr, cov_times)
            v_interp = interp1d(v_arr[0,:], v_arr[1,:])

            root_folder ='pickles/'

            wnc = False

            """ NOW DO SYNTHETIC """
            for cov_index in range(cov_times.size):
                cov_time = cov_times[cov_index]
                v_source = v_interp(cov_time)
                v_source = vv[np.argmin([abs(v_source -x) for x in vv])]
                spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle, num_els, num_freqs, v_source, cov_time, wn_gain)
                spo.get_bathy_corr()
                spo.wnc_out -= np.max(spo.wnc_out)
                spo.bart_out -= np.max(spo.bart_out)

                if cov_index == 0:
                    range_vals = np.zeros((cov_times.size, spo.corr_grid_r.size))
                    depth_vals = np.zeros((cov_times.size, spo.corr_grid_z.size))
                    range_est = np.zeros((cov_times.size))

                range_vals[cov_index, :] = np.max(spo.bart_out, axis=0)
                depth_vals[cov_index, :] = np.max(spo.bart_out, axis=1)
                est = spo.corr_grid_r[np.argmax(spo.bart_out) % spo.corr_grid_r.size]
                range_est[cov_index] = est

            sq_diffs = get_hit_prcnt(cov_times, range_est)
            snr_vals = snr_t(cov_times)

            axis.scatter(snr_vals, np.sqrt(sq_diffs), color=colors[proj_count])

    
        proj_count += 1
    axis.set_ylim([0, 3])
    plt.show()




