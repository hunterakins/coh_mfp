import numpy as np
from matplotlib import pyplot as plt

"""
Description:
Estimate the tilt of the s5 event using matched field inversion

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
def run_tilt_mfp(suffix):
    incoh = False
    wnc = False
    proj_str = 's5_deep'
    freqs = get_proj_tones(proj_str)
    subfolder = '2048'+ suffix
    #vv = np.linspace(-1.8, -2.5, 12)
    #vv = np.array([round(x, 4) for x in vv])
    num_snap_list = [4]
    num_freq_list = [13]
    num_synth_el_list = [1]
    tilt_angles = np.linspace(0, -2, 5)
    print(tilt_angles)
    tilt_angles = np.array([round(x, 4) for x in tilt_angles])
    dr_runs = DRPRuns(proj_str, subfolder, num_snap_list, num_freq_list, num_synth_el_list, vv, tilt_angles, incoh, wnc)
    dr_runs.run_all()
    dr_runs.save('tilt_est')
   
def load_mfp_results(suffix):
    drp = get_drp_runs('s5_deep', '2048'+suffix, 'tilt_est')
    return drp

if __name__ == '__main__':
    fig, ax = plt.subplots(1,1)
    for sec in ['', '_sec1', '_sec2']:
        #run_tilt_mfp(sec)
        drp = load_mfp_results(sec)
        #drp.show_param_corrs()
        print(drp.tilt_angles, drp.vv)
        dr, vel_arr, tilt_arr = drp.get_param_arrs(4, 1, 13)
         
        max_tilt_corr_inds = np.argmax(tilt_arr, axis=0)
        #plt.figure()
        ax.plot(dr.cov_t, drp.tilt_angles[max_tilt_corr_inds])

        tilt_db = 10*np.log10(tilt_arr)
        plt.figure()
        levels = np.linspace(-1, 0, 11)
        plt.contourf(dr.cov_t,drp.tilt_angles, tilt_db, levels=levels)
    plt.show()
