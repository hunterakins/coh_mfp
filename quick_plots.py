import numpy as np
import pickle
from matplotlib import pyplot as plt
from coh_mfp.data_test import DRPRuns, DataRun, make_save_loc
import time
from swellex.audio.config import get_proj_zs, get_proj_zr, get_proj_tones

"""
Description:
Just a throwaway script for generating some figs

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


def get_drp_runs(proj_str, subfolder, drp_name):
    """
    Project string, subfolder and name, load up 
    the appropriate drpruns object, return it
    """
    fname = make_save_loc(proj_str, subfolder) + drp_name + '.pickle'
    with open(fname, 'rb') as f:
        drp_obj = pickle.load(f)
    print(drp_obj.vv)
    return drp_obj

def drp_plots(proj_str, subfolder, drp_name):
    dr_runs = get_drp_runs(proj_str, subfolder, drp_name)
    print('incoh', dr_runs.incoh)
    print('num snaps', dr_runs.num_snap_list)
    #dr_runs.make_amb_mov()
    dr_runs.get_best_range_guesses()
    #dr_runs.show_param_corrs()
    dr_runs.show_vel_best_guess()

if __name__ == '__main__':
    #proj_str = 's5_deep'
    #proj_str = 's5_quiet2'
    for proj_str in ['s5_deep', 's5_quiet1']:#, 's5_quiet3', 's5_quiet4']:

        #proj_str = 's5_quiet1'
        print(get_proj_tones(proj_str))
    #proj_str = 's5_quiet1'
    #proj_str = 's5_quiet4'

        subfolder = '2048'
        drp_plots(proj_str, subfolder, '4_snaps')
        #drp_plots(proj_str, subfolder, 'beefier_test')
        plt.show()
    
    #drp_plots(proj_str, subfolder, 'beefier_test')



