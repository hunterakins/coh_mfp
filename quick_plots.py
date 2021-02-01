import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.data_test import DRPRuns, DataRun
import time
from swellex.audio.config import get_proj_zs, get_proj_zr, get_proj_tones

"""
Description:
Just a throwaway script for generating some figs

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

proj_str = 's5_deep'
#proj_str = 's5_quiet4'
#proj_str = 's5_quiet3'
proj_str = 's5_quiet1'
#proj_str = 's5_quiet2'

subfolder = '8096'

def drp_plots():
    t0 = time.time()
    zs = get_proj_zs(proj_str)
    vv = np.linspace(-2.6, -1.8, 10)
    vv = np.array([round(x, 4) for x in vv])
    num_snap_list = [15]
    num_freq_list = [13]
    num_synth_el_list = [1, 15]
    dr_runs = DRPRuns(proj_str, subfolder, num_snap_list, num_freq_list, num_synth_el_list, vv)
    dr_runs.make_amb_mov()
    dr_runs.get_best_range_guesses()
    dr_runs.show_vel_corrs()

drp_plots()

