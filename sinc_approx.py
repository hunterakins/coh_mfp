import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.vel_estimation import load_vel_arr

"""
Description:
mainlobe sinc model approximation

Date:
4/1/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego

"""


def true_win(kbar, num_synth_els, range_err):
    return 1/num_synth_els/num_synth_els*np.square(np.sin(kbar * range_err * num_synth_els / 2)/np.sin(kbar *range_err /2))

def sinc_win(kbar, num_synth_els, range_err):
    return np.square(np.sinc(kbar * range_err * num_synth_els / 2 / np.pi))

f = 100
c = 1500


def check_approx(f, c):
    kbar = 2*np.pi*f/c
    range_errs = np.linspace(-5, 5, 100)
    num_synth_els = 5

    for num_synth_els in [5, 6, 7, 8]:
        true = true_win(kbar, num_synth_els, range_errs)
        plt.plot(range_errs, true, color='r')
        approx = sinc_win(kbar, num_synth_els, range_errs)
        plt.plot(range_errs, approx, color='b')
        plt.show()



def v_mainlobe(freq, c, num_synth_els, T):
    v = 2.8 * c / 2 / np.pi / num_synth_els / T / freq
    return v


for f in [100, 200, 300, 400]:
    num_synth_els = 5
    T = 25
    v = v_mainlobe(f, c, num_synth_els, T)
    print('f, v', f, v)


proj_str = 's5_deep'
subfolder = '2048'
num_snapshots = 36
num_synth_els = 5
v_arr = load_vel_arr(proj_str, subfolder, num_snapshots, num_synth_els)
plt.plot(v_arr[0,:], v_arr[1,:])
plt.show()
