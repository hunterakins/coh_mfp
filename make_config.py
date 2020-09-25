import numpy as np
import coh_mfp.config as conf
from env.env.json_reader import write_json

'''
Description:
Store project parameters

Date: 
8/25/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


""" Pick source params
and receiver config """


#acc_amp = 0.2 #m/s
acc_amp = 0.0
acc_T = 10 # second
freqs = [49, 64]#, 79, 94, 109]#, 112, 127, 130, 145, 148]
source_vel = 3 # ship range rate in m/s
fft_len = 2048
fft_spacing = 1024 # 
SNR = 0 # after fft gain is accounted for
fs = 1500 # sampling rate
ship_dr = source_vel * fft_spacing / fs

""" Replica grid parameters """
dz =  5
zmax = 216.5
dr = 50
rmax = 2*1e3

""" 
Generate a fake range rate track at 1500 Hz sampling rate
"""
r0 = 1000
r1 = 1200 

""" Calculate time domain field with zero initial phase """
#zr = np.array([94.125, 99.755, 105.38, 111.00, 116.62, 122.25, 127.88, 139.12, 144.74, 150.38, 155.99, 161.62, 167.26, 172.88, 178.49, 184.12, 189.76, 195.38, 200.99, 206.62, 212.25])
zr = np.linspace(94.125, 212.24, 10)
zs = 54



""" coavriance integration time (seconds)"""
cov_int_time = 10
num_ranges = 5

""" STFT Params"""
n_overlap = fft_len-fft_spacing

fig_folder = '/home/fakins/code/coh_mfp/pics/'

num_realizations = 5


exp_id = 0

proj_root = '/oasis/tscc/scratch/fakins/coh_mfp/' + str(exp_id) + '/'


if __name__ == '__main__':
    """
    Run as main if you want to generate a json file for this experiment
    """
    var_dict = locals()
    save_dict = conf.get_conf_dict(var_dict)
    print(save_dict)
    conf_name = conf.make_conf_name(exp_id)
    write_json(conf_name, save_dict)