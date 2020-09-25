import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.config import ExpConf, load_config
from coh_mfp.sim import run_sim
from coh_mfp.get_dvecs import gen_dvecs
from coh_mfp.get_cov import make_cov_mat_seq
from coh_mfp.bartlett import get_amb_surf


'''
Description:
Run a full simulation, including time domain generation,
STFT, covariance estimation, and matched field

Date: 
09/14/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

exp_id = 0
exp_conf = load_config(exp_id)

""" Generate synthetic data as well as replicas """
#run_sim(exp_conf)


""" Do STFT to extract the data vectors from the time series 
synthetic data """
#for sim_iter in range(exp_conf.num_realizations):
#    gen_dvecs(sim_iter, exp_conf)

""" Form sequence of sample covariance mats """
for freq in exp_conf.freqs:
    for sim_iter in range(exp_conf.num_realizations):
        cov_int_time, proj_root, num_ranges = exp_conf.cov_int_time, exp_conf.proj_root, exp_conf.num_ranges
        #make_range_super_cov_mat_seq(freq, cov_int_time, sim_iter, proj_root, phase_key='source_correct', num_ranges=num_ranges)
        make_cov_mat_seq([freq], cov_int_time, sim_iter, exp_conf, 'none')
        kwargs = {'num_ranges':2, 'phase_key':'source_correct'}
        make_cov_mat_seq([freq], cov_int_time, sim_iter, exp_conf, 'range', **kwargs)
        #get_amb_surf(freq, sim_iter)




#make_super_cov_mat_seq(freqs, cov_int_time, 'naive')
#make_super_cov_mat_seq(freqs, cov_int_time, 'source_correct')
#make_super_cov_mat_seq(freqs, cov_int_time, 'MP_norm')

