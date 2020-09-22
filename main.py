import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.config import num_ranges, cov_int_time, freqs
from coh_mfp.sim import run_sim
from coh_mfp.get_dvecs import gen_dvecs
from coh_mfp.get_cov import make_super_cov_mat_seq, make_range_super_cov_mat_seq, make_cov_mat_seq

'''
Description:
Run a full simulation, including time domain generation,
STFT, covariance estimation, and matched field

Date: 
09/14/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


""" Generate synthetic data as well as replicas """
run_sim()


""" Do STFT to extract the data vectors from the time series 
synthetic data """
gen_dvecs()

""" Form sequence of sample covariance mats """
for freq in freqs:
    make_range_super_cov_mat_seq(freq, cov_int_time, phase_key='source_correct', num_ranges=num_ranges)
    make_cov_mat_seq(freq, cov_int_time)


#make_super_cov_mat_seq(freqs, cov_int_time, 'naive')
#make_super_cov_mat_seq(freqs, cov_int_time, 'source_correct')
#make_super_cov_mat_seq(freqs, cov_int_time, 'MP_norm')

