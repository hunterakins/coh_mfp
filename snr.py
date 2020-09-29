import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.config import freqs
from coh_mfp.get_dvecs import load_dvec, load_tgrid

'''
Description:
Estimate the SNR from the stft data

Date: 
8/27/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

t = load_tgrid()

noise_freqs = [x + 2 for x in freqs]
noise_pows = np.zeros((len(noise_freqs), t.size))
i=0
"""
Go through noise frequencies and form array average power 
"""
for freq in noise_freqs:
    tmp = load_dvec(freq)
    noise_pows[i,:] = np.mean(np.square(abs(tmp)), axis=0)
    i += 1

""" Now average the array averages over frequency """
avg_noise_pow = np.mean(noise_pows, axis=0)

""" Now go through and estimate signal power from dvecs """
for freq in freqs:
    dvec = load_dvec(freq)
    power = np.square(abs(dvec))
    array_avg_pow = np.mean(power, axis=0)
    total_avg = np.mean(power)
    snr = array_avg_pow/avg_noise_pow
    snr_db = 10*np.log10(snr)
    fig = plt.figure()
    plt.plot(t, snr)
    plt.savefig(str(freq) + '_snr_db.png')
    plt.close(fig)
    
