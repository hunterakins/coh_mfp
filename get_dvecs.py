import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
from coh_mfp.sim import make_raw_ts_name
#from coh_mfp.config import freqs, source_vel, fft_len, fft_spacing, lizations, PROJ_ROOT

'''
Description:
Get the data vectors from the simulated time series by means
of a windowed stft

Date: 
8/25/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def load_ts(sim_iter, proj_root):
    x = np.load(make_raw_ts_name(sim_iter, proj_root))
    return x

def check_small_samp(x):
    """
    Load 10 seconds
    Compute fft
    plot PSD on first element
    """
    x0 = x[0,:]
    ind1 = 2048
    x0 = x0[:ind1]
    fvals = np.fft.fft(x0)
    freq = np.fft.fftfreq(x0.size, 1/1500)
    thing = np.zeros((2, fvals.size))
    thing[0,:] = np.square(abs(fvals))
    thing[1,:] = freq
    np.save('check_vals.npy', thing)
    fig = plt.figure()
    plt.plot(freq, abs(fvals))
    plt.savefig('check.png')
    plt.close(fig)

def make_dvec_name(freq, sim_iter, proj_root):
    """
    Create a string that is the absolute file
    path of the "dvec" for the given source freq"""
    return proj_root + str(freq) + '_dvec' + str(sim_iter) +'.npy'

def load_dvec(freq, sim_iter, proj_root):
    """ load the dvec into a numpy array
    Input
    freq - float
        source frequency band of dvecs you want """
    dvec_name = make_dvec_name(freq, sim_iter, proj_root=proj_root)
    x = np.load(dvec_name)
    return x

def make_tgrid_name(proj_root):
    """create string for absolute file 
    path of the corresponding time values for the
    given data vectors (freq. domain field) """
    return proj_root + 'times.npy'

def load_tgrid(proj_root):
    """
    Load the times associated with the ~beginning~ of the 
    fft windows used to form the data vectors
    """
    x = np.load(make_tgrid_name(proj_root))
    return x

def gen_dvecs(sim_iter, conf):
    n_overlap = conf.fft_len-conf.fft_spacing # number of samples ot overlap
    x = load_ts(sim_iter, conf.proj_root)
    check_small_samp(x)
    
    fgrid, times, vals = stft(x, fs=1500, nperseg=conf.fft_len, noverlap=n_overlap, nfft=8192)
    v = conf.source_vel
    
    for freq in conf.freqs:
        f_dop = freq - freq*v/1500
        print('Doppler shifted_freq', f_dop)
        f_ind = np.argmin([abs(f_dop - fgrid[i]) for i in range(len(fgrid))])
        print(f_ind, 'find')
        print(fgrid[f_ind], 'f_grid f')
        print(fgrid[f_ind-1], 'adjacent f_grid f')
        print(fgrid[f_ind+1], 'adjacent f_grid f')
        fvals = vals[:,f_ind,:]
        dvec_name = make_dvec_name(freq, sim_iter, conf.proj_root)
        np.save(dvec_name, fvals)
    noise_freqs = [x + 2 for x in conf.freqs]
    for freq in noise_freqs:
        print('Noise freq', freq)
        #f_dop = freq - freq*v/1500
        f_ind = np.argmin([abs(freq - fgrid[i]) for i in range(len(fgrid))])
        fvals = vals[:,f_ind,:]
        dvec_name = make_dvec_name(freq, sim_iter, conf.proj_root)
        np.save(dvec_name, fvals)
    np.save(make_tgrid_name(conf.proj_root), times)


if __name__ == '__main__':
    gen_dvecs(0)
