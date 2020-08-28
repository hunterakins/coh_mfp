import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
from coh_mfp.sim import freqs, source_vel, fft_len

'''
Description:
Get the data vectors from the simulated time series by means
of a windowed stft

Date: 
8/25/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''
PROJ_ROOT = '/oasis/tscc/scratch/fakins/coh_mfp/'

def load_ts(proj_root=PROJ_ROOT):
    x = np.load(proj_root+'sim_data.npy')
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

def make_dvec_name(freq, proj_root =PROJ_ROOT):
    """
    Create a string that is the absolute file
    path of the "dvec" for the given source freq"""
    return proj_root + str(freq) + '_dvec.npy'

def load_dvec(freq, proj_root=PROJ_ROOT):
    """ load the dvec into a numpy array
    Input
    freq - float
        source frequency band of dvecs you want """
    dvec_name = make_dvec_name(freq, proj_root=proj_root)
    x = np.load(dvec_name)
    return x

def make_tgrid_name(proj_root=PROJ_ROOT):
    """create string for absolute file 
    path of the corresponding time values for the
    given data vectors (freq. domain field) """
    return proj_root + '_times.npy'

def load_tgrid(proj_root=PROJ_ROOT):
    """
    Load the tgrid  
    """
    x = np.load(make_tgrid_name(proj_root=PROJ_ROOT))
    return x

fft_spacing = 1024 # spacingn in samples
n_overlap = fft_len-fft_spacing # number of samples ot overlap

if __name__ == '__main__':
    x = load_ts()
    check_small_samp(x)
    
    fgrid, times, vals = stft(x, fs=1500, nperseg=fft_len, noverlap=n_overlap, nfft=4096)
    print(x.shape, fgrid.shape, times.shape, vals.shape)
    v = source_vel
    print('Source velocity', source_vel)
    
    for freq in freqs:
        print('Freq', freq)
        f_dop = freq - freq*v/1500
        print('Doppler shifted_freq', f_dop)
        f_ind = np.argmin([abs(f_dop - fgrid[i]) for i in range(len(fgrid))])
        print(f_ind, 'find')
        print(fgrid[f_ind], 'f_grid f')
        print(fgrid[f_ind-1], 'adjacent f_grid f')
        print(fgrid[f_ind+1], 'adjacent f_grid f')
        fvals = vals[:,f_ind,:]
        dvec_name = make_dvec_name(freq)
        np.save(dvec_name, fvals)
    noise_freqs = [x + 2 for x in freqs]
    for freq in noise_freqs:
        print('Freq', freq)
        #f_dop = freq - freq*v/1500
        f_ind = np.argmin([abs(freq - fgrid[i]) for i in range(len(fgrid))])
        fvals = vals[:,f_ind,:]
        dvec_name = make_dvec_name(freq)
        np.save(dvec_name, fvals)
    np.save(make_tgrid_name(), times)

    #check_small_samp(x)


        
        
