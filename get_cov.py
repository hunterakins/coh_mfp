import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.sim import freqs, fs
from coh_mfp.get_dvecs import load_dvec, load_tgrid, fft_spacing, PROJ_ROOT

'''
Description:
Form the sample covariance matrix for the simulated 
data

Date: 
8/26/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def make_cov_name(freq, proj_root=PROJ_ROOT):
    name = proj_root + str(freq) + '_cov.npy' 
    return name

def make_cov_time_name(proj_root=PROJ_ROOT):
    name = proj_root + 't_cov.npy'
    return name

def get_frame_len(cov_int_time):
    """
    Get number of dvecs required to 
    form a sample covariance matrix over
    cov_int_time seconds 
    fft_spacing is a global project variable
    (the number of samples between fft windows)
    fs is also a global variable, the sampling
    rate of the system
    """
    delta_t = fft_spacing/fs # time between adjacent d_vecs
    frame_len = int(cov_int_time / delta_t) # num dvecs per
    return frame_len

def get_num_frames(tgrid, frame_len):   
    """
    frame_len - int
        number of frames used in cov est.
    tgrid - np 1d array
        time grid associated with stqrt
        points of each data vec
    Ignores the number of samples that hang off 
    the end of data that frame_len doesn't evenly
    divide (number of ignored frames is 
    tgrid.size % frame_len
    """
    num_frames = tgrid.size // frame_len
    return num_frames

def make_cov_mat_seq(freq, cov_int_time):
    """
    For a given frequency band and a covariance
    integration time in seconds, produce a sequence
    of covariance estimates with no overlap
    Save the cov estimates in a numpy ndarray,
    the last axis is time
    Input
    freq - int
        source frequency to analyze
    cov_int_time - float
        integration time in seconds
    Output-
    K_samp - np ndarray
        last aaxis is time, first two store cov mats
    tvals - np 1darray
        first time stamp associated with the corresponding
        sample covariance matrix
    """
    tgrid = load_tgrid()

    """ Get number of dvecs in each cov estimation"""
    frame_len = get_frame_len(cov_int_time)
    print('num frames used in covariance estimation', frame_len)
    num_frames = get_num_frames(tgrid, frame_len)
    print('total number of chunks analyzed', num_frames)
    print('time spacing of cov estimates', tgrid[frame_len] - tgrid[0])

    """ Load up the relevant dvecs """
    dvecs = load_dvec(freq)
    dvecs = dvecs / np.linalg.norm(dvecs, axis=0)
    num_rcvrs = dvecs.shape[0]
    print(np.linalg.norm(dvecs[:,0]))
    print(np.linalg.norm(dvecs[:,-1]))
    
    """ Initialize sample covariance array """
    K_samp = np.zeros((num_rcvrs, num_rcvrs, num_frames), dtype=np.complex128)
    """ Iterate through frames and add sample covs to the mat """
    for i in range(num_frames):
        inds = slice(i*frame_len, (i+1)*frame_len)
        tmp_K = np.cov(dvecs[:,inds])
        K_samp[:,:,i] = tmp_K

    """ Save it """
    fname = make_cov_name(freq)
    tvals = tgrid[::frame_len]
    print(tvals.shape, K_samp.shape)
    tvals = tvals[:num_frames]
    np.save(fname, K_samp)
    tname = make_cov_time_name()
    np.save(tname, tvals)
    return tvals, K_samp

def load_cov(freq, proj_root=PROJ_ROOT):
    """ Load up cov estimates
    for source frequency freq 
    Input 
    freq - int
        source frequency
    Output 
    tvals - np 1darray of floats
        times associated with the left-hand side
        of the data chunk used in the cov estim.  
    K_samp - np ndarray 
        sequence of cov estimates for the data
    """
    fname = make_cov_name(freq, proj_root=proj_root)
    tname = make_cov_time_name(proj_root=proj_root)
    K_samp = np.load(fname)
    tvals = np.load(tname)
    return tvals, K_samp
    
cov_int_time = 10 # number of seconds to integrate to form sample cov

if __name__ == '__main__':
    for freq in freqs:
        make_cov_mat_seq(freq, cov_int_time)



