import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.config import freqs, fs, source_vel, fft_spacing, PROJ_ROOT
from coh_mfp.get_dvecs import load_dvec, load_tgrid

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

def make_super_cov_name(phase_key='naive', proj_root=PROJ_ROOT):
    name = proj_root + phase_key +  '_supercov.npy' 
    return name

def make_range_super_cov_name(freq, num_ranges, phase_key='naive', proj_root=PROJ_ROOT):
    name = proj_root + str(freq) + '_' + str(num_ranges) + '_' + phase_key +  '_supercov.npy' 
    return name

def make_cov_time_name(proj_root=PROJ_ROOT,range_stack=False):
    if range_stack==False:
        name = proj_root + 't_cov.npy'
    else:
        name = proj_root + 'rangestack_t_cov.npy'
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

def stack_dvecs(dvec_list, phase_key='naive', freqs=None):
    """
    Take in list of normalized data vectors
    Each list element is a full array of data vectors
    with the first axis being receiver depth, second
    axis being fft time
    Stack them into a supervector,
    applying the appropriate phase correction   
    at each frequency 
    Input 
    dvec_list - list of numpy ndarrays
        each element is a dvec (see make_super_cov_mat_seq)
    phase_key - string
        optional key to specify a phase correction/normalization
    freqs - list or type(None)
        if source freq knowledge is assumed supply these
    """
    tgrid = load_tgrid()
    num_times = tgrid.size
    tmp = dvec_list[0]
    num_rcvrs = tmp.shape[0]
    num_freqs = len(dvec_list)
    super_dvecs = np.zeros((num_freqs*num_rcvrs, num_times), dtype=np.complex128)
    for i in range(num_freqs):
        dvecs = dvec_list[i]
        if phase_key == 'naive':
            pass
        elif phase_key == 'source_correct':
            source_phase = 2*np.pi*freqs[i]*tgrid
            phase_correct = np.exp(complex(0,1)*-1*source_phase)
            dvecs *= phase_correct
        elif phase_key == 'MP_norm':
            first_elem = dvecs[0,:] 
            phase = np.angle(first_elem)
            phase_correct = np.exp(complex(0,1)*-phase)
            dvecs *= phase_correct
        else:
            raise ValueError('Invalid phase key provided. Options are naive, source_correct, and MP_norm')
        super_dvecs[i*num_rcvrs:(i+1)*num_rcvrs, :] = dvecs
    return super_dvecs

def range_stack_dvecs(dvecs, freq, phase_key='naive', num_ranges=2):
    """
    Similar to stack_dvecs, but instead of stacking in frequency, 
    stack in range
    Input 
    dvec_list - list of numpy ndarrays
        each element is a dvec (see make_super_cov_mat_seq)
    phase_key - string
        optional key to specify a phase correction/normalization
    freqs - list or type(None)
        if source freq knowledge is assumed supply these
    """
    tgrid = load_tgrid()
    num_times = tgrid.size
    num_rcvrs = dvecs.shape[0]
    if phase_key == 'naive':
        pass
    elif phase_key == 'source_correct':
        source_phase = 2*np.pi*freq*tgrid
        phase_correct = np.exp(complex(0,1)*-1*source_phase)
        dvecs *= phase_correct
    else:
        raise ValueError('Invalid phase key provided. Options are naive, source_correct, and MP_norm')
    lam = 1500 / freq
    replica_dr = source_vel * fft_spacing/fs
    stride = int(lam/2 / replica_dr)
    super_dvecs = np.zeros((num_ranges*num_rcvrs, num_times), dtype=np.complex128)
    print('stride', stride)
    for i in range(num_ranges):
        rel_dvecs = dvecs[:,i*stride:]
        print(rel_dvecs.shape[1])
        super_dvecs[i*num_rcvrs:(i+1)*num_rcvrs, :rel_dvecs.shape[1]] = rel_dvecs
    return super_dvecs
    
def make_super_cov_mat_seq(freq, cov_int_time, phase_key='naive'):
    """
    For the frequency bands inf freqs and a covariance
    integration time in seconds, produce a sequence
    of covariance estimates with no overlap
    Save the cov estimates in a numpy ndarray,
    the last axis is time
    phase_key allows for selection of some phase correction/
    normalization business 
    Input
    freqs - list of ints
        list of source frequencies to analyze and stack
    cov_int_time - float
        integration time in seconds
    phase_key - string
        switch for adding phase corrections to the 
        independent frequencies before stacking into the 
        supervector
    Output-
    K_samp - np ndarray
        last axis is time, first two store cov mats
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
    num_freqs = len(freqs)
    print('number of frequencies stacked', num_freqs)

    """ Load up the relevant dvecs """
    dvec_list = []
    for freq in freqs:
        freq_dvecs = load_dvec(freq)
        freq_dvecs = freq_dvecs / np.linalg.norm(freq_dvecs, axis=0)
        num_rcvrs = freq_dvecs.shape[0]
        dvec_list.append(freq_dvecs)

    dvecs = stack_dvecs(dvec_list, phase_key=phase_key, freqs=freqs)
    print('supervector dims', dvecs.shape)
    print('expected to be', num_freqs*num_rcvrs, len(tgrid))
    
    """ Initialize sample covariance array """
    K_samp = np.zeros((num_rcvrs*num_freqs, num_rcvrs*num_freqs, num_frames), dtype=np.complex128)
    """ Iterate through frames and add sample covs to the mat """
    for i in range(num_frames):
        inds = slice(i*frame_len, (i+1)*frame_len)
        tmp_K = np.cov(dvecs[:,inds])
        K_samp[:,:,i] = tmp_K

    """ Save it """
    fname = make_super_cov_name(phase_key=phase_key)
    tvals = tgrid[::frame_len]
    print(tvals.shape, K_samp.shape)
    tvals = tvals[:num_frames]
    np.save(fname, K_samp)
    tname = make_cov_time_name()
    np.save(tname, tvals)
    return tvals, K_samp

def make_range_super_cov_mat_seq(freq, cov_int_time, phase_key='naive', num_ranges=2):
    """
    For the covariance
    integration time in seconds, produce a sequence
    of covariance estimates with no overlap, using data vectors
    that stack in range using as many as num_ranges
    Save the cov estimates in a numpy ndarray,
    the last axis is time
    phase_key allows for selection of some phase correction/
    normalization business 
    Input
    freqs - list of ints
        list of source frequencies to analyze and stack
    cov_int_time - float
        integration time in seconds
    phase_key - string
        switch for adding phase corrections to the 
        independent frequencies before stacking into the 
        supervector
    Output-
    K_samp - np ndarray
        last axis is time, first two store cov mats
    tvals - np 1darray
        first time stamp associated with the corresponding
        sample covariance matrix
    """
    tgrid = load_tgrid()
    tgrid = tgrid[:1-num_ranges]

    """ Get number of dvecs in each cov estimation"""
    frame_len = get_frame_len(cov_int_time)
    print('num dvecs used in covariance estimation', frame_len)
    num_frames = get_num_frames(tgrid, frame_len)
    print('total number of chunks analyzed', num_frames)
    print('time spacing of cov estimates', tgrid[frame_len] - tgrid[0])

    """ Load up the relevant dvecs """
    dvecs = load_dvec(freq)
    num_rcvrs = dvecs.shape[0]
    dvecs = dvecs / np.linalg.norm(dvecs, axis=0)
    dvecs = range_stack_dvecs(dvecs, freq, phase_key=phase_key, num_ranges=num_ranges)
    print('supervector dims', dvecs.shape)
    print('expected to be', num_ranges*num_rcvrs, len(tgrid))
    
    """ Initialize sample covariance array """
    K_samp = np.zeros((num_rcvrs*num_ranges, num_rcvrs*num_ranges, num_frames), dtype=np.complex128)
    """ Iterate through frames and add sample covs to the mat """
    for i in range(num_frames):
        inds = slice(i*frame_len, (i+1)*frame_len)
        tmp_K = np.cov(dvecs[:,inds])
        K_samp[:,:,i] = tmp_K

    """ Save it """
    fname = make_range_super_cov_name(freq, num_ranges, phase_key=phase_key)
    tvals = tgrid[::frame_len]
    print(tvals.shape, K_samp.shape)
    tvals = tvals[:num_frames]
    np.save(fname, K_samp)
    tname = make_cov_time_name(range_stack=True)
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

def load_super_cov(phase_key='naive', proj_root=PROJ_ROOT):
    """ Load up cov estimates
    for source frequency freq 
    Output 
    tvals - np 1darray of floats
        times associated with the left-hand side
        of the data chunk used in the cov estim.  
    K_samp - np ndarray 
        sequence of cov estimates for the data
    """
    fname = make_super_cov_name(phase_key=phase_key, proj_root=proj_root)
    tname = make_cov_time_name(proj_root=proj_root)
    K_samp = np.load(fname)
    tvals = np.load(tname)
    return tvals, K_samp

def load_range_super_cov(freq, num_ranges, phase_key='naive', proj_root=PROJ_ROOT):
    """ Load up cov estimates
    for source frequency freq 
    Output 
    tvals - np 1darray of floats
        times associated with the left-hand side
        of the data chunk used in the cov estim.  
    K_samp - np ndarray 
        sequence of cov estimates for the data
    """
    fname = make_range_super_cov_name(freq, num_ranges, phase_key=phase_key, proj_root=PROJ_ROOT)
    tname = make_cov_time_name(proj_root=proj_root, range_stack=True)
    K_samp = np.load(fname)
    tvals = np.load(tname)
    tvals = tvals[:1-num_ranges]
    return tvals, K_samp
    
cov_int_time = 10 # number of seconds to integrate to form sample cov

if __name__ == '__main__':
    #for freq in freqs:
    #    make_cov_mat_seq(freq, cov_int_time)

    make_super_cov_mat_seq(freqs, cov_int_time, 'naive')
    make_super_cov_mat_seq(freqs, cov_int_time, 'source_correct')
    make_super_cov_mat_seq(freqs, cov_int_time, 'MP_norm')



