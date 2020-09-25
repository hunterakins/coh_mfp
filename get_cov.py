import numpy as np
from matplotlib import pyplot as plt
#from coh_mfp.config import freqs, fs, source_vel, fft_spacing, PROJ_ROOT, num_realizations
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

def make_cov_name(freq, sim_iter, proj_root,super_type, **kwargs):
    if super_type =='none':
        name = proj_root + str(freq) + '_cov' + str(sim_iter) + '.npy' 
    elif super_type == 'range':
        print(kwargs)
        num_ranges = kwargs['num_ranges'] 
        phase_key = kwargs['phase_key']
        name = make_range_super_cov_name(freq, num_ranges, sim_iter, proj_root, phase_key) 
    elif super_type == 'freq':
        phase_key = kwargs['phase_key']
        name = make_super_cov_name(sim_iter, proj_root, phase_key)
    else:
        raise ValueError('Incorrect super_type key supplied')
    return name

def make_super_cov_name(sim_iter, proj_root, phase_key='naive'):
    name = proj_root + phase_key +  '_supercov' + str(sim_iter) + '.npy' 
    return name

def make_range_super_cov_name(freq, num_ranges, sim_iter, proj_root, phase_key='naive'):
    name = proj_root + str(freq) + '_' + str(num_ranges) + '_' + phase_key +  '_supercov' + str(sim_iter) + '.npy' 
    return name

def make_cov_time_name(proj_root):
    name = proj_root + 'cov_t.npy'   
    return name

def get_frame_len(cov_int_time,fft_spacing, fs):
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

def stack_dvecs(dvec_list, proj_root, phase_key='naive', freqs=None):
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
    tgrid = load_tgrid(proj_root)
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

def range_stack_dvecs(dvecs, freq, proj_root, source_vel, fft_spacing, fs, phase_key='naive', num_ranges=2):
    """
    Similar to stack_dvecs, but instead of stacking in frequency, 
    stack in range
    Input 
    dvec_list - list of numpy ndarrays
        each element is a dvec (see make_super_cov_mat_seq)
    freq - int 
        source frequency, used to get the lambda / 2 range spacing
    phase_key - string
        optional key to specify a phase correction/normalization
    num_ranges - int
        how many lambda/2 spacings to stack
    Output -
    super_dvecs - numpy nd array
        first two axis are virtual receiver index
        (num_rcvrs * num_ranges)
        last axis is time (set by fft spacing) in conf
    """
    tgrid = load_tgrid(proj_root)
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

def get_K_samp(dvecs, num_frames, frame_len):
    """
    Form sample covariance matrix for dvecs
    Input 
    dvecs - numpy 2d array
        First axis is receiver index (note that
        in the case of some sort of coherent range
        or frequency stacking, the receiver index
        becomes the index for a virtual array
    num_frames - int
        number of data snapshots in the entire time record
    frame_len - int
        number of data vectors used in each frame
    Output-
    K_samp - numpy ndarrayt
        first two axes are receiver index, last axis is time
    t
    """

    num_rcvrs = dvecs.shape[0]
    """ Initialize sample covariance array """
    K_samp = np.zeros((num_rcvrs, num_rcvrs, num_frames), dtype=np.complex128)
    """ Iterate through frames and add sample covs to the mat """
    for i in range(num_frames):
        inds = slice(i*frame_len, (i+1)*frame_len)
        tmp_K = np.cov(dvecs[:,inds])
        K_samp[:,:,i] = tmp_K

def get_freq_dvec_list(freqs, sim_iter, proj_root):
    """
    Form a list of the dvecs for each frequency in freqs
    Input
    freqs - list of ints
        source frequencies to evaluate
    sim_iter - integer
       noise realization identifier
    proj_root - string  
        the project root storage location (conf.proj_root)
    Output -
    dvec_list - list of numpy ndarrays
        ith element is the dvecs for the ith frequecy in freqs
    """
    dvec_list = []
    for freq in freqs:
        freq_dvecs = load_dvec(freq, sim_iter, proj_root)
        freq_dvecs = freq_dvecs / np.linalg.norm(freq_dvecs, axis=0)
        dvec_list.append(freq_dvecs)
    return dvec_list

def make_cov_mat_seq(freqs,cov_int_time, sim_iter, conf, super_type, **kwargs):
    """
    For a given frequency band and a covariance
    integration time in seconds, produce a sequence
    of covariance estimates with no overlap
    Save the cov estimates in a numpy ndarray,
    the last axis is time
    Input
    freqs - list of ints
        source frequencies to analyze
    sim_iter - int
        integer identifying which noise realization is under consideration
    cov_int_time - float
        integration time in seconds
    conf - ExpConf object
    super_type - string key
    **kwargs - dict pointer
        optional dict args for the supervector processors
    Output-
    K_samp - np ndarray
        last aaxis is time, first two store cov mats
    tvals - np 1darray
        first time stamp associated with the corresponding
        sample covariance matrix
    """
    proj_root = conf.proj_root
    tgrid = load_tgrid(proj_root) # time stamps associated with beginning of fft windows

    frame_len = get_frame_len(cov_int_time, conf.fft_spacing, conf.fs)
    num_frames = get_num_frames(tgrid, frame_len)

    if super_type=='none':
        freq = freqs[0]
        dvecs = load_dvec(freq, sim_iter, proj_root)
        dvecs = dvecs / np.linalg.norm(dvecs, axis=0)
    elif super_type == 'range':
        freq = freqs[0]
        num_ranges = kwargs['num_ranges']
        phase_key=kwargs['phase_key']
        print('num ranges', num_ranges)
        print('phase_key', phase_key)
        dvecs = load_dvec(freq, sim_iter, proj_root)
        dvecs = dvecs / np.linalg.norm(dvecs, axis=0)
        dvecs = range_stack_dvecs(dvecs, freq, proj_root, conf.source_vel, conf.fft_spacing, conf.fs, phase_key=phase_key, num_ranges=num_ranges)
    elif super_type == 'freq':
        dvec_list = get_freq_dvec_list(freqs, sim_iter, proj_root)
        dvecs = stack_dvecs(dvec_list, phase_key=phase_key, freqs=freqs)
    else: 
        raise ValueError('Incorrect super_type key supplied. Options are none, range, and freq. You supplied ' + super_type)
        
    K_samp = get_K_samp(dvecs, num_frames, frame_len)

    """ Save it """
    fname = make_cov_name(freq, sim_iter, conf.proj_root, super_type, **kwargs)
    tvals = tgrid[::frame_len]
    tvals = tvals[:num_frames]
    np.save(fname, K_samp)
    tname = make_cov_time_name(conf.proj_root)
    np.save(tname, tvals)
    return tvals, K_samp

#def make_super_cov_mat_seq(freq, cov_int_time, sim_iter, phase_key='naive'):
#    """
#    For the frequency bands inf freqs and a covariance
#    integration time in seconds, produce a sequence
#    of covariance estimates with no overlap
#    Save the cov estimates in a numpy ndarray,
#    the last axis is time
#    phase_key allows for selection of some phase correction/
#    normalization business 
#    Input
#    freqs - list of ints or just an int
#        list of source frequencies to analyze and stack
#    cov_int_time - float
#        integration time in seconds
#    phase_key - string
#        switch for adding phase corrections to the 
#        independent frequencies before stacking into the 
#        supervector
#    Output-
#    K_samp - np ndarray
#        last axis is time, first two store cov mats
#    tvals - np 1darray
#        first time stamp associated with the corresponding
#        sample covariance matrix
#    """
#    tgrid = load_tgrid(proj_root)
#
#    """ Get number of dvecs in each cov estimation"""
#    frame_len = get_frame_len(cov_int_time)
#    num_frames = get_num_frames(tgrid, frame_len)
#    num_freqs = len(freqs)
#
#    get_freq_dvec_list(freqs, sim_iter, proj_root)
#    dvecs = stack_dvecs(dvec_list, phase_key=phase_key, freqs=freqs)
#    K_samp = get_K_samp(dvecs, num_frames, frame_len)
#
#    fname = make_super_cov_name(phase_key=phase_key)
#
#    tvals = tgrid[::frame_len]
#    tvals = tvals[:num_frames]
#    np.save(fname, K_samp)
#    tname = make_cov_time_name()
#    np.save(tname, tvals)
#    return tvals, K_samp

#def make_range_super_cov_mat_seq(freq, cov_int_time, sim_iter, proj_root, phase_key='naive', num_ranges=2):
#    """
#    For the covariance
#    integration time in seconds, produce a sequence
#    of covariance estimates with no overlap, using data vectors
#    that stack in range using as many as num_ranges
#    Save the cov estimates in a numpy ndarray,
#    the last axis is time
#    phase_key allows for selection of some phase correction/
#    normalization business 
#    Input
#    freqs - list of ints
#        list of source frequencies to analyze and stack
#    cov_int_time - float
#        integration time in seconds
#    sim_iter - int
#        key for simulation index
#    proj_root - string
#        save location for cov mats
#    phase_key - string
#        switch for adding phase corrections to the 
#        independent frequencies before stacking into the 
#        supervector
#    Output-
#    K_samp - np ndarray
#        last axis is time, first two store cov mats
#    tvals - np 1darray
#        first time stamp associated with the corresponding
#        sample covariance matrix
#    """
#    tgrid = load_tgrid(proj_root)
#    tgrid = tgrid[:1-num_ranges]
#
#    """ Get number of dvecs in each cov estimation"""
#    frame_len = get_frame_len(cov_int_time)
#    print('num dvecs used in covariance estimation', frame_len)
#    num_frames = get_num_frames(tgrid, frame_len)
#    print('total number of chunks analyzed', num_frames)
#    print('time spacing of cov estimates', tgrid[frame_len] - tgrid[0])
#
#    """ Load up the relevant dvecs """
#    dvecs = load_dvec(freq, sim_iter, proj_root)
#    num_rcvrs = dvecs.shape[0]
#    dvecs = dvecs / np.linalg.norm(dvecs, axis=0)
#    dvecs = range_stack_dvecs(dvecs, freq, phase_key=phase_key, num_ranges=num_ranges)
#    print('supervector dims', dvecs.shape)
#    print('expected to be', num_ranges*num_rcvrs, len(tgrid))
#    
#    """ Initialize sample covariance array """
#    K_samp = np.zeros((num_rcvrs*num_ranges, num_rcvrs*num_ranges, num_frames), dtype=np.complex128)
#    """ Iterate through frames and add sample covs to the mat """
#    for i in range(num_frames):
#        inds = slice(i*frame_len, (i+1)*frame_len)
#        tmp_K = np.cov(dvecs[:,inds])
#        K_samp[:,:,i] = tmp_K
#
#    """ Save it """
#    fname = make_range_super_cov_name(freq, num_ranges, sim_iter, phase_key=phase_key)
#    tvals = tgrid[::frame_len]
#    print(tvals.shape, K_samp.shape)
#    tvals = tvals[:num_frames]
#    np.save(fname, K_samp)
#    tname = make_cov_time_name(range_stack=True)
#    np.save(tname, tvals)
#    return tvals, K_samp

def load_cov(freq, sim_iter, proj_root, super_type, **kwargs):
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
    fname = make_cov_name(freq, sim_iter,  proj_root, super_type, **kwargs)
    tname = make_cov_time_name(proj_root)
    K_samp = np.load(fname)
    tvals = np.load(tname)
    return tvals, K_samp

    
cov_int_time = 10 # number of seconds to integrate to form sample cov

if __name__ == '__main__':
    print('nothin here...')
