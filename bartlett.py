import numpy as np
import sys
from matplotlib import pyplot as plt
from coh_mfp.sim import get_sim_folder, make_sim_name
from coh_mfp.config import source_vel, freqs, num_ranges, r0, fft_spacing, fs, SNR, PROJ_ROOT
from coh_mfp.get_cov import load_cov, load_super_cov, load_range_super_cov
from pyat.pyat.readwrite import read_shd
import time
'''
Description:
Perform Bartlett processing on sequence of covariance estimates
Save a pic for each one and form a movie of the ambiguity surfaces

Date: 
8/28/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''



def load_replicas(freq):
    """
    Load the replicas from sim.py for source frequency freq
    input
    freq - int or float
    output 
    pfield - np ndarray
        num_rcvrs, num_depths, num_ranges = pfield.shape
    pos - pyat Pos object
        correpsonding source positions
    """
    fname = get_sim_folder() + make_sim_name(freq) + '.shd'
    [x,x,x,x, pos, pfield] = read_shd(fname)
    pfield = np.squeeze(pfield)
    return pfield, pos

def load_super_replicas(freqs):
    """
    For multi-frequency coherent processing,
    I need to stack my replicas into a 'supervector'
    Input 
    freqs - list of frequencies
    Output 
    """ 
    num_freqs = len(freqs)
    pfield, pos = load_replicas(freqs[0])
    num_rcvrs, num_depths, num_ranges = pfield.shape
    super_pfield = np.zeros((num_rcvrs*num_freqs, num_depths, num_ranges), dtype=np.complex128)
    super_pfield[:num_rcvrs, :, :] = pfield[:,:,:]
    for i in range(1, num_freqs):
        pfield, pos = load_replicas(freqs[i])
        super_pfield[i*num_rcvrs:(i+1)*num_rcvrs, :, :] = pfield[:,:,:]
    return super_pfield, pos

def load_range_super_replicas(freq, num_range_stack):
    """
    For multi-range coherent processing,
    I need to stack my replicas into a 'supervector'
    Input 
    freq - int
    num_range_stack - int 
        number of ranges used in stacking
    Output 
    """ 
    pfield, pos = load_replicas(freq)
    num_rcvrs, num_depths, num_ranges = pfield.shape
    super_pfield = 1e-17*np.ones((num_rcvrs*num_range_stack, num_depths, num_ranges), dtype=np.complex128)
    lam = 1500 / freq
    replica_dr = source_vel * fft_spacing/fs
    stride = int(lam/2 / replica_dr)
    print('stride', stride)
    for i in range(num_range_stack):
        rel_pfield = pfield[:,:,i*stride:]
        super_pfield[i*num_rcvrs:(i+1)*num_rcvrs,:,:rel_pfield.shape[2]] = rel_pfield
    return super_pfield, pos

def plot_single_snapshot_amb(pos, tvals, bartlett, int_id, save_id):
    """ Ploat the ambiguity surface on db scale
    and save figure to pics/ with the integer id
    """
    fig = plt.figure()
    print('bartlett_max', np.max(abs(bartlett)))
    b_db = np.log10(abs(bartlett)/np.max(abs(bartlett)))
    max_loc = np.argmax(abs(bartlett))
    num_ranges = pos.r.range.size
    max_depth = max_loc // num_ranges
    max_range = max_loc % num_ranges

    levels = np.linspace(-2, 0, 10)
    CS = plt.contourf(pos.r.range, pos.r.depth, b_db, levels=levels)
    plt.suptitle('SNR: ' + str(SNR) + ' db')
    plt.plot([r0 + source_vel*tvals[int_id]], [54], 'b+')
    plt.plot(pos.r.range[max_range], pos.r.depth[max_depth], 'r+')
    plt.xlabel('Range (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar()
    plt.savefig('pics/' + save_id +  str(int_id).zfill(3) + '.png')
    plt.close(fig)
    return

def bartlett(tvals, K_samp, pfield, pos, save_id=''):
    """
    Form ambiguity surface for bartlett processor
    Input
    tvals - np 1darray
        beginning times of the snapshots
    K_samp - np ndarray
        first two dims are the covariance mat for the ith snapshot
        num_rcvrs, num_rcvrs, num_snapshots = np.shape(K_samp)
    pfield - np nadarray
        grid of replicas. The first axis is the receiver dimension
        Second axis is source depth, third (final) axis is range)
        So pfield[0,1,2] is the replica for the first phone from the 
        second source depth and the the third range
    pos - Pos object (pyat)
        Grid of source positions for the replicas
    Output  
    bartlett - np ndarray
        first axis is source depths, second is source range,
        final axis is beginning time of snapshot
    """
    num_rcvrs = pfield.shape[0]
    num_depths = pfield.shape[1]
    num_ranges = pfield.shape[2]
    num_positions = num_depths*num_ranges
    pfield = pfield.reshape(num_rcvrs, num_positions)
    pfield /= np.linalg.norm(pfield, axis=0)
    output = np.zeros((num_depths, num_ranges, tvals.size))
    for i in range(tvals.size):
        right_prod = K_samp[:,:,i]@pfield
        full_prod = pfield.conj()*right_prod
        power = np.sum(full_prod, axis=0)
        power = power.real
        power = power.reshape(num_depths, num_ranges)
        output[:,:,i] = power# there might be a  complex part from roundoff err.
        plot_single_snapshot_amb(pos, tvals, output[:,:,i], i, save_id)
    return output

def get_bartlett_name(freq,proj_root=PROJ_ROOT):
    fname = proj_root + str(freq) + 'bartlett.npy'
    return fname

def get_super_bartlett_name(proj_root=PROJ_ROOT, phase_key='naive'):
    fname = proj_root + 'super_bart_' + phase_key + '.npy'
    return fname

def get_range_super_bartlett_name(freq, num_ranges,proj_root=PROJ_ROOT):
    fname = proj_root + 'range_super_bart_' + str(freq) + '_' + str(num_ranges) + '.npy'
    return fname

def get_amb_surf(freq):
    pfield, pos = load_replicas(freq)
    tvals, K_samp = load_cov(freq)
    output = bartlett(tvals, K_samp, pfield, pos, str(freq) + '_')
    fname = get_bartlett_name(freq)
    np.save(fname, output)
    b_db = np.log10(abs(output)/np.max(abs(output)))
    return b_db

def load_bartlett(freq, proj_root=PROJ_ROOT):
    fname =  get_bartlett_name(freq, proj_root)
    x = np.load(fname)
    return x

def get_super_bart(phase_key):
    tvals, super_samp = load_super_cov(phase_key=phase_key)
    output = bartlett(tvals, super_samp, pfield, pos, save_id=phase_key + '_')
    fname = get_super_bartlett_name(phase_key=phase_key)
    np.save(fname, output)
    return output

def get_range_super_bart(freq, num_ranges):
    pfield, pos = load_range_super_replicas(freq, num_ranges)
    tvals, super_samp = load_range_super_cov(freq, num_ranges, phase_key='source_correct')
    output = bartlett(tvals, super_samp, pfield, pos, save_id=str(freq) + '_range_super')
    fname = get_range_super_bartlett_name(freq, num_ranges)
    np.save(fname, output)
    return output

def get_incoh_name(proj_root=PROJ_ROOT):
    return proj_root+'incoh_bartlett.npy'

def incoh_bart(freqs):
    """ Form an incoherent db summation
    of the ambiguity surfaces of the different 
    frequencies """

    f0 = freqs[0]
    x0 = load_bartlett(f0)
    pfield, pos = load_replicas(f0)
    pfield = 0
    tvals, K_samp = load_cov(f0)
    incoh_sum = np.zeros(x0.shape)
    x0_db = 10*np.log10(abs(x0)/np.max(abs(x0), axis=(0,1)))
    incoh_sum += x0_db
    for f in freqs[1:]:
        tmp = load_bartlett(f)
        tmp_db = 10*np.log10(abs(tmp)/np.max(abs(tmp),axis=(0,1)))
        incoh_sum += tmp_db
    incoh_sum /= len(freqs)
    fname = get_incoh_name()
    np.save(fname, incoh_sum)

    """ Now plot it """
    for i in range(tvals.size):
        num_ranges = pos.r.range.size
        max_loc = np.argmax(incoh_sum[:,:,i])
        max_depth = max_loc // num_ranges
        max_range = max_loc % num_ranges
        fig = plt.figure()
        levels = np.linspace(-20, 0, 10)
        CS = plt.contourf(pos.r.range, pos.r.depth, incoh_sum[:,:,i], levels=levels)
        plt.plot([r0 + source_vel*tvals[i]], [54], 'b+')
        plt.plot(pos.r.range[max_range], pos.r.depth[max_depth], 'r+')
        plt.colorbar()
        plt.savefig('pics/incoh_sum_' + str(i).zfill(3) +'.png')
        plt.close(fig)
    return incoh_sum


if __name__ == '__main__':
    now = time.time()
#
    for freq in freqs:
        print('freq', freq)
        get_amb_surf(freq)
#
#    incoh_bart(freqs)
#    print('elapsed time', time.time() - now)
#    
#    
#    pfield, pos = load_super_replicas(freqs)
#    for phase_key in ['source_correct', 'MP_norm', 'naive']:
#        now = time.time()
#        get_super_bart(phase_key)
#        print('elapsed time ', phase_key,  time.time() - now)
#
    for freq in freqs:
        print('freq', freq)
        get_range_super_bart(freq, num_ranges)
    sys.exit(0)
