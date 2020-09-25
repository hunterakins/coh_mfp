import numpy as np
import pickle
import sys, os
from matplotlib import pyplot as plt
from coh_mfp.sim import get_sim_folder, make_sim_name
from coh_mfp.config import source_vel, freqs, num_ranges, r0, fft_spacing, fs, SNR, PROJ_ROOT, zs, num_realizations, ship_dr, dr, fig_folder
from coh_mfp.get_cov import load_cov, load_super_cov, load_range_super_cov
from signal_proc.mfp.wnc import run_wnc
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
    Pick out replicas at lambda / 2
    Stack them.
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

def plot_single_snapshot_amb(pos, tvals, bartlett, int_id, fig_leaf):
    """ Ploat the ambiguity surface on db scale
    and save figure to pics/ with the integer id
    """
    fig = plt.figure()
    print('bartlett_max', np.max(abs(bartlett)), fig_leaf)
    b_db = np.log10(abs(bartlett)/np.max(abs(bartlett)))
    max_loc = get_max_locs(b_db)
    max_depth = max_loc[0,0]
    max_range = max_loc[1,0]

    levels = np.linspace(-2, 0, 10)
    CS = plt.contourf(pos.r.range, pos.r.depth, b_db, levels=levels)
    plt.suptitle('SNR: ' + str(SNR) + ' db')
    plt.plot([r0 + source_vel*tvals[int_id]], [zs], 'b+')
    plt.plot(pos.r.range[max_range], pos.r.depth[max_depth], 'r+')
    plt.xlabel('Range (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar()
    plt.savefig(fig_folder + fig_leaf +  str(int_id).zfill(3) + '.png')
    plt.close(fig)
    return

def plot_amb_series(pos, tvals, bf_out, fig_leaf):
    """
    Plot the sequence of ambiguity surfaces for 
    a beamformer
    Input
    pos - Pos object
    tvals - np array 
        time values of beginning of snapshot
    bf_out - np 3d array
        first two axes are beamforer power
        last axis is time
    fig_leaf - string
        specific figure location for this amb. surface
        sequence (trailing /)
    """
    for i in range(tvals.size):
        curr_out = bf_out[:,:,i]
        plot_single_snapshot_amb(pos, tvals, curr_out, i, fig_leaf)
    return

def bartlett(K_samp, pfield):
    """
    Form ambiguity surface for bartlett processor
    Input
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
    num_times = K_samp.shape[-1]
    num_rcvrs = pfield.shape[0]
    num_depths = pfield.shape[1]
    num_ranges = pfield.shape[2]
    num_positions = num_depths*num_ranges
    pfield = pfield.reshape(num_rcvrs, num_positions)
    pfield /= np.linalg.norm(pfield, axis=0)
    output = np.zeros((num_depths, num_ranges, num_times))
    for i in range(num_times):
        right_prod = K_samp[:,:,i]@pfield
        full_prod = pfield.conj()*right_prod
        power = np.sum(full_prod, axis=0)
        power = power.real
        power = power.reshape(num_depths, num_ranges)
        output[:,:,i] = power# there might be a  complex part from roundoff err.
    return output

def get_bartlett_name(freq,sim_iter, proj_root=PROJ_ROOT):
    fname = proj_root + str(freq) + 'bartlett_' + str(sim_iter) + '.npy'
    return fname

def get_super_bartlett_name(sim_iter, proj_root=PROJ_ROOT, phase_key='naive'):
    fname = proj_root + 'super_bart_' + phase_key + '_' + str(sim_iter) + '.npy'
    return fname

def get_range_super_bartlett_name(freq, num_ranges, sim_iter, proj_root=PROJ_ROOT):
    fname = proj_root + 'range_super_bart_' + str(freq) + '_' + str(num_ranges) + '_' + str(sim_iter) +  '.npy'
    return fname

def get_amb_surf(freq, sim_iter):
    pfield, pos = load_replicas(freq)
    tvals, K_samp = load_cov(freq, sim_iter)
    output = bartlett(tvals, K_samp, pfield, pos, str(freq) + '_')
    fname = get_bartlett_name(freq, sim_iter)
    np.save(fname, output)
    b_db = np.log10(abs(output)/np.max(abs(output)))
    return b_db

def load_bartlett(freq, sim_iter=0, proj_root=PROJ_ROOT):
    fname =  get_bartlett_name(freq, sim_iter, proj_root)
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

def get_incoh_name(sim_iter, proj_root=PROJ_ROOT):
    return proj_root+'incoh_bartlett' + '_' + str(sim_iter) + '.npy'

def incoh_bart(freqs, sim_iter):
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
    for f in freqs:
        tmp = load_bartlett(f, sim_iter)
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
        plt.plot([r0 + source_vel*tvals[i]], [zs], 'b+')
        plt.plot(pos.r.range[max_range], pos.r.depth[max_depth], 'r+')
        plt.colorbar()
        plt.savefig(fig_folder + '/incoh_sum_' + str(i).zfill(3) +'.png')
        plt.close(fig)
    return incoh_sum

def get_mc_name(freq, proc_key):
    return proj_root + str(freq) + '_' + proc_key + '_mcout.pickle'
    

class MCOutput:
    def __init__(self, freq,  tvals, true_range, true_depth, pos, SNR, num_realizations, sim_outputs, proc_key):
        """
        Save output of Monte Carlo simulation
        Input -
        freq - int (or float)
            source freq
        tvals - np 1darray
            time value at start of each snapshot 
        true_range - np 1darray
            position at the beginning of each snapshot (meters)
        true_depth - float
            depth of source
        pos - Pos object 
            each position in Pos is a replica location
        SNR - float (or potentially int)
            SNR of the realizations (see config)
        num_realizations - int
            number of simulation runs 
        sim_outputs - list of SimOutput objs
            save the tracking results of the simulation runs
        """
        self.freq = freq
        self.tvals = tvals
        self.true_range = true_range
        self.true_depth = true_depth
        self.pos = pos
        self.SNR = SNR
        self.num_realizations = num_realizations
        self.sim_outputs = sim_outputs
        self.proc_key= proc_key
        return

    def save(self):
        """
        Save pickled version of self 
        """
        name = get_mc_name(self.freq, self.proc_key)
        with open(name, 'wb') as f:
            pickle.dump(self, f)
        return

def get_sim_out_name(sim_iter, proj_root=PROJ_ROOT):
    return proj_root + 'sim_output_'+str(sim_iter) + '.pickle'

class SimOutput:
    def __init__(self, sim_iter, max_locs):
        """
        Save output of a single simulation run 
        Input 
        sim_iter - integer
            index of specific sim realization
        max_locs - np ndarray
            each column is a single time
            first row is INDEX of depth max location in bart amb sufra
            second is INDEX of range of max location in bart amb surface
            To get the actual location, get pos from MCOutput and the 
            location for time i is (pos.r.range[max_locs[1,i]], pos.r.depth[max_locs[0,i]])
        """
        self.sim_iter = sim_iter    
        self.max_locs = max_locs
        return

    def save(self, proj_root=PROJ_ROOT):
        name = get_sim_out_name(self.sim_iter, proj_root)
        with open(name, 'wb') as f:
            pickle.dump(self, f)

def get_max_locs(bart_out):
    """
    Get 2d array of argmax of bartlett power surface
    Input 
    bart_out - numpy 3d array
        last axis is "time". if dims are only 2, assume it'sa single time
        first two are depth and range
    Output - max_locs
        numpy 2d array
        first row is the index of the depth
        second is index of the range
    """
    print(bart_out.shape)
    if len(bart_out.shape) == 2:
        num_time = 1
        num_depth, num_range = bart_out.shape
    else:
        num_depth, num_range, num_time = bart_out.shape
    bart_out = np.reshape(bart_out, (num_depth*num_range, num_time), order='F')
    flattened_max = np.argmax(bart_out, axis=0)
    max_locs = np.zeros((2, num_time), dtype=int)
    max_locs[0,:] = (flattened_max % num_depth).astype(int)
    max_locs[1,:] = (flattened_max // num_depth).astype(int)
    return max_locs

def get_fig_leaf(freq, sim_iter, proc, fig_root):
    fig_leaf= str(freq) +'/'+proc + '/' + str(sim_iter) + '/'
    if str(freq) not in os.listdir(fig_root):
        print('making freq folder', fig_root+fig_leaf)
        os.mkdir(fig_root + str(freq))
    if proc not in os.listdir(fig_root + str(freq)):
        os.mkdir(fig_root + str(freq) +'/'+ proc)
    if str(sim_iter) not in os.listdir(fig_root + str(freq) + '/' + proc):
        os.mkdir(fig_root + fig_leaf)
    return fig_leaf
    
def MCBartlett(freq):
    """
    Run Bartlett on the realizations of the dvecs
    Save a MCOutput object
    Input - 
    freq - int 
        source freq
    Output
    mc_out - MCOutput obj
    """
    sim_outs = []
    plt.figure()
    for sim_iter in range(num_realizations):
        pfield, pos = load_replicas(freq)
        stride = int(dr // ship_dr)
        pfield = pfield[:,:,::stride]
        print('stride', stride, 'pfield dims', pfield.shape)
        pos.r.range = pos.r.range[::stride]
        tvals, K_samp = load_cov(freq, sim_iter)
        output = bartlett(K_samp, pfield)
        fig_leaf = get_fig_leaf(freq, sim_iter, 'bart', fig_folder)
        plot_amb_series(pos, tvals, output, fig_leaf)
        #output = run_wnc(K_samp, pfield, -2)
        #fig_leaf = get_fig_leaf(freq, sim_iter, 'wnc', fig_folder)
        #plot_amb_series(pos, tvals, output, fig_leaf)
        max_locs = get_max_locs(output)
        #best_depth = pos.r.depth[max_locs[0,:]]
        #best_range = pos.r.range[max_locs[1,:]]
        #plt.plot(tvals, best_range, color='g')
        sim_out = SimOutput(sim_iter, max_locs)
        sim_outs.append(sim_out)
    true_range = r0 + source_vel*tvals
    true_depth = zs
    mc_out = MCOutput(tvals, true_range, true_depth, pos, SNR, num_realizations, sim_outs, 'wnc')
    mc_out.save()
    return
        

if __name__ == '__main__':
    now = time.time()
#
    for freq in freqs:
        print('freq', freq)
        #get_amb_surf(freq)
        MCBartlett(freq)
        sys.exit(0)
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
