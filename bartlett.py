import numpy as np
import itertools
import pickle
import sys, os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from coh_mfp.sim import get_sim_folder, make_sim_name
#from coh_mfp.config import source_vel, freqs, num_ranges, r0, fft_spacing, fs, SNR, PROJ_ROOT, zs, num_realizations, ship_dr, dr, fig_folder
from coh_mfp.config import ExpConf, load_config
from coh_mfp.get_cov import load_cov
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc
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



def load_replicas(freq, conf):
    """
    Load the replicas from sim.py for source frequency freq
    input
    freq - int or float
    conf - ExpConf object
    output 
    pfield - np ndarray
        num_rcvrs, num_depths, num_ranges = pfield.shape
    pos - pyat Pos object
        correpsonding source positions
    """
    fname = get_sim_folder(conf.proj_root) + make_sim_name(freq) + '.shd'
    [x,x,x,x, pos, pfield] = read_shd(fname)
    pfield = np.squeeze(pfield)
    return pfield, pos

def load_super_replicas(freqs, conf):
    """
    For multi-frequency coherent processing,
    I need to stack my replicas into a 'supervector'
    Input 
    freqs - list of frequencies
    Output 
    """ 
    num_freqs = len(freqs)
    pfield, pos = load_replicas(freqs[0], conf)
    num_rcvrs, num_depths, num_ranges = pfield.shape
    super_pfield = np.zeros((num_rcvrs*num_freqs, num_depths, num_ranges), dtype=np.complex128)
    super_pfield[:num_rcvrs, :, :] = pfield[:,:,:]
    for i in range(1, num_freqs):
        pfield, pos = load_replicas(freqs[i], conf)
        super_pfield[i*num_rcvrs:(i+1)*num_rcvrs, :, :] = pfield[:,:,:]
    return super_pfield, pos

def load_range_super_replicas(freq, conf):
    """
    For multi-range coherent processing,
    I need to stack my replicas into a 'supervector'
    Pick out replicas at lambda / 2
    Stack them.
    Dealing with the edge is annoying...
    Input 
    freq - int
    conf - ExpConf object
    Output 
    """ 
    num_range_stack = conf.num_ranges
    pfield, pos = load_replicas(freq, conf)
    num_rcvrs, num_depths, num_ranges = pfield.shape
    super_pfield = 1e-17*np.ones((num_rcvrs*num_range_stack, num_depths, num_ranges), dtype=np.complex128)
    lam = 1500 / freq
    replica_dr = conf.source_vel * conf.fft_spacing/conf.fs
    stride = int(lam/2 / replica_dr)
    print('stride', stride)
    for i in range(num_range_stack):
        rel_pfield = pfield[:,:,i*stride:]
        if i == 0:
            min_pfield_len = rel_pfield.shape[2]
        else:
            if rel_pfield.shape[2]<min_pfield_len:
                min_pfield_len = rel_pfield.shape[2] 
        super_pfield[i*num_rcvrs:(i+1)*num_rcvrs,:,:rel_pfield.shape[2]] = rel_pfield
    return super_pfield, pos

def plot_single_snapshot_amb(pos, tvals, bartlett, int_id, fig_leaf, conf):
    """ Ploat the ambiguity surface on db scale
    and save figure to pics/ with the integer id
    """
    r0 = conf.r0
    source_vel = conf.source_vel
    fig_folder = conf.fig_folder
    SNR = conf.SNR
    zs = conf.zs 
    fig = plt.figure()
    
    print('bartlett_max', np.max(abs(bartlett)), fig_leaf)
    b_db = np.log10(abs(bartlett)/np.max(abs(bartlett)))
    max_loc = get_max_locs(b_db)
    max_depth = max_loc[0,0]
    max_range = max_loc[1,0]

    levels = np.linspace(-2, 0, 10)
    CS = plt.contourf(pos.r.range, pos.r.depth, b_db, levels=levels)
    plt.suptitle('SNR: ' + str(SNR) + ' db' + ', ' + fig_leaf)
    plt.plot([r0 + source_vel*tvals[int_id]], [zs], 'b+')
    plt.plot(pos.r.range[max_range], pos.r.depth[max_depth], 'r+')
    plt.xlabel('Range (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar()
    plt.savefig(fig_folder + fig_leaf +  str(int_id).zfill(3) + '.png')
    plt.close(fig)
    return

def plot_amb_series(pos, tvals, bf_out, fig_leaf, conf):
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
        plot_single_snapshot_amb(pos, tvals, curr_out, i, fig_leaf, conf)
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

def get_bartlett_name(freq,sim_iter, proj_root):
    fname = proj_root + str(freq) + 'bartlett_' + str(sim_iter) + '.npy'
    return fname

def get_super_bartlett_name(sim_iter, proj_root, phase_key='naive'):
    fname = proj_root + 'super_bart_' + phase_key + '_' + str(sim_iter) + '.npy'
    return fname

def get_range_super_bartlett_name(freq, num_ranges, sim_iter, proj_root):
    fname = proj_root + 'range_super_bart_' + str(freq) + '_' + str(num_ranges) + '_' + str(sim_iter) +  '.npy'
    return fname

def get_amb_surf(freq, conf, sim_iter, super_type):
    proj_root = conj.proj_root
    pfield, pos = load_replicas(freq, conf)
    tvals, K_samp = load_cov(freq, sim_iter, proj_root, super_type)
    output = bartlett(tvals, K_samp, pfield, pos, str(freq) + '_')
    fname = get_bartlett_name(freq, sim_iter, proj_root)
    np.save(fname, output)
    b_db = 10*np.log10(abs(output)/np.max(abs(output)))
    return b_db

def load_bartlett(freq, sim_iter, proj_root):
    fname =  get_bartlett_name(freq, sim_iter, proj_root)
    x = np.load(fname)
    return x

def get_range_super_bart(freq, num_ranges):
    pfield, pos = load_range_super_replicas(freq, num_ranges)
    tvals, super_samp = load_range_super_cov(freq, num_ranges, phase_key='source_correct')
    output = bartlett(tvals, super_samp, pfield, pos, save_id=str(freq) + '_range_super')
    fname = get_range_super_bartlett_name(freq, num_ranges)
    np.save(fname, output)
    return output

def get_incoh_name(sim_iter, proj_root):
    """ jack stupy"""
    return proj_root+'incoh_bartlett' + '_' + str(sim_iter) + '.npy'

def get_mc_name(freq, proj_root):
    return proj_root + str(freq) +'_mcout.pickle'

def load_mc(freq, proj_root):
    name = get_mc_name(freq, proj_root)
    with open(name, 'rb') as f:
        mc_out = pickle.load(f)
    return mc_out

def get_pos_from_loc(pos, max_locs):
    """
    From the max_locs array which indexs the pos,
    get the range and depth array """
    ranges = pos.r.range[max_locs[1,:]]
    depths = pos.r.depth[max_locs[0,:]]
    return ranges, depths

class MCOutput:
    def __init__(self, freq, tvals, pos, conf, sim_outputs):
        """
        Save output of Monte Carlo simulation
        Input -
        freq - int (or float)
            source freq
        tvals - np 1darray
            time value at start of each snapshot 
        pos - Pos object 
            each position in Pos is a replica location
        conf - ExpConf object
            all the sim settings
        sim_outputs - list of SimOutput objs
            save the tracking results of the simulation runs
        """
        self.freq = freq
        self.tvals = tvals
        self.true_range = conf.r0 + conf.source_vel*tvals
        self.true_depth = conf.zs
        self.pos = pos
        self.SNR = conf.SNR
        self.num_realizations = conf.num_realizations
        self.sim_outputs = sim_outputs
        self.conf = conf
        self.proc_keys = list(set([x.proc_key for x in sim_outputs]))
        return

    def save(self):
        """
        Save pickled version of self 
        """
        name = get_mc_name(self.freq, self.conf.proj_root)
        with open(name, 'wb') as f:
            pickle.dump(self, f)
        return

    def make_track_plot(self, proc_key):
        true_range = self.true_range
        num_points = true_range.size
        true_depth = self.true_depth
        sim_outputs = [x for x in self.sim_outputs if x.proc_key == proc_key]
        pos = self.pos
        colors = cm.rainbow(np.linspace(0,1,num_points))
        fig = plt.figure()
        for i in range(num_points):
            true_ax = plt.plot(true_range[i], true_depth, color=colors[i], marker='o')
        marker = itertools.cycle((',', '+', '.', 'o', '*')) 
        r_mat, z_mat = get_loc_mat(self.pos, sim_outputs)
        mean_r = np.mean(r_mat, axis=0)
        mean_z = np.mean(z_mat, axis=0)
        median_r = np.median(r_mat, axis=0)
        median_z = np.median(z_mat, axis=0)
        print(mean_r.shape, mean_z.shape, median_r.shape, median_z.shape, num_points)
        for i in range(num_points-1):
            mean_ax = plt.plot(mean_r[i], mean_z[i], color=colors[i], marker='+')
            med_ax = plt.plot(median_r[i], median_z[i], color=colors[i], marker='*')
        #for i, sim in enumerate(sim_outputs): 
        #    max_locs = sim.max_locs
        #    r, z = get_pos_from_loc(pos, max_locs)
            #for j in range(num_points-1):
            #    plt.scatter(r[j], z[j], color=colors[j], marker=next(marker))
        #plt.xlim([.9*np.min(true_range), 1.1*np.max(true_range)])
        plt.legend((true_ax[0], mean_ax[0], med_ax[0]), ('True track', 'Mean track', 'Median track'))
        plt.suptitle('SNR ' + str(self.SNR) + ' , ' + proc_key)
        save_loc = self.conf.proj_root+ proc_key + '.png'
        print('save loc', save_loc)
        plt.savefig(save_loc)

    def make_realiz_scatter(self, proc_key):
        true_range = self.true_range
        num_points = true_range.size
        true_depth = self.true_depth
        sim_outputs = [x for x in self.sim_outputs if x.proc_key == proc_key]
        pos = self.pos
        colors = cm.rainbow(np.linspace(0,1,num_points))
        fig = plt.figure()
        num_points = 1
        plt.scatter(true_range[0], true_depth, color='b', marker='o')
        #marker = itertools.cycle((',', '+', '.', 'o', '*')) 
        r_mat, z_mat = get_loc_mat(self.pos, sim_outputs)
        mean_r = np.mean(r_mat, axis=0)
        mean_z = np.mean(z_mat, axis=0)
        median_r = np.median(r_mat, axis=0)
        median_z = np.median(z_mat, axis=0)
        print(mean_r.shape, mean_z.shape, median_r.shape, median_z.shape, num_points)
        num_realizations = self.num_realizations
        for i in range(num_realizations):
            plt.scatter(r_mat[i,0], z_mat[i,0], color='r', marker='*')
        plt.suptitle('SNR ' + str(self.SNR) + ' , ' + proc_key)
        save_loc = self.conf.proj_root+ proc_key + '.png'
        print('save loc', save_loc)
        plt.savefig(save_loc)

    def get_track_stats(self):
        """
        For each processor, form a mean track (over realizations)
        Compute error for each track realization
        Compute mean and variance, media as well
        """
        true_range = self.true_range
        true_depth = self.true_depth
        for key in self.proc_keys:
            sim_outputs = [x for x in self.sim_outputs if x.proc_key == proc_key]
            rmat, zmat = get_loc_mat(self.pos, sim_outputs)
        return

def get_loc_mat(pos, sim_outputs):
    """
    For a list of simulatio outputs, 
    create two numpy arrays.
    Each column contains the range estimates
    for a given time, each row corresponds
    to a single realization of the noise
    Input -
    pos - Position object
    sim_outputs - SimOutputs obj
    Output
    r_mat - np 2d array
    z_mat - np 2darray
    """
    first = False
    num_real = len(sim_outputs)
    for i, sim in enumerate(sim_outputs):
        r, z = get_pos_from_loc(pos, sim.max_locs)
        if i == 0:
            r_mat, z_mat = np.zeros((num_real, r.size)), np.zeros((num_real, z.size))
        r_mat[i,:] = r
        z_mat[i,:] = z
    return r_mat, z_mat
    
def get_sim_out_name(sim_iter, proj_root):
    return proj_root + 'sim_output_'+str(sim_iter) + '.pickle'

class SimOutput:
    def __init__(self, sim_iter, max_locs, proc_key, **kwargs):
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
        self.proc_key = proc_key
        if proc_key == 'wnc':
            self.wn_gain = kwargs['wn_gain']
        return

    def save(self, proj_root):
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
    
def MCBartlett(freq, conf):
    """
    Run Bartlett on the realizations of the dvecs
    Save a MCOutput object
    Input - 
    freq - int 
        source freq
    Output
    mc_out - MCOutput obj
    """
    source_vel = conf.source_vel
    sim_outs = []
    wn_gain = -.01

    for sim_iter in range(conf.num_realizations):
        pfield, pos = load_replicas(freq, conf)
        stride = int(conf.dr // conf.ship_dr)
        pfield = pfield[:,:,::stride]
        pos.r.range = pos.r.range[::stride]
        print('pos shape', pos.r.range.size)
    
        """ Naive covariance """
        tvals, K_samp = load_cov(freq, sim_iter, conf.proj_root, 'none')
        print('tvals size', tvals.size)

        output = bartlett(K_samp, pfield)
        print(output.shape)
        fig_leaf = get_fig_leaf(freq, sim_iter, 'bart', conf.fig_folder)
        plot_amb_series(pos, tvals, output, fig_leaf, conf)
        max_locs = get_max_locs(output)
        print('bart', max_locs.shape)
        bart_out = SimOutput(sim_iter, max_locs, 'bart')
    
        output = lookup_run_wnc(K_samp, pfield, wn_gain)
        #output = run_wnc(K_samp, pfield, wn_gain)
        fig_leaf = get_fig_leaf(freq, sim_iter, 'wnc', conf.fig_folder)
        plot_amb_series(pos, tvals, output, fig_leaf, conf)
        max_locs = get_max_locs(output)
        kwargs = {'wn_gain' : wn_gain}
        wnc_out = SimOutput(sim_iter, max_locs, 'wnc', **kwargs)

        sim_outs.append(bart_out)
        sim_outs.append(wnc_out)


        """ Range covariance """
        pfield, pos = load_range_super_replicas(freq, conf)
        pfield = pfield[:,:,::stride]
        pos.r.range = pos.r.range[::stride]
        print('pos shape', pos.r.range.size)
        kwargs = {'num_ranges':conf.num_ranges, 'phase_key': 'source_correct'}
        tvals, K_samp = load_cov(freq, sim_iter, conf.proj_root, 'range', **kwargs)
        print('tvals size', tvals.size)
        #tvals, K_samp = tvals[::5], K_samp[:,:,::5]
        output = bartlett(K_samp, pfield)
        fig_leaf = get_fig_leaf(freq, sim_iter, 'bart_range', conf.fig_folder)
        plot_amb_series(pos, tvals, output, fig_leaf, conf)
        max_locs = get_max_locs(output)
        print('range', max_locs.shape)
        bart_out = SimOutput(sim_iter, max_locs, 'bart_range')
    
        output = lookup_run_wnc(K_samp, pfield, wn_gain)
        #output = run_wnc(K_samp, pfield, wn_gain)
        fig_leaf = get_fig_leaf(freq, sim_iter, 'wnc_range', conf.fig_folder)
        plot_amb_series(pos, tvals, output, fig_leaf, conf)
        max_locs = get_max_locs(output)
        kwargs = {'wn_gain' : wn_gain}
        wnc_out = SimOutput(sim_iter, max_locs, 'wnc_range', **kwargs)

        sim_outs.append(bart_out)
        sim_outs.append(wnc_out)
    mc_out = MCOutput(freq, tvals, pos, conf, sim_outs)
    mc_out.save()
    return

def MCFSum(freqs, conf):
    """
    Run Bartlett on the realizations of the dvecs
    Save a MCOutput object
    Input - 
    freq - int 
        source freq
    Output
    mc_out - MCOutput obj
    """
    source_vel = conf.source_vel
    sim_outs = []
    wn_gain = -2

    for sim_iter in range(conf.num_realizations):
        first = True
        for freq in freqs:
            pfield, pos = load_replicas(freq, conf)
            stride = int(conf.dr // conf.ship_dr)
            pfield = pfield[:,:,::stride]
            pos.r.range = pos.r.range[::stride]
            print('pos shape', pos.r.range.size)
        
            """ Naive covariance """
            tvals, K_samp = load_cov(freq, sim_iter, conf.proj_root, 'none')

            output = bartlett(K_samp, pfield)
            if first == True:
                stack_output = output
            else:
                stack_output *= output
            #fig_leaf = get_fig_leaf(freq, sim_iter, 'bart', conf.fig_folder)
            #plot_amb_series(pos, tvals, output, fig_leaf, conf)

            """ Range covariance """
            pfield, pos = load_range_super_replicas(freq, conf)
            pfield = pfield[:,:,::stride]
            pos.r.range = pos.r.range[::stride]
            print('pos shape', pos.r.range.size)
            kwargs = {'num_ranges':conf.num_ranges, 'phase_key': 'source_correct'}
            tvals, K_samp = load_cov(freq, sim_iter, conf.proj_root, 'range', **kwargs)
            tvals, K_samp = tvals[::5], K_samp[:,:,::5]
            output = bartlett(K_samp, pfield)
            if first == True:
                range_stack_output = output
                first = False
            else:
                range_stack_output *= output
        max_locs = get_max_locs(stack_output)
        print('bart', max_locs.shape)
        bart_out = SimOutput(sim_iter, max_locs, 'bart')
        sim_outs.append(bart_out)

        max_locs = get_max_locs(range_stack_output)
        print('range', max_locs.shape)
        bart_out = SimOutput(sim_iter, max_locs, 'bart_range')
        sim_outs.append(bart_out)

    mc_out = MCOutput(freqs[0], tvals, pos, conf, sim_outs)
    mc_out.save()
    return
        

if __name__ == '__main__':
    now = time.time()

    exp_id = 7
    exp_conf = load_config(exp_id)
    for freq in exp_conf.freqs:
        print('freq', freq)
        #get_amb_surf(freq)
        MCBartlett(freq, exp_conf)
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
