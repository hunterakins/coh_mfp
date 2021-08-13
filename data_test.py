import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
import swellex.audio.make_snapshots as ms
from swellex.audio.config import get_proj_tones, get_proj_zr, get_proj_zs
from swellex.ship.ship import good_time
from scipy.interpolate import interp1d
from coh_mfp.vel_mcm import form_replicas, get_v_dr_grid, get_r_arr
from coh_mfp.simplest_sim import get_amb_surf, plot_amb_surf
from env.env.envs import factory
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc
from signal_proc.mfp.mcm import run_mcm, run_wnc_mcm
import os

"""
Description:
Test out synth stuff on swellex 96 data

Date:
12/9/2020

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


def load_x(freq, proj_str, subfolder,data_root='/home/hunter/data/'):
    name = ms.make_fname(proj_str, data_root, freq,subfolder)
    tname = ms.make_tgrid_fname(proj_str, data_root,subfolder)
    #print('name', name)
    x = np.load(name)
    t = np.load(tname)
    return t, x

def rm_source_phase(t, x, freq):
    """
    Name says it all
    """
    phase_corr = np.exp(complex(0,-1)*2*np.pi*freq*t)
    x *= phase_corr
    return x

def restrict_track(t, x, delta_t=1024/1500): #start_ind, end_ind = 1000, 2000
    #start_ind, end_ind = 5800, 6300
    #start_min = 6
    #start_ind = int(start_min*60/delta_t)
    #start_ind, end_ind = 3515, 4830 # roughly the period from 40 mins to 55 mins
    #start_ind, end_ind = 3515, 3600
    #start_ind, end_ind = 1640, 1840
    #t = t[start_ind:]
    #if len(x.shape) == 2:
    #    x = x[:,start_ind:]
    #else:
    #    x = x[start_ind:]
    return t, x

def get_env(proj_str, source_freq):
    env_builder = factory.create('swellex')
    env = env_builder()
    zr= get_proj_zr(proj_str)
    zs = get_proj_zs(proj_str)
    dz,dr,zmax = 10, 20, 216.5
    folder, fname=  'at_files/', 'swell'
    env.add_source_params(source_freq, zr, zr)
    env.add_field_params(dz, zmax, dr, rmax)
    return env

def get_stride(freq, delta_t):
    """
    For source frequency freq and time spacing between
    data vectors delta_t, get the stride to get a supervector
    """
    lam = 1500/ freq
    T = lam / (2.5) # time to traverse lam
    #T *= 3 # pick a nice broad swath
    #T *= 2
    stride = int(T // delta_t)
    if stride < 1:
        print('stride less than 1, increasing to 1')
        stride = 1
    return stride

def stride_data(t, x, offset, stride):
    """
    Get a rough lambda / 2 synthetic track
    return strided data (ready for synthetic aperture)
    also return T, the time interval corresponding to the stride
     
    """
    #print('stride', stride)
    stride_t = t[offset::stride]
    stride_x = x[:,offset::stride]
    return stride_t, stride_x, T
    
def get_ship_track(): 
    r_t = np.load('/home/hunter/research/code/swellex/ship/npys/t_grid.npy')
    r_r = np.load('/home/hunter/research/code/swellex/ship/npys/r_vals.npy')
    return r_t, r_r

def get_single_el_reps(env):
    folder=  'at_files/'
    fname = 'swell'
    v= 2.05 # doesn't matter
    T = 1 #doesn't matter
    num_synth_els = 1
    r, z, synth_reps = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder, fname)
    return r, z, synth_reps

"""
def get_rmin(v, T, num_synth_els):
    rmin = 400
    if v < 0:
        rmin = abs(v)*T*num_synth_els + grid_dr
    return rmin
"""

def get_r_grid():
    r = np.arange(rmin, rmax+grid_dr, grid_dr)
    return r

def get_mult_el_reps(env, num_synth_els, v, T,fname='swell', adiabatic=False, tilt_angle=0):
    """
    Generate the replicas for multiple synthetic elements
    with an assumed range rate of v and snapshot time separation
    of T
    """
    folder=  'at_files/'
    r, z, synth_reps = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder, fname, adiabatic=adiabatic, tilt_angle=tilt_angle)
    return r, z, synth_reps

def get_d(x, offset_ind):
    tmp = x[:, offset_ind]
    d1 = tmp.reshape(tmp.size, 1)
    d1 /= np.linalg.norm(d1, axis=0)
    return d1

def get_K_samp(start_ind, num_snapshots, x):
    """
    For an initial index start_ind (Referenced to 
    start of x, number of snapshots 'num_snapshots'
    and data x, get the sample cov
    """
    for loop_count in range(num_snapshots):
        offset_ind = loop_count+start_ind
        d = get_d(x, offset_ind)
        if loop_count == 0:
            K_samp = d@(d.T.conj())
        else:
            K_samp += d@(d.T.conj())
    return K_samp
    
def make_sim_comp(rr, x, synth_x):
    """
    Compare x and ynthetic data
    """
    synth_x = np.squeeze(synth_x)
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].plot(rr, x[0,:])
    axes[1].plot(rr, synth_x[0,:])
    return

def comp_tl(x, synth_pow, rr, r):
    """compare intesnity curves of snythetiszed dataa
    and datq x
    """
    x_pow = abs(x)
    synth_pow *= np.sqrt(r)
    x_pow *= np.sqrt(rr)
    fig, axes = plt.subplots(2,1)
    axes[0].contourf(r, zr, 10*np.log10(synth_pow / np.max(synth_pow)))
    axes[1].contourf(rr, zr, 10*np.log10(x_pow / np.max(x_pow)))
    return

def get_r_cov_seq(t, x, num_snap_per_cov, rr):
    """
    data array x, number of snapshots per covariance 
    each cov is formed by averaging num_snapshots data window things
    form the cov mats without any overlap
    return the range positions with the center of each cov mat thing
    """
    total_snapshot_num = x.shape[1]
    num_covs = int((total_snapshot_num-num_snap_per_cov)//num_snap_per_cov)

    source_r = []
    K_samps = []
    for cov_ind in range(num_covs):
        start_ind = num_snap_per_cov*cov_ind
        """ Form cov mat """

        K_samp = get_K_samp(start_ind, num_snap_per_cov, x)
        K_samps.append(K_samp)
        #wnc = run_wnc(K_samp.reshape(1, K_samp.shape[0], K_samp.shape[1]), synth_reps, wn_gain)
        #wnc = wnc[0, :,:]
        #wnc = 10*np.log10(abs(wnc)/np.max(abs(wnc)))

        #plot_amb_surf(-10, r, z, wnc, 'wnc ' + str(num_synth_els) + ' num synth els', r0, zs) 
        #plt.savefig('pics/wnc_' + str(num_synth_els) + '.png')
        center_ind = start_ind + int(num_snap_per_cov//2)
        center_t = t[center_ind] 
        range_ind = np.argmin([abs(x - center_t) for x in t])
        r_center = rr[range_ind]
        source_r.append(r_center)
    return source_r, K_samps

def get_stacked_x(x, num_synth_els, start_ind, stride):
    """ Form the supervector data from the full data set, x, 
    with an offset index start_ind and stride stride """
    num_rcvrs = x.shape[0]
    num_synth_rcvrs = num_rcvrs*num_synth_els
    stacked_x = np.zeros((num_synth_rcvrs,1), dtype=np.complex128)
    for k in range(num_synth_els):
        #tmp = x[:, start_ind+k:start_ind + k + num_synth_els*stride:stride]
        tmp = x[:, start_ind+k*stride]
        norm_fact=  np.linalg.norm(tmp, axis=0)
        tmp /= norm_fact
        stacked_x[num_rcvrs*k:num_rcvrs*(k+1),0] = tmp
    #norm_fact=  np.linalg.norm(stacked_x, axis=0)
    #stacked_x /= norm_fact
    return stacked_x

def make_incoh_mask(num_rcvrs, num_synth_els):
    """
    Make a mask to zero out the inter-synthetic element covariances
    """
    num_synth_rcvrs = num_rcvrs * num_synth_els
    for i in range(num_synth_els):
        dvec = np.zeros((num_synth_rcvrs, 1))
        dvec[num_rcvrs*i:num_rcvrs*(i+1), 0] = 1
        cov = dvec@dvec.T
        if i == 0:
            mask = cov
        else:
            mask += cov
    return mask
    
def get_r_super_cov_seq(t, x, num_snap_per_cov, r_interp, num_synth_els, stride, incoh=False):
    """
    data array x, number of snapshots per covariance 
    each cov is formed by averaging num_snapshots data window things
    form the cov mats without any overlap
    return the range positions with the center of each cov mat thing
    Input
    t - np array
        grid of times at beginning of each data vector
    x - np array
        grid of data vectors
    num_snap_per_cov - int
        number of snapshots per cov matrix
    r_interp - function 
        gives range as functio of time (derived from gps)
    num_synth_els - int
        how many ranges to combine
    stride - int
        spacing of synthetic elements as an index of time grid
    incoh - bool
        zero out "range coherent" terms if true
    """
    total_snapshot_num = x.shape[1]
    num_covs = int((total_snapshot_num-num_snap_per_cov)//num_snap_per_cov)

    source_r = []
    t_centers = []
    num_rcvrs = x.shape[0]
    num_synth_rcvrs = num_rcvrs*num_synth_els
    num_covs = int((total_snapshot_num- num_synth_els*stride - num_snap_per_cov)//num_snap_per_cov)
    K_samps = np.zeros((num_covs, num_synth_rcvrs, num_synth_rcvrs), dtype=np.complex128)

    if incoh == True:
        mask = make_incoh_mask(num_rcvrs, num_synth_els)

    for i in range(num_covs):
        cov_start_ind = i*num_snap_per_cov
        for j in range(num_snap_per_cov):
            start_ind = cov_start_ind + j
            stacked_x = get_stacked_x(x, num_synth_els, start_ind, stride)
            if j ==0:
                ith_cov = stacked_x @ (stacked_x.T.conj())
            else:
                ith_cov += stacked_x @ (stacked_x.T.conj())
        center_ind = cov_start_ind + int(num_snap_per_cov//2)
        t_center = t[center_ind]
        source_r.append(r_interp(t_center))

        if incoh==True:
            ith_cov *= mask

        ith_cov /= np.trace(ith_cov)

        K_samps[i,:,:] = ith_cov
        t_centers.append(t_center)
    t_centers = np.array(t_centers)
    return source_r, t_centers, K_samps

def make_cov_check_fig(K_samp, best_rep):
    guess_K = best_rep@(best_rep.T.conj())
    fig, axes = plt.subplots(2,1)
    axes[0].imshow(abs(K_samp))
    axes[1].imshow(abs(guess_K)) 

def get_best_guesses(outputs):
    """
    For list of ambiguity surfaces outputs and search grid of range, r,
    get the argmax """
    best_guesses = []
    r = get_r_grid()
    for i in range(len(outputs)):
        output = outputs[i]
        #output /= len(freqs)
        best_ind = np.argmax(output)
        best_depth_ind, best_range_ind = int(best_ind // output.shape[1]), best_ind % output.shape[1]
        #K_samp = K_samps[i]
        #best_rep = synth_reps[:, best_depth_ind, best_range_ind]
        #best_rep = best_rep.reshape(best_rep.size, 1)
        #make_cov_check_fig(K_samp, best_rep)
        #plt.show()
        best_guesses.append(r[best_range_ind])
    return best_guesses

def make_comp_plot(single_r, r, z, single_el_out, output, db_lev, r_true, zs, title_str):
    levels = np.linspace(db_lev, 0, 20)
    fig, axes = plt.subplots(2,1)
    axes[0].contourf(single_r, z, single_el_out, levels=levels, extend='both')
    cont = axes[1].contourf(r, z, output, levels=levels, extend='both')
    axes[0].scatter(r_true, zs, zs, marker='+', color='r')
    axes[1].scatter(r_true, zs, zs, marker='+', color='r')
    ind = np.argmax(single_el_out)
    inds = (ind % single_r.size, int(ind // single_r.size))
    axes[0].scatter(single_r[inds[0]], z[inds[1]], marker='+', color='b')
    ind = np.argmax(output)
    inds = (ind % r.size, int(ind // r.size))
    axes[1].scatter(r[inds[0]], z[inds[1]], marker='+', color='b')
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    fig.colorbar(cont, ax=axes.ravel().tolist())
    fig.suptitle(title_str)
    return fig

def make_comp_plots(vv, T, proj_str, num_synth_els, z, single_el_outs, outputs, db_lev, r_center, zs, wnc_mcm=False):
    for i in range(len(outputs)):
        single_el_out = single_el_outs[i]
        output = outputs[i]
        single_r = get_r_grid()
        r = get_r_grid()
        if wnc_mcm==True:
            r = get_r_grid(np.median(vv), T, num_synth_els)
            tit = 'wnc mcm'
        else:
            tit = 'bartlett '
        fig = make_comp_plot(single_r, r, z,single_el_out, output, db_lev, r_center[i], zs, tit + ', snapshot ' + str(i))
        plt.savefig('pics/' +proj_str+ '/' +  str(i).zfill(3)+ '.png')
        plt.clf()
        plt.close(fig)

def make_range_track_name(proj_str, source_freq, num_synth_els, suffix=''):
    return proj_str + '/' + str(source_freq) + '_range_track_' + str(num_synth_els) + suffix

def deal_with_t_x_r(t, x, source_freq):
        """
        Time grid t, data array x, source freq (to remove phase), 
        data_stride is the receiver stride 
        """
        x = rm_source_phase(t, x, source_freq)

        rt, rr = get_ship_track()
        r_interp = interp1d(rt, rr)
        t, x = restrict_track(t, x)
        rt, rr = restrict_track(rt, rr)
        rt = t
        rr = r_interp(rt)
        vv = np.zeros(rr.size)
        vv[:-1] = (rr[1:] - rr[:-1]) / (rt[1] - rt[0])
        vv[-1] = vv[-2]
        fig = plt.figure()
        plt.plot(vv)
        plt.savefig('pics/vv.png')
        plt.close(fig)
        return t, x, rt, rr, vv, r_interp

def make_range_perform_plot(proj_str, num_snapshots, r_center, t, best_guesses, source_freq, num_synth_els, suffix=''):
    plt.figure()
    rel_t = t[::num_snapshots][:len(r_center)]
    plt.plot(rel_t, best_guesses)
    plt.plot(rel_t, r_center)
    bad_inds = [i for i in range(len(r_center)) if not good_time(rel_t[i])]
    plt.scatter([rel_t[x] for x in bad_inds], [r_center[x] for x in bad_inds], c='r')
    pic_name = make_range_track_name(proj_str, source_freq, num_synth_els, suffix=suffix)
    print(pic_name)
    plt.savefig('pics/' + pic_name)
    return

def get_mcm_synth_reps(ship_dr, num_synth_els, num_constraints, kbar, T, env):
    folder=  'at_files/'
    fname = 'swell'
    dr_grid = get_v_dr_grid(ship_dr, num_synth_els, num_constraints, kbar)
    v_grid = dr_grid/T
    print('v_grid', v_grid)
    r, z, r_arr = get_r_arr(env, rmin, rmax, grid_dr, v_grid, T, num_synth_els,folder, fname)
    return r, z, r_arr

def gen_vid(proj_str, source_freq, num_synth_els, title_str, outputs):
    """
    Aggregate the make comp plots into a single mp4 file
    """
    os.system('ffmpeg -loglevel quiet -r 3 -f image2 -s 1920x1080 -i pics/' + proj_str + '/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p pics/' + proj_str + '/' +  str(source_freq) + '_' + str(num_synth_els) + '_' + title_str + '.mp4')

    for i in range(len(outputs)):
        os.system('rm pics/' + proj_str + '/'+ str(i).zfill(3) + '.png')
    return

def populate_env(env, source_freq, zs, zr, rr, fname='swell'):
    """ 
    Run once on the array to get the modes and kbar,
    then update zs to be zr for replica generation  
    """
    env.add_source_params(source_freq, zs, zr)
    synth_x, _ = env.run_model('kraken_custom_r', 'at_files/', fname, zr_flag=True, zr_range_flag=False, custom_r = rr)
    kbar = np.mean(env.modes.k).real
    env.add_source_params(source_freq, zr, zr)
    return env

def get_synth_arr_params(source_freq, t, num_snapshots): 
    """
    get_stride does the determination ofthe array spacing,
    but this gets the snapshot stride 
    just do snapshots tride...
    """
    delta_t = t[1]-t[0]
    #stride = get_stride(source_freq, delta_t) 
    #if stride < num_snapshots:
        #print('warning, overlapping covs in synth els, increasoing stride to num_snapshots')
        #stride = num_snapshots
    stride = num_snapshots
    T = delta_t*stride
    return delta_t, stride, T

def augment_outputs(outputs, r_center, K_samps, env, num_synth_els, T, specific_v):
    if outputs == []:
        first_freq = True # flag
    else:
        first_freq = False
    last_v = -1000 # dummy value
    for i, r0 in enumerate(r_center):
        v = specific_v[i]
        K_samp = K_samps[i,:,:]
        if v != last_v:
            r, z, synth_reps = get_mult_el_reps(env, num_synth_els, v, T)
            last_v = v
        output = get_amb_surf(r, z, K_samp, synth_reps,matmul=True)
        if first_freq == True:
            outputs.append(output)
        else:
            outputs[i] += output
    return r,z

def gen_incoh_avg_plots(outputs, r, z, r_center, zs, db_lev, freqs, proj_str, title_str):
    for i in range(len(outputs)):
        out = outputs[i]
        out /= len(freqs)
        fig = plot_amb_surf(-10, r, z, out,  title_str + str(i), r_center[i], zs)
        plt.savefig('pics/' + proj_str + '/' +str(i).zfill(3) + '.png')
        plt.close(fig)
    return
 
def make_str_id(num_synth_els, num_snapshots, num_freqs, v, tilt_angle,incoh, wnc):
    str_id = str(num_synth_els) + '_' + str(num_snapshots) + '_' + str(num_freqs) + '_' + str(v) + '_' + str(tilt_angle)
    if incoh == True:
        str_id += '_incoh'
    if wnc == True:
        str_id += '_wnc'
    return str_id


class DataRun:
    def __init__(self, proj_str, num_snapshots, v, subfolder, num_freqs, num_synth_els, tilt_angle, incoh=False, wnc=False):
        self.proj_str = proj_str
        self.num_snapshots = num_snapshots
        self.v = v
        self.subfolder = subfolder
        self.num_freqs = num_freqs
        self.num_synth_els= num_synth_els
        self.str_id = make_str_id(num_synth_els, num_snapshots, num_freqs, v, tilt_angle,incoh, wnc)
        self.incoh = incoh
        self.wnc = wnc
        self.tilt_angle=tilt_angle

    def run(self, **kwargs):
        if self.wnc==False:
            r_center, cov_t, r, z, outputs, max_vals = synth_stacked_bart(self)
        else:
            r_center, cov_t, r, z, outputs, max_vals, wnc_outs, wnc_max_vals = synth_stacked_bart(self, **kwargs)
            self.wnc_outs = wnc_outs
            self.wnc_max_vals = wnc_max_vals
        self.r_center = r_center
        self.cov_t = cov_t # time at the center of the fft window
        self.r = r
        self.z = z
        self.outputs = outputs
        self.max_vals = max_vals
        return

    def save(self):
        save_loc = make_save_loc(self.proj_str, self.subfolder)
        prefix = self.str_id 
        fname = save_loc + prefix  + '.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        return

def load_dr(dr):
    """
    Use a semi initialized dr object to load a fully populated one
    Maybe this is a bad idea?
    """
    save_loc = make_save_loc(dr.proj_str, dr.subfolder)
    prefix = dr.str_id 
    fname = save_loc + prefix  + '.pickle'
    with open(fname, 'rb') as f:
        dr = pickle.load(f)
    return dr

def get_outputs(drp, wnc=False, **kwargs):
    """
    For data run params drp, calculate the 
    stacked ambiguity surface (incoherently sum log 
    outputs over freq), and return the max vals of the amb 
    surf for each freq and time
    By default computes a simple bartlett
    If wnc == True, then also computes wnc
    and wn_gain is a dict key
    """
    freqs = get_proj_tones(drp.proj_str)[:drp.num_freqs]
    zs, zr = get_proj_zs(drp.proj_str), get_proj_zr(drp.proj_str)

    if wnc==True:
        wn_gain = kwargs['wn_gain']


    for freq_ind, source_freq in enumerate(freqs):
        #print('Now processing freq', source_freq)
        fc = ms.get_fc(source_freq, drp.v)  
        t, x =  load_x(fc, drp.proj_str, drp.subfolder)
        num_rcvrs = x.shape[0]
        if freq_ind == 0:
            rcvr_stride = int(zr.size // num_rcvrs)
            zr = zr[::rcvr_stride]

        t,x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq)

        env = get_env(drp.proj_str, source_freq)
        env = populate_env(env, source_freq, zs, zr, rr, fname=drp.proj_str)
        kbar = np.mean(env.modes.k.real)
        delta_k = np.max(env.modes.k.real) - np.min(env.modes.k.real)
        #print('delta_k', delta_k)
        #print('range cell', 2 * np.pi / delta_k)
        time_in_cell = 2*np.pi / delta_k / 2.5
        #print('time in cell', 2*np.pi / delta_k  / 2.5)

        delta_t = t[1]-t[0]      
        #print('good num snaps',  time_in_cell/ delta_t)
        delta_t, stride, T = get_synth_arr_params(source_freq, t, drp.num_snapshots)
        #print('stride, T', stride, T)
     
        r_center, cov_t, K_samps = get_r_super_cov_seq(t, x,drp.num_snapshots, r_interp, drp.num_synth_els, stride, incoh=drp.incoh)

        #print(' num k samps', cov_t.size)
        t_before = time.time()
        r, z, synth_reps = get_mult_el_reps(env, drp.num_synth_els, drp.v, T,fname=drp.proj_str,adiabatic=False, tilt_angle=drp.tilt_angle)
        #print('Time to compute replcias =' , time.time() - t_before)
        if source_freq == freqs[0]:
            bart_outs = np.zeros((len(r_center), z.size, r.size))
            max_vals = np.zeros((len(freqs), len(r_center)))
            if wnc == True:
                wnc_outs = np.zeros((len(r_center), z.size, r.size))
                wnc_max_vals  = np.zeros((len(freqs), len(r_center)))
            """ Annoying thing where I need to truncate based on
            first freq 
            """
            num_times = cov_t.size

        """ Compute bartlett """
        t_before = time.time()
        for i in range(num_times):
            K_samp = K_samps[i,...]
            #plt.figure()
            #plt.imshow((K_samp).imag)
            output, max_val = get_amb_surf(r, z, K_samp, synth_reps, matmul=True) # output is in db
            ind = np.argmax(output)
            best_rep = synth_reps[:, ind//r.size, ind%r.size]
            best_rep = best_rep.reshape(best_rep.size,1)
            #plt.figure()
            #plt.imshow((best_rep.conj()@(best_rep.T)).imag)
            #print('max_val', max_val)
            #print(best_rep.T.conj()@K_samp@best_rep)
            #plt.show()
            
            """ Remove normalization """
            output += 10*np.log10(max_val)
            """ Add to bartlett mat """
            bart_outs[i,:,:] += output
            max_vals[freq_ind, i] = max_val
        #print('Time to compute bartlett =' , time.time() - t_before)

        """ Comput wnc if it's a thing  """
        t_before = time.time()
        if wnc == True:
            wnc_out_freq = lookup_run_wnc(K_samps[:num_times, :,:], synth_reps, wn_gain)
            tmp = wnc_out_freq.reshape(num_times, wnc_out_freq.shape[1]*wnc_out_freq.shape[2])
            """ don't normalize """
            #wnc_max_val_freq = np.max(tmp, axis=1)
            #wnc_max_vals[freq_ind, :] = wnc_max_val_freq
            #for i in range(num_times):
            #    wnc_out_freq[i,:,:] /= wnc_max_val_freq[i]
            wnc_out_freq = 10*np.log10(wnc_out_freq)
            wnc_outs += wnc_out_freq
        #print('Time to compute wnc =' , time.time() - t_before)
            
    r_center = r_center[:num_times]
    cov_t = cov_t[:num_times] # make sure we truncate based on first freq
    if wnc == True:
        #wnc_outs /= len(freqs)
        return r_center, cov_t, r, z, bart_outs, max_vals, wnc_outs, wnc_max_vals
    else:
        return r_center, cov_t, r, z, bart_outs, max_vals
           
def synth_stacked_bart(drp,**kwargs):
    """
    Run a bartlett proc on the dta from proj_str with num_synth_els synthetic elements
    drp - DataRunParams obj holds errythagn
    """
    wnc = drp.wnc
    freqs = get_proj_tones(drp.proj_str)
    freqs = freqs[:drp.num_freqs]
    if wnc==False:
        r_center, cov_t, r, z, outputs, max_vals = get_outputs(drp)
        return r_center, cov_t, r, z, outputs, max_vals
    if wnc==True:
        r_center, cov_t, r, z, outputs, max_vals, wnc_outs, wnc_max_vals = get_outputs(drp, wnc=wnc,**kwargs)
        return r_center, cov_t, r, z, outputs, max_vals, wnc_outs, wnc_max_vals

def make_save_loc(proj_string, subfolder):
    if proj_string not in os.listdir('npy_files'):
        os.mkdir('npy_files/' + proj_string)
    if subfolder not in os.listdir('npy_files/' + proj_string):
        os.mkdir('npy_files/' + proj_string + '/' + subfolder)
    return 'npy_files/' + proj_string + '/' + subfolder + '/'

def get_vel_arr(vv, tilt_angles, max_val_list):
    """
    Integrate out tilt angles to get the velocity
    values for each time 
    """
    max_val_arr = get_max_val_arr(max_val_list)
    best_param_inds = np.argmax(max_val_arr, axis=0)
    best_tilt_inds = best_param_inds // vv.size
    num_snaps = max_val_arr.shape[1]
    best_vel_arr = np.zeros((vv.size, num_snaps))
    for i in range(num_snaps):
        bracket = best_tilt_inds[i]*vv.size
        best_vel_arr[:,i] = max_val_arr[bracket:bracket+vv.size, i]
    return best_vel_arr

def get_tilt_arr(vv, tilt_angles, max_val_list):
    """
    Integrate out velocity to get tilt 
    values for each time 
    """
    num_tilt_angles = tilt_angles.size
    max_val_arr = get_max_val_arr(max_val_list)
    best_param_inds = np.argmax(max_val_arr, axis=0)
    best_vel_inds = best_param_inds % vv.size
    num_snaps = max_val_arr.shape[1]
    best_tilt_arr = np.zeros((num_tilt_angles, num_snaps))
    print(max_val_arr.shape, num_tilt_angles)
    print('num snaps', num_snaps)
    for i in range(num_snaps):
        inds = [best_vel_inds[i] + x*vv.size for x in range(num_tilt_angles)]
        best_tilt_arr[:,i] = max_val_arr[inds, i]
    return best_tilt_arr

class DRPRuns:
    """
    Class to hold params for a suite of MFP runs 
    """
    def __init__(self, proj_str, subfolder, num_snap_list, num_freq_list, num_synth_el_list, vv, tilt_angles, incoh, wnc):
        self.proj_str = proj_str
        self.subfolder = subfolder
        self.num_snap_list = num_snap_list
        self.num_freq_list = num_freq_list  
        self.num_synth_el_list = num_synth_el_list
        self.vv = vv
        self.tilt_angles=tilt_angles
        self.incoh = incoh
        self.wnc=wnc

    def run_all(self, **kwargs):
        """
        For each element in the tensor product of snapshot numbers, freqs, and synth els and velocities,
        create a drp, produce 
        """
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    for tilt_angle in self.tilt_angles:
                        print('Getting data run for tilt = ', tilt_angle, ' degrees')
                        for v in self.vv:
                            print('Running data run for v = ', v)
                            dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, num_freqs, num_synth_els, tilt_angle, self.incoh, self.wnc)
                            dr.run(**kwargs)
                            dr.save()

    def save(self, str_id):
        save_loc = make_save_loc(self.proj_str, self.subfolder)
        prefix = str_id
        fname = save_loc + prefix  + '.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        return

    def get_best_range_guesses(self):
        """ Best best tilt and velocity """
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    dr_list = []
                    for tilt_angle in self.tilt_angles:
                        for v in self.vv:
                            dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, num_freqs, num_synth_els, tilt_angle, self.incoh, self.wnc)
                            dr = load_dr(dr)
                            dr_list.append(dr)
                            
                    best_outs = form_best_output_mat(dr_list)
                    r_center  = dr.r_center
                    best_r_guesses = []
                    best_z_guesses = []
                    for i in range(len(r_center)):
                        amb_surf = best_outs[i,:,:]
                        best_ind = int(np.argmax(amb_surf) % dr.r.size)
                        best_r_guess = dr.r[best_ind]    
                        best_z_ind = int(np.argmax(amb_surf) // dr.r.size)
                        best_z_guess = dr.z[best_z_ind]
                        best_r_guesses.append(best_r_guess)
                        best_z_guesses.append(best_z_guess)

                    plt.figure()
                    plt.suptitle('_'.join([str(num_snaps), str(num_freqs), str(num_synth_els)]) + ' num snaps, num freqs, num_synth els')
                    plt.scatter(dr.cov_t, r_center)
                    plt.scatter(dr.cov_t, best_r_guesses)
                        
        plt.show()

    def show_param_corrs(self):
        """ Get the velocity and tilt amb surface """
        #self.tilt_angles = np.array(self.tilt_angles)
        self.tilt_angles = np.array([round(x, 4) for x in self.tilt_angles])
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    fig,axes = plt.subplots(2,1)
                    dr, vel_arr, tilt_arr = self.get_param_arrs(num_snaps, num_synth_els, num_freqs)
                    vel_arr = 10*np.log10(vel_arr)
                    tilt_arr = 10*np.log10(tilt_arr)
                    levels = np.linspace(-10, 0, 10)
                    axes[0].contourf(dr.cov_t, self.vv, vel_arr,levels=levels)
                    levels = np.linspace(-1, 0, 10)
                    axes[1].contourf(dr.cov_t, self.tilt_angles, tilt_arr, levels=levels)
                    plt.suptitle('_'.join([str(num_snaps), str(num_freqs), str(num_synth_els)]) + ' num snaps, num freqs, num_synth els')
                plt.show()

    def get_param_arrs(self, num_snaps, num_synth_els, num_freqs):
        dr_list = []
        max_val_list=[]
        for tilt_angle in self.tilt_angles:
            for v in self.vv:
                dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, num_freqs, num_synth_els, tilt_angle, self.incoh, self.wnc)
                dr = load_dr(dr)
                dr_list.append(dr)
                max_val_list.append(dr.max_vals)
            vel_arr = get_vel_arr(self.vv, self.tilt_angles, max_val_list)
            vel_arr /= np.max(vel_arr, axis=0)
        tilt_arr = get_tilt_arr(self.vv, self.tilt_angles, max_val_list)
        tilt_arr /= np.max(tilt_arr, axis=0)
        return dr, vel_arr, tilt_arr

    def show_vel_best_guess(self):
        """ Plot max vels from the runs"""
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    dr_list = []
                    max_val_list=[]
                    for tilt_angle in self.tilt_angles:
                        for v in self.vv:
                            dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, num_freqs, num_synth_els, tilt_angle, self.incoh, self.wnc)
                            dr = load_dr(dr)
                            dr_list.append(dr)
                            max_val_list.append(dr.max_vals)
                    max_val_arr = get_max_val_arr(max_val_list)
                    best_param_ind = np.argmax(max_val_arr, axis=0)
                    if len(self.tilt_angles == 1):
                        plt.figure()
                        plt.scatter(dr.cov_t, np.array(self.vv)[best_param_ind])
                        plt.show()
                    else:
                        best_v_ind = best_param_ind % len(self.tilt_angles)
                        best_tilt_ind = best_param_ind 
                        plt.figure()
                        plt.scatter(dr.cov_t, self.vv[best_v_ind])
                        plt.suptitle(str(num_snaps) +'_' + str(num_synth_els))
                        plt.show()

    def make_amb_mov(self):
        """ 
        Make an ambiguity surface mov
        """
        zs = get_proj_zs(self.proj_str)
        folder = 'pics/' + self.proj_str + '/' + self.subfolder + '/'
        if self.proj_str not in os.listdir('pics'):
            os.mkdir('pics/' + self.proj_str)
        if self.subfolder not in os.listdir('pics/' + self.proj_str):
            os.mkdir(folder)
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    dr_list = []
                    for tilt_angle in self.tilt_angles:
                        if num_synth_els == 1:
                            v = -2.3
                            dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, num_freqs, num_synth_els, tilt_angle, self.incoh, self.wnc)
                            dr = load_dr(dr)
                            best_outs = dr.outputs
                            r_center = dr.r_center
                            dr_list.append(dr)
                        else:
                            for v in self.vv:
                                dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, num_freqs, num_synth_els, tilt_angle, self.incoh, self.wnc)
                                dr = load_dr(dr)
                                dr_list.append(dr)
                                r_center=dr.r_center
                    best_outs = form_best_output_mat(dr_list)
                    for i in range(len(r_center)):
                        db_max = np.max(best_outs[i,:,:])
                        db_min = db_max - 20
                        fig = plot_amb_surf([db_min, db_max], dr.r, dr.z, best_outs[i,:,:], 'bart ' + str(num_synth_els), r_center[i], zs)
                        plt.savefig(folder + str(i).zfill(3) + '.png')
                        plt.close(fig)
                    os.system('ffmpeg -loglevel quiet -r 3 -f image2 -s 1920x1080 -i ' + folder + '/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ' + folder +  str(num_snaps) + '_' + str(num_synth_els) + '_' + str(num_freqs) + '_' + str(self.incoh) + '.mp4')
                    os.system('rm ' + folder + '*png')

def get_max_val_arr( max_val_list):
    """
    Each entry in max_val list is a numpy array that corredsponds
    to some parameter configuration
    Each numpy array contains a
    row for each processed frequency and a column for each snapshot
    It contains the maximum value of the correlation surface for each snapshot
    The idea is toassociate rows with parameter configuratiopns
    and columns to snapshots
    """
    num_snaps = max_val_list[0].shape[1]
    max_val_arr = np.zeros((len(max_val_list), num_snaps))
    for j in range(len(max_val_list)):
        max_vals = max_val_list[j]
        max_val_arr[j, :] = np.sum(max_vals, axis=0)
    return max_val_arr

def form_best_output_mat(dr_list):
    """
    For a list of data runs with varying v and tilt angle,
    pick out the best v and tilt angle for each snapshot,
    and form a single output that has as each entry the best one
    """
    vv = [] 
    tilt_angles = []
    max_val_list = []
    outputs_list = []
    for dr in dr_list:
        max_val_list.append(dr.max_vals)
        vv.append(dr.v)
        tilt_angles.append(dr.tilt_angle)
        outputs_list.append(dr.outputs)
    max_val_arr = get_max_val_arr(max_val_list)
    max_inds = np.argmax(max_val_arr, axis=0)
    best_outs = np.zeros((dr.outputs.shape))
    for i in range(len(dr.r_center)):
        best_outs[i,:,:] = outputs_list[max_inds[i]][i,:,:]
    return best_outs
    
rmax = 10*1e3
grid_dr = 50
rmin = 500

if __name__ == '__main__':
    #simple_stacked_bart()

    incoh = False
    wnc = False
    wn_gain = -1
    N_fft = 8096
    subfolder = str(N_fft)
    num_snap_list = [10]
    for proj_str in ['s5_deep']:#, 's5_quiet1']:#, 's5_quiet2', 's5_quiet3', 's5_quiet4']:
        freqs = get_proj_tones(proj_str)
        t0 = time.time()
        #num_freqs= 4
        #freqs = get_proj_tones(proj_str)[:num_freqs]
        #num_synth_els = 1
        zs = get_proj_zs(proj_str)
        vv = np.linspace(-1.8, -2.5, 10)
        vv = np.array([round(x, 4) for x in vv])
        
        num_freq_list = [13]
        num_synth_el_list = [1]
        #tilt_angles = np.linspace(-0.5, -1.75, 6)
        #tilt_angles = np.linspace(-1, -1.5, 1)
        #tilt_angles = np.array([round(x, 4) for x in tilt_angles])
        tilt_angles = np.array([-1])
        print('tilt_angles', tilt_angles)

        dr_runs = DRPRuns(proj_str, subfolder, num_snap_list, num_freq_list, num_synth_el_list, vv, tilt_angles, incoh, wnc)
        dr_runs.run_all(**{'wn_gain':wn_gain})
        #dr_runs.run_all()
        dr_runs.save('wnc_-1_test')
        """
        output_list = []
        vv= [-2.5]
        for v in vv:
            v = round(v, 4)
            drp = DataRun(proj_str, num_snaps, v, subfolder, num_freqs, num_synth_els)
            #r_center, cov_t,r,z, outputs, max_vals = load_results(drp)
            drp.run()
            drp.save()

            #synth_stacked_bart(2,v) 
        """

