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
from signal_proc.mfp.wnc import run_wnc
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

proj_str = 's5_quiet2'
#proj_str = 's5_quiet2'
#proj_str = 's5_quiet_tones'
#proj_str = 's5_deep' #proj_str = 's5_shallow'
freqs = get_proj_tones(proj_str)

grid_dr = 50
rmax = 10*1e3
rmin = 500

def load_x(freq,subfolder):
    data_root = '/home/hunter/data/'
    name = ms.make_fname(proj_str, data_root, freq,subfolder)
    tname = ms.make_tgrid_fname(proj_str, data_root,subfolder)
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
    start_ind, end_ind = 500, 3000
    start_min, end_min = 40, 55
    start_ind = int(start_min*60/delta_t)
    end_ind = int(end_min*60/delta_t)
    print('s ind, e ind', start_ind, end_ind)
    #start_ind, end_ind = 3515, 4830 # roughly the period from 40 mins to 55 mins
    #start_ind, end_ind = 3515, 3600
    #start_ind, end_ind = 1640, 1840
    t = t[start_ind:end_ind]
    if len(x.shape) == 2:
        x = x[:,start_ind:end_ind]
    else:
        x = x[start_ind:end_ind]
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
        print('stride less than 1!!')
    return stride
    
def stride_data(t, x, offset, stride):
    """
    Get a rough lambda / 2 synthetic track
    return strided data (ready for synthetic aperture)
    also return T, the time interval corresponding to the stride
     
    """
    print('stride', stride)
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

def get_mult_el_reps(env, num_synth_els, v, T,fname='swell'):
    """
    Generate the replicas for multiple synthetic elements
    with an assumed range rate of v and snapshot time separation
    of T
    """
    folder=  'at_files/'
    r, z, synth_reps = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder, fname)
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

def get_r_super_cov_seq(t, x, num_snap_per_cov, r_interp, num_synth_els, stride):
    """
    data array x, number of snapshots per covariance 
    each cov is formed by averaging num_snapshots data window things
    form the cov mats without any overlap
    return the range positions with the center of each cov mat thing
    """
    total_snapshot_num = x.shape[1]
    num_covs = int((total_snapshot_num-num_snap_per_cov)//num_snap_per_cov)

    source_r = []
    t_centers = []
    num_rcvrs = x.shape[0]
    num_synth_rcvrs = num_rcvrs*num_synth_els
    num_covs = int((total_snapshot_num- num_synth_els*stride - num_snap_per_cov)//num_snap_per_cov)
    K_samps = np.zeros((num_covs, num_synth_rcvrs, num_synth_rcvrs), dtype=np.complex128)
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
        #K_samps.append(ith_cov)
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

def make_comp_plots(vv, T, num_synth_els, z, single_el_outs, outputs, db_lev, r_center, zs, wnc_mcm=False):
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

def deal_with_t_x_r(t, x, source_freq, data_stride):
        x = rm_source_phase(t, x, source_freq)
        x = x[::data_stride, :]

        rt, rr = get_ship_track()
        r_interp = interp1d(rt, rr)
        #t, x = restrict_track(t, x)
        #rt, rr = restrict_track(rt, rr)
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

def make_range_perform_plot(num_snapshots, r_center, t, best_guesses, source_freq, num_synth_els, suffix=''):
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

def gen_vid(source_freq, num_synth_els, title_str, outputs):
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
    print('kbar', kbar)
    env.add_source_params(source_freq, zr, zr)
    return env

def get_synth_arr_params(source_freq, t, num_snapshots): 
    """
    get_stride does the determination ofthe array spacing,
    but this gets the snapshot stride 
    """
    delta_t = t[1]-t[0]
    stride = get_stride(source_freq, delta_t) 
    T = delta_t*stride
    if stride < num_snapshots:
        print('warning, overlapping covs in synth els, increase stride or decrease snapshot num')
    return delta_t, stride, T

def synthetic_ap_eval():
    zs = get_proj_zs(proj_str)
    zr = get_proj_zr(proj_str)
    data_stride = 4
    zr = zr[::data_stride]
    print('zr', zr)
    #freqs = freqs[2:]
    #freqs = freqs[6:10]
    #freqs = freqs[:4]
    freqs = freqs[:5] # reduce number of frequency bands used
    wn_gain = -5 # gain in wnc
    num_constraints = 3 # num velocity constraints for vel_mcm
    num_snapshots = 30 # num snaps in cov matrix formation


    for source_freq in freqs:
        for num_synth_els in [1,2]:
            outputs=[]
            wnc_outputs = []
            mcm_outputs = []
            print('source freq', source_freq)

            t, x =  load_x(source_freq)
            t,x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq, data_stride)

            env = get_env(proj_str, source_freq)
            env = populate_env(env, source_freq, zs, zr, rr)
            kbar = np.mean(env.modes.k.real)
           

            delta_t, stride, T = get_synth_arr_params(source_freq, t, num_snapshots)

            r_center,K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, num_synth_els, stride)

            specific_v = vv[::num_snapshots][:len(r_center)] 
            loop_count = 0
            last_v = -1000 # dummy value
            ship_dr = np.median(specific_v)*T
            """ MCM Part 
            if num_synth_els > 1:
                r, z, mcm_synth_reps = get_mcm_synth_reps(ship_dr, num_synth_els, num_constraints, kbar, T, env)

                look_ind = int(num_constraints//2)
                print(mcm_synth_reps.shape, look_ind)

                #mcm_outputs = run_mcm(K_samps, mcm_synth_reps, look_ind = look_ind)
                wnc_mcm_outputs = run_wnc_mcm(K_samps, mcm_synth_reps, wn_gain, look_ind = look_ind)

                #norm_fact = np.max(abs(mcm_outputs), axis=0)
                #mcm_outputs = 10*np.log10(abs(mcm_outputs) / norm_fact)
                #print(mcm_outputs.shape)
                norm_fact = np.max(abs(wnc_mcm_outputs), axis=0)
                wnc_mcm_outputs = 10*np.log10(abs(wnc_mcm_outputs) / norm_fact)
                print(wnc_mcm_outputs.shape)

                #for j in range(wnc_mcm_outputs.shape[0]):
                #    plot_amb_surf(-10, r, z, wnc_mcm_outputs[j,:,:], 'wnc wnc_mcm', r_center[0], zs)
                #plt.show()
            """
            for i, r0 in enumerate(r_center):
                v = specific_v[i]
                K_samp = K_samps[i,:,:]
                if v != last_v:
                    r, z, synth_reps = get_mult_el_reps(env, num_synth_els, v, T)
                    last_v = v
                output = get_amb_surf(r, z, K_samp, synth_reps)
                outputs.append(output)
                K_samp = K_samp.reshape(1, K_samp.shape[0], K_samp.shape[1])
                #wnc_output = run_wnc(K_samp, synth_reps, wn_gain)[0,:,:]
                #wnc_output = 10*np.log10(abs(wnc_output/np.max(abs(wnc_output))))
                #wnc_outputs.append(wnc_output)
                loop_count += 1

            """
            r, z, synth_reps = get_mult_el_reps(env, num_synth_els, np.median(specific_v), T)
            wnc_outputs = run_wnc(K_samps, synth_reps, wn_gain)
            wnc_outputs = 10*np.log10(abs(wnc_outputs/np.max(abs(wnc_outputs), axis=0)))
            """


            """ Store simple outputs for comparison """
            if num_synth_els == 1:
                single_el_outs = [x for x in outputs]
                single_el_wnc_outs = [x for x in wnc_outputs]

            best_guesses = get_best_guesses(outputs)
            best_wnc_guesses = get_best_guesses(wnc_outputs)

            """ If synthetic, compare..."""
            if num_synth_els > 1:
               make_comp_plots(specific_v, T, num_synth_els, z, single_el_outs, outputs, -10, r_center, zs) 

            make_range_perform_plot(num_snapshots, r_center, t, best_guesses, source_freq, num_synth_els)

            if num_synth_els > 1:
                gen_vid(source_freq, num_synth_els, bart, outputs)

            """ If synthetic, compare wnc"""
            if num_synth_els > 1:
               wnc_out_list = [wnc_mcm_outputs[i,:,:] for i in range(wnc_mcm_outputs.shape[0])]
               #make_comp_plots(specific_v, T, num_synth_els, z, single_el_wnc_outs, wnc_outputs, -10, r_center, zs) 
               make_comp_plots(specific_v, T, num_synth_els, z, single_el_wnc_outs, wnc_out_list, -10, r_center, zs,wnc_mcm=True) 

            if num_synth_els > 1:
                gen_vid(source_freq, num_synth_els, 'wnc', outputs)

def simple_stacked_bart():
    """ Run a simple incoherent average bartlett surface
    """
    zs = get_proj_zs(proj_str)
    zr = get_proj_zr(proj_str)
    data_stride = 4
    zr = zr[::data_stride]
    print('zr', zr)
    freqs = get_proj_tones(proj_str)

    num_constraints = 3 # num velocity constraints for vel_mcm
    num_snapshots = 30 # num snaps in cov matrix formation

    freqs = freqs[:13]
    outputs = []

    for source_freq in freqs:
        print('source freq', source_freq)

        t, x =  load_x(source_freq)
        t,x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq, data_stride)

        env = get_env(proj_str, source_freq)
        env = populate_env(env, source_freq, zs, zr, rr)
        kbar = np.mean(env.modes.k.real)
      
     
        r_center,K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, 1, 10)
        i = 0 
        for K_samp in K_samps:
            r, z, synth_reps = get_mult_el_reps(env, 1, 10, 10)
            output = get_amb_surf(r, z, K_samp, synth_reps)
            if source_freq == freqs[0]:
                outputs.append(output)
            else:
                outputs[i] += output
            i += 1

    for i in range(len(outputs)):
        out = outputs[i]
        out /= len(freqs)
        fig = plot_amb_surf(-10, r, z, out, 'incoh avg bart for snapshot ' + str(i), r_center[i], zs)
        plt.savefig('pics/' + proj_str + '/' +str(i).zfill(3) + '.png')
        plt.close(fig)

    os.system('ffmpeg -loglevel quiet -r 3 -f image2 -s 1920x1080 -i pics/' + proj_str + '/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p pics/' + proj_str + '/' +  'bart_stacked.mp4')

    for i in range(len(outputs)):
        os.system('rm pics/' + proj_str + '/'+ str(i).zfill(3) + '.png')
    return


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
        output = get_amb_surf(r, z, K_samp, synth_reps)
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
 
def synth_stacked_bart(num_synth_els,v, t):
    """ Run a simple incoherent average bartlett surface
    with synthetic aperture elements
    """
    zs = get_proj_zs(proj_str)
    zr = get_proj_zr(proj_str)
    data_stride = 4
    zr = zr[::data_stride]
    print('zr', zr)
    freqs = get_proj_tones(proj_str)

    num_snapshots = 2 # num snaps in cov matrix formation

    freqs = freqs[:13]

    """ GET BARTLETTS FOR SYNTH ARRAY """    
    outputs = []
    for source_freq in freqs: 
        print('source freq', source_freq)

        t, x =  load_x(source_freq,tag)
        print(t[0])
        t,x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq, data_stride)

        env = get_env(proj_str, source_freq)
        env = populate_env(env, source_freq, zs, zr, rr)
        kbar = np.mean(env.modes.k.real)
      
        delta_t, stride, T = get_synth_arr_params(source_freq, t, num_snapshots)
        r_center,K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, num_synth_els, stride)
        if source_freq == freqs[0]:
            num_frames = len(r_center)
        else:
            r_center, K_samps = r_center[:num_frames], K_samps[:num_frames]
            

        specific_v = vv[::num_snapshots][:len(r_center)] 
        ship_dr = np.median(specific_v)*T
        r,z=augment_outputs(outputs, r_center, K_samps, env, num_synth_els, T, specific_v)

    db_lev=  -10

    title_str = 'incoh avg synth stacked for snapshot '
    gen_incoh_avg_plots(outputs, r, z, r_center, zs, db_lev, freqs, proj_str, title_str)
    
    print(np.max(outputs[-1]), np.min(outputs[-1]))
    outputs = [x/len(freqs) for x in outputs]
    for i in range(len(outputs)):
        out = outputs[i]
        fig = plot_amb_surf(-10, r, z, out, 'incoh avg bart for snapshot ' + str(i), r_center[i], zs)
        plt.savefig('pics/' + proj_str + '/' +str(i).zfill(3) + '.png')
        plt.close(fig)

    os.system('ffmpeg -loglevel quiet -r 3 -f image2 -s 1920x1080 -i pics/' + proj_str + '/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p pics/' + proj_str + '/' +  str(num_synth_els) + '_synth_stacked.mp4')

    for i in range(len(outputs)):
        os.system('rm pics/' + proj_str + '/'+ str(i).zfill(3) + '.png')

    print(np.max(outputs[-1]), np.min(outputs[-1]))
    best_guesses = get_best_guesses(outputs)
    print(np.max(outputs[-1]), np.min(outputs[-1]))
    make_range_perform_plot(num_snapshots, r_center, t, best_guesses, source_freq, num_synth_els)
    

    
    """ GET BARTLETT FOR SINGLE EL ARRAY """
    single_el_outs = []

    for source_freq in freqs:
        print('source freq', source_freq)

        t, x =  load_x(source_freq)
        t,x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq, data_stride)

        env = get_env(proj_str, source_freq)
        env = populate_env(env, source_freq, zs, zr, rr)
        kbar = np.mean(env.modes.k.real)
      
     
        r_center,K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, 1, 10)
        specific_v = vv[::num_snapshots][:len(r_center)] 
        r,z=augment_outputs(single_el_outs, r_center, K_samps, env, 1, T, specific_v)

    title_str = 'incoh avg for simple vla for snapshot '
    gen_incoh_avg_plots(single_el_outs, r, z, r_center, zs, db_lev, freqs, proj_str, title_str)

    single_el_outs = [x/len(freqs) for x in single_el_outs]
    for i in range(len(single_el_outs)):
        out = single_el_outs[i]
        fig = plot_amb_surf(-10, r, z, out, 'incoh avg bart for snapshot ' + str(i), r_center[i], zs)
        plt.savefig('pics/' + proj_str + '/' +str(i).zfill(3) + '.png')
        plt.close(fig)

    if 'bart_stacked.mp4' not in os.listdir('pics/' + proj_str):
        os.system('ffmpeg -loglevel quiet -r 3 -f image2 -s 1920x1080 -i pics/' + proj_str + '/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p pics/' + proj_str + '/' +  'bart_stacked.mp4')

    for i in range(len(single_el_outs)):
        os.system('rm pics/' + proj_str + '/'+ str(i).zfill(3) + '.png')

    print(np.max(outputs[-1]), np.min(outputs[-1]))

    make_comp_plots(specific_v, T, num_synth_els, z, single_el_outs, outputs, db_lev, r_center, zs, wnc_mcm=False)
    title_str = 'synth_stacked'
    gen_vid(source_freq, num_synth_els, title_str, outputs)



    best_guesses = get_best_guesses(single_el_outs)
    make_range_perform_plot(num_snapshots, r_center, t, best_guesses, source_freq, 1)
    return

def make_str_id(num_synth_els, num_snapshots, num_freqs, v):
    str_id = str(num_synth_els) + '_' + str(num_snapshots) + '_' + str(num_freqs) + '_' + str(v)
    return str_id

class DataRun:
    def __init__(self, proj_str, num_snapshots, v, subfolder, rcvr_stride, num_freqs, num_synth_els):
        self.proj_str = proj_str
        self.num_snapshots = num_snapshots
        self.v = v
        self.subfolder = subfolder
        self.rcvr_stride = rcvr_stride
        self.num_freqs = num_freqs
        self.num_synth_els= num_synth_els
        self.str_id = make_str_id(num_synth_els, num_snapshots, num_freqs, v)

    def run(self):
        r_center, cov_t, r, z, outputs, max_vals = new_synth_stacked_bart(self)
        self.r_center = r_center
        self.cov_t = cov_t
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


def get_bart_outputs(drp):
    """
    For data run params drp, calculate the 
    stacked ambiguity surface (incoherently sum log 
    outputs over freq), and return the max vals of the amb 
    surf for each freq and time
    """
    freqs = get_proj_tones(drp.proj_str)[:drp.num_freqs]
    zs, zr = get_proj_zs(drp.proj_str), get_proj_zr(drp.proj_str)
    zr = zr[::drp.rcvr_stride]


    for freq_ind, source_freq in enumerate(freqs):
        t, x =  load_x(source_freq, drp.subfolder)
        t,x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq, drp.rcvr_stride)

        env = get_env(proj_str, source_freq)
        env = populate_env(env, source_freq, zs, zr, rr, fname=drp.proj_str)
        kbar = np.mean(env.modes.k.real)

        delta_t = t[1]-t[0]      
        delta_t, stride, T = get_synth_arr_params(source_freq, t, drp.num_snapshots)
        print('stride, T', stride, T)
     
        r_center, cov_t, K_samps = get_r_super_cov_seq(t, x,drp.num_snapshots, r_interp, drp.num_synth_els, stride)

        print(' num k samps', cov_t.size)
        r, z, synth_reps = get_mult_el_reps(env, drp.num_synth_els, drp.v, T,fname=drp.proj_str)
        if source_freq == freqs[0]:
            bart_outs = np.zeros((len(r_center), z.size, r.size))
            max_vals = np.zeros((len(freqs), len(r_center)))
            """ Annoying thing where I need to truncate based on
            first freq 
            """
            num_times = cov_t.size

        for i in range(num_times):
            K_samp = K_samps[i,:,:]
            output, max_val = get_amb_surf(r, z, K_samp, synth_reps)
            bart_outs[i,:,:] += output
            max_vals[freq_ind, i] = max_val
    bart_outs /= len(freqs)
    r_center = r_center[:num_times]
    cov_t = cov_t[:num_times] # make sure we truncate based on first freq
    return r_center, cov_t, r, z, bart_outs, max_vals
           
def new_synth_stacked_bart(drp):
    """
    Run a bartlett proc on the dta from proj_str with num_synth_els synthetic elements
    drp - DataRunParams obj holds errythagn
    """
    freqs = get_proj_tones(drp.proj_str)
    freqs = freqs[:drp.num_freqs]
    r_center, cov_t, r, z, outputs, max_vals = get_bart_outputs(drp)
    return r_center, cov_t, r, z, outputs, max_vals


def make_save_loc(proj_string, subfolder):
    if proj_string not in os.listdir('npy_files'):
        os.mkdir('npy_files/' + proj_string)
    if subfolder not in os.listdir('npy_files/' + proj_string):
        os.mkdir('npy_files/' + proj_string + '/' + subfolder)
    return 'npy_files/' + proj_string + '/' + subfolder + '/'

    
   
class DRPRuns:
    """
    Class to hold params for a suite of MFP runs 
    """
    def __init__(self, proj_str, subfolder, num_snap_list, rcvr_stride, num_freq_list, num_synth_el_list, vv):
        self.proj_str = proj_str
        self.subfolder = subfolder
        self.num_snap_list = num_snap_list
        self.rcvr_stride = rcvr_stride
        self.num_freq_list = num_freq_list  
        self.num_synth_el_list = num_synth_el_list
        self.vv = vv

    def run_all(self):
        """
        For each element in the tensor product of snapshot numbers, freqs, and synth els and velocities,
        create a drp, produce 
        """
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    if num_synth_els == 1:
                        v = -2.5
                        dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, self.rcvr_stride, num_freqs, num_synth_els)
                        dr.run()
                        dr.save()
                    else:
                        for v in self.vv:
                            dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, self.rcvr_stride, num_freqs, num_synth_els)
                            dr.run()
                            dr.save()

    def save(self, str_id):
        save_loc = make_save_loc(self.proj_str, self.subfolder)
        prefix = str_id
        fname = save_loc + prefix  + '.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        return

    def get_best_range_guesses(self):
        for num_snaps in self.num_snap_list:
            for num_freqs in self.num_freq_list:
                for num_synth_els in self.num_synth_el_list:
                    if num_synth_els == 1:
                        v = -2.5
                        dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, self.rcvr_stride, num_freqs, num_synth_els)
                        dr = load_dr(dr)
                        best_outs = dr.outputs
                        r_center = dr.r_center
                    else:
                        dr_list = []
                        for v in self.vv:
                            dr = DataRun(self.proj_str, num_snaps, v, self.subfolder, self.rcvr_stride, num_freqs, num_synth_els)
                            dr = load_dr(dr)
                            dr_list.append(dr)
                        best_outs = form_best_output_mat(dr_list)
                        r_center  = dr.r_center
                    best_guesses = []
                    for i in range(len(r_center)):
                        amb_surf = best_outs[i,:,:]
                        best_ind = int(np.argmax(amb_surf) % dr.r.size)
                        best_guess = dr.r[best_ind]    
                        best_guesses.append(best_guess)

                    plt.figure()
                    plt.suptitle('_'.join([str(num_snaps), str(num_freqs), str(num_synth_els)]) + 'bart')
                    plt.plot(dr.cov_t, r_center)
                    plt.plot(dr.cov_t, best_guesses)
        plt.show()
            

def get_max_val_arr(vv, r_center, max_val_list):
    max_val_arr = np.zeros((len(vv), len(r_center)))
    for j in range(len(vv)):
        v = vv[j]
        max_vals = max_val_list[j]
        max_val_arr[j, :] = np.sum(max_vals, axis=0)
    return max_val_arr

    max_val_arr = get_max_val_arr(vv, r_center, max_val_list)

    print('total_time', time.time() - t0)
    return max_val_arr

def form_best_output_mat(dr_list):
    """
    For a set of data runs where the only different parameter is
    velocity, form an array that has the maximum correlation value for each range
    """
    vv = [] 
    max_val_list = []
    outputs_list = []
    for dr in dr_list:
        max_val_list.append(dr.max_vals)
        vv.append(dr.v)
        outputs_list.append(dr.outputs)
    max_val_arr = get_max_val_arr(vv, dr.r_center, max_val_list)
    max_inds = np.argmax(max_val_arr, axis=0)
    best_outs = np.zeros((dr.outputs.shape))
    for i in range(len(dr.r_center)):
        best_outs[i,:,:] = outputs_list[max_inds[i]][i,:,:]
    return best_outs
    

if __name__ == '__main__':
    #simple_stacked_bart()
    t0 = time.time()
    num_snaps = 2
    subfolder = '4096'
    rcvr_stride = 4
    #num_freqs= 4
    #freqs = get_proj_tones(proj_str)[:num_freqs]
    num_synth_els = 1
    zs = get_proj_zs(proj_str)
    max_val_list = []
    vv = np.linspace(-2.6, -1.8, 15)
    vv = np.array([round(x, 4) for x in vv])
    num_snap_list = [10, 20]
    num_freq_list = [13]
    num_synth_el_list = [1, 5, 10, 15]

    dr_runs = DRPRuns(proj_str, subfolder, num_snap_list, rcvr_stride, num_freq_list, num_synth_el_list, vv)
    dr_runs.run_all()
    dr_runs.save('more_snaps')
    #with open('npy_files/' + proj_str + '/' + subfolder + '/' + 'quietest_fullset_run.pickle', 'rb') as f:
    #    dr_runs =  pickle.load(f)
    #    dr_runs.get_best_range_guesses()
    """
    output_list = []
    vv= [-2.5]
    for v in vv:
        v = round(v, 4)
        drp = DataRun(proj_str, num_snaps, v, subfolder, rcvr_stride, num_freqs, num_synth_els)
        #r_center, cov_t,r,z, outputs, max_vals = load_results(drp)
        drp.run()
        drp.save()

        #synth_stacked_bart(2,v) 
    """

