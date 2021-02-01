import numpy as np
from numba import jit, cuda
import sys
from copy import deepcopy
from matplotlib import pyplot as plt
from env.env.envs import factory
import multiprocessing as mp
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc
from signal_proc.mfp.mcm import run_mcm
import time

'''
Description:
Just do a really simple simulation with no source phase
A planar array versus a vla

Add the noise as a diagonal matrix to the 'covariance' matrix

Date: 
10/19/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def add_noise(p_true, snr_db):
    mean_pow = np.mean(np.square(abs(p_true)))
    noise_var = mean_pow/(np.power(10, snr_db/10))
    noise_vec = np.sqrt(noise_var/2)* np.random.randn(p_true.size) + complex(0,1)*np.sqrt(noise_var/2)*np.random.randn(p_true.size)
    noise_vec = noise_vec.reshape(p_true.shape)
    mat = np.outer(noise_vec, noise_vec.conj())
    plt.figure()
    plt.imshow(abs(mat))
    p_true = p_true + noise_vec
    return p_true

def add_noise_cov(p_true, snr_db):
    mean_pow = np.mean(np.square(abs(p_true)))
    noise_var = mean_pow/(np.power(10, snr_db/10))
    #noise_vec = np.sqrt(noise_var/2)* np.random.randn(p_true.size) + complex(0,1)*np.sqrt(noise_var/2)*np.random.randn(p_true.size)
    #noise_vec = noise_vec.reshape(p_true.shape)
    K_true = np.outer(p_true, p_true.conj())
    noise_K = noise_var*np.identity(p_true.size)  #+ complex(0,1)*noise_var/2 * np.identity(p_true.size)
    return K_true+noise_K

def get_amb_surf(r, z, K_true, replicas):
    amb_surf = np.zeros((z.size, r.size))
    for i in range(z.size):
        for j in range(r.size):
            replica = replicas[:,i,j]
            #out = replica.conj().dot(K_true).dot(replica)
            replica= replica.reshape(replica.size, 1)
            out = replica.T.conj()@K_true@replica
            amb_surf[i,j] = abs(out)
    max_val = np.max(amb_surf)
    amb_surf /= np.max(amb_surf)
    amb_surf = 10*np.log10(amb_surf)
    return amb_surf, max_val

def get_mvdr_amb_surf(r, z, K_true, replicas):
    amb_surf = np.zeros((z.size, r.size))
    cond_num = np.linalg.cond(K_true)
    eps = 1e-12
    loop_count = 0
    while cond_num > 1e9:
        print('covariance matrix is ill-conditioned, regularizing before inversion', cond_num)
        K_true += eps*np.identity(K_true.shape[0])
        cond_num = np.linalg.cond(K_true)
        print('new cond num', cond_num)
        loop_count += 1
        if loop_count % 10 == 0:
            eps = eps*10
            print('increasing epsilon', eps)
    K_true_inv = np.linalg.inv(K_true)
    for i in range(z.size):
        for j in range(r.size):
            replica = replicas[:,i,j]
            denom = (replica.T.conj()@K_true_inv@replica).real
            amb_surf[i,j] = 1/denom
    amb_surf /= np.max(abs(amb_surf))
    amb_surf = 10*np.log10(abs(amb_surf))
    return amb_surf

def plot_amb_surf(db_range, r, z, amb_surf, title_str, r_true, zs, show=False):
    """
    """
    levels = np.linspace(db_range[0], db_range[1], 20)
    fig = plt.figure()
    plt.contourf(r, z, amb_surf, levels=levels, extend='both')
    plt.colorbar()
    plt.scatter(r_true, zs, zs, marker='+', color='r')
    ind = np.argmax(amb_surf)
    inds = (ind % r.size, int(ind // r.size))
    plt.scatter(r[inds[0]], z[inds[1]], marker='+', color='b')
    plt.gca().invert_yaxis()
    fig.suptitle(title_str)
    if show == True:
        plt.show()
    return fig

def mvdr_quick_plot(r, z, K_true, replicas, db_scale, title_str, r_true, zs):
    output = get_mvdr_amb_surf(r, z, K_true, replicas)
    plot_amb_surf(db_scale, r, z, output, title_str + 'mvdr', r_true, zs, show=False)
    return

def bart_quick_plot(r, z, K_true, replicas, db_scale, title_str, r_true, zs):
    output = get_amb_surf(r, z, K_true, replicas)
    plot_amb_surf(db_scale, r, z, output, title_str + 'bart', r_true, zs, show=False)
    return

def wnc_quick_plot(r, z, K_true, replicas, db_scale, wn_gain, title_str, r_true, zs):
    output = lookup_run_wnc(K_true, replicas, wn_gain)
    output = 10*np.log10(abs(output/np.max(abs(output))))
    plot_amb_surf(db_scale, r, z, output[:,:,0], title_str + 'wn gain ' + str(wn_gain), r_true, zs, show=False)
    return
    
def quick_plot_suite(r, z, K_true, replicas, db_scale, title_str, r_true, zs):
    """
    p1 = mp.Process(target = mvdr_quick_plot, args=(r, z, K_true, replicas, db_scale, title_str, r_true, zs))
    p1.start()
    #mvdr_quick_plot(r, z, K_true, replicas, db_scale, title_str, r_true, zs)
    p2 = mp.Process(target = wnc_quick_plot, args = (r, z, K_true, replicas, db_scale, -1, title_str, r_true, zs))
    p2.start()
    p3 = mp.Process(target = wnc_quick_plot, args = (r, z, K_true, replicas, db_scale, -3, title_str, r_true, zs))
    p3.start()
    p4 = mp.Process(target = wnc_quick_plot, args = (r, z, K_true, replicas, db_scale, -6, title_str, r_true, zs))
    p4.start()
    p5 = mp.Process(target = bart_quick_plot, args = (r, z, K_true, replicas, db_scale, title_str, r_true, zs))
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    """
    mvdr_quick_plot(r, z, K_true[:,:,0], replicas, db_scale, title_str, r_true, zs)
    wnc_quick_plot(r, z, K_true, replicas, db_scale, -1, title_str, r_true, zs)
    wnc_quick_plot(r, z, K_true, replicas, db_scale, -3, title_str, r_true, zs)
    wnc_quick_plot(r, z, K_true, replicas, db_scale, -6, title_str, r_true, zs)
    bart_quick_plot(r, z, K_true[:,:,0], replicas, db_scale, title_str, r_true, zs)
    return

if __name__ == '__main__':

    start_time = time.time()
    freq = 50
    num_rcvrs = 2
    zr = np.linspace(50, 200, num_rcvrs)
    zs = 50


    dz = 5
    zmax = 216.5
    dr = 25
    rmax = 1e4

    env_builder = factory.create('swellex')
    env = env_builder()
    folder = 'at_files/'
    fname = 'simple'
    env.add_source_params(freq, zs, zr)
    env.add_field_params(dz, zmax, dr, rmax)
    p, pos = env.run_model('kraken', folder, fname, zr_flag=True, zr_range_flag=False)
    p1 = p
    env.add_source_params(freq, zr, zr)
    replicas, pos = env.run_model('kraken', folder, fname, zr_flag=False, zr_range_flag=False)
    replicas /= np.linalg.norm(replicas, axis=0)
    #env.add_source_params(freq, zs, zr)
    #custom_r = np.arange(dr, rmax+dr, dr)
    #x, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=custom_r)


    r0 = 5e3
    true_ind = np.argmin(np.array([abs(r0 - x) for x in pos.r.range]))
    r0 = pos.r.range[true_ind]
    p_true = p[:,true_ind]
    r_true = pos.r.range[true_ind]
    print('r true, zs', r_true, zs)
    snr_db = -15

    K_tmp = add_noise_cov(p_true, snr_db)
    K_true = K_tmp
    print('Reshaping K_true to hae a time axis')
    K_true=K_true.reshape(1, K_true.shape[0], K_true.shape[1])
    
    #output = lookup_run_wnc(K_true, replicas, -1)
    #p_true = add_noise(p_true, snr_db)
    #K_true = np.outer(p_true, p_true.conj())

    """
    MCM Test

    """
    r_list = [replicas[:, 1:-1,1:-1], replicas[:, :-2, :-2], replicas[:, 2:, :-2], replicas[:, :-2, 2:], replicas[:, 2:, 2:]]
    output = run_mcm(K_true,r_list)
    output = output[:,:,0]
    output = 10*np.log10(abs(output) / np.max(abs(output)))
    #plot_amb_surf(-10, pos.r.range[1:-1], pos.r.depth[1:-1],  output, 'mcm', r0, zs)


    now = time.time()
    db_scale = -5
    quick_plot_suite(pos.r.range,pos.r.depth, K_true, replicas, db_scale, 'standard ',r_true, zs)

    print('time elapsed' , time.time()-now)


    stride = 1
    num_synth_els = 5
    synth_dr = stride*dr
    synthetic_array = p[:,true_ind:true_ind+10:stride]
    synth_p_true = synthetic_array.reshape(synthetic_array.size,1, order='F')
    synth_p_true = deepcopy(synth_p_true)
    synth_K_true =add_noise_cov(synth_p_true, snr_db)
    print('Reshaping K_true to hae a time axis')
    synth_K_true=synth_K_true.reshape(1, synth_K_true.shape[0], synth_K_true.shape[1])
    #synth_p_true = add_noise(synth_p_true, snr_db)
    #synth_K_true = np.outer(synth_p_true, synth_p_true.conj())

    synth_r = pos.r.range[true_ind:true_ind+10:stride]
    print('synth spacing', synth_r[1]-synth_r[0], ' meters')
    num_synth_els = synth_r.size
    print('number synth els' ,num_synth_els)


    num_ranges=  pos.r.range.size
    num_depths = pos.r.depth.size
    synth_rep = np.zeros((num_rcvrs*num_synth_els, num_depths, num_ranges), dtype=np.complex128)
    hanger= 0
    for i in range(num_synth_els):
        vals = replicas[:,:,stride*i:]
        synth_rep[i*num_rcvrs:(i+1)*num_rcvrs, :, :vals.shape[-1]] = vals
        num_leftovers = num_ranges - vals.shape[-1]
        hanger = np.max([num_leftovers, hanger])
        print('num left', num_leftovers)

    synth_rep = synth_rep[:,:,:-hanger]
    synth_pos = deepcopy(pos)
    synth_pos.r.range = synth_pos.r.range[:-hanger]
    print(synth_rep.shape, synth_pos.r.range.shape, synth_pos.r.depth.shape)
    synth_rep = synth_rep.copy(order='c')
    now = time.time()

    quick_plot_suite(synth_pos.r.range, synth_pos.r.depth, synth_K_true, synth_rep, db_scale, 'synthetic ', r_true, zs)
    print('synth quick plot', time.time() - now)
    print('total_time', time.time() - start_time)

    plt.show()
