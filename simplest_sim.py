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
import pickle
from matplotlib import rc

rc('text', usetex=True)
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

def plot_amb_surf(db_range, r, z, amb_surf, title_str, r_true, zs, show=False):
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

def add_noise_cov(p_true, snr_db, num_snapshots):
    mean_pow = np.mean(np.square(abs(p_true)))
    noise_var = mean_pow/(np.power(10, snr_db/10))

    K_true = np.outer(p_true, p_true.conj())
    noise_K = np.zeros((K_true.shape), dtype=np.complex128)
    for i in range(num_snapshots):
        noise_vec = np.sqrt(noise_var/2)* np.random.randn(p_true.size) + complex(0,1)*np.sqrt(noise_var/2)*np.random.randn(p_true.size)
        noise_vec = noise_vec.reshape(p_true.shape)
        noise_K += np.outer(noise_vec, noise_vec.conj())
    noise_K /= num_snapshots

    #noise_K = noise_var*np.identity(p_true.size)  #+ complex(0,1)*noise_var/2 * np.identity(p_true.size)
    return K_true+noise_K

def get_amb_surf(r, z, K_true, replicas, matmul=False):
    amb_surf = np.zeros((z.size, r.size))
    if matmul == False:
        for i in range(z.size):
            for j in range(r.size):
                replica = replicas[:,i,j]
                #out = replica.conj().dot(K_true).dot(replica)
                replica= replica.reshape(replica.size, 1)
                out = replica.T.conj()@K_true@replica
                amb_surf[i,j] = abs(out)
    else:
        """ Get the replicas on the right side """
        """ Replicas shape is rcvrs.size, z.size, r.size """
        left_op = np.transpose(replicas, (1, 2,0)).conj()
        left_prod = left_op @ K_true
        out = np.einsum('ijk,kij->ij', left_prod, replicas)
        amb_surf = abs(out)
    max_val = np.max(amb_surf)
    #amb_surf /= np.max(amb_surf)
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

def get_env():
    env_builder = factory.create('swellex')
    env = env_builder()
    source_freq = 100
    dz = 5 
    zmax = 216.5
    dr = 2.5
    rmax = 1e4
    num_rcvrs = 21
    zr = np.linspace(100, 200, num_rcvrs)
    folder, fname=  'at_files/', 'simple'
    env.add_source_params(source_freq, zr, zr)
    env.add_field_params(dz, zmax, dr, rmax)
    return env

def form_replicas(env, rmin, rmax, grid_dr, v, cov_T, num_synth_els, folder, fname, adiabatic=False,tilt_angle=0):
    synth_dr = v*cov_T
    og_r = np.arange(rmin, rmax+ grid_dr, grid_dr)
    num_r = og_r.size
    custom_r = og_r[:]

    """ Form the range grid """
    for i in range(1, num_synth_els):
        tmp = og_r + synth_dr*i
        custom_r = np.concatenate((custom_r, tmp))

    """ Run the model """
    if adiabatic==False:
        replica, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=False, zr_range_flag=False, custom_r = custom_r, tilt_angle=tilt_angle)
    else:
        replica, pos = env.s5_approach_adiabatic_replicas(folder, fname, custom_r, tilt_angle=tilt_angle)

    """ Stack the replicas """
    synth_replicas = np.zeros((num_synth_els*env.zr.size, env.pos.r.depth.size,  og_r.size), dtype=np.complex128)
    for i in range(num_synth_els):
        synth_replicas[i*(env.zr.size):(i+1)*env.zr.size, :,:] = replica[:, :, i*num_r:(i+1)*num_r]
    synth_replicas /= np.linalg.norm(synth_replicas , axis=0)
    return og_r, pos.r.depth, synth_replicas

def gen_synth_data(env, num_synth_els, r0, ship_dr, folder, fname):
    """ 
    Initialize the environment
    Generate a synthetic data set
    """
    if num_synth_els == 1:
        true_r = np.array([r0])
    else:
        true_r = np.linspace(r0, r0 + (num_synth_els -1)*ship_dr, num_synth_els)
    print('True ranges', true_r)
    
    synth_data, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=true_r)
    """ Reshape true data into supervector """
    synth_p_true = synth_data.reshape(synth_data.size, 1, order='F')
    synth_p_true /= np.linalg.norm(synth_p_true)
    return true_r, synth_data, synth_p_true, pos

def get_mult_el_reps(env, num_synth_els, v, cov_T,fname='swell', adiabatic=False, tilt_angle=0):
    """
    Generate the replicas for multiple synthetic elements
    with an assumed range rate of v and snapshot time separation
    of T
    """
    folder=  'at_files/'
    r, z, synth_reps = form_replicas(env, rmin, rmax, grid_dr, v, cov_T, num_synth_els, folder, fname, adiabatic=adiabatic, tilt_angle=tilt_angle)
    return r, z, synth_reps

class MCObj:
    def __init__(self, snr_arr, rmse_arr, num_synth_els, num_snapshots, proc_type):
        self.snr_arr = snr_arr
        self.rmse_arr = rmse_arr
        self.proc_type = proc_type
        self.num_synth_els = num_synth_els
        self.num_snapshots = num_snapshots

rmin = 500
rmax = 1e4
grid_dr=  50

def get_MCObj(num_synth_els, num_snapshots, wnc, wn_gain):
    env = get_env()
    zs = 50
    T = 5 # time between snapshots
    cov_T = T*num_snapshots
    v = 2.5
    v_err = 1.00
    ship_dr= v*cov_T
    print('ship_dr', ship_dr)

    r, z, synth_reps = get_mult_el_reps(env, num_synth_els, v*v_err, cov_T, fname='swell')

    r0 = 5e3
    true_ind = np.argmin(np.array([abs(r0 - x) for x in r]))


    """ Get some data """
    env.zs = zs

    synth_p_snaps = []
    for k in range(num_snapshots):
        rs = r0 + k*v*T
        true_r, synth_data, synth_p_true, pos = gen_synth_data(env, num_synth_els, rs, ship_dr, 'at_files/', 'swell')
        synth_p_snaps.append(synth_p_true)

    r_true = r0 + (num_snapshots-1)/2*v*T
    print('r true', r_true)



    num_realizations = 1000
    snr_db_list = np.linspace(5, -30,  25)
    bart_rmse_arr = np.zeros((len(snr_db_list)))
    wnc_rmse_arr = np.zeros((len(snr_db_list)))
    for snr_ind, snr_db in enumerate(snr_db_list):

        print('snr db', snr_db)
        bart_sq_err_arr = np.zeros((num_realizations))
        wnc_sq_err_arr = np.zeros((num_realizations))
    

        for i in range(num_realizations):
            """ FOrm sample cov """
            for k in range(num_snapshots):
                K_tmp = add_noise_cov(synth_p_snaps[k], snr_db, 1)
                if k == 0:
                    K_true = K_tmp
                else:
                    K_true += K_tmp

            """ Get bart amb surf and sq err """
            bart_out, bart_max_val = get_amb_surf(r, z, K_true, synth_reps, matmul=True)
            best_ind = np.argmax(bart_out)
            best_range = r[best_ind % r.size]
            best_depth = r[best_ind // r.size]
            range_err = best_range - r_true
            sq_err = np.square(range_err)
            bart_sq_err_arr[i] = sq_err
            #plt.figure()
            #plt.xlabel('Range (m)')
            #plt.ylabel('Depth (z)')
            #plt.pcolormesh(r, z, bart_out/bart_max_val,vmax=0, vmin=-20)
            #plt.contourf(r, z, bart_out-np.max(bart_out))
            #plt.gca().invert_yaxis()
            #cb = plt.colorbar()
            #cb.set_label('dB', rotation='horizontal')
            #plt.scatter(r_true, zs, marker='+', color='k')
            #plt.show()

            """ Get wnc amb surf and sq err """
            if wnc == True:
                K_true = K_true.reshape(1, K_true.shape[0], K_true.shape[1])
                wnc_out = lookup_run_wnc(K_true, synth_reps, wn_gain)
                best_ind = np.argmax(wnc_out)
                best_range = r[best_ind % r.size]
                best_depth = r[best_ind // r.size]
                range_err = best_range - r_true
                sq_err = np.square(range_err)
                wnc_sq_err_arr[i] = sq_err
                wnc_db = 10*np.log10(wnc_out / np.max(wnc_out))
                levels = np.linspace(-10, 0, 20)
                #plt.figure()
                #plt.xlabel('Range (m)')
                #plt.ylabel('Depth (z)')
                ##plt.pcolormesh(r, z, wnc_db,vmax=0, vmin=-20)
                #print(wnc_db.shape)
                #plt.contourf(r, z, wnc_db[0,...])
                #plt.gca().invert_yaxis()
                #cb = plt.colorbar()
                #cb.set_label('dB', rotation='horizontal')
                #plt.scatter(r_true, zs, marker='+', color='k')
                #plt.show()


        bart_rmse = np.sqrt(np.mean(bart_sq_err_arr))
        bart_rmse_arr[snr_ind] = bart_rmse
        if wnc == True:
            wnc_rmse = np.sqrt(np.mean(wnc_sq_err_arr))
            wnc_rmse_arr[snr_ind] = wnc_rmse

    bart_mc = MCObj(np.array(snr_db_list), bart_rmse_arr, num_synth_els, num_snapshots, 'bart')
    if wnc == True:
        wnc_mc = MCObj(np.array(snr_db_list), wnc_rmse_arr, num_synth_els, num_snapshots, 'wnc_' + str(wn_gain))
        return bart_mc, wnc_mc
    else:
        return bart_mc


def mv_demo():
    num_synth_els = 5
    num_snapshots = 4
    wn_gain = -3

    env = get_env()
    zs = 50
    T = 5 # time between snapshots
    cov_T = T*num_snapshots
    v = 2.5
    v_err = 1.00
    ship_dr= v*cov_T
    print('ship_dr', ship_dr)

    r, z, synth_reps = get_mult_el_reps(env, num_synth_els, v*v_err, cov_T, fname='swell')

    r0 = 5e3
    true_ind = np.argmin(np.array([abs(r0 - x) for x in r]))


    """ Get some data """
    env.zs = zs

    synth_p_snaps = []
    for k in range(num_snapshots):
        rs = r0 + k*v*T
        true_r, synth_data, synth_p_true, pos = gen_synth_data(env, num_synth_els, rs, ship_dr, 'at_files/', 'swell')
        synth_p_snaps.append(synth_p_true)

    r_true = r0 + (num_snapshots-1)/2*v*T
    print('r true', r_true)



    num_realizations = 5
    snr_db_list = [-15,-10, -20]
    bart_rmse_arr = np.zeros((len(snr_db_list)))
    wnc_rmse_arr = np.zeros((len(snr_db_list)))
    for snr_ind, snr_db in enumerate(snr_db_list):

        print('snr db', snr_db)
        bart_sq_err_arr = np.zeros((num_realizations))
        wnc_sq_err_arr = np.zeros((num_realizations))
    

        for i in range(num_realizations):
            """ FOrm sample cov """
            for k in range(num_snapshots):
                K_tmp = add_noise_cov(synth_p_snaps[k], snr_db, 1)
                if k == 0:
                    K_true = K_tmp
                else:
                    K_true += K_tmp

            """ Get bart amb surf and sq err """
            bart_out, bart_max_val = get_amb_surf(r, z, K_true, synth_reps, matmul=True)
            best_ind = np.argmax(bart_out)
            best_range = r[best_ind % r.size]
            best_depth = r[best_ind // r.size]
            range_err = best_range - r_true
            sq_err = np.square(range_err)
            bart_sq_err_arr[i] = sq_err
            #plt.figure()
            #plt.xlabel('Range (m)')
            #plt.ylabel('Depth (z)')
            #plt.pcolormesh(r, z, bart_out/bart_max_val,vmax=0, vmin=-20)
            #plt.contourf(r, z, bart_out-np.max(bart_out))
            #plt.gca().invert_yaxis()
            #cb = plt.colorbar()
            #cb.set_label('dB', rotation='horizontal')
            #plt.scatter(r_true, zs, marker='+', color='k')
            #plt.show()

            K_true = K_true.reshape(1, K_true.shape[0], K_true.shape[1])
            wnc_out = lookup_run_wnc(K_true, synth_reps, wn_gain)
            best_ind = np.argmax(wnc_out)
            best_range = r[best_ind % r.size]
            best_depth = r[best_ind // r.size]
            range_err = best_range - r_true
            sq_err = np.square(range_err)
            wnc_sq_err_arr[i] = sq_err
            wnc_db = 10*np.log10(wnc_out / np.max(wnc_out))
            #levels = np.linspace(-10, 0, 20)
            #plt.figure()
            #plt.xlabel('Range (m)')
            #plt.ylabel('Depth (z)')
            ##plt.pcolormesh(r, z, wnc_db,vmax=0, vmin=-20)
            #print(wnc_db.shape)
            #plt.contourf(r, z, wnc_db[0,...])
            #plt.gca().invert_yaxis()
            #cb = plt.colorbar()
            #cb.set_label('dB', rotation='horizontal')
            #plt.scatter(r_true, zs, marker='+', color='k')
            #plt.show()


        bart_rmse = np.sqrt(np.mean(bart_sq_err_arr))
        bart_rmse_arr[snr_ind] = bart_rmse
        wnc_rmse = np.sqrt(np.mean(wnc_sq_err_arr))
        wnc_rmse_arr[snr_ind] = wnc_rmse

    bart_mc = MCObj(np.array(snr_db_list), bart_rmse_arr, num_synth_els, num_snapshots, 'bart')
    wnc_mc = MCObj(np.array(snr_db_list), wnc_rmse_arr, num_synth_els, num_snapshots, 'wnc_' + str(wn_gain))
    return bart_mc, wnc_mc


def make_mc_name(mc_obj, root_folder='pickles/'):
    name = '_'.join([str(mc_obj.num_synth_els), str(mc_obj.num_snapshots), mc_obj.proc_type])
    pname = root_folder + name + '.pickle'
    return pname

def save_mc(mc_obj, root_folder):
    pname = make_mc_name(mc_obj, root_folder)
    with open(pname, 'wb') as f:
        pickle.dump(mc_obj, f)
    return

def load_mc(num_synth_els, num_snapshots, proc_type, root_folder='pickles/'):
    name = '_'.join([str(num_synth_els), str(num_snapshots), proc_type]) + '.pickle'
    pname = root_folder + name
    with open(pname, 'rb') as f:
        mc_obj = pickle.load(f)
    return mc_obj
    
    

if __name__ == '__main__':
    
    #bart_mc = load_mc(5,4,'bart')
    #wnc_mc = load_mc(5,4,'wnc_-3')
    #bart_mc, wnc_mc = mv_demo()
    #save_mc(bart_mc, 'rebut/')
    #save_mc(wnc_mc, 'rebut/')
    #fig, ax = plt.subplots(1,1)
    #ax.plot(bart_mc.snr_arr, bart_mc.rmse_arr/1000, color='r', marker='*',linestyle='-')
    #ax.plot(wnc_mc.snr_arr, wnc_mc.rmse_arr/1000, color='b', marker='*',linestyle='-')
    #plt.show()

    start_time = time.time()


    fig, ax = plt.subplots(1,1)
    plt.grid()
    wn_gain = -1
    wnc=True
    load = False
    #load = True

    if load == False:
        if wnc == False:
            bart_mc = get_MCObj(1,4,wnc,wn_gain)
            save_mc(bart_mc, 'pickles/')
        else:
            bart_mc, wnc_mc = get_MCObj(1,4,wnc,wn_gain)
            save_mc(bart_mc, 'pickles/')
            save_mc(wnc_mc, 'pickles/')
    else:
        bart_mc = load_mc(1, 4, 'bart') 
        if wnc == True:
            wnc_mc = load_mc(1, 4, 'wnc') 
            

    ax.plot(bart_mc.snr_arr, bart_mc.rmse_arr/1000, color='r', marker='*',linestyle='-')

    if load == False:
        if wnc == False:
            bart_mc = get_MCObj(5, 4, wnc, wn_gain)
            save_mc(bart_mc, 'pickles/')
        else:
            bart_mc, wnc_obj = get_MCObj(5, 4, wnc, wn_gain)
            save_mc(bart_mc, 'pickles/')
            save_mc(wnc_mc, 'pickles/')
    else:
        bart_mc = load_mc(5, 4, 'bart') 
        if wnc == True:
            wnc_mc = load_mc(5, 4, 'wnc') 

    ax.plot(bart_mc.snr_arr, bart_mc.rmse_arr/1000, color='b', marker='x', linestyle='-.')

    if load == False:
        if wnc == False:
            bart_mc = get_MCObj(10, 4, wnc, wn_gain)
            save_mc(bart_mc, 'pickles/')
        else:
            bart_mc, wnc_mc = get_MCObj(10, 4, wnc, wn_gain)
            save_mc(bart_mc, 'pickles/')
            save_mc(wnc_mc, 'pickles/')
    else:
        bart_mc = load_mc(10, 4, 'bart')
        if wnc == True:
            wnc_mc = load_mc(10, 4, 'bart')

    ax.plot(bart_mc.snr_arr, bart_mc.rmse_arr/1000, color='k', marker='+', linestyle='--')

    plt.legend(['No synth els', '$N_{syn} = 5$', '$N_{syn}=10$'])
    ax.set_ylabel('RMSE (km)', fontsize=15)
    ax.text(-19.5, 2 * .8/20, 'b)', fontsize=15,color='k')
    ax.set_xlabel('Input SNR (dB)', fontsize=15)
    ax.set_xlim([-20, 5])
    ax.set_ylim([0, 0.02])


    #fig_name = '/home/hunter/research/coherent_matched_field/paper/pics/mc_sim.png'
    #plt.savefig(fig_name)
    plt.show()


    bart_mc
    

