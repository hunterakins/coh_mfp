import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from env.env.envs import factory
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc
from signal_proc.mfp.mcm import run_mcm
from coh_mfp.simplest_sim import get_amb_surf, get_mvdr_amb_surf, plot_amb_surf

'''
Description:
Main function willlll....
Simulate the field, then simulate replicas with velocity error
to look at sensitivity of ambiguity surface to velocity knowledge

I also want to export a function that generates some 
canonical field as a function of assumed velocity 


Date: 
10/19/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


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

def get_mainlobe_sinc_approx(v_grid, T, rough_k, num_synth_els, ship_dr):
    replica_drs = v_grid*T
    range_step_errs = ship_dr - replica_drs
    modeled_shape = np.sinc(rough_k*range_step_errs*num_synth_els/2/np.pi)
    return modeled_shape

def get_rough_phase(kr, true_r, synth_data):
    rough_k = np.mean(kr).real
    rough_phase = rough_k*(true_r-true_r[0]) + np.angle(synth_data[0,0])
    rough_phase = np.angle(np.exp(complex(0,1)*rough_phase))
    return rough_k, rough_phase

def get_bartlett_fn_of_v(v_grid, T, r0, num_synth_els, folder, fname, synth_p_true, synth_replica):
    bartlett_grid = np.zeros((v_grid.size))
    for i, v in enumerate(v_grid):
        replica_ship_dr = v*T
        custom_r = np.linspace(r0, r0 + (num_synth_els -1)*replica_ship_dr, num_synth_els)
        synth_replica, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=custom_r)
        synth_replica = synth_replica.reshape(synth_replica.size, 1, order='F')
        synth_replica/=np.linalg.norm(synth_replica, axis=0)
        tmp_bartlett = np.square(abs(synth_p_true.conj().T@synth_replica))
        bartlett_grid[i] = tmp_bartlett
    return bartlett_grid

def get_v_replicas(replica_v, T, env, folder, fname, num_synth_els, dz, zmax, rmax):
    replica_ship_dr = replica_v*T
    custom_r = np.arange(replica_ship_dr, rmax, replica_ship_dr)
    synth_replicas = np.zeros((zr.size, env.pos.r.depth.size, custom_r.size), dtype=np.complex128)
    env.add_field_params(dz, zmax, replica_ship_dr, rmax)
    env.add_source_params(freq, zr, zr)
    synth_replica, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=False, zr_range_flag=False, custom_r=custom_r)
    #tmp_bartlett = np.square(abs(synth_p_true.conj().T@synth_replica))
    synth_replica/=np.linalg.norm(synth_replica, axis=0)
    return synth_replica, pos

#def run_mcm


if __name__ == '__main__':

    """ Set basic experimental params """
    freq = 50
    num_rcvrs = 25
    zr = np.linspace(50, 200, num_rcvrs) # array depths
    zs = 50 # source depth

    r0 = 5*1e3 # initial source range
    ship_v = 5 # m/s
    T = 10 # time between snapshots in synth array
    ship_dr = ship_v*T
    num_synth_els = 10

    dz = 5
    zmax = 216.5
    dr = 50
    rmax = 1e4
    env_builder = factory.create('swellex')
    env = env_builder()
    folder = 'at_files/'
    fname = 'simple'
    env.add_source_params(freq, zs, zr)
    env.add_field_params(dz, zmax, dr, rmax)

    true_r, synth_data, synth_p_true, pos = gen_synth_data(env, num_synth_els, r0, ship_dr, folder, fname)

    
    p_true = synth_p_true[:zr.size]
    p_true /= np.linalg.norm(p_true)
    K_true = np.outer(p_true, p_true.conj())
    synth_replica, rep_pos =get_v_replicas(ship_v, T, env, folder, fname, num_synth_els, dz, zmax, rmax)
    
    rep_r_ind = np.argmin((np.array([abs(x*1e3 - r0) for x in rep_pos.r.range])))
    rep_z_ind = np.argmin((np.array([abs(zs - x) for x in rep_pos.r.depth])))
    rep_true = synth_replica[:, rep_z_ind, rep_r_ind]
    output = get_amb_surf(rep_pos.r.range, rep_pos.r.depth, K_true, synth_replica)
    print(rep_true.conj().T@p_true)
    print(rep_pos.r.range)
    plot_amb_surf(-20, rep_pos.r.range*1e3, rep_pos.r.depth, output, 'Correct velocity', r0, zs)
    plt.show()


    
    modes = env.modes
    kr = modes.k

    rough_k, rough_phase = get_rough_phase(kr, true_r, synth_data)

    """ Generate a set of replicas for the true initial position 
    but with a velocity error """
    v_grid = np.linspace(1, 10, 200)

    modeled_shape = get_mainlobe_sinc_approx(v_grid, T, rough_k, num_synth_els, ship_dr)
    modeled_shape = np.square(abs(modeled_shape))
    bartlett_grid = get_bartlett_fn_of_v(v_grid, T, r0, num_synth_els, folder, fname, synth_p_true, synth_replica)

    fig = plt.figure()
    plt.plot(v_grid*T, bartlett_grid)
    plt.xlabel('Synthetic aperture replica range spacing (meters)')
    plt.scatter(ship_v*T, 1, marker='+', color='r')
    plt.text(ship_v*.2*T, .75, 'red marker corresponds to true range spacing')
    plt.ylabel('Bartlett power')
    plt.suptitle('Bartlett output for correct initial range\n but incorrect synthetic array range calibration')



    fig = plt.figure()
    plt.plot(v_grid*T, bartlett_grid)
    plt.plot(v_grid*T, modeled_shape)
    plt.xlabel('Synthetic aperture replica range spacing (meters)')
    plt.ylabel('Bartlett power')
    plt.suptitle('Comparison of sinc function model for mainlobe to the actual mainlobe')
    plt.legend(['True Bartlett shape', 'sinc approximation to Bartlett\n response using mean wavenumber'])


    plt.figure()
    peak_inds, peak_heights = find_peaks(bartlett_grid, height = 0.9)
    for peak_ind in peak_inds:
        v = v_grid[peak_ind]
        replica_ship_dr = v*T
        custom_r = np.linspace(r0, r0 + (num_synth_els -1)*replica_ship_dr, num_synth_els)
        synth_replica, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=custom_r)
        for j in range(synth_replica.shape[0]):
            phase = np.angle(synth_replica[j,:])
            plt.plot(phase, color='b')
        synth_replica = synth_replica.reshape(synth_replica.size, 1, order='F')
        synth_replica/=np.linalg.norm(synth_replica)

    for j in range(synth_data.shape[0]):
        phase = np.angle(synth_data[j,:])
        plt.plot(phase , color='r')
    plt.show()

        
    print(peak_inds, peak_heights)


