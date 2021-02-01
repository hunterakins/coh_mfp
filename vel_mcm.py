import numpy as np
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from coh_mfp.vel_err_sim import get_mainlobe_sinc_approx, gen_synth_data
from coh_mfp.simplest_sim import get_mvdr_amb_surf, get_amb_surf, get_mvdr_amb_surf, plot_amb_surf, add_noise, add_noise_cov
from env.env.envs import factory
from signal_proc.mfp.mcm import run_mcm, run_wnc_mcm
from signal_proc.mfp.wnc import run_wnc

"""
Description:
Run MCM on a synthetic aperture using 'adjacent' velocity values


Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


def get_v_dr_grid(ship_dr, num_synth_els, num_constraints, k_bar):
    """
    Select a grid of velocity points in the vicinity of 
    ship_v, with the constraint that they sit in the 
    bartlett mainlobe (w.r.t. assumed velocity as the 
    look parameter)
    sinc(x) = sin(x)/x
    The sinc function has argument:
        sinc(x), x = k_bar * dr_err * num_synth_els / 2
    sinc has 3db  down point at x approx 1.9
    so the max range step error in the constraint mainlobe
    is dr_err = 1.9 * (2 pi / kbar / num_synth_els)
    Input 
    ship_dr - float
        true value of the ship's motion between data frames
    num_synth_els - int
        number of elements in the synthetic aperture
    num_constraints - int
        number of other guesses for dr
    k_bar - float
        estimate of modal wavenumber
    """
    dr_err_max = 1.9 * 2 / k_bar / num_synth_els
    v_dr_grid = np.linspace(ship_dr - dr_err_max, ship_dr + dr_err_max, num_constraints)
    return v_dr_grid
    
def check_constraint_points(v_grid, T, kbar, num_synth_els, ship_dr):
    modeled_shape = get_mainlobe_sinc_approx(v_grid, T, kbar, num_synth_els, ship_dr)
    v_alt = np.linspace(3, 7, 100)
    full_shape = get_mainlobe_sinc_approx(v_alt, T, kbar, num_synth_els, ship_dr)
    plt.scatter(v_grid, modeled_shape, color='r')
    plt.plot(v_alt, full_shape)
    plt.xlabel('Velocity used to calibrate the synthetic array (m/s)')
    plt.ylabel('Bartlett response as a function of assumed v for correct r0')
    plt.legend(['Constraint velocities', 'Approximate mainlobe shape'])
    plt.show()
    return

def form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder, fname, adiabatic=False):
    synth_dr = v*T
    og_r = np.arange(rmin, rmax+ grid_dr, grid_dr)
    for i in range(num_synth_els):
        custom_r = og_r + synth_dr*i
        if adiabatic==False:
            replica, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=False, zr_range_flag=False, custom_r = custom_r)
        else:
            replica, pos = env.s5_approach_adiabatic_replicas(folder, fname, custom_r)
        if i == 0 :
            synth_replicas = np.zeros((num_synth_els*env.zr.size, env.pos.r.depth.size,  og_r.size), dtype=np.complex128)
        synth_replicas[i*(env.zr.size):(i+1)*env.zr.size, :,:] = replica
    synth_replicas /= np.linalg.norm(synth_replicas , axis=0)
    return og_r, pos.r.depth, synth_replicas

def mcm_plot(K_true, r_arr, db_down, look_ind, r, z, rs, zs):
    output = run_wnc_mcm(K_true.reshape(1, K_true.shape[0], K_true.shape[1]), r_arr, db_down, look_ind=look_ind)
    output= 10*np.log10(abs(output) / np.max(output))
    output = output[0,:,:]
    plot_amb_surf(-5, r, z, output, 'mcm wnc ' + str(db_down), rs, zs, show=True)

def mcm_mvdr_comp(env, rmin, rmax, grid_dr, r0, v_grid, T, num_synth_els, K_true, folder, fname,rs, zs, f_err=0):
    """
    Compare MCM and MVDR processor, using assumed source velocity as constraint points
    Input 
    env
    """ 
    K_true = K_true.reshape(1, K_true.shape[0], K_true.shape[1])
    wn_gain = -3
    fig, axis = plt.subplots(1,1)
    plt.suptitle('Compare MCM and WNC output at the correct source depth')
    axis.set_xlabel('Range (m)')
    axis.set_ylabel('WNC output (dB)')
    for i, v in enumerate(v_grid):
        r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder, fname)
        print('synth dr in func ', v*T)
        if i == 0:
            r_arr = np.zeros((len(v_grid), num_synth_els*env.zr.size, z.size, r.size), dtype=np.complex128)
        synth_replicas = add_f_err(synth_replicas, num_synth_els, T, env.freq, env.freq+f_err)
        r_arr[i,...] = synth_replicas[...]
        z_ind = np.argmin(np.array([abs(zs- x) for x in z]))
        #output = get_amb_surf(r, z, K_true, synth_replicas)
        #plot_amb_surf(-5, r, z, output, 'bartlett, v = ' + str(v), rs, zs)
        mvdr = run_wnc(K_true, synth_replicas, wn_gain)
        mvdr = mvdr[0,...]
        mvdr = 10*np.log10(abs(mvdr)/np.max(mvdr))
        #plot_amb_surf(-10, r, z, mvdr, 'g, v = ' + str(v), rs, zs)
        axis.plot(r, mvdr[z_ind, :])
    num_constraints = len(v_grid)
    look_ind = int(num_constraints // 2) 
    print('look ind', look_ind)

    leg = [str(x)[:4]+ ' m /s assumed speed' for x in v_grid]
    if num_synth_els > 1:
        mcm = run_wnc_mcm(K_true,r_arr, wn_gain, look_ind=look_ind)
        mcm = mcm[0,...]  
        mcm = 10*np.log10(abs(mcm)/np.max(mcm))
        axis.plot(r, mcm[z_ind,:])
        plot_amb_surf(-10, r, z, mcm, 'mcm', rs, zs)
        leg += ['MCM']
    axis.legend(leg)
    
    plt.show()
   
def add_f_err(replicas, num_synth_els, T, f_true, f_assumed):
    f_err = f_assumed-f_true
    zr_size = int(replicas.shape[0]//num_synth_els)
    for i in range(num_synth_els):
        phase_err = 2*np.pi*f_err*(T*i)
        multiplier = np.exp(complex(0,1)*phase_err)
        replicas[i*zr_size:(i+1)*zr_size,:,:] *= multiplier
    return replicas
    
def examine_f_sens(env, rmin, rmax, grid_dr, v, T, num_synth_els, K_true, zs, rs, folder, fname):
    f_true = env.freq 
    b_vals = []
    df_db_down = 1.9 / (T*num_synth_els) / np.pi
    f_min = f_true - 2*df_db_down
    f_max = f_true + 2*df_db_down
    f_vals = np.linspace(f_min, f_max, 40)
    plt.figure() 
    for f_assumed in f_vals:
        r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder,fname)
        r_ind = np.argmin(np.array([abs(rs - x) for x in r]))
        z_ind = np.argmin(np.array([abs(zs- x) for x in z]))
        synth_replicas = add_f_err(synth_replicas, num_synth_els, T, f_true, f_assumed)
        synth_rep = synth_replicas[:, z_ind, r_ind]
        synth_rep /= np.linalg.norm(synth_rep)
        #output = get_amb_surf(r, z, K_true, synth_replicas)
        output = synth_rep.T.conj()@K_true@synth_rep
        output = output.real
        b_vals.append(output)

    plt.plot(f_vals, b_vals, color='b')
    f_min = f_true - df_db_down
    f_max = f_true + df_db_down
    f_vals = np.linspace(f_min, f_max, 10)
    c_vals = []
    for f_assumed in f_vals:
        r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder_fname)
        r_ind = np.argmin(np.array([abs(rs - x) for x in r]))
        z_ind = np.argmin(np.array([abs(zs- x) for x in z]))
        synth_replicas = add_f_err(synth_replicas, num_synth_els, T, f_true, f_assumed)
        synth_rep = synth_replicas[:, z_ind, r_ind]
        synth_rep /= np.linalg.norm(synth_rep)
        #output = get_amb_surf(r, z, K_true, synth_replicas)
        output = synth_rep.T.conj()@K_true@synth_rep
        output = output.real
        c_vals.append(output)
    plt.scatter(f_vals, c_vals, color='r')

def get_r_arr(env, rmin, rmax, grid_dr, v_grid, T, num_synth_els, folder, fname, f_err=0):
    """
    For the environment, generate array of replicas
    First axis is constraint point, second is receiver index, third
    is depth, fourth is range 
    """
    for i, v in enumerate(v_grid):
        r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, v, T, num_synth_els, folder, fname)
        if i == 0:
            r_arr = np.zeros((len(v_grid), num_synth_els*env.zr.size, z.size, r.size), dtype=np.complex128)
        synth_replicas = add_f_err(synth_replicas, num_synth_els, T, env.freq, env.freq+f_err)
        synth_replicas /= np.linalg.norm(synth_replicas, axis=0)
        r_arr[i,:,:,:] = synth_replicas[:,:,:]
    return r, z, r_arr

def three_plot_comp(env, rmin, rmax, grid_dr, ship_v, v_grid, T, num_synth_els, K_true, snr, rs, zs,f_err=0):
    """ 
    Compare bartlett to MCM and MVDR for synthetic aperture
    """
    wnc_db = -3

    """ 
    Produce Bartlett 
    """
    r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, ship_v, T, num_synth_els, folder, fname)
    print('new k', env.modes.k)
    bart = get_amb_surf(r, z, K_true, synth_replicas)

    """ 
    Now do MCM for v_grid 
    """


    r, z, r_arr = get_r_arr(env, rmin, rmax, grid_dr, v_grid, T, num_synth_els, folder, fname, f_err)

    num_constraints= len(v_grid)
    look_ind = int(num_constraints // 2) 
    print('look ind', look_ind)
    now = time.time()
    mcm = run_mcm(K_true.reshape(1, K_true.shape[0], K_true.shape[1]), r_arr, look_ind=look_ind)
    print('mcm time', time.time()-now)
    mcm = mcm[0, :,:]  
    mcm = 10*np.log10(abs(mcm)/np.max(mcm))

    now = time.time()
    mcm_wnc = run_wnc_mcm(K_true.reshape(1, K_true.shape[0], K_true.shape[1]), r_arr, wnc_db, look_ind=look_ind)
    print('wnc time', time.time()-now)
    mcm_wnc = mcm_wnc[0, :,:]  
    print('-----------------', np.max(mcm_wnc), np.min(mcm_wnc))
    mcm_wnc = 10*np.log10(abs(mcm_wnc)/np.max(mcm_wnc))

    for x in [-15, -10]:
        db_lev = x
        fig, axes = plt.subplots(3,1, sharex=True)
        axes[1].set_ylabel('Depth (m)')
        axes[-1].set_xlabel('Range (m)')
        plt.suptitle('Bartlett, MCM, and MCM with WNC for synthetic aperture\nSNR = ' + str(snr) + ' dB, WNC level is ' + str(wnc_db))
        levels = np.linspace(db_lev, 0, 20)


        cmesh0 = axes[0].contourf(r, z, bart, levels=levels, extend='both')
        cmesh1 = axes[1].contourf(r, z, mcm, levels=levels, extend='both')
        cmesh2 = axes[2].contourf(r, z, mcm_wnc, levels=levels, extend='both')
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()
        axes[2].invert_yaxis()
        fig.colorbar(cmesh0,ax=axes[0])
        fig.colorbar(cmesh1,ax=axes[1])
        fig.colorbar(cmesh2,ax=axes[2])
        """ Put source positions on the ambiguity surfaces """
        for i in range(3):
            axes[i].scatter(rs, zs, color='r', marker='+', linewidth=0.5)
        plt.savefig('pics/' + str(snr) + '_snr_' + str(x) + '_dbscale.png', dpi=500, bbox_inches='tight')
    return


def single_sensor_plot_comp(env, rmin, rmax, grid_dr, K_true, snr_db, folder, fname, rs, zs, f_err=0.0):
    """ 
    Just us the VLA to produce Bartlett, MVDR, and WNC
    """
    wn_gain = -3
    r, z, tmp_replicas = form_replicas(env, rmin, rmax, grid_dr, 5, 5, 1, folder, fname)
    bart = get_amb_surf(r, z, K_true, tmp_replicas)

    mvdr = get_mvdr_amb_surf(r, z, K_true, tmp_replicas)
    
    K_true = K_true.reshape(1, K_true.shape[0], K_true.shape[1])    
    wnc = run_wnc(K_true, tmp_replicas, wn_gain)
    wnc = wnc[0, :,:]
    wnc = 10*np.log10(abs(wnc)/np.max(abs(wnc)))

    for x in [-15, -10]:
        db_lev = x
        fig, axes = plt.subplots(3,1, sharex=True)
        axes[1].set_ylabel('Depth (m)')
        axes[-1].set_xlabel('Range (m)')
        plt.suptitle('Bartlett,MVDR and WNC for VLA\nSNR = ' + str(snr_db) + ' dB, WNC level is ' + str(wn_gain))
        levels = np.linspace(db_lev, 0, 20)


        cmesh0 = axes[0].contourf(r, z, bart, levels=levels, extend='both')
        cmesh1 = axes[1].contourf(r, z, mvdr, levels=levels, extend='both')
        cmesh2 = axes[2].contourf(r, z, wnc, levels=levels, extend='both')
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()
        axes[2].invert_yaxis()
        fig.colorbar(cmesh0,ax=axes[0])
        fig.colorbar(cmesh1,ax=axes[1])
        fig.colorbar(cmesh2,ax=axes[2])
        """ Put source positions on the ambiguity surfaces """
        for i in range(3):
            axes[i].scatter(rs, zs, color='r', marker='+', linewidth=0.5)
        plt.savefig('pics/' + str(snr_db) + '_snr_' + str(x) + '_dbscale_VLA.png', dpi=500, bbox_inches='tight')

def get_stacked_cov(synth_p_true, num_synth_els, snr_db):
    num_rcvrs = synth_p_true.size // num_synth_els
    K = np.zeros((num_rcvrs, num_rcvrs), dtype=np.complex128)
    for i in range(num_synth_els):
        p = synth_p_true[i*num_rcvrs:(i+1)*num_rcvrs,0]
        K += np.outer(p, p.conj())
    mean_pow = np.mean(np.square(abs(synth_p_true)))
    noise_var = mean_pow / (np.power(10, snr_db/10))
    noise_var *= num_synth_els
    noise_cov = noise_var*np.identity(num_rcvrs)
    K += noise_cov
    return K

def add_mismatch(env):
    """
    Add some environmental mismatch to the environment 
    """
    env.change_depth(200)
    k = env.modes.k
    print(k)
    return env
    

if __name__ == '__main__':
    """ Set simulation parameters """
    now = time.time()

    freq = 50
    num_rcvrs = 25
    zr = np.linspace(50, 200, num_rcvrs) # array depths
    zs = 50 # source depth

    snr_db = 0

    r0 = 5*1e3 # initial source range
    ship_v = 5 # m/s
    T = 5# time between snapshots in synth array
    ship_dr = ship_v*T
    num_synth_els = 3


    dz = 10
    zmax = 216.5
    grid_dr = 500
    rmax = 1e4
    env_builder = factory.create('swellex')
    env = env_builder()
    folder = 'at_files/'
    fname = 'simple'
    env.add_source_params(freq, zs, zr)
    fig = env.gen_env_fig(r0)
    ax_list = fig.axes
    ax2= ax_list[1]
    for i in range(num_synth_els):
        ax2.scatter(r0 + i*ship_dr, zs, color='b', marker='+')
    env.add_field_params(dz, zmax, grid_dr, rmax)
    plt.show()

    tmp_R, tmp_data, tmp_p_true, tmp_pos = gen_synth_data(env, 1, r0, ship_dr, folder, fname)
    tmp_K_true = add_noise_cov(tmp_p_true, snr_db)
    rmin = grid_dr
    #single_sensor_plot_comp(env, rmin, rmax, grid_dr, tmp_K_true, snr_db)



    env.add_source_params(freq, zs, zr)
    true_r, synth_data, synth_p_true, pos = gen_synth_data(env, num_synth_els, r0, ship_dr, folder, fname)

    stacked_K = get_stacked_cov(synth_p_true, num_synth_els, snr_db)
    plt.figure()
    plt.imshow(abs(stacked_K))
    env.add_source_params(freq, zr, zr)


    env = add_mismatch(env)
    single_sensor_plot_comp(env, rmin, rmax, grid_dr, stacked_K, snr_db, folder, fname, r0, zs)
    
    noiseless_p = synth_p_true[:] 
    plt.figure()
    plt.imshow(abs(np.outer(noiseless_p, noiseless_p.conj())))
    #synth_p_true = add_noise(synth_p_true, snr_db)
    #K_true = np.outer(synth_p_true, synth_p_true.conj())
    K_true = add_noise_cov(synth_p_true, snr_db)
    plt.figure()
    plt.imshow(abs(K_true))
    #eps = 1e-14
    #K_true += eps*np.identity(K_true.shape[0])

    modes = env.modes
    kbar = np.mean(modes.k).real 
    num_constraints = 5 
    ship_dr_bias = 0.0
    dr_grid = get_v_dr_grid(ship_dr+ship_dr_bias, num_synth_els, num_constraints, kbar)
    v_grid = dr_grid/T
    print('v_grid', v_grid)


    tmp = dr_grid[-1] - dr_grid[0]
    rmin = tmp + grid_dr
    print('rmin' ,rmin)
    env.add_source_params(freq, zr, zr)

    three_plot_comp(env, rmin, rmax, grid_dr, ship_v, v_grid, T, num_synth_els, K_true, snr_db, r0, zs,f_err=0)

    print('total run time', time.time() - now)
    plt.show()

    #check_constraint_points(v_grid, T, kbar, num_synth_els, ship_dr)


    #examine_f_sens(env, rmin, rmax, grid_dr, ship_v, T, num_synth_els, K_true, zs, r0)

    #r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, ship_v, T, num_synth_els, folder, fname)
    #output = get_amb_surf(r, z, K_true, synth_replicas)
    #plot_amb_surf(-5, r, z, output, 'bartlett ' + str(ship_v), r0, zs)
    #plt.show()

    #r, z, synth_replicas = form_replicas(env, rmin, rmax, grid_dr, ship_v+1, T, num_synth_els, folder, fname)
    #output = get_amb_surf(r, z, K_true, synth_replicas)
    #plot_amb_surf(-5, r, z, output, 'bartlett, v = ' + str(ship_v+1), r0, zs)
    #plt.show()

    #mcm_mvdr_comp(env, rmin, rmax, grid_dr, r0, v_grid, T, num_synth_els, K_true, folder, fname, f_err=0.0)


