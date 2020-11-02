import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from env.env.envs import factory
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc

'''
Description:
Simulate the field, then simulate replicas with velocity error
to look at sensitivity of ambiguity surface to velocity knowledge

Date: 
10/19/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


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

""" 
Initialize the environment
Generate a synthetic data set
"""
env_builder = factory.create('swellex')
env = env_builder()
folder = 'at_files/'
fname = 'simple'
env.add_source_params(freq, zs, zr)
env.add_field_params(dz, zmax, dr, rmax)
true_r = np.linspace(r0, r0 + (num_synth_els -1)*ship_dr, num_synth_els)
print('True ranges', true_r)

synth_data, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=true_r)

modes = env.modes
kr = modes.k
rough_k = np.mean(kr).real
rough_phase = rough_k*(true_r-true_r[0]) + np.angle(synth_data[0,0])
rough_phase = np.angle(np.exp(complex(0,1)*rough_phase))

""" Reshape true data into supervector """
synth_p_true = synth_data.reshape(synth_data.size, 1, order='F')
synth_p_true /= np.linalg.norm(synth_p_true)
print(synth_data.shape)

""" Generate a set of replicas for the true initial position 
but with a velocity error """
v_grid = np.linspace(1, 10, 200)
bartlett_grid = np.zeros((v_grid.size))
array_errs = np.zeros((v_grid.size))

replica_drs = v_grid*T
range_step_errs = ship_dr - replica_drs
modeled_shape = np.sinc(rough_k*range_step_errs*num_synth_els/2/np.pi)
modeled_shape = np.square(abs(modeled_shape))

for i, v in enumerate(v_grid):
    replica_ship_dr = v*T
    custom_r = np.linspace(r0, r0 + (num_synth_els -1)*replica_ship_dr, num_synth_els)
    array_calib_err = np.linalg.norm(custom_r - true_r)
    array_errs[i] = array_calib_err
    synth_replica, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=custom_r)
    synth_replica = synth_replica.reshape(synth_replica.size, 1, order='F')
    synth_replica/=np.linalg.norm(synth_replica)
    tmp_bartlett = np.square(abs(synth_p_true.conj().T@synth_replica))
    bartlett_grid[i] = tmp_bartlett

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
plt.legend(['True Bartlett shape', 'sinc approximation to Bartlett response using mean wavenumber'])


plt.figure()
peak_inds, peak_heights = find_peaks(bartlett_grid, height = 0.9)
for peak_ind in peak_inds:
    v = v_grid[peak_ind]
    replica_ship_dr = v*T
    custom_r = np.linspace(r0, r0 + (num_synth_els -1)*replica_ship_dr, num_synth_els)
    array_calib_err = np.linalg.norm(custom_r - true_r)
    array_errs[i] = array_calib_err
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


plt.show()


