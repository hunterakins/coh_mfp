import numpy as np
from matplotlib import pyplot as plt
from pyat.pyat.readwrite import read_modes
from env.env.envs import factory

'''
Description:
Some scripts to analyze how sensitivity the Bartlett will be 
to errors in the velocity assumption
An error in the velocity corresponds to error in the synthetic
array calibration, so in some ways is very similar to array tilt.
Thus, it's reasonable to think that we can apply the multiple
tilt constraints kind of method to the data to get better results. 
We also can use my precise frequency estimates to get precise ranging results on the data
(eliminate that error)


Date: 10/23/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def get_p(modes, r):
    krs = modes.k
    phi = modes.phi
    mode_amp = phi[0,:]*phi[1,:]
    mode_amp = mode_amp.reshape(1, mode_amp.size)
    range_dep = np.exp(complex(0,1)*krs*r) / np.sqrt(krs*r)
    range_dep = range_dep.reshape(range_dep.size, 1)
    p = mode_amp@range_dep
    return p

def get_array_p(modes, r):
    krs = modes.k
    phi = modes.phi
    source_amp = phi[0,:]
    mode_amp = source_amp * phi[1:,:]
    num_rcvrs = phi.shape[0]-1
    num_modes = len(krs)
    mode_amp = mode_amp.reshape(num_rcvrs, num_modes)
    range_dep = np.exp(complex(0,1)*krs*r) / np.sqrt(krs*r)
    range_dep = range_dep.reshape(range_dep.size, 1)
    p = mode_amp@range_dep
    return p

def single_sensor_analysis(freq):
    num_rcvrs = 1
    zr = 70
    zs = 50


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
    p, pos = env.run_model('kraken', folder, fname, zr_flag=True, zr_range_flag=False)
    modes = read_modes(**{'freq': freq, 'fname':folder + fname + '.mod'})
    r1 = 6*1e3
    delta_r = np.linspace(0, 10, 30)
    p_r1 = get_p(modes, r1)
    p_r1 /= abs(p_r1)
    P_vals = []
    for dr in delta_r:
        p_tmp = get_p(modes, r1+dr)
        p_tmp /= abs(p_tmp)
        P = p_r1.conj()*p_tmp
        plt.scatter(dr, np.angle(p_tmp))
        P_vals.append(P)
    print(np.mean(modes.k.real))
    plt.plot(delta_r, np.mean(modes.k.real)*delta_r)
    P_vals = np.array(P_vals)
    plt.show()
        
def array_analysis(freq):
    num_rcvrs = 20
    zr = np.linspace(100, 200, num_rcvrs)
    zs = 50


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
    p, pos = env.run_model('kraken', folder, fname, zr_flag=True, zr_range_flag=False)
    modes = read_modes(**{'freq': freq, 'fname':folder + fname + '.mod'})
    r1 = 6*1e3
    delta_r = np.linspace(0, 150, 30)
    p_r1 = get_array_p(modes, r1)
    p_r1 /= abs(p_r1)
    P_vals = []
    for dr in delta_r:
        p_tmp = get_array_p(modes, r1+dr)
        p_tmp /= abs(p_tmp)
        print(p_tmp.shape, p_r1.shape)
        P = abs(p_r1.conj().T@p_tmp)
        #plt.scatter(dr, abs(P))
        P_vals.append(P)
    print(np.mean(modes.k.real))
    #plt.plot(delta_r, np.mean(modes.k.real)*delta_r)
    P_vals = np.array(P_vals)
    P_vals /= np.max(P_vals)
    P_vals = 10*np.log10(P_vals)
    plt.scatter(delta_r, P_vals)
    plt.ylim([np.min(P_vals), np.max(P_vals)])
    plt.xlabel('Range error (m)')
    plt.ylabel('Magnitude of Bartlett processor (dB)')
    plt.suptitle('Sensitivity of array to range errors. \nArray has ' + str(num_rcvrs) + ' elements spaced from ' + str(np.min(zr)) + ' to ' + str(np.max(zr)) + ' m depth\nSource is at ' + str(zs) + ' m depth.')
    plt.show()
    
#single_sensor_analysis(50)
array_analysis(50)
    
 
