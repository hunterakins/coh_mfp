import numpy as np
from matplotlib import pyplot as plt
from env.env.envs import factory
from pyat.pyat.readwrite import read_modes
import os


'''
Description:
Generate some synthetic data for the coherent MFP test stuff

Just use a uniform source traveling from 1 km to 8 km
at v_b

Keep the first term in a mach number [source_vel / c] expansion
of the true doppler modal solution. 

Set the SNR of the simulation here.
This will be the RANGE AVERAGE SNR ~AFTER~ FFT GAIN IS ACCOUNTED FOR
(factor of N^2 for signal and factor of N for white noise). 
Since signal power decreases as 1/r for cylindrical spreading, 
average power is such that 10*log10(P_avg/P_noise) = SNR
I'm not sure exactly which range the SNR condition best fits...
I think it involves a log of the total range interval?
Anyways, SNR is higher at r0 and lower at r1
Also hann window cuts power by about 1/3 (maybe exactly)



Date: 
8/25/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''



def make_sim_name(freq):
    """ File stem of simulated data"""
    return 'swell_' + str(freq)

def get_sim_folder(proj_root):
    """
    Get the path to the folder where the simulation
    outputs live 
    """
    if 'at_files' not in os.listdir(proj_root):
        os.mkdir(proj_root + 'at_files/')
    return proj_root + 'at_files/'

def run_sim(conf):
    """ 
    Generate a fake range rate track at 1500 Hz sampling rate
    """
    R = conf.r1-conf.r0 # total distance traveled1
    T = R / conf.source_vel # total time in seconds
    print(T) #
    dt = 1/conf.fs
    t = np.arange(0, T+dt, dt)
    print(t.size)
    acc_f = 1/conf.acc_T
    r = conf.r0 + conf.source_vel*t + conf.acc_amp/(2*np.pi*acc_f)*np.sin(2*np.pi*acc_f*t)


    """ Calculate time domain field with zero initial phase """
    field = np.zeros((conf.zr.size, r.size))
    intensity_list = []

    for freq in conf.freqs:
        env_builder  =factory.create('swellex')
        env = env_builder()
        num_rcvrs = conf.zr.size

        """ Run kraken for this specific frequency at every single r """
        folder = get_sim_folder(conf.proj_root)
        fname = make_sim_name(freq)
        env.add_source_params(freq, conf.zs, conf.zr)
        env.add_field_params(conf.dz, conf.zmax, conf.ship_dr, conf.rmax)

        freq_contrib, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=r)
        """ Only a single source, so take the first entry  """
        freq_contrib = freq_contrib[0,...]
        """ Add the source signal for this frequency to the field """
        freq_contrib *= np.exp(complex(0,1)*2*np.pi*freq*t)
        freq_contrib = freq_contrib.real # this just sets the original phase of the signal

        """ Estimate sample variance over time """
        power = np.square(abs(freq_contrib))
        power = np.mean(power)
        intensity_list.append(power)
        field += freq_contrib



    sim_data = field
    sim_name = make_raw_ts_name(conf.proj_root)
    np.save(sim_name, sim_data)
    return

def make_raw_ts_name(proj_root):
    return proj_root + 'sim_data' + '.npy'

if __name__ == '__main__':
    print('hi')
