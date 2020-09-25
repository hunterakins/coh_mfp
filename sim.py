import numpy as np
from matplotlib import pyplot as plt
from env.env.envs import factory
from pyat.pyat.readwrite import read_modes


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

def get_sim_folder():
    return 'at_files/'

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

        """ Run the model and get the modes and wavenumbers
        to do a time domain simulation that neglects doppler"""
        """ Run once to get the modes shapes on the array and the kr """
        folder = get_sim_folder()
        fname = make_sim_name(freq)
        env.add_source_params(freq, conf.zs, conf.zr)
        env.add_field_params(conf.dz, conf.zmax, conf.ship_dr, conf.rmax)
        p, pos = env.run_model('kraken', folder, fname, zr_flag=True, zr_range_flag=True)
        print(p.shape)
        modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
        print(modes.phi.shape)
        As = modes.phi[0,:] # extract source exctiation
        phi = modes.phi[1:,:] # throw out source depth val
        krs = modes.k
        num_modes = krs.size

        """ Run a second time to generate the replicas """
        """ Put a source at each of the receiver positoins
        reciprocity gives you the field of replicas for each source  """
        """ Settin zs as zr is just a placeholder, it doesn't matter """
        env.add_source_params(freq, conf.zr, [conf.zs])
        p, pos = env.run_model('kraken', folder, fname, zr_flag=False,zr_range_flag=False)
        print('replica dr', pos.r.range[1] - pos.r.range[0])
        print(p.shape, 'expected shape 21 by like 40 by like ', conf.rmax/conf.ship_dr )



        """ Manually compute field to get high range resolution required
        for the time domain sim """
        """ Add the source signal for this frequency to the field """
        freq_contrib = np.zeros((conf.zr.size, r.size), dtype=np.complex128)
        """Compute modal sum """
        for i in range(num_modes):
            mode_stren =  As[i]*(phi[:,i])
            mode_range_dep = np.exp(-complex(0,1)*krs[i]*r) / np.sqrt(krs[i])
            mode_term = mode_stren.reshape(mode_stren.size, 1) * mode_range_dep
            freq_contrib += mode_term
        """ Add prefactors to modal sum """
        freq_contrib /= np.sqrt(r)
        freq_contrib *= np.exp(complex(0,1)*2*np.pi*freq*t)
        freq_contrib *= np.exp(complex(0,1)*np.pi/4)/np.sqrt(8*np.pi)
        freq_contrib = freq_contrib.real

        """ Estimate sapmle variance over time """
        power = np.square(abs(freq_contrib))
        power = np.mean(power)
        intensity_list.append(power)
        field += freq_contrib


    """ Now add some white gaussian noise """
    """ Recall that the variance of the DFT will be N times the 
    variance of the noise sequence, where N is the DFT length """
    """ We can assume all the signal variance is in a single bin """
    mean_signal_var = np.mean(intensity_list)
    post_fft_signal_pow = 1/3*conf.fft_len*conf.fft_len*mean_signal_var
    intensity_ratio = np.power(10, conf.SNR/10)
    noise_var = post_fft_signal_pow / intensity_ratio
    sample_var = noise_var / conf.fft_len
    noise = np.sqrt(sample_var)*np.random.randn(field.size).reshape(field.shape)

    for sim_iter in range(conf.num_realizations):
        noise = np.sqrt(sample_var)*np.random.randn(field.size).reshape(field.shape)
        sim_data = field+noise
        sim_name = make_raw_ts_name(sim_iter, conf.proj_root)
        np.save(sim_name, sim_data)
    return

def make_raw_ts_name(sim_iter, proj_root):
    return proj_root + 'sim_data_' + str(sim_iter) + '.npy'

if __name__ == '__main__':
    print('hi')
