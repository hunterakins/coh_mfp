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


""" Pick source params
and receiver config """
freqs = [49, 64, 79, 94, 109]#, 112, 127, 130, 145, 148]
source_vel = 3 # ship range rate in m/s
fft_len = 2048
SNR = 20 # after fft gain is accounted for
fs = 1500 # sampling rate

if __name__ == '__main__':
    """ 
    Generate a fake range rate track at 1500 Hz sampling rate
    """
    r0 = 1000
    r1 = 8000 
    R = r1-r0 # total distance traveled1
    T = R / source_vel # total time in seconds
    print(T) #
    dt = 1/fs
    t = np.arange(0, T+dt, dt)
    print(t.size)
    r = r0 + source_vel*t

    """ Calculate time domain field with zero initial phase """
    zr = np.array([94.125, 99.755, 105.38, 111.00, 116.62, 122.25, 127.88, 139.12, 144.74, 150.38, 155.99, 161.62, 167.26, 172.88, 178.49, 184.12, 189.76, 195.38, 200.99, 206.62, 212.25])
    field = np.zeros((zr.size, r.size))

    
    intensity_list = []

    for freq in freqs:
        env_builder  =factory.create('swellex')
        env = env_builder()
        zs = 54
        num_rcvrs = zr.size
        dz =  5
        zmax = 216.5
        dr = 1
        rmax = 10*1e3

        """ Run the model and get the modes and wavenumbers
        to do a time domain simulation that neglects doppler"""

        folder = 'at_files/'
        fname = 'swell_'+str(freq)
        env.add_source_params(freq, zs, zr)
        env.add_field_params(dz, zmax, dr, rmax)
        p, pos = env.run_model('kraken', folder, fname, zr_range_flag=False)
        modes = read_modes(**{'fname':folder+fname+'.mod', 'freq':freq})
        print(modes.phi.shape)
        As = modes.phi[0,:] # extract source exctiation
        phi = modes.phi[1:,:] # throw out source depth val
        krs = modes.k
        num_modes = krs.size

        """ Add the source signal for this frequency to the field """
        freq_contrib = np.zeros((zr.size, r.size), dtype=np.complex128)
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
    post_fft_signal_pow = 1/3*fft_len*fft_len*mean_signal_var
    intensity_ratio = np.power(10, SNR/10)
    noise_var = post_fft_signal_pow / intensity_ratio
    sample_var = noise_var / fft_len
    noise = np.sqrt(sample_var)*np.random.randn(field.size).reshape(field.shape)
    field += noise
    np.save('/oasis/tscc/scratch/fakins/coh_mfp/sim_data.npy', field)
