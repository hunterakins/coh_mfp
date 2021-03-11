import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.config import get_proj_tones
from coh_mfp.data_test import get_r_super_cov_seq, deal_with_t_x_r, load_x

'''
Description:
Estimate the SNR from the stft data

Date: 
8/27/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

if __name__ == '__main__':
    proj_str = 's5_deep'
    subfolder = '2048'
    subfolder = '2048_doppler'
    #subfolder = '8096_doppler'
    num_snapshots = 15
    tones = get_proj_tones(proj_str)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    

    num_colors = len(tones)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])

    for source_freq in tones:
        print(source_freq)
        t, x = load_x(source_freq, proj_str, subfolder)
        noise_t1, noise_x1 = load_x(source_freq-1, proj_str, subfolder)
        noise_t2, noise_x2 = load_x(source_freq+2, proj_str, subfolder)

        x_pow =  np.sum(np.square(abs(x)), axis=0) / x.shape[0]
        nx1_pow = np.sum(np.square(abs(noise_x1)), axis=0) / noise_x1.shape[0]
        nx2_pow = np.sum(np.square(abs(noise_x2)), axis=0) / noise_x2.shape[0]


        num_samples = noise_t1.size
        num_covs = (num_samples - num_snapshots) // num_snapshots
        x_avg_pow = np.zeros((num_covs))
        nx1_avg_pow = np.zeros((num_covs))
        nx2_avg_pow = np.zeros((num_covs))
        cov_t = np.zeros((num_covs))
        for i in range(num_covs):
            tmp = np.mean(np.square(abs(x[:,num_snapshots*i:num_snapshots*(i+1)])))
            x_avg_pow[i] = tmp

            tmp = np.mean(np.square(abs(noise_x1[:, num_snapshots*i:num_snapshots*(i+1)])))
            nx1_avg_pow[i] = tmp

            tmp = np.mean(np.square(abs(noise_x2[:, num_snapshots*i:num_snapshots*(i+1)])))
            nx2_avg_pow[i] = tmp
            
            tmp = np.mean(t[num_snapshots*i:num_snapshots*(i+1)])
            cov_t[i] = tmp


        #plt.figure()
        #plt.plot(x_avg_pow)
        #plt.plot(nx1_avg_pow)
        #plt.plot(nx2_avg_pow)
        #plt.show()


        n_avg_pow = (nx1_avg_pow + nx2_avg_pow) / 2
        sig_avg_pow = x_avg_pow - n_avg_pow
        #plt.figure()
        #plt.plot(sig_avg_pow)
    
        #plt.figure()
        snr = sig_avg_pow / n_avg_pow
        snr_db = 10*np.log10(snr)
        plt.plot(cov_t, snr_db)
    plt.legend([str(x) for x in tones])
    plt.show()
            
        
