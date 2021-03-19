import numpy as np
from matplotlib import pyplot as plt
from swellex.audio.config import get_proj_tones
import swellex.audio.make_snapshots as ms
from coh_mfp.data_test import get_r_super_cov_seq, deal_with_t_x_r, load_x

'''
Description:
Estimate the SNR from the stft data

Date: 
8/27/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

def get_snr(source_freq, vv, Nfft, fact, num_snapshots, proj_str):
    subfolder = str(Nfft)
    v_counter = 0
    for v in vv:
        fc = ms.get_fc(source_freq, v)
        t, x = load_x(fc, proj_str, subfolder)
        noise_t1, noise_x1 = load_x(ms.get_fc(source_freq-1, v), proj_str, subfolder)
        noise_t2, noise_x2 = load_x(ms.get_fc(source_freq+2, v), proj_str, subfolder)

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

        n_avg_pow = (nx1_avg_pow + nx2_avg_pow) / 2
        sig_avg_pow = x_avg_pow - n_avg_pow
        snr = sig_avg_pow / n_avg_pow

        if v==vv[0]:
            full_snr = np.zeros((vv.size, snr.size))
        full_snr[v_counter, :] = snr
        v_counter += 1
    best_snr = np.max(full_snr, axis=0)
    best_inds = np.argmax(full_snr, axis=0)
    best_v = vv[best_inds]
    snr_db = 10*np.log10(best_snr)
    return cov_t, snr_db, best_v

if __name__ == '__main__':
    proj_str = 's5_deep'
    #subfolder = '2048'
    
    #Nfft = 8096 
    for fact in [4, 8, 16, 32, 64, 128]:
        Nfft = 2048*fact
        num_snapshots = int(36 / fact)
        if num_snapshots < 1:
            num_snapshots = 1
        print('num snaps', num_snapshots)
        #subfolder = '16192'
        #num_snapshots = 7
        vv = ms.get_vv()
        
        tones = get_proj_tones(proj_str)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        num_colors = len(tones)
        cm = plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])
        for source_freq in tones:
            cov_t, snr_db, best_v = get_snr(source_freq, vv, Nfft, fact, num_snapshots, proj_str)
            plt.plot(cov_t,snr_db)
        plt.legend([str(x) for x in tones])
        fig.suptitle('SNR with Nfft = ' + str(Nfft))
        ax.set_ylim([10, 38])
    plt.show()
                
            
