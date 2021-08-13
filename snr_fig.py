import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.snr import get_snr
import swellex.audio.make_snapshots as ms
from coh_mfp.data_test import get_env
from swellex.audio.config import get_proj_tones
from matplotlib import rc

rc('text', usetex=True)
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


"""
Description:
Plot SNR for 388 Hz as function of FFT length

Date:
3/18/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

base_Nfft=  2048
base_num_snapshots = 36
vv = ms.get_vv()

fig_name = 'snr.png'

proj_str = 's5_deep'

tones = get_proj_tones(proj_str)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(6, 3)
plt.grid(linestyle='--', color='k')


freq_count = 0
freqs = [tones[0], tones[4], tones[8], tones[12]]
linestyle=['solid', 'dashed', 'dashdot', 'dotted']
markers=['.','x', '*', '+']
freqs = freqs[::-1]
num_freqs = len(freqs)
ax_list = []
for freq_count, source_freq in enumerate(freqs):
    cmap = plt.get_cmap('hsv')
    col = cmap(1*freq_count /( num_freqs))

    snr_list = []

    fact_list = [1, 2, 4, 8, 16, 32, 64, 128]

    #fig, axes = plt.subplots(len(fact_list), 1)

    print('source_freq' ,source_freq)
    for i, fact in enumerate(fact_list):
        Nfft=  base_Nfft * fact
        num_snapshots = int(base_num_snapshots / fact)
        if num_snapshots < 1:
            num_snapshots = 1
        cov_t, snr_db, best_v = get_snr(source_freq, vv, Nfft, fact, num_snapshots, proj_str)
        snr_avg = np.mean(snr_db)
        snr_list.append(snr_avg)
        #axes[i].plot(best_v)


    snr_arr =np.array(snr_list)

    domain = np.array([np.log2(x) for x in fact_list])
    base_T = base_Nfft / 1500
    fft_T = [np.power(2,x)*base_T for x in range(len(fact_list))]

    tmp = ax.plot(fft_T, snr_list, marker=markers[freq_count], color=col,linestyle=linestyle[freq_count])
    ax_list.append(tmp[0])
    ax.set_xscale('log')
    ax.set_xlabel('FFT Length (s)', fontsize=15)

    env = get_env(proj_str, source_freq)
    env.run_model('kraken_custom_r', 'at_files/', 'tmp', zr_flag=True, zr_range_flag=False, custom_r = np.array([1000]))
    k = env.modes.k.real
    kmax = np.max(k)
    kmin = np.min(k)
    delta_k = kmax - kmin

    H = np.ones((snr_arr.size, 2))
    H[:,1] = domain

    alpha = np.linalg.inv(H.T@H) @ H.T@snr_arr
    print(alpha)

    if freq_count == 0:
        vbar = abs(np.mean(best_v))
    coh_t = 4*np.pi/vbar/delta_k

    plt.plot([coh_t]*10, np.linspace(0, 35, 10), color=col, alpha=0.8, linestyle=linestyle[freq_count], lw=2)
    #plt.plot(fft_T, alpha[0] + alpha[1]*domain, color='k', alpha=0.6, lw=4)



    ax.set_ylabel('SNR (dB)', fontsize=15)


ax.set_ylim([0, 35])
plt.legend(ax_list, [str(x) + ' Hz' for x in freqs])
fig.tight_layout()
plt.savefig('/home/hunter/research/coherent_matched_field/paper/pics/' + fig_name, dpi=500)
plt.show()


