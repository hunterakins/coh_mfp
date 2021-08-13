import numpy as np
import os
from matplotlib import pyplot as plt
from coh_mfp.data_test import get_max_val_arr, load_results, DataRunParams
from coh_mfp.simplest_sim import plot_amb_surf
from swellex.audio.config import get_proj_tones, get_proj_zr, get_proj_zs
'''
Description:

Date: 

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

proj_str = 's5_quiet_tones'
num_snaps = 2
subfolder = '4096'
rcvr_stride = 4
num_freqs= 10
freqs = get_proj_tones(proj_str)[:num_freqs]
num_synth_els = 1
zs = get_proj_zs(proj_str)
max_val_list = []
#vv = np.linspace(-2.6, -1.8, 15)
vv = [-2.5]
#num_synth_els = 1
output_list = []
max_val_list = []
for v in vv:
    v = round(v, 4)
    drp = DataRunParams(proj_str, num_snaps, v, subfolder, rcvr_stride, num_freqs, num_synth_els)
    r_center, cov_t,r,z, outputs, max_vals = load_results(drp)
    max_val_list.append(max_vals)
    output_list.append(outputs)
    

best_outs = np.zeros((outputs.shape))

   
max_val_arr = get_max_val_arr(vv, r_center, max_val_list) 

#plt.contourf(cov_t, vv, max_val_arr)

max_inds = np.argmax(max_val_arr, axis=0)

best_v = [vv[x] for x in max_inds]
plt.figure()
plt.plot(best_v)

fig_prefix = str(num_synth_els) + '_best_' 
bguesses = []
for i in range(len(r_center)):
    best_ind = max_inds[i]
    print('best ind, best v', best_ind, vv[best_ind])
    best_amb_surf = output_list[best_ind][i,:,:]/num_freqs
    best_guess = r[int(np.argmax(best_amb_surf)%r.size)]
    bguesses.append(best_guess)
    best_outs[i, :, :] = best_amb_surf
    #fig = plot_amb_surf(-10, r,z,best_outs[i,:,:], 'snap ' + str(i), r_center[i], zs)
    #plt.savefig('pics/' + proj_str + '/' + subfolder + '/' + fig_prefix + str(i).zfill(3) + '.png')
    #plt.close(fig)


#os.system('ffmpeg -loglevel quiet -r 3 -f image2 -s 1920x1080 -i pics/' + proj_str + '/' + subfolder + '/' + fig_prefix+ '%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p pics/' + proj_str + '/' + subfolder + '/' + fig_prefix  + '.mp4')

plt.figure()
plt.plot(cov_t,  r_center)
plt.plot(cov_t, bguesses)
plt.show()
