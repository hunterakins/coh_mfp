import numpy as np
import sys
from matplotlib import pyplot as plt
#from coh_mfp.config import freqs, fig_folder, num_realizations
from coh_mfp.config import ExpConf, load_config
from coh_mfp.bartlett import get_fig_leaf
import os

'''
Description:
Copy over figures from tscc

Date: 
9/16/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

exp_id = int(sys.argv[1])
conf = load_config(exp_id, local=True)
fig_folder = conf.fig_folder
print(fig_folder)

num_realizations = conf.num_realizations
freqs = conf.freqs
local_folder = 'pics/' + str(exp_id) + '/'
if str(exp_id) not in os.listdir('pics'):
    os.mkdir('pics/'+str(exp_id))



def copy_fig(proc_key, local_folder):
    os.system('scp fakins@tscc-login.sdsc.edu:' + conf.proj_root + proc_key + '.png ' + local_folder)
    #os.system('scp fakins@tscc-login.sdsc.edu:' + conf.fig_folder + 'cov_show.png ' + local_folder)

copy_fig('', local_folder)

proc_keys = ['wnc', 'bart', 'wnc_range', 'bart_range']
proc_keys = ['bart', 'bart_range']
for key in proc_keys:
    copy_fig(key, local_folder)
    os.system('xdg-open ' + local_folder + key+'.png')

for freq in freqs:
    local_root = local_folder
    os.system('rsync -r fakins@tscc-login.sdsc.edu:' +fig_folder + ' ' + local_root)
    #for i in range(num_realizations):
    #    fig_leaf = get_fig_leaf(freq, i, 'bart', local_root)
        #os.chdir('pics/' + fig_leaf)
        #os.system('ffmpeg -loglevel quiet -r 5 -f image2 -s 1920x1080 -i ' + local_root+ fig_leaf + '%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p '+ local_root + fig_leaf + str(freq)+'.mp4')
        #fig_leaf = get_fig_leaf(freq, i, 'bart_range', local_root)
        #print(local_root + fig_leaf)
       #os.system('ffmpeg -loglevel quiet -r 5 -f image2 -s 1920x1080 -i ' + local_root+ fig_leaf + '%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p '+ local_root + fig_leaf + str(freq)+'.mp4')
