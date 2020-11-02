import numpy as np
import time
from matplotlib import pyplot as plt
import sys
now = time.time()
from coh_mfp.bartlett import MCOutput, load_mc, SimOutput
print('import time', time.time() - now)
from coh_mfp.config import ExpConf, load_config

'''
Description:
Analyze Monte Carlo outputs, look at processor metrics

Date: 
09/24/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''



if __name__ == '__main__':
    exp_id = sys.argv[1]
    conf = load_config(exp_id)
    proj_root=  conf.proj_root
    freq = conf.freqs[0]
    
    mc_out = load_mc(freq, proj_root)
    #range_super = [x for x in mc_out.sim_outputs if x.proc_key == 'wnc_range']
    #mc_out.make_track_plot('bart')
    mc_out.make_realiz_scatter('bart')
    #mc_out.make_track_plot('wnc')
    mc_out.make_realiz_scatter('bart_range')
    #mc_out.make_track_plot('wnc_range')


