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

cnstitution: UC San Diego, Scripps Institution of Oceanography

'''

if __name__ == '__main__':
    exp_id = sys.argv[1]
    conf = load_config(exp_id)
    proj_root=  conf.proj_root
    freq = conf.freqs[0]
    
    mc_out = load_mc(freq, proj_root)
    range_super = [x for x in mc_out.sim_outputs if x.proc_key == 'wnc_range']
    print(len(mc_out.sim_outputs))
    print(range_super)
    print(range_super[0].max_locs)
    mc_out.sim_compare('bart')
    mc_out.sim_compare('wnc')
    mc_out.sim_compare('bart_range')
    mc_out.sim_compare('wnc_range')

