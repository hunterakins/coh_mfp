import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.sim import get_sim_folder, make_sim_name, source_vel
from coh_mfp.get_cov import load_cov
from coh_mfp.get_dvecs import PROJ_ROOT
from pyat.pyat.readwrite import read_shd

'''
Description:
Perform Bartlett processing on sequence of covariance estimates
Save a pic for each one and form a movie of the ambiguity surfaces

Date: 
8/28/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''



def load_replicas(freq):
    fname = get_sim_folder() + make_sim_name(freq) + '.shd'
    [x,x,x,x, pos, pfield] = read_shd(fname)
    pfield = np.squeeze(pfield)
    print(pfield.shape)
    print(pos.r.range.shape)
    print(pos.r.depth.shape)
    return pfield, pos


pfield, pos = load_replicas(49)
tvals, K_samp = load_cov(49)
num_rcvrs = pfield.shape[0]
num_depths = pfield.shape[1]
num_ranges = pfield.shape[2]
num_positions = num_depths*num_ranges
for i in range(tvals.size):
    bartlett = np.zeros((num_depths, num_ranges))
    for j in range(num_depths):
        for k in range(num_ranges):
            r = pfield[:,j,k]
            r /= np.linalg.norm(r)
            power = (r.conj()).T@K_samp[:,:,i]@r
            bartlett[j,k] = power.real
    fig = plt.figure()
    b_db = np.log10(abs(bartlett)/np.max(abs(bartlett)))
    max_loc = np.argmax(abs(bartlett))
    max_depth = max_loc // num_ranges
    max_range = max_loc % num_ranges
    
    levels = np.linspace(-2, 0, 10)
    CS = plt.contourf(pos.r.range, pos.r.depth, b_db, levels=levels)
    plt.plot([1000 + source_vel*tvals[i]], [54], 'b+')
    plt.plot(pos.r.range[max_range], pos.r.depth[max_depth], 'r+')
    plt.colorbar()
    plt.savefig('pics/' + str(i).zfill(3) + '.png')
    plt.close(fig)
