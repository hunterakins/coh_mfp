import numpy as np
import sys
from copy import deepcopy
from matplotlib import pyplot as plt
from env.env.envs import factory
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc

'''
Description:
Just do a really simple simulation with no source phase
A planar array versus a vla

Add the noise as a diagonal matrix to the 'covariance' matrix

Date: 
10/19/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


freq = 150
num_rcvrs = 15
zr = np.linspace(50, 200, num_rcvrs)
zs = 50


dz = 5
zmax = 216.5
dr = 50
rmax = 1e4

env_builder = factory.create('swellex')
env = env_builder()
folder = 'at_files/'
fname = 'simple'
env.add_source_params(freq, zs, zr)
env.add_field_params(dz, zmax, dr, rmax)
p, pos = env.run_model('kraken', folder, fname, zr_flag=True, zr_range_flag=False)
p1 = p
env.add_source_params(freq, zr, zr)
replicas, pos = env.run_model('kraken', folder, fname, zr_flag=False, zr_range_flag=False)

#env.add_source_params(freq, zs, zr)
#custom_r = np.arange(dr, rmax+dr, dr)
#x, pos = env.run_model('kraken_custom_r', folder, fname, zr_flag=True, zr_range_flag=False, custom_r=custom_r)

def add_noise(p_true, snr_db):
    mean_pow = np.mean(np.square(abs(p_true)))
    noise_var = mean_pow/(np.power(10, snr_db/10))
    noise_vec = np.sqrt(noise_var/2)* np.random.randn(p_true.size) + complex(0,1)*np.sqrt(noise_var/2)*np.random.randn(p_true.size)
    noise_vec = noise_vec.reshape(p_true.shape)
    mat = np.outer(noise_vec, noise_vec.conj())
    print(np.mean(np.square(abs(noise_vec))), mean_pow)
    p_true = p_true + noise_vec
    return p_true

def add_noise_cov(p_true, snr_db):
    mean_pow = np.mean(np.square(abs(p_true)))
    noise_var = mean_pow/(np.power(10, snr_db/10))
    print(mean_pow, noise_var)
    #noise_vec = np.sqrt(noise_var/2)* np.random.randn(p_true.size) + complex(0,1)*np.sqrt(noise_var/2)*np.random.randn(p_true.size)
    #noise_vec = noise_vec.reshape(p_true.shape)
    #print(np.mean(np.square(abs(noise_vec))), mean_pow)
    K_true = np.outer(p_true, p_true.conj())
    noise_K = noise_var*np.identity(p_true.size)  #+ complex(0,1)*noise_var/2 * np.identity(p_true.size)
    return K_true+noise_K

def get_amb_surf(pos, K_true, replicas):
    amb_surf = np.zeros((pos.r.depth.size, pos.r.range.size))
    for i in range(pos.r.depth.size):
        for j in range(pos.r.range.size):
            replica = replicas[:,i,j]
            replica /= np.linalg.norm(replica)
            #out = replica.conj().dot(K_true).dot(replica)
            replica= replica.reshape(replica.size, 1)
            out = replica.T.conj()@K_true@replica
            amb_surf[i,j] = abs(out)
    amb_surf /= np.max(amb_surf)
    amb_surf = 10*np.log10(amb_surf)
    return amb_surf

def get_mvdr_amb_surf(pos, K_true, replicas):
    amb_surf = np.zeros((pos.r.depth.size, pos.r.range.size))
    K_true_inv = np.linalg.inv(K_true)
    for i in range(pos.r.depth.size):
        for j in range(pos.r.range.size):
            replica = replicas[:,i,j]
            replica /= np.linalg.norm(replica)
            denom = replica.T.conj()@K_true_inv@replica
            amb_surf[i,j] = 1/abs(denom)
    amb_surf /= np.max(abs(amb_surf))
    amb_surf = 10*np.log10(abs(amb_surf))
    return amb_surf

def plot_amb_surf(db_lev, pos, amb_surf):
    levels = np.linspace(db_lev, 0, 20)
    plt.figure()
    plt.contourf(pos.r.range, pos.r.depth, amb_surf, levels=levels, extend='both')
    plt.colorbar()
    plt.scatter(r_true, zs, marker='+', color='r')
    plt.gca().invert_yaxis()


p_true = p[:,150]
r_true = pos.r.range[150]
print('r true', r_true)
snr_db = -15

K_tmp = add_noise_cov(p_true, snr_db)
#K_true = np.zeros((p_true.size, p_true.size), dtype=np.complex128)
K_true = K_tmp
#p_true = add_noise(p_true, snr_db)
#K_true = np.outer(p_true.conj(), p_true)

#output = run_wnc(K_true, replicas, 0)
#output = 10*np.log10(output/np.max(abs(output)))
#p_true = add_noise(p_true, snr_db)
output = get_mvdr_amb_surf(pos, K_true, replicas)
plot_amb_surf(-1, pos, output)
plt.suptitle('mvdr')

#output = run_wnc(K_true, replicas, -3)
#output = 10*np.log10(output/np.max(abs(output)))
#plot_amb_surf(-10, pos, output[:,:,0])


amb_surf = get_amb_surf(pos, K_true, replicas)

stride = 2
synthetic_array = p[:,150:170:stride]
synth_p_true = synthetic_array.reshape(synthetic_array.size,1, order='F')
synth_K_true =add_noise_cov(synth_p_true, snr_db)

synth_r = pos.r.range[150:170:stride]
print('synth spacing', synth_r[1]-synth_r[0], ' meters')
num_synth_els = synth_r.size
print('number synth els' ,num_synth_els)


num_ranges=  pos.r.range.size
num_depths = pos.r.depth.size
synth_rep = np.zeros((num_rcvrs*num_synth_els, num_depths, num_ranges), dtype=np.complex128)
hanger= 0
for i in range(num_synth_els):
    vals = replicas[:,:,stride*i:]
    synth_rep[i*num_rcvrs:(i+1)*num_rcvrs, :, :vals.shape[-1]] = vals
    num_leftovers = num_ranges - vals.shape[-1]
    hanger = np.max([num_leftovers, hanger])
    print('num left', num_leftovers)

#synth_rep = synth_rep[:,:,:-hanger]
#synth_pos = deepcopy(pos)
#synth_pos.r.range = synth_pos.r.range[:-hanger]
synth_pos = pos

output = get_mvdr_amb_surf(synth_pos, synth_K_true, synth_rep)
plot_amb_surf(-1, synth_pos, output)
plt.suptitle('Synth mvdr')

synth_amb_surf = get_amb_surf(synth_pos, synth_K_true, synth_rep)
plot_amb_surf(-10, synth_pos, synth_amb_surf)
plt.suptitle('Synthetic aperture, SNR: ' +  str(snr_db))
plot_amb_surf(-10, pos, amb_surf)
plt.suptitle('Normal VLA, SNR ' + str(snr_db))
#plot_amb_surf(-5, pos, synth_amb_surf)
#plt.suptitle('Synthetic aperture, SNR: ' +  str(snr_db))
#plot_amb_surf(-5, pos, amb_surf)
#plt.suptitle('Normal VLA, SNR ' + str(snr_db))
#plot_amb_surf(-15, synth_pos, synth_amb_surf)
#plt.suptitle('Synthetic aperture, SNR: ' +  str(snr_db))
#plot_amb_surf(-15, pos, amb_surf)
#plt.suptitle('Normal VLA, SNR ' + str(snr_db))
plot_amb_surf(-1, synth_pos, synth_amb_surf)
plt.suptitle('Synthetic aperture, SNR: ' +  str(snr_db))
plot_amb_surf(-1, pos, amb_surf)
plt.suptitle('Normal VLA, SNR ' + str(snr_db))
plt.show()

