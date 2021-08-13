import numpy as np
import sys
import time
from matplotlib import pyplot as plt
from coh_mfp.data_test import get_r_super_cov_seq, deal_with_t_x_r, load_x, get_env, populate_env, get_mult_el_reps
from coh_mfp.simplest_sim import get_amb_surf
from swellex.audio.config import get_proj_tones, get_proj_zr, get_proj_zs
from signal_proc.mfp.wnc import run_wnc, lookup_run_wnc
from coh_mfp.proc_out import SwellProcObj, make_spo_pickle_name
from coh_mfp.vel_estimation import load_vel_arr
from scipy.interpolate import interp1d
import swellex.audio.make_snapshots as ms

"""
Description:
Messin around with wnc

Date:
2/19/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


    
def get_noise_K_samps(source_freq, proj_str, subfolder, num_snapshots, num_synth_els, stride, incoh, rm_diag=False):
    for noise_freq in range(source_freq - 2, source_freq + 3):
        t, x = load_x(noise_freq, proj_str, subfolder)
        t, x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq)
        _, _, tmp = get_r_super_cov_seq(t, x, num_snapshots, r_interp, num_synth_els, stride, incoh=incoh)
        if rm_diag == True:
            for i in range(tmp.shape[0]):
                tmp[i,...] -= np.diag(np.diag(tmp[i,...]))
        if noise_freq == (source_freq - 2):
            K_samps = tmp
        elif noise_freq == source_freq:
            pass
        else:
            K_samps += tmp
    return K_samps / 5

def get_single_freq_wnc(proj_str, subfolder, source_freq, num_synth_els, best_v, wn_gain, num_snapshots, cov_index, incoh=False, tilt_angle=-1):
    print('wn_gain', wn_gain)
    for num_synth_els in [num_synth_els]:
        vv=[best_v]
        for v in vv:
            print('-----------------------------------------')
            now = time.time()
            fc = ms.get_fc(source_freq, v)
            t, x = load_x(fc, proj_str, subfolder)
            num_rcvrs = x.shape[0]
            t, x, rt, rr, vv, r_interp = deal_with_t_x_r(t, x, source_freq)
            zs, zr = get_proj_zs(proj_str), get_proj_zr(proj_str)
            rcvr_stride = int(zr.size // num_rcvrs)
            zr = zr[::rcvr_stride]

            env = get_env(proj_str, source_freq)
            env = populate_env(env, source_freq, zs, zr, rr, fname=proj_str)
            
            kbar = np.mean(env.modes.k.real)
            delta_k = np.max(env.modes.k.real) - np.min(env.modes.k.real)
            #print('delta_k', delta_k)
            range_cell = (2*np.pi / delta_k)
            range_cell /= 4 # do quarter range cells
            #print('range cell', range_cell)
            time_in_cell = range_cell / abs(v)
            #print('time in cell', time_in_cell)
            #T = time_in_cell 


            delta_t = t[1]-t[0]      
            print('num snapshots', num_snapshots)
            print('good num snaps per cov mat, total_num data snapshots',  num_snapshots, x.shape[1])
            stride = num_snapshots
            T = stride * delta_t


            print('v', v)
            r, z, reps = get_mult_el_reps(env, num_synth_els, v, T,fname=proj_str,adiabatic=False, tilt_angle=tilt_angle)


            r_center, cov_t, K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, num_synth_els, stride, incoh=incoh)
            r_center, cov_t, K_samps = r_center[cov_index], cov_t[cov_index], K_samps[cov_index,...]


            #output = lookup_run_wnc(K_samps, reps, wbn_gain, noise_K_samps)
            K_samps = K_samps.reshape(1, K_samps.shape[0], K_samps.shape[1])
            output = lookup_run_wnc(K_samps, reps, wn_gain)
            out_db = 10*np.log10(output)
            print('wnc time', time.time() - now)
            print('-----------------------------------------')
    return r, z, r_center, zs, out_db, cov_t

#def get_single_freq_mcm(proj_str, subfolder, source_

def get_single_freq_bart(proj_str, subfolder, source_freq, num_synth_els, v, num_snapshots, cov_index, incoh=False, tilt_angle=-1):
    rm_diag = False
    for num_synth_els in [num_synth_els]:
        print('-----------------------------------------')
        now = time.time()
        fc = ms.get_fc(source_freq, v)
        t, x = load_x(fc, proj_str, subfolder)
        num_rcvrs = x.shape[0]
        t, x, rt, rr, vvv, r_interp = deal_with_t_x_r(t, x, source_freq)
        zs, zr = get_proj_zs(proj_str), get_proj_zr(proj_str)
        rcvr_stride = int(zr.size // num_rcvrs)
        zr = zr[::rcvr_stride]

        env = get_env(proj_str, source_freq)
        env = populate_env(env, source_freq, zs, zr, rr, fname=proj_str)
        
        kbar = np.mean(env.modes.k.real)
        delta_k = np.max(env.modes.k.real) - np.min(env.modes.k.real)
        #print('delta_k', delta_k)
        range_cell = (2*np.pi / (delta_k/2))
        range_cell /= 4 # do quarter range cells
        #print('range cell', range_cell)
        time_in_cell = range_cell / abs(v)

        delta_t = t[1]-t[0]      
        print('source freq', source_freq)
        print(' range cell snaps', time_in_cell*4 / delta_t)
        print(' range cell size ', range_cell*4)
        #print('num snapshots', num_snapshots)
        #print('good num snaps per cov mat, total_num data snapshots',  num_snapshots, x.shape[1])
        #stride = num_snapshots
        stride = num_snapshots
        print('stride, num_snaps', stride, num_snapshots)
        T = stride * delta_t


        print('v', v)
        r, z, reps = get_mult_el_reps(env, num_synth_els, v, T,fname=proj_str,adiabatic=False, tilt_angle=tilt_angle)

        r_center, cov_t, K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, num_synth_els, stride, incoh=incoh)
        r_center, cov_t, K_samps = r_center[cov_index], cov_t[cov_index], K_samps[cov_index,...]
        bart_db, max_val = get_amb_surf(r, z, K_samps, reps, matmul=True)
        bart_db += 10*np.log10(max_val)
        out_db = bart_db
        #out_db = 10*np.log10(output)
        #print('bart time', time.time() - now)
    return r, z, r_center, zs, out_db, cov_t

def get_stacked_wnc(proj_str, subfolder, num_synth_els, best_v, wn_gain, num_snapshots, cov_index, incoh=False,num_freqs=None, tilt_angle=-1):
    if type(num_freqs) == type(None):
        tones = get_proj_tones(proj_str)
    else:
        tones= get_proj_tones(proj_str)[:num_freqs]
    for source_freq in tones:
        r, z, r_center, zs, out_db, cov_t = get_single_freq_wnc(proj_str, subfolder, source_freq, num_synth_els, best_v, wn_gain, num_snapshots, cov_index, incoh, tilt_angle=tilt_angle)
        out_db -= np.max(out_db)
        if source_freq == tones[0]:
            out = out_db
        else:
            out += out_db
    out /= len(tones)
    return r, z, r_center, zs, out

def get_stacked_bart(proj_str, subfolder, num_synth_els, v, num_snapshots, cov_index, incoh=False, num_freqs=None, tilt_angle=-1):
    if type(num_freqs) == type(None):
        tones = get_proj_tones(proj_str)
    else:
        tones= get_proj_tones(proj_str)[:num_freqs]
    for source_freq in tones:
        r, z, r_center, zs, out_db, cov_t = get_single_freq_bart(proj_str, subfolder, source_freq, num_synth_els, v, num_snapshots, cov_index, incoh, tilt_angle=tilt_angle)
        out_db += np.max(out_db)
        if source_freq == tones[0]:
            out = out_db
        else:
            out += out_db
    out /= len(tones)
    return r, z, r_center, zs, out, cov_t

def plot_bart(r, z, out_db, r_center, zs):
    db_max = np.max(out_db)
    #db_max = np.max(out_db[i,:,:])
    db_min = db_max - 15
    print(db_max, db_min)
    levels = np.linspace(db_min, db_max, 20)
    plt.figure()
    #plt.suptitle(str(num_synth_els) + ',  v= ' + str(v) + ', wn gain ' + str(wn_gain))
    #plt.contourf(r, z, out_db[i,:,:],levels=levels, extend='both')
    plt.contourf(r, z, out_db,levels=levels, extend='both')
    plt.colorbar()
    plt.scatter(r_center, zs, marker='+', color='r')
    plt.gca().invert_yaxis()
    return

def plot_wnc(r, z, out_db, r_center, zs):
    db_max = np.max(out_db[0,:,:])
    db_min = db_max - 15
    print(db_max, db_min)
    levels = np.linspace(db_min, db_max, 20)
    plt.figure()
    #plt.suptitle(str(num_synth_els) + ',  v= ' + str(v) + ', wn gain ' + str(wn_gain))
    plt.contourf(r, z, out_db[0,:,:],levels=levels, extend='both')
    #plt.contourf(r, z, out_db,levels=levels, extend='both')
    plt.colorbar()
    plt.scatter(r_center, zs, marker='+', color='r')
    plt.gca().invert_yaxis()
    return

def get_cov_time(proj_str, subfolder, num_snapshots, num_synth_els):
    source_freq = get_proj_tones(proj_str)[0]
    fc = ms.get_fc(source_freq, 0)
    t, x = load_x(fc, proj_str, subfolder)
    t, x, rt, rr, vvv, r_interp = deal_with_t_x_r(t, x, source_freq)
    stride = num_snapshots
    r_center, cov_t, K_samps = get_r_super_cov_seq(t, x, num_snapshots, r_interp, num_synth_els, stride)
    return cov_t

def check_v_arr(v_arr, cov_times):
    if v_arr[0, -1] < cov_times[-1]:
        new_t = cov_times[-1]
        new_v = v_arr[1,-1]
        new_entry = np.array([new_t, new_v]).reshape(2,1)
        v_arr = np.concatenate((v_arr, new_entry), axis=1)

    if v_arr[0, 0] > cov_times[0]:
        new_t = cov_times[0]
        new_v = v_arr[1,0]
        new_entry = np.array([new_t, new_v]).reshape(2,1)
        v_arr = np.concatenate((new_entry, v_arr), axis=1)
    return v_arr

#def get_full_set_estimates(proj_str, subfolder, num_synth_els, num_snapshots, wn_gain,


if __name__ == '__main__':

    #range_cell_test(source_freq)
    #sys.exit(0)
    """
    Example usage
    python wnc_test.py s5_deep 1 -2 _sec1
    """
    proj_str = sys.argv[1]
    fact = int(sys.argv[2])
    wn_gain = float(sys.argv[3])
    num_synth_els = int(sys.argv[4])
    if len(sys.argv) == 6: # get segment tail
        subfolder_suffix = sys.argv[5]
        tilt_angle = -0.5
    else: # segment tail is null because i look at sec3
        subfolder_suffix = ''
        tilt_angle = -1
        

    print('Running wnc_test on ', proj_str, 'with fft fact ', fact, ' and wn gina', wn_gain)

    N_fft = 2048
    num_snapshots = 36
    N_fft = fact*N_fft
    num_snapshots = int(num_snapshots / fact)

    print('N_fft, num snaps', N_fft, num_snapshots)

    
    
    for wn_gain in [wn_gain]:
        num_freqs =13
        num_synth_els_list = [num_synth_els]
        for subfolder in [str(N_fft) + subfolder_suffix]:#, str(N_fft) + '_sec1']:
            print(subfolder)
        
            #subfolder =str(N_fft) + '_sec1'
            vv = ms.get_vv()
            #for proj_str in ['s5_deep', 's5_quiet2', 's5_quiet1', 's5_quiet3']:#, 's5_quiet4']:
        #for proj_str in ['s5_quiet1']:

            proj_tones = get_proj_tones(proj_str)
            source_freq = proj_tones[7]
            print('soruce freq', source_freq)



            for num_synth_els in num_synth_els_list:
                if num_synth_els == 1:
                    cov_times = get_cov_time(proj_str, subfolder, num_snapshots, 5)
                    v_arr = load_vel_arr(proj_str, subfolder, num_snapshots, 5)
                else:
                    cov_times = get_cov_time(proj_str, subfolder, num_snapshots, num_synth_els)
                    v_arr = load_vel_arr(proj_str, subfolder, num_snapshots, num_synth_els)
                v_arr = check_v_arr(v_arr, cov_times)
                v_interp = interp1d(v_arr[0,:], v_arr[1,:])

                tones = proj_tones[:num_freqs]
                #for cov_index in range(cov_times.size):
                for cov_index in range(cov_times.size):

                    cov_time = cov_times[cov_index]
                    print('cov time based on index', cov_time)
                    
                    v_source = v_interp(cov_time)
                    v_source = vv[np.argmin([abs(v_source -x) for x in vv])] 
                    print('v source', v_source)


                    r,z, r_center, zs, out_db, cov_t = get_stacked_bart(proj_str, subfolder, num_synth_els, v_source, num_snapshots, cov_index, incoh=False, num_freqs=num_freqs,tilt_angle=tilt_angle)
                    spo = SwellProcObj(tones, num_snapshots, proj_str, subfolder, r_center, zs, cov_t, v_source, tilt_angle, num_synth_els, r, z, out_db)

                    spo.get_bathy_corr()
                    
                    r,z,r_center, zs, out_db = get_stacked_wnc(proj_str, subfolder, num_synth_els, v_source, wn_gain, num_snapshots, cov_index, num_freqs=num_freqs, tilt_angle=tilt_angle)

                    spo.add_wnc(wn_gain, out_db[0,:,:])
                    name = spo.save()
                    print('name', name)
                    #plot_wnc(spo.corr_grid_r, spo.corr_grid_z, out_db, r_center, zs)
                    
