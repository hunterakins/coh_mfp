import numpy as np
from matplotlib import pyplot as plt
  
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import colors as colors
from matplotlib import rc
from wnc_test import get_cov_time
from proc_out import SwellProcObj, load_spo
from vel_estimation import load_vel_arr
from scipy.interpolate import interp1d
from copy import deepcopy

rc('text', usetex=True)
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

PLOT_STRIDE = 1

"""
Description:
Create multi panel figure comparing our method to conventional methods.

Ultimately I will be focusing on a specific snapshot time to 
do this comparison. So this should be a parameter.

Date:
3/1/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""



def get_tracking_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle,num_freqs, num_tracking_els, v_interp, cov_index, wnc=False, wn_gain=None):
    cov_times=get_cov_time(proj_str, subfolder, num_snapshots, num_tracking_els)
    delta_t = cov_times[1] - cov_times[0]
    print('delta t' , delta_t)
    for i in range(num_tracking_els): 
        print('i', i)
        cov_time = cov_times[cov_index + i]
        print('cov time', cov_time)
        v_source = v_interp(cov_time)
        spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle,1, num_freqs, v_source, cov_time, wn_gain)
        spo.get_bathy_corr()
        print(np.min(spo.corr_grid_r), np.max(spo.corr_grid_r))

        if i == 0:
            ship_dr = v_source*delta_t  # amount of space between synth els
            incoh_spo = deepcopy(spo)
            incoh_bart = incoh_spo.bart_out    
            if wnc==True:
                incoh_wnc = spo.wnc_out
        else:
            grid_dr = spo.corr_grid_r[1] - spo.corr_grid_r[0]
            index_shift = int(round((ship_dr*i)/grid_dr))
            print('index shift', index_shift)
            #index_shift = -index_shift
            if index_shift < 0: # moving closer, so throw away last points
                rel_bart = spo.bart_out[:, :index_shift]
                incoh_bart[:,-index_shift:] += rel_bart
                incoh_bart[:, :-index_shift] = incoh_bart[:, -index_shift:-2*index_shift]
                if wnc == True:
                    rel_wnc = spo.wnc_out[:, -index_shift:]
                    incoh_wnc[:,-index_shift:] += rel_wnc
                    incoh_wnc[:, :-index_shift] = incoh_wnc[:, -index_shift:2*-index_shift]
            elif index_shift == 0:
                rel_bart = spo.bart_out[:,:]
                incoh_bart += rel_bart
                if wnc == True:
                    rel_wnc = spo.wnc_out[:,:]
                    incoh_wnc += rel_wnc
            else:
                rel_bart = spo.bart_out[:, :-index_shift]
                incoh_bart[:,:-index_shift] += rel_bart
                incoh_bart[:, -index_shift:] = incoh_bart[:, -2*index_shift:-index_shift]
                if wnc == True:
                    rel_wnc = spo.wnc_out[:, :-index_shift]
                    incoh_wnc[:,:-index_shift] += rel_wnc
                    incoh_wnc[:, -index_shift:] = incoh_wnc[:, -2*index_shift:-index_shift]
    incoh_spo.bart_out = incoh_bart/num_tracking_els
    if wnc == True:
        incoh_spo.wnc_out = incoh_wnc / num_tracking_els
    return incoh_spo
       
        

def add_inset(axis, spo, wnc=False):
    #spo = deepcopy(spo)
    inset_axes = zoomed_inset_axes(axis, 2, 1)
    if wnc == True:
        inset_axes.pcolormesh(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.wnc_out[:,::PLOT_STRIDE],
                                   vmin=db_min, vmax=0, cmap=cmap)
    else:
        inset_axes.pcolormesh(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.bart_out[:,::PLOT_STRIDE],
                                   vmin=db_min, vmax=0, cmap=cmap)
    inset_axes.set_xlim(spo.rs - 400, spo.rs + 400)
    inset_axes.set_ylim(40, 70)
    #inset_axes.scatter(spo.rs, spo.zs, color='w', marker='+', s=24)
    #inset_axes.plot(spo.rs, spo.zs, color='w', marker='.',  markeredgecolor='k', markeredgewidth=1, markerfacecolor='w', markersize=12)

    inset_axes.plot(spo.rs, spo.zs, color='w', marker='+',  markeredgecolor='k', markeredgewidth=1)
    inset_axes.plot(spo.rs, spo.zs, color='w', marker='+',  markeredgecolor='w', markeredgewidth=.5)
    inset_axes.set_xticks([])
    inset_axes.set_yticks([])
    inset_axes.invert_yaxis()
    

    patch, pp1, pp2 = mark_inset(axis, inset_axes, loc1=2, loc2=4, linestyle='dashed', linewidth=0.8)
    pp1.loc1 = 3
    pp1.loc2 = 1
    pp2.loc1 = 2
    pp2.loc2 = 4
    pp1.set_linewidth(0.6)
    pp2.set_linewidth(0.6)
    patch.set_linestyle('solid')
    print('patch wid', patch.get_linewidth())

    #rect = inset_axes.patch
    #rect.set_linestyle('dashed')
    return

        


if __name__ == '__main__':

    cov_index = 4
    synth_cov_index = 3
    proj_str = 's5_quiet3'
    #proj_str = 's5_deep'
    N_fft = 2048
    num_snapshots = 36
    #num_snapshots = 15
    fact = 8
    N_fft = fact*N_fft

    subfolder =str(N_fft)
    num_snapshots = int(num_snapshots / fact)
    num_synth_els = 5
    num_tracking_els = num_synth_els
    tilt_angle = -1 
    num_freqs = 13
    wn_gain = -2.0
    synth_wn_gain = -.5
    cov_times=get_cov_time(proj_str, subfolder, num_snapshots, num_synth_els)
    cov_time = cov_times[cov_index]
    print(cov_time)
    v_arr = load_vel_arr(proj_str, subfolder, num_snapshots)
    if v_arr[0, -1] < cov_times[-1]:
        new_t = cov_times[-1]
        new_v = v_arr[1,-1]
        new_entry = np.array([new_t, new_v]).reshape(2,1)
        v_arr = np.concatenate((v_arr, new_entry), axis=1)
    v_interp = interp1d(v_arr[0,:], v_arr[1,:])
    v_source = v_interp(cov_time)
    print('v source', v_source)
    root_folder ='pickles/'

    #cmap =plt.cm.get_cmap('binary')
    #fig_name = proj_str + '_proc_comp_bw.png'
    cmap = plt.cm.get_cmap('viridis')
    fig_name = proj_str + '_proc_comp.png'

    """ Get tracking spo """
    tracking_spo = get_tracking_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle,num_freqs, num_tracking_els, v_interp, cov_index, wnc=True, wn_gain = wn_gain)

    """ synthetic array results """

    #synth_folder = str(N_fft*fact)
    synth_folder = subfolder
    #synth_snaps = int(num_snapshots / fact)
    synth_snaps = num_snapshots
    synth_cov_times=get_cov_time(proj_str, synth_folder, synth_snaps, num_synth_els)
    synth_cov_time = synth_cov_times[synth_cov_index]
    synth_v_arr = load_vel_arr(proj_str,synth_folder, synth_snaps) 
    if synth_v_arr[0, -1] < synth_cov_times[-1]:
        new_t = synth_cov_times[-1]
        new_v = synth_v_arr[1,-1]
        new_entry = np.array([new_t, new_v]).reshape(2,1)
        synth_v_arr = np.concatenate((synth_v_arr, new_entry), axis=1)
    synth_v_interp = interp1d(synth_v_arr[0,:], synth_v_arr[1,:])
    synth_v_source = synth_v_interp(synth_cov_time)
    spo = load_spo(root_folder, proj_str, synth_folder, synth_snaps, tilt_angle, num_synth_els, num_freqs, synth_v_source, synth_cov_time, synth_wn_gain)
    spo.get_bathy_corr()

    print('cov time, synth cov time', cov_time, synth_cov_time)


    fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
    db_max = np.max(spo.bart_out[:,::PLOT_STRIDE])
    spo.bart_out[:,::PLOT_STRIDE] -= db_max
    db_min = -10
    levels = np.linspace(-15, 0, 25)
    #CS1 = axes[0,0].contourf(spo.corr_grid_r, spo.corr_grid_z, spo.bart_out[:,::PLOT_STRIDE], 
    #cmap.set_under('blue')
    pcm = axes[0,2].pcolormesh(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.bart_out[:,::PLOT_STRIDE],
                                    vmin=db_min, vmax=0, cmap=cmap)
    #add_inset(axes[0,2], spo, wnc=False)
    
    db_max = np.max(spo.wnc_out[:,::PLOT_STRIDE])
    spo.wnc_out[:,::PLOT_STRIDE] -= db_max
    pcm1 = axes[1,2].pcolormesh(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.wnc_out[:,::PLOT_STRIDE],
                                    vmin=db_min, vmax=0, cmap=cmap)

    #add_inset(axes[1,2], spo, wnc=True)


   

    """
    Standard result """
    num_synth_els = 1
    spo = load_spo(root_folder, proj_str, subfolder, num_snapshots, tilt_angle, num_synth_els, num_freqs, v_source, cov_time, wn_gain)
    spo.get_bathy_corr()
    db_max = np.max(spo.bart_out[:,::PLOT_STRIDE])
    spo.bart_out[:,::PLOT_STRIDE] -= db_max
    #CS1 = axes[0,0].contourf(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.bart_out[:,::PLOT_STRIDE], 
    #cmap.set_under('blue')
    pcm = axes[0,0].pcolormesh(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.bart_out[:,::PLOT_STRIDE],
                                    vmin=db_min, vmax=0, cmap=cmap)
    #add_inset(axes[0,0], spo, wnc=False)

    db_max = np.max(spo.wnc_out[:,::PLOT_STRIDE])
    spo.wnc_out[:,::PLOT_STRIDE] -= db_max
    pcm1 = axes[1,0].pcolormesh(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.wnc_out[:,::PLOT_STRIDE],
                                    vmin=db_min, vmax=0, cmap=cmap)

    #add_inset(axes[1,0], spo,  wnc=True)


    """ tracking results """
    db_max = np.max(tracking_spo.bart_out[:,::PLOT_STRIDE])
    print('db_max' ,db_max)
    tracking_spo.bart_out[:,::PLOT_STRIDE] -= db_max
    #CS1 = axes[0,0].contourf(spo.corr_grid_r[::PLOT_STRIDE], spo.corr_grid_z, spo.bart_out[:,::PLOT_STRIDE], 
    #cmap.set_under('blue')
    pcm = axes[0,1].pcolormesh(tracking_spo.corr_grid_r[::PLOT_STRIDE], tracking_spo.corr_grid_z, tracking_spo.bart_out[:,::PLOT_STRIDE],
                                    vmin=db_min, vmax=0, cmap=cmap)
    #add_inset(axes[0,1], tracking_spo,  wnc=False)
    db_max = np.max(tracking_spo.wnc_out[:,::PLOT_STRIDE])
    tracking_spo.wnc_out[:,::PLOT_STRIDE] -= db_max
    print(spo.corr_grid_r[np.argmax(tracking_spo.wnc_out[:,::PLOT_STRIDE]) % spo.corr_grid_r.size])
    pcm1 = axes[1,1].pcolormesh(tracking_spo.corr_grid_r[::PLOT_STRIDE], tracking_spo.corr_grid_z, tracking_spo.wnc_out[:,::PLOT_STRIDE],
                                    vmin=db_min, vmax=0, cmap=cmap)

    #add_inset(axes[1,1], tracking_spo,  wnc=True)

    axes[1,0].invert_yaxis()


    letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    i = 0
    for ax in axes.ravel().tolist():
        #ax.plot(spo.rs, spo.zs, color='w', marker='+',  markeredgecolor='k', markeredgewidth=5, markerfacecolor='w', markersize=6)
        ax.plot(spo.rs, spo.zs, color='w', marker='+',  markeredgecolor='k', markeredgewidth=1)
        ax.plot(spo.rs, spo.zs, color='w', marker='+',  markeredgecolor='w', markeredgewidth=.5)
        ax.text(7500, 175, letters[i], color='w', fontsize=15)
        i+=1



    cb = fig.colorbar(pcm1, ax=axes.ravel().tolist())
    cb.set_label('dB', rotation='horizontal')

    cols = ['Traditional', 'MFT', 'Range-coherent']
    for i in range(len(cols)):
        ax = axes[0,i]
        ax.set_title(cols[i])

    rows = ['Bartlett', 'WNC']
    for i in range(len(rows)):
        ax = axes[i,0]
        ax.set_ylabel(rows[i])

    fig.text(0.5, 0.02, 'Range (m)', ha='center')
    fig.text(0.02, 0.5, 'Depth (m)', va='center', rotation='vertical')
    """
    fig.text(0.02, 0.75, 'Bartlett', va='center')#, rotation='vertical')
    fig.text(0.02, 0.25, 'WNC', va='center')#, rotation='vertical')
    fig.text(0.15,.9, 'Simple VLA')
    fig.text(0.4, .9, 'MFT')
    fig.text(0.6, .9, 'Range-coherent')
    """
 

    

    fig.set_size_inches(8, 4)
    plt.savefig('/home/hunter/research/coherent_matched_field/paper/pics/' + fig_name, dpi=500, orientation='landscape')

    plt.show()
