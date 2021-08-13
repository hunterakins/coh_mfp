import numpy as np
from matplotlib import pyplot as plt
from env.env.env_loader import s5_approach_D
import os
import pickle

"""
Description:
Processor output object

Date:
3/1/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


def load_spo(root_folder, proj_str, subfolder, num_snaps, tilt_angle, num_synth_els,num_freqs, v_source, t, wn_gain=None):
    fname = root_folder + proj_str +'/' + subfolder +'/' + make_leaf_name(num_snaps, tilt_angle, num_synth_els, num_freqs, v_source, t, wn_gain)
    with open(fname, 'rb') as f:
        spo_obj = pickle.load(f)
    return spo_obj

def make_leaf_name(num_snaps, tilt_angle, num_synth_els, num_freqs, v_source, t, wn_gain=None):
    name = '_'.join([str(x) for x in [num_snaps, tilt_angle, num_synth_els, num_freqs, v_source,  t]]) + '.pickle'
    if type(wn_gain) != type(None):
        name = str(wn_gain) + '_' + name
    return name

def make_spo_pickle_name(spo, root_folder, wn_gain=None):
    """
    Make a pickly name for this little motherfucker 
    """
    if spo.proj_str not in os.listdir(root_folder):
        os.mkdir(root_folder +spo.proj_str)
    root_folder += spo.proj_str 
    if spo.subfolder not in os.listdir(root_folder):
        os.mkdir(root_folder + '/' + spo.subfolder)
    root_folder += '/' + spo.subfolder + '/'
    name = make_leaf_name(spo.num_snapshots, spo.tilt_angle, spo.num_synth_els, len(spo.source_freq), spo.v, spo.t, wn_gain)
    return root_folder + name
    

class SwellProcObj:
    """
    For data assembled into a cov mat, store
    the GPS true range and depth, experiment time
    associated with the center data frame in cov mat
    as well as the un-normalized bartlett amb surf in db, 
    assumed velocity, assumed tilt
    """
    def __init__(self, source_freq, num_snapshots, proj_str, subfolder, rs, zs, cov_t, v_source, tilt_angle, num_synth_els, r, z, bart_out):
        """
        Source freq may be a list if a stacking has been performed
        """
        self.source_freq = source_freq
        self.proj_str = proj_str
        self.subfolder = subfolder
        self.num_snapshots = num_snapshots
        self.rs = rs
        self.zs = zs
        self.t = cov_t
        self.v = v_source
        self.num_synth_els= num_synth_els
        self.tilt_angle = tilt_angle
        self.grid_r = r
        self.grid_z = z
        self.bart_out = bart_out

    def add_wnc(self, wn_gain, wnc_out):
        self.wn_gain = wn_gain
        self.wnc_out = wnc_out
        return

    def get_bathy_corr(self):
        """
        Use the bathymetry to correct the MFP peak location
        """
        d_r= s5_approach_D()
        true_depth = d_r(self.rs)
        self.d_r = true_depth
        corr_factor = true_depth /216.5
        self.corr_factor = corr_factor
        self.corr_grid_r = corr_factor * self.grid_r
        self.corr_grid_z = corr_factor * self.grid_z
        return

    def save(self, root_folder='pickles/'):
        if hasattr(self, 'wn_gain'):
            name = make_spo_pickle_name(self, root_folder, self.wn_gain)
        else:
            name = make_spo_pickle_name(self, root_folder)
        with open(name, 'wb') as f:
            pickle.dump(self,f)
        return name
        
        
        

