import numpy as np
import sys
from matplotlib import pyplot as plt
from env.env.json_reader import write_json, read_json

'''
Description:
Create experiment class to store all vars
json read write interface

Date: 
8/25/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


""" Pick source params
and receiver config """

conf_folder = '/oasis/tscc/scratch/fakins/coh_mfp/confs/'

class ExpConf:
    def __init__(self, var_dict):
        self.freqs = var_dict['freqs']
        self.fft_len = var_dict['fft_len']
        self.acc_amp = var_dict['acc_amp']
        self.acc_T = var_dict['acc_T']
        self.source_vel = var_dict['source_vel']
        self.fft_len = var_dict['fft_len']
        self.fft_spacing = var_dict['fft_spacing']
        self.SNR = var_dict['SNR']
        self.fs = var_dict['fs']
        self.ship_dr = var_dict['ship_dr']
        self.dz = var_dict['dz']
        self.zmax = var_dict['zmax']
        self.dr = var_dict['dr']
        self.rmax = var_dict['rmax']
        self.r0 = var_dict['r0']
        self.r1 = var_dict['r1']
        self.zr = var_dict['zr']
        self.zs = var_dict['zs']
        self.cov_int_time = var_dict['cov_int_time']
        self.num_ranges = var_dict['num_ranges']
        self.n_overlap = var_dict['n_overlap']
        self.proj_root = var_dict['proj_root']
        self.fig_folder = var_dict['fig_folder']
        self.num_realizations = var_dict['num_realizations']
        self.exp_id = var_dict['exp_id']
   
def load_config(exp_id):
    name = make_conf_name(exp_id)
    var_dict = read_json(name)
    exp_conf = ExpConf(var_dict)
    return exp_conf

def get_conf_dict(var_dict):
    save_dict = var_dict.copy()

    builtin_keys = [x for x in save_dict.keys() if x[:2] == '__']
    mod_keys = ['np', 'conf', 'write_json', 'os']
    other_keys = ['var_dict']

    annoying_keys = mod_keys + builtin_keys + other_keys
    for key in annoying_keys:
        save_dict.pop(key)
    return save_dict

def make_conf_name(exp_id):
    return conf_folder + str(exp_id) + '.json'

