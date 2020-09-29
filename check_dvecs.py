import numpy as np
from matplotlib import pyplot as plt
from coh_mfp.config import freqs
from coh_mfp.get_dvecs import load_dvec, load_tgrid

'''
Description:
Look at the data vectors extracted using get_dvecs

Date: 8/26/2020

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''

if __name__ == '__main__':
    tgrid = load_tgrid()
    for freq in freqs:
        dvec = load_dvec(freq)
        fig = plt.figure()
        plt.plot(tgrid, dvec[0,:])
        plt.savefig(str(freq) + '_d0vec.png')
        plt.close(fig)

