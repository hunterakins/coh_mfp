import numpy as np
from matplotlib import pyplot as plt

'''
Description:

Date: 

Author: Hunter Akins

Institution: UC San Diego, Scripps Institution of Oceanography

'''


x = np.load('check_vals.npy')
plt.plot(x[1,:], x[0,:])
plt.show()
