import numpy as np
from matplotlib import pyplot as plt

"""
Description:
Show how to combine snapshots into a synthetic array

Date:
3/23/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""



fig = plt.figure()
ax = fig.add_subplot(111)

rcvrs = np.linspace(0, 20, 21)

dot_dot_dot = np.array([np.median(rcvrs)]*3)

dom = np.linspace(0, 20, 21)

ax.scatter([dom[0]]*rcvrs.size, rcvrs, color='k')
ax.scatter([0.2]*rcvrs.size, rcvrs, color='k')
ax.scatter([0.4]*rcvrs.size, rcvrs, color='k')
ax.scatter([dom[1]]*rcvrs.size, rcvrs, color='k')
ax.text(0.1, 0, '{', rotation=90, fontsize=154)

dot_dot_dot_dom = [0.45, 0.5, 0.55]
ax.scatter(dot_dot_dot_dom, dot_dot_dot, color='k', s=10)


ax.set_xlim([-.5, 1.5])
ax.set_xticks([])
ax.set_yticks([])
plt.show()

