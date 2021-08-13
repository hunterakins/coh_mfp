import numpy as np
from matplotlib import pyplot as plt

"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego


"""


a = np.array([x for x in range(8)])
a = a.reshape(4,2)

b = np.array([x for x in range(8)])
b = b.reshape(2,4)

print(a, b)

c = np.einsum('ij,ji->i', a, b)
print(c)

