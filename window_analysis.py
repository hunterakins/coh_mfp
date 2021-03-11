import numpy as np
from matplotlib import pyplot as plt

"""
Description:
Make sure the windowing effects are well-understood

Date:
3/9/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


from scipy.signal.windows import get_window
hamm = get_window('hamming', 2048)

transfer_func = np.fft.fft(hamm)
freqs = np.fft.fftfreq(2048, 1/1500)

transfer_func /= np.max(abs(transfer_func))
trans_db = 10*np.log10(transfer_func)

plt.figure()
plt.plot(freqs, trans_db)


hamm_pad = np.pad(hamm, (0, 8096-2048),'constant')
transfer_func = np.fft.fft(hamm_pad)
freqs = np.fft.fftfreq(8096, 1/1500)

transfer_func /= np.max(abs(transfer_func))
trans_db = 10*np.log10(transfer_func)

plt.plot(freqs, trans_db)
plt.show()
