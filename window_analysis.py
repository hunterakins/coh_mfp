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

window_len = 1024
N_fft = 4*8096
hamm = get_window('hamming', window_len)

transfer_func = np.fft.fft(hamm)
freqs = np.fft.fftfreq(window_len, 1/1500)

transfer_func = np.square(abs(transfer_func))
transfer_func /= np.max(transfer_func)
trans_db = 10*np.log10(transfer_func)

plt.figure()
plt.plot(freqs, trans_db)


hamm_pad = np.pad(hamm, (0, N_fft-window_len),'constant')
transfer_func = np.fft.fft(hamm_pad)
freqs = np.fft.fftfreq(N_fft, 1/1500)

transfer_func = np.square(abs(transfer_func))
transfer_func /= np.max(transfer_func)
trans_db = 10*np.log10(transfer_func)

plt.plot(freqs, trans_db)
plt.show()
