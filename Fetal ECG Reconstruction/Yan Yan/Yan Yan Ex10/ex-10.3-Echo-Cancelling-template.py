import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from lms_filter_template import lms_filter
from wiener_filter_template import wiener_filter

# ---------------------------------------------------------------------------------------
# load data
fs, data = wavfile.read('audio.wav')

y_clean = data[:, 0].astype(float)
y_mixed = data[:, 1].astype(float)

# noramlize for numerical stability
y_clean /= max(abs(y_clean)) # voice of p1
y_mixed /= max(abs(y_mixed)) # voide of p2 and echo from p1


dummy_array = np.zeros_like(y_clean)

# ---------------------------------------------------------------------------------------
# LMS and Wiener Filter

#USE_FILER = 'LMS'
USE_FILER = 'WIENER'

L = 1200 # filter length
mu = 1.2e-2 # LMS step size

# CHANGE NEXT TWO LINES !!!
u = y_clean
y = y_mixed

if USE_FILER == 'LMS':
    y_hat, h_hat, error = lms_filter(u, y, L, mu)

if USE_FILER == 'WIENER':
    y_hat, h_hat = wiener_filter(u, y, L)

# CHANGE NEXT LINE !!!
y_out = y_mixed-y_hat


# ---------------------------------------------------------------------------------------
# plot and wav export

# export wav: DO NOT CHANGE!
float32_data = y_out/np.ptp(y_out) 
float32_data = float32_data*32767
wavfile.write('audio_processed.wav', fs, float32_data.astype(np.int16))

# plot
fig, axs = plt.subplots(3, sharex='all')

axs[0].plot(y_clean, c='k', lw=0.8, label=r'$u$ (clean)')
axs[1].plot(y_mixed, c='k', lw=0.8, label=r'$y$ (mixed up)')
axs[1].plot(y_hat, c='b', lw=1.2, label=r'$\hat{y}} (estimated)')
axs[2].plot(y_out, c='b', lw=0.8, label=r'$y_{\text{out}}$ (estimated)')

for ax in axs:
    ax.grid(lw=0.5, c='grey', ls=':')
    ax.legend(loc=1)
    ax.set_yticks([])
axs[-1].set_xlabel('k')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------------------
'''
(b)

the Wiener Filter is offline, there is a time offset, and this filter runs slower.
The Wiener ﬁﬁlter is a common method for estimating ﬁﬁlter 
coeﬃﬃﬃcients that do not change over time.
Compared with LMS filter, the Wiener filter can give back the optimum least square solution for the entire signal y.
'''