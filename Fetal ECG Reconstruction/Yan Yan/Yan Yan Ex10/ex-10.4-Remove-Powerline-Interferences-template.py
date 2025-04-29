import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

from lms_filter_template import lms_filter

# ---------------------------------------------------------------------------------------
# load data
ecg_data_table = np.genfromtxt('secg-poweline-50Hz-fs-512Hz.csv', delimiter=',', dtype=float)
fs = 512
t = ecg_data_table[:, 0]  # time
y_ecg_powerline = ecg_data_table[:, 1]  # observations
y_ecg = ecg_data_table[:, 2]   # "true" signal
y_powerline = ecg_data_table[:, 3]   # powerline

dummy_array = np.zeros_like(y_ecg_powerline)

# Generate reference signal
# CHANGE & COMPLETE CODE !
ind, _ = find_peaks(y_powerline, distance=10)
y_dirac_comb_50hz = np.zeros_like(t)
y_dirac_comb_50hz[ind] = 1
# ---------------------------------------------------------------------------------------
# LMS Filter

L = 50 # filter length
mu = 20e-3 # LMS step size

# CHANGE NEXT TWO LINES !!!
u = y_dirac_comb_50hz
y = y_ecg_powerline

y_hat, h_hat, error = lms_filter(u, y, L, mu)

# CHANGE NEXT LINE !!!
y_out = y_ecg_powerline - y_hat


# ---------------------------------------------------------------------------------------
# plot
fig, axs = plt.subplots(4, sharex='all')
axs[0].plot(y_ecg_powerline, c='k', lw=0.8, label=r'$y$ (ecg + powerline interference)')
axs[1].plot(y_dirac_comb_50hz, c='k', lw=0.8, label=r'$u$ (powerline pulse train)')
axs[2].plot(y_hat, c='b', lw=1.2, label=r'$\hat{y}$ (estimated powerline in $u$)')
axs[3].plot(y_out, c='b', lw=0.8, label=r'$\hat{y}_\text{ecg}$ (estimated ecg)')

for ax in axs:
    ax.grid(lw=0.5, c='grey', ls=':')
    ax.legend(loc=1)
    ax.set_yticks([])
axs[-1].set_xlabel('k')

plt.tight_layout()
plt.show()
