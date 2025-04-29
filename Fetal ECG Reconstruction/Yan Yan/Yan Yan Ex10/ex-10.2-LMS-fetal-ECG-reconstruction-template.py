import numpy as np
import matplotlib.pyplot as plt

from lms_filter_template import lms_filter


# ---------------------------------------------------------------------------------------
# load data
data = np.genfromtxt("ecg_maternal_fetal.csv", delimiter=",", skip_header=1)

y_maternal = data[:, 0]
y_maternal_fetal = data[:, 1]

dummy_array = np.zeros_like(y_maternal)

# ---------------------------------------------------------------------------------------
# LMS Filter
L = 10 # filter length
mu = 0.25 # step size

# CHANGE NEXT TWO LINES !!!
u = y_maternal
y = y_maternal_fetal

y_hat, h_hat, error = lms_filter(u, y, L, mu)

# CHANGE NEXT LINE !!!
y_fetal = y_maternal_fetal-y_hat

# ---------------------------------------------------------------------------------------
# plot
fig, axs = plt.subplots(4, sharex='all')

axs[0].plot(y_maternal, c='k', lw=0.8, label=r'$\text{ECG}_{\text{maternal}}$')
axs[1].plot(y_maternal_fetal, c='k', lw=0.8, label=r'$\text{ECG}_{\text{maternal+fetal}}$')
axs[1].plot(y_hat, c='b', lw=1.2, label=r'$\text{ECG}_{\text{maternal}}$ (estimated)')
axs[2].plot(y_fetal, c='b', lw=0.8, label=r'$\text{ECG}_{\text{fetal}}$ (estimated)')
axs[3].plot(h_hat, label=[r'$\hat{h}$ (estimated filter weights)']+[None]*(L-1))

for ax in axs:
    ax.grid(lw=0.5, c='grey', ls=':')
    ax.legend(loc=1)
    ax.set_yticks([])
    ax.set_title("L=10")
axs[-1].set_xlabel('k')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------
'''
(c) 
if we change the value of L, we can see that the larger the L is , the estimated ECG curve changes more drastically, and h_hat varies more
the smaller the L is, the estimated ECG curve and h_hat change more smoothly and stable
'''