import numpy as np
import matplotlib.pyplot as plt

from wiener_filter_template import wiener_filter

# ---------------------------------------------------------------------------------------
# load data
data = np.genfromtxt("ecg_maternal_fetal.csv", delimiter=",", skip_header=1)

y_maternal = data[:, 0]
y_maternal_fetal = data[:, 1]

dummy_array = np.random.randn(len(y_maternal))*1e-3

# ---------------------------------------------------------------------------------------
# Wiener Filter
L = 10 # filter length

# CHANGE NEXT TWO LINES !!!
u = y_maternal
y = y_maternal_fetal

y_hat, h_hat = wiener_filter(u, y, L)

# CHANGE NEXT LINE !!!
y_fetal = y-y_hat

# print coefficients
print("Estimate Coefficients: \n", np.round(h_hat, 2))

# ---------------------------------------------------------------------------------------
# plot
fig, axs = plt.subplots(3, sharex='all')

axs[0].plot(y_maternal, c='k', lw=0.8, label=r'$\text{ECG}_{\text{maternal}}$')
axs[1].plot(y_maternal_fetal, c='k', lw=0.8, label=r'$\text{ECG}_{\text{maternal+fetal}}$')
axs[1].plot(y_hat, c='b', lw=1.2, label=r'$\text{ECG}_{\text{maternal}}$ (estimated)')
axs[2].plot(y_fetal, c='b', lw=0.8, label=r'$\text{ECG}_{\text{fetal}}$ (estimated)')

for ax in axs:
    ax.grid(lw=0.5, c='grey', ls=':')
    ax.legend(loc=1)
    ax.set_yticks([])
axs[-1].set_xlabel('k')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------
'''
optional
'''
y_maternal_clean=data[:,3]
y_maternal_fetal_clean=data[:,4]

u=y_maternal_clean
y=y_maternal_fetal_clean
_, h_hat = wiener_filter(u, y, L)
print(f"h coefficient: {h_hat}")
