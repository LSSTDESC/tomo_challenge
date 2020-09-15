import numpy as np
import matplotlib.pyplot as plt
from snc import SnCalc


# Let's start with a "large" set of 12 samples
n_bins = 12
z_cent = np.linspace(0.2, 1.4, n_bins+1)+0.5/n_bins
sigma_z = 0.05 * (1+z_cent)
z_arr = np.linspace(0, 2, 1024)
# Total number of galaxies in each sample
n_gals = (z_cent/0.5)**2*np.exp(-(z_cent/0.5)**1.5) * 10. * 1.8E4 * 60**2


def pz_gau(z, zc, sz):
    return np.exp(-0.5*((z-zc)/sz)**2)/np.sqrt(2*np.pi*sz**2)


# Create the N(z) arrays, which should integrate to the total number
# of galaxies in each sample
nzs = [ng * pz_gau(z_arr, zc, sz)
       for ng, zc, sz in zip(n_gals, z_cent, sigma_z)]

# Let's plot them
plt.figure()
for nz in nzs:
    plt.plot(z_arr, nz)
plt.xlabel(r'$z$', fontsize=16)
plt.ylabel(r'$N(z)$', fontsize=16)


c = SnCalc(z_arr, nzs)
# Now let's do a dumb thing and bunch them up into
# 4, 3, 2 and 1 subsamples
assign4 = [np.array([0, 1, 2]),
           np.array([3, 4, 5]),
           np.array([6, 7, 8]),
           np.array([9, 10, 11])]
assign3 = [np.array([0, 1, 2, 3]),
           np.array([4, 5, 6, 7]),
           np.array([8, 9, 10, 11])]
assign2 = [np.array([0, 1, 2, 3, 4, 5]),
           np.array([6, 7, 8, 9, 10, 11])]
assign1 = [np.arange(n_bins)]


# OK, caclulate S/N for all of these groupings
snr_d4 = c.get_sn(assign4, full_output=True)
print('4 bins, S/N = %.1lf' % (snr_d4['snr']))
snr_d3 = c.get_sn(assign3, full_output=True)
print('3 bins, S/N = %.1lf' % (snr_d3['snr']))
snr_d2 = c.get_sn(assign2, full_output=True)
print('2 bins, S/N = %.1lf' % (snr_d2['snr']))
snr_d1 = c.get_sn(assign1, full_output=True)
print('1 bins, S/N = %.1lf' % (snr_d1['snr']))


# Note that the calculator also returns the power
# spectra, if you wanna look at them.
plt.figure()
for i in range(4):
    for j in range(i, 4):
        if i == j:
            plt.plot(snr_d4['ls'],
                     snr_d4['cl'][:, i, j],
                     'k-')
            plt.plot(snr_d4['ls'],
                     snr_d4['nl'][0, i, j]*np.ones_like(snr_d4['ls']), 'k--')
        else:
            plt.plot(snr_d4['ls'],
                     snr_d4['cl'][:, i, j],
                     '-', c='#AAAAAA')
plt.loglog()
plt.show()
