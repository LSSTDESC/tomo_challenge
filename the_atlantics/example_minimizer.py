import numpy as np
import matplotlib.pyplot as plt
from snc import SnCalc
from scipy.optimize import minimize


# Let's start with a "large" set of 12 samples
n_bins = 100
z_cent = np.linspace(0.2, 1.4, n_bins+1)+0.5/n_bins
sigma_z = 0.05 * (1+z_cent)
z_arr = np.linspace(0, 2, 1024)
# Total number of galaxies in each sample
n_gals = (z_cent/0.5)**2*np.exp(-(z_cent/0.5)**1.5) * 10. * 1.8E4 * 60**2


def pz_gau(z, zc, sz):
    return np.exp(-0.5*((z-zc)/sz)**2)/np.sqrt(2*np.pi*sz**2)


# Create the N(z) arrays, which should integrate to the total number
# of galaxies in each sample.
nzs = [ng * pz_gau(z_arr, zc, sz)
       for ng, zc, sz in zip(n_gals, z_cent, sigma_z)]

# Let's plot them
plt.figure()
for nz in nzs:
    plt.plot(z_arr, nz)
plt.xlabel(r'$z$', fontsize=16)
plt.ylabel(r'$N(z)$', fontsize=16)

# Now initialize a S/N calculator for these initial groups.
c_wl = SnCalc(z_arr, nzs, use_clustering=False)
c_gc = SnCalc(z_arr, nzs, use_clustering=True)
print("Initializing WL")
c_wl.get_cl_matrix(fname_save='wl_nb%d.npz' % n_bins)
print("Initializing GC")
c_gc.get_cl_matrix(fname_save='gc_nb%d.npz' % n_bins)

# Let's focus on WL for now.
# OK, let's brute-force explore the total S/N as a function
# of the bin edge for a 2-bin scenario.
print("Exploring 2-bins")
zs = np.linspace(0.01, 1.5, 128)
sns_wl = np.array([c_wl.get_sn_from_edges(np.array([z])) for z in zs])
sns_gc = np.array([c_gc.get_sn_from_edges(np.array([z])) for z in zs])
plt.figure()
plt.plot(zs, sns_wl/np.amax(sns_wl), 'r-', label='Lensing')
plt.plot(zs, sns_gc/np.amax(sns_gc), 'k-', label='Clustering')
plt.xlabel('$z$')
plt.ylabel(r'$(S/N)/(S/N)_{\rm max}$')
plt.legend(loc='upper left', frameon=False)

# Now let's do a 3-bin case (more coarsely-grained so we can do it quickly)
print("Exploring 3-bins")
zs = np.linspace(0.01, 1.5, 32)
sns = np.array([[c_wl.get_sn_from_edges(np.array([z1, z2])) for z1 in zs]
                for z2 in zs])

plt.figure()
plt.imshow(sns, vmin=1440, extent=[0.01, 1.5, 0.01, 1.5],
           origin='lower')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
cb = plt.colorbar()
cb.set_label('$S/N$')


# Finally, let's write the function to minimize and optimize for a 4-bin case.
def minus_sn(edges, calc):
    return -calc.get_sn_from_edges(edges)


print("Optimizing WL")
edges_0 = np.array([0.3, 0.6, 0.9])
res = minimize(minus_sn, edges_0, method='Powell', args=(c_wl,))
print("WL final edges: ", res.x)
print("Maximum S/N: ", c_wl.get_sn_from_edges(res.x))
print(" ")

# That was for weak lensing. Let's do the same thing for clustering.
print("Optimizing GC")
res = minimize(minus_sn, edges_0, method='Powell', args=(c_gc,))
print("GC final edges: ", res.x)
print("Maximum S/N: ", c_gc.get_sn_from_edges(res.x))
plt.show()
