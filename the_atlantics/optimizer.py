from argparse import ArgumentParser
import numpy as np
from snc import SnCalc, assign_params_default
from scipy.interpolate import interp1d
from scipy.optimize import minimize, brentq
from scipy.integrate import simps
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--nz-file", default="nzs.npy", type=str,
                    help="Path to input file containing the N(z) of all initial groups")
parser.add_argument("--prefix-out", default='out', type=str,
                    help="Prefix for all output files")
parser.add_argument("--n-bins", default=4, type=int,
                    help="Number of bins (excluding a possible trash bin")
parser.add_argument("--prob-in-bin", default=-1., type=float,
                    help="Minimum fraction of the N(z) that should be inside the bin."
                    "If <= 0, this will not be taken into account in the optimization.")
parser.add_argument("--prob-out-bin", default=-1., type=float,
                    help="Maximum fraction of the N(z) that should be inside other bins."
                    "If <= 0, this will not be taken into account in the optimization.")
o = parser.parse_args()

nzs = np.load(o.nz_file)
# We should probably read z on input too
z_arr = np.linspace(0, 2, 124)
zz = (z_arr[1:]+z_arr[:-1])*0.5
# Renormalize to use the same sample definitions as the
# official metric calculators.
fsky = 0.25
ndens_arcmin = 20.
area_arcmin = (180*60/np.pi)**2*4*np.pi*fsky
ng_tot = ndens_arcmin * area_arcmin
ng_each_small = simps(nzs, x=zz)
ng_tot_small = np.sum(ng_each_small)
ng_each = ng_each_small * ng_tot / ng_tot_small
nzs = nzs * ng_tot / ng_tot_small

# Initialize calculator
sc = SnCalc(zz, nzs, fsky=fsky)
sc.get_cl_matrix(fname_save=o.prefix_out + 'cl_wl.npz')

# Initial edge guess (defined as having equal number of galaxies)
nz_tot = np.sum(nzs, axis=0)
cumulative_fraction = np.cumsum(nz_tot) / np.sum(nz_tot)
cumul_f = interp1d(zz, cumulative_fraction, bounds_error=False,
                   fill_value=(0, 1))
edges_0 = np.array([brentq(lambda z : cumul_f(z) - q, 0, 2)
                    for q in (np.arange(o.n_bins-1)+1.)/o.n_bins])


# Minimize
# Binning parameters
params = assign_params_default.copy()
if o.prob_in_bin > 0:
    params['p_inbin_thr'] = o.prob_in_bin
    params['use_p_inbin'] = True
if o.prob_out_bin > 0:
    params['p_outbin_thr'] = o.prob_out_bin
    params['use_p_outbin'] = True


def minus_sn(edges, calc):
    return -calc.get_sn_from_edges(edges, assign_params=params)

    
# Actual optimization
res = minimize(minus_sn, edges_0, method='Powell', args=(sc,))
edges_1 = res.x

# Post-process:
# S/N
sn = sc.get_sn_from_edges(edges_1, assign_params=params)
# N(z)s of the final bins
nz_best = sc.get_nzs_from_edges(edges_1, assign_params=params)
# Group assignment
assign = sc.assign_from_edges(edges_1, assign_params=params,
                              get_ids=True)


def get_bin_name(i):
    if i == -1:
        return 'bin_trash'
    else:
        return 'bin_%d' % i

print("Redshift edges: ", edges_1)
print("Optimal S/N: ", sn)
n_tot = simps(np.sum(nz_best, axis=0),
              x=zz)

d_out = {'sn': sn,
         'edges': edges_1}
plt.figure()
for (i, a), n in zip(assign, nz_best):
    name = get_bin_name(i)
    d_out[name+'_groups'] = a
    d_out[name+'_nz'] = n
    print("Bin %d, groups: " % i, a)
    n_here = simps(n, x=zz)
    f = n_here / n_tot
    frac = "%.1lf" % (100 * f)
    plt.plot(zz, n/n_here,
             label=f'Bin {i} ({frac} %)')
plt.legend()
plt.xlabel(r'$z$', fontsize=15)
plt.ylabel(r'$p(z)$', fontsize=15)
plt.show()

np.savez(o.prefix_out + '_bin_info.npz', **d_out)
