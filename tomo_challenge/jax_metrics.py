import jax.numpy as np
import jax.random as rand
from jax import lax, jit, vmap, grad

from jax_cosmo.redshift import kde_nz
from jax_cosmo.core import Cosmology
from jax_cosmo.tracers import get_lensing_tracer_fn
from jax_cosmo.angular_cl import angular_cl

#SNR_SCORE_BASELINE = 266.5
# This is the score for the linear power spectrum only
SNR_SCORE_BASELINE = 138.4

@jit
def compute_mean_covariance(weights, labels, kernel_bandwidth=0.01):
    """
    JAX compatible version of the tomo challenge function

    Compute a mean and covariance for the chosen distribution of objects.

    This assumes a cosmology, and the varions parameters affecting the noise
    and signal: the f_sky, ell choices, sigma_e, and n_eff
    """
    # plausible limits I guess
    ell_max = 2000
    n_ell = 100

    # 10,000 sq deg
    f_sky = 0.25

    # pretend there is no evolution in measurement error.
    # because LSST is awesome
    sigma_e = 0.26

    # assumed total over all bins, divided proportionally
    n_eff_total_arcmin2 = 20.0

    # Use this fiducial cosmology, which is what I have for TXPipe
    cosmo = Cosmology(
        Omega_c = 0.22,
        Omega_b = 0.0447927,
        h = 0.71,
        n_s = 0.963,
        sigma8 = 0.8,
        Omega_k=0.,
        w0=-1., wa=0.
    )

    # choose ell bins from params  above
    ell = np.logspace(2, np.log10(ell_max), n_ell)

    # work out the number density per steradian
    steradian_to_arcmin2 = 11818102.86004228
    n_eff_total = n_eff_total_arcmin2 * steradian_to_arcmin2

    # Get the number of galaxiex.
    ngal = len(weights)

    nbins = weights.shape[-1]

    # # Generate CCL tracers and get total counts, for the noise
    counts = []
    for b in range(nbins):
        # total number of objects in this histogram bin
        counts.append(weights[:,b].sum())

    # Get the fraction of the total possible number of objects
    # in each bin, and the consequent effective number density.
    # We pretend here that all objects have the same weight.
    # JAX modif: here instead we still know that all galaxies add up to 1,
    # but they have bin specific weights
    fractions = np.array([c / len(labels) for c in counts])
    n_eff = np.array([n_eff_total * f for f in fractions])

    # Define an ordering of the theory vector
    blocks = []
    for i in range(nbins):
        for j in range(i, nbins):
            blocks.append((i,j))

    def find_index(a, b):
        if (a,b) in blocks:
            return blocks.index((a,b))
        else:
            return blocks.index((b,a))

    # Define an ordering for the covariance matrix blocks
    cov_blocks = []
    for (i,j) in blocks:
        for (m,n) in blocks:
            cov_blocks.append((find_index(i,m),
                               find_index(j,n),
                               find_index(i,n),
                               find_index(j,m)))
    blocks = np.array(blocks)
    cov_blocks = np.array(cov_blocks)

    @jit
    def get_cl(inds):
        nz1 = kde_nz(zcat=labels, weight=weights[:,inds[0]], bw=kernel_bandwidth, zmax=4.)
        nz2 = kde_nz(zcat=labels, weight=weights[:,inds[1]], bw=kernel_bandwidth, zmax=4.)
        return angular_cl(cosmo, ell, get_lensing_tracer_fn(nz1), get_lensing_tracer_fn(nz2))

    cl_signal = lax.map(get_cl, blocks)

    def get_noise_cl(inds):
        i,j = inds
        delta = 1. - np.clip(np.abs(i-j), 0., 1.)
        return sigma_e**2/n_eff[i]*delta * np.ones(n_ell)

    cl_noise = lax.map(get_noise_cl, blocks)

    # Adding noise to auto-spectra
    cl_obs = cl_signal + cl_noise

    norm = (2*ell + 1) * np.gradient(ell) * f_sky

    def get_cov_block(inds):
        a, b, c, d = inds
        cov = (cl_obs[a]*cl_obs[b] + cl_obs[c]*cl_obs[d])*np.eye(n_ell) / norm
        return cov

    cov_mat = lax.map(get_cov_block, cov_blocks)

    # Reshape covariance matrix into proper matrix
    cov_mat = cov_mat.reshape((len(blocks), len(blocks), n_ell, n_ell))
    cov_mat = cov_mat.transpose(axes=(0,2,1,3)).reshape((n_ell*len(blocks),
                                                         n_ell*len(blocks)))

    return cl_signal.flatten(), cov_mat

def compute_snr_score(weights, labels):
    """Compute a score metric based on the total spectrum S/N

    This is given by sqrt(mu^T . C^{-1} . mu) - baseline
    where mu is the theory prediction and C the Gaussian covariance
    for this set of bins. The baseline is the score for no tomographic binning.

    Parameters
    ----------
    tomo_bin: array
        Tomographic bin choice (0 .. bin_max) for each object in the survey
    z: array
        True redshift for each object

    Returns
    -------
    score: float
        Metric for this configuration
    """
    mu, C = compute_mean_covariance(weights, labels)
    # S/N for correlated data, I assume, from generalizing
    # sqrt(sum(mu**2/sigma**2))
    P = np.linalg.inv(C)
    score = (mu.T @ P @ mu)**0.5 - SNR_SCORE_BASELINE

    return score
