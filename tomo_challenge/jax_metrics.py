import jax.numpy as np
import jax.random as rand
from jax import lax, jit, vmap, grad
from functools import partial
import jax_cosmo as jc

SNR_SCORE_BASELINE = 266.5

def ell_binning():
    # we put this here to make sure it's used consistently
    # plausible limits I guess
    ell_max = 2000
    n_ell = 100
    # choose ell bins from 10 .. 2000 log spaced
    ell_edges  = np.logspace(2, np.log10(ell_max), n_ell+1)
    ell = 0.5*(ell_edges[1:]+ell_edges[:-1])
    delta_ell =(ell_edges[1:]-ell_edges[:-1])
    return ell, delta_ell

@jit
def compute_mean_covariance(weights, labels, params=None, kernel_bandwidth=0.01):
    """
    JAX compatible version of the tomo challenge function

    By default we are only considering the 3x2pt, because I'm too lazy to
    handle flags.
    Compute a mean and covariance for the chosen distribution of objects.

    If params are provided, it's understood to be an array of the following
    parameters:
      Omega_c
      Omega_b
      h
      sigma8
      n_s

    This assumes a cosmology, and the varions parameters affecting the noise
    and signal: the f_sky, ell choices, sigma_e, and n_eff
    """
    what = '3x2'
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
    if params is None:
        cosmo = jc.Cosmology(
            Omega_c = 0.27,
            Omega_b = 0.045,
            h = 0.67,
            n_s = 0.96,
            sigma8 = 0.8404844953840714,
            Omega_k=0.,
            w0=-1., wa=0.
        )
    else:
        cosmo = jc.Cosmology(
            Omega_c = params[0],
            Omega_b = params[1],
            h = params[2],
            n_s =  params[3],
            sigma8 = params[4],
            Omega_k=0.,
            w0=-1., wa=0.
        )

    # choose ell bins from params  above
    ell, delta_ell = ell_binning()

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
    n_eff = np.array([n_eff_total_arcmin2 * f for f in fractions])
    # Create redshift bins
    nzs = []
    for i in range(nbins):
      nzs.append(jc.redshift.kde_nz(labels, weights[:,i], bw=kernel_bandwidth,
                                    gals_per_arcmin2=n_eff[i], zmax=4.))

    probes = []

    # start with number counts
    if (what == 'gg' or what == '3x2'):
      # Define a bias parameterization
      bias = jc.bias.inverse_growth_linear_bias(cosmo, 1.)
      probes.append(jc.probes.NumberCounts(nzs, bias))

    if (what == 'ww' or what == '3x2'):
      probes.append(jc.probes.WeakLensing(nzs, sigma_e=sigma_e))

    # Let's the mean and covariance
    mu, C = jc.angular_cl.gaussian_cl_covariance(cosmo, ell, probes, f_sky=f_sky)

    # TODO: I'm not too sure about this, should we use cl_obs or cl_sig for S/N?
    # For now, I'm doing the same thing as the upstream package, so I'm removing
    # the noise contribution from the signal mu
    cl_noise = jc.angular_cl.noise_cl(ell, probes)
    mu = mu - cl_noise.flatten()
    return mu , C

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
