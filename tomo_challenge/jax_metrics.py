import jax.numpy as np
import jax.random as rand
from jax import lax, jit, vmap, grad

import jax_cosmo as jc

SNR_SCORE_BASELINE = 266.5

@jit
def compute_mean_covariance(weights, labels, what, kernel_bandwidth=0.01):
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
    cosmo = jc.Cosmology(
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
    tracer_type = get_tracer_type(nbin, what)

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
