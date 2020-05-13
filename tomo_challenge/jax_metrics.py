import jax.numpy as np
import jax.random as rand
from jax import lax, jit, vmap, grad
from functools import partial
import jax_cosmo as jc
import jax

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
def get_probes(weights, labels, kernel_bandwidth=0.05):
    """
    JAX function that builds the 3x2pt probes, which can
    then be used within any metri
    """
    what = '3x2'
    # pretend there is no evolution in measurement error.
    # because LSST is awesome
    sigma_e = 0.26

    # assumed total over all bins, divided proportionally
    n_eff_total_arcmin2 = 20.0

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
        bias = jc.bias.inverse_growth_linear_bias(1.)
        probes.append(jc.probes.NumberCounts(nzs, bias))

    if (what == 'ww' or what == '3x2'):
        probes.append(jc.probes.WeakLensing(nzs, sigma_e=sigma_e))

    return probes

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
    # Retrieve the probes
    probes = get_probes(weights, labels)
    # instantiates fiducial cosmology
    cosmo = jc.Cosmology(
        Omega_c = 0.27,
        Omega_b = 0.045,
        h = 0.67,
        n_s = 0.96,
        sigma8 = 0.8404844953840714,
        Omega_k=0.,
        w0=-1., wa=0.)

    # choose ell bins from params  above
    ell, delta_ell = ell_binning()

    # Compute mean and covariance
    mu, C = jc.angular_cl.gaussian_cl_covariance(cosmo, ell, probes, f_sky=0.25)

    # S/N for correlated data, I assume, from generalizing
    # sqrt(sum(mu**2/sigma**2))
    P = np.linalg.inv(C)
    score = (mu.T @ P @ mu)**0.5
    return score

def compute_fom_score(weights, labels, inds=[0,4]):
    """
    Computes the omega_c, sigma8 Figure of Merit
    Actually the score returned is - area, I think it's more stable
    """
    # Retrieve the probes
    probes = get_probes(weights, labels)

    ell, delta_ell = ell_binning()

    # Compute the derivatives of the data vector
    @jax.jit
    def mean(params):
        cosmo = jc.Cosmology(
            Omega_c = params[0],
            Omega_b = params[1],
            h = params[2],
            n_s =  params[3],
            sigma8 = params[4],
            Omega_k=0.,
            w0=-1., wa=0.
        )
        return jc.angular_cl.angular_cl(cosmo, ell, probes)

    # Compute the jacobian of the data vector at fiducial cosmology
    fid_params = np.array([0.27, 0.045, 0.67, 0.96, 0.840484495])
    jac_mean = jax.jacfwd(lambda x: mean(x).flatten())

    mu = mean(fid_params)
    dmu = jac_mean(fid_params)

    # Compute the covariance matrix
    cl_noise = jc.angular_cl.noise_cl(ell, probes)
    C = jc.angular_cl.gaussian_cl_covariance2(ell, probes, mu, cl_noise)

    invCov = np.linalg.inv(C)

    # Compute both terms of the Fisher matrix and combine them
    #t1 = 0.5*np.einsum('nqa,ql,lmb,mn', dc, invCov, dc, invCov)
    t2 = np.einsum('pa,pq,qb->ab', dmu, invCov, dmu)
    F = t2 #t1+t2

    # Compute covariance
    i,j = inds
    covmat_chunk = np.linalg.inv(F)[:, [i, j]][[i, j], :]

    # And get the FoM, the inverse area of the 2 sigma contour
    # area.
    area = 6.17 * np.pi * np.sqrt(np.linalg.det(covmat_chunk))
    # Actually, maybe we should just return the AREA, which a quantity we want to optimize
    return - area #np.linalg.det(covmat_chunk)/9.277426e-09
