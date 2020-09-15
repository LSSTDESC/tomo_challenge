import warnings
import numpy as onp
from functools import partial

try:
    import jax.numpy as np
    from jax import lax, jit, vmap, grad
    import jax_cosmo as jc
    import jax
except:
    print ("Warning: Couldn't import JAX or jax-cosmo, some metrics may be unavailable.")

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

def get_probes(weights, labels, kernel_bandwidth=0.02, what='3x2', binned_nz=False):
    """
    JAX function that builds the 3x2pt probes, which can
    then be used within any metric
    """
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
        if binned_nz:
            # In this case, we use a histogram instead of a KDE, to make things a lot faster
            h, he = onp.histogram(labels, bins=512, range=[0,4], weights=weights[:,i], density=True)
            he = 0.5*(he[1:]+he[:-1])
            nz = jc.redshift.kde_nz(he, h, bw=4./512, gals_per_arcmin2=n_eff[i], zmax=4.)
        else:
            nz = jc.redshift.kde_nz(labels, weights[:,i], bw=kernel_bandwidth,
                                      gals_per_arcmin2=n_eff[i], zmax=4.)
        nzs.append(nz)
    probes = []
    # start with number counts
    if (what == 'gg' or what == '3x2'):
        # Define a bias parameterization
        bias = jc.bias.inverse_growth_linear_bias(1.)
        probes.append(jc.probes.NumberCounts(nzs, bias))

    if (what == 'ww' or what == '3x2'):
        probes.append(jc.probes.WeakLensing(nzs, sigma_e=sigma_e))

    return probes

def compute_snr_score(weights, labels, what='3x2', binned_nz=False):
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
    probes = get_probes(weights, labels, what=what, binned_nz=binned_nz)
    ell, delta_ell = ell_binning()

    @jax.jit
    def snr_fn(probes, ell):
        # instantiates fiducial cosmology
        cosmo = jc.Cosmology(
            Omega_c = 0.27,
            Omega_b = 0.045,
            h = 0.67,
            n_s = 0.96,
            sigma8 = 0.8404844953840714,
            Omega_k=0.,
            w0=-1., wa=0.)

        # Compute mean and covariance
        mu, C = jc.angular_cl.gaussian_cl_covariance_and_mean(cosmo, ell, probes,
                                                              f_sky=0.25,
                                                              nonlinear_fn=jc.power.halofit,
                                                              sparse=True)

        # S/N for correlated data, I assume, from generalizing
        # sqrt(sum(mu**2/sigma**2))
        P = jc.sparse.inv(C)
        score = (mu.T @ jc.sparse.sparse_dot_vec(P, mu))**0.5
        return score

    return snr_fn(probes, ell)

def compute_fom(weights, labels, inds=[0,4], what='3x2', binned_nz=False):
    """
    Computes the omega_c, sigma8 Figure of Merit
    Actually the score returned is - area, I think it's more stable
    """
    # Retrieve the probes
    probes = get_probes(weights, labels, what=what, binned_nz=binned_nz)
    ell, delta_ell = ell_binning()

    @jax.jit
    def fisher_fn(probes, ell):
        # Compute the derivatives of the data vector
        def mean(params):
            cosmo = jc.Cosmology(
                Omega_c = params[0],
                Omega_b = params[1],
                h = params[2],
                n_s =  params[3],
                sigma8 = params[4],
                Omega_k=0.,
                w0=params[5], wa=params[6]
            )
            return jc.angular_cl.angular_cl(cosmo, ell, probes, nonlinear_fn=jc.power.halofit)

        # Compute the jacobian of the data vector at fiducial cosmology
        fid_params = np.array([0.27, 0.045, 0.67, 0.96, 0.840484495, -1.0, 0.0])
        jac_mean = jax.jacfwd(lambda x: mean(x).flatten())

        mu = mean(fid_params)
        dmu = jac_mean(fid_params)

        # Compute the covariance matrix
        cl_noise = jc.angular_cl.noise_cl(ell, probes)
        C = jc.angular_cl.gaussian_cl_covariance(ell, probes, mu, cl_noise, sparse=True)

        invCov = jc.sparse.inv(C)

        # Compute Fisher matrix for constant covariance
        F = jc.sparse.dot(dmu.T, invCov, dmu)
        return F

    F = fisher_fn(probes, ell)
    # Compute covariance
    i,j = inds
    covmat_chunk = np.linalg.inv(F)[:, [i, j]][[i, j], :]

    # And get the FoM, the inverse area of the 2 sigma contour
    # area.
    area = 6.17 * np.pi * np.sqrt(np.linalg.det(covmat_chunk))

    return 1. / area

def compute_scores(tomo_bin, z, metrics='all'):
    """Compute a set of score metrics.

    Metric 1
    ========
    Score metric based on the total spectrum S/N

    This is given by sqrt(mu^T . C^{-1} . mu) - baseline
    where mu is the theory prediction and C the Gaussian covariance
    for this set of bins. The baseline is the score for no tomographic binning.

    Metric 2
    ========
    WL FoM in (currently) Omega_c - sigma_8.

    Generated using a Fisher matrix calculation

    Parameters
    ----------
    tomo_bin: array
        Tomographic bin choice (0 .. bin_max) for each object in the survey
    z: array
        True redshift for each object

    metrics: str or list of str
        Which metrics to compute. If all it will return all metrics,
        otherwise just those required (see below)

    Returns
    -------
    scores: dict
         A dictionary of scores. The following dict keys are present

        "SNR_ww", "SNR_gg", "SNR_3x2": float
        SNR scores for shear-shear, galaxy clustering and full 3x2pt

        "FOM_ww", "FOM_gg", "FOM_3x2": float
        FOM metric derived from SNR above

    """
    tomo_bin = jax.nn.one_hot(tomo_bin, tomo_bin.max() + 1)
    scores = {}
    if metrics == 'all':
        metrics = ["SNR_ww", "SNR_gg", "SNR_3x2",
                   "FOM_ww", "FOM_gg", "FOM_3x2",
                   "FOM_DETF_ww", "FOM_DETF_gg", "FOM_DETF_3x2"]
    for what in ["ww", "gg", "3x2"]:
        if ("SNR_"+what in metrics) or ("FOM_"+what in metrics):
            scores['SNR_'+what] = float(compute_snr_score(tomo_bin, z, what=what, binned_nz=True))
            if "FOM_"+what in metrics:
                scores['FOM_'+what] = float(compute_fom(tomo_bin, z, what=what, binned_nz=True))
            if "FOM_DETF_"+what in metrics:
                scores['FOM_DETF_'+what] = float(compute_fom(tomo_bin, z, inds=[5,6],
                                                        what=what, binned_nz=True))
    return scores
