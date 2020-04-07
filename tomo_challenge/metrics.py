import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

def compute_snr_score(tomo_bin, z):
    """Compute a score metric based on the total spectrum S/N

    This is given by mu^T . C^{-1} . mu
    where mu is the theory prediction and C the Gaussian covariance
    for this set of bins.

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
    mu, C = compute_mean_covariance(tomo_bin, z)
    # S/N for correlated data, I assume, from generalizing
    # sqrt(sum(mu**2/sigma**2))
    P = np.linalg.inv(C)
    score = (mu.T @ P @ mu)**0.5

    return score

def compute_mean_covariance(tomo_bin, z):
    """Compute a mean and covariance for the chosen distribution of objects.

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
    cosmo = ccl.Cosmology(
        Omega_c = 0.22,
        Omega_b = 0.0447927,
        h = 0.71,
        n_s = 0.963,
        sigma8 = 0.8,
    )


    # choose ell bins from params  above
    ell = np.logspace(2, np.log10(ell_max), n_ell)

    # work out the number density per steradian
    steradian_to_arcmin2 = 11818102.86004228
    n_eff_total = n_eff_total_arcmin2 * steradian_to_arcmin2


    nbin = int(tomo_bin.max()) + 1

    # redshift grid we use to compute theory spectra.
    # reasonable size given the numbers we're talking about here
    dz = 0.01
    z_max = z.max() + dz / 2
    z_edge = np.arange(0, z_max, dz)
    z_mid = (z_edge[1:] + z_edge[:-1]) / 2


    # Generate CCL tracers and get total counts, for the noise
    counts = np.zeros(nbin)
    tracers = []
    for b in range(nbin):
        n_of_z, _ = np.histogram(z[tomo_bin == b], bins=z_edge)
        tracers.append(ccl.WeakLensingTracer(cosmo, dndz=(z_mid, n_of_z)))
        # total number of objects in this histogram bin
        counts[b] = n_of_z.sum()

    # Get the fraction of the total possible number of objects
    # in each bin, and the consequent effective number density.
    # We pretend here that all objects have the same weight.
    fractions = [c / tomo_bin.size for c in counts]
    n_eff = [n_eff_total * f for f in fractions]

    # Define an ordering of the theory vector
    blocks = []
    for i in range(nbin):
        for j in range(i, nbin):
            blocks.append((i,j))


    # Get all the spectra, both the signal-only version (for the mean)
    # and the version with noise (for the covmat)
    C_sig = {}
    C_obs = {}
    for i, j in blocks:
        Ti = tracers[i]
        Tj = tracers[j]
        C_sig[i, j] = ccl.angular_cl(cosmo, Ti, Tj, ell)

        # Noise contribution, if an auto-bin
        if i == j:
            C_obs[i, j] = C_sig[i, j] + sigma_e**2 / n_eff[i]
        else:
            C_obs[i, j] = C_sig[i, j]
            C_obs[j, i] = C_sig[i, j]


    # concatenate all the theory predictions as our mean
    mu = np.concatenate([C_sig[i, j] for i, j in blocks])

    # empty space for the covariance matrix
    C = np.zeros((mu.size, mu.size))

    # normalization.  This and the bit below are Takada & Jain
    # equation 14
    norm = (2*ell + 1) * np.gradient(ell) * f_sky

    # Fill in each covmat block.  We waste time here
    # doing the flip elements but not significant
    for a, (i, j) in enumerate(blocks[:]):
        for b, (m, n) in enumerate(blocks[:]):
            start_a = a * n_ell
            start_b = b * n_ell
            end_a = start_a + n_ell
            end_b = start_b + n_ell
            c2 = (C_obs[i, m] * C_obs[j, n] + C_obs[i, n] * C_obs[j, m])
            C[start_a:end_a, start_b:end_b] = (c2 / norm) * np.eye(n_ell)

    return mu, C


def plot_distributions(z, tomo_bin, filename, nominal_edges=None):
    fig = plt.figure()
    nbin = int(tomo_bin.max()) + 1
    for i in range(nbin):
        w = np.where(tomo_bin == i)
        plt.hist(z[w], bins=50)

    # Plot verticals at nominal edges, if given
    if nominal_edges is not None:
        for x in nominal_edges:
            plt.axvline(x, color='k', linestyle=':')

    plt.savefig(filename)
    plt.close()
