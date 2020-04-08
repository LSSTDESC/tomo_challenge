import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import sacc
import firecrown
import sacc
import pathlib
import tempfile
import yaml

# if you just put all the objects into one bin you get something
# like this.
SNR_SCORE_BASELINE = 266.5

def compute_scores(tomo_bin, z):
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

    Returns
    -------
    score1: float
        SNR metric

    score2: float
        FOM metric

    """
    mu, C = compute_mean_covariance(tomo_bin, z)
    # S/N for correlated data, I assume, from generalizing
    # sqrt(sum(mu**2/sigma**2))
    P = np.linalg.inv(C)
    score1 = (mu.T @ P @ mu)**0.5 - SNR_SCORE_BASELINE


    sacc_data = make_sacc(tomo_bin, z, nbin, mu, C)
    score2 = figure_of_merit(sacc_data)

    return score1, score2


def compute_fom_metric(tomo_bin, z):
    # make a mean and covariance
    # make a sacc data object using firecrown
    # use firecrown to run a fisher analysis on it
    pass


def get_n_of_z(tomo_bin, z):
    nbin = int(tomo_bin.max()) + 1

    # redshift grid we use to compute theory spectra.
    # reasonable size given the numbers we're talking about here
    dz = 0.01
    z_max = z.max() + dz / 2
    z_edge = np.arange(0, z_max, dz)
    z_mid = (z_edge[1:] + z_edge[:-1]) / 2


    # Generate CCL tracers and get total counts, for the noise
    counts = np.zeros(nbin)
    n_of_z = []
    for b in range(nbin):
        nz_bin, _ = np.histogram(z[tomo_bin == b], bins=z_edge)
        n_of_z.append(nz_bin)

    return z_mid, n_of_z

def ell_binning():
    # we put this here to make sure it's used consistently
    # plausible limits I guess
    ell_max = 2000
    n_ell = 100
    # choose ell bins from 10 .. 2000 log spaced
    ell = np.logspace(2, np.log10(ell_max), n_ell)
    return ell



def compute_mean_covariance(tomo_bin, z):
    """Compute a mean and covariance for the chosen distribution of objects.

    This assumes a cosmology, and the varions parameters affecting the noise
    and signal: the f_sky, ell choices, sigma_e, and n_eff
    """
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

    # ell values we will use.  Computed centrally
    # since we want to avoid mismatches elsewhere.
    # should really sort this out better.
    ell = ell_binning()
    n_ell = len(ell)


    # work out the number density per steradian
    steradian_to_arcmin2 = 11818102.86004228
    n_eff_total = n_eff_total_arcmin2 * steradian_to_arcmin2


    nbin = int(tomo_bin.max()) + 1

    z_mid, n_of_z = get_n_of_z(tomo_bin, z)
    tracers = []
    counts = [nz_bin.sum() for nz_bin in n_of_z]
    tracers = [
        ccl.WeakLensingTracer(cosmo, dndz=(z_mid, nz_bin))
        for nz_bin in n_of_z
    ]


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
        plt.hist(z[w], bins=50, histtype='step')

    # Plot verticals at nominal edges, if given
    if nominal_edges is not None:
        for x in nominal_edges:
            plt.axvline(x, color='k', linestyle=':')

    plt.savefig(filename)
    plt.close()


def make_sacc(tomo_bin, z, mu, C):
    # Basic numbers
    nbin = tomo_bin.max() + 1
    z_mid, n_of_z = get_n_of_z(tomo_bin, z)
    npair = (nbin * (nbin + 1)) // 2

    # Must be the same as above
    ell = ell_binning()
    n_ell = len(ell)
    # Just EE for now
    EE = sacc.data_types.standard_types.galaxy_shear_cl_ee

    # Start with empty sacc.
    S = sacc.Sacc()

    # Add all the tracers
    for i in range(nbin):
        S.add_tracer("NZ", f'source_{i}', z_mid, n_of_z[i])

    # Now put in all the data points
    n = 0
    for i in range(nbin):
        for j in range(i, nbin):
            S.add_ell_cl(EE, f'source_{i}', f'source_{j}', ell, mu[n:n+n_ell])
            n += n_ell

    # And finally add the covmat
    S.add_covariance(C)
    return S


def figure_of_merit(sacc_data):
    nbin = len(sacc_data.tracers)

    # Load the baseline configuration
    config = yaml.safe_load(open("./tomo_challenge/config.yml"))

    # Override pieces of the configuration.
    # Start with the tracer list.
    config['two_point']['sources'] = {
        f'source_{b}' : {
            'kind': 'WLSource',
            'sacc_tracer': f'source_{b}',
            'systematics': {}
        } for b in range(nbin)
    }

    # Override pieces of the configuration.
    # Then the statistics list (all pairs).
    config['two_point']['statistics'] = {
        f'cl_{i}_{j}': {
          'sources': [f'source_{i}', f'source_{j}'],
          'sacc_data_type': 'galaxy_shear_cl_ee'
        }
        for i in range(nbin) for j in range(i, nbin)
    }

    # The C_ell values going into the data vector (all of them)
    config['two_point']['likelihood']['data_vector'] = [
            f'cl_{i}_{j}'
            for i in range(nbin) for j in range(i, nbin)
    ]

    # and finally override the sacc_data object itself.
    config['two_point']['sacc_data'] = sacc_data

    # Now run firecrown in a temporary directory
    # and load the resulting fisher matrix
    conf, data = firecrown.parse(config)
    with tempfile.TemporaryDirectory() as tmp:
        firecrown.run_cosmosis(conf, data, pathlib.Path(tmp))
        fisher_matrix = np.loadtxt(f'{tmp}/chain.txt')

    # Pull out the correct indices.  We would like to use
    # w0 - wa but can't yet because 
    param_names = list(config['cosmosis']['parameters'].keys())
    i = param_names.index('Omega_c')
    j = param_names.index('sigma8')
    
    # Convert the Fisher matrix to a covariance
    covmat = np.linalg.inv(fisher_matrix)
    
    # Pull out the 2x2 chunk of the covariance matrix
    covmat_chunk = covmat[:, [i,j]][[i,j], :]
    
    # And get the FoM, the inverse area of the 2 sigma contour
    # area.
    area = 6.17 * np.pi * np.sqrt(np.linalg.det(covmat_chunk))
    fom = 1 / area

    return fom