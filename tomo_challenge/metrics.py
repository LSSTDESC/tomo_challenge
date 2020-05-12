import numpy as np
import pyccl as ccl
import sacc
import firecrown
import sacc
import pathlib
import tempfile
import yaml

this_directory = pathlib.Path(__file__).resolve().parent
default_config_path = this_directory.parent / 'tomo_challenge' / 'config.yml'


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

    scores = {}
    if metrics == 'all':
        metrics = ["SNR_ww", "SNR_gg", "SNR_3x2", "FOM_ww", "FOM_gg", "FOM_3x2"]
    for what in ["ww", "gg", "3x2"]:
        if ("SNR_"+what in metrics) or ("FOM_"+what in metrics):
            mu, C, galaxy_galaxy_tracer_bias = compute_mean_covariance(
                tomo_bin, z, what)
            # S/N for correlated data, I assume, from generalizing
            # sqrt(sum(mu**2/sigma**2))
            P = np.linalg.inv(C)
            scores['SNR_'+what] = (mu.T @ P @ mu)**0.5
            if "FOM_"+what in metrics:
                sacc_data = make_sacc(tomo_bin, z, what, mu, C)
                scores['FOM_'+what] = figure_of_merit(
                    sacc_data, what, galaxy_galaxy_tracer_bias)

    return scores



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
    ell_edges  = np.logspace(2, np.log10(ell_max), n_ell+1)
    ell = 0.5*(ell_edges[1:]+ell_edges[:-1])
    delta_ell =(ell_edges[1:]-ell_edges[:-1])
    return ell, delta_ell


def get_tracer_type(nbin, what):
    """ returns a string of what tracers do we have with g for galaxy and w for weak lensing.
    Utility func -- see below """

    tracer_type = ""
    if (what == 'gg' or what == '3x2'):
        tracer_type += "g"*nbin
    if (what == 'ww' or what == '3x2'):
        tracer_type += "w"*nbin
    return tracer_type


def compute_mean_covariance(tomo_bin, z, what):
    """Compute a mean and covariance for the chosen distribution of objects.

    value can be 'ww' for shear-shear only, 'gg' for galaxy clustering and
    '3x2' for full '3x2pt'
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
    config = yaml.safe_load(open(default_config_path))
    cosmo = ccl.Cosmology(**config['parameters'])

    # ell values we will use.  Computed centrally
    # since we want to avoid mismatches elsewhere.
    # should really sort this out better.
    ell, delta_ell = ell_binning()
    n_ell = len(ell)

    # work out the number density per steradian
    steradian_to_arcmin2 =  (180*60/np.pi)**2
    n_eff_total = n_eff_total_arcmin2 * steradian_to_arcmin2

    nbin = int(tomo_bin.max()) + 1

    z_mid, n_of_z = get_n_of_z(tomo_bin, z)
    bz = 1/ccl.growth_factor(cosmo, 1/(1+z_mid))
    bofz = (z_mid, bz)
    galaxy_galaxy_tracer_bias = [
        (bz*nz_bin).sum()/(nz_bin).sum() for nz_bin in n_of_z]
    #print ('biases=',galaxy_galaxy_tracer_bias)

    tracers = []
    counts = [nz_bin.sum() for nz_bin in n_of_z]
    tracers = []
    tracer_type = get_tracer_type(nbin, what)
    # start with number counts
    if (what == 'gg' or what == '3x2'):
        tracers += [ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_mid, nz_bin), bias=bofz)
                    for nz_bin in n_of_z]
    if (what == 'ww' or what == '3x2'):
        tracers += [ccl.WeakLensingTracer(cosmo, dndz=(z_mid, nz_bin))
                    for nz_bin in n_of_z]

    ntracers = len(tracers)
    #print (tracer_type)
    # Get the fraction of the total possible number of objects
    # in each bin, and the consequent effective number density.
    # We pretend here that all objects have the same weight.
    fractions = [c / tomo_bin.size for c in counts]
    n_eff = [n_eff_total * f for f in fractions]

    # Define an ordering of the theory vector
    blocks = []
    for i in range(ntracers):
        for j in range(i, ntracers):
            blocks.append((i, j))

    # Get all the spectra, both the signal-only version (for the mean)
    # and the version with noise (for the covmat)
    C_sig = {}
    C_obs = {}
    for ci, cj in blocks:
        # ci, cj are tracer numbers as in (g1, g2, g3, g4, s1, s2,s3,s4 etc.)
        # nbin is 4 in the example above
        # so ci %n bin gives the tomographic bin
        i = ci % nbin
        j = cj % nbin
        Ti = tracers[ci]
        Tj = tracers[cj]
        C_sig[ci, cj] = ccl.angular_cl(cosmo, Ti, Tj, ell)
        # Noise contribution, if an auto-bin
        if ci == cj:
            if tracer_type[ci] == 'g':
                C_obs[ci, cj] = C_sig[ci, cj] + 1/n_eff[i]
            elif tracer_type[ci] == 'w':
                C_obs[ci, cj] = C_sig[ci, cj] + sigma_e**2 / n_eff[i]
            else:
                raise NotImplementedError("Unknown tracer")
        else:
            C_obs[ci, cj] = C_sig[ci, cj]
            C_obs[cj, ci] = C_sig[ci, cj]

    # concatenate all the theory predictions as our mean
    mu = np.concatenate([C_sig[i, j] for i, j in blocks])

    # empty space for the covariance matrix
    C = np.zeros((mu.size, mu.size))

    # normalization.  This and the bit below are Takada & Jain
    # equation 14
    norm = (2*ell + 1) * delta_ell * f_sky

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

    return mu, C, galaxy_galaxy_tracer_bias


def plot_distributions(z, tomo_bin, filename, nominal_edges=None):
    import matplotlib.pyplot as plt
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


def make_sacc(tomo_bin, z, what, mu, C):
    # Basic numbers
    nbin = int(tomo_bin.max()) + 1
    tracer_type = get_tracer_type(nbin, what)
    ntot = nbin*2 if what == '3x2' else nbin

    z_mid, n_of_z = get_n_of_z(tomo_bin, z)
    npair = (ntot * (ntot + 1)) // 2

    # Must be the same as above
    ell, _ = ell_binning()
    n_ell = len(ell)
    # Just EE for now
    EE = sacc.data_types.standard_types.galaxy_shear_cl_ee
    GG = sacc.data_types.standard_types.galaxy_density_cl
    GE = sacc.data_types.standard_types.galaxy_shearDensity_cl_e
    # Start with empty sacc.
    S = sacc.Sacc()

    # Add all the tracers
    for ci in range(ntot):
        i = ci % nbin
        if tracer_type[ci] == 'g':
            S.add_tracer("NZ", f'lens_{i}', z_mid, n_of_z[i])
        elif tracer_type[ci] == 'w':
            S.add_tracer("NZ", f'source_{i}', z_mid, n_of_z[i])
        else:
            raise NotImplemented

    # Now put in all the data points
    n = 0
    for ic in range(ntot):
        i = ic % nbin
        name_i = f'lens_{i}' if tracer_type[ic] == 'g' else f'source_{i}'
        for jc in range(ic, ntot):
            j = jc % nbin
            name_j = f'lens_{j}' if tracer_type[jc] == 'g' else f'source_{j}'
            if tracer_type[ic]+tracer_type[jc] == 'gg':
                S.add_ell_cl(GG, name_i, name_j, ell, mu[n:n+n_ell])
            elif tracer_type[ic]+tracer_type[jc] == 'ww':
                S.add_ell_cl(EE, name_i, name_j, ell, mu[n:n+n_ell])
            else:
                S.add_ell_cl(GE, name_i, name_j, ell, mu[n:n+n_ell])
            n += n_ell

    # And finally add the covmat
    S.add_covariance(C)
    return S


def figure_of_merit(sacc_data, what, galaxy_tracer_bias):
    ntot = len(sacc_data.tracers)
    nbin = ntot//2 if what == "3x2" else ntot
    tracer_type = get_tracer_type(nbin, what)

    # Load the baseline configuration
    config = yaml.safe_load(open(default_config_path))

    # Override pieces of the configuration.
    # Start with the tracer list.
    sourcename = []
    if what == "gg" or what == "3x2":
        for b in range(nbin):
            config['parameters'][f'bias_{b}'] = galaxy_tracer_bias[b]
            config['two_point']['sources'][f'lens_{b}'] = {
                'kind': 'NumberCountsSource',
                'sacc_tracer': f'lens_{b}',
                'bias': f'bias_{b}',
                'systematics': {}
            }
            sourcename.append(f"lens_{b}")
    if what == "ww" or what == "3x2":
        for b in range(nbin):
            config['two_point']['sources'][f'source_{b}'] = {
                'kind': 'WLSource',
                'sacc_tracer': f'source_{b}',
                'systematics': {}
            }
            sourcename.append(f"source_{b}")

    def corrtype(i, j, tracer_type):
        tt = tracer_type[i]+tracer_type[j]
        if tt == 'gg':
            return "galaxy_density_cl"
        elif tt == 'ww':
            return "galaxy_shear_cl_ee"
        else:
            return "galaxy_shearDensity_cl_e"

    # Override pieces of the configuration.
    # Then the statistics list (all pairs).
    config['two_point']['statistics'] = {
        f'cl_{i}_{j}': {
            'sources': [sourcename[i], sourcename[j]],
            'sacc_data_type': corrtype(i, j, tracer_type)
        }
        for i in range(ntot) for j in range(i, ntot)
    }

    # The C_ell values going into the data vector (all of them)
    config['two_point']['likelihood']['data_vector'] = [
        f'cl_{i}_{j}'
        for i in range(ntot) for j in range(i, ntot)
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
    covmat_chunk = covmat[:, [i, j]][[i, j], :]

    # And get the FoM, the inverse area of the 2 sigma contour
    # area.
    area = 6.17 * np.pi * np.sqrt(np.linalg.det(covmat_chunk))
    fom = 1 / area

    return fom
