import numpy as np
import pyccl as ccl
from scipy.integrate import simps
import os


assign_params_default = {'p_inbin_thr': 0.5,
                         'p_outbin_thr': 0.2,
                         'use_p_inbin': False,
                         'use_p_outbin': False}


class SnCalc(object):
    edges_large = 100.

    def __init__(self, z_arr, nz_list, fsky=0.4, lmax=2000, d_ell=10,
                 s_gamma=0.28, use_clustering=False):
        """ S/N calculator

        Args:
            z_arr (array_like): array of redshifts at which all the N(z)s are
                sampled. They should be linearly spaced.
            nz_list (list): list of arrays containing the redshift
                distributions of all the initial groups (each array in the
                list corresponds to one group). The arrays should contain the
                redshift distribution sampled at `z_arr`, with its integral
                over `z_arr` equal to the number of galaxies in that group.
            fsky (float): sky fraction (for SNR calculation and for
                transforming number of galaxies into number densities).
            lmax (float): maximum ell.
            d_ell (float): ell bandpower width.
            s_gamma (float): rms ellipticity scatter (relevant if
                `use_clustering=False`).
            use_clustering (bool): if `True`, SNR will be computed for
                clustering instead of lensing.
        """
        self.s_gamma = s_gamma
        self.fsky = fsky
        self.lmax = lmax
        self.d_ell = d_ell
        self.larr = np.arange(2, lmax, d_ell)+d_ell/2
        self.n_ell = len(self.larr)
        self.n_samples = len(nz_list)
        self.z_arr = z_arr
        self.nz_list = nz_list
        self.n_gals = np.array([simps(nz, x=self.z_arr)
                                for nz in self.nz_list])
        self.pz_list = self.nz_list / self.n_gals[:, None]
        self.z_means = np.array([simps(self.z_arr*nz, x=self.z_arr)
                                 for nz in self.nz_list]) / self.n_gals
        self.n_dens = self.n_gals / (4*np.pi*self.fsky)
        self.cls = None
        self.use_clustering = use_clustering

    def _bz_model(self, cosmo, z):
        return 0.95/ccl.growth_factor(cosmo, 1./(1+z))

    def get_cl_matrix(self, fname_save=None, recompute=False):
        """ Computes matrix of power spectra between all the initial groups.

        Args:
            fname_save (string): if not None, the result will be saved to a
                file by this name, and if present and `recompute=False`, the
                matrix will be read from that file.
            recompute (bool): if `True`, any previous calculations of the
                C_ell matrix stored on file will be ignored, and the matrix
                will be recomputed.
        """
        if fname_save is not None:
            if os.path.isfile(fname_save) and not recompute:
                self.cls = np.load(fname_save)['cls']
                return

        # Cosmology
        cosmo = ccl.CosmologyVanillaLCDM()

        # Tracers
        if self.use_clustering:
            trs = [ccl.NumberCountsTracer(cosmo, False, (self.z_arr, nz),
                                          bias=(self.z_arr,
                                                self._bz_model(cosmo,
                                                               self.z_arr)))
                   for nz in self.nz_list]
        else:
            trs = [ccl.WeakLensingTracer(cosmo, (self.z_arr, nz))
                   for nz in self.nz_list]

        # Cls
        self.cls = np.zeros([self.n_ell, self.n_samples, self.n_samples])
        for i in range(self.n_samples):
            for j in range(i, self.n_samples):
                cl = ccl.angular_cl(cosmo, trs[i], trs[j], self.larr)
                self.cls[:, i, j] = cl
                if j != i:
                    self.cls[:, j, i] = cl

        if fname_save is not None:
            np.savez(fname_save, cls=self.cls)

    def get_resample_metadata(self, assign):
        """ Transform an assignment list into a weight matrix and the
        corresponding number densities.

        Args:
            assign (list): list of arrays. Each element of the list should
                be a numpy array of integers, corresponding to the indices
                of the initial groups that are now resampled into a larger
                group.
        """
        n_resamples = len(assign)
        weight_res = np.zeros([n_resamples, self.n_samples])
        n_dens_res = np.zeros(n_resamples)
        for ia, a in enumerate(assign):
            ndens = self.n_dens[a]
            ndens_tot = np.sum(ndens)
            n_dens_res[ia] = ndens_tot
            weight_res[ia, a] = ndens / ndens_tot
        return weight_res, n_dens_res

    def _get_cl_resamples(self, weights):
        """ Gets C_ell matrix for resampled groups from weights matrix.
        """
        if self.cls is None:
            self.get_cl_matrix()
        return np.einsum('jl,km,ilm', weights, weights, self.cls)

    def _get_nl_resamples(self, n_dens):
        """ Gets noise contribution to C_ell matrix for resampled groups
        from list of number densities.
        """
        if self.use_clustering:
            return np.diag(1./n_dens)[None, :, :]
        else:
            return np.diag(self.s_gamma**2/n_dens)[None, :, :]

    def get_sn_wn(self, weights, n_dens, full_output=False):
        """ Compute signal-to-noise ratio from weights matrix and number
        densities of the resampled groups.

        Args:
            weights (array_like): weights matrix of shape
                `(N_resample, N_initial)`, where `N_resample` is the
                number of new groups, and `N_initial` is the original
                number of groups. Each entry corresponds to the weight
                which with a given initial group enters the new set of
                groups.
            n_dens (array_like): number density (in sterad^-1) of the
                new groups.
            full_output (bool): if true, a dictionary with additional
                information will be returned. Otherwise just total S/N.

        Returns:
            If `full_output=True`, dictionary containing S/N, power
            spectra and noise spectra. Otherwise just S/N.
        """
        sl = self._get_cl_resamples(weights)
        nl = self._get_nl_resamples(n_dens)
        cl = sl + nl
        icl = np.linalg.inv(cl)
        sn2_1pt = np.sum(sl[:, :, :, None] * icl[:, None, :, :], axis=2)
        sn2_ell = np.sum(sn2_1pt[:, :, :, None] * sn2_1pt[:, None, :, :],
                         axis=2)
        trsn2_ell = np.trace(sn2_ell, axis1=1, axis2=2)
        snr = np.sqrt(np.sum(trsn2_ell * (2*self.larr + 1) *
                             self.d_ell / self.fsky))

        if full_output:
            dret = {'snr': snr, 'cl': sl, 'nl': nl, 'ls': self.larr}
            return dret
        else:
            return snr

    def get_sn(self, assign, full_output=False):
        """ Compute signal-to-noise ratio from a bin assignment list.

        Args:
            assign (list): list of arrays. Each element of the list should
                be a numpy array of integers, corresponding to the indices
                of the initial groups that are now resampled into a larger
                group.
            full_output (bool): if true, a dictionary with additional
                information will be returned. Otherwise just total S/N.

        Returns:
            If `full_output=True`, dictionary containing S/N, power
            spectra and noise spectra. Otherwise just S/N.
        """
        w, ndens = self.get_resample_metadata(assign)
        return self.get_sn_wn(w, ndens,
                              full_output=full_output)

    def check_edges(self, edges):
        """ Returns `True` if there's something wrong with the edges.
        """
        return np.any(edges < 0) or \
            np.any(edges > self.edges_large) or \
            np.any(np.diff(edges) < 0)

    def assign_from_edges(self, edges, assign_params=assign_params_default):
        """ Get assignment list from edges and assign parameters.

        Args:
            edges (array_like): array of bin edges.
            assign_params (dict): dictionary of assignment parameters
                (see `assign_params_default`).

        Returns:
            List of assignment arrays ready to be used in e.g. `get_sn`.
        """
        nbins = len(edges) + 1
        # Bin IDs based on mean z
        ids = np.digitize(self.z_means, bins=edges)
        if assign_params['use_p_inbin'] or assign_params['use_p_outbin']:
            # Calculate probabilities in each bin
            zgroups = np.digitize(self.z_arr, bins=edges)
            masks = [zgroups == i for i in range(nbins)]
            if assign_params['use_p_outbin']:
                # Matrix of probabilities in each bin
                pzs = np.array([[simps(pz[m], x=self.z_arr[m])
                                 for m in masks]
                                for pz in self.pz_list])
                pzd = np.array([pzs[j, ids[j]]
                                for j in range(self.n_samples)])
            else:
                # Overlaps in own bin
                pzd = np.array([simps(pz[masks[i]],
                                      x=self.z_arr[masks[i]])
                                for pz, i in zip(self.pz_list, ids)])

            # Throw away based on in-bin probability
            if assign_params['use_p_inbin']:
                ids[pzd < assign_params['p_inbin_thr']] = -1
            # Throw away based on off-bin probability
            if assign_params['use_p_outbin']:
                ids[np.array([np.sum(p > assign_params['p_outbin_thr']) > 1
                              for p in pzs])] = -1
        # Assignment list
        return [np.where(ids == i)[0] for i in np.unique(ids)]

    def get_sn_from_edges(self, edges, full_output=False,
                          assign_params=assign_params_default):
        """ Compute signal-to-noise ratio from a set of edges and assignment
        parameters.

        Args:
            edges (array_like): array of bin edges.
            assign_params (dict): dictionary of assignment parameters
                (see `assign_params_default`).
            full_output (bool): if true, a dictionary with additional
                information will be returned. Otherwise just total S/N.

        Returns:
            If `full_output=True`, dictionary containing S/N, power
            spectra and noise spectra. Otherwise just S/N.
        """
        if self.check_edges(edges):
            return 0
        assign = self.assign_from_edges(edges, assign_params=assign_params)
        return self.get_sn(assign, full_output=full_output)
