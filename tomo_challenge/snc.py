import numpy as np
import pyccl as ccl
from scipy.integrate import simps
import os


assign_params_default = {'p_inbin_thr': 0.5,
                         'p_outbin_thr': 0.2,
                         'use_p_inbin': False,
                         'use_p_outbin': False}


class SnCalc(object):
    edges_large = 3.

    def __init__(self, z_arr, nz_list, fsky=0.4, lmax=2000, n_ell=100,
                 s_gamma=0.26, use_clustering=False, use_3x2=False,
                 integrator='qag_quad'):#spline'):
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
            use_3x2 (bool): if `True`, SNR will be computed for 3x2pt.
            integrator (string): CCL integration method. Either 'qag_quad'
                or 'spline'.
        """
        self.integrator = integrator
        self.s_gamma = s_gamma
        self.fsky = fsky
        self.lmax = lmax
        self.n_ell = n_ell
        ell_edges = np.logspace(2, np.log10(lmax), n_ell+1)
        self.larr = 0.5*(ell_edges[1:]+ell_edges[:-1])
        self.d_ell = (ell_edges[1:]-ell_edges[:-1])
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
        self.use_3x2 = use_3x2
        if use_3x2:
            self.n_tracers = 2*self.n_samples
        else:
            self.n_tracers = self.n_samples

    def _bz_model(self, cosmo, z):
        return 1./ccl.growth_factor(cosmo, 1./(1+z))

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
        cosmo = ccl.Cosmology(Omega_c=0.25,
                              Omega_b=0.05,
                              h=0.67, n_s=0.96,
                              sigma8=0.81)

        # Tracers
        bz = self._bz_model(cosmo, self.z_arr)
        if self.use_3x2:
            trs_gc = [ccl.NumberCountsTracer(cosmo, False, (self.z_arr, nz),
                                             bias=(self.z_arr, bz))
                      for nz in self.nz_list]
            trs_wl = [ccl.WeakLensingTracer(cosmo, (self.z_arr, nz))
                      for nz in self.nz_list]
            trs = trs_gc + trs_wl
        else:
            if self.use_clustering:
                trs = [ccl.NumberCountsTracer(cosmo, False, (self.z_arr, nz),
                                              bias=(self.z_arr, bz))
                       for nz in self.nz_list]
            else:
                trs = [ccl.WeakLensingTracer(cosmo, (self.z_arr, nz))
                       for nz in self.nz_list]

        # Cls
        self.cls = np.zeros([self.n_ell, self.n_tracers, self.n_tracers])
        for i in range(self.n_tracers):
            for j in range(i, self.n_tracers):
                cl = ccl.angular_cl(cosmo, trs[i], trs[j], self.larr,
                                    limber_integration_method=self.integrator)
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
        if self.use_3x2:
            w_use = np.tile(weights, (2, 2))
        else:
            w_use = weights
        return np.sum(np.sum(w_use[None, None, :, :] *
                             self.cls[:, :, None, :],
                             axis=-1)[:, None, :, :] *
                      w_use[None, :, :, None],
                      axis=-2)

    def _get_nl_resamples(self, n_dens):
        """ Gets noise contribution to C_ell matrix for resampled groups
        from list of number densities.
        """
        if self.use_3x2:
            n_diag = np.array(list(1./n_dens) +
                              list(self.s_gamma**2/n_dens))
            return np.diag(n_diag)[None, :, :]
        else:
            if self.use_clustering:
                return np.diag(1./n_dens)[None, :, :]
            else:
                return np.diag(self.s_gamma**2/n_dens)[None, :, :]

    def get_sn_max(self, full_output=False):
        """ Compute maximum signal-to-noise ratio given the input
        groups
        Args:
            full_output (bool): if true, a dictionary with additional
                information will be returned. Otherwise just total S/N.
        Returns:
            If `full_output=True`, dictionary containing S/N, power
            spectra and noise spectra. Otherwise just S/N.
        """
        weights = np.eye(self.n_samples)
        n_dens = self.n_dens
        return self.get_sn_wn(weights, n_dens, full_output=full_output)

    def get_kltrans(self):
        """ Compute KL decomposition.

        Returns:
            Fisher matrix element for each of the KL modes.
        """
        nl = self._get_nl_resamples(self.n_dens)
        nij = nl*np.ones(self.n_ell)[:, None, None]
        cij = self.cls + nij
        sij = cij - nij
        inv_nij = np.linalg.inv(nij)
        metric = inv_nij

        def change_basis(c, m, ev):
            return np.array([np.diag(np.dot(ev[i].T,
                                            np.dot(m[i],
                                                   np.dot(c[i],
                                                          np.dot(m[i],
                                                                 ev[i])))))
                             for i in range(self.n_ell)])

        def diagonalize(c, m):
            im = np.linalg.inv(m)
            ll = np.linalg.cholesky(m)
            ill = np.linalg.cholesky(im)
            cl = np.array([np.dot(np.transpose(ll[i]),
                                  np.dot(c[i], ll[i]))
                           for i in range(self.n_ell)])
            c_p, v = np.linalg.eigh(cl)
            ev = np.array([np.dot(np.transpose(ill[i]), v[i])
                           for i in range(self.n_ell)])
            # iden = change_basis(im, m, ev)
            return ev, c_p

        e_v, c_p = diagonalize(cij, metric)
        s_p = change_basis(sij, metric, e_v)
        nmodes_l = (self.fsky * self.d_ell * (self.larr+0.5))
        fish_kl = nmodes_l[:, None]*(s_p/c_p)**2
        isort = np.argsort(-np.sum(fish_kl, axis=0))
        e_o = e_v[:, :, isort]
        # f_o = np.array([np.dot(inv_nij[l], e_o[l, :, :])
        #                 for l in range(self.n_ell)])
        c_p = change_basis(cij, metric, e_o)
        s_p = change_basis(sij, metric, e_o)

        fish_kl = nmodes_l[:, None]*(s_p/c_p)**2
        fish_permode = np.sum(fish_kl, axis=0)
        fish_cumul = np.cumsum(fish_permode)
        return fish_kl, fish_permode, fish_cumul

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
        snr = np.sqrt(np.sum(trsn2_ell * (self.larr + 0.5) *
                             self.d_ell * self.fsky))

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
        swapped = False
        if np.ndim(edges) > 0:
            swapped = np.any(np.diff(edges) < 0)
        return np.any(edges < 0) or \
            np.any(edges > self.edges_large) or \
            swapped

    def get_nzs_from_edges(self, edges, assign_params=assign_params_default):
        assign = self.assign_from_edges(edges, assign_params=assign_params)
        nzs = np.array([np.sum(self.nz_list[a, :], axis=0)
                        for a in assign])
        return nzs

    def assign_from_edges(self, edges, assign_params=assign_params_default,
                          get_ids=False):
        """ Get assignment list from edges and assign parameters.
        Args:
            edges (array_like): array of bin edges.
            assign_params (dict): dictionary of assignment parameters
                (see `assign_params_default`).
            get_ids (bool): if `True`, output assignment arrays will
                be accompanied by the associated bin id (with -1 for the
                trash bin).
        Returns:
            List of assignment arrays ready to be used in e.g. `get_sn`.
        """
        edges = np.atleast_1d(edges)
        nbins = len(edges) + 1
        # Bin IDs based on mean z
        ids = np.digitize(self.z_means, bins=edges)
        if assign_params['use_p_inbin'] or assign_params['use_p_outbin']:
            def integrate_safe(p, z, m):
                if np.sum(m) == 0:
                    return 0
                else:
                    return simps(p[m], x=z[m])
            # Calculate probabilities in each bin
            zgroups = np.digitize(self.z_arr, bins=edges)
            masks = [zgroups == i for i in range(nbins)]
            if assign_params['use_p_outbin']:
                # Matrix of probabilities in each bin
                pzs = np.array([[integrate_safe(pz, self.z_arr, m)
                                 for m in masks]
                                for pz in self.pz_list])
                pzd = np.array([pzs[j, ids[j]]
                                for j in range(self.n_samples)])
            else:
                # Overlaps in own bin
                pzd = np.array([integrate_safe(pz, self.z_arr, masks[i])
                                for pz, i in zip(self.pz_list, ids)])
            # Throw away based on in-bin probability
            if assign_params['use_p_inbin']:
                ids[pzd < assign_params['p_inbin_thr']] = -1
            # Throw away based on off-bin probability
            if assign_params['use_p_outbin']:
                ids[np.array([np.sum(p > assign_params['p_outbin_thr']) > 1
                    for p in pzs])] = -1
                # Assignment list
        if get_ids:
            return [(i, np.where(ids == i)[0]) for i in np.unique(ids)]
        else:
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
