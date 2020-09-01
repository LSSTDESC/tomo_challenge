from .base import Tomographer
import numpy as np

try:
    from zotbin.group import groupbins
    from zotbin.binned import get_zedges_chi
except ImportError:
    print('You need to install the zotbin package:\n  pip install git+https://github.com/dkirkby/zotbin.git')


class ZotBin(Tomographer):
    """ ZotBin method.
    """

    # valid parameter -- see below
    valid_options = ['bins']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = True

    def __init__ (self, bands, options):
        """Constructor

        Parameters:
        -----------
        bands: str
          string containg valid bands, like 'riz' or 'griz'
        options: dict
          options come through here. Valid keys are listed as valid_options
          class variable.

        Note:
        -----
        Valiad options are:
            'bins' - number of tomographic bins

        """
        self.bands = bands
        self.opt = options
        if self.opt['loadgroups'] is not None:
            self.zedges, self.fedges, self.grpid, _, _ = load_groups(self.opt['loadgroups'])

    def prepare(self, data, band='i'):
        # Use colors and i-band magnitude as the training features.
        colors = np.diff(data, axis=1)
        i = self.bands.index(band)
        return np.concatenate((colors, data[:, i:i + 1]), axis=1)

    def train (self, data, z):
        """Trains the classifier

        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """
        npct = self.opt['npct']
        nzbin = self.opt['nzbin']
        ngrp = self.opt['ngrp']
        weighted = self.opt['weighted']
        # Prepare the training data.
        X = self.prepare(data)
        # Calculate redshift slices that are equally spaced in comoving distance.
        self.zedges = get_zedges_chi(z, nzbin)
        # Calculate feature-space groups.
        self.fedges, self.grpid, zhist, zsim = groupbins(
            X, z, self.zedges, npct, min_groups=ngrp, weighted=weighted)
        if self.opt['savegroups'] is not None:
            save_groups(self.opt['groupfile'], self.zedges, self.fedges, self.grpid, zhist, zsim)

    def apply (self, data):
        """Applies training to the data.

        Note that a bin number of -1 indicates that a galaxy should not be used.

        Parameters:
        -----------
        Data: numpy array, size Ngalaxes x Nbands
          testing data, each row is a galaxy, each column is a band as per
          band defined above

        Returns:
        tomographic_selections: numpy array, int, size Ngalaxies
          tomographic selection for galaxies return as bin number for
          each galaxy.
        """
        ngrp = self.opt['ngrp']
        X = self.prepare(data)
        sample_bin = fdigitize(X, self.fedges)
        tomo_sel = np.full(sample_bin, -1)
        for igrp in range(ngrp):
            grp_bins = np.where(self.grpid == igrp)[0]
            grp_samples = np.isin(sample_bin, grp_bins)
            tomo_sel[grp_samples] = igrp
        return tomo_sel
