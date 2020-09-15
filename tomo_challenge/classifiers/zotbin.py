from pathlib import Path

from .base import Tomographer
import numpy as np

try:
    import jax.experimental.optimizers
except ImportError:
    print('The ZotBin classifier needs the jax and jax-cosmo packages.')

try:
    #from zotbin.group import groupbins, load_groups, save_groups, fdigitize
    #from zotbin.binned import get_zedges_chi
    from zotbin.util import prepare, get_signature, get_file
    from zotbin.flow import learn_flow
    from zotbin.binned import load_binned
    from zotbin.group import groupbins, load_groups, fdigitize, assign_bins
    from zotbin.optimize import optimize
except ImportError:
    print('The ZotBin classifierr needs the zotbin package:\n  pip install git+https://github.com/dkirkby/zotbin.git')


class ZotBin(Tomographer):
    """ ZotBin method.
    """

    # valid parameter -- see below
    valid_options = [
        'bins', 'init', 'npct', 'ngrp', 'similarity',
        'metric', 'ntrial', 'nsteps', 'interval', 'eta', 'seed']
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
        self.preprocessor = None
        self.fedges = None
        similarity = options['similarity']
        if similarity not in ('cosine', 'weighted', 'EMD'):
            raise ValueError(f'Invalid similarity: "{similarity}".')
        metric = options['metric']
        if metric not in ('SNR_3x2', 'FOM_3x2', 'FOM_DETF_3x2'):
            raise ValueError(f'Invalid optimization metric: "{metric}".')
        self.init_data = load_binned(get_file(options['init']))

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
        print(f'train: input data shape is {data.shape}.')
        # Prepare input features.
        features, detected = prepare(data, self.bands)
        features = features[detected]
        z = z[detected]
        # Use cached preprocessed data if available.
        signature = get_signature(features)
        pname = Path('preprocessed_{0}.npy'.format(signature))
        if pname.exists():
            print('Using cached preprocessed data.')
            U = np.load(pname)
        else:
            # Learn a preprocessing transform to an approximately uniform distribution of features.
            print('Learning preprocessor normalizing flow...')
            self.preprocessor = learn_flow(features[:400000])
            # Proprocess the input features.
            U = self.preprocessor(features)
            # Cache the preprocessed data for next time.
            np.save(pname, U)
            print('Cached preprocessed data.')
        # Load or calculate groups in feature space.
        method = self.opt['similarity']
        npct = self.opt['npct']
        ngrp = self.opt['ngrp']
        fname = f'groups_{method}_{npct}_{ngrp}_{signature}.npz'
        if not Path(fname).exists():
            print(f'Calculating {ngrp} feature space groups with npct={npct}...')
            groupbins(U, z, self.init_data[0], npct, ngrp_save=[ngrp], method=method,
                      plot_interval=None, savename=fname)
        _, self.fedges, self.grpid, self.zhist, _ = load_groups(fname)
        print(f'Loaded {ngrp} groups with npct={npct}.')
        # Optimize the weights for combining groups into the requested number of nbins
        # for the specified metric.
        args = {k: self.opt[k] for k in ('metric', 'ntrial', 'interval', 'seed')}
        args['nbin'] = self.opt['bins']
        args['opt_args'] = dict(
            optimizer=jax.experimental.optimizers.adam(self.opt['eta']),
            nsteps=self.opt['nsteps'])
        print(f'Optimizing final bins with {args}...')
        best_scores, self.weights, self.dndz_bin, _ = optimize(
            mixing_matrix=self.zhist, init_data=self.init_data, **args)
        print(f'Best scores after optimization: {best_scores}')

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
        print(f'apply: input data shape is {data.shape}.')
        features, detected = prepare(data, self.bands)
        tomo_sel = np.full(len(features), -1, int)
        # Use cached preprocessed data if available.
        features = features[detected]
        signature = get_signature(features)
        pname = Path('preprocessed_{0}.npy'.format(signature))
        if pname.exists():
            print('Using cached preprocessed data.')
            U = np.load(pname)
        elif self.preprocessor is None:
            raise RuntimeError('No preprocessor defined: has the train step been run?')
        else:
            # Apply the learned transform.
            print('Preprocessing...')
            U = self.preprocessor(features)
            # Cache the preprocessed data for next time.
            np.save(pname, U)
            print('Cached preprocessed data.')
        # Assign galaxies to feature groups.
        if self.fedges is None:
            raise RuntimeError('No groups defined: has the train step been run?')
        feature_bin = fdigitize(U, self.fedges)
        feature_grp = self.grpid[feature_bin]
        nempty = np.count_nonzero(feature_grp == -1)
        print(f'Found {nempty} galaxies outside the training feature space.')
        # Assign feature groups to output bins.
        tomo_sel[detected] = assign_bins(feature_grp, self.weights, self.opt['seed'])
        # Save results before returning.
        nbin, ngrp = self.weights.shape
        metric = self.opt['metric']
        method = self.opt['similarity']
        npct = self.opt['npct']
        fname = f'zotbin_{metric}_{nbin}_{method}_{npct}_{ngrp}_{signature}.npz'
        np.savez(fname, idx=tomo_sel.astype(np.uint8), weights=self.weights, dndz=self.dndz_bin)
        print(f'Saved {fname}')

        return tomo_sel
