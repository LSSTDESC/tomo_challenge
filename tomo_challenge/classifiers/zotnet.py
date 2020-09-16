from pathlib import Path

from .base import Tomographer
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    print('The ZotNet classifier needs the jax and jax-cosmo packages.')

try:
    #from zotbin.group import groupbins, load_groups, save_groups, fdigitize
    #from zotbin.binned import get_zedges_chi
    from zotbin.util import prepare, get_signature, get_file
    from zotbin.flow import learn_flow
    from zotbin.binned import load_binned
    from zotbin.nnet import learn_nnet
except ImportError:
    print('The ZotNet classifierr needs the zotbin package:\n  pip install git+https://github.com/dkirkby/zotbin.git')


class ZotNet(Tomographer):
    """ ZotNet method.
    """

    # valid parameter -- see below
    valid_options = [
        'bins', 'init', 'ndata', 'nhidden', 'nlayer', 'trainfrac', 'batchsize',
        'metric', 'ntrial', 'nepoch', 'interval', 'eta', 'seed']
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
        # Train a neural network using the specified metric as the -loss function.
        args = {k: self.opt[k] for k in (
            'nhidden', 'nlayer', 'trainfrac', 'batchsize',
            'metric', 'ntrial', 'nepoch', 'interval', 'eta', 'seed')}
        print(f'Learning neural network with {args}...')
        ndata = self.opt['ndata']
        X = jnp.array(3 * (U[:ndata] - 0.5))
        z = jnp.array(z[:ndata])
        best_scores, self.weights, self.dndz_bin, _, self.apply_nnet = learn_nnet(
            self.opt['bins'], X, z, init_data=self.init_data, **args)
        print(f'Best scores after training: {best_scores}')

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
        # Apply the learned network to calculate sample weights.
        X = jnp.array(3 * (U - 0.5))
        weights = self.apply_nnet(X)
        # Randomly assign each sample to an output bin.
        print('Assigning output bins...')
        cdf = np.cumsum(weights, axis=1)
        gen = np.random.RandomState(self.opt['seed'])
        u = gen.uniform(size=len(X))
        idx = np.empty(len(X), int)
        for i in range(len(X)):
            idx[i] = np.searchsorted(cdf[i], u[i])
        tomo_sel[detected] = idx
        # Save results before returning.
        nbin = self.opt['bins']
        metric = self.opt['metric']
        fname = f'zotnet_{metric}_{nbin}_{signature}.npz'
        np.savez(fname, idx=tomo_sel.astype(np.uint8))
        print(f'Saved {fname}')

        return tomo_sel
