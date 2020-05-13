import numpy as np

import os.path
import pickle

# Some JAX imports
import jax
import jax.numpy as np
import jax.random as rand

# Import JAX-based Neural Network library
from flax import nn, optim, serialization

# And some good old sklearn
from sklearn.preprocessing import StandardScaler

# Conviniently stores the number of features
n_features = {'riz':6, 'griz':10}

class NeuralNetwork:
    """ Neural Network Classifier """

    # valid parameter -- see below
    valid_options = ['bins']

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

        self.n_features = n_features[bands]
        self.n_bin = options['bins']

        # Create classifier
        self.model = self.get_classifier()
        # Create scaler
        self.features_scaler = StandardScaler()

    def get_classifier(self):
        """
        Function creating the neural network for a specific number of bins
        """
        n_bin = self.n_bin
        n_features = self.n_features
        # Let's create a cute little neural network for classification
        class BinningNN(nn.Module):
            def apply(self, x):
                net = nn.leaky_relu(nn.Dense(x,   500, name='fc1'))
                net = nn.leaky_relu(nn.Dense(net, 500, name='fc2'))
                return nn.softmax(nn.Dense(net, n_bin))
        # Initializing neural network weights for this configuration
        _, initial_params = BinningNN.init_by_shape( rand.PRNGKey(0),
                                                    [((1, n_features), np.float32)])
        # This instantiates the model, now ready to use
        return nn.Model(BinningNN, initial_params)

    def train (self, training_data, training_z):
        """Trains the classifier

        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """
        # create scaler
        self.features_scaler.fit(training_data)

        # For now, I'm just grabbing the trained 2b neural network
        with open('models/BinningNN_3x2_2b_FoM.flax', 'rb') as file:
            self.model = serialization.from_bytes(self.model, pickle.load(file))

    def apply (self, data):
        """Applies training to the data.

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
        features = np.array(self.features_scaler.transform(data))

        # Retrieves the classification data as bin probabilities, by batch
        bs = 10000
        s = len(features)
        weights = np.concatenate([self.model(features[bs*i:min((bs*(i+1)), s)]) for i
                                  in range(s//bs + 1)])

        # let's just check we didn't forget anyone
        assert len(weights) == s

        # Retrieve most likely bin for each galaxy
        tomo_bin = weights.argmax(axis=-1)
        return tomo_bin
