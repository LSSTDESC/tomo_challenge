import numpy as onp

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

import tomo_challenge.jax_metrics as metrics

# Conviniently stores the number of features
n_features = {'riz':6, 'griz':10}

# Function creating the neural network for a specific number of bins
def get_classifier(n_bin, n_features):
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

# This defines the optimization step, literally maximizing the S/N
def get_optimizing_step(metric):
    if metric == 'SNR':
        score_fn = metrics.compute_snr_score
    elif metric == 'FOM':
        score_fn = metrics.compute_fom_score

    @jax.jit
    def train_step(optimizer, batch):
        # This is the loss function
        def loss_fn(model):
            # Apply classifier to features
            w = model(batch['features'])
            # returns - score, because we want to maximize score
            return - score_fn(w, batch['labels'])
        # Compute gradients
        loss, g = jax.value_and_grad(loss_fn)(optimizer.target)
        # Perform gradient descent
        optimizer = optimizer.apply_gradient(g)
        return optimizer, loss

    return train_step

class NeuralNetwork:
    """ Neural Network Classifier """

    # valid parameter -- see below
    valid_options = ['bins', 'metric']

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
        self.metric = options['metric']
        # Build a name for the model based on bands and options
        self.export_name = f'models/{self.metric}_{bands}_{self.n_bin}.flax'

        # Create classifier
        self.model = get_classifier(self.n_bin, self.n_features)
        # Create scaler
        self.features_scaler = StandardScaler()

    def train (self, training_data, training_z,
              batch_size=2000, niter=1500):
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
        features = self.features_scaler.fit_transform(training_data)
        labels = training_z

        # If model is already trained, we just load the weights
        if os.path.exists(self.export_name):
            with open(self.export_name, 'rb') as file:
                self.model = serialization.from_bytes(self.model, pickle.load(file))
            return

        # Otherwise we train
        if self.metric == 'FOM':
            lr = 0.1
        if self.metric == 'SNR':
            lr = 0.001
        optimizer = optim.Momentum(learning_rate=lr, beta=0.9).create(self.model)

        @jax.jit
        def train_step(optimizer, batch):
            # This is the loss function
            def loss_fn(model):
                # Apply classifier to features
                w = model(batch['features'])
                # returns - score, because we want to maximize score
                if self.metric == 'SNR':
                    return - metrics.compute_snr_score(w, batch['labels'])
                elif self.metric == 'FOM':
                    return - metrics.compute_fom_score(w, batch['labels'])
                else:
                  raise NotImplementedError
            # Compute gradients
            loss, g = jax.value_and_grad(loss_fn)(optimizer.target)
            # Perform gradient descent
            optimizer = optimizer.apply_gradient(g)
            return optimizer, loss
        #
        # # Create training fn for specific metric
        # train_step = get_optimizing_step(self.metric)

        # This function provides random batches of data, TODO: convert to JAX
        print("Size of dataset", len(labels))
        def get_batch():
            inds = onp.random.choice(len(labels), batch_size)
            return {'labels': labels[inds], 'features': features[inds]}

        losses = []
        for i in range(niter):
            optimizer, loss = train_step(optimizer, get_batch())
            losses.append(loss)
            if i%100 == 0:
                print('iter: %d; Loss : %f'%(i,loss))

        # Export model to disk
        with open(self.export_name, 'wb') as file:
            pickle.dump(serialization.to_bytes(optimizer.target), file)

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
