"""
This is an example tomographic bin generator
using a random forest.

Feel free to use any part of it in your own efforts.
Thanks Joe! Will do!
"""
import time
import sys
import os.path
import pickle

import numpy as onp

# Some JAX imports
import jax
import jax.numpy as np
import jax.random as rand

# Import JAX-based Neural Network library
from flax import nn, optim, serialization

# And some good old sklearn
from sklearn.preprocessing import StandardScaler

# Import modified version of challenge metrics
from . import jax_metrics as metrics
from . import metrics as original_metrics
from .data import load_magnitudes_and_colors, load_redshift

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
@jax.jit
def train_step(optimizer, batch):
    # This is the loss function
    def loss_fn(model):
        # Apply classifier to features
        w = model(batch['features'])
        # returns - S/N
        return -metrics.compute_snr_score(w, batch['labels'])
    # Compute gradients
    loss, g = jax.value_and_grad(loss_fn)(optimizer.target)
    # Perform gradient descent
    optimizer = optimizer.apply_gradient(g)
    return optimizer, loss

def train_neural_network(filename, bands, n_bin, export_name,
                         batch_size=10000, niter=1500):
    """
    Trains a neural network by back-propagation through the metric.
    """
    features_scaler = StandardScaler()

    # Loading the data
    features = np.array(features_scaler.fit_transform(load_magnitudes_and_colors(filename, bands)))
    labels = np.array(load_redshift(filename))

    model = get_classifier(n_bin, n_features[bands])

    # Here is the optimizer
    optimizer = optim.Momentum(learning_rate=0.001, beta=0.9).create(model)

    # This function provides random batches of data
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
    with open(export_name, 'wb') as file:
        pickle.dump(serialization.to_bytes(optimizer.target), file)

    # Returns the trained model
    return model

def apply_neural_network(classifier, filename, bands):
    features_scaler = StandardScaler()
    features = np.array(features_scaler.fit_transform(load_magnitudes_and_colors(filename, bands)))

    # Retrieves the classification data as bin probabilities, by batch
    bs = 10000
    s = len(features)
    weights = np.concatenate([classifier(features[bs*i:min((bs*(i+1)), s)]) for i
                              in range(s//bs + 1)])

    # let's just check we didn't forget anyone
    assert len(weights) == s

    # Retrieve most likely bin for each galaxy
    tomo_bin = weights.argmax(axis=-1)
    return tomo_bin

def main(bands, n_bin):
    # Assume data in standard locations relative to current directory
    training_file = f'{bands}/training.hdf5'
    validation_file = f'{bands}/validation.hdf5'
    output_file = f'nn_{bands}_{n_bin}.png'
    export_file =f'{bands}_{n_bin}.flax'

    # Let's check if model already exists, in which case we just load it
    if os.path.exists(export_file):
        with open(export_file, 'rb') as file:
            classifier = serialization.from_bytes(get_classifier(n_bin, n_features[bands]),
                                     pickle.load(file))
    else:
        classifier = train_neural_network(training_file, bands, n_bin,
                                          export_file)

    tomo_bin = apply_neural_network(classifier, validation_file, bands)

    # Get a score
    z = load_redshift(validation_file)
    original_metrics.plot_distributions(z, tomo_bin, output_file)

    score =1.
    #score = original_metrics.compute_snr_score(tomo_bin, z)
    # Return. Command line invovation also prints out
    return score

if __name__ == '__main__':
    # Command line arguments
    try:
        bands = sys.argv[1]
        n_bin_max = int(sys.argv[2])
        assert bands in ['riz', 'griz']
    except:
        sys.stderr.write("Script takes two arguments, 'riz'/'griz' and n_bin_max\n")
        sys.exit(1)

    # Run main code
    for n_bin in range(1, n_bin_max+1):
        score = main(bands, n_bin)
        print(f"Score for {n_bin} bin(s) = {score:.1f}")
