from .base import Tomographer

import numpy as onp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from flax import nn, optim

import tomo_challenge as tc
from tomo_challenge import jax_metrics

from jax_cosmo.redshift import kde_nz

import os

class JaxCNN(Tomographer):
    """ Neural Network Classifier """
    
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
        
        assert self.bands in ["riz", "griz"]

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
        
        n_bins = self.opt['bins']
        print("Finding bins for training data")
        
        # Simple CNN from flax
        class CNN(nn.Module):
            def apply(self, x):
                b = x.shape[0]
                x = nn.Conv(x, features=128, kernel_size=(4,), padding='SAME')
                x = nn.BatchNorm(x)
                x = nn.leaky_relu(x)
                x = nn.avg_pool(x, window_shape=(2,), padding='SAME')
                x = nn.Conv(x, features=256, kernel_size=(4,), padding='SAME')
                x = nn.BatchNorm(x)
                x = nn.leaky_relu(x)
                x = nn.avg_pool(x, window_shape=(2,), padding='SAME')
                x = x.reshape(b, -1)
                x = nn.Dense(x, features=128)
                x = nn.BatchNorm(x)
                x = nn.leaky_relu(x)
                x = nn.Dense(x, features=n_bins)
                x = nn.softmax(x)
                return x
        
        # Hyperparameters
        prng = jax.random.PRNGKey(0)
        learning_rate = 0.001
        input_shape = (1, training_data.shape[1], 1)
        batch_size = 5000
        epochs = 250
        
        # Initialize model and optimizer
        def create_model_optimizer(n_bins):
            _, initial_params = CNN.init_by_shape(prng, [(input_shape, jnp.float32)])
            model = nn.Model(CNN, initial_params)
            optimizer = optim.Adam(learning_rate=learning_rate).create(model)
            return model, optimizer
        
        # Helper function
        def get_batch():
            inds = onp.random.choice(len(training_z), batch_size)
            return {'labels': training_z[inds], 'features': training_data[inds]}
        
        @jax.jit
        def train_step(optimizer, batch):
            # Define loss function as 1 / FOM
            def loss_fn(model):
                w = model(batch['features'][..., jnp.newaxis])
                return 1. / jax_metrics.compute_fom(w, batch['labels'], inds=[5,6])
            loss, g = jax.value_and_grad(loss_fn)(optimizer.target)
            optimizer = optimizer.apply_gradient(g)
            return optimizer, loss
        
        
        model, optimizer = create_model_optimizer(n_bins)
        
        losses = []
        # Training
        for epoch in range(epochs):
            batch = get_batch()
            optimizer, loss = train_step(optimizer, batch)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
        
        
        # Plotting the loss curve
#        figure = plt.figure(figsize=(10, 6))
#        plt.plot(range(epochs), losses)
#        plt.xlabel('Epoch')
#        plt.ylabel('1 / FOM')
#        plt.yscale('log')
#        plt.savefig(f'../../{n_bins}-bins_{self.bands}.png')
#        plt.close()
        
        self.model = optimizer.target
        

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
        print("Applying classifier")
        batch_size = 10000
        data = jnp.array(data)
        Ngalaxies = len(data)
        tomo_bin = jnp.concatenate(
            [self.model(data[batch_size * i : min((batch_size * (i + 1)), Ngalaxies)][..., jnp.newaxis]) 
             for i in range(Ngalaxies // batch_size + 1)]
        )
        
        return jnp.argmax(tomo_bin, axis=-1)

