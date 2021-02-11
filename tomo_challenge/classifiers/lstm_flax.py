"""
Flax LSTM 
This is an example tomographic bin generator using a convolutional LSTM neural network using JAX/FLAX
We also added a custom data loader we tested.
This solution was developed by the Brazilian Center for Physics Research AI 4 Astrophysics team.
Authors: Clecio R. Bom, Bernardo M. Fraga, Gabriel Teixeira, Eduardo Cypriano and Elizabeth Gonzalez.
contact: debom |at |cbpf| dot| br
Every classifier module needs to:
 - have construction of the type 
       __init__ (self, bands, options) (see examples below)
 -  implement two functions: 
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.
See Classifier Documentation below.
"""

from .base import Tomographer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import h5py
from flax import nn, optim, jax_utils
import jax.random as rand
import jax.numpy as jnp
import jax
from .. import jax_metrics as j_metrics

def get_classifier(n_bin, n_features):
    class BinningNN(nn.Module):
        def apply(self, x):
            batch_size = x.shape[0]
            net = nn.Conv(x, features=64, kernel_size=(2,), padding='SAME', name='fc1')
            net = nn.tanh(net)
            net = nn.max_pool(net, window_shape=(2,), strides=(2,), padding='SAME')
            net = nn.Conv(net, 128, kernel_size=(2,), padding='SAME', name='fc2')
            net = nn.tanh(net)
            net = nn.max_pool(net, (2,), strides=(2,), padding='SAME')
            carry = nn.LSTMCell.initialize_carry(rand.PRNGKey(0), (batch_size,), 972)
            _, outputs = jax_utils.scan_in_dim(nn.LSTMCell.partial(), carry, net, axis=1)
            net_dense = outputs.reshape((outputs.shape[0], -1))
            net_dense = nn.Dense(net_dense, 486, name='fc3')
            net_dense = nn.tanh(net_dense)
            net_dense = nn.Dense(net_dense, 486)
            net_dense = nn.tanh(net_dense)
            return nn.softmax(nn.Dense(net_dense, n_bin))
    _, initial_params = BinningNN.init_by_shape( rand.PRNGKey(0), [((1, n_features,1), jnp.float32)])
    return nn.Model(BinningNN, initial_params)

class Flax_LSTM(Tomographer):
    """ LSTM using flax, optimising for SNR """
    
    # valid parameter -- see below
    valid_options = ['bins', 'n_feats']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = True
    skips_zero_flux = False
    
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
        Valid options are:
            'bins' - number of tomographic bins
            'n_feats' - number of features

        """
        self.bands = bands
        self.opt = options
        self.n_bins = options['bins']
        self.n_features = options['n_feats']
        self.model = get_classifier(self.n_bins, self.n_features)#, self.n_bins)
        self.scaler = MinMaxScaler()
    
    def load_data(self, fname, take_colors=True, cutoff=0.0):
        
        data = h5py.File(fname, 'r')
        r_mag = data['r_mag']
        g_mag = data['g_mag']
        i_mag = data['i_mag']
        z_mag = data['z_mag']
        redshift = data['redshift_true']
        all_mags = np.vstack([g_mag, r_mag, i_mag, z_mag])
        all_mags = all_mags.T
        mask = (all_mags != np.inf).all(axis=1)
        all_mags = all_mags[mask,:]
        redshift = redshift[mask]
        gr_color = all_mags[:,0] - all_mags[:,1]
        ri_color = all_mags[:,1] - all_mags[:,2]
        iz_color = all_mags[:,2] - all_mags[:,3]
        all_colors = np.vstack([gr_color, ri_color, iz_color])
        all_colors = all_colors.T
        p = np.linspace(0, 100, self.n_bins+1)
        z_edges = np.percentile(redshift, p)
        train_bin = np.zeros(all_mags.shape[0])
        for i in range(self.n_bins):
            z_low = z_edges[i]
            z_high = z_edges[i+1]
            train_bin[(redshift > z_low) & (redshift <= z_high)] = i
        if cutoff != 0.0:
            cut = np.random.uniform(0, 1, all_mags.shape[0]) < cutoff
            train_bin = train_bin[cut].reshape(-1,1)
            all_mags = all_mags[cut]
            all_colors = all_colors[cut]
            redshift = redshift[cut]
        else:
            train_bin = train_bin.reshape(-1,1)
        if take_colors:
            return np.hstack([all_mags, all_colors]), redshift, train_bin.astype(int), z_edges
        else:
            return mags, redshift, train_bin.astype(int), z_edges


         
    def train(self, training_data, training_z, batch_size=512, epochs=20):
        x_train = self.scaler.fit_transform(training_data)
        x_train = np.expand_dims(x_train, axis=-1)
        lr = 0.001
        optimizer = optim.Adam(learning_rate=lr).create(self.model)

        @jax.jit
        def train_step(optimizer, x, y):
        # This is the loss function
            def loss_fn(model):
            # Apply classifier to features
                w = model(x)
                return 1000./ j_metrics.compute_snr_score(w, y)
        # Compute gradients
            loss, g = jax.value_and_grad(loss_fn)(optimizer.target)
        # Perform gradient descent
            optimizer = optimizer.apply_gradient(g)
            return optimizer, loss

        def get_batches():
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, training_z))
            train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batch_size)
            return train_dataset

        losses = []
        for e in range(epochs):
            print("Running training epoch ", e)
            for i, (x_train1, labels) in enumerate(get_batches().as_numpy_iterator()):
                optimizer, loss = train_step(optimizer, x_train1, labels)
                losses.append(loss)

            print('Epoch {}\nLoss = {}'.format(e, loss))

   
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
        
        x_test = self.scaler.transform(data)
        x_test = np.expand_dims(x_test, axis=-1)
        bs = 512
        batches = tf.data.Dataset.from_tensor_slices(x_test).batch(bs)
        
        preds = []
        for i, test in enumerate(batches.as_numpy_iterator()):
            p = self.model(test)
            preds.append(p)
        
        result = np.concatenate([p for p in preds], axis=0)
        tomo_bin = np.argmax(result, axis=1)
        return tomo_bin
