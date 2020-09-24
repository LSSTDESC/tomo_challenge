"""
TCN classifier
This is an example tomographic bin generator using a temporal convolutional neural network (TCN).
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
from tcn import TCN
import h5py


class TCN(Tomographer):
    """ TCN deep classifier """
    
    # valid parameter -- see below
    valid_options = ['bins']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = True
    skips_zero_flux = True
    
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

        """
        self.bands = bands
        self.opt = options
    
    def load_data(fname, take_colors=True, cutoff=0.0):
        n_bins = self.opt['bins']
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
        p = np.linspace(0, 100, n_bins+1)
        z_edges = np.percentile(redshift, p)
        train_bin = np.zeros(all_mags.shape[0])
        for i in range(n_bins):
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

        n_bin = self.opt['bins']
        print("Finding bins for training data")
        # Now put the training data into redshift bins.
        # Use zero so that the one object with minimum
        # z in the whole survey will be in the lowest bin
        training_bin = np.zeros(training_z.size)

        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
        p = np.linspace(0, 100, n_bin + 1)
        z_edges = np.percentile(training_z, p)

        # Now find all the objects in each of these bins
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            training_bin[(training_z > z_low) & (training_z < z_high)] = i
        #cut = np.random.uniform(0, 1, training_z.size) < 0.05
        #training_data = training_data[cut]
        #training_bin = training_bin[cut]
        

        inp = keras.layers.Input(shape=(training_data.shape[1], 1))
        x = TCN(nb_filters=[128, 128], kernel_size=2, dilations=[1, 2], nb_stacks=2, activation='relu', return_sequences=False, use_batch_norm=True, use_skip_connections=True)(inp) 
    
        x = keras.layers.Dense(n_bin, activation='softmax')(x)
    
        model = keras.models.Model(inp, x)
    
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
    
        # Can be replaced with any classifier
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(training_data)
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = np.expand_dims(training_bin, axis=-1)
        print("Fitting classifier")
        # Lots of data, so this will take some time
        model.fit(x_train, y_train, epochs=20, verbose=0)

        self.classifier = model
        self.z_edges = z_edges
        self.scaler = scaler


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
        preds = self.classifier.predict(x_test)
        tomo_bin = np.argmax(preds, axis=1)
        return tomo_bin

