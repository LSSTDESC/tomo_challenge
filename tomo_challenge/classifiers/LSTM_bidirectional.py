"""
Bidirectional LSTM Classifier

This is an example tomographic bin generator using a convolutional bidirectional LSTM neural network.

This solution was developed by the Brazilian Center for Physics Research AI 4 Astrophysics team.

Authors: Clecio R. Bom, Bernardo M. Fraga, Gabriel Teixeira and Elizabeth Gonzalez.

In our preliminary tests we found a SNR 3X2 of  ~1930 for n=10 bins. 

This is an example tomographic bin generator using a convolutional bidirectional LSTM neural network.


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

class LSTM(Tomographer):
    """ Bidirectional LSTM deep classifier """
    
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
        Valid options are:
            'bins' - number of tomographic bins

        """
        self.bands = bands
        self.opt = options

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

        
        lstm_out = np.int(486*2)
        inp = keras.layers.Input(shape=inp_shape)
    
        x = keras.layers.Conv1D(64, 3, padding='same')(inp)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.MaxPooling1D(2, padding='same')(x)
    
        x = keras.layers.Conv1D(128, 3, padding='same')(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.MaxPooling1D(2, padding='same')(x)
    
        x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_out, return_sequences=False), merge_mode='concat')(x)
    
        x = keras.layers.Dense(lstm_out // 2, activation='tanh')(x)
        x = keras.layers.Dense(lstm_out // 2, activation='tanh')(x)
    
        x = keras.layers.Dense(n_bins, activation='softmax')(x)
    
        model = keras.models.Model(inp, x)
    
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
    
        # Can be replaced with any classifier
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(training_data)
        print("Fitting classifier")
        # Lots of data, so this will take some time
        model.fit(x_train, training_bin, epochs=15, verbose=0)

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
        preds = self.classifier.predict(x_test)
        tomo_bin = np.argmax(preds, axis=1)
        return tomo_bin

