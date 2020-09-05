from .base import Tomographer
import numpy as np

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
# from dbn.tensorflow import SupervisedDBNClassification # version conflict with tf_v2 (had to put it inside the DBN class)
    
class TensorFlow_FFNN(Tomographer):
    """ TensorFlow Deep Neural Network Classifier """

    # valid parameter -- see below
    valid_options = ['bins','train_percent','epochs','activation','optimizer']
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
        np.random.seed(1905)

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
        train_percent = self.opt['train_percent'] if 'train_percent' in self.opt else 2
        epochs = self.opt['epochs'] if 'epochs' in self.opt else 3
        activation = self.opt['activation'] if 'activation' in self.opt else 'relu'
        optimizer = self.opt['optimizer'] if 'optimizer' in self.opt else 'adam'
        print("Finding bins for training data")

        # Data scaling # max mag 30?
        training_data = (training_data / 30).astype(np.float32)

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

        if train_percent<100:
            # for speed, cut down to ?% of original size
            print(f'Cutting down to {train_percent}% of original size for speed.')
            cut = np.random.uniform(0, 1, training_z.size) < train_percent/100
            training_bin = training_bin[cut]
            training_data = training_data[cut]
        elif train_percent>100:
            raise ValueError('train_percent>100 is not valid')

        print('Setting up the layers')
        # Set up the layers
        classifier = keras.Sequential([
            keras.layers.Flatten(input_shape=(training_data.shape[1],)),
            keras.layers.Dense(8, activation=activation),
            keras.layers.Dense(n_bin)
        ])

        print('Compiling the model')
        # Compile the model
        classifier.compile(optimizer=optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        # Train the model
        print("Fitting classifier")
        # Lots of data, so this will take some time
        classifier.fit(training_data, training_bin, epochs=epochs)

        self.classifier = classifier
        self.z_edges = z_edges

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

        # Data scaling # max mag 30?
        data = (data / 30).astype(np.float32)

        # Make predictions
        probability_model = tf.keras.Sequential([self.classifier, 
                                                 tf.keras.layers.Softmax()])

        # Get the probabilities
        probs = probability_model.predict(data)
        
        # Find the index of the most probable ones
        tomo_bin = np.argmax(probs, axis=1)
        
        return tomo_bin

class TensorFlow_DBN(Tomographer):
    """ TensorFlow Deep Belief Network Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins','train_percent','n_epochs_rbm','hidden_layers_structure',
                     'activation','learning_rate_rbm','learning_rate','n_iter_backprop','batch_size','dropout_p']
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
        np.random.seed(1905)

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

        from dbn.tensorflow import SupervisedDBNClassification
            
        n_bin = self.opt['bins']
        train_percent = self.opt['train_percent'] if 'train_percent' in self.opt else 1
        n_epochs_rbm = self.opt['n_epochs_rbm'] if 'n_epochs_rbm' in self.opt else 2
        activation = self.opt['activation'] if 'activation' in self.opt else 'relu'
        learning_rate_rbm = self.opt['learning_rate_rbm'] if 'learning_rate_rbm' in self.opt else 0.05
        learning_rate = self.opt['learning_rate'] if 'learning_rate' in self.opt else 0.1
        n_iter_backprop = self.opt['n_iter_backprop'] if 'n_iter_backprop' in self.opt else 25
        batch_size = self.opt['batch_size'] if 'batch_size' in self.opt else 32
        dropout_p = self.opt['dropout_p'] if 'dropout_p' in self.opt else 0.2
        hidden_layers_structure = self.opt['hidden_layers_structure'] if 'hidden_layers_structure' in self.opt else [256, 256]
                                    
        print("Finding bins for training data")

        # Data scaling # max mag 30?
        training_data = (training_data / 30).astype(np.float32)

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

        if train_percent<100:
            # for speed, cut down to ?% of original size
            print(f'Cutting down to {train_percent}% of original size for speed.')
            cut = np.random.uniform(0, 1, training_z.size) < train_percent/100
            training_bin = training_bin[cut]
            training_data = training_data[cut]
        elif train_percent>100:
            raise ValueError('train_percent>100 is not valid')

        print('Setting up the layers')
        # Set up the layers
        classifier = SupervisedDBNClassification(hidden_layers_structure=hidden_layers_structure,
                                                 learning_rate_rbm=learning_rate_rbm,
                                                 learning_rate=learning_rate,
                                                 n_epochs_rbm=n_epochs_rbm,
                                                 n_iter_backprop=n_iter_backprop,
                                                 batch_size=batch_size,
                                                 activation_function=activation,
                                                 dropout_p=dropout_p)

        # Train the model
        print("Fitting classifier")
        classifier.fit(training_data, training_bin)

        self.classifier = classifier
        self.z_edges = z_edges

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

        # Data scaling
        data = (data / 30).astype(np.float32)


#         from sklearn.metrics import accuracy_score
        
        # Find predictions
        tomo_bin = self.classifier.predict(data)
#         print(f'Done.\nAccuracy: {accuracy_score(data, tomo_bin)}') # error ValueError: Classification metrics can't handle a mix of continuous-multioutput and binary targets
        
        return np.array(tomo_bin)
