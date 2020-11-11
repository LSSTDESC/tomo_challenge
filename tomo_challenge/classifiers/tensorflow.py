from .base import Tomographer
import numpy as np

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# from scipy import interpolate
import warnings
import h5py

import tensorflow as tf
from tensorflow import keras
# from dbn.tensorflow import SupervisedDBNClassification # version conflict with tf_v2 (had to put it inside the DBN class)

# Data normalization with sklearn
from sklearn import preprocessing

np.random.seed(1905)

# Helper functions
def colors_for_bands(bands):
    for i,b in enumerate(bands):
        for c in bands[i+1:]:
            yield b, c

def adjacent_colors_for_bands(bands):
    for i,b in enumerate(bands[0:-1]):
        yield b, bands[i+1]

def band_triplets_for_bands(bands):
    clrs = list(adjacent_colors_for_bands(bands))
    bts = colors_for_bands(clrs)
    bts_selected = [bt for bt in bts if bt[0][1]==bt[1][0]] # difference of two adjacent colors, e.g. (g-r)-(r-i)
    return bts_selected

def adjacent_colors_for_bands(bands):
    for i,b in enumerate(bands[0:-1]):
        yield b, bands[i+1]

def add_colors(data, bands, errors=False, band_triplets_errors=False):
    nband = len(bands)
    nobj = data[bands[0]].size
    ncolor = nband * (nband - 1) // 2

    # also get colors as data, from all the
    # (non-symmetric) pairs.  Note that we are getting some
    # redundant colors here, and some incorrect colors based
    # on the choice to set undetected magnitudes to 30.
    for b,c in colors_for_bands(bands):
        data[f'{b}{c}'] = data[f'{b}'] - data[f'{c}']
        if errors or band_triplets_errors:
            data[f'{b}{c}_err'] = np.sqrt(data[f'{b}_err']**2 + data[f'{c}_err']**2)

def load_mags(filename, bands, errors=False, band_triplets_errors=False, heal_undetected=False):

    # Warn about non-detections being set mag=30.
    # The system is only supposed to warn once but on
    # shifter it is warning every time and I don't understand why.
    # Best guess is one of the libraries we load sets some option.
    if heal_undetected:
        warnings.warn("Setting inf (undetected) bands to the mag for which S/N=1 (i.e. mag_err=2.5*log10(2))")
    else:
        warnings.warn("Setting inf (undetected) bands to mag=30 and mag_err=30")

    data = {}

    with h5py.File(filename, 'r') as f:
        # load all bands
        for b in bands:
            data[b] = f[f'{b}_mag'][:]

            if errors or band_triplets_errors:
                data[f'{b}_err'] = f[f'{b}_mag_err'][:]

    # Set undetected objects to S/N=1 mags and errors if heal_undetected=True, otherwise to mag 30 +/- 30
    print(f'Analyzing sample from {filename}')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    for ib, b in enumerate(bands):
        bad = ~np.isfinite(data[b])
        if heal_undetected:
            magerr_snr1=2.5*np.log10(2)
            mag_snr1 = my_npinterp(magerr_snr1, data[f'{b}_err'][~bad], data[b][~bad]) # mag_snr1, (A,B) = ...
            data[b][bad] = mag_snr1
            # print(f'band {b}: S/N~1, mag={mag_snr1}, mag_err={magerr_snr1} [replaced {sum(bad)} non-detections ~{100*sum(bad)/len(data[b]):.2f}%]')
            ## Make some informative plots to present
            # ax.scatter(data[f'{b}_err'][~bad], data[b][~bad], s=2, color=f'C{ib}', alpha=0.4)
            # mlist = np.linspace(1e-4,1,60)
            # ax.plot(mlist,A*np.log10(mlist)+B, label=b, color=f'C{ib}', alpha=0.8)
            # ax.set_xlabel('mag_err')
            # ax.set_ylabel('mag')
            # ax.set_xlim(-0.1,1)
            # ax.set_ylim(11,30)
            # ax.hlines(mag_snr1, ax.get_xlim()[0], magerr_snr1, colors=f'C{ib}', linestyles='--', alpha=0.8)
            # plt.legend(loc='lower right')
            # sample_name = filename.replace("/", " ").split()[-1].split(".")[-2]
            # if ib==len(bands)-1:
            #     ax.vlines(magerr_snr1, ax.get_ylim()[0], ax.get_ylim()[1], colors='k', linestyles='--', alpha=0.8)
            #     plt.savefig(f'./errormodel-{sample_name}.png')
            #     plt.close(fig)
        else:
            data[b][bad] = 30.0

        if errors or band_triplets_errors:
            if heal_undetected:
                data[f'{b}_err'][bad] = magerr_snr1
            else:
                data[f'{b}_err'][bad] = 30.0

    return data

def my_npinterp(x0,x,y, model='log'):
    ' my custom numpy interpolation/exterapolation '
    if x0<x.max():
        ix = x.argsort()
        return np.interp(x0,x[ix],y[ix])
    elif model=='linear':
        A, B = np.polyfit(x, y, 1) # y = Ax + B
        return A*x0+B #, (A,B)
    elif model=='log':
        A, B = np.polyfit(np.log10(x), y, 1) # y = A log(x) + B
        return A*np.log10(x0)+B #, (A,B)

def load_data(filename, bands, colors=False,
              errors=False, band_triplets=False, band_triplets_errors=False,
              array=False, heal_undetected=False):
    data = load_mags(filename, bands, errors=errors, band_triplets_errors=band_triplets_errors, heal_undetected=heal_undetected)

    if colors:
        add_colors(data, bands, errors=errors, band_triplets_errors=band_triplets_errors)

    if array:
        data = dict_to_array(data, bands, errors=errors, colors=colors, band_triplets=band_triplets, band_triplets_errors=band_triplets_errors)

    return data

def load_redshift(filename):
    """Load a redshift column from a training or validation file"""
    f = h5py.File(filename, 'r')
    z = f['redshift_true'][:]
    f.close()
    return z

def get_valueadded_data(training_file, validation_file, bands, errors, colors, band_triplets,
                        band_triplets_errors, heal_undetected, wants_arrays):
    """Make a value-added sample from training and validation files"""
    training_data = load_data(
        training_file,
        bands,
        array=wants_arrays,
        errors=errors,
        colors=colors,
        band_triplets=band_triplets,
        band_triplets_errors=band_triplets_errors,
        heal_undetected=heal_undetected
    )
    validation_data = load_data(
        validation_file,
        bands,
        array=wants_arrays,
        errors=errors,
        colors=colors,
        band_triplets=band_triplets,
        band_triplets_errors=band_triplets_errors,
        heal_undetected=heal_undetected
    )
    training_z = load_redshift(training_file)
    validation_z = load_redshift(validation_file)
    return training_data, validation_data, training_z, validation_z

def dict_to_array(data, bands, errors=False, colors=False, band_triplets=False, band_triplets_errors=False):
    nobj = data[bands[0]].size
    nband = len(bands)
    ncol = nband
    if colors:
        ncol += nband * (nband - 1) // 2
    if errors:
        ncol *= 2
    if band_triplets:
        ncol += (nband - 2)
    if band_triplets_errors:
        ncol += (nband - 2)

    arr = np.empty((ncol, nobj))
    i = 0
    for b in bands:
        arr[i] = data[b]
        i += 1

    if colors:
        for b, c in colors_for_bands(bands):
            arr[i] = data[f'{b}{c}']
            i += 1
            
    if errors:
        for b in bands:
            arr[i] = data[f"{b}_err"]
            i += 1

    if errors and colors:
        for b, c in colors_for_bands(bands):
            arr[i] = data[f'{b}{c}_err']
            i += 1
    
    if band_triplets:
        for (b1,b2), (c1,c2) in band_triplets_for_bands(bands):
            arr[i] = data[f'{b1}{b2}']-data[f'{c1}{c2}']
            i += 1

    if band_triplets_errors:
        for (b1,b2), (c1,c2) in band_triplets_for_bands(bands):
            arr[i] = np.sqrt(data[f'{b1}{b2}_err']**2+data[f'{c1}{c2}_err']**2)
            
    return arr.T

class TensorFlow_FFNN(Tomographer):
    """ TensorFlow Deep Neural Network Classifier """

    # valid parameter -- see below
    valid_options = ['bins','train_percent','test_percent','epochs','activation','optimizer',
                     'data_scaler','heal_undetected','band_triplets','band_triplets_errors',
                     'training_file','validation_file'] #, 'excess_prob_percent']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = True # redundant in this line for this method
    
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

        wants_arrays = True
        self.bands = bands
        # self.excess_prob_percent = options['excess_prob_percent']
        self.opt = options
        np.random.seed(1905)
        
        # Create value-added data
        print("Creating value-added data")
        self.training_data, self.validation_data, self.training_z, self.validation_z = get_valueadded_data(
             options['training_file'], options['validation_file'], bands, options['errors'], options['colors'],
             options['band_triplets'], options['band_triplets_errors'], options['heal_undetected'], wants_arrays
        )

    def train (self, *args, **kwargs):
        """Trains the classifier
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample
        """

        del args, kwargs # we already loaded our value-added data in __init__
        data_scaler = self.opt['data_scaler'] if 'data_scaler' in self.opt else 'MinMaxScaler'
        n_bin = self.opt['bins']
        train_percent = self.opt['train_percent'] if 'train_percent' in self.opt else 2
        epochs = self.opt['epochs'] if 'epochs' in self.opt else 3
        activation = self.opt['activation'] if 'activation' in self.opt else 'relu'
        optimizer = self.opt['optimizer'] if 'optimizer' in self.opt else 'adam'
        
        # Data rescaling
        self.scaler = getattr(preprocessing, data_scaler)()
        
        print(f"Using {data_scaler} to rescale data for better results")
    
        # Fit scaler on data and use the same scaler in the future when needed
        self.scaler.fit(self.training_data)

        print("Finding bins for training data")

        # apply transform to get rescaled values
        self.training_data = self.scaler.transform(self.training_data) # inverse: data_original = scaler.inverse_transform(data_rescaled)
        
        # Now put the training data into redshift bins.
        # Use zero so that the one object with minimum
        # z in the whole survey will be in the lowest bin
        training_bin = np.zeros(self.training_z.size)

        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
        p = np.linspace(0, 100, n_bin + 1)
        z_edges = np.percentile(self.training_z, p)

        # Now find all the objects in each of these bins
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            training_bin[(self.training_z > z_low) & (self.training_z < z_high)] = i

        if 0<train_percent<100:
            # for speed, cut down to ?% of original size
            print(f'Cutting down to {train_percent}% of original training sample size for speed.')
            cut = np.random.uniform(0, 1, self.training_z.size) < train_percent/100
            training_bin = training_bin[cut]
            self.training_data = self.training_data[cut]
        elif train_percent==100:
            pass
        else:
            raise ValueError('train_percent is not valid')

        print('Setting up the layers')
        # Set up the layers
        classifier = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.training_data.shape[1],)),
            keras.layers.Dense((n_bin+self.training_data.shape[1])//2, activation=activation),
            #keras.layers.Dense((3*n_bin+self.training_data.shape[1])//4, activation=activation),
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
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # This callback will stop the training when there is no improvement in
        # the validation loss for three consecutive epochs. (This is to avoid overfitting)
        history = classifier.fit(self.training_data, training_bin, epochs=epochs, callbacks=[callback])
        epochs_ran = len(history.history["loss"])
        if epochs_ran<epochs:
            print(f'Only {epochs_ran} out of maximum {epochs} epochs are run to avoid overfitting.')

        self.classifier = classifier
        self.z_edges = z_edges

    def apply (self, *args, **kwargs):
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
        
        del args, kwargs # we already loaded our value-added data in __init__

        # Apply transform to get rescaled values using the scaler we already fit to the training data
        self.validation_data = self.scaler.transform(self.validation_data)

        # Make predictions
        probability_model = tf.keras.Sequential([self.classifier, 
                                                 tf.keras.layers.Softmax()])

        # Get the probabilities
        probs = probability_model.predict(self.validation_data)
        
        # Find the index of the most probable ones
        tomo_bin = np.argmax(probs, axis=1)
        
        # if self.excess_prob_percent:
        #     rejected = np.max(probs, axis=1)*probs.shape[1]<(1+self.excess_prob_percent/100)
        #     tomo_bin[rejected] = -1
        #     print(f"{100*sum(rejected)/probs.shape[0]}% of validation data has been rejected ")
        
        return tomo_bin

class TensorFlow_DBN(Tomographer):
    """ TensorFlow Deep Belief Network Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins','train_percent','test_percent','n_epochs_rbm','hidden_layers_structure',
                     'activation','learning_rate_rbm','learning_rate','n_iter_backprop','batch_size',
                     'dropout_p','data_scaler','heal_undetected','band_triplets','band_triplets_errors',
                     'training_file','validation_file']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = True # redundant in this line for this method

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
    
        wants_arrays = True
        self.bands = bands
        self.opt = options
        np.random.seed(1905)
        
        # Create value-added data
        print("Creating value-added data")
        self.training_data, self.validation_data, self.training_z, self.validation_z = get_valueadded_data(
             options['training_file'], options['validation_file'], bands, options['errors'], options['colors'],
             options['band_triplets'], options['band_triplets_errors'], options['heal_undetected'], wants_arrays
        )
        
    def train (self, *args, **kwargs):
        """Trains the classifier
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample
        """

        del args, kwargs # we already loaded our value-added data in __init__
        from dbn.tensorflow import SupervisedDBNClassification
            
        data_scaler = self.opt['data_scaler'] if 'data_scaler' in self.opt else 'MinMaxScaler'
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

        # Data rescaling
        self.scaler = getattr(preprocessing, data_scaler)()
        
        print(f"Using {data_scaler} to rescale data for better results")
    
        # Fit scaler on data and use the same scaler in the future when needed
        self.scaler.fit(self.training_data)
        
        # apply transform to get rescaled values
        self.training_data = self.scaler.transform(self.training_data) # inverse: data_original = scaler.inverse_transform(data_rescaled)
        
        # Now put the training data into redshift bins.
        # Use zero so that the one object with minimum
        # z in the whole survey will be in the lowest bin
        training_bin = np.zeros(self.training_z.size)

        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
        p = np.linspace(0, 100, n_bin + 1)
        z_edges = np.percentile(self.training_z, p)

        # Now find all the objects in each of these bins
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            training_bin[(self.training_z > z_low) & (self.training_z < z_high)] = i

        if 0<train_percent<100:
            # for speed, cut down to ?% of original size
            print(f'Cutting down to {train_percent}% of original training sample size for speed.')
            cut = np.random.uniform(0, 1, self.training_z.size) < train_percent/100
            training_bin = training_bin[cut]
            self.training_data = self.training_data[cut]
        elif train_percent==100:
            pass
        else:
            raise ValueError('train_percent is not valid')

        print('Setting up the layers for DBN')
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
        classifier.fit(self.training_data, training_bin)

        self.classifier = classifier
        self.z_edges = z_edges

    def apply (self, *args, **kwargs):
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

        del args, kwargs # we already loaded our value-added data in __init__

        # Apply transform to get rescaled values using the scaler we already fit to the training data
        self.validation_data = self.scaler.transform(self.validation_data)
        
        # Find predictions
        tomo_bin = self.classifier.predict(self.validation_data)
        
        return np.array(tomo_bin)
