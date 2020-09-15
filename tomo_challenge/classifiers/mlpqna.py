"""
This is an example tomographic bin generator using a random forest.

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
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.preprocessing import StandardScaler




class mlpqna(Tomographer):
    """ Random Forest Classifier """
    
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

        # for speed, cut down to 5% of original size
        cut = np.random.uniform(0, 1, training_z.size) < 0.05
        training_bin = training_bin[cut]
        training_data = training_data[cut]

        # Can be replaced with any classifier
        actFunc ="tanh"
        decayVal =0.1
        h1Layers = 2
        h1LayerNodes = 2* training_data.shape[1] -1
        h2LayerNodes = training_data.shape[1]+1
        trainEpochs = 20000
        seed = 3214234
        tolerance = 1e-20
        verboseFlag = True
        warmFlag = False
        maxFun = 30000        
        classifier= mlp(activation=actFunc, solver='lbfgs', alpha=decayVal, hidden_layer_sizes=(h1LayerNodes, h2LayerNodes), max_iter=trainEpochs, random_state=seed, tol=tolerance, verbose=verboseFlag, warm_start=warmFlag, max_fun=maxFun)
        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(training_data)
        # Now apply the transformations to the data:
        X_norm_train = scaler.transform(training_data)

        print("Fitting classifier")
        # Lots of data, so this will take some time
        classifier.fit(X_norm_train, training_bin)
        self.scaler=scaler
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
        X_norm_data = self.scaler.transform(data)
        tomo_bin = self.classifier.predict(X_norm_data)
        return tomo_bin

