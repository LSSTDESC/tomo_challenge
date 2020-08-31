"""
This is an example tomographic bin generator using a random forest for logarithmic bins.

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
from sklearn.ensemble import RandomForestClassifier


class RF_Log_CombineBins(Tomographer):
    """ Random Forest Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = True
    
    def __init__ (self, bands, options, newbins):
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
        self.newbins = newbins

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
        newbins = self.newbins
        
        print("Finding bins for training data")
        # Now put the training data into redshift bins.
        # Use zero so that the one object with minimum
        # z in the whole survey will be in the lowest bin
        training_bin = np.zeros(training_z.size)

        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
#         p = np.linspace(0, 100, n_bin + 1)
        p = np.logspace(0, 2, n_bin+1)
        print('the bins are in logspace')
        z_edges = np.percentile(training_z, p)

        # Now find all the objects in each of these bins
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            training_bin[(training_z > z_low) & (training_z < z_high)] = i
        
        #combine bins 
        newbins.sort()
        training_bin[training_bin == newbins[1]] = newbins[0]
        
        #rename/move bins
        if newbins[1] != np.max(training_bin):
            training_bin[training_bin == np.max(training_bin)] = newbins[1]
        
        # for speed, cut down to 5% of original size
        cut_percent = 0.05
        print('cut percent:', cut_percent)
        cut = np.random.uniform(0, 1, training_z.size) < cut_percent
        training_bin = training_bin[cut]
        training_data = training_data[cut]

        # Can be replaced with any classifier
        classifier = RandomForestClassifier()

        print("Fitting classifier")
        # Lots of data, so this will take some time
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
        tomo_bin = self.classifier.predict(data)
        return tomo_bin

