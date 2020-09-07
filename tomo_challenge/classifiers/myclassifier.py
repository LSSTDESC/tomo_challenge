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
from sklearn.ensemble import RandomForestClassifier


class funbins(Tomographer):
    """ Random Forest Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins', 'seed', 'method', 'combinebins']
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
            'seed' - the seed to use for RandomState
            'method' - what method to use for binning: 'log', 'random', 'linear'
            'combinebins' - what 2 bins to combine as a list

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
        #reading in elements from yaml
        n_bin = self.opt['bins']
        
        if self.opt['seed'] is None:
            seed = 123
            print('The default seed is 123')
        else:
            seed = self.opt['seed']
            print(f'The seed is {seed}')
        
        if self.opt['method'] is None:
            method = 'log'
            print('The default method for binning has been set to log.')
        else:
            method = self.opt['method']
            assert method in ['log', 'random', 'linear'], 'The method must be log, random, or linear'
            print(f'The method has been set to {method}')
        
        if self.opt['combinebins'] is None:
            combine = None
        elif self.opt['combinebins'] is not None:
            combine = self.opt['combinebins']
            assert len(combine == 2), "You can only combine 2 bins at one time right now!"
            print(f'You are combining bins {combine[0]} and {combine[1]}')
            
        #set up a reproducible random state
        gen = np.random.RandomState(seed)
        
        #find bins
        print("Finding bins for training data")
        
        training_bin = np.zeros(training_z.size)        
        
        if method == 'log':
            #creating percentile binning in logspace
            p = np.logspace(0, 2, n_bin + 1)
            z_edges = np.percentile(training_z, p)
        
        if method == 'random':
            #creating a random binning by pulling from a uniform distribution
            z_edges = gen.uniform(0, 3, n_bin - 1)
            z_edges = np.insert(z_edges, 0, np.min(training_z))
            z_edges = np.insert(z_edges, 0, np.max(training_z))
            z_edges.sort()
        
        if method == 'linear':
            #the given random forest linear binning
            p = np.linspace(0, 100, n_bin +1)
            z_edges = np.percentile(training_z, p)
            
        #can add stuff here about david's method if it can be generalized to n bins
        #n_bin = 8
        #training_bin = np.load('dc2-labels.npy')#[:len(training_z)]
        #print(len(training_bin))        
        
        # Now find all the objects in each of these bins
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            training_bin[(training_z > z_low) & (training_z < z_high)] = i
        
        if combine:
            max_ = np.max(training_bin)
            if combine[1] == max_:
                #combine the bins, combining the higher/max bin into the lower bin is also moving because its max bin
                training_bin[training_bin == combine[1]] = combine[0]
            if combine[1] != max_:
                #combine the bins
                training_bin[training_bin == combine[1]] = combine[0]
                #move the max bin into the empty spot
                training_bin[training_bin == max_] = combine[1]
        
        # for speed, cut down to 5% of original size
        cut = gen.uniform(0, 1, training_z.size) < 0.05
        training_bin = training_bin[cut]
        training_data = training_data[cut]

        # Can be replaced with any classifier
        classifier = RandomForestClassifier()

        print("Fitting classifier")
        # Lots of data, so this will take some time
        classifier.fit(training_data, training_bin)

        self.classifier = classifier
        #self.z_edges = z_edges


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

