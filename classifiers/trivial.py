"""
Trivial classifiers. Used as example and testing.

Every classifier module needs to:
 - have construction of the type 
       __init__ (self, bands, options) (see examples below)
 -  implement two functions: 
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.

See Classifier Documentation below.


"""
import numpy as np

class Random:
    """Completely random classifier. 

    Every object goes into a random bin. 
    """

    ## see constructor below
    valid_options = ['bins','seed']
    
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
            'seed' - random number seed (passed to numpy.random) 

        """
        self.opt = options
        if 'seed' not in options: self.opt['seed'] = 123
        if 'bins' not in options: self.opt['bins'] = 3

    def train (self,training_data, training_z):
        """Trains the classifier
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """
        pass

    def apply (self,data):
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

        np.random.seed(self.opt['seed'])
        nbins = self.opt["bins"]
        tomo_bin = np.random.randint(0,nbins,len(data))
        return tomo_bin


class ILover:
    """Classifier for people who love nothing more than the i-band. 
       Classifies in uniform bins in the i-band mag. """

    valid_options = ['bins']

    def __init__ (self, bands, options):
        self.indx = bands.find("i")
        self.opt = options
        if 'bins' not in options: self.opt['bins'] = 3
        
    def train (self,training_data, training_z):
        pass

    def apply (self,data):
        nbins = self.opt["bins"]
        iband = data[:,self.indx]
        tomo_bin = np.digitize(iband, np.linspace(iband.min(),iband.max(), nbins))
        tomo_bin -= 1 ## we need start with 0
        print(tomo_bin.min(), tomo_bin.max())
        return tomo_bin



