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
from .base import Tomographer

class SummerSlasher(Tomographer):
    """
     There is some magic here.
    """

    ## see constructor below
    valid_options = ['n_slashes','seed', 'pop_size']
    
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
        if 'n_slashes' not in options: self.opt['n_slashes'] = 3
        if 'pop_size' not in options: self.opt['pop_size'] = 1000
        self.bands = bands
        self.Nd = len(bands)
        
        
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
        ## first let's get train data into a nice array
        data = np.vstack([training_data[band] for band in self.bands]).T
        self.pop = [Slasher(self.opt['n_slashes'],data) for i in range(self.opt['pop_size'])]
        sels = self.pop[0].get_selections()
        print (np.histogram(sels))

        
        stop()
        

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


class Slash:
    def __init__ (self, train_data):
        Ns = train_data.shape[1]
        self.w = np.random.uniform(-1,1,Ns)
        self.C = np.median((train_data*self.w).sum(axis=1))

    def apply (self,data):
        return (((data*self.w).sum(axis=1)-self.C)>0).astype(int)
    

        
class Slasher:
    def __init__ (self, n_slashes, train_data):
         self.data = train_data
         self.slashes = [Slash(self.data) for i in range(n_slashes)]
         
    def get_selections(self):
        i=1
        sel = np.zeros(self.data.shape[0],int)
        for slash in self.slashes:
            sel+=slash.apply(self.data)*i
            i*=2
        return sel
