""" This is an example tomographic bin generator using the code GPz

Every classifier module needs to:
 - have construction of the type 
       __init__ (self, bands, options) (see examples below)
 -  implement two functions: 
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.

See Classifier Documentation below.
"""

# The only extra code it needs is GPz, which can be accessed at
#pip3 install --upgrade 'git+https://github.com/OxfordML/GPz#egg=GPz' 
# This is unfortunately only in python2.7 at the moment...
# It also calls two python2 scripts (GPz is in python2), classifier_train_GPz.py and classifier_predict_GPz.py
# Train requires file_prefix to tell it where you put these files

## Options:
# bins - number of bins
# edge_strictness - how close to the edges of the redshift bins relative to the uncertainty on the redshift permitted (higher is stricter)
# extrapolate_threshold - how much extrapolation is permitted (lower is stricter). This is probably not hugely valuable here, but might be if the test and training data were different.

from .base import Tomographer
import numpy as np
from . import GPz

import subprocess


class GPzBinning(Tomographer):
    """ GPz Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins','edge_strictness','extrapolate_threshold']
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
            'edge_strictness [default=0.0] Essentially how big error bars can be compared to the bin edges
            'extrapolate_threshold' [default=1.0] Essentially how much extrapolation should be allowed

        """
        self.bands = bands
        self.opt = options

    def train (self, training_data, training_z):

        
        X_train = training_data
        n_train,d = X_train.shape
        
        # np.savetxt('train_data.csv',training_data)
        # np.savetxt('training_z.csv',training_z)


        # subprocess.run(["python2", file_prefix+"classifier_train_GPz.py"])
        self.model = GPz.train(training_data, training_z)


        # Sort out bin edges
        n_bin = self.opt['bins']
        print("Finding bins for training data")
        # Now put the training data into redshift bins.
        # Use zero so that the one object with minimum
        # z in the whole survey will be in the lowest bin
        #training_bin = np.zeros(training_z.size)

        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
        p = np.linspace(0, 100, n_bin + 1)
        z_edges = np.percentile(training_z, p)

        # Now find all the objects in each of these bins
        #for i in range(n_bin):
        #    z_low = z_edges[i]
        #    z_high = z_edges[i + 1]
        #    training_bin[(training_z > z_low) & (training_z < z_high)] = i

        #self.photoz_predictor = model
        self.z_edges = z_edges


    def apply (self, testing_data):
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
        
        # Save data
        # np.savetxt('test_data.csv',testing_data)

        # Run GPz for predictions
        # subprocess.run(["python2", file_prefix+"classifier_predict_GPz.py"])
        data = GPz.predict(self.model, testing_data)

        

        # data= np.genfromtxt('prediction_data.csv')
        mu=data[:,0]
        sigma=data[:,1]
        modelV=data[:,2]
        noiseV=data[:,3]
        
        z_edges=self.z_edges
        n_bin = self.opt['bins']
        
        edge_strictness=self.opt['edge_strictness']
        extrapolate_threshold=self.opt['extrapolate_threshold']
        
        tomo_bin = 0*mu
        
        for i in range(len(mu)):
            
            tomo_bin[i]=-1
            
            for j in range(n_bin):
                
                if mu[i]-edge_strictness*sigma[i]**0.5>z_edges[j] and mu[i]+edge_strictness*sigma[i]**0.5<z_edges[j+1]:
                    tomo_bin[i]=j
                    
                if modelV[i]>extrapolate_threshold*sigma[i]:
                    tomo_bin[i]=-1
                    
            
        
        

        return tomo_bin

