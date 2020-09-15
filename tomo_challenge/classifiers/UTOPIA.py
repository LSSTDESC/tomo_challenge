"""
This is a classifier based on k-nearest neighbour matching.
Angus H Wright's UTOPIA: Useless Technique that OutPerforms Intelligent Alternatives

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
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpack
from rpy2.robjects.vectors import StrVector, IntVector, DataFrame, FloatVector, FactorVector
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

#Check that all the needed packages are installed
# R package nameo
packnames = ('RANN','RANN')
base=ro.packages.importr("base")
utils=ro.packages.importr("utils")
stats=ro.packages.importr("stats")
gr=ro.packages.importr("graphics")
dev=ro.packages.importr("grDevices")
utils.chooseCRANmirror(ind=1)
# Selectively install what needs to be installed.
names_to_install = [x for x in packnames if not rpack.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
rann=ro.packages.importr("RANN")

class UTOPIA(Tomographer):
    """ Simple kNN Classifier """
    
    # valid parameter -- see below
    valid_options = ['bins','sparse_frac']
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = False
    
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
            'sparse_frac' - the sparse-sampling fraction used for training

        """
        self.bands = bands
        self.opt = options

    def train (self, training_data, training_z):
        """Trains the SOM and outputs the resulting bins
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """

        print("Initialising")
        #Number of tomographic bins 
        n_bin = self.opt['bins']
        #Sparse Frac 
        sparse_frac = self.opt['sparse_frac'] 

        #Define the SOM variables
        if self.bands == 'riz':
            #riz bands
            expressions = ("r-i","r-z","i-z",
                           "z","r-i-(i-z)")
        elif self.bands == 'griz':
            #griz bands
            expressions = ("g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","g-r-(r-i)",
                           "r-i-(i-z)")
        elif self.bands == 'grizy':
            #grizy bands
            expressions = ("g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")
        elif self.bands == 'ugriz':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","g-r","g-i",
                           "g-z","r-i","r-z","i-z",
                           "z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)")
        elif self.bands == 'ugrizy':
            #ugrizy bands
            expressions = ("u-g","u-r","u-i","u-z","u-y","g-r","g-i",
                           "g-z","g-y","r-i","r-z","r-y","i-z","i-y",
                           "z-y","z","u-g-(g-r)","g-r-(r-i)",
                           "r-i-(i-z)","i-z-(z-y)")

        print("Preparing the data")
        training_data = pd.DataFrame.from_dict(training_data)

        if sparse_frac < 1:
            print("Sparse Sampling the training data")
            cut = np.random.uniform(0, 1, training_z.size) < sparse_frac
            training_data = training_data[cut]
            training_z = training_z[cut]

        #Construct the training data frame (just a python-to-R data conversion)
        print("Converting the data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              train_df = ro.conversion.py2rpy(training_data)

        print("Outputting training data")
        self.train_df = train_df
        self.train_z = FloatVector(training_z)

    def apply (self, data):
        """Matched the validation data to the training data.
        
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
        
        #Number of tomographic bins 
        n_bin = self.opt['bins']
        #Training photometry
        train_df = self.train_df
        #Training z 
        train_z = self.train_z

        print("Preparing the data")
        valid_data = pd.DataFrame.from_dict(data)

        #Construct the validation data frame (just a python-to-R data conversion)
        print("Converting the validation data to R format")
        with localconverter(ro.default_converter + pandas2ri.converter):
              valid_df = ro.conversion.py2rpy(valid_data)

        print("Matching the validation data to the training data")
        match = rann.nn2(train_df,valid_df,k=1)

        print("Assign matching redshift to validation sources")
        valid_z = np.array(train_z.rx(match.rx2['nn.idx']))

        print("Assign the validation data to tomographic bins")
        # Find the edges that split the redshifts into n_z bins of
        # equal number counts in each
        p = np.linspace(0, 100, n_bin + 1)
        z_edges = np.percentile(valid_z, p)

        # Now find all the objects in each of these bins
        valid_bin = np.zeros(valid_z.size)
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            valid_bin[(valid_z > z_low) & (valid_z < z_high)] = i

        #Assign the sources, by group, to tomographic bins
        print("Output source tomographic bin assignments")

        return valid_bin

