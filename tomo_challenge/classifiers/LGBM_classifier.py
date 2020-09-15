"""
LGBM Classifier
This is an example tomographic bin generator using a Light Gradient Boosted Machine as Classifier.
This solution was developed by the Brazilian Center for Physics Research AI 4 Astrophysics team.
Authors: Clecio R. Bom, Gabriel Teixeira, Bernardo M. Fraga, Eduardo Cypriano.
contact: debom |at |cbpf| dot| br
In our preliminary tests we found a SNR 3X2 of  ~1926 for n=10 bins.
Every classifier module needs to:
 - have construction of the type
       __init__ (self, bands, options) (see examples below)
 -  implement two functions:
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.
"""

from .base import Tomographer
import numpy as np
from lightgbm import LGBMClassifier


class LGBM(Tomographer):
    """ Light Gradient Boosted Machine Classifier """

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

        # For speed, it's possible cut down a percentage of original size
        #cut = np.random.uniform(0, 1, training_z.size) < 0.05
        #training_data = training_data[cut]
        #training_bin = training_bin[cut]


        model = LGBMClassifier
        # Can be replaced with any classifier


        print("Fitting classifier")
        # Lots of data, so this will take some time
        model.fit(training_data, training_bin)

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


        preds = self.classifier.predict(data)
        tomo_bin = np.argmax(preds, axis=1)
        return tomo_bin
