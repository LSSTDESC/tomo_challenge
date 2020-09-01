from .base import Tomographer
import numpy as np


class RandomForest(Tomographer):
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
        pass

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
        pass
