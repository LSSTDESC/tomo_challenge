"""
Deep Ensemble Classifier
This is an example tomographic bin generator using a Deep Ensemble Classifier with JAX/FLAX capable of optimizing to multiple losses including SNR/FOM
This solution was developed by the Brazilian Center for Physics Research AI 4 Astrophysics team.
Authors: Clecio R. Bom, Bernardo M. Fraga, Gabriel Teixeira, Eduardo Cypriano and Elizabeth Gonzalez.
contact: debom |at |cbpf| dot| br
Every classifier module needs to:
 - have construction of the type
       __init__ (self, bands, options) (see examples below)
 -  implement two functions:
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.
"""


#from tomo_challenge.utils.utils import create_directory
import os
from .base import Tomographer
import numpy as np
import subprocess




class ENSEMBLE(Tomographer):
    """ ENSEMBLE Classifier """

    # valid parameter -- see below
    valid_options = ['bins']
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
        dir_out: str
          directory for the outputs of training
        Note:
        -----
        Valid options are:
            'bins' - number of tomographic bins
        """

        self.bands = bands
        self.opt = options
        self.CLASSIFIERS = ['resnet', 'autolstm']

    def train (self, training_data, validation_data):
        """Trains the classifier

        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample
        """

        #Create a directory for the results and the models
        path_parent = os.getcwd()
        dir_out = path_parent+'/ENSEMBLE'
        if not os.path.exists(dir_out):
            create_directory(dir_out)
        self.dir_out = dir_out

        data_path = path_parent + '/data/'
        training_path = data_path + 'training.hdf5'
        validation_path = data_path + 'validation.hdf5'
        classifier_path = os.path.dirname(os.path.abspath(__file__))
        #print('path = ', classifier_path)
        n_bin = self.opt['bins']

        print("Fitting classifier")
        for classifier_name in self.CLASSIFIERS:
            print('classifier_name', classifier_name)

            output_directory = self.dir_out +'/' + classifier_name + '/'
            create_directory(output_directory)

            print("Training")
            print(os.getcwd())
            self.p = subprocess.Popen(['python3','-u', str(classifier_path) + '/' + classifier_name + '.py',
                              training_path,
                              validation_path,
                              output_directory,
                              str(n_bin)])
            subprocess.Popen.wait(self.p)

            create_directory(output_directory + '/DONE')
            print('\t\t\t\tDONE')
        # Can be replaced with any classifier


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

        from tomo_challenge.classifiers import nne_jax as nne
        result_dir = self.dir_out+'/ensemble_result/'
        create_directory(self.dir_out+'/ensemble_result/')
        ensemble = nne.Classifier_NNE(result_dir)

        preds = ensemble.predict()
        tomo_bin = np.argmax(preds, axis=1)
        return tomo_bin


def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            
            return None 
        return directory_path
