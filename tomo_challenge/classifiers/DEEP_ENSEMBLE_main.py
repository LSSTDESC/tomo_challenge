"""
Deep Ensemble Classifier
This is an example tomographic bin generator using a Deep Ensemble Classifier.
This solution was developed by the Brazilian Center for Physics Research AI 4 Astrophysics team.
Authors: Clecio R. Bom, Gabriel Teixeira, Bernardo M. Fraga, Eduardo Cypriano and Elizabeth Gonzalez.
contact: debom |at |cbpf| dot| br
In our preliminary tests we found a SNR 3X2 of  ~1930 for n=10 bins.
Every classifier module needs to:
 - have construction of the type
       __init__ (self, bands, options) (see examples below)
 -  implement two functions:
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.
"""

from tomo_challenge.utils.utils import transform_labels
from tomo_challenge.utils.utils import create_directory
from sklearn.preprocessing import MinMaxScaler
import sklearn
import os
from .base import Tomographer
import numpy as np
import sys



class ENSEMBLE(Tomographer):
    """ ENSEMBLE Classifier """

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
        dir_out: str
          directory for the outputs of training
        Note:
        -----
        Valid options are:
            'bins' - number of tomographic bins
        """

        self.bands = bands
        self.opt = options
        self.CLASSIFIERS = ['fcn', 'autolstm', 'resnet']

    def propare_data(self, training_data, traning_bin):

        x_train = training_data
        y_train = traning_bin

        nb_classes = len(np.unique(y_train))
        print(f'n_bins = {nb_classes}')


        # make the min to zero of labels
        y_train = transform_labels(y_train)

        # transform the labels from integers to one hot vectors
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(y_train.reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension and normalizing
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train, nb_classes, scaler

    def fit_classifier(self, classifier_name, x_train, y_train, nb_classes, output_directory,  load_weights=False):

        input_shape = x_train.shape[1:]

        classifier = self.create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                                       load_weights=load_weights)

        classifier.fit(x_train, y_train)

    def create_classifier(self, classifier_name, input_shape, nb_classes, output_directory, verbose=True,
                          build=True, load_weights=False):
        if classifier_name == 'fcn':
            from tomo_challenge.classifiers import de_fcn as fcn
            return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose, build=build)

        if classifier_name == 'autolstm':
            from tomo_challenge.classifiers import de_autolstm as autolstm 
            return autolstm.Classifier_LSTM(output_directory, input_shape, nb_classes, verbose, build=build)

        if classifier_name == 'resnet':
            from tomo_challenge.classifiers import de_resnet as resnet 
            return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose,
                                            build=build, load_weights=load_weights)

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

        #Create a directory for the results and the models
        path_parent = os.getcwd()
        dir_out = path_parent+'/ENSEMBLE'
        if not os.path.exists(dir_out):
            create_directory(dir_out)
        self.dir_out = dir_out



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

        x_train, y_train, nb_classes, scaler = self.propare_data(training_data=training_data, traning_bin=training_bin)
        print("Fitting classifier")
        for classifier_name in self.CLASSIFIERS:
            print('classifier_name', classifier_name)

            output_directory = self.dir_out +'/' + classifier_name + '/'
            create_directory(output_directory)

            print("Training")

            self.fit_classifier(classifier_name=classifier_name, x_train=x_train, y_train=y_train,
                           nb_classes=nb_classes, output_directory=output_directory)

            create_directory(output_directory + '/DONE')
            print('\t\t\t\tDONE')
        # Can be replaced with any classifier



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
        data = self.scaler.transform(data)
        if len(data.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension and normalizing
            data = data.reshape((data.shape[0], data.shape[1], 1))

        from tomo_challenge.classifiers import de_nne as nne
        result_dir = self.dir_out+'/ensemble_result/'
        create_directory(self.dir_out+'/ensemble_result/')
        ensemble = nne.Classifier_NNE(result_dir)

        preds = ensemble.predict(data)
        tomo_bin = np.argmax(preds, axis=1)
        return tomo_bin
