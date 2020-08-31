# NNE model for Deep Ensemble
import tensorflow.keras as keras
import numpy as np

from tomo_challenge.utils.utils import create_directory
from tomo_challenge.utils.utils import check_if_file_exits
import gc
import time
import sys


class Classifier_NNE:
    def create_classifier(self, model_name, input_shape, nb_classes, output_directory, verbose=False,
                          build=True, load_weights=False):
        if model_name == 'fcn':
            from tomo_challenge.classifiers import fcn
            return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'autolstm':
            from tomo_challenge.classifiers import autolstm
            return autolstm.Classifier_LSTM(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'resnet':
            from tomo_challenge.classifiers import resnet
            return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose,
                                            build=build, load_weights=load_weights)


    def __init__(self, output_directory, verbose=False):
        self.classifiers = ['fcn', 'autolstm', 'resnet']

        self.output_directory = output_directory
        create_directory(self.output_directory)
        self.verbose = verbose
        self.models_dir = output_directory.replace('ensemble_results','classifier')
        self.iterations = 1

    def predict(self, x_test):
        # no training since models are pre-trained
        start_time = time.time()

        l = 0
        # loop through all classifiers
        for model_name in self.classifiers:

            # loop through different initialization of classifiers

            curr_dir = self.models_dir.replace('classifier',model_name)

            model = self.create_classifier(model_name, None, None,
                                           curr_dir, build=False)
            print(model_name)

            predictions_file_name = curr_dir+'y_pred.npy'
   
            print(f"geting predict for {model_name}")
            curr_y_pred = model.predict(x_test=x_test)
            keras.backend.clear_session()

            np.save(predictions_file_name,curr_y_pred)

            if l == 0:
                y_pred = np.zeros(shape=curr_y_pred.shape)

            y_pred = y_pred+curr_y_pred
            l+=1


        # average predictions
        y_pred = y_pred / l

        # save predictiosn
        np.save(self.output_directory+'y_pred.npy', y_pred)

        duration = time.time() - start_time

        # the creation of this directory means
        create_directory(self.output_directory + '/DONE')

        return y_pred
