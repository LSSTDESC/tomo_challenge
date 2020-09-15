#import tensorflow.keras as keras
import numpy as np

#from tomo_challenge.utils.utils import create_directory
import os
#from tomo_challenge.utils.utils import check_if_file_exits
import gc
import time
import sys


class Classifier_NNE:
    def create_classifier(self, model_name, input_shape, nb_classes, output_directory, verbose=False,
                          build=True, load_weights=False):
        if model_name == 'autolstm':
            from tomo_challenge.classifiers import autolstm_jax as autolstm 
            return autolstm.Classifier_LSTM(output_directory, input_shape, nb_classes, verbose, build=build)
        if model_name == 'resnet':
            from tomo_challenge.classifiers import resnet_jax as resnet 
            return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose,
                                            build=build, load_weights=load_weights)


    def __init__(self, output_directory, verbose=False):
        self.classifiers = ['autolstm', 'resnet']

        self.output_directory = output_directory
        create_directory(self.output_directory)
        self.verbose = verbose
        self.models_dir = output_directory.replace('ensemble_result','classifier')
        self.iterations = 1

    def predict(self):
        # no training since models are pre-trained
        start_time = time.time()

        l = 0
        # loop through all classifiers
        for model_name in self.classifiers:

            # loop through different initialization of classifiers

            curr_dir = self.models_dir.replace('classifier',model_name)
            print('curr_dir:' ,curr_dir)
            #model = self.create_classifier(model_name, None, None,
            #                               curr_dir, build=False)
            #print(model_name)

            predictions_file_name = curr_dir+'y_pred.npy'
   
            print(f"geting predict for {model_name}")
            curr_y_pred = np.load(predictions_file_name)

            if l == 0:
                y_pred = np.zeros(shape=curr_y_pred.shape)
            else:
                max_val = min(y_pred.shape[0], curr_y_pred.shape[0])
                print(max_val)
                y_pred = y_pred[:max_val]
                print(y_pred.shape)
                curr_y_pred = curr_y_pred[:max_val]
                print(curr_y_pred.shape)
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

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            
            return None 
        return directory_path

def check_if_file_exits(file_name):
    return os.path.exists(file_name)
