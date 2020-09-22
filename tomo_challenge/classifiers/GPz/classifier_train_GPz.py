from . import GPz
from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

import h5py


########### Model options ###############

method = 'VC'               # select method, options = GL, VL, GD, VD, GC and VC [required]
m = 100                      # number of basis functions to use [required]
joint = True                # jointly learn a prior linear mean function [default=true]
heteroscedastic = True      # learn a heteroscedastic noise process, set to false interested only in point estimates
csl_method = 'normal'       # cost-sensitive learning option: [default='normal']
                            #       'balanced':     to weigh rare samples more heavly during train
                            #       'normalized':   assigns an error cost for each sample = 1/(z+1)
                            #       'normal':       no weights assigned, all samples are equally important
                            #
binWidth = 0.1              # the width of the bin for 'balanced' cost-sensitive learning [default=range(z_spec)/100]
decorrelate = True          # preprocess the data using PCA [default=False]

########### Training options ###########

maxIter = 500                  # maximum number of iterations [default=200]
maxAttempts = 50              # maximum iterations to attempt if there is no progress on the validation set [default=infinity]
trainSplit = 0.5               # percentage of data to use for training
validSplit = 0.5               # percentage of data to use for validation
testSplit  = 0.0               # percentage of data to use for testing

########### Start of script ###########



# X_train = np.genfromtxt('train_data.csv')
# n_train,d = X_train.shape
# Y_train = np.genfromtxt('training_z.csv')
# Y_train = Y_train.reshape(-1,1)




# # sample training, validation and testing sets from the data
# training,validation,testing = GPz.sample(n_train,trainSplit,validSplit,testSplit)


# # get the weights for cost-sensitive learning
# omega = GPz.getOmega(Y_train, method=csl_method)


# # initialize the initial model
# model = GPz.GP(m,method=method,joint=joint,heteroscedastic=heteroscedastic,decorrelate=decorrelate)

# # train the model
# model.train(X_train.copy(), Y_train.copy(), omega=omega, training=training, validation=validation, maxIter=maxIter, maxAttempts=maxAttempts)


# import pickle 
# file_pi = open('model.obj', 'w') 
# pickle.dump(model, file_pi)


def train(X_train, Y_train):
    n_train,d = X_train.shape
    Y_train = Y_train.reshape(-1,1)

    # sample training, validation and testing sets from the data
    training,validation,testing = GPz.sample(n_train,trainSplit,validSplit,testSplit)

    # get the weights for cost-sensitive learning
    omega = GPz.getOmega(Y_train, method=csl_method)

    # initialize the initial model
    model = GPz.GP(m,method=method,joint=joint,heteroscedastic=heteroscedastic,decorrelate=decorrelate)

    # train the model
    model.train(X_train.copy(), Y_train.copy(), omega=omega, training=training, validation=validation, maxIter=maxIter, maxAttempts=maxAttempts)

    return model
