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

# import pickle 
# file_pi = open('model.obj', 'r') 
# model = pickle.load(file_pi)

# X_test = np.genfromtxt('test_data.csv')
# n_test,d = X_test.shape


# ########### NOTE ###########
# # you can train the model gain, eve using different data, by executing:
# # model.train(model,X,Y,options)

# # use the model to generate predictions for the test set
# data=model.predict(X_test[:,:].copy())

# predictions=np.zeros([n_test,4])

# for i in range(n_test):
#     predictions[i,0]=data[0][i][0] #mu
#     predictions[i,1]=data[1][i][0] #variance
#     predictions[i,2]=data[2][i][0] #modelVariance
#     predictions[i,3]=data[3][i][0] #noiseVariance


# np.savetxt('prediction_data.csv',predictions)

# print 'predictions made'

def predict(model, X_test):
    n_test,d = X_test.shape

    ########### NOTE ###########
    # you can train the model gain, eve using different data, by executing:
    # model.train(model,X,Y,options)

    # use the model to generate predictions for the test set
    data=model.predict(X_test[:,:].copy())

    predictions=np.zeros([n_test,4])

    for i in range(n_test):
        predictions[i,0]=data[0][i][0] #mu
        predictions[i,1]=data[1][i][0] #variance
        predictions[i,2]=data[2][i][0] #modelVariance
        predictions[i,3]=data[3][i][0] #noiseVariance

    print('predictions made')

    return predictions

