"""
This is an example tomographic bin generator using a random forest.

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
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
#JEC 15/7/20 use joblib to save the model
from joblib import dump, load


class myCombinedClassifiers(Tomographer):
    """ Multi Classifiers """
    
    # valid parameter -- see below
    # JEC 15/7/20 savefile opt to dump the model
    valid_options = ['bins', 'n_estimators', 'savefile']
    
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
        #JEC 
        n_estimators=self.opt['n_estimators']
        
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



        #JEC 23/7/2020
        #Resultat de la sepoaration a # de gal constant per bin
##         z_edges = np.array([0.02450252, 0.29473031, 0.44607811, 0.58123241,\
##                             0.70399809,0.8120476 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])

##         #Chge1 z_edges[1]
        
##         z_edges = np.array([0.02450252, 0.15961642, 0.44607811, 0.58123241,\
##                             0.70399809,0.8120476 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])

##         #Chge2 z_edges[2]

##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.58123241,\
##                             0.70399809,0.8120476 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])


##         #Chge2 z_edges[3]
##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
##                             0.70399809,0.8120476 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])



##         #Chge2 z_edges[4]
##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
##                             0.58955, 0.8120476 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])



##         #Chge2 z_edges[5]
##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
##                             0.58955, 0.700775 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])


##         #Chge2 z_edges[6] apres des essais +/- pas d'amlioration 
##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
##                             0.58955, 0.700775 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])


##         #Chge2 z_edges[7] apres des essais +/- pas d'amlioration 
##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
##                             0.58955, 0.700775 , 0.91746771, 1.03793607,\
##                             1.20601892, 1.49989052, 3.03609133])

##         #Chge2 z_edges[8]
##         z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
##                             0.58955, 0.700775 , 0.91746771, 1.03793607,\
##                             1.352945, 1.49989052, 3.03609133])

        #Chge2 z_edges[9]
        z_edges = np.array([0.02450252, 0.15961642, 0.37035, 0.475175,\
                            0.58955, 0.700775 , 0.91746771, 1.03793607,\
                            1.352945, 1.8, 3.03609133])




        print("new set: ",z_edges)



        # Now find all the objects in each of these bins
        for i in range(n_bin):
            z_low = z_edges[i]
            z_high = z_edges[i + 1]
            training_bin[(training_z > z_low) & (training_z < z_high)] = i

        # for speed, cut down to 5% of original size
        cut = np.random.uniform(0, 1, training_z.size) < 0.05
        training_bin = training_bin[cut]
        training_data = training_data[cut]

        # Can be replaced with any classifier
        
        
        estimators = [ ('gd', make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=n_estimators, verbose=1))),
                        ('rf', make_pipeline(StandardScaler(),
                                             RandomForestClassifier(n_estimators=n_estimators,verbose=1)))  ]
        classifier = StackingClassifier(estimators=estimators,
                                        final_estimator=LogisticRegression(max_iter=5000))

        print("Fitting classifier")
        # Lots of data, so this will take some time
        classifier.fit(training_data, training_bin)

        self.classifier = classifier
        self.z_edges = z_edges

        #JEC 15/7/20 dump clf
        dump(classifier, self.opt['savefile'])

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
        tomo_bin = self.classifier.predict(data)
        return tomo_bin

