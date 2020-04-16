"""
This is an example tomographic bin generator using a random forest.

Every classifier module needs to implement two functions: 
 train (self, training_data,training_z)
and
 apply (self, data)

See Classifier Documentation below.
"""

import time
import sys
import code

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    """ Random Forest Classifier """
    # Which bands are valid to call this with
    _valid_bads = ["riz","griz"]
    # List of options we want to be called in competition, i.e. each dict
    # in the list is an "entry" into competition
    _opions = [{"bins":1},{"bins":2}, {"bins":3},{"bins":4},{"bins":5},{"bins":10}]

    def __init__ (self, bands, options = {"bins":3}):
        self.bands = bands
        self.opt = options

    def train (self, training_data, training_z):
        n_bin = self.opt['bins']
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

        # for speed, cut down to 5% of original size
        cut = np.random.uniform(0, 1, training_z.size) < 0.05
        training_bin = training_bin[cut]
        training_data = training_data[cut]

        # Can be replaced with any classifier
        classifier = RandomForestClassifier()

        t0 = time.perf_counter()
        # Lots of data, so this will take some time
        classifier.fit(training_data, training_bin)
        duration = time.perf_counter() - t0

        self.classifier = classifier
        self.z_edges = z_edges


    def apply (self, data):
        tomo_bin = self.classifier.predict(data)
        return tomo_bin

