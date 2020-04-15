"""
This is an example tomographic bin generator using a random forest.

Every classifier module needs to implement define _provides variable 
that list classifiers. 

See Classifier Documentation below
"""
import numpy as np

class Random:
    # Which bands are valid to call this with
    _valid_bads = ["riz","griz"]
    # List of options we want to be called in competition, i.e. each dict
    # in the list is an "entry" into competition
    _opions = [{"bins":1},{"bins":2}, {"bins":10}]

    def __init__ (self, bands, options = {"bins":3}):
        self.opt = options

    def train (self,training_data, training_z):
        pass

    def apply (self,data):
        nbins = self.opt["bins"]
        tomo_bin = np.random.uniform(0,nbins,len(data))
        return tomo_bin



