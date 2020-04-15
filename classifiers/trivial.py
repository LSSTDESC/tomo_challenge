"""
This is an example tomographic bin generator using a random forest.

Every classifier module needs to implement define _provides variable 
that list classifiers. 

See Classifier Documentation below
"""
import numpy as np

class Random:
    """ Completely random classifier. """
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
        tomo_bin = np.random.randint(0,nbins,len(data))
        return tomo_bin


class ILover:
    """Classifier for people who love nothing more than the i-band. 
       Classifies in uniform bins in the i-band mag. """
    # Which bands are valid to call this with
    _valid_bads = ["riz","griz"]
    # List of options we want to be called in competition, i.e. each dict
    # in the list is an "entry" into competition
    _opions = [{"bins":1},{"bins":2}, {"bins":10}]

    def __init__ (self, bands, options = {"bins":3}):
        self.indx = bands.find("i")
        self.opt = options

    def train (self,training_data, training_z):
        pass

    def apply (self,data):
        nbins = self.opt["bins"]
        iband = data[:,self.indx]
        tomo_bin = np.digitize(iband, np.linspace(iband.min(),iband.max(), nbins))
        tomo_bin -= 1 ## we need start with 0
        print(tomo_bin.min(), tomo_bin.max())
        return tomo_bin



