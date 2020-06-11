import numpy as np
from .base import Tomographer

class IBandOnly(Tomographer):
    """Classifier for people who love nothing more than the i-band. 
       Classifies in uniform bins in the i-band mag. """

    valid_options = ['bins']

    def __init__ (self, bands, options):
        self.opt = options
        if 'bins' not in options:
            self.opt['bins'] = 3
        
    def train (self,training_data, training_z):
        pass

    def apply(self, data):
        iband = data['i']
        nbin = self.opt['bins']
        # edges should be just beyond the limits of the data
        tomo_bin = np.digitize(iband, np.linspace(iband.min()-1e-6, iband.max()+1e-6, nbin))
        tomo_bin -= 1 # we need start with 0; digitize starts at 1
        return tomo_bin
