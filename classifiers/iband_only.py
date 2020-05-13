import numpy as np

class IBandOnly:
    """Classifier for people who love nothing more than the i-band.
       Classifies in uniform bins in the i-band mag. """

    valid_options = ['bins']

    def __init__ (self, bands, options):
        self.indx = bands.find('i')
        self.opt = options
        if 'bins' not in options:
            self.opt['bins'] = 3

    def train (self,training_data, training_z):
        pass

    def apply (self,data):
        nbins = self.opt["bins"]
        iband = data[:, self.indx]
        tomo_bin = np.digitize(iband, np.linspace(iband.min()-1e-6, iband.max()+1e-6, nbins))
        tomo_bin -= 1 ## we need start with 0
        return tomo_bin
