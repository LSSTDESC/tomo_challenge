"""
Trivial classifiers. Used as example and testing.

Every classifier module needs to:
 - have construction of the type 
       __init__ (self, bands, options) (see examples below)
 -  implement two functions: 
        train (self, training_data,training_z)
        apply (self, data).
 - define valid_options class varible.

See Classifier Documentation below.


"""
import numpy as np
from .base import Tomographer

class MineCraft(Tomographer):
    """Completely random classifier. 

    Every object goes into a random bin. 
    """

    ## see constructor below
    valid_options = ['blocks','zbins']
    
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
            'bins' - number of blocks per in direction

        """
        self.opt = {'blocks':40, 'zbins':10}
        self.bands = bands
        self.Nb = len(bands)
        self.opt.update(options)


    def train (self,training_data, training_z):
        """Trains the classifier
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """
        #data = np.vstack([training_data[band] for band in self.bands]).T
        bins=[]
        self.bmin,self.bmax = [],[]
        for band in self.bands:
            da=training_data[band]
            mn,mx  = da.min(), da.max()*1.00001
            self.bmin.append(mn)
            self.bmax.append(mx)
            step = (mx-mn)/self.opt['blocks']
            bins.append(((da-mn)/step).astype(int))
        bins = np.array(bins).T
        print (bins[:4])
        zmin,zmax = training_z.min(), training_z.max()
        dz= (zmax-zmin)/self.opt['zbins']
        self.target = {}
        for i in range(self.opt['zbins']):
            czmin = zmin+i*dz
            czmax = czmin+dz
            print (czmin,czmax)
            w = np.where((training_z>czmin) & (training_z<=czmax))
            print (bins.shape)
            xbins = bins[w[0],:]
            xbins = set([tuple(b) for b in xbins])
            ## now that we have unique elements, let's add
            for bin in xbins:
                if bin in self.target:
                    self.target[bin].append(i)
                else:
                    self.target[bin] = [i]
        ## now reorganize
        for key in self.target.keys():
            self.target[key] = tuple(self.target[key])#[# key])
            # if len(l)<6:
            #     self.target[key] = tuple(self.target[key])
            # else:
            #     self.target[key] = ()
        self.idmap={}
        for binid,vals in enumerate(set(self.target.values())):
            self.idmap[vals]=binid
            print ('id = ',binid,':',vals)
        self.idmap[()] = len(self.target.keys())  #last +1
        print (self.idmap)
        #self.apply(training_data)
        #stop()
        
        
    def apply (self,data):
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
        bins=[]
        for band,mn,mx in zip(self.bands,self.bmin,self.bmax):
            da=data[band]
            step = (mx-mn)/self.opt['blocks']
            bins.append(((da-mn)/step).astype(int))
        bins = [tuple(b) for b in np.array(bins).T]
        print (bins[:4])
        tomo_bin = np.array([self.idmap[dict.get(self.target,b,())] for b in bins])
        No = len(tomo_bin)
        binc=np.bincount(tomo_bin)
        print ('bincount before=',binc/No)
        minobj = int(No*0.03)
        ## now combined all the samples that are less than 5%:
        w=np.where(binc<minobj)[0]
        for j in w[1:]:
            tomo_bin [tomo_bin==j] = w[0]
        ## now reindex, again, to get rid of intermediate zeros:
        for i,j in enumerate(sorted(set(tomo_bin))):
            if (i!=j):
                tomo_bin[tomo_bin==j]=i
            
        binc=np.bincount(tomo_bin)
        ## now get rid of zeros:
        
        print ('bincount after=',binc/No)
        return tomo_bin
