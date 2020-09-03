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
from copy import deepcopy
from random import choices
import pyccl
from .base import Tomographer
from tomo_challenge import compute_scores, compute_mean_covariance

class SummerSlasher(Tomographer):
    """
     There is some magic here.
    """

    ## see constructor below
    valid_options = ['n_cuts','seed', 'pop_size', 'children','target_metric','outroot',
                     'letdie', 'downsample']
    
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
            'seed' - random number seed (passed to numpy.random) 

        """
        self.opt = { 'seed':123,'n_cuts':3, 'pop_size':100, 'children':100,
                     'target_metric':'gg','outroot':'cuter', 'letdie':False,
                     'downsample':2}
        self.opt.update(options)
        self.bands = bands
        self.n_cuts=self.opt['n_cuts']
        self.nbins = 2**self.n_cuts
        self.Nd = len(bands)
        
        
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
        ## first let's get train data into a nice array
        data = np.vstack([training_data[band] for band in self.bands]).T
        downsample = self.opt['downsample']
        data=data[::downsample,:]
        training_z = training_z[::downsample]
        print ("initializing")
        self.pop = [Astronomer(self.opt['n_cuts'],data, id=i) for i in range(self.opt['pop_size'])]
        print ("getting initial scores")
        for cuter in self.pop:
            cuter.get_score(training_z,self.opt['target_metric'])
        self.pop.sort(key = lambda x:x.score, reverse=True)
        self.sof = open (self.opt['outroot']+'_scores.txt','w')
        self.som = open (self.opt['outroot']+'_mut.txt','w')
        self.write_status()
        ## now the main genetic algorithgm:
        NP = self.opt['pop_size']
        while True:
            children=[]
            for nc in range(self.opt['children']):
                i,j=np.random.randint(0,NP,2)
                children.append (Astronomer(self.opt['n_cuts'],data,
                                         mate = [self.pop[i],
                                                 self.pop[j]]))
            ## we will parallelize this later
            for cuter in children:
                cuter.get_score(training_z,self.opt['target_metric'])



            ## add children to population
            if self.opt['letdie']:
                self.pop = children
            else:
                self.pop += children
            ### sort by score and take top 
            self.pop.sort(key = lambda x:x.score, reverse=True)
            print ("Scores:")
            for s in self.pop:
                print ("    %f : %i, %s "%(s.score,s.actual_bins,s.id))
            self.pop = self.pop[:NP]
            self.write_status()

        

    def write_status(self):
        for s in self.pop:
            self.sof.write("%f "%s.score)
            self.som.write("%f "%s.Pmutate)
        self.sof.write("\n")
        self.sof.flush()
        self.som.write("\n")
        self.som.flush()







    #BELOW is what a fast calculation would do
        # z_min = training_z.min()
        # finesels = ((training_z-z_min)/0.01).astype(int)
        # mu_fine, C_fine, galaxy_galaxy_tracer_bias = compute_mean_covariance(
        #         finesels, training_z, 'ww')
        # P_fine = np.linalg.inc(C_fine)
        # ## now express sels as finesels
        # T=[]
        # for bin in range(sels.nbins):
        #     x=np.bincount(((training_z[sels==bin]-z_min)/0.01).astype(int),minlength=len(finesels))
        #     T.append(x/finesels)
        # T=np.array(T)
        # mymu = (T @ mu_fine)
        # myP = (T.T @ P_fine @ T)
        # myscore = (mymu.T @ myP @ mymu)**0.5
        # print ('myscore = ',myscore)



        
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

        np.random.seed(self.opt['seed'])
        nbins = self.opt["bins"]
        tomo_bin = np.random.randint(0,nbins,len(data))
        return tomo_bin


class Cut:
    def __init__ (self, train_data):
        Ns = train_data.shape[1]
        self.w = np.random.uniform(-1,1,Ns)
        self.C = np.median((train_data*self.w).sum(axis=1))

        
    def apply (self,data):
        return (((data*self.w).sum(axis=1)-self.C)>0).astype(int)
    
    def mutate (self):
        self.w *=np.random.normal(1,0.003,len(self.w))
        self.C *=np.random.normal(1,0.001)
        
class Astronomer:
    def __init__ (self, n_cuts, train_data, mate = None, id = None):
        self.data = train_data
        self.Nm = 2**n_cuts
        self.n_cuts = n_cuts
        if mate:
            self.rompy_pompy(*mate)
        else:
            self.cuts = [Cut(self.data) for i in range(n_cuts)]
            self.Pmutate = np.random.normal(0.3,0.1)
            self.id = set([id])
            
    def rompy_pompy(self, parentA, parentB):
        self.Pmutate=1.0#np.sqrt(parentA.Pmutate*parentB.Pmutate)*np.random.normal(1.0,0.05)
        ## we make a copy of genes!
        self.cuts=deepcopy(choices(parentA.cuts+parentB.cuts,k=self.n_cuts))
        for i in np.where(np.random.uniform(0,1,self.n_cuts)<self.Pmutate)[0]:
            self.cuts[i].mutate()
        self.id = parentA.id.union(parentB.id)
         
    def get_selections(self):
        i=1
        sel = np.zeros(self.data.shape[0],int)
        for cut in self.cuts:
            sel+=cut.apply(self.data)*i
            i*=2
        ### now see if any bin is empty and fix this.
        bc=np.bincount(sel)
        j=0
        for i, v in enumerate(bc):
            if v==0:
                sel[sel>j]-=1
            else:
                j+=1
        self.actual_bins = j+1
        return sel

    def get_score(self,training_z, target_metric):
        sels = self.get_selections()
        #print (np.bincount(sels),'<<<<<<<<')
        try:
            mu, C, galaxy_galaxy_tracer_bias = compute_mean_covariance(
                sels, training_z, target_metric )
        except pyccl.errors.CCLError:
            print (np.bincount(sels),'<<<<<<<<')
            print ("BARF^^")
            self.score=-1
            #stop()
            return
        
        P = np.linalg.inv(C)
        self.score = (mu.T @ P @ mu)**0.5
        return self.score
