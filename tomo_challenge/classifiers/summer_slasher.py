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
import pyccl
from .base import Tomographer
from tomo_challenge import compute_scores, compute_mean_covariance

class SummerSlasher(Tomographer):
    """
     There is some magic here.
    """

    ## see constructor below
    valid_options = ['n_slashes','seed', 'pop_size', 'children','target_metric','outroot',
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
        self.opt = { 'seed':123,'n_slashes':3, 'pop_size':100, 'children':100,
                     'target_metric':'gg','outroot':'slasher', 'letdie':False,
                     'downsample':2}
        self.opt.update(options)
        self.bands = bands
        self.n_slashes=self.opt['n_slashes']
        self.nbins = 2**self.n_slashes
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
        self.pop = [Slasher(self.opt['n_slashes'],data, id=i) for i in range(self.opt['pop_size'])]
        print ("getting initial scores")
        for slasher in self.pop:
            slasher.get_score(training_z,self.opt['target_metric'])
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
                children.append (Slasher(self.opt['n_slashes'],data,
                                         mate = [self.pop[i],
                                                 self.pop[j]]))
            ## we will parallelize this later
            for slasher in children:
                slasher.get_score(training_z,self.opt['target_metric'])



            ## add children to population
            if self.opt['letdie']:
                self.pop = children
            else:
                self.pop += children
            ### sort by score and take top 
            self.pop.sort(key = lambda x:x.score, reverse=True)
            print ("Scores:")
            for s in self.pop:
                print ("    %f : %s "%(s.score,s.id))
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


class Slash:
    def __init__ (self, train_data):
        Ns = train_data.shape[1]
        self.w = np.random.uniform(-1,1,Ns)
        self.C = np.median((train_data*self.w).sum(axis=1))

        
    def apply (self,data):
        return (((data*self.w).sum(axis=1)-self.C)>0).astype(int)
    
    def mutate (self):
        self.w *=np.random.normal(1,0.01,len(self.w))
        self.C *=np.random.normal(1,0.01)
        
class Slasher:
    def __init__ (self, n_slashes, train_data, mate = None, id = None):
        self.data = train_data
        self.Nm = 2**n_slashes
        if mate:
            self.rompy_pompy(*mate)
        else:
            self.slashes = [Slash(self.data) for i in range(n_slashes)]
            self.Pmutate = np.random.normal(0.3,0.1)
            self.id = str(id)
            
    def rompy_pompy(self, parentA, parentB):
        self.Pmutate=np.sqrt(parentA.Pmutate*parentB.Pmutate)*np.random.normal(1.0,0.05)
        self.slashes=[]
        for slashA,slashB in zip(parentA.slashes, parentB.slashes):
            slash = slashA if  np.random.randint(2) else slashB
            if np.random.uniform(0,1)<self.Pmutate:
                #print ('---------A')
                #print (np.sum(slash.apply(self.data))/len(self.data))
                slash.mutate()
                #print(np.sum(slash.apply(self.data))/len(self.data))
                #print ('---------B')
            self.slashes.append(slash)
        self.id = "(%s,%s)"%(parentA.id, parentB.id)
         
    def get_selections(self):
        i=1
        sel = np.zeros(self.data.shape[0],int)
        for slash in self.slashes:
            sel+=slash.apply(self.data)*i
            i*=2
        ### now see if any bin is empty and fix this.
        bc=np.bincount(sel)
        j=0
        for i, v in enumerate(bc):
            if v==0:
                sel[sel>j]-=1
            else:
                j+=1
        
        return sel

    def get_score(self,training_z, target_metric):
        sels = self.get_selections()
        #print (np.bincount(sels),'<<<<<<<<')
        try:
            mu, C, galaxy_galaxy_tracer_bias = compute_mean_covariance(
                sels, training_z, target_metric )
        except pyccl.errors.CCLError:
            #print ("BARF^^")
            self.score=-1
            return
        
        P = np.linalg.inv(C)
        self.score = (mu.T @ P @ mu)**0.5
        return self.score
