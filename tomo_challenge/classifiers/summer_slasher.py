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
import pickle
import pyccl
import sys
from .base import Tomographer
from tomo_challenge import compute_scores, compute_mean_covariance

# try:
# import mpi4py
# from mpi4py import MPI

have_mpi = False

# except:
#     have_mpi = False
#     print ("No MPI available.")

def task(el, training_z):
    el.get_score(training_z, self.opt["target_metric"])
    return el

class SummerSlasher(Tomographer):
    """
     There is some magic here.
    """

    ## see constructor below
    valid_options = [
        "n_cuts",
        "seed",
        "pop_size",
        "children",
        "target_metric",
        "outroot",
        "letdie",
        "downsample",
        "apply_pickled",
        "processes",
    ]

    def __init__(self, bands, options):
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
        self.opt = {
            "seed": 123,
            "n_cuts": 3,
            "pop_size": 100,
            "children": 100,
            "target_metric": "gg",
            "outroot": "cuter",
            "letdie": False,
            "downsample": 2,
            "processes": 0,
        }
        self.opt.update(options)
        self.bands = bands
        self.n_cuts = self.opt["n_cuts"]
        self.nbins = 2 ** self.n_cuts
        self.Nd = len(bands)

    def get_scores(self, tlist, training_z):
        if have_mpi and (self.mpi_size > 0):
            ## first broadcast everything
            # print ("here",self.mpi_rank)
            tlist = self.mpi_comm.bcast(tlist, root=0)
            # print ("here",self.mpi_rank)

            for i, el in enumerate(tlist):
                if i % self.mpi_size == self.mpi_rank:
                    el.get_score(training_z, self.opt["target_metric"])
                    # print (self.mpi_rank,el.score)
            # sys.stdout.flush()
            # self.mpi_comm.Barrier()

            for i, el in enumerate(tlist):
                k = i % self.mpi_size
                # print('XXX',self.mpi_rank,i,k)
                # sys.stdout.flush()
                if k > 0:
                    if (self.mpi_rank > 0) and (self.mpi_rank == k):
                        # print (self.mpi_rank,'sending')
                        # sys.stdout.flush()
                        self.mpi_comm.send(el, dest=0)
                    elif self.mpi_rank == 0:
                        # print (self.mpi_rank,'receiving',k)
                        # sys.stdout.flush()
                        tlist[i] = self.mpi_comm.recv(source=k)

            # print (self.mpi_rank,'at barrier')
            self.mpi_comm.Barrier()
            # if (self.mpi_rank==0):
            #    print ([t.score for t in tlist])
            #    sys.stdout.flush()
        elif self.opt['processes'] > 1:
            import multiprocessing
            with multiprocessing.Pool(self.opt['processes']) as pool:
                args = [(el, training_z) for el in tlist]
                tlist = pool.map(task, args)
        else:
            for cuter in tlist:
                cuter.get_score(training_z, self.opt["target_metric"])

        return tlist

    def train(self, training_data, training_z):
        """Trains the classifier
        
        Parameters:
        -----------
        training_data: numpy array, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """

        self.bestfn = self.opt["outroot"] + "_best.pickle"
        if self.opt["apply_pickled"]:
            return

        if have_mpi:
            self.mpi_comm = MPI.COMM_WORLD
            self.mpi_size = self.mpi_comm.Get_size()
            self.mpi_rank = self.mpi_comm.Get_rank()
            print("mpi rank = ", self.mpi_rank, "/", self.mpi_size)
        else:
            self.mpi_rank = 0

        ## first let's get train data into a nice array
        data = np.vstack([training_data[band] for band in self.bands]).T
        downsample = self.opt["downsample"]
        data = data[::downsample, :]
        training_z = training_z[::downsample]
        print("initializing")
        self.pop = [
            Astronomer(self.opt["n_cuts"], data, id=i)
            for i in range(self.opt["pop_size"])
        ]
        print("getting initial scores")
        self.pop = self.get_scores(self.pop, training_z)
        if self.mpi_rank == 0:
            self.pop.sort(key=lambda x: x.score, reverse=True)
            self.sof = open(self.opt["outroot"] + "_scores.txt", "w")
            self.som = open(self.opt["outroot"] + "_mut.txt", "w")
            self.write_status()
        ## now the main genetic algorithgm:
        NP = self.opt["pop_size"]
        while True:
            children = []
            for nc in range(self.opt["children"]):
                i, j = np.random.randint(0, NP, 2)
                children.append(
                    Astronomer(
                        self.opt["n_cuts"], data, mate=[self.pop[i], self.pop[j]]
                    )
                )
            ## we will parallelize this later
            children = self.get_scores(children, training_z)

            if self.mpi_rank == 0:
                ## add children to population
                if self.opt["letdie"]:
                    self.pop = children
                else:
                    self.pop += children
                ### sort by score and take top
                self.pop.sort(key=lambda x: x.score, reverse=True)
                print("Scores:")
                for s in self.pop:
                    print("    %f : %i, %s " % (s.score, s.actual_bins, s.id))
                self.pop = self.pop[:NP]
                self.write_status()
            sys.stdout.flush()

    def write_status(self):
        for s in self.pop:
            self.sof.write("%f " % s.score)
            self.som.write("%f " % s.Pmutate)
        self.sof.write("\n")
        self.sof.flush()
        self.som.write("\n")
        self.som.flush()
        bestfn = "%s_%g_best.pickle"%(self.opt["outroot"],self.pop[0].score)
        pickle.dump(self.pop[0], open(bestfn, "wb"))

    def apply(self, data):
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

        astronomer = pickle.load(open(self.bestfn,'rb'))
        data = np.vstack([data[band] for band in self.bands]).T
        astronomer.set_data(data)
        return astronomer.get_selections()


class Cut:
    def __init__(self, train_data):
        Ns = train_data.shape[1]
        self.w = np.random.uniform(-1, 1, Ns)
        self.C = np.median((train_data * self.w).sum(axis=1))

    def apply(self, data):
        return (((data * self.w).sum(axis=1) - self.C) > 0).astype(int)

    def mutate(self):
        self.w *= np.random.normal(1, 0.003, len(self.w))
        self.C *= np.random.normal(1, 0.001)


class Astronomer:
    def __init__(self, n_cuts, train_data, mate=None, id=None):
        self.data = train_data
        self.Nm = 2 ** n_cuts
        self.n_cuts = n_cuts
        if mate:
            self.rompy_pompy(*mate)
        else:
            self.cuts = [Cut(self.data) for i in range(n_cuts)]
            self.Pmutate = np.random.normal(0.3, 0.1)
            self.id = set([id])

    def set_data(self,data):
        self.data = data
            
    def rompy_pompy(self, parentA, parentB):
        self.Pmutate = (
            1.0  # np.sqrt(parentA.Pmutate*parentB.Pmutate)*np.random.normal(1.0,0.05)
        )
        ## we make a copy of genes!
        self.cuts = deepcopy(choices(parentA.cuts + parentB.cuts, k=self.n_cuts))
        for i in np.where(np.random.uniform(0, 1, self.n_cuts) < self.Pmutate)[0]:
            self.cuts[i].mutate()
        self.id = parentA.id.union(parentB.id)

    def get_selections(self):
        i = 1
        sel = np.zeros(self.data.shape[0], int)
        for cut in self.cuts:
            sel += cut.apply(self.data) * i
            i *= 2
        ### now see if any bin is empty and fix this.
        bc = np.bincount(sel)
        j = 0
        for i, v in enumerate(bc):
            if v == 0:
                sel[sel > j] -= 1
            else:
                j += 1
        self.actual_bins = j + 1
        return sel

    def get_score(self, training_z, target_metric):
        sels = self.get_selections()
        # print (np.bincount(sels),'<<<<<<<<')
        try:
            mu, C, galaxy_galaxy_tracer_bias = compute_mean_covariance(
                sels, training_z, target_metric
            )
        except pyccl.errors.CCLError:
            print(np.bincount(sels), "<<<<<<<<")
            print("BARF^^")
            self.score = -1
            # stop()
            return

        P = np.linalg.inv(C)
        self.score = (mu.T @ P @ mu) ** 0.5
        return self.score
