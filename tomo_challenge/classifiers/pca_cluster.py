"""
This is a binning method using principal component analysis
and the BAG algorithm outlined in binning_as_clustering.ipynb

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

import jax
import jax.numpy as jnp
from jax.core import UnexpectedTracerError
from jax import config
config.update("jax_enable_x64", True)

from tomo_challenge.jax_metrics import compute_snr_score, compute_fom

class PCACluster(Tomographer):
    """ PCA based clustering algorithm """
    
    # valid parameter -- see below
    valid_options = ["bins", "metric", "verbose"]
    # this settings means arrays will be sent to train and apply instead
    # of dictionaries
    wants_arrays = False
    
    def __init__ (self, bands, options):
        """Constructor
        
        Parameters:
        -----------
        bands: str
          string containg valid bands. PCACluster is designed for griz with colors and errors
        options: dict
          options come through here. Valid keys are listed as valid_options
          class variable. 

        Note:
        -----
        Valiad options are:
            "bins" - number of tomographic bins
            "metric" - the metric to optimize for, one of {"SNR", "FOM", "FOM_DETF"}
            "verbose" - Whether to print verbosely. This prints continual training updates. 

        """
        assert bands == "griz", "This method is designed to use griz with colors and errors!"
        self.bands = bands
        self.opt = options
        
        self.eigs = [] # To store eigen vectors from training to use during testing
        self.centroids = [] # Centroids for classification purposes.

    def train (self, training_data, training_z):
        """Trains the classifier
        
        Parameters:
        -----------
        training_data: dict, size Ngalaxes x Nbands
          training data, each row is a galaxy, each column is a band as per
          band defined above
        training_z: numpy array, size Ngalaxies
          true redshift for the training sample

        """
        impl = self.opt["metric"].lower()
        verbose = self.opt["verbose"]
        num_centroids = self.opt["bins"]
        
        # Gets the color data and the errors which we will use for weights.
        color_data = []
        for c in ["r", "gr", "ri", "rz"]:
            color_data.append(training_data[c])
        color_data = np.asarray(color_data).T
        errs = training_data["r_err"].reshape(-1, 1)
        
        # Converts the errors to weights by using 1/err^2
        # Since errors very close to 0 blow up, this code will set errors
        # below the threshold to 1 and scale everything s.t. the weights
        # are in the range (0, 1]
        err_thresh = 0.01
        err_cond = errs >= err_thresh
        weights = np.where(errs < err_thresh, 1, 1/errs**2)
        weights[err_cond] = weights[err_cond] / np.max(weights[err_cond])
        
        # Make the mean zeroish by subtracting weighted mean then find the covariance matrix
        # divde by len-1 because sample and not population covariance
        # Not that at this size this is going to matter though.
        color_shifted = color_data - np.average(color_data[:,0].reshape(-1, 1), weights=weights)
        cov = color_shifted.T @ color_shifted / (color_shifted.shape[0] - 1)
        
        # My own implementation of finding a PCA eigenvector with weights
        # using an EM-based algorithm
        def find_eigenvector(data, weights=None):
            # Start with position 1985. 1985 is the year my favourite
            # movie came out. No other reason than that. 
            phi = data[1985].reshape(1, -1)

            if weights is None:
                weights = np.ones_like(phi)

            thresh = 1e-6
            cond = False
            i = 0
            while not cond:
                # Find the coefficients that match the eigen vector to the data vector
                coeffs = data @ phi.T

                # Project the data along phi axis by multiplying the data by the coefficient
                proj = data * coeffs * weights

                # Sum all the projected ones to find the new eigenvector and then divide by the
                # length of the vector to reduce it to unit vector length.
                phi_new = np.sum(proj, axis=0)
                phi_new = phi_new / np.linalg.norm(phi_new)

                # If all of the dimensions changes by less than thresh then the
                # condition is set to true and the loop breaks
                cond = np.all((phi_new - phi) < thresh)

                phi = phi_new.reshape(1, -1)
                i += 1
            return phi
    
        # We here find the eigenvectors. We only need two since I'll be working in 2-D
        num_eigs = 3
        eigs = np.zeros((color_shifted.shape[1], num_eigs))
        temp_data = np.copy(color_shifted)
        if verbose: print("Finding eigenvectors for dimensionality reduction")
        for i in range(num_eigs):
            v = find_eigenvector(temp_data, None)
            eigs[:,i] = (v)

            # Subtract the projections of the found eigen vector to start finding the next one.
            coeffs = temp_data @ v.T
            temp_data = temp_data - coeffs * v
            
        # The selection of principal axis vectors that will reduce dimensionality
        self.eigs = eigs
        data_reduced = color_data @ self.eigs
        
        # I took this cut from the random forest example.
        # I cut after doing the PCA in case the cut changes the
        # principal axes and I want to avoid that.
        np.random.seed(1985) # To ensure this is the same every time.
        cut = np.random.uniform(0, 1, data_reduced.shape[0]) < 0.05
        data_cut =  data_reduced[cut]
        z_cut = training_z[cut]
        
        # These next "few" lines are helper functions for the training.
        @jax.jit
        def softmax(x, beta=1):
            return jnp.exp(beta * x) / jnp.sum(jnp.exp(beta * x), axis=0)

        @jax.jit
        def dist(points, centroids, beta=1):
            # Finds the distance between the points and the centroids
            dist = []
            for center in centroids:
                shift = points - center
                dist.append(jnp.linalg.norm(shift, axis=1))

            # Converting to numpy array so we can use boolean indexing
            dist = jnp.asarray(dist)

            # Which category these would be assigned to based on their distances
            # soft min, don't have to one_hot then and the gradient should work.
            return softmax(-dist, beta).T 

        # This is by far not the best way to do this but alas...
        @jax.jit
        def dist_snr(points, centroids, z, beta=1):
            cat = dist(points, centroids, beta)
            return -compute_snr_score(cat, z, binned_nz=True)

        @jax.jit
        def dist_fom(points, centroids, z, beta=1):
            cat = dist(points, centroids, beta)
            return -compute_fom(cat, z, binned_nz=True)
        
        @jax.jit
        def dist_fom_detf(points, centroids, z, beta=1):
            cat = dist(points, centroids, beta)
            return -compute_fom(cat, z, inds=[5, 6], binned_nz=True)
        
        def get_equality_centroids(data, redshift, n_bins=3):
            # Find the edges that split the redshifts into n_bins bins of
            # equal number counts in each
            p = np.linspace(0, 100, n_bins + 1)
            z_edges = np.percentile(redshift, p)

            training_bin = np.zeros_like(data[:, 0])

            # Now find all the objects in each of these bins
            for i in range(n_bins):
                z_low = z_edges[i]
                z_high = z_edges[i + 1]
                training_bin[(redshift > z_low) & (redshift <= z_high)] = i

            centroids = []
            for i in range(0, int(training_bin.max()) + 1):
                cond = training_bin == i
                centroids.append(data[cond].mean(axis=0))

            return np.asarray(centroids)
        
        # points = data_cut
        # redshift = z_cut
        beta = np.ones(1) * num_centroids
        if verbose: print(f"Using beta: {beta}")

        # The function we're optimizing, can't use string inputs in functions 
        # we're differentiating, hence this. The name 'd2' is a historical artifact
        # from when I was implementing various distance functions named dist, d2, and d3.
        if impl == "fom":
            d2 = dist_fom
        elif impl == "fom_detf":
            d2 = dist_fom_detf
        else:
            d2 = dist_snr

        # Technically this is the actual training loop but I call it twice
        # hence the abstraction to a function.
        def loop_and_improve(val, num_epochs):
            to_improve = np.copy(val)
            val_history = []

            # These top and bottom are designed for DETF.
            # Need to reduce by factor of ~10^1 for SNR and
            # ~10^2 for FOM, since DETF is order ~10, SNR ~10^2, FOM ~10^3
            top = -1.5
            bottom = -3

            if impl == "fom": 
                top -= 2
                bottom -= 2
            elif impl == "snr":
                top -= 1
                bottom -= 1

            lr_arr = np.logspace(bottom, top, num_epochs // 2) * 2.5
            lr_arr = np.concatenate([lr_arr, np.flip(lr_arr, 0)])

            # Do this first so we know where we start.
            val, grads = jax.value_and_grad(d2, 1)(data_cut, to_improve, z_cut, beta)
            print(f"Starting {impl.upper()}: {-val}")
            best = np.copy(to_improve)
            best_score = val

            # Terminate at the number of epochs or if the change 
            # in snr is too small to be meaningful. We also force 
            # a minimum of (num epochs // 3) epochs.
            i = 0
            delta_val = 1
            min_change = 0.25
            while (i < num_epochs and abs(delta_val) > min_change) or i < num_epochs // 3:
                try:
                    cur_lr = lr_arr[i]
                    if verbose: print(f"Epoch {i + 1} LR {np.round(cur_lr, 6)}")
                    to_improve += -(grads) * cur_lr

                    # Finding the resultang value and then the grad for the next epoch.
                    val, grads = jax.value_and_grad(d2, 1)(data_cut, to_improve, z_cut, beta)

                    if verbose: print(f"{impl.upper()}: {-val}")
                    val_history.append(val)

                    # Storing the best found score in case we jump out of the minimum
                    # (possible... if not likely in some situations)
                    if val < best_score:
                        best = np.copy(to_improve)
                        best_score = val

                    i += 1
                    if len(val_history) > 1:
                        delta_val = val - val_history[-2]
                        if verbose: print(f"Delta {impl.upper()}: {-delta_val}")

                except UnexpectedTracerError:
                    # I swear this isn't my fault.
                    if verbose: print("Tracer Error, retrying epoch")
                    continue

            return (best, best_score, val_history)

        # This finds the mean in all directions then organizes
        # the starting centroids in an equally spaced circle around the x-y center.
        cent = np.mean(data_cut, axis=0)
        rad_diff = 2 * np.pi / num_centroids
        l = []
        for i in range(num_centroids):
            p = 0.15 * np.asarray([np.cos(i * rad_diff), np.sin(i * rad_diff), 0])
            l.append(p + cent)

        # Run twice with two different starting centroids and use the best one.
        num_epochs = 100
        if verbose: print("\nStart 1")
        centroids = np.asarray(l)
        c1, score1, _ = loop_and_improve(centroids, num_epochs)

        if verbose: print("\nStart 2")
        centroids = get_equality_centroids(data_cut, z_cut, num_centroids)
        c2, score2, _ = loop_and_improve(centroids, num_epochs)

        if score1 < score2:
            if verbose: print(f"Circular start used. {-score1} > {-score2}")
            self.centroids = c1
        else:
            if verbse: print(f"Equality start used. {-score2} > {-score1}")
            self.centroids = c2

    
    # In theory for maximum speed you could jit compile this, since it's
    # all pure jax numpy math.
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
        
        data_valid = []
        for c in ["r", "gr", "ri", "rz"]:
            data_valid.append(data[c])
        data_valid = np.asarray(data_valid).T
        data_valid_r = data_valid @ self.eigs
        
        # Finds the distance between the points and the centroids
        dist = []
        for center in self.centroids:
            shift = data_valid_r - center
            dist.append(jnp.linalg.norm(shift, axis=1))

        # Converting to numpy array so we can use axis for argmin.
        dist = jnp.asarray(dist)

        # Which category these would be assigned to based on their distances
        return jnp.argmin(dist, axis=0)
