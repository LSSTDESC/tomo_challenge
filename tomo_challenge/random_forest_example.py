"""
This is an example tomographic bin generator
using a random forest.

Feel free to use any part of it in your own efforts.
"""
import time
import sys

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .score import compute_score

def load_magnitudes_and_colors(filename, bands):
    """Load magnitudes and compute colors
    from a training or validation file
    """

    # Open the data file
    f = h5py.File(filename)

    # Get the number of features (mags + colors)
    # and data points
    ndata = f['ra'].size
    nband = len(bands)
    ncolor = (nband * (nband - 1)) // 2
    nfeature = nband + ncolor

    print(f"Loading {nband} columns and {ndata} rows from {filename}")

    # np.empty is like np.zeros except it doesn't
    # bother filling in the data with zeros, just
    # allocates space.  We can use it because we
    # are filling it in in a moment
    data = np.empty((nfeature, ndata))

    # Read the magnitudes into the array
    for i, b in enumerate(bands):
        data[i] = f['mcal_mag_{}'.format(b)][:]

    f.close()
    print(f"Loaded magnitudes. Setting infinite (undetected) bands to 30")
    data[:nband][~np.isfinite(data[:nband])] = 30.0

    # Starting column for the colors
    n = nband

    # also get colors as data, from all the
    # (non-symmetric) pairs.  Note that we are getting some
    # redundant colors here.
    for i in range(nband):
        for j in range(i+1, nband):
            data[n] = data[i] - data[j]
            n += 1

    print(f"Computed colors")

    
    # Return the data. sklearn wants it the other way around
    # because data scientists are weird and think of data as
    # lots of rows instead of lots of columns.
    return data.T

def load_redshift(filename):
    """Load a redshift column from a training or validation file"""
    f = h5py.File(filename)
    print(f"Loading redshift from {filename}")
    z = f['redshift_true'][:]
    f.close()
    return z    


def build_random_forest(filename, bands, n_bin, **kwargs):
    # Load the training data
    training_data = load_magnitudes_and_colors(filename, bands)

    # Get the truth information
    z = load_redshift(filename)

    # Now put the training data into redshift bins.
    # Use zero so that the one object with minimum
    # z in the whole survey will be in the lowest bin
    training_bin = np.zeros(z.size)

    # Find the edges that split the redshifts into n_z bins of
    # equal number counts in each
    p = np.linspace(0, 100, n_bin + 1)
    z_edges = np.percentile(z, p)

    # Now find all the objects in each of these bins
    for i in range(n_bin):
        z_low = z_edges[i]
        z_high = z_edges[i + 1]
        training_bin[(z > z_low) & (z <= z_high)] = i

    print("Cutting down data for speed to 5% of original size")
    print("If you do this, do so at random - the data has an ordering to it")
    cut = np.random.uniform(0, 1, z.size) < 0.05
    training_bin = training_bin[cut]
    training_data = training_data[cut]

    # Can be replaced with any classifier
    classifier = RandomForestClassifier(**kwargs)

    print("Fitting data ...")
    t0 = time.perf_counter()
    # Lots of data, so this will take some time
    classifier.fit(training_data, training_bin)
    duration = time.perf_counter() - t0
    print(f"... complete: fitting took {duration:.1f} seconds")

    return classifier


def apply_random_forest(classifier, filename, bands):
    data = load_magnitudes_and_colors(filename, bands)
    print("Applying classifier")
    tomo_bin = classifier.predict(data)
    return tomo_bin

def main(bands, n_bin):
    # Assume data in standard locations relative to current directory
    training_file = f'{bands}/training.hdf5'
    validation_file = f'{bands}/validation.hdf5'
    classifier = build_random_forest(training_file, bands, n_bin,
                                     max_depth=10,
                                     max_features=None,
                                     n_estimators=20,
    )

    tomo_bin = apply_random_forest(classifier, validation_file, bands)

    # Get a score
    z = load_redshift(validation_file)
    score = compute_score(z, tomo_bin)

    # Return. Command line invovation also prints out
    return score


if __name__ == '__main__':
    # Command line arguments
    try:
        bands = sys.argv[1]
        n_bin = int(sys.argv[2])
        assert bands in ['riz', 'griz']
    except:
        sys.stderr.write("Script takes two arguments, 'riz'/'griz' and nbin\n")
        sys.exit(1)

    # Run main code
    score = main(bands, n_bin)
    print(f"Score = {score}")
