import os
from urllib.request import urlretrieve

import h5py
import numpy as np

nersc_path = '/global/projecta/projectdirs/lsst/groups/WL/users/zuntz/tomo_challenge_data/'
url_root =  'https://portal.nersc.gov/project/lsst/txpipe/tomo_challenge_data/'


def download_data():
    """Download challenge data (about 4GB) to current directory.

    This will create directories ./riz and ./griz with the training
    and validation files in.

    If on NERSC this will just generate links to the data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if os.environ.get("NERSC_HOST"):
        # If we are on NERSC just make some links
        os.symlink(nersc_path + 'riz', 'riz')
        os.symlink(nersc_path + 'griz', 'griz')
    else:
        # Otherwise actually download both data sets
        for bands in ['riz', 'griz']:
            # These will raise an exception if the directories
            # already exist, preventing downloading twice
            os.makedirs(bands)
            # Download each of the two files for these bands
            for f in ['training', 'validation']:
                filename = f'{bands}/{f}.hdf5'
                urlretrieve(url_root + filename, filename)



def load_magnitudes_and_colors(filename, bands):
    """Load magnitudes, and compute colors from them,
    from a training or validation file.

    Note that there are other columns available in
    the files that this function does not load, but are
    available for your methods (mag errors, size, s/n).

    Parameters
    ----------
    filename: str
        The name of the file to read, e.g. riz/training.hdf5

    bands: str
        The list of bands to read from the data

    Returns
    -------
    data: array
        Dimension is nfeature x nrow, where nfeature = nband + ncolor
        and ncolor = nband * (nband - 1) / 2
    """

    # Open the data file
    f = h5py.File(filename)

    # Get the number of features (mags + colors)
    # and data points
    ndata = f['ra'].size
    nband = len(bands)
    ncolor = (nband * (nband - 1)) // 2
    nfeature = nband + ncolor

    # np.empty is like np.zeros except it doesn't
    # bother filling in the data with zeros, just
    # allocates space.  We can use it because we
    # are filling it in in a moment.  This gets
    # transposed before we return it to match
    # what sklearn expects
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
    
    # Return the data. sklearn wants it the other way around
    # because data scientists are weird and think of data as
    # lots of rows instead of lots of columns.
    return data.T

def load_redshift(filename):
    """Load a redshift column from a training or validation file"""
    f = h5py.File(filename)
    z = f['redshift_true'][:]
    f.close()
    return z    


if __name__ == '__main__':
    download_data()
