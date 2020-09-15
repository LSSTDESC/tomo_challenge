import os
from urllib.request import urlretrieve
import warnings
import h5py
import numpy as np

# Path to original challenge dataset
nersc_path = '/global/projecta/projectdirs/lsst/groups/WL/users/zuntz/tomo_challenge_data/ugrizy'
url_root =  'https://portal.nersc.gov/cfs/lsst/txpipe/tomo_challenge_data/ugrizy'

# Path to Buzzard version of the challenge dataset
nersc_path_buzzard = '/global/projecta/projectdirs/lsst/groups/WL/users/flanusse/tomo_challenge_buzzard'
url_root_buzzard =  'https://portal.nersc.gov/cfs/lsst/txpipe/tomo_challenge_data/ugrizy_buzzard'

# This is not supposed to be needed - I don't understand why in my shifter env the warning
# is being repeated.
warned = False

class MyProgressBar:
    def __init__(self):
        self.pbar = None
        try:
            import progressbar
            self.module = progressbar
        except ImportError:
            self.module = None

    def __call__(self, block_num, block_size, total_size):
        if self.module is None:
            return

        if self.pbar is None:
            self.pbar = self.module.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_data():
    """Download challenge data (about 6GB) to current directory.

    This will create directories ./data with the training
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
        os.symlink(nersc_path, 'data/')
        os.symlink(nersc_path_buzzard, 'data_buzzard/')
    else:
        # Otherwise actually download both data sets
        os.makedirs('data', exist_ok=True)
        # Download each of the two files for these bands
        for f in ['validation', 'training']:
            filename = f'{f}.hdf5'
            progress = MyProgressBar()
            urlretrieve(f'{url_root}/{filename}', f'data/{filename}', reporthook=progress)

        os.makedirs('data_buzzard', exist_ok=True)
        # Download each of the two files for these bands
        for f in ['validation', 'training']:
            filename = f'{f}.hdf5'
            progress = MyProgressBar()
            urlretrieve(f'{url_root_buzzard}/{filename}', f'data_buzzard/{filename}', reporthook=progress)
            

def load_mags(filename, bands, errors=False, band_triplets_errors=False, heal_undetected=False):


    # Warn about non-detections being set mag=30.
    # The system is only supposed to warn once but on
    # shifter it is warning every time and I don't understand why.
    # Best guess is one of the libraries we load sets some option.
    global warned
    if not warned:
        if heal_undetected:
            warnings.warn("Setting inf (undetected) bands to the mag for which S/N=1 (i.e. mag_err=2.5*log10(2))")
        else:
            warnings.warn("Setting inf (undetected) bands to mag=30 and mag_err=30")
        warned = True

    data = {}

    with h5py.File(filename, 'r') as f:
        # load all bands
        for b in bands:
            data[b] = f[f'{b}_mag'][:]

            if errors or band_triplets_errors:
                data[f'{b}_err'] = f[f'{b}_mag_err'][:]

    # Set undetected objects to S/N=1 mags and errors if heal_undetected=True, otherwise to mag 30 +/- 30
    print(f'Analyzing sample from {filename}')
    for b in bands:
        bad = ~np.isfinite(data[b])
        if heal_undetected:
            mag_err_snr1=2.5*np.log10(2)
            mag_snr1=np.interp(mag_err_snr1, data[f'{b}_err'][~bad], data[b][~bad])
            data[b][bad] = mag_snr1
            print(f'band {b}: S/N~1, mag={mag_snr1}, mag_err={mag_err_snr1}')
        else:
            data[b][bad] = 30.0

        if errors or band_triplets_errors:
            if heal_undetected:
                data[f'{b}_err'][bad] = mag_err_snr1
            else:
                data[f'{b}_err'][bad] = 30.0

    return data

def add_colors(data, bands, errors=False, band_triplets_errors=False):
    nband = len(bands)
    nobj = data[bands[0]].size
    ncolor = nband * (nband - 1) // 2

    # also get colors as data, from all the
    # (non-symmetric) pairs.  Note that we are getting some
    # redundant colors here, and some incorrect colors based
    # on the choice to set undetected magnitudes to 30.
    for b,c in colors_for_bands(bands):
        data[f'{b}{c}'] = data[f'{b}'] - data[f'{c}']
        if errors or band_triplets_errors:
            data[f'{b}{c}_err'] = np.sqrt(data[f'{b}_err']**2 + data[f'{c}_err']**2)

def dict_to_array(data, bands, errors=False, colors=False, band_triplets=False, band_triplets_errors=False):
    nobj = data[bands[0]].size
    nband = len(bands)
    ncol = nband
    if colors:
        ncol += nband * (nband - 1) // 2
    if errors:
        ncol *= 2
    if band_triplets:
        ncol += (nband - 2)
    if band_triplets_errors:
        ncol += (nband - 2)

    arr = np.empty((ncol, nobj))
    i = 0
    for b in bands:
        arr[i] = data[b]
        i += 1

    if colors:
        for b, c in colors_for_bands(bands):
            arr[i] = data[f'{b}{c}']
            i += 1
            
    if errors:
        for b in bands:
            arr[i] = data[f"{b}_err"]
            i += 1

    if errors and colors:
        for b, c in colors_for_bands(bands):
            arr[i] = data[f'{b}{c}_err']
            i += 1
    
    if band_triplets:
        for (b1,b2), (c1,c2) in band_triplets_for_bands(bands):
            arr[i] = data[f'{b1}{b2}']-data[f'{c1}{c2}']
            i += 1

    if band_triplets_errors:
        for (b1,b2), (c1,c2) in band_triplets_for_bands(bands):
            arr[i] = np.sqrt(data[f'{b1}{b2}_err']**2+data[f'{c1}{c2}_err']**2)
            i += 1
            
    return arr.T


def colors_for_bands(bands):
    for i,b in enumerate(bands):
        for c in bands[i+1:]:
            yield b, c

def adjacent_colors_for_bands(bands):
    for i,b in enumerate(bands[0:-1]):
        yield b, bands[i+1]
        
def band_triplets_for_bands(bands):
    clrs = list(adjacent_colors_for_bands(bands))
    bts = colors_for_bands(clrs)
    bts_selected = [bt for bt in bts if bt[0][1]==bt[1][0]] # difference of two adjacent colors, e.g. (g-r)-(r-i)
    return bts_selected

def load_data(filename, bands, colors=False, errors=False, band_triplets=False, band_triplets_errors=False, array=False, heal_undetected=False):
    data = load_mags(filename, bands, errors=errors, band_triplets_errors=band_triplets_errors, heal_undetected=heal_undetected)

    if colors:
        add_colors(data, bands, errors=errors, band_triplets_errors=band_triplets_errors)

    if array:
        data = dict_to_array(data, bands, errors=errors, colors=colors, band_triplets=band_triplets, band_triplets_errors=band_triplets_errors)

    return data



def load_redshift(filename):
    """Load a redshift column from a training or validation file"""
    f = h5py.File(filename, 'r')
    z = f['redshift_true'][:]
    f.close()
    return z


if __name__ == '__main__':
    download_data()
