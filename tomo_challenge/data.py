import os
from urllib.request import urlretrieve

nersc_path = '/project/projectdirs/lsst/www/txpipe/tomo_challenge_data/'
url_root =  'https://portal.nersc.gov/project/lsst/txpipe/tomo_challenge_data/'


def download_data():
    """Download challenge data (about 4GB) to current directory.

    If on NERSC this will just generate links to the data.
    """
    if os.environ.get("NERSC_HOST"):
        # If we are on NERSC just make some links
        os.symlink("riz", nersc_path + 'riz')
        os.symlink("griz", nersc_path + 'griz')
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



if __name__ == '__main__':
    download_data()
