#JEC To use the model after fitting

import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

from  metrics import compute_scores, plot_distributions
from data import dict_to_array, load_data, load_redshift

import classifiers

def apply_model(classifier, filename, bands):
    data = load_data(filename, bands, colors=True, errors=False)
    data =  dict_to_array(data, bands, colors=True, errors=False)
    tomo_bin = classifier.predict(data)
    return tomo_bin


def main(bands, n_bin, model_file, output_file):
    # Assume data in standard locations relative to current directory
    training_file = '../data/training.hdf5'
    validation_file = '../data/validation.hdf5'

    clf = load(model_file)
    print('clf parameters: ',clf.get_params())

    #Apply model
    print('Apply model to validation set')
    tomo_bin = apply_model(clf, validation_file, bands)
    
    # Get a score
    z = load_redshift(validation_file)
    scores = compute_scores(tomo_bin, z, metrics="SNR_3x2,FOM_3x2")


    # Get the galaxy distribution of redshifts into n_z bins of
    # equal number counts in each
    p = np.linspace(0, 100, n_bin + 1)
    z_edges = np.percentile(z, p)

    plot_distributions(z, tomo_bin, output_file, z_edges)

    # return 
    return scores
    
if __name__ == '__main__':
    # Command line arguments
    try:
        bands = sys.argv[1]
        n_bin = int(sys.argv[2])
        model_file = sys.argv[3]
        output_file = sys.argv[4]
        assert bands in ['riz', 'griz']
    except:
        sys.stderr.write("Script takes 4 arguments: 'riz'/'griz', n_bin, model_file, output fig\n")
        sys.exit(1)

    # Run main code
    scores = main(bands, n_bin, model_file, output_file)
    print(f"Scores for {n_bin} bin(s) : ")
    for k,v in scores.items():
        print ("      %s : %4.1f"%(k,v))
