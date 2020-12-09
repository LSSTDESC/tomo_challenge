import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
import sys
sys.path.append(dir_path)

import tomo_challenge
import numpy as np
import h5py
import yaml

import task_queue

def setup(queue):

    classifiers = [
        ("NeuralNetwork", 1), 
        ("NeuralNetwork", 2), 
        ("Autokeras_LSTM", 0), 
        ("CNN", 0), 
        ("ENSEMBLE1", 0), 
        ("Flax_LSTM", 0), 
        ("JaxCNN", 0), 
        ("JaxResNet", 0), 
        ("LSTM", 0), 
        ("TCN", 0), 
        ("ZotBin", 0), 
        ("ZotNet", 0), 
        ("myCombinedClassifiers", 0), 
        ("IBandOnly", 0),
        ("Random", 0),
        ("mlpqna", 0),
        ("ComplexSOM", 0),
        ("SimpleSOM", 0),
        ("PCACluster", 0),
        ("GPzBinning", 0),
        ("funbins", 0),
        ("UTOPIA", 0),
        ("LGBM", 0),
        ("RandomForest", 0),
        ("SummerSlasher", 0),
        ("MineCraft", 0),
    ]
    
    no_nbin = {
        "SummerSlasher",
        "MineCraft",
    }


    for (classifier, config_index) in classifiers:
        if classifier in no_nbin:
            bins = [0]
        else:
            bins = [3, 5, 7, 9]

        for nbin in bins:
            name = f"{classifier}_{nbin}_{config_index}"
            print(name)
            queue.add_job(name, {})



def task(name):
    print(name)
    results_dir = "/global/cscratch1/sd/zuntz/tomo_challenge_results"
    result_file = f'{results_dir}/{name}.npy'
    img_file = f'{results_dir}/plots/{name}.png'
    metric_file = f'{results_dir}/metrics/{name}.yml'

    if os.path.exists(img_file) and os.path.exists(metric_file):
        print("Done already")
        return 0

    bins = np.load(result_file).astype(int)
    unique_bins = np.unique(bins)
    
    if -1 in unique_bins:
        # if this is just a tiny number then put them in bin 0
        if (bins==-1).sum() < 0.03 * bins.size:
            bins[bins==-1] = 0
        # otherwise push everything up by one
        else:
            bins += 1
            unique_bins += 1

    counts = {int(b): int((bins==b).sum()) for b in unique_bins}
    
    tomo_challenge.metrics.plot_distributions(truth_z, bins, f'{results_dir}/plots/{name}.png', metadata={})

    metrics = tomo_challenge.jc_compute_scores(bins, truth_z)
    output = {"name": name, "counts":counts}
    output.update(metrics)
    print(output)
    with open(metric_file, 'w') as f:
        yaml.dump(output, f)

    return 0


import argparse
parser = argparse.ArgumentParser(description='Batch run jobs')
parser.add_argument('--setup', action='store_true', help='Set up the DB and exit')





def main():
    args = parser.parse_args()
    db = "metrics.db"
    queue = task_queue.TaskQueue(db, task, {})
    if args.setup:
        setup(queue)
    else:
        global truth_z
        with h5py.File("data/validation.hdf5", "r") as f:#change later
            truth_z = f["redshift_true"][:]
        queue.run_loop()


if __name__ == '__main__':
    main()