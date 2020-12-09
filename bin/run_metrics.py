import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
import sys
sys.path.append(dir_path)

import tomo_challenge
import numpy as np
import glob
import h5py
import yaml

with h5py.File("data/validation.hdf5", "r") as f:#change later
    truth_z = f["redshift_true"][:]
print("Loaded truth data")

results_dir = "/global/cscratch1/sd/zuntz/tomo_challenge_results/"
completed = sorted(glob.glob(results_dir + "*.npy"))

for result_file in completed:
    name = result_file[len(results_dir):]
    img_file = f'{results_dir}/plots/{name}.png'
    metric_file = f'{results_dir}/metrics/{name}.yml'

    if os.path.exists(metric_file):
        continue

    print(name)
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




#     for b in unique_bins:


# tomo_challenge.jc_compute_scores
# # for each job we find in the results
# # load the file
# # make a plot
# # get the metrics


#     db = 'db.sqlite3'
