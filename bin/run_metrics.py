#import tomo_challenge
import numpy as np
import glob

# with h5py.File("data/validation.hdf5", "r") as f:#change later
#     truth_z = f["redshift_true"][:]


results_dir = "/global/cscratch1/sd/zuntz/tomo_challenge_results/"
completed = sorted(glob.glob(results_dir + "*.npy"))

for result_file in completed:
    bins = np.load(result_file).astype(int)

    unique_bins = np.unique(bins)
    print(result_file[len(results_dir):])
    for b in unique_bins:
        print("    ", b, (bins==b).sum())

#     for b in unique_bins:


# tomo_challenge.jc_compute_scores
# # for each job we find in the results
# # load the file
# # make a plot
# # get the metrics


#     db = 'db.sqlite3'
