import tomo_challenge



# with h5py.File("data/validation.hdf5", "r") as f:#change later
#     truth_z = f["redshift_true"][:]


completed = glob.glob("/global/cscratch1/sd/zuntz/tomo_challenge_results/*.npy")

for result_file in completed:
    bins = np.load(result_file).astype(int)

    unique_bins = np.unique(bins)
    print(result_file, unique_bins)
#     for b in unique_bins:


# tomo_challenge.jc_compute_scores
# # for each job we find in the results
# # load the file
# # make a plot
# # get the metrics


#     db = 'db.sqlite3'
