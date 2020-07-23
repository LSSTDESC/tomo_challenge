import h5py
import numpy as np


# input files
indir = "/global/projecta/projectdirs/lsst/groups/WL/users/flanusse/buzzard-inputs/"

mcal_file = h5py.File(indir + "shear_catalog.hdf5", 'r')
mcal_group = mcal_file['shear']

photo_file = h5py.File(indir + 'photometry_catalog.hdf5', 'r')
z = photo_file['photometry/redshift_true'][:]

# use all bands available
bands = 'griz'

# do basic shear cuts on s/n, flag, and size / psf size
sel = mcal_group['mcal_s2n'][:] > 10
print("Read s/n")
sel &= mcal_group['mcal_flags'][:] == 0
print("Read flag")
sel &= (mcal_group['mcal_T'][:] / mcal_group['mcal_psf_T_mean'][:]) > 0.5
print("Read sizes")

# convert to an index
sel = np.where(sel)[0]
ntot = sel.size

# Randomly reorder the data as it is in z-order
# by default, and we don't want that
reorder = np.arange(ntot)
np.random.shuffle(reorder)
sel = sel[reorder]

print("Loaded cuts")

# define subsets and their fractions.
training = 0
testing = 1
validation = 2
probs = [0.25, 0.25, 0.5]
subset = np.random.choice(3, size=ntot, p=probs)

# output files
files = [
    h5py.File("training.hdf5", "w"),
    h5py.File("testing.hdf5", "w"),
    h5py.File("validation.hdf5", "w"),
]

# get indices for each file, into the galaxies that
# pass the initial cut
selections = [
    np.where(subset==i)[0] for i in range(3)
]

# copy in redshifts and print sizes
print("Created selections")
for s,f in zip(selections, files):
    print(f"Selection {f}:  {s.size}")


# Cols we need from each file
shear_cols = ['ra', 'dec', 'mcal_T', 'mcal_s2n']
photo_cols = (['redshift_true'] 
            + [f'mag_{b}' for b in bands] 
            + [f'mag_{b}_err' for b in bands])

for col in shear_cols:
    print(f"Saving {col}")
    d = mcal_group[col][:]
    for s, f in zip(selections, files):
        f.create_dataset(col, data=d[sel][s])

for col in photo_cols:
    print(f"Saving {col}")
    d = photo_file[f'photometry/{col}'][:]
    for s, f in zip(selections, files):
        f.create_dataset(col, data=d[sel][s])


for f in files:
    f.close()
