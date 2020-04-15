import h5py
import numpy as np

indir = "/global/projecta/projectdirs/lsst/" \
          "groups/WL/users/zuntz/data/cosmoDC2-1.1.4_oneyear_unit_response/"


mcal_file = h5py.File(indir + "shear_catalog.hdf5")
mcal_group = mcal_file['metacal']

photo_file = h5py.File(indir + 'photometry_catalog.hdf5')
z = photo_file['photometry/redshift_true'][:]

bands = 'ugrizy'
g = False

sel = mcal_group['mcal_s2n'][:] > 10
print("Read s/n")
sel &= mcal_group['mcal_flags'][:] == 0
print("Read flag")
sel &= (mcal_group['mcal_T'][:] / mcal_group['mcal_psf_T_mean'][:]) > 0.5
print("Read sizes")
sel = np.where(sel)[0]
ntot = sel.size

# Randomly reorder the data as it is in z-order
# by default
reorder = np.arange(ntot)
np.random.shuffle(reorder)
sel = sel[reorder]

print("Loaded cuts")

training = 0
testing = 1
validation = 2
probs = [0.25, 0.25, 0.5]
sel2 = np.random.choice(3, size=ntot, p=probs)


files = [
    h5py.File("training.hdf5", "w"),
    h5py.File("testing.hdf5", "w"),
    h5py.File("validation.hdf5", "w"),
]

selections = [
    np.where(sel2==i)[0] for i in range(3)
]

print("Created selections")

for s,f in zip(selections, files):
    print(f"Selection {f}:  {s.size}")
    f.create_dataset('redshift_true', data=z[sel][s])

cols = ['ra', 'dec', 'mcal_T', 'mcal_s2n'] + \
       [f'mcal_mag_{b}' for b in bands] + \
       [f'mcal_mag_err_{b}' for b in bands]



for col in cols:
    print(f"Saving {col}")
    d = mcal_group[col][:]
    for s, f in zip(selections, files):
        f.create_dataset(col, data=d[sel][s])

if g:
    print(f"Saving mcal_mag_g")
    d = photo_file['photometry/g_mag'][:]
    for s, f in zip(selections, files):
        f.create_dataset('mcal_mag_g', data=d[sel][s])
    print(f"Saving mcal_mag_g")
    d = photo_file['photometry/g_mag_err'][:]
    for s, f in zip(selections, files):
        f.create_dataset('mcal_mag_err_g', data=d[sel][s])


for f in files:
    f.close()
