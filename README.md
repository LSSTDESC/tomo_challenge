# The Idealised Tomography Challenge

In this challenge you are asked to group galaxies into tomographic bins using only the quantities we generate using the metacalibration method.  These quantities are the only ones for which we can compute a shear bias correction associcated with the division.

We provide training and validation sets of data, and once everyone has added methods, we will run on the secret testing data.

This test is highly idealised: we have a huge complete training sample, simple noise models, no outliers, and no variation in depth or any other observing conditions.

Galaxy magnitudes are generated by adding noise to the CosmoDC2 data and applying a preliminary selection cut (SNR>10, metacal flag=0, metacal size > PSF size / 2).

All entrants will be on the author list for the upcoming TXPipe-CosmoDC2 paper, which these results will go into.  The winner will additionally receive glory.

The challenge has been extended and is open until the end of August 2020.


## Installing requirements

In general you can install python requirements with `pip install -r requirements.txt`

On NERSC it's easiest to use shifter (I've had problems with CCL there):

```
shifter --image=joezuntz/txpipe-tomo bash
```

This will put you in a shell with all requirements.


## Getting training data

Run `python -m tomo_challenge.data` in this directory to download the full set of challenge data, about 4.5GB.  You can also get the individual files from here if you prefer:  https://portal.nersc.gov/project/lsst/txpipe/tomo_challenge_data/

## Installing dependencies

From the root folder:
```
$ pip install -r requirements.txt
```


## Metric

The first metric is the S/N on the spectra generated with the method:
```
score^2 = sqrt(mu^T . C^{-1} . mu) - baseline
```
where mu is the theory spectrum and C the Gaussian covariance.

The second is a Fisher-based Figure Of Merit, currently sigma8-omegac, though we will later add w0-wa.


## Entering the contest.

You can enter the contest by pull request.  Add a python file with your method in it.


## Example Method

In `tomo_challenge/classifiers/random_forest.py` you can find an example of using a scikit-learn classified with a simple galaxy split to assign objects.  Run it by doing, e.g.:

```bash
$ python bin/challenge.py example/example.yaml
```

This will compute the metrics and write an output file for some test methods.

You are welcome to adapt any part of that code in your methods.


The example random forest implementation gets the following scores using the spectrum S/N metric.  For griz:

```
# nbin  score
1  0.0
2  28.2
3  35.4
4  37.9
5  39.7
6  40.3
7  40.5
8  41.1
9  41.5
10  41.8
```

and for riz:

```
# nbin  score
1  0.1
2  21.6
3  26.4
4  28.5
5  29.7
6  30.1
7  30.1
8  29.4
9  29.1
10  30.1
```


## FAQ

- **How do I enter?**

Create a pull request that adds a python file to the `tomo_challenge/classifiers` directory containing a class with `train` and `apply` methods, following the random forest example as a template.

---

- **Can I use a different programming language?**

Only if it can be easily called from python.

---

- **What general software requirements are needed?**
- Relatively recent compilers
- MPI
- Lapack
- Python 3.6+
- cmake
- swig
- if you find others please let us know

On ubuntu 20 you can install these apt/yum packages:
```
gfortran
cmake
swig
libopenmpi-dev
liblapack3
liblapack64-dev
libopenblas-dev
```

- **What are the metrics?**

1. The total S/N (including covariance) of all the power spectra of weak lensing made using your bins
2. The inverse area of the w0-wa Fisher matrix (due to [a technical problem](https://github.com/LSSTDESC/CCL/issues/779) the current metric is the sigma8-omega_c Fisher matrix)

Each can be run on ww (lensing-lensing) gg (lss-lss) and 3x2 (both + cross-corr), so the full list is: `SNR_ww, SNR_gg, SNR_3x2, FOM_ww, FOM_gg, FOM_3x2`

---

- **How can I change hyper-parameters or otherwise configure my method?**

Add a class variable `valid_options` in your method. Then variables in it are accessible in the dictionary `self.opt`.  See the random forest file for an example.

Then you can set the variables for a given run in your yaml file, as in how `bins` is set in the example yaml file.

---

- **Why is this needed?**

The metacal method can correct galaxy shear biases associated with putting galaxies into bins (which arise because noise on magnitudes correlates with noise on shape), but only if the selection is done with quantities measured jointly with the shear.

This only affects shear catalogs - for lens bins we can do what we like.

---

- **What is the input data?**

[CosmoDC2](https://arxiv.org/pdf/1907.06530.pdf) galaxies with mock noise added.

---

- **How many bins should I use, and what are the target distributions?**

As many as you like - it's likely that more bins will add to your score as long as they're well-separated in redshift, so you probably want to push the number upwards.  You can experiment with what edges give you best metrics; historically most approaches have tried to divide into bins with roughly equal numbers, so that may be a good place to start.

---

- **What do I get out of this?**

You can be an author on the paper we write if you submit a working method.  

---

- **Can non-DESC people enter?**

Yes - we have now been told by the publication board that this is fine, and the results paper can be non-DESC.

---

- **How realistic is this?**

This is the easiest possible challenge - the training set is large and drawn from the same population as the test data, and the data selection is relatively simple.

If you think it's too unrealistic then you should do really really well.

---

- **Do I have to use machine learning methods?**

No - we call the methods `train` and `apply`, but that's just terminology, you can train however you like.

---

- **Do I have to assign every galaxy to a bin?**

No, you can leave out galaxies if you want.  If you leave out too many the decrease in number density will start to hit your score of course.

---

- **Can I use a simpler metric?**

Yes, you can train however you like, including with your own metrics.  The final score will be on a suite of metrics including the two here.  We reserve the right to add more metrics to better understand things.

---

- **When does the challenge close?**

The end of August 2020. (We have updated this from July 2020 since we felt it was too soon).

---

- **What does the winner get?**

Recognition.
