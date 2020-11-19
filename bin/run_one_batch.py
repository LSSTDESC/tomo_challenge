import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
import sys
sys.path.append(dir_path)

task_name, classifier, nbin, index = sys.argv[1:]
nbin = int(nbin)
index = int(index)


scratch = os.environ['SCRATCH']


import tomo_challenge as tc
from challenge import run_one
import yaml
import time
import sys
import traceback
import faulthandler
faulthandler.enable()

if index == 0:
    index = ""

fn = f'evaluation/{name}{index}.yml'
full_config = yaml.safe_load(open(fn))['run'][name]
settings = list(full_config.values())[0]



training_file = 'data-train/training-cut.hdf5'
validation_file = 'data/validation.hdf5'
bands = 'riz'

training_data = tc.load_data(
    training_file,
    bands,
    errors=True,
    colors=True,
    size=True
)

validation_data = tc.load_data(
    validation_file,
    bands,
    errors=True,
    colors=True,
    size=True
)

training_z = tc.load_redshift(training_file)
validation_z = tc.load_redshift(validation_file)
metrics_fn = None
metrics = None


settings['bins'] = nbin

# hack because this code is extra special and
# can't just load the data like everyone else
if classifier == "Flax_LSTM":
    n = settings['bins']
    if settings['colors']:
        n += n * (n - 1) // 2
    if settings['errors']:
        n *= 2
    settings['n_feats'] = n
    print(f"Set Flax_LSTM n_feats to {n}")
    

results = run_one(classifier, bands, settings, training_data, training_z, validation_data,
             validation_z, metrics, metrics_fn)


np.save(f'{scratch}/tomo_challenge_results/{task_name}.npy', results)