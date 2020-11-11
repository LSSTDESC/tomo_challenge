import os
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
import sys
sys.path.append(dir_path)

import tomo_challenge as tc
from challenge import run_one
import yaml
import time
import sys
import traceback
import faulthandler
faulthandler.enable()

status_file = sys.argv[1]
name = sys.argv[2]

if len(sys.argv) > 3:
    index = sys.argv[3]
else:
    index = ""

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
metrics_fn = tc.jc_compute_scores
metrics = ['SNR_3x2']

with open(status_file, 'a') as status_file:
    try:
        fn = f'evaluation/{name}{index}.yml'
        full_config = yaml.safe_load(open(fn))['run'][name]
        settings = list(full_config.values())[0]
    except FileNotFoundError:
        status_file.write(f'{name} - no config {fn}\n')
        traceback.print_exc()
        sys.exit(1)
    except KeyError:
        status_file.write(f'{name}{index} - config malformed\n')
        traceback.print_exc()
        sys.exit(1)
    except Exception as error:
        status_file.write(f'{name}{index} - (load) {error}\n')
        traceback.print_exc()
        sys.exit(1)


    # hack because this code is extra special and
    # can't just load the data like everyone else
    if name == "Flax_LSTM":
        n = settings['bins']
        if settings['colors']:
            n += n * (n - 1) // 2
        if settings['errors']:
            n *= 2
        settings['n_feats'] = n
        print(f"Set Flax_LSTM n_feats to {n}")
        
        
    try:
        t0 = time.time()
        scores = run_one(name, bands, settings, training_data, training_z, validation_data,
                     validation_z, metrics, metrics_fn)
    except Exception as error:
        t = time.time() - t0
        tb = traceback.format_exc()
        traceback.print_exc()
        status_file.write(f'{name}{index} - (run, {t:.2f}) {error}\n')
        status_file.write(tb + '\n')
        sys.exit(1)

    t = time.time() - t0
    score = scores['SNR_3x2']
    status_file.write(f'{name}{index} - (success, {t:.2f}) {score}\n')
