import tomo_challenge as tc
from challenge import run_one
import yaml
import time
import sys
import os
import traceback

training_file = 'data/mini_training.hdf5'
validation_file = 'data/mini_validation.hdf5'
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

name = 'MineCraft'
full_config = yaml.safe_load(open(f'evaluation/{name}.yml'))['run'][name]
settings = list(full_config.values())[0]
scores = run_one(name, bands, settings, training_data, training_z, validation_data,
             validation_z, metrics, metrics_fn)

# zstatus_file = open('status.txt', 'w')

# names = tc.Tomographer._subclasses.keys()
# names = ['PQNLD']
# for name in names:
#     status_file.flush()
#     try:
#         full_config = yaml.safe_load(open(f'evaluation/{name}.yml'))['run'][name]
#         settings = list(full_config.values())[0]
#     except FileNotFoundError:
#         status_file.write(f'{name} - no config\n')
#         continue
#     except KeyError:
#         status_file.write(f'{name} - config malformed\n')
#         continue
#     except Exception as error:
#         status_file.write(f'{name} - (load) {error}\n')
#         continue

#     try:
#         t0 = time.time()
#         scores = run_one(name, bands, settings, training_data, training_z, validation_data,
#                      validation_z, metrics, metrics_fn)
#     except Exception as error:
#         t = time.time() - t0
#         tb = traceback.format_exc()
#         status_file.write(f'{name} - (run, {t:.2f}) {error}\n')
#         status_file.write(tb + '\n')
#         continue

#     t = time.time() - t0
#     score = scores['SNR_3x2']
#     status_file.write(f'{name} - (success, {t:.2f}) {score}\n')
