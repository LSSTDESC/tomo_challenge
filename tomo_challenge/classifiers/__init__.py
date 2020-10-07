import os
import sys
from .base import Tomographer
import glob
import ctypes

# Initialization for GPus
# Set up what JAX needs to load
cuda_dir = os.environ.get('CUDA_DIR', '.')
no_gpu = os.environ.get('TOMO_NO_GPU', 0)

#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

def try_load_lib(fn):
    if os.path.exists(fn):
        ctypes.cdll.LoadLibrary(fn)

import ctypes

# hack until global installation on cuillin
if no_gpu:
    gpus = []
else:
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_dir}'
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/jzuntz/tomo_challenge/cuda/cuda/lib64"
    for fn in glob.glob("/home/jzuntz/tomo_challenge/cuda/cuda/lib64/lib*.so.8"):
        print(fn)
        try_load_lib(fn)

    from jax.lib import xla_bridge
    print("Running JAX on: ", xla_bridge.get_backend().platform)

    # Tell tensorflow not to steal all the memory too
    import tensorflow as tf
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except AttributeError:
        gpus = []

if gpus:
    print("Running tensorflow on GPU")
else:
    print("Running tensorflow on CPU")


# def all_python_files():
#     root_dir = os.path.dirname(__file__)
#     names = []
#     for filename in os.listdir(root_dir):
#         if filename.endswith('.py') and filename != '__init__.py':
#             name = filename[:-3]
#             names.append(name)
#     return names

# for name in all_python_files():
#     try:
#         __import__(name, globals(), locals(), level=1)
#     except Exception as error:
#         sys.stderr.write(f"Failed to import {name}: '{error}'")
