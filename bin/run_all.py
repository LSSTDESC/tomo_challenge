import os
import sys

mode = sys.argv[1]
if mode not in ['cpu', 'gpu']:
    raise ValueError("set cpu or gpu")

status_file = f"{mode}-status.txt"
open(status_file, "w").close()

if mode == 'gpu':
    os.environ["LD_LIBRARY_PATH"] += ":/home/jzuntz/tomo_challenge/cuda/cuda/lib64"

if mode == "gpu":
    names = [
#        "Flax_LSTM",
#        "JaxResNet",
#        "JaxCNN",
#        "LSTM",
#        "ZotBin",
        "ENSEMBLE1",
        "ENSEMBLE2",
        "CNN",
        "Autokeras_LSTM",
        "TCN",
        "NeuralNetwork 1",
        "NeuralNetwork 2",
        "ZotNet",
    ]
else:
    names = [
        "myCombinedClassifiers",
        "LGBM",
        "funbins",
        "RandomForest",
        "PQNLD",
        "PCACluster",
        "MineCraft",
        "GPzBinning",
        "IBandOnly",
        "mlpqna",
        "SummerSlasher",
        "ComplexSOM",
        "UTOPIA",
        "SimpleSOM",
    ]

for name in names:
    s = os.system(f"python bin/run_one.py {status_file} {name}")
    print(f"{name} status = {s}")
