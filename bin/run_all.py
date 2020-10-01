import os
open("status.txt", "w").close()
os.environ["LD_LIBRARY_PATH"] += ":/home/jzuntz/tomo_challenge/cuda/cuda/lib64"

for name in [
    "ComplexSOM 1",
    "ComplexSOM 2",
    "JaxResNet",
    "SummerSlasher",
    "JaxCNN",
    "Flax_LSTM",
    "UTOPIA 1",
    "UTOPIA 1",
    "LSTM",
    "ZotBin",
    "funbins",
    "SimpleSOM",
    "ENSEMBLE2",
    "LGBM",
    "ENSEMBLE1",
    "myCombinedClassifiers",
    "RandomForest",
    "PQNLD",
    "CNN",
    "Autokeras_LSTM",
    "TCN",
    "PCACluster",
    "Random",
    "MineCraft",
    "GPzBinning",
    "IBandOnly",
    "mlpqna",
    "NeuralNetwork 1",
    "NeuralNetwork 2",
    "ZotNet",
    ]:
    s = os.system(f"python bin/run_one.py {name}")
    print(f"{name} status = {s}")
