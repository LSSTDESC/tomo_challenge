import task_queue
import subprocess

CPU=0
GPU=1

off_by_one = {
    "LSTM",
    "LGBM",
    "CNN",
    "GPzBinning"
}

def setup(queue):

    classifiers = [
        ("NeuralNetwork", GPU, 1), 
        ("NeuralNetwork", GPU, 2), 
        ("Autokeras_LSTM", GPU, 0), 
        ("CNN", GPU, 0), 
        ("ENSEMBLE1", GPU, 0), 
        ("Flax_LSTM", GPU, 0), 
        ("JaxCNN", GPU, 0), 
        ("JaxResNet", GPU, 0), 
        ("LSTM", GPU, 0), 
        ("TCN", GPU, 0), 
        ("ZotBin", GPU, 0), 
        ("ZotNet", GPU, 0), 
        ("myCombinedClassifiers", GPU, 0), 
        ("IBandOnly", CPU, 0),
        ("Random", CPU, 0),
        ("mlpqna", CPU, 0),
        ("ComplexSOM", CPU, 0),
        ("SimpleSOM", CPU, 0),
        ("PCACluster", CPU, 0),
        ("GPzBinning", CPU, 0),
        ("funbins", CPU, 0),
        ("UTOPIA", CPU, 0),
        ("LGBM", CPU, 0),
        ("RandomForest", CPU, 0),
        ("SummerSlasher", CPU, 0),
        ("MineCraft", CPU, 0),
    ]
    
    no_nbin = {
        "SummerSlasher",
        "MineCraft",
    }


    for (classifier, gpu_or_cpu, config_index) in classifiers:
        for bands in ['riz', 'griz']:
            if classifier in no_nbin:
                bins = [0]
            else:
                bins = [3, 5, 7, 9]

            for nbin in bins:
                name = f"{classifier}_{nbin}_{bands}_{config_index}"
                print(name)
                task = {"classifier":classifier, "bands":bands, "nbin":nbin, "config_index":config_index}
                queue.add_job(name, task, subset=gpu_or_cpu)


def task(name, bands, classifier, nbin, config_index):
    f = open(f"logs/{name}_log.txt", "w")
    print("Running",name, classifier, nbin)
    if classifier in off_by_one:
        nbin -= 1
    cmd = f"python bin/run_one_batch.py {name} {bands} {classifier} {nbin} {config_index}"
    status = subprocess.call(cmd.split(), stderr=subprocess.STDOUT, stdout=f)
    return status


import argparse
parser = argparse.ArgumentParser(description='Batch run jobs')
parser.add_argument('cpu_or_gpu', type=int, choices=[GPU, CPU], help='Run jobs for CPU (0) or GPU (1)')
parser.add_argument('--setup', action='store_true', help='Set up the DB and exit')
parser.add_argument('--query', action='store_true', help='Display next job')
parser.add_argument('--list', action='store_true', help='List remaining jobs')


def main():
    args = parser.parse_args()
    db = "classify.db"
    queue = task_queue.SlurmTaskQueue(db, task, {"classifier": str, "bands":str, "nbin": int, "config_index":int})
    if args.setup:
        setup(queue)
    elif args.query:
        job = queue.choose_next_job(subset=args.cpu_or_gpu, dry_run=True)
        print(job)
    elif args.list:
        for j in queue.list_all_remaining(args.cpu_or_gpu):
            print(j)
    else:
        queue.run_loop(subset=args.cpu_or_gpu)


if __name__ == '__main__':
    main()
