import sys
import os
import subprocess
import sqlite3

def get_my_job_id():
    job_id = os.environ['SLURM_JOB_ID']
    return job_id

def parse_time_listing(time_str):
    """returns time in seconds"""
    # days part, if present
    if '-' in time_str:
        days, time_str = time_str.split('-')
        t = int(days) * 60*60*24
    else:
        t = 0

    # H, M, S parts
    parts = time_str.split(":")

    t += int(parts[-1]) # seconds

    if len(parts) > 1:
        t += int(parts[-2]) * 60 # minutes

    if len(parts) > 2:
        t += int(parts[-3]) * 3060 # hours

    return t

def check_job_time_left(job_id):
    cmd = f"squeue -o %L -j {job_id}"
    r = subprocess.run(cmd.split(), capture_output=True)
    time_str = r.stdout.decode('ascii').split('\n')[1]
    return parse_time_listing(time_str)

def is_running(job_id):
    cmd = f"squeue -o %L -j {job_id}"
    r = subprocess.run(cmd.split(), capture_output=True)
    return r.returncode == 0

def choose_next_job(database, gpu):
    my_job_id = get_my_job_id()
    my_time_left = check_job_time_left(my_job_id)
    con = sqlite3.connect(database, timeout=30)
    con.isolation_level = 'EXCLUSIVE'
    con.execute('BEGIN EXCLUSIVE')
    # Exclusive access starts here.

    # find all jobs which:
    # are not finished
    # have not started and are still running
    # have not started and their process has died and we don't have a fair bit more time than them
    cur = con.cursor()
    q = "select * from all_jobs"
    all_jobs = {j[0]: j for j in cur.execute(q)}

    q = "select * from started_jobs"
    started_jobs = {j[0]: j for j in cur.execute(q)}

    q = "select * from completed_jobs"
    completed_jobs = {j[0]: j for j in cur.execute(q)}

    job = None
    for name in all_jobs.keys():
        if name in completed_jobs:
            print("Skip completed ", name)
            continue

        gpu_job = all_jobs[name][1]
        if gpu_job != gpu:
            print("Skip gpu ", name)
            continue

        # get info (if any) on the previous time
        # the job was started
        job_start = started_jobs.get(name)

        # job has never been started, so star it
        if job_start is None:
            job = all_jobs[name]
            print("Using non-started", name)
            break

        # job has been started but not finished.
        prev_id = job_start[1]

        # if job is still running, skip it
        if is_running(prev_id):
            print("Skipping still-running", name)
            continue

        # otherwise check how long it had left to run.
        # if we have at least 30% more time than it used
        # then give it a go
        min_time_needed = job_start[2]
        if my_time_left > min_time_needed * 1.3:
            job = all_jobs[name]
            print("Using previously outrun", name)
            break
        print("Skipping previously outrun", name)

        # otherwise we don't have enough time to run this
        # job, so just carry on.
        

    if job:
        # add start information
        q = "replace into started_jobs(name, job_id, remaining_job_time) values(?, ?, ?);"
        print(q)
        cur.execute(q, (job[0], my_job_id, my_time_left))

    else:
        print("No jobs could be found to start")


    con.commit()
    con.close()

    return job      



def create_initial_db(database):
    gpu_classifiers = [
        "NeuralNetwork",
        "Autokeras_LSTM",
        "CNN",
        "ENSEMBLE1",
        "Flax_LSTM",
        "JaxCNN",
        "JaxResNet",
        "LSTM",
        "TCN",
        "ZotBin",
        "ZotNet",
        "myCombinedClassifiers",
    ]

    cpu_classifiers = [
        "IBandOnly",
        "Random",
        "mlpqna",
        "ComplexSOM",
        "SimpleSOM",
        "PCACluster",
        "GPzBinning",
        "funbins",
        "UTOPIA",
        "LGBM",
        "RandomForest",
    ]

        # "PQNLD",
        # "TensorFlow_DBN",
        # "TensorFlow_FFNN",

    no_nbin_classifiers = [
        "SummerSlasher",
        "MineCraft",
    ]

    indices = {
        'NeuralNetwork': [1,2],
    }



    con = sqlite3.connect(database)
    cur = con.cursor()
    sql = """
    CREATE TABLE all_jobs (
        name varchar(64) PRIMARY KEY,
        gpu boolean NOT NULL,
        classifier varchar(64) NOT NULL,
        nbin integer NOT NULL,
        index1 integer NOT NULL
    );
    """
    cur.execute(sql)

    sql = """INSERT INTO all_jobs(name, gpu, classifier, nbin, index1) VALUES(?,?,?,?,?)"""
    for name in cpu_classifiers:
        for nbin in [3,5,7,9]:
            print(name, nbin)
            if name in indices:
                for index in indices[name]:
                    cur.execute(sql, (f"{name}_{nbin}_{index}", False, name, nbin, index))
            else:
                cur.execute(sql, (f"{name}_{nbin}_0", False, name, nbin, 0))

    for name in gpu_classifiers:
        for nbin in [3,5,7,9]:
            print(name, nbin)
            if name in indices:
                for index in indices[name]:
                    cur.execute(sql, (f"{name}_{nbin}_{index}", True, name, nbin, index))
            else:
                cur.execute(sql, (f"{name}_{nbin}_0", True, name, nbin, 0))

    for name in no_nbin_classifiers:
        print(name, 0)
        cur.execute(sql, (f"{name}_0", True, name, 0, 0))


    sql = """
    CREATE TABLE completed_jobs (
        name varchar(64) PRIMARY KEY,
        job_id integer NOT NULL,
        status integer NOT NULL
    );

    """
    cur.execute(sql)

    sql = """
    CREATE TABLE started_jobs (
        name varchar(64) PRIMARY KEY,
        job_id integer NOT NULL,
        remaining_job_time real NOT NULL
    );

    """
    cur.execute(sql)

    con.commit()
    con.close()



def execute_job(job):
    print('execute', job)
    name, _, classifier, nbin, index = job
    f = open(f"logs/{name}_log.txt", "w")
    print("Running",name, classifier, nbin)
    cmd = f"python bin/run_one_batch.py {name} {classifier} {nbin} {index}"
    status = subprocess.call(cmd.split(), stderr=subprocess.STDOUT, stdout=f)
    print(f"Job {name} finished with status {status}")
    return status

def write_completed_job(db, job, status):
    print('complete', job)
    name, _, classifier, nbin, index = job
    job_id = get_my_job_id()
    con = sqlite3.connect(db, timeout=30)
    con.isolation_level = 'EXCLUSIVE'
    con.execute('BEGIN EXCLUSIVE')
    cur = con.cursor()

    sql = """INSERT INTO completed_jobs(name, job_id, status) VALUES(?,?,?)""" 

    cur.execute(sql, (name, job_id, status))
    con.commit()
    con.close()

def main(db, gpu):
    while True:
        job = choose_next_job(db, gpu)
        sys.stdout.flush()
        if not job:
            break

        status = execute_job(job)
        sys.stdout.flush()
        write_completed_job(db, job, status)


if __name__ == '__main__':
    db = 'db.sqlite3'
    if sys.argv[1] == 'gpu':
        gpu = True
    elif sys.argv[1] == 'cpu':
        gpu = False
    else:
        raise ValueError("say gpu or cpu")
    main(db, gpu)
    # create_initial_db(db)


# a little sqlite db containing table:
# all jobs
#   - name
#   - cpu_or_gpu
#   - classifier
#   - nbin
#   - index
# started jobs
#   - name
#   - job_id
#   - remaining_job_time
# completed jobs
#   - name
#   - job_id
#   - status


# for each class of method (GPU, CPU)
# get list of things to run
# check what has been run
# check if last thing failed or used full time
# get next thing to run
# run in subprocess

# record what time the job starts, and how long is left on the timer
