import contextlib
import sqlite3
import subprocess
import psutil
import os
import sys

@contextlib.contextmanager
def database_lock(db, timeout=30):
    con = sqlite3.connect(db, timeout=timeout)
    try:
        # allows dict-like access
        con.row_factory = sqlite3.Row
        con.isolation_level = 'EXCLUSIVE'
        con.execute('BEGIN EXCLUSIVE')
        cursor = con.cursor()
        yield cursor
    finally:
        con.close()



class TaskQueue:
    def __init__(self, database, job_function, job_args, subsets=None):
        """

        job_args: dict
            mapping from argument names to types
        """
        self.job_function = job_function
        self.job_args = job_args
        self.subsets = subsets or [0]
        self.database = database

        if not os.path.exists(database):
            self.init_database()


    def init_database(self):
        type_map = {
            str: "text",
            int: "integer",
            float: "real",
        }
        args = {key: type_map[t] for key, t in self.job_args.items()}
        job_columns = ",".join(f"{k} {t}" for (k, t) in args.items())
        # Allow for empty arg lists
        if job_columns:
            job_columns = ", " + job_columns

        # Set up DB
        with database_lock(self.database) as cursor:
            q = """
            CREATE TABLE all_jobs (
                name text PRIMARY KEY,
                subset integer NOT NULL
                {0}
            );

            CREATE TABLE completed_jobs (
                name text PRIMARY KEY,
                process_id integer NOT NULL,
                status integer NOT NULL
            );

            CREATE TABLE started_jobs (
                name text PRIMARY KEY,
                process_id integer NOT NULL,
                remaining_job_time real NOT NULL
            );

            """.format(job_columns)
            print(q)
            cursor.executescript(q)
            cursor.connection.commit()

    def list_all_remaining(self, subset=0):
        with database_lock(self.database) as cursor:
            q = "select * from all_jobs where subset=?"
            all_jobs = {j['name']: j for j in cursor.execute(q, [subset])}
            q = "select * from completed_jobs"
            completed_jobs = {j['name']: j for j in cursor.execute(q)}
            jobs = [j for j in all_jobs if j not in completed_jobs]
        return jobs


    def choose_next_job(self, subset=0, dry_run=False, force_job="_not_a_real_job"):
        my_process_id = self.get_my_process_id()
        my_time_left = self.get_process_time_remaining(my_process_id)
        print("Choosing task")
        with database_lock(self.database) as cursor:
            # find all jobs which:
            # are not finished
            # have not started and are still running
            # have not started and their process has died and we don't have a fair bit more time than them
            q = "select * from all_jobs where subset=?"
            all_jobs = {j['name']: j for j in cursor.execute(q, [subset])}

            q = "select * from started_jobs"
            started_jobs = {j['name']: j for j in cursor.execute(q)}

            q = "select * from completed_jobs"
            completed_jobs = {j['name']: j for j in cursor.execute(q)}

            job = None
            for name, candidate in all_jobs.items():
                if name == force_job:
                    job = candidate
                    break

                if name in completed_jobs:
                    continue

                # get info (if any) on the previous time
                # the job was started
                candidate_prev_run = started_jobs.get(name)

                # job has never been started, so star it
                if candidate_prev_run is None:
                    job = candidate
                    break

                # job has been started but not finished.
                # if job is still running, skip it
                if self.is_running(candidate_prev_run['process_id']):
                    continue

                # otherwise check how long it had left to run.
                # if we have at least 30% more time than it used
                # then give it a go
                min_time_needed = candidate_prev_run['remaining_job_time']

                if my_time_left > min_time_needed * 1.3:
                    job = candidate
                    break

                # otherwise we don't have enough time to run this
                # job, so just carry on.

            if job:
                # add start information
                if not dry_run:
                    q = """replace into started_jobs(name, process_id, remaining_job_time) values(?, ?, ?)"""
                    cursor.execute(q, (name, my_process_id, my_time_left))

            cursor.connection.commit()

        if job is None:
            return job
        else:
            return dict(job)

    def add_job(self, name, args, subset=0):
        arg_names = args.keys()
        keys = ','.join(arg_names)
        vals = [name, subset] + list(args.values())
        n = len(arg_names)
        qq = ", ?" * n
        if keys:
            keys = ", " + keys
        sql = f"INSERT INTO all_jobs(name, subset {keys}) VALUES(?, ?{qq})"
        print(sql)
        with database_lock(self.database) as cursor:
            cursor.execute(sql, vals)
            cursor.connection.commit()

    def execute(self, job):
        job = job.copy()
        job.pop("subset")
        name = job.pop("name")
        status = self.job_function(name, **job)
        return status

    def run_loop(self, subset=0):
        while True:
            job = self.choose_next_job(subset=subset)
            sys.stdout.flush()
            if not job:
                break

            status = self.execute(job)
            sys.stdout.flush()
            self.write_completed_job(job, status)


    def write_completed_job(self, job, status):
        process_id = self.get_my_process_id()
        with database_lock(self.database) as cursor:
            sql = "INSERT INTO completed_jobs(name, process_id, status) VALUES(?,?,?)"
            cursor.execute(sql, (job['name'], process_id, status))
            cursor.connection.commit()


    @staticmethod
    def get_my_process_id():
        return os.getpid()

    @classmethod
    def get_process_time_remaining(cls, process_id):
        return 1_000_000_000

    @staticmethod
    def is_running(process_id):
        return psutil.pid_exists(process_id)



class SlurmTaskQueue(TaskQueue):

    @staticmethod
    def get_my_process_id():
        job_id = os.environ['SLURM_JOB_ID']
        return job_id

    @staticmethod
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
            t += int(parts[-3]) * 3600 # hours

        return t

    @classmethod
    def get_process_time_remaining(cls, process_id):
        cmd = f"squeue -o %L -j {process_id}"
        r = subprocess.run(cmd.split(), capture_output=True)
        time_str = r.stdout.decode('ascii').split('\n')[1]
        return cls.parse_time_listing(time_str)

    @staticmethod
    def is_running(process_id):
        cmd = f"squeue -o %L -j {process_id}"
        r = subprocess.run(cmd.split(), capture_output=True)
        return r.returncode == 0


def test():
    import time
    def task(name, potato, n, x):
        time.sleep(5)
        print(f"Done {name} {potato} {n} {x}")
        if n < 10:
            return 0
        else:
            return n

    new = not os.path.exists("./test.db")
    t = SlurmTaskQueue("./test.db", task, {"potato": str, "n": int, "x":float})

    if new:
        for i in range(5, 15):
            t.add_job(f"task_{i}", {'potato':f'king{i}', 'x':44.5, 'n':i})

    t.run_loop()



if __name__ == '__main__':
    test()
