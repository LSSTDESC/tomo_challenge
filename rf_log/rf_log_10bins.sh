#!/bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -L SCRATCH
#SBATCH --job-name=log_10bins
#SBATCH --output=/global/cscratch1/sd/abault/tomo_challenge/rf_log/log_10bins.out
#SBATCH --error=/global/cscratch1/sd/abault/tomo_challenge/rf_log/log_10bins.err
#SBATCH --constraint=haswell

source activate tomo
python /global/cscratch1/sd/abault/tomo_challenge/bin/challenge.py /global/cscratch1/sd/abault/tomo_challenge/rf_log/riz_10bins.yaml