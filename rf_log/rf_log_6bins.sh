#!/bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH --job-name=log_6bins
#SBATCH --output=/global/cscratch1/sd/abault/tomo_challenge/rf_log/log_6bins.out
#SBATCH --error=/global/cscratch1/sd/abault/tomo_challenge/rf_log/log_6bins.err
#SBATCH --constraint=haswell

source activate tomo
python /global/cscratch1/sd/abault/tomo_challenge/bin/challenge.py /global/cscratch1/sd/abault/tomo_challenge/rf_log/riz_6bins.yaml