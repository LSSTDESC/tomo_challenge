#!/bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -L SCRATCH
#SBATCH --job-name=u_2bins
#SBATCH --output=/global/cscratch1/sd/abault/tomo_challenge/rf_uniform/u_2bins.out
#SBATCH --error=/global/cscratch1/sd/abault/tomo_challenge/rf_uniform/u_2bins.err
#SBATCH --constraint=haswell

source activate tomo
python /global/cscratch1/sd/abault/tomo_challenge/bin/challenge.py /global/cscratch1/sd/abault/tomo_challenge/rf_uniform/riz_2bins.yaml