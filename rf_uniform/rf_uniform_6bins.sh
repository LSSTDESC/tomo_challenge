#!/bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -L SCRATCH
#SBATCH --job-name=riz_u_6bins
#SBATCH --output=/global/cscratch1/sd/abault/tomo_challenge/rf_uniform/u_6bins.out
#SBATCH --error=/global/cscratch1/sd/abault/tomo_challenge/rf_uniform/u_6bins.err
#SBATCH --constraint=haswell

source activate tomo
python /global/cscratch1/sd/abault/tomo_challenge/bin/challenge.py /global/cscratch1/sd/abault/tomo_challenge/rf_uniform/riz_6bins.yaml