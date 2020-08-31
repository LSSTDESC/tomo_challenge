#!/bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -L SCRATCH
#SBATCH --job-name=u_4bins
#SBATCH --output=/global/cscratch1/sd/abault/tomo_challenge/rf_uniform/u_4bins.out
#SBATCH --error=/global/cscratch1/sd/abault/tomo_challenge/rf_uniform/u_4bins.err
#SBATCH --constraint=haswell

source activate tomo
python /global/cscratch1/sd/abault/tomo_challenge/bin/challenge.py /global/cscratch1/sd/abault/tomo_challenge/rf_uniform/riz_4bins.yaml