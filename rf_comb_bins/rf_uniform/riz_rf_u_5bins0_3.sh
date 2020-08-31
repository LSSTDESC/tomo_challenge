#!/bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -L SCRATCH
#SBATCH --job-name=riz_rf_u_5bins0_3
#SBATCH --output=/global/cscratch1/sd/abault/tomo_challenge/rf_comb_bins/riz_rf_u_5bins0_3.out
#SBATCH --error=/global/cscratch1/sd/abault/tomo_challenge/rf_comb_bins/riz_rf_u_5bins0_3.err
#SBATCH --constraint=haswell

source activate tomo
python /global/cscratch1/sd/abault/tomo_challenge/bin/challenge_comb.py /global/cscratch1/sd/abault/tomo_challenge/rf_comb_bins/rf_uniform/riz_rf_u_5bins0_3.yaml