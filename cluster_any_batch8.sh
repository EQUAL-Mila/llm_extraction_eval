#!/bin/bash

#SBATCH --job-name=any_batch8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 400 --maxtokens 400 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 500 --maxtokens 500 --modelsize pythia-1.4b