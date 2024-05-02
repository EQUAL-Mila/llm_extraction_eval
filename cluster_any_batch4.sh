#!/bin/bash

#SBATCH --job-name=any_batch4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 100 --maxtokens 100 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 200 --maxtokens 200 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 300 --maxtokens 300 --modelsize pythia-1.4b