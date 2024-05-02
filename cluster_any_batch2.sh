#!/bin/bash

#SBATCH --job-name=any_batch2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 100 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 200 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 300 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 400 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --modelsize pythia-1.4b