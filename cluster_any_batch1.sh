#!/bin/bash

#SBATCH --job-name=any_batch1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 10 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 20 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 30 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 40 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 60 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 70 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 80 --modelsize pythia-1.4b
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 90 --modelsize pythia-1.4b