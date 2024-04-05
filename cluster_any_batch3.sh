#!/bin/bash

#SBATCH --job-name=any_batch3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 10 --maxtokens 10
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 20 --maxtokens 20
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 30 --maxtokens 30
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 40 --maxtokens 40
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 60 --maxtokens 60
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 70 --maxtokens 70
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 80 --maxtokens 80
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 90 --maxtokens 90