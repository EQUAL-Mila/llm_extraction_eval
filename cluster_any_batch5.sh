#!/bin/bash

#SBATCH --job-name=any_batch5
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=30:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype skipalt
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype end
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype corner
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype cornerdel