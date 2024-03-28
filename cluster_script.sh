#!/bin/bash

#SBATCH --job-name=demo
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile demoidx100000.csv --batchsize 10000 --promptlen 50
python experiment.py --evalfile demoidx100000.csv --batchsize 10000 --promptlen 50 --instructions demo
python experiment.py --evalfile demoidx100000.csv --batchsize 10000 --promptlen 200
python experiment.py --evalfile demoidx100000.csv --batchsize 10000 --promptlen 200 --instructions demo