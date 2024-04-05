#!/bin/bash

#SBATCH --job-name=any_batch2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 100
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 200
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 300
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 400
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500