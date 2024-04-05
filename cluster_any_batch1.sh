#!/bin/bash

#SBATCH --job-name=any_batch1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 10
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 20
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 30
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 40
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 60
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 70
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 80
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 90