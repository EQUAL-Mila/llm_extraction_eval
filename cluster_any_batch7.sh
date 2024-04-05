#!/bin/bash

#SBATCH --job-name=any_batch7
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step100000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step105000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step110000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step115000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step120000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step125000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step130000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step135000