#!/bin/bash

#SBATCH --job-name=a100_batch3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 3
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 4
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 5