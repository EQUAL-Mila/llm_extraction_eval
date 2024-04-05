#!/bin/bash

#SBATCH --job-name=any_batch6
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --instructions assistant
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --instructions short
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --instructions follows
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --instructions excerpt
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --instructions dan
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.1
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.2
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.3
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.4
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.5
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.6
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.7
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.8
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.9
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 1.0