#!/bin/bash

#SBATCH --job-name=pythia-2.8b_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=7:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --modelsize pythia-2.8b 
echo "Done!"
