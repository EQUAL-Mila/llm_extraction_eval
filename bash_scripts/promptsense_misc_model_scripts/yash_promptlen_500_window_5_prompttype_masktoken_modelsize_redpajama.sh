#!/bin/bash

#SBATCH --job-name=500_5_masktoken_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=30:00:00
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate vllm
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen 500 --window 5 --prompttype masktoken --modelsize redpajama
echo "Done!"
