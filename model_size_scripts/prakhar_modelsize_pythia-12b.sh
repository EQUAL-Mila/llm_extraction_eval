
#!/bin/bash

#SBATCH --job-name=pythia-12b_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=35:00:00
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --modelsize pythia-12b 
echo "Done!"