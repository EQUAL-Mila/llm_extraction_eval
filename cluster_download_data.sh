#!/bin/bash

#SBATCH --job-name=unshard_pile
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python utils/unshard_memmap.py --input_file /home/mila/p/prakhar.ganesh/scratch/pile-pythia/datasets--EleutherAI--pile-deduped-pythia-preshuffled/snapshots/4647773ea142ab1ff5694602fa104bbf49088408/document-00000-of-00020.bin --num_shards 21 --output_dir /home/mila/p/prakhar.ganesh/scratch/pile-pythia/pile-deduped/