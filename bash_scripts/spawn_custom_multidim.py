"""
This script is used to generate the sbatch files for the experiments with different prompt lengths and model sizes
It runs the experiments for the following dimension:
- Prompt length + Model step + Model size (different combinations of each)
"""


import argparse
parser = argparse.ArgumentParser(description='Run the evaluation script')
parser.add_argument('--basefolder', type=str, default='./', help='Base folder for the scripts')
parser.add_argument('--conda_env', type=str, default='vllm', help='Conda environment to use')
parser.add_argument('--runner', type=str, default='person1', help='Name of the person running the script')

args = parser.parse_args()

prompt_lengths = [100, 200, 300]
model_steps = ['step100000','step120000','step140000']
# model_sizes = ['pythia-1.4b','pythia-2.8b','pythia-6.9b']
model_sizes = ['pythia-1.4b-deduped','pythia-2.8b-deduped','pythia-6.9b-deduped']

runner = args.runner
base_folder = args.basefolder

deduped = True # Change this to False if you want to run the non-deduped models


# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths
custom_folder = base_folder + "multidim_scripts/"

if deduped:
    custom_folder = base_folder + "multidim_deduped_scripts/"

for prompt_len in prompt_lengths:
    for model_step in model_steps:
        for model_size in model_sizes:
            if model_size =='pythia-6.9b':
                time = "30:00:00"
            else:
                time = "20:00:00"


            base_string = f"""#!/bin/bash

#SBATCH --job-name={prompt_len}_{model_step}_{model_size}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=48000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate vllm
python experiment.py --evalfile finalidx100000.csv --complen 50 --maxtokens 50 --promptlen {prompt_len} --modelsize {model_size} --modelstep {model_step} 
echo "Done!"
"""

            print(base_string)

            with open(f'{custom_folder}{runner}_promptlen_{prompt_len}_modelsize_{model_size}_modelstep_{model_step}.sh', 'w') as f:
                f.write(base_string)
