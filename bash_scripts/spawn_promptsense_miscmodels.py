"""
This script runs the experiments for prompt sensitivity.
It generates the sbatch files for the experiments varying on the following dimensions:
- Prompt length + Window size + Prompt type + Model type (except for pythia and olmo models)
(combinations of each type)
"""

import argparse
parser = argparse.ArgumentParser(description='Run the evaluation script')
parser.add_argument('--basefolder', type=str, default='./', help='Base folder for the scripts')
parser.add_argument('--conda_env', type=str, default='vllm', help='Conda environment to use')
parser.add_argument('--runner', type=str, default='person1', help='Name of the person running the script')

args = parser.parse_args()


prompt_lengths = [100,300,500]
window_sizes = [5,10,15,20]
prompt_types = ['reduce', 'masktoken']
model_sizes = ['gemma2', 'gemma7', 'phi','mpt','gpt2','redpajama','falcon']

runner = args.runner
base_folder = args.basefolder


# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths
custom_folder = base_folder + "promptsense_misc_model_scripts/"


for model_size in model_sizes:
    for prompt_len in prompt_lengths:
        for window_size in window_sizes:
            for prompt_type in prompt_types:
                if prompt_len == 100:
                    time = "20:00:00"
                else:
                    time = "30:00:00"

                base_string = f"""#!/bin/bash

#SBATCH --job-name={prompt_len}_{window_size}_{prompt_type}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} --window {window_size} --prompttype {prompt_type} --modelsize {model_size}
echo "Done!"
"""
                if model_size in ['gemma2', 'gemma7', 'mpt']:
                    base_string = f"""#!/bin/bash

#SBATCH --job-name={prompt_len}_{window_size}_{prompt_type}_modelsize_{model_size}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=48000M
#SBATCH --gres=gpu:a100l:1

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} --window {window_size} --prompttype {prompt_type} --modelsize {model_size}
echo "Done!"
"""
                print(base_string)

                with open(
                        f'{custom_folder}{runner}_promptlen_{prompt_len}_window_{window_size}_prompttype_{prompt_type}_modelsize_{model_size}.sh',
                        'w') as f:
                    f.write(base_string)
