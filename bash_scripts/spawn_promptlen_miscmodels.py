"""
This script is used to generate the sbatch files for the experiments involving looking at the effects of prompt length for miscellaneous models.
The script generates the sbatch files to run experiments for the following dimensions:
- Prompt length + Model size  (different combinations of each)
"""

import argparse
parser = argparse.ArgumentParser(description='Run the evaluation script')
parser.add_argument('--basefolder', type=str, default='./', help='Base folder for the scripts')
parser.add_argument('--conda_env', type=str, default='vllm', help='Conda environment to use')
parser.add_argument('--runner', type=str, default='person1', help='Name of the person running the script')

args = parser.parse_args()

prompt_lengths = [ 100, 200, 300, 400, 500]
model_steps = ['step100000','step105000','step110000', 'step115000','step120000','step125000','step130000','step135000','step140000']
model_sizes = ['gemma2', 'gemma7','llama', 'olmo','phi','mpt','gpt2','redpajama','falcon']

runner = args.runner
base_folder = args.basefolder


# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths
custom_folder = base_folder + "prompt_miscmodel_scripts/"
for prompt_len in prompt_lengths:
    for model_size in model_sizes:

        if prompt_len < 250 and model_size in ['gemma2', 'phi', 'gpt2']:
            time = "15:00:00"

        if prompt_len < 250 and model_size in [
                'falcon', 'redpajama', 'mpt', 'gemma7','llama', 'olmo'
        ]:
            time = "25:00:00"

        if prompt_len >= 250 and model_size in [
            'falcon', 'redpajama', 'mpt', 'gemma7', 'llama', 'olmo'
        ]:
            time = "35:00:00"
        if prompt_len >= 250 and model_size in ['gemma2', 'phi', 'gpt2', 'llama']:
            time = "25:00:00"



        base_string = f"""#!/bin/bash
#SBATCH --job-name={prompt_len}_miscmodel_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} --modelsize {model_size}
echo "Done!"
"""

        if model_size in ['gemma2', 'gemma7', 'mpt']:
            base_string = f"""#!/bin/bash
#SBATCH --job-name={prompt_len}_miscmodel_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} --modelsize {model_size}
echo "Done!"
"""

        print(base_string)

        with open(f'{custom_folder}{runner}_promptlen_{prompt_len}_modelsize_{model_size}.sh',
                  'w') as f:
            f.write(base_string)
