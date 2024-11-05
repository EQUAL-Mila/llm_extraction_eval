"""
This script is used to generate the sbatch files for the experiments involving looking at the effects of prompt length, model step and model size on the performance of the model.
The script generates the sbatch files to run experiments for the following dimensions:
- Prompt length
- Model step (checkpoint)
- Model size
(individual axis)
"""

import argparse
parser = argparse.ArgumentParser(description='Run the evaluation script')
parser.add_argument('--basefolder', type=str, default='./', help='Base folder for the scripts')
parser.add_argument('--conda_env', type=str, default='vllm', help='Conda environment to use')
parser.add_argument('--runner', type=str, default='person1', help='Name of the person running the script')

args = parser.parse_args()


prompt_lengths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
model_steps = ['step100000','step105000','step110000', 'step115000','step120000','step125000','step130000','step135000','step140000']
model_sizes = [
    'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'
]
model_sizes_deduped = [ 'pythia-1b-deduped', 'pythia-1.4b-deduped', 'pythia-2.8b-deduped', 'pythia-6.9b-deduped', 'pythia-12b-deduped']

deduped = False
runner = args.runner
base_folder = args.basefolder

if deduped:
    model_sizes = model_sizes_deduped
# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths
custom_folder = base_folder + "prompt_len_scripts/"

if deduped:
    custom_folder = base_folder + "prompt_len_scripts_deduped/"

for prompt_len in prompt_lengths:

    if prompt_len <250:
        time = "20:00:00"

    if prompt_len >= 250:
        time = "30:00:00"


    base_string = f"""#!/bin/bash

#SBATCH --job-name={prompt_len}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} --modelsize pythia-6.9b-deduped
echo "Done!"
"""

    print(base_string)

    with open(f'{custom_folder}{runner}_promptlen_{prompt_len}.sh', 'w') as f:
        f.write(base_string)

# --------------------------------------------------------
# Making the sbatch file for the experiments model steps
custom_folder = base_folder + "model_step_scripts/"

if deduped:
    custom_folder = base_folder + "model_step_scripts_deduped/"

for model_step in model_steps:
    time = "15:00:00"

    base_string = f"""#!/bin/bash

#SBATCH --job-name={model_step}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --modelstep {model_step} --modelsize pythia-6.9b-deduped
echo "Done!"
"""

    print(base_string)
    with open(f'{custom_folder}{runner}_modelstep_{model_step}.sh', 'w') as f:
        f.write(base_string)

# --------------------------------------------------------
# Making the sbatch file for the experiments model sizes
custom_folder = base_folder + "model_size_scripts/"
if deduped:
    custom_folder = base_folder + "model_size_scripts_deduped/"

for model_size in model_sizes:

    if model_size is 'pythia-1b':
        time = "5:00:00"
    if model_size is 'pythia-1.4b':
        time = "5:00:00"
    if model_size is 'pythia-2.8b':
        time = "8:00:00"
    if model_size is 'pythia-6.9b':
        time = "15:00:00"
    if model_size is 'pythia-12b':
        time = "35:00:00"


    base_string = f"""#!/bin/bash

#SBATCH --job-name={model_size}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate {args.conda_env}
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --modelsize {model_size} 
echo "Done!"
"""

    print(base_string)

    with open(f'{custom_folder}{runner}_modelsize_{model_size}.sh', 'w') as f:
        f.write(base_string)
