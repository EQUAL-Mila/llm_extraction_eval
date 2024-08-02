prompt_lengths = [100,300,500]
window_sizes = [5,10,15,20]
prompt_types = ['reduce', 'masktoken']

runner = 'yash'
if runner is 'prakhar':
    base_folder = "/network/scratch/p/prakhar.ganesh/"

elif runner is 'yash':
    base_folder = "./"

# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths
custom_folder = base_folder + "promptsense_scripts/"

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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate vllm
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} --window {window_size} --prompttype {prompt_type} 
echo "Done!"
"""

            print(base_string)

            with open(
                    f'{custom_folder}{runner}_promptlen_{prompt_len}_window_{window_size}_prompttype_{prompt_type}.sh',
                    'w') as f:
                f.write(base_string)
