prompt_lengths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
model_steps = ['step300000','step320000','step340000', 'step360000','step380000','step400000','step420000','step440000','step460000', 'step480000', 'step500000']

runner = 'yash'
deduped = False

if runner is 'prakhar':
    base_folder = "./"

elif runner is 'yash':
    base_folder = "./"

# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths

custom_folder = base_folder + "olmo_prompt_len_scripts/"

for prompt_len in prompt_lengths:

    if prompt_len <250:
        time = "20:00:00"

    if prompt_len >= 250:
        time = "30:00:00"


    base_string = f"""#!/bin/bash

#SBATCH --job-name={prompt_len}_olmo_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yash.more@mila.quebec

module load miniconda/3
conda activate vllm
python experiment.py --evalfile finalidx300000_olmo.csv --modelstep step500000 --promptlen {prompt_len} --modelsize olmo-7b
echo "Done!"
"""

    print(base_string)

    with open(f'{custom_folder}{runner}_promptlen_{prompt_len}.sh', 'w') as f:
        f.write(base_string)

# --------------------------------------------------------
# Making the sbatch file for the experiments model steps
custom_folder = base_folder + "olmo_model_step_scripts/"

for model_step in model_steps:
    time = "20:00:00"

    base_string = f"""#!/bin/bash

#SBATCH --job-name={model_step}_olmo_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate vllm
python experiment.py --evalfile finalidx300000_olmo.csv --modelstep {model_step} --modelsize olmo-7b
echo "Done!"
"""

    print(base_string)
    with open(f'{custom_folder}{runner}_modelstep_{model_step}.sh', 'w') as f:
        f.write(base_string)


# -------------------------------
