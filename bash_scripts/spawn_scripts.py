


prompt_lengths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
model_steps = ['step100000','step105000','step110000', 'step115000','step120000','step125000','step130000','step135000','step140000']
model_sizes = ['pythia-1b','pythia-1.4b','pythia-2.8b','pythia-6.9b','pythia-12b']


runner = 'prakhar'

if runner is 'prakhar':
    base_folder = "./"
    
elif runner is 'yash':
    base_folder = "./"

# --------------------------------------------------------
# Making the sbatch file for the experiments prompt lengths
custom_folder = base_folder + "prompt_len_scripts/"
for prompt_len in prompt_lengths:

    if prompt_len <250:
        time = "20:00:00"

    if prompt_len >= 250:
        time = "30:00:00"
    

    base_string = f"""
#!/bin/bash

#SBATCH --job-name={prompt_len}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --promptlen {prompt_len} 
echo "Done!"
"""

    print(base_string)

    with open(f'{custom_folder}{runner}_promptlen_{prompt_len}.sbatch', 'w') as f:
        f.write(base_string)

# Making the sbatch file for the experiments model steps
custom_folder = base_folder + "model_step_scripts/"
for model_step in model_steps:
    time = "15:00:00"
    
    base_string = f"""
#!/bin/bash

#SBATCH --job-name={model_step}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --modelstep {model_step} 
echo "Done!"
"""

# --------------------------------------------------------
# Making the sbatch file for the experiments model sizes
custom_folder = base_folder + "model_size_scripts/"
for model_size in model_sizes:

    if model_size is 'pythia-1b':
        time = "4:00:00"
    if model_size is 'pythia-1.4b':
        time = "5:00:00"
    if model_size is 'pythia-2.8b':
        time = "7:00:00"
    if model_size is 'pythia-6.9b':
        time = "15:00:00"
    if model_size is 'pythia-12b':
        time = "35:00:00"


    base_string = f"""
#!/bin/bash

#SBATCH --job-name={model_size}_pythia_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time={time}
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate pythia
python experiment.py --evalfile finalidx100000.csv --complen 500 --maxtokens 500 --modelsize {model_size} 
echo "Done!"
"""

    print(base_string)

    with open(f'{custom_folder}{runner}_modelsize_{model_size}.sbatch', 'w') as f:
        f.write(base_string)
