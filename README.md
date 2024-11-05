
# Towards Realistic Extraction Attacks

<insert abstract?>

## Repository Structure

- **bash_scripts/**: Directory containing scripts for managing and running tasks on clusters or other environments.
- **.gitignore**: Specifies files and directories to ignore in version control.
- **README.md**: Project overview and documentation (you're reading this file).
- **config_local.yaml**: Configuration file for local settings, containing adjustable parameters for experiments.
- **download_dataset.py**: Script to download and prepare datasets for experiments.
- **download_olmdata_v2.py**: Enhanced version of the dataset downloader specifically for OLmo data.
- **experiment.py**: Main script to set up and run experiments.
- **finalidx100000.csv**: CSV file containing final index data for a subset of 100,000 entries.
- **finalidx300000_olmo_filtered.csv**: Filtered OLmo data with a final index of 300,000 entries.
- **generate_eval_file.py**: Script to generate evaluation files, possibly for model performance assessment.
- **mmap_dataset.py**: Memory-mapped dataset loader for efficient data access and handling.
- **model_utils.py**: Utility functions for model handling and evaluation.
- **prompt_loader.py**: Script to load prompts for model or evaluation purposes.
- **score.py**: Scoring script optimized for performance, evaluating models or outputs.
- **utils.py**: General utility functions used across the project.

## Getting Started

### Prerequisites

- Python 3.10

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/prakhargannu/llm_extraction_eval.git
   cd llm_extraction_eval
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Download Instructions

### Pile
**Downloading Pile Dataset**
Follow the steps here to install the required packages and then download the dataset - https://github.com/EleutherAI/pythia?tab=readme-ov-file#exploring-the-dataset

If the 'hf_hub_download' function doesn't work (as it didn't for me), you can use the alternative 'snapshot_download' (see the file in this repo - `download_dataset.py`; Make sure to change the path according to your cluster location)

**Preparing the Pile Dataset**
Continue following the steps in the above link to 'unshard' the dataset after downloading it. More specifically,
```
git clone https://github.com/EleutherAI/pythia
cd pythia
python3 utils/unshard_mmap.py --input_file "path/to/download/folder/document-00000-of-00020.bin" --num_shards 21 --output_dir "path/to/final/dataset/folder/"
cp path/to/download/folder/document.idx path/to/final/dataset/folder/document.idx
```

**Using index file**
An index file allows you index the downloaded data in a particular way. As per the paper, we uniformly sample 100,000 sequences from the first 100k steps (batches) of the training data. The indices for these steps are stored in `finalidx100000.csv`

  
### Olmo

**Configuration**
Please follow the instructions at the (original repo)[https://github.com/allenai/OLMo]
```git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[all]
```
To download the dataset(cache it) used for olmo, you can also use the `download_olmodata.py`. This file also updates the `config_local.yaml` required to load specific indexes of the training dataset. Edit `config_local.yaml` to customize parameters such as paths, model settings, and other experiment configurations.
To use the default configs, refer to this (link)[https://github.com/allenai/OLMo/blob/main/configs/official/OLMo-7B.yaml].

**Using index file**
Filtered indices used for olmo are stored in `finalidx300000_olmo_filtered.csv`. Steps to filter are outlined in the paper.

## Usage
To run an experiment, you can run the following code:


**Changing prompt length**
The `promptlen` flag specifies the input prompt length you want the model to test on.   
```
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 10
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 20
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 30
```

**Changing completion length**
The `complen` flag specifies the completion length (for the model), and the `maxtokens` is the max number of tokens you request from the model completion.  

python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 10 --maxtokens 10
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 20 --maxtokens 20
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 30 --maxtokens 30
**Changing Prompt type**

python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype skipalt
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype end
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype corner
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype cornerdel

## SLURM Usage
For ease of replication of our experiments with model size, model checkpoints and varying prompt lenghts, we write custom scripts that you can schedule on your respective slurm server(s).
One can modify the following scripts to match their local-environment configurations.


## Contributing
If you want to contribute, feel free to create a pull request or reach out to the maintainers.

## Contact
For questions related to the repository, please contact @prakhar at prakhargannu@gmail.com, or @sert121 at yash.more@alumni.ashoka.edu.in

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
