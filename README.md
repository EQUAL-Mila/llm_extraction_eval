
# Towards Realistic Extraction Attacks

<insert repo structure/understanding>

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
   cd your_project
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

**Downloading the Olmo Dataset**



## Configuration

Edit `config_local.yaml` to customize parameters such as paths, model settings, and other experiment configurations.



  - To fetch the dolma dataset, you can use `download_
- **Running Experiments**: Execute `experiment.py` to start the main experiment.
- **Evaluation**: Use `generate_eval_file.py` and `score.py` to evaluate models or outputs.

## Contributing

If you want to contribute, feel free to create a pull request or reach out to the maintainers.


## Contact
For questions related to the repository, please contact @prakhar at prakhargannu@gmail.com, or @sert121 at yash.more@alumni.ashoka.edu.in

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- [x] Keep increasing prompt length and see how the results change. Does it keep on increasing always? Does it plateau? Does it maybe even decrease for some prompts? (Code note: Simply change the values of the flag `promptlen` and see how things change)
- [x] Skip every alternate token in the prompt and see how extraction rates change. The extraction rate should still be some reasonably high value, to show that it's the context we care about, not the exact order of tokens. (Code note: Simply change the flag `prompttype` to `skipalt` to skip alternate tokens in the prompt)
- [x] Keep the last few tokens as is, but randomly shuffle all tokens at the start. Same expectation as above. (Code note: Simply change the flag `prompttype` to `end50` to shuffle the initial tokens in the prompt, while keeping the last 50 tokens fixed)
- [x] Keep the last few and the first few tokens as is, but randomly shuffle all tokens in the middle, or even remove them. Same expectation as above. (Code note: Simply change the flag `prompttype` to `corner50` to shuffle the middle tokens in the prompt, while keeping the last and first 50 tokens each fixed. Change the flag to `corner50del` for the same but middle tokens are deleted instead of being shuffled)

- [x] Introduce instructions into the prompt to further elicit information out of the model. Are there instructions that can help with data extraction? (Code note: Each instruction set has its own identifier. Use the flag `instructions` and the identifier as input to add those instructions to the prompt. Check function `get_instruction_ids` in file `utils.py` to see those instruction identifiers, or add new instructions)

- [x] Increase beam length and max tokens to see if the generation changes. (Code note: Simply set the flags `beamwidth` and `maxtokens` to change the relevant parameters)

- [x] Perform extraction attacks over different time steps. (Code note: Simple change the flag `modelstep` to `stepxxxxxx`, where `xxxxxx` can be anything between 100000 to 140000, at steps of 1000 only)

- [x] Changing scoring from perfect match at a fixed length to the length of perfect match. Observe the distribution. (Code note: Use the `scoring` flag `length` to check the length of exact match)
- [ ] Incorporate levenshtein distance or other measures to study insertions, deletions or substitutions. (Code note: Use the `scoring` flag `levenshtein` to check the levenshtein distance between generated text and original completion)
- [ ] Study semantic similarity instead of token level matching, by using cosine similarity. (Code note: TODO. This is not implemented yet)
- [ ] Print out original sentences, given indices, to qualitatively analyse the results. (Code note: TODO. This is not implemented yet)

