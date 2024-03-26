## Experiments TODO

- [ ] Changing scoring from perfect match at a fixed length to the length of perfect match. Observe the distribution. (Code note: TODO. This is not implemented yet)
- [ ] Incorporate levenshtein distance or other measures to study insertions, deletions or substitutions. (Code note: TODO. This is not implemented yet)
- [ ] Study semantic similarity instead of token level matching, by using cosine similarity. (Code note: TODO. This is not implemented yet)

- [ ] Keep increasing prompt length and see how the results change. Does it keep on increasing always? Does it plateau? Does it maybe even decrease for some prompts? (Code note: Simply change the values of the flag `promptlen` and see how things change)
- [ ] Skip every alternate token in the prompt and see how extraction rates change. The extraction rate should still be some reasonably high value, to show that it's the context we care about, not the exact order of tokens. (Code note: Simply change the flag `prompttype` to `skipalt` to skip alternate tokens in the prompt)
- [ ] Keep the last few tokens as is, but randomly shuffle all tokens at the start. Same expectation as above. (Code note: TODO. This is not implemented yet)
- [ ] Keep the last few and the first few tokens as is, but randomly shuffle all tokens in the middle, or even remove them. Same expectation as above. (Code note: TODO. This is not implemented yet)

- [ ] Introduce instructions into the prompt to further elicit information out of the model. Are there instructions that can help with data extraction? (Code note: TODO. This is not implemented yet)


- [ ] Increase beam length and max tokens to see if the generation changes. (Code note: Simply set the flags `beamwidth` and `maxtokens` to change the relevant parameters)

- [ ] Perform extraction attacks over different time steps. (Code note: Simple change the flag `modelstep` to `stepxxxxxx`, where `xxxxxx` can be anything between 100000 to 140000, at steps of 1000 only)

## Dataset Download Instructions

### Downloading Dataset
Follow the steps here to install the required packages and then download the dataset - https://github.com/EleutherAI/pythia?tab=readme-ov-file#exploring-the-dataset

If the 'hf_hub_download' function doesn't work (as it didn't for me), you can use the alternative 'snapshot_download' (see the file in this repo - `download_dataset.py`; Make sure to change the path according to your cluster location)

### Preparing Dataset
Continue following the steps in the above link to 'unshard' the dataset after downloading it. More specifically,
```
git clone https://github.com/EleutherAI/pythia
cd pythia
python3 utils/unshard_mmap.py --input_file "path/to/download/folder/document-00000-of-00020.bin" --num_shards 21 --output_dir "path/to/final/dataset/folder/"
cp path/to/download/folder/document.idx path/to/final/dataset/folder/document.idx
```