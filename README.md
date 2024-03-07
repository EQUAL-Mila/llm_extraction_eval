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