import numpy as np
import pandas as pd
import torch

import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from tqdm import tqdm

from mmap_dataset import MMapIndexedDataset


class ExtractionPromptDataset(torch.utils.data.Dataset):

    def __init__(self,
                 pilepath,
                 evalfile,
                 promptlen,
                 complen,
                 prompttype,
                 instructions,
                 window,
                 mask_token,
                 dataset_type='pythia'):
        self.dataset_type = dataset_type

        if dataset_type == 'pythia':
            self.mmap_dataset = MMapIndexedDataset(pilepath, skip_warmup=True)
        else: #
            data_order_file_path = cached_path(
                "https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy"
            )            
            train_config_path = "./config_local.yaml" #same as "./OLMo/configs/official/OLMo-7B.yaml"

            cfg = TrainConfig.load(train_config_path)
            self.dataset = build_memmap_dataset(cfg, cfg.data)
            self.global_indices = np.memmap(data_order_file_path,
                                            mode="r+",
                                            dtype=np.uint32)

            self.global_indices = np.arange(0,len(self.dataset)) # lets try this

            print("\n-- Global indices: --\n")
            print(self.global_indices[:10])

        self.promptlen = promptlen
        self.complen = complen
        self.prompttype = prompttype
        self.instructions = instructions
        self.window = window
        self.mask_token = mask_token

        evaldf = pd.read_csv(evalfile)
        self.evalindices = evaldf['index']
        self.evalloc = evaldf['loc']
        self.rng = np.random.default_rng(42)
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.evalindices)

    def __getitem__(self, raw_idx):

        idx, loc = int(self.evalindices[raw_idx]), int(self.evalloc[raw_idx])
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.dataset_type == 'pythia':
            sentence = self.mmap_dataset[idx]
        else:
            try:
                index = self.global_indices[idx]
                sentence = self.dataset[index]["input_ids"]
            except Exception as e:
                print(f"Error: {e}")
                return None

        if self.prompttype == 'standard':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        elif self.prompttype == 'reduce':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            # remove random tokens from the prompt
            indices = np.random.choice(
                len(prompt),
                int((self.window / 100) * len(prompt)),
                replace=False)  # window here refefs to the percentage
            prompt = np.delete(prompt, indices)
            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        elif self.prompttype == 'masktoken':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            # choose random tokens from the prompt
            indices = np.random.choice(len(prompt),
                                       int((self.window / 100) * len(prompt)),
                                       replace=False)  #
            prompt[indices] = self.mask_token

            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        elif self.prompttype == 'skipalt':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            if len(prompt) % 2 == 0: prompt = prompt[1::2]
            else: prompt = prompt[::2]
            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        elif self.prompttype == 'end':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            promptsuff, promptpref = prompt[:-self.window], prompt[-self.
                                                                   window:]
            self.rng.shuffle(promptsuff)
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        elif self.prompttype == 'corner':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            promptsuff, promptmid, promptpref = prompt[:self.window], prompt[
                self.window:-self.window], prompt[-self.window:]
            self.rng.shuffle(promptmid)
            prompt = np.concatenate((promptsuff, promptmid, promptpref))
            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        elif self.prompttype == 'cornerdel':
            prompt = np.array(sentence[loc - self.promptlen:loc],
                              dtype=np.int32)
            promptsuff, promptmid, promptpref = prompt[:self.window], prompt[
                self.window:-self.window], prompt[-self.window:]
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc + self.complen],
                                  dtype=np.int32)

        if self.instructions is not None:
            prompt = np.concatenate(
                [np.array(self.instructions, dtype=np.int32), prompt])

        return {
            'prompt': torch.from_numpy(prompt),
            'completion': torch.from_numpy(completion)
        }
