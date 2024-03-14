import numpy as np
import pandas as pd
import torch

from mmap_dataset import MMapIndexedDataset

class ExtractionPromptDataset(torch.utils.data.Dataset):
    def __init__(self, pilepath, evalfile, promptlen, complen, prompttype, instructions):
        self.mmap_dataset = MMapIndexedDataset(pilepath, skip_warmup = True)
        self.promptlen = promptlen
        self.complen = complen
        self.prompttype = prompttype
        self.instructions = instructions

        evaldf = pd.read_csv(evalfile)
        self.evalindices = evaldf['index']
        self.evalloc = evaldf['loc']

    def __len__(self):
        return len(self.evalindices)
    
    def __getitem__(self, raw_idx):
        idx, loc = int(self.evalindices[raw_idx]), int(self.evalloc[raw_idx])
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.mmap_dataset[idx]
        
        ### TODO: All Different Variations of Prompt Creation go here!! 
        ### We can also create separate functions for them for readability.
        if self.prompttype=='standard':
            prompt = np.array(sentence[loc:loc+self.promptlen], dtype=np.int32)
            completion = np.array(sentence[loc+self.promptlen:loc+self.promptlen+self.complen], dtype=np.int32)

        if self.instructions is not None:
            prompt = np.concatenate([np.array(self.instructions, dtype=np.int32), prompt])

        return {'prompt': torch.from_numpy(prompt), 'completion': torch.from_numpy(completion)}