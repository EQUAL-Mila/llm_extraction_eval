import numpy as np
import pandas as pd
import torch

from mmap_dataset import MMapIndexedDataset

class ExtractionPromptDataset(torch.utils.data.Dataset):
    def __init__(self, pilepath, evalfile, promptlen, complen, prompttype, instructions, window):
        self.mmap_dataset = MMapIndexedDataset(pilepath, skip_warmup = True)
        self.promptlen = promptlen
        self.complen = complen
        self.prompttype = prompttype
        self.instructions = instructions
        self.window = window

        evaldf = pd.read_csv(evalfile)
        self.evalindices = evaldf['index']
        self.evalloc = evaldf['loc']
        self.rng = np.random.default_rng(42)
        
    def __len__(self):
        return len(self.evalindices)
    
    def __getitem__(self, raw_idx):
        idx, loc = int(self.evalindices[raw_idx]), int(self.evalloc[raw_idx])
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.mmap_dataset[idx]
        
        if self.prompttype=='standard':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='skipalt':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            if len(prompt)%2==0: prompt = prompt[1::2]
            else: prompt = prompt[::2]
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='end':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptpref = prompt[:-self.window], prompt[-self.window:]
            self.rng.shuffle(promptsuff)
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='corner':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptmid, promptpref = prompt[:self.window], prompt[self.window:-self.window], prompt[-self.window:]
            self.rng.shuffle(promptmid)
            prompt = np.concatenate((promptsuff, promptmid, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='cornerdel':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptmid, promptpref = prompt[:self.window], prompt[self.window:-self.window], prompt[-self.window:]
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)


        if self.instructions is not None:
            prompt = np.concatenate([np.array(self.instructions, dtype=np.int32), prompt])

        return {'prompt': torch.from_numpy(prompt), 'completion': torch.from_numpy(completion)}