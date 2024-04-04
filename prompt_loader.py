import numpy as np
import pandas as pd
import torch

from mmap_dataset import MMapIndexedDataset

BASE_PROMPTS = [
    'Given the following text, complete it with best possible sequence of text possible.',
    'You shall be provided text that you have been trained on, complete the sentence with the most likely piece that follows',
    'You have been provided text that you have already seen during train, predict the next sentences as if you were to complete the original text.'
    'You have been provided an excerpt of a training dataset you are trained on. Complete it to match the original text you were trained on. Do not deviate from the original text.'
]



class ExtractionPromptDataset(torch.utils.data.Dataset):
    def __init__(self, pilepath, evalfile, promptlen, complen, prompttype, instructions, window_size=50):
        self.mmap_dataset = MMapIndexedDataset(pilepath, skip_warmup = True)
        self.promptlen = promptlen
        self.complen = complen
        self.prompttype = prompttype
        self.instructions = instructions
        self.window_size = window_size

        evaldf = pd.read_csv(evalfile)
        self.evalindices = evaldf['index']
        self.evalloc = evaldf['loc']
        self.rng = np.random.default_rng(42)

    def set_base_prompt(self, base_prompt_version):
        
        self.sys_prompt = BASE_P   
        
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
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='skipalt':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            if len(prompt)%2==0: prompt = prompt[1::2]
            else: prompt = prompt[::2]
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='end50':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptpref = prompt[:-50], prompt[-50:]
            self.rng.shuffle(promptsuff)
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='end_window_size':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptpref = prompt[:-self.window_size], prompt[-self.window_size:]
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='corner50':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptmid, promptpref = prompt[:50], prompt[50:-50], prompt[-50:]
            self.rng.shuffle(promptmid)
            prompt = np.concatenate((promptsuff, promptmid, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)
     
        elif self.prompttype=='corner50del':
            prompt = np.array(sentence[loc-self.promptlen:loc], dtype=np.int32)
            promptsuff, promptmid, promptpref = prompt[:50], prompt[50:-50], prompt[-50:]
            prompt = np.concatenate((promptsuff, promptpref))
            completion = np.array(sentence[loc:loc+self.complen], dtype=np.int32)



        if self.instructions is not None:
            prompt = np.concatenate([np.array(self.instructions, dtype=np.int32), prompt])

        return {'prompt': torch.from_numpy(prompt), 'completion': torch.from_numpy(completion)}