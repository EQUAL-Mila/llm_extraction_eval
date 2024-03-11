import numpy as np
import torch

from mmap_dataset import MMapIndexedDataset

class ExtractionPromptDataset(torch.utils.data.Dataset):
    def __init__(self, pilepath, evalfile, promptlen, complen, promptloc, prompttype, instructions):
        ### TODO: Incorporate all the inputs
        self.mmap_dataset = MMapIndexedDataset(pilepath, skip_warmup = True)

    def __len__(self):
        return len(self.mmap_dataset)

    def set_window_length(self,prefix_len=None,suffix_len=None, sentence=""):
        """
        vary prefix/suffix window length
        """

        if prefix_len is None and suffix_len is None:
            prefix = sentence[:50]
            suffix = sentence[50:100]
        
        prefix = sentence[:prefix_len]
        suffix = sentence[ prefix_len: prefix_len + suffix_len ]
        return prefix, suffix

    def set_additional_instructions(self, sentence="", instruction=""):
        """
        add additional instructions to the prompt
        """
        return instruction + sentence
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.mmap_dataset[idx]

        ## Divide it into prompt and completion.
        ## TODO: All Different Variations of Prompt Creation Go Here!!

        prompt, completion = np.array(sentence[:50], dtype=np.int32), np.array(sentence[50:100], dtype=np.int32)

        return {'prompt': torch.from_numpy(prompt), 'completion': torch.from_numpy(completion)}