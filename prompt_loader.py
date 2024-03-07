import numpy as np
import torch

from mmap_dataset import MMapIndexedDataset

class ExtractionPromptDataset(torch.utils.data.Dataset):
    def __init__(self, pile_path):
        self.mmap_dataset = MMapIndexedDataset(pile_path, skip_warmup = True)

    def __len__(self):
        return len(self.mmap_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.mmap_dataset[idx]

        ## Divide it into prompt and completion.
        ## TODO: All Different Variations of Prompt Creation Go Here!!
        prompt, completion = np.array(sentence[:50], dtype=np.int32), np.array(sentence[50:100], dtype=np.int32)

        return {'prompt': torch.from_numpy(prompt), 'completion': torch.from_numpy(completion)}