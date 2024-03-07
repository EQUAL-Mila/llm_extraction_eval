import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import torch

from pythia_utils import load_pythia_model
from prompt_loader import ExtractionPromptDataset
from utils import get_filename

from transformers import logging
logging.set_verbosity_error()
path_to_scratch = os.environ.get("SCRATCH")

def single_eval_run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_pythia_model(args.modelsize, args.modelstep, device=device)
    
    if 'deduped' in args.modelsize:
        prompt_dataset = ExtractionPromptDataset(path_to_scratch + "/pile-pythia/pile-deduped/document")
    else:
        prompt_dataset = ExtractionPromptDataset(path_to_scratch + "/pile-pythia/pile-standard/document")
    ## TODO: As of now, the whole Pile dataset is converted into prompts. 
    ## We need to get a subset here of only some randomly sampled indices.
    prompt_loader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batch_size, shuffle=False)

    score_arr = []
    for sample in tqdm(prompt_loader):
        prompt, completion = sample['prompt'].to(device), sample['completion'].to(device)

        ## TODO: Max new tokens is also a hyperparameter choice we need to make.
        ## TODO: Do we or do we not want stochasticity in generation? 
        ## TODO: Also other hyperparameters like temperature needs to be decided.
        ## TODO: Maybe something needs to be done here too for VLLM acceleration??
        outgen = model.generate(prompt, max_new_tokens=50)
        outgen_completion = outgen[:, prompt.shape[1]:]

        ## TODO: It records the perfect match for now, but we want to record how much matches later.
        outscores = torch.all(completion==outgen_completion, dim=-1)
        score_arr.extend(outscores.detach().cpu().numpy())

    ## TODO: Filename will only become useful after we make sure to put all distinguishing hyperparameters in argparse flags
    ## TODO: We might want to further divide the results or name them differently or save something else instead of score_arr
    with open('results/' + get_filename(args), "wb") as fp:
        pickle.dump(score_arr, fp)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction Attack: Single Evaluation Run')
    parser.add_argument('--modelsize', default='pythia-1.4b', type=str, help='Model Size')
    parser.add_argument('--modelstep', default='step100000', type=str, help='Training Step for Checkpoint')
    parser.add_argument('--batch_size', default=1, type=int, help='Evaluation Batch Size')
    # .... other arguments to be added
    args = parser.parse_args()

    single_eval_run(args)