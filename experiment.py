import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import torch

from pythia_utils import load_pythia_model
from prompt_loader import ExtractionPromptDataset
from utils import get_filename, prompt_scoring

from transformers import logging
logging.set_verbosity_error()
path_to_scratch = os.environ.get("SCRATCH")

def single_eval_run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_pythia_model(args.modelsize, args.modelstep, device=device)
    
    if 'deduped' in args.modelsize: promptfile = path_to_scratch + "/pile-pythia/pile-deduped/document"
    else: promptfile = path_to_scratch + "/pile-pythia/pile-standard/document"

    prompt_dataset = ExtractionPromptDataset(pile_path=promptfile, evalfile=args.evalfile,
                                             promptlen=args.promptlen, complen=args.complen, promptloc=args.promptloc,
                                             prompttype=args.prompttype, instructions=args.instructions)
    prompt_loader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batch_size, shuffle=False)

    score_arr = []
    for sample in tqdm(prompt_loader):
        prompt, completion = sample['prompt'].to(device), sample['completion'].to(device)

        ## TODO: Where exactly do we add generation hyperparameters? Here or when defining the model?
        outgen = model.generate(prompt, max_new_tokens=50)
        outgen_completion = outgen[:, prompt.shape[1]:]

        match_score = prompt_scoring(completion, outgen_completion, scoring=args.scoring)
        score_arr.extend(match_score)

    ## TODO: We might want to further divide the results or name them differently or save something else instead of score_arr
    with open('results/' + get_filename(args), "wb") as fp:
        pickle.dump(score_arr, fp)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction Attack: Single Evaluation Run')
    ### Arguments to control Prompt Creation for Evaluation
    parser.add_argument('--promptlen', default=10, type=int, help='Length of Prompt')
    parser.add_argument('--complen', default=50, type=int, help='Length of Completion')
    parser.add_argument('--promptloc', default=None, type=str, help='Filename containing locations of Prompt in each Sentence')
    parser.add_argument('--prompttype', default='standard', type=str, help='Prompt Formatting Before Feeding it to the Model')
    parser.add_argument('--instructions', default=None, type=str, help='Any Additional Instructions Added Before the Prompt')

    ### LLM Hyperparameters for Generation
    parser.add_argument('--temperature', default=0., type=float, help='Temperature for LLM Generation')
    parser.add_argument('--beamwidth', default=1, type=int, help='Beam Search Width for LLM Generation')
    parser.add_argument('--maxtokens', default=50, type=int, help='Max tokens for LLM Generation')
    parser.add_argument('--sampling', action='store_true', help='Do sampling for LLM Generation')

    ### Model Configuration
    parser.add_argument('--modelsize', default='pythia-1.4b', type=str, help='Model Size')
    parser.add_argument('--modelstep', default='step100000', type=str, help='Training Step for Checkpoint')

    ### Scoring Method for Evaluation
    parser.add_argument('--scoring', default='exact', type=str, help='Scoring Method for Evaluation')

    ### Other Arguments
    parser.add_argument('--batchsize', default=1, type=int, help='Evaluation Batch Size')
    parser.add_argument('--evalfile', type=str, help='Text File with Indices of Input Sentences to Evaluate')

    args = parser.parse_args()
    ## TODO: Wandb Setup

    single_eval_run(args)
