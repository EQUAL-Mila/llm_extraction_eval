import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import torch

from pythia_utils import load_pythia_model, VLLMModelWrapper
from prompt_loader import ExtractionPromptDataset
from utils import get_filename, prompt_scoring, get_instruction_ids

from transformers import logging
logging.set_verbosity_error()
path_to_scratch = os.environ.get("SCRATCH")

def single_eval_run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pythia_model(args.modelsize, args.modelstep, device=device)
    model = VLLMModelWrapper(model=model, temperature=args.temperature, best_of=args.beamwidth, 
                             use_beam_search=not args.sampling, max_tokens=args.maxtokens)
    
    if 'deduped' in args.modelsize: pilepath = path_to_scratch + "/pile-pythia/pile-deduped/document"
    else: pilepath = path_to_scratch + "/pile-pythia/pile-standard/document"

    prompt_dataset = ExtractionPromptDataset(pilepath=pilepath, evalfile=args.evalfile,
                                             promptlen=args.promptlen, complen=args.complen,
                                             prompttype=args.prompttype, instructions=get_instruction_ids(args.instructions))
    prompt_loader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batchsize, shuffle=False)

    score_arr = []
    for sample in tqdm(prompt_loader):
        prompt_ids, completion_ids = sample['prompt'].detach().cpu().tolist(), sample['completion'].detach().cpu().tolist()

        outgen = model.generate_text(prompt_token_ids=prompt_ids)
        outgen_ids = [ele.outputs[0].token_ids for ele in outgen]

        match_score = prompt_scoring(completion_ids, outgen_ids, scoring=args.scoring)
        score_arr.extend(match_score)
        print(np.sum(score_arr), np.mean(score_arr))

    ## TODO: We might want to further divide the results or name them differently or save something else instead of score_arr
    with open('results/' + get_filename(args, args_ignore=['batchsize']), "wb") as fp:
        pickle.dump(score_arr, fp)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction Attack: Single Evaluation Run')
    ### Arguments to control Prompt Creation for Evaluation
    parser.add_argument('--evalfile', required=True, type=str, help='File with Indices and Location of Sentences to Evaluate')
    parser.add_argument('--promptlen', default=10, type=int, help='Length of Prompt')
    parser.add_argument('--complen', default=50, type=int, help='Length of Completion')
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

    args = parser.parse_args()
    ## TODO: Wandb Setup

    single_eval_run(args)
