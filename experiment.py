import os
import numpy as np
import pickle
from tqdm import tqdm
import torch

import wandb
import random

from pythia_utils import load_pythia_model, VLLMModelWrapper
from prompt_loader import ExtractionPromptDataset
from utils import setup_parser, get_filename, get_instruction_ids

from transformers import logging
logging.set_verbosity_error()
path_to_scratch = "/network/scratch/p/prakhar.ganesh/"

def single_eval_run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_pythia_model(args.modelsize, args.modelstep, device=device, numgpus=args.numgpus)
    model = VLLMModelWrapper(model=model, temperature=args.temperature, best_of=args.beamwidth, 
                             use_beam_search=not args.sampling, max_tokens=args.maxtokens)
    
    if 'deduped' in args.modelsize: pilepath = path_to_scratch + "/pile-pythia/pile-deduped/document"
    else: pilepath = path_to_scratch + "/pile-pythia/pile-standard/document"

    prompt_dataset = ExtractionPromptDataset(pilepath=pilepath, evalfile=args.evalfile,
                                             promptlen=args.promptlen, complen=args.complen,
                                             prompttype=args.prompttype, instructions=get_instruction_ids(args.instructions), 
                                             window=args.window)
    prompt_loader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batchsize, shuffle=False)

    gen_arr = []
    counter = 0
    for sample in tqdm(prompt_loader):
        prompt_ids, completion_ids = sample['prompt'].detach().cpu().tolist(), sample['completion'].detach().cpu().tolist()

        outgen = model.generate_text(prompt_token_ids=prompt_ids)
        outgen_ids = [ele.outputs[0].token_ids for ele in outgen]

        gen_arr.append({'prompt_ids': prompt_ids, 'completion_ids': completion_ids, 'outgen_ids': outgen_ids})
        counter += 1
        if counter%100==0:
            wandb.log({'counter': counter})

    with open(path_to_scratch + '/extraction_results/' + get_filename(args, args_ignore=['scoring', 'batchsize', 'numgpus']), "wb") as fp:
        pickle.dump(gen_arr, fp)

    

if __name__=="__main__":
    parser = setup_parser()
    args = parser.parse_args()
    ## NOTE: Wandb Setup
    wandb.login(key='177301ceceab56316ac99630a79d09a45b1da3d6')
    wandb.init(
        project='llm-extraction-eval',
        name=f"run_{args.modelsize}_{args.modelstep}_{args.promptlen}_{random.randint(0, 1000000)}",
        config={**vars(args), 
        'completed': False,
        'counter': 0
        }
    )
    
    single_eval_run(args)

    wandb.log({'completed': True})
    wandb.finish()