import os
import numpy as np
import pickle
from tqdm import tqdm
import torch

from pythia_utils import load_pythia_model, VLLMModelWrapper
from prompt_loader import ExtractionPromptDataset
from utils import setup_parser, get_filename, get_instruction_ids

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
                                             prompttype=args.prompttype, instructions=get_instruction_ids(args.instructions), 
                                             window=args.window)
    prompt_loader = torch.utils.data.DataLoader(prompt_dataset, batch_size=args.batchsize, shuffle=False)

    gen_arr = []
    for sample in tqdm(prompt_loader):
        prompt_ids, completion_ids = sample['prompt'].detach().cpu().tolist(), sample['completion'].detach().cpu().tolist()

        outgen = model.generate_text(prompt_token_ids=prompt_ids)
        outgen_ids = [ele.outputs[0].token_ids for ele in outgen]

        gen_arr.append({'prompt_ids': prompt_ids, 'completion_ids': completion_ids, 'outgen_ids': outgen_ids})

    with open(path_to_scratch + '/extraction_results/' + get_filename(args, args_ignore=['scoring', 'batchsize']), "wb") as fp:
        pickle.dump(gen_arr, fp)

if __name__=="__main__":
    parser = setup_parser()
    args = parser.parse_args()
    ## TODO: Wandb Setup

    single_eval_run(args)
