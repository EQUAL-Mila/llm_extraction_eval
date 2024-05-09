import os
import numpy as np
import pickle
from tqdm import tqdm
import torch

import wandb
import random

from model_utils import load_pythia_model, VLLMModelWrapper, load_llama_together, load_mpt_7b, load_phi, load_falcon, load_gpt2
from model_utils import load_gemma_7, load_gemma_2, load_redpajama_base

from prompt_loader import ExtractionPromptDataset
from utils import setup_parser, get_filename, get_instruction_ids, get_mask_token_id

from transformers import logging
from transformers import AutoTokenizer
logging.set_verbosity_error()
path_to_scratch = "/network/scratch/p/prakhar.ganesh/"


def single_eval_run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if 'pythia' in args.modelsize:
        model = load_pythia_model(args.modelsize,
                                  args.modelstep,
                                  device=device,
                                  numgpus=args.numgpus)
    if 'llama' in args.modelsize:
        model = load_llama_together()
    if 'mpt' in args.modelsize:
        model = load_mpt_7b()
    if 'phi' in args.modelsize:
        model = load_phi()
    if 'redpajama' in args.modelsize:
        model = load_redpajama_base()
    if 'gpt2' in args.modelsize:
        model = load_gpt2()
    if 'falcon' in args.modelsize:
        model = load_falcon()
    if 'gemma2' in args.modelsize:
        model = load_gemma_2()
    if 'gemma7' in args.modelsize:
        model = load_gemma_7()

    model = VLLMModelWrapper(model=model,
                             temperature=args.temperature,
                             best_of=args.beamwidth,
                             use_beam_search=not args.sampling,
                             max_tokens=args.maxtokens)

    if 'deduped' in args.modelsize:
        pilepath = path_to_scratch + "/pile-pythia/pile-deduped/document"
    else:
        pilepath = path_to_scratch + "/pile-pythia/pile-standard/document"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")

    prompt_dataset = ExtractionPromptDataset(
        pilepath=pilepath,
        evalfile=args.evalfile,
        promptlen=args.promptlen,
        complen=args.complen,
        prompttype=args.prompttype,
        instructions=get_instruction_ids(args.instructions),
        window=args.window,
        mask_token=get_mask_token_id(args.modelsize))

    prompt_loader = torch.utils.data.DataLoader(prompt_dataset,
                                                batch_size=args.batchsize,
                                                shuffle=False)

    gen_arr = []
                    
                    
    counter = 0
    for sample in tqdm(prompt_loader):
        prompt_ids, completion_ids = sample['prompt'].detach().cpu().tolist(
        ), sample['completion'].detach().cpu().tolist()

        if "pythia" in args.modelsize:
            outgen = model.generate_text(prompt_token_ids=prompt_ids)
            outgen_ids = [ele.outputs[0].token_ids for ele in outgen]
        else:
            prompts = tokenizer.batch_decode(np.array(prompt_ids))
            outgen = model.generate_text(prompts=prompts)
            outgen = [str(ele.outputs[0].text) for ele in outgen]
            outgen_ids = tokenizer.encode(outgen, is_split_into_words=True)
            # for ele in outgen:
            #     print(ele)
            #     outgen_ids = tokenizer.encode(ele)

        gen_arr.append({
            'prompt_ids': prompt_ids,
            'completion_ids': completion_ids,
            'outgen_ids': outgen_ids
        })
        counter += 1
        wandb.log({'counter': counter})

    with open(path_to_scratch + '/extraction_results/' + get_filename(
             args, args_ignore=['scoring', 'batchsize', 'numgpus']),
             "wb") as fp:
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
