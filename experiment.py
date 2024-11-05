import os
import numpy as np
import pickle
from tqdm import tqdm
import torch

import wandb
import random

from model_utils import load_pythia_model, VLLMModelWrapper, load_llama_together, load_mpt_7b, load_phi, load_falcon, load_gpt2
from model_utils import load_gemma_7, load_gemma_2, load_redpajama_base, load_olmo
from model_utils import load_gpt_large, load_gpt_xl, load_opt_6b

from prompt_loader import ExtractionPromptDataset
from utils import setup_parser, get_filename, get_instruction_ids, get_mask_token_id

import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

from tqdm import tqdm
import time
from transformers import logging
from transformers import AutoTokenizer
logging.set_verbosity_error()
path_to_scratch = "<add path to base_dir here>"


def single_eval_run(args):

    """
    Runs a single evaluation pass using the specified model and dataset.

    This function loads the appropriate model and dataset based on provided arguments,
    generates text completions for prompts, and stores the output in a specified directory.
    The results are also logged to Weights & Biases (wandb) [optional].

    Args:
        args (Namespace): Parsed arguments containing the configuration for the evaluation run, 
                          including model type, dataset path, prompt length, etc.

    Side Effects:
        - Logs progress and outputs to Weights & Biases.
        - Saves generated completion results to a file.
    """
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model based on args.modelsize
    print("...1. loading model")
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
    if 'opt6b' in args.modelsize:
        model = load_opt_6b()
    if 'gptlarge' in args.modelsize:
        model = load_gpt_large()
    if 'gptxl' in args.modelsize:
        model = load_gpt_xl()
    if 'olmo' in args.modelsize:
        model = load_olmo(modelstep=args.modelstep)

    model = VLLMModelWrapper(model=model,
                             temperature=args.temperature,
                             best_of=args.beamwidth,
                             use_beam_search=not args.sampling,
                             max_tokens=args.maxtokens)

    end = time.time()
    print(f"Time taken to load model: {end-start}")

    print("\n...2. loading prompt dataset")
    start = time.time()

    # Define path for dataset based on model type
    if 'deduped' in args.modelsize:
        pilepath = path_to_scratch + "/pile-pythia/pile-deduped/document"
    else:
        pilepath = path_to_scratch + "/pile-pythia/pile-standard/document"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")

    dataset_type = 'pythia'
    dataset_type = 'olmo' if 'olmo' in args.modelsize else dataset_type
    
    # Initialize the prompt dataset
    prompt_dataset = ExtractionPromptDataset(
        pilepath=pilepath,
        evalfile=args.evalfile,
        promptlen=args.promptlen,
        complen=args.complen,
        prompttype=args.prompttype,
        instructions=get_instruction_ids(args.instructions,
                                         dataset_type=dataset_type),
        window=args.window,
        mask_token=get_mask_token_id(args.modelsize,
                                     dataset_type=dataset_type),
        dataset_type=dataset_type)

    print(f"Length of prompt dataset: {len(prompt_dataset)}")
    print("batchsize:", args.batchsize)
    prompt_loader = torch.utils.data.DataLoader(prompt_dataset,
                                                batch_size=args.batchsize,
                                                shuffle=False)

    end = time.time()
    print(f"Time taken to load prompt dataset: {end-start}")

    gen_arr = []

    counter = 0
    print("\n...3. generating completions")
    start = time.time()

    # Generate completions for each sample in the dataset
    for sample in tqdm(prompt_loader):
        prompt_ids, completion_ids = sample['prompt'].detach().cpu().tolist(
        ), sample['completion'].detach().cpu().tolist()

        # Generate text completions based on model type
        if "pythia" in args.modelsize or "olmo" in args.modelsize:
            outgen = model.generate_text(prompt_token_ids=prompt_ids)
            outgen_ids = [ele.outputs[0].token_ids for ele in outgen]

        else:
            prompts = tokenizer.batch_decode(np.array(prompt_ids))
            outgen = model.generate_text(prompts=prompts)
            outgen = [str(ele.outputs[0].text) for ele in outgen]
            outgen_ids = tokenizer(outgen)


        gen_arr.append({
            'prompt_ids': prompt_ids,
            'completion_ids': completion_ids,
            'outgen_ids': outgen_ids
        })
        counter += 1
        wandb.log({'counter': counter})

    end = time.time()
    print(f"Time taken to generate completions: {end-start}")
    # Save generated completions to a file

    with open(
            path_to_scratch + '/extraction_results/' + get_filename(
                args, args_ignore=['scoring', 'batchsize', 'numgpus']),
            "wb") as fp:
        pickle.dump(gen_arr, fp)



if __name__=="__main__":
    parser = setup_parser()
    args = parser.parse_args()
    ## NOTE: Wandb Setup: Please replace 'your_wandb_key' with your own wandb key
    wandb.login(key='your_wandb_key')
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
