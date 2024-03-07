import argparse
import torch

from pythia_utils import load_pythia_model

def single_eval_run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_pythia_model(args.modelsize, args.modelstep, device=device)
    
    ### Create the dataset and prompt

    ### Run the Evaluation
    ### Save results in a file

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extraction Attack: Single Evaluation Run')
    parser.add_argument('--modelsize', default='pythia-1.4b', type=str, help='Model Size')
    parser.add_argument('--modelstep', default='step100000', type=str, help='Training Step for Checkpoint')
    # .... other arguments to be added
    args = parser.parse_args()

    single_eval_run(args)