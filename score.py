import os
import numpy as np
import pickle

from utils import setup_parser, get_filename, prompt_scoring

path_to_scratch = os.environ.get("SCRATCH")

def single_eval_score(args):
    with open(path_to_scratch + '/extraction_results/' + get_filename(args, args_ignore=['scoring', 'batchsize']), "rb") as fp:
        gen_arr = pickle.load(fp)

    score_arr = []
    for batch in gen_arr:
        scores = prompt_scoring(batch['completion_ids'], batch['outgen_ids'], args.scoring)
        score_arr.extend(scores)
    
    # score_arr = np.array(score_arr)
    # print(np.mean(score_arr>=10))
    # print(np.mean(score_arr))

if __name__=="__main__":
    parser = setup_parser()
    args = parser.parse_args()

    single_eval_score(args)
