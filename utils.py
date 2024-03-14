import numpy as np
import torch

def get_filename(args, args_subset=None):
    args_dict = vars(args)
    filename = ''
    for ele in args_dict:
        if args_subset is not None and ele not in args_subset:
            continue
        filename += ele + '-' + str(args_dict[ele]) + '__'
    
    return filename[:-2]

def prompt_scoring(orig_ids, gen_ids, scoring='exact'):
    orig_ids, gen_ids = np.array(orig_ids), np.array(gen_ids)
    ## TODO: It records the perfect match for now, but we want to record how much matches later.
    if scoring=='exact':
        outscores = np.all(orig_ids==gen_ids, axis=-1)
        return outscores
    else:
        raise NotImplementedError("Evaluation scoring method %s is not implemented" % scoring)