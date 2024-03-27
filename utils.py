import numpy as np
import torch

def get_filename(args, args_ignore=None):
    args_dict = vars(args)
    filename = ''
    for ele in args_dict:
        if args_ignore is not None and ele in args_ignore:
            continue
        filename += ele + '-' + str(args_dict[ele]) + '__'
    
    return filename[:-2] + '.log'

def prompt_scoring(orig_ids, gen_ids, scoring='exact'):
    orig_ids, gen_ids = np.array(orig_ids), np.array(gen_ids)
    ## TODO: It records the perfect match for now, but we want to record how much matches later.
    if scoring=='exact':
        outscores = np.all(orig_ids==gen_ids, axis=-1)
        return outscores
    elif scoring=='length':
        outbools = orig_ids!=gen_ids
        print(outbools)
        outscores = np.argmax(outbools, axis=-1)
        print(outscores)
        exit()
    else:
        raise NotImplementedError("Evaluation scoring method %s is not implemented" % scoring)
    
def get_instruction_ids(instruction_format):
    if instruction_format is None:
        return None
    ### TODO: The instructions flag will only contain a marker for what kind of instruction needs to be added.
    ### First, we will need to use that to get the actual long form instruction string. 
    ### And then we need to tokenize it before sending it to the prompt dataset creator.