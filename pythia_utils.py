import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer
path_to_scratch = os.environ.get("SCRATCH")

def load_pythia_model(modelsize, modelstep, device='cpu', padding_side='right', truncation_side='right', model_max_length=2048):
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/%s" % modelsize,
        revision=modelstep,
        cache_dir=path_to_scratch + "/%s/%s/" % (modelsize, modelstep),
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/%s" % modelsize,
        revision=modelstep,
        cache_dir=path_to_scratch + "/%s/%s/" % (modelsize, modelstep),
        padding_side=padding_side, truncation_side=truncation_side, model_max_length=model_max_length
    )

    return model, tokenizer

if __name__=="__main__":
    model, tokenizer = load_pythia_model('pythia-1.4b', 'step100000')
    print(model)