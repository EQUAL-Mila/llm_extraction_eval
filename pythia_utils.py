import os
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers import GPTNeoXConfig
import vllm
from dotenv import load_dotenv
load_dotenv()

path_to_scratch = os.environ.get("SCRATCH")


def cache_check_tokenizer(modelsize, modelstep,):
    '''
    Check if the tokenizer is already cached/saved
    '''
    if os.path.exists(path_to_scratch + "/%s/%s/" % (modelsize, modelstep) + "tokenizer_config.json"):
        return True
    
def load_pythia_model(modelsize, modelstep, device='cuda', padding_side='right', truncation_side='right', model_max_length=2048):

    # check if the tokenizer is already cached/saved
    if not cache_check_tokenizer(modelsize, modelstep):
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/%s" % modelsize,
            revision=modelstep,
            padding_side=padding_side, truncation_side=truncation_side, model_max_length=model_max_length
        )        
        tokenizer.save_pretrained(path_to_scratch + "/%s/%s/" % (modelsize, modelstep))
    
    model = vllm.LLM(model = f"EleutherAI/{modelsize}", revision = modelstep ,tokenizer= path_to_scratch + "/%s/%s/" % (modelsize, modelstep), trust_remote_code = True)

    # testing model outputs
    l = model.generate("Hello, eincorp!")
    print(l)

    return model, tokenizer


if __name__=="__main__":
    model, tokenizer = load_pythia_model('pythia-1.4b', 'step100000')
    
    MODEL_NAMES = [ "pythia-14m",
                    "pythia-70m",
                    "pythia-160m",
                    "pythia-410m",
                    "pythia-1b",
                    "pythia-1.4b",
                    "pythia-2.8b",
                    "pythia-6.9b",
                    "pythia-12b",
                    "pythia-70m-deduped",
                    "pythia-160m-deduped",
                    "pythia-410m-deduped",
                    "pythia-1b-deduped",
                    "pythia-1.4b-deduped",
                    "pythia-2.8b-deduped",
                    "pythia-6.9b-deduped",
                    "pythia-12b-deduped"]


