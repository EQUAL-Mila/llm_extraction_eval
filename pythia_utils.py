import os
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers import GPTNeoXConfig
import vllm
from dotenv import load_dotenv
load_dotenv()

path_to_scratch = os.environ.get("SCRATCH")


from vllm import SamplingParams

class VLLMModelWrapper:
    def __init__(self, model, temperature=0, use_beam_search=False,seed=None, max_tokens=200, **kwargs):
        '''
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        use_beam_search: If True, beam search is used instead of sampling.
        more on: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        '''

        self.model = model
        self.sampling_params = SamplingParams(temperature=temperature,max_tokens=max_tokens,use_beam_search=use_beam_search)

    def set_base_prompt(self, prompt):
        self.prompt = prompt

    def generate_text(self, prompt):
        if prompt is None:
            prompt = self.prompt
        return self.model.generate_text(prompt,self.sampling_params)

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


if __name__ == "__main__":

    MODEL_NAMES = [
        "pythia-14m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
        "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
        "pythia-70m-deduped", "pythia-160m-deduped", "pythia-410m-deduped",
        "pythia-1b-deduped", "pythia-1.4b-deduped", "pythia-2.8b-deduped",
        "pythia-6.9b-deduped", "pythia-12b-deduped"
    ]

    DEFUALT_MODEL, DEFAULT_REVISION = "pythia-1.4b", 'step100000'

    model, tokenizer = load_pythia_model(DEFUALT_MODEL, DEFAULT_REVISION)

    # configure the parameters for the generation (model, temperature, etc)
    model = VLLMModelWrapper(model=model,
                             temperature=0.5,
                             use_beam_search=False,
                             max_tokens=200)

    model.generate_text(prompt = "Hello, eincorp!")
