import os
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers import GPTNeoXConfig
import vllm
from dotenv import load_dotenv
load_dotenv()

path_to_scratch = os.environ.get("SCRATCH")

class VLLMModelWrapper:
    def __init__(self, model, temperature=0, best_of=1, use_beam_search=False, seed=0, max_tokens=200, **kwargs):
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
        self.sampling_params = vllm.SamplingParams(temperature=temperature, best_of=best_of, max_tokens=max_tokens,
                                                   use_beam_search=(best_of>1 and use_beam_search), seed=seed, ignore_eos=True)

    def generate_text(self, prompt=None, prompt_token_ids=None):
        return self.model.generate(prompts=prompt, sampling_params=self.sampling_params, prompt_token_ids=prompt_token_ids)

def cache_check_tokenizer(modelsize, modelstep,):
    '''
    Check if the tokenizer is already cached/saved
    '''
    if os.path.exists(path_to_scratch + "/%s/%s/" % (modelsize, modelstep) + "tokenizer_config.json"):
        return True

def load_pythia_model(modelsize, modelstep, device='cuda', padding_side='right', truncation_side='right', model_max_length=2048):

    modelloc = path_to_scratch + "/%s/%s/" % (modelsize, modelstep)
    if not cache_check_tokenizer(modelsize, modelstep):
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/%s" % modelsize,
            revision=modelstep,
            padding_side=padding_side, truncation_side=truncation_side, model_max_length=model_max_length
        )
        tokenizer.save_pretrained(modelloc)

    model = vllm.LLM(model = f"EleutherAI/{modelsize}", revision = modelstep,
                     tokenizer= modelloc, download_dir= modelloc, trust_remote_code = True)

    return model

if __name__ == "__main__":

    MODEL_NAMES = [
        "pythia-14m",
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]

    DEFAULT_MODEL, DEFAULT_REVISION = "pythia-1.4b", 'step100000'

    model, tokenizer = load_pythia_model(DEFAULT_MODEL, DEFAULT_REVISION)

    # configure the parameters for the generation (model, temperature, etc)
    model = VLLMModelWrapper(model=model,
                             temperature=0.5,
                             use_beam_search=False,
                             max_tokens=200)

    prompt = "The quick brown fox jumps over the lazy dog."
    model.generate_text(prompt=prompt)
