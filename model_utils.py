import os
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers import GPTNeoXConfig
from huggingface_hub import list_repo_refs

import vllm
from dotenv import load_dotenv
load_dotenv()

path_to_scratch = "/network/scratch/p/prakhar.ganesh/"



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

    def generate_text(self, prompts=None, prompt_token_ids=None):
        return self.model.generate(prompts=prompts, sampling_params=self.sampling_params, prompt_token_ids=prompt_token_ids)

def cache_check_tokenizer(modelsize, modelstep,):
    '''
    Check if the tokenizer is already cached/saved
    '''
    if os.path.exists(path_to_scratch + "/%s/%s/" % (modelsize, modelstep) + "tokenizer_config.json"):
        return True


def load_pythia_model(modelsize,
                      modelstep,
                      device='cuda',
                      padding_side='right',
                      truncation_side='right',
                      model_max_length=2048,
                      gpu_memory_utilization=0.9,
                      numgpus=1):

    """
    Loads the Pythia model and tokenizer with specified configuration.

    Args:
        modelsize (str): Model size identifier.
        modelstep (str): Model step/revision identifier.
        device (str): Device to load the model on.
        padding_side (str): Padding side for tokenizer.
        truncation_side (str): Truncation side for tokenizer.
        model_max_length (int): Maximum length for the model.
        gpu_memory_utilization (float): GPU memory utilization limit.
        numgpus (int): Number of GPUs to use.

    Returns:
        vllm.LLM: Loaded model instance.
    """

    modelloc = path_to_scratch + "/%s/%s/" % (modelsize, modelstep)
    if not cache_check_tokenizer(modelsize, modelstep):
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/%s" % modelsize,
            revision=modelstep,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_max_length=model_max_length)
        tokenizer.save_pretrained(modelloc)

    if gpu_memory_utilization != 0.9:
        model = vllm.LLM(model=f"EleutherAI/{modelsize}",
                         revision=modelstep,
                         tokenizer=modelloc,
                         download_dir=modelloc,
                         trust_remote_code=True,
                         gpu_memory_utilization=gpu_memory_utilization,
                         tensor_parallel_size=numgpus)
        return model

    model = vllm.LLM(model=f"EleutherAI/{modelsize}",
                     revision=modelstep,
                     tokenizer=modelloc,
                     download_dir=modelloc,
                     trust_remote_code=True,
                     tensor_parallel_size=numgpus)

    return model

def load_opt_6b():
    """
    Loads the OPT 6.7B model.

    Returns:
        vllm.LLM: Loaded OPT 6.7B model instance.
    """

    model_name = 'facebook/opt-6.7b'
    model = vllm.LLM(
        model = model_name,
        trust_remote_code=True,
        tensor_parallel_size = 1
    )
    return model


def load_opt_2b():
    """
    Loads the OPT 2.7B model.

    Returns:
        vllm.LLM: Loaded OPT 2.7B model instance.
    """
    model_name = 'facebook/opt-2.7b'
    model = vllm.LLM(model=model_name,
                     trust_remote_code=True,
                     tensor_parallel_size=1)
    return model


def load_gpt_large():
    """
    Loads the GPT-2 Large model.
    """
    model_name = 'openai-community/gpt2-large'
    model = vllm.LLM(model=model_name,
                     trust_remote_code=True,
                     tensor_parallel_size=1)
    return model

def load_gpt_xl():
    """
    Loads the GPT-2 XL model.
    """
    model_name = 'openai-community/gpt2-xl'
    model = vllm.LLM(model=model_name,
                     trust_remote_code=True,
                     tensor_parallel_size=1)
    return model

def load_gemma_2():
    """
    Loads the GEMMA 2B model.
    """
    model_name = 'google/gemma-2b'
    gemma = vllm.LLM(model=model_name,
                        trust_remote_code=True,
                        max_model_len=2048,
                        tensor_parallel_size=1)
    return gemma


def load_gemma_7():
    """
    Loads the GEMMA 7B model.
    """
    model_name = 'google/gemma-7b'
    gemma = vllm.LLM(model=model_name,
                        trust_remote_code=True,
                        max_model_len=2048,
                        tensor_parallel_size=1, prompt_logprobs=1)
    return gemma


def load_olmo(modelstep):
    """
    Loads the OLMo 7B model.
    """
    model_name = 'allenai/OLMo-7B'
    model_name = 'allenai/OLMo-7B-0724-hf'

    out = list_repo_refs("allenai/OLMo-7B")
    branches = [b.name for b in out.branches]

    for branch in branches:
        if modelstep in branch:
            modelstep = branch
            break

    olmo = vllm.LLM(model=model_name,
                    trust_remote_code=True,
                    max_model_len=2048,
                    revision=modelstep,
                    tensor_parallel_size=1)
    return olmo


def load_llama_together():
    """
    Loads the LLaMA 2.7B model by Together Computer.
    """
    model_name = "togethercomputer/LLaMA-2-7B-32K"
    llama = vllm.LLM(model=model_name,
                     trust_remote_code=True, max_model_len=2048, tensor_parallel_size=1)
    return llama

def load_mpt_7b():
    """
    Loads the MPT 7B model.
    """
    model_name = "mosaicml/mpt-7b"
    mpt = vllm.LLM(model=model_name,
                     trust_remote_code=True,max_model_len=2048, tensor_parallel_size=1)
    return mpt


def load_phi():
    """
    Loads the PHI 2B model.
    """
    model_name = "microsoft/phi-2"
    phi = vllm.LLM(model=model_name, trust_remote_code=True, max_model_len=2048, tensor_parallel_size=1)
    return phi


def load_gpt2():
    """
    Loads the community GPT-2 model.
    """
    model_name = "openai-community/gpt2"
    gpt2 = vllm.LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1)
    return gpt2

def load_redpajama_chat():
    """
    Loads the RedPajama INCITE 7B Chat model.
    """
    model_name = "togethercomputer/RedPajama-INCITE-7B-Chat"
    redpajama = vllm.LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1)
    return redpajama


def load_redpajama_base():
    """
    Loads the RedPajama INCITE 7B Base model.
    """
    model_name = "togethercomputer/RedPajama-INCITE-7B-Base"
    redpajama = vllm.LLM(model=model_name,
                         trust_remote_code=True,
                         tensor_parallel_size=1)
    return redpajama


def load_falcon():
    """
    Loads the Falcon 7B model.
    """
    model_name = "tiiuae/falcon-7b"
    falcon = vllm.LLM(model=model_name,
                      trust_remote_code=True,
                      tensor_parallel_size=1)
    return falcon


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
    prompt = "The quick brown fox jumps over the lazy dog."
    chosen = 'redpajama_base'

    if chosen == 'pythia':
        model, tokenizer = load_pythia_model(DEFAULT_MODEL, DEFAULT_REVISION)
        # configure the parameters for the generation (model, temperature, etc)

    if chosen == 'llama':
        # llama
        model = load_llama_together()

    if chosen == 'mpt':
        model = load_mpt_7b()

    if chosen == 'phi':
        model = load_phi()

    if chosen == 'redpajama_chat':
        model = load_redpajama_chat()
        template = "<human>: {instruction}\n<bot>:"
        prompt = template.format(instruction=prompt)

    if chosen == 'redpajama_base':
        model = load_redpajama_base()



    model = VLLMModelWrapper(model=model,
                             temperature=0.5,
                             use_beam_search=False,
                             max_tokens=200)

    s = model.generate_text(prompt=prompt)
    print(s)
