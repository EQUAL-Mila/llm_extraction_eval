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