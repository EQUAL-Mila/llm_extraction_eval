import os
import itertools
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import seaborn as sns

from model_utils import load_gemma_7, load_gemma_2, load_redpajama_base, load_olmo
from model_utils import load_gpt_large, load_gpt_xl, load_opt_6b, load_pythia_model

sns.set(rc={'figure.facecolor':'white'}, font_scale=1.6)
# , 'figure.facecolor':'cornflowerblue'
palette = itertools.cycle(sns.color_palette())

from utils import setup_parser, get_filename, prompt_scoring, zlib_ratio

path_to_scratch = "/network/scratch/p/prakhar.ganesh/"


def print_single_statement(args, idx):
    """
    Prints a single prompt-completion pair from the generated data based on the provided index.

    Args:
        args: Parsed arguments containing configuration details.
        idx: Index of the example to retrieve and print.
    """
    print("using .. ")
    print(path_to_scratch + '/extraction_results/' +
          get_filename(args, args_ignore=['scoring', 'batchsize', 'numgpus']))

    with open(
            path_to_scratch + '/extraction_results/' + get_filename(
                args, args_ignore=['scoring', 'batchsize', 'numgpus']),
            "rb") as fp:
        gen_arr = pickle.load(fp)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b",
                                              revision="step140000")

    for batch in gen_arr:

        if idx >= len(batch['prompt_ids']):
            idx = idx - len(batch['prompt_ids'])
            continue
        chosen_example = (batch['prompt_ids'][idx],
                          batch['completion_ids'][idx],
                          batch['outgen_ids'][idx])

        print("\n----\n")
        print('prompt_ids shape', len(batch['prompt_ids']),
              len(batch['prompt_ids'][1]),
              tokenizer.batch_decode(batch['prompt_ids'][:2]),
              len(tokenizer.batch_decode(batch['prompt_ids'])))
        print('completion_ids shape', len(batch['completion_ids']),
              len(batch['completion_ids'][0]))
        print("comparing indexes of prompt and batch completions")

        break

    print("Prompt")
    print("------")
    print(tokenizer.decode(chosen_example[0]))
    print("-------------------")
    print("Original Completion")
    print("-------------------")
    print(tokenizer.decode(chosen_example[1]))
    print("--------------------")
    print("Generated Completion")
    print("--------------------")
    print(tokenizer.decode(chosen_example[2]))
    print("--------------------")

def single_eval_score(args):
    """
    Evaluates a single prompt-completion pair and calculates the prompt scoring.

    Args:
        args: Parsed arguments containing configuration details.
    """

    with open(path_to_scratch + '/extraction_results/' + get_filename(args, args_ignore=['scoring', 'batchsize', 'numgpus']), "rb") as fp:
        gen_arr = pickle.load(fp)

    score_arr = []
    for batch in gen_arr:
        scores = prompt_scoring(batch['completion_ids'], batch['outgen_ids'], args.scoring, levenshtein_delta=1.)
        score_arr.extend(scores)

    score_arr = np.array(score_arr)
    print(np.mean(score_arr))
    exit()
    # score_arr = np.logical_and(score_arr<5, score_arr>0)
    # score_arr = score_arr>=500
    # print(np.mean(score_arr))

    # for ite, ind in enumerate(np.where(score_arr)[0]):
    #     print_single_statement(args, ind)
    #     if ite>100:
    #         break

    # score_arr_nonzero = score_arr[score_arr>0]
    score_arr_nonzero = score_arr
    color1 = next(palette)
    H, bins = np.histogram(score_arr_nonzero, bins=50)
    H = np.cumsum(H)/1000
    plt.plot(H[:11], color=color1)
    # plt.hist(score_arr_nonzero, color=color1, bins=50)

    plt.fill_between(range(0, 11), H[:11], 0, color=color1, alpha=.5)
    # plt.yscale('log')
    # plt.ylabel('Number of Sentences', labelpad=10)
    # plt.xlabel('Length of Exact Match', labelpad=10)
    plt.ylabel('Extraction Rate', labelpad=10)
    plt.xlabel('Levenshtein Distance Threshold', labelpad=10)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    plt.tight_layout()
    # plt.savefig('hist_length.pdf')
    plt.savefig('hist_levenshtein.pdf')


def multiple_eval_single_axis(args, var_name, var_list, ld):
    """
    Performs multiple evaluations on a single variable axis and plots extraction rates.

    Args:
        args: Parsed arguments containing configuration details.
        var_name: Name of the variable to vary in evaluations.
        var_list: List of values for the specified variable.
        ld: Levenshtein distance threshold.
    """

    all_scores = []
    all_acc = []
    temp = path_to_scratch + '/extraction_results/' + get_filename(
        args, args_ignore=['scoring', 'batchsize', 'numgpus'])
    print("... Using filename:", temp)
    for var_val in tqdm(var_list):
        setattr(args, var_name, var_val)
        t = get_filename(args, args_ignore=['scoring', 'batchsize', 'numgpus'])
        print("... Using sub-filename:", t)
        with open(
                path_to_scratch + '/extraction_results/' + get_filename(
                    args, args_ignore=['scoring', 'batchsize', 'numgpus']),
                "rb") as fp:
            gen_arr = pickle.load(fp)

        score_arr = []
        for batch in gen_arr:
            scores = prompt_scoring(batch['completion_ids'],
                                    batch['outgen_ids'],
                                    args.scoring,
                                    levenshtein_delta=ld)
            score_arr.extend(scores)

        score_arr = np.array(score_arr)
        all_scores.append(score_arr)

        all_acc.append(100 * np.mean(score_arr))

    ## Extracted in at least one
    combined_scores = np.max(all_scores, axis=0)

    color1 = next(palette)
    plt.scatter(var_list, all_acc, color=color1)
    plt.plot(var_list, all_acc, '--', color=color1)
    plt.plot(var_list, [100 * np.mean(combined_scores)] * len(var_list),
             '--',
             color='red',
             linewidth=4)
    plt.ylabel('Extraction Rate', labelpad=10)
    plt.xlabel(var_name, labelpad=10)
    plt.ylim(0, 4.)
    plt.tight_layout()
    plt.savefig('extraction_single_axis_%s_%f_multi.png' % (var_name, ld))


def multiple_eval_multiple_axis(args, var1_name, var1_list, var2_name,
                                var2_list, var3_name, var3_list, ld):

    """
    Evaluates multiple variable combinations on separate axes .

    Args:
        args: Parsed arguments containing configuration details.
        var1_name: Name of the first variable.
        var1_list: List of values for the first variable.
        var2_name: Name of the second variable.
        var2_list: List of values for the second variable.
        var3_name: Name of the third variable.
        var3_list: List of values for the third variable.
        ld: Levenshtein distance threshold.
    """

    all_scores = []
    all_acc = []
    for var1_val in tqdm(var1_list):
        all_scores1 = []
        all_acc1 = []
        for var2_val in tqdm(var2_list):
            all_scores2 = []
            all_acc2 = []
            for var3_val in tqdm(var3_list):
                setattr(args, var1_name, var1_val)
                setattr(args, var2_name, var2_val)
                setattr(args, var3_name, var3_val)
                try:
                    t = get_filename(args, args_ignore=['scoring', 'batchsize', 'numgpus'])
                    print("using filename:", t)
                    with open(
                            path_to_scratch + '/extraction_results/' +
                            get_filename(args,
                                         args_ignore=[
                                             'scoring', 'batchsize', 'numgpus'
                                         ]), "rb") as fp:
                        gen_arr = pickle.load(fp)
                except:
                    print(args)

                score_arr = []
                for batch in gen_arr:
                    scores = prompt_scoring(batch['completion_ids'],
                                            batch['outgen_ids'],
                                            args.scoring,
                                            levenshtein_delta=ld)
                    score_arr.extend(scores)

                score_arr = np.array(score_arr)
                all_scores2.append(score_arr)

                all_acc2.append(100 * np.mean(score_arr))
            all_scores1.append(all_scores2)
            all_acc1.append(all_acc2)
        all_scores.append(all_scores1)
        all_acc.append(all_acc1)

    ## Extracted in at least one
    # print(np.shape(all_scores))
    print(all_acc)
    print(np.mean(np.max(all_scores, axis=0), axis=-1))
    print(np.mean(np.max(all_scores, axis=1), axis=-1))
    print(np.mean(np.max(all_scores, axis=2), axis=-1))
    print(np.mean(np.max(all_scores, axis=(0, 1)), axis=-1))
    print(np.mean(np.max(all_scores, axis=(1, 2)), axis=-1))
    print(np.mean(np.max(all_scores, axis=(0, 2)), axis=-1))
    print(np.mean(np.max(all_scores, axis=(0, 1, 2)), axis=-1))
    # combined_scores = np.max(all_scores, axis=0)
    # print(combined_scores)

def multiple_eval_combined(args):
    """
    Evaluates different variables and their respective values, for example : prompt length, model size, model step etc.
    and reports a combined score across the chosen variables. To choose a variable, please uncommment respective keys.
    Args:
        args: Parsed arguments containing configuration details.
    """

    args_change = {
                    # 'modelstep': ['step100000', 'step105000', 'step110000', 'step115000', 'step120000',
                    #               'step125000', 'step130000', 'step135000', 'step140000'],
                    'promptlen': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500],
                    # 'promptlen': [500],
                    # 'prompttype': ['standard', 'skipalt', 'end', 'corner', 'cornerdel'],
                    # 'instructions': [None, 'assistant', 'short', 'follows', 'excerpt', 'dan'],
                    # 'temperature': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    # 'beamwidth': [1, 2, 3, 4, 5],
                  }

    assert args.scoring!='length'

    all_scores = []
    all_acc = []
    for args_var in tqdm(args_change):
        for args_val in args_change[args_var]:
            restore_val = getattr(args, args_var)
            setattr(args, args_var, args_val)

            with open(path_to_scratch + '/extraction_results/' + get_filename(args, args_ignore=['scoring', 'batchsize', 'numgpus']), "rb") as fp:
                gen_arr = pickle.load(fp)

            score_arr = []
            for batch in gen_arr:
                scores = prompt_scoring(batch['completion_ids'], batch['outgen_ids'], args.scoring)
                score_arr.extend(scores)

            score_arr = np.array(score_arr)
            all_scores.append(score_arr)

            all_acc.append(100*np.mean(score_arr))

            setattr(args, args_var, restore_val)

    ## Extracted in at least one
    combined_scores = np.max(all_scores, axis=0)

    # ## Extracted in all
    # combined_scores = np.min(all_scores, axis=0)

    print(np.mean(combined_scores))

    # xlabels = ['100k', '105k', '110k', '115k', '120k', '125k', '130k', '135k', '140k']
    xlabels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    color1 = next(palette)
    plt.scatter(xlabels, all_acc, color=color1)
    plt.plot(xlabels, all_acc, '--', color=color1)
    plt.plot(xlabels, [100*np.mean(combined_scores)]*len(xlabels), '--', color='red', linewidth=4)
    plt.ylabel('Extraction Rate', labelpad=10)
    # plt.xlabel('Training Step', labelpad=10)
    plt.xlabel('Prompt Length', labelpad=10)
    plt.text(0, 3.6, 'Worst Case Extraction Rate', color='maroon')
    # plt.ylim(1., 1.8)
    plt.ylim(0, 4.)
    # plt.xticks(xlabels, ['100k', '', '110k', '', '120k', '', '130k', '', '140k'])
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], ['0', '', '100', '', '200', '', '300', '', '400', '', '500'])
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    plt.yticks(ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], labels=['0.0%', '', '1.0%', '', '2.0%', '', '3.0%', '', '4.0%'])
    # ax.set_facecolor('white')
    plt.tight_layout()
    # plt.savefig('extraction_over_time.pdf')
    plt.savefig('extraction_over_promptlen.pdf')


def zlib_eval(args):

    """
    Computes the ratio of perplexity to zlib-compression entropy for each completion.

    Args:
        args: Parsed arguments containing configuration details.
    """
    
    MODEL_USAGE = 'vllm'

    if MODEL_USAGE == 'hf':
        if 'pythia' in args.modelsize:
            model = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/{args.modelsize}",
                revision=args.modelstep,
            )
            padding_side = 'right'
            truncation_side = 'right'
            model_max_length = 2048
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/%s" % args.modelsize,
                revision=args.modelstep,
                padding_side=padding_side,
                truncation_side=truncation_side,
                model_max_length=model_max_length)

        if 'olmo' in args.modelsize:

            model = OLMoForCausalLM.from_pretrained(
                "allenai/OLMo-7B",
                revision=args.modelstep,
            )
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

    if MODEL_USAGE == 'vllm':
        if 'pythia' in args.modelsize:
            model = load_pythia_model(modelsize=args.modelsize,
                                      modelstep=args.modelstep)

            padding_side = 'right'
            truncation_side = 'right'
            model_max_length = 2048
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/%s" % args.modelsize,
                revision=args.modelstep,
                padding_side=padding_side,
                truncation_side=truncation_side,
                model_max_length=model_max_length)

        if 'olmo' in args.modelsize:
            model = load_olmo(args.modelstep)
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

    # computes the ratio of perplexity to zlib-compression entropy for each completion
    with open(
            path_to_scratch + '/extraction_results/' + get_filename(
                args, args_ignore=['scoring', 'batchsize', 'numgpus']),
            "rb") as fp:
        gen_arr = pickle.load(
            fp)  # reading the collection of generated completions

    ORIG_RATIOS = []
    GEN_RATIOS = []

    for batch in tqdm(gen_arr):

        prompt_generations = tokenizer.batch_decode(
            batch['prompt_ids']
        )  # list of X generations with Px tokens each (decoded)
        output_generations = tokenizer.batch_decode(
            batch['outgen_ids']
        )  # list of X generations with Ox tokens each (decoded)
        comp_gen = tokenizer.batch_decode(
            batch['completion_ids']
        )  # list of X original gens with C tokens each (decoded)

        #join them together
        completion_texts = [
            prompt + output
            for prompt, output in zip(prompt_generations, output_generations)
        ]
        original_texts = [
            prompt + suffix
            for prompt, suffix in zip(prompt_generations, comp_gen)
        ]

        original_gen_ratios, output_gen_ratios = zlib_ratio(
            output_generations=completion_texts,
            orig_generations=original_texts,
            model=model,
            tokenizer=tokenizer)

        print("original generation ratios:")
        print(original_gen_ratios[:10])
        print("\n -------- \n")
        print("output generation ratios:")
        print(output_gen_ratios[:10])

        ORIG_RATIOS.extend(original_gen_ratios)
        GEN_RATIOS.extend(output_gen_ratios)

        print("\n ######\n###### \n")

    with open(
            path_to_scratch + '/zlib_scores/' + get_filename(
                args, args_ignore=['scoring', 'batchsize', 'numgpus']),
            "wb") as fp:

        pickle.dump({
            'original_gen_ratios': ORIG_RATIOS,
            'output_gen_ratios': GEN_RATIOS
        }, fp)

    # print('now testing loading')

    # with open(
    #     path_to_scratch + '/zlib_scores/' + get_filename(
    #         args, args_ignore=['scoring', 'batchsize', 'numgpus']),
    #     "rb") as fp:
    #     data = pickle.load(fp)
    # print(data)




if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    scoring_mode = 'olmo'

    scoring_mode = 'pythia'

    if scoring_mode == 'olmo':
        assert (args.evalfile == 'finalidx300000_olmo_filtered.csv')

        # single_eval_score(args)
        # multiple_eval_single_axis(args, 'promptlen', [100, 150, 200, 250, 300, 350, 400, 450, 500], 0.9)
        multiple_eval_single_axis(args, 'modelstep', [
            'step300000', 'step360000', 'step400000', 'step440000',
            'step480000'
        ], 0.9)

    # single_eval_score(args)

    # multiple_eval_single_axis(args, 'promptlen', [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], 0.9)
    # multiple_eval_single_axis(args, 'modelstep', ['step100000', 'step105000', 'step110000', 'step115000',
    #                                               'step120000', 'step125000', 'step130000', 'step135000', 'step140000'], 0.7)
    # multiple_eval_single_axis(args, 'modelsize', ['pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'], 1.)

    # multiple_eval_multiple_axis(args, 'promptlen', [100, 300, 500], 'modelstep', ['step100000', 'step120000', 'step140000'],
    #                             'modelsize', ['pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b'], 1.)

    # multiple_eval_combined(args)
    # print_single_statement(args, 0)

    zlib_eval(args)
