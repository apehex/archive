import json
import os
import random
import sys

import tqdm

import dsl
import generators
import utils
import verifiers

from dsl import *
from utils import *

# CONSTANTS ####################################################################

PATH = 'data/temp/'
SEED = 1337

# UTILITIES ####################################################################

def get_generators() -> dict:
    """
    returns mapper from task identifiers (keys) to example generator functions
    """
    prefix = 'generate_'
    return {
        strip_prefix(n, prefix): getattr(generators, n) for n in dir(generators) if n.startswith(prefix)
    }


def get_verifiers() -> dict:
    """
    returns mapper from task identifiers (keys) to example verifier functions
    """
    prefix = 'verify_'
    return {
        strip_prefix(n, prefix): getattr(verifiers, n) for n in dir(verifiers) if n.startswith(prefix)
    }


def get_rng_difficulty(
    example: dict
) -> float:
    """
    RNG-Difficulty: proxy measure for example difficulty, defined as the mean of sampled floats within example generation
    """
    rng = getattr(utils, 'rng')
    setattr(utils, 'rng', [])
    return sum(rng) / len(rng)


def get_pso_difficulty(
    example: dict
) -> float:
    """
    PSO-Difficulty: proxy measure for example difficulty, defined as weighted sum of #Pixels, #Symbols, #Objects
    """
    i, o = example['input'], example['output']
    hwi = height(i) * width(i)
    hwo = height(o) * width(o)
    pix_pct = (hwi + hwo) / 1800
    col_pct = len(palette(i) | palette(o)) / 10
    obj_dens = (len(objects(i, T, F, F)) / hwi + len(objects(o, T, F, F)) / hwo) / 2
    return (pix_pct + col_pct + obj_dens) / 3


def demo_generator(key, n=6):
    with open(f'arc_original/training/{key}.json', 'r') as fp:
        original_task = json.load(fp)
    original_task = original_task['train'] + original_task['test']
    generator = getattr(generators, f'generate_{key}')
    generated_examples = [generator(0, 1) for k in range(n)]
    plot_task(original_task)
    plot_task(generated_examples)

# INSPECT ######################################################################

def check_sample(sample: dict, verifier_fn: callable) -> bool:
    return (
        utils.is_sample(sample)
        and sample['input'] != sample['output']
        and verifier_fn(sample['input']) == sample['output'])

# GENERATE #####################################################################

def generate_list(generator_fn: callable, verifier_fn: callable, count: int=3, diff_lb: float=0.0, diff_ub: float=1.0) -> tuple:
    __valid = []
    while len(__valid) < count:
        try:
            __s = generator_fn(diff_lb, diff_ub)
            if check_sample(sample=__s, verifier_fn=verifier_fn):
                __valid.append(__s)
        except:
            pass
    return __valid

def generate_task(generator_fn: callable, verifier_fn: callable, n_train: int=3, n_test: int=1, diff_lb: float=0.0, diff_ub: float=1.0) -> tuple:
    __n_train = round(random.uniform(2, max(1, n_train)))
    __n_test = round(random.uniform(1, max(1, n_test)))
    return {
        'train': generate_list(generator_fn=generator_fn, verifier_fn=verifier_fn, count=__n_train, diff_lb=diff_lb, diff_ub=diff_ub),
        'test': generate_list(generator_fn=generator_fn, verifier_fn=verifier_fn, count=__n_test, diff_lb=diff_lb, diff_ub=diff_ub),}

def generate_dataset(
    path: str=PATH,
    n_tasks: int=256,
    n_train: int=3,
    n_test: int=1,
    diff_lb: float = 0.0,
    diff_ub: float = 1.0
) -> None:
    """
    generates dataset

    path: which folder to save data to
    n_tasks: number of distinct task per challenge
    n_train: number of training examples per task
    n_test: number of testing examples per task
    diff_lb: lower bound for difficulty
    diff_ub: upper bound for difficulty
    """
    __generators = get_generators()
    __verifiers = get_verifiers()
    # iterate over
    __keys = sorted(__generators.keys())
    __status = 'challenge {{c}}/{n_challenges}, task {{t}}/{n_tasks}'.format(n_challenges=len(__keys), n_tasks=n_tasks)
    __pbar = tqdm.tqdm(enumerate(__keys), desc=__status.format(c=0, t=0), position=0, leave=True, total=len(__keys))
    # iterate over challenges
    for __i, __k in __pbar:
        __gen = __generators[__k]
        __che = __verifiers[__k]
        for __j in range(n_tasks):
            # generate an independent task
            __t = generate_task(generator_fn=__gen, verifier_fn=__che, n_train=n_train, n_test=n_test, diff_lb=diff_lb, diff_ub=diff_ub)
            # display progress
            __pbar.set_description(__status.format(c=__i, t=__j))
            # export the results
            with open(os.path.join(path, f'{__k}.{__j}.json'), 'w') as __f:
                json.dump(__t, __f)

# DISPLAY ######################################################################

def demo_dataset(
    folder: str = 're_arc',
    n: int = 8,
    s: int = 0,
    e: int = 400
) -> None:
    """
    visualizing snippets from a generated dataset (original, easy, medium and hard instances for each task)
    """
    with open(f'{folder}/metadata.json', 'r') as fp:
        metadata = json.load(fp)
    for i, fn in enumerate(sorted(os.listdir(f'{folder}/tasks'))):
        if s <= i < e:
            key = fn[:8]
            with open(f'arc_original/training/{key}.json', 'r') as fp:
                original_task = json.load(fp)
            with open(f'{folder}/tasks/{key}.json', 'r') as fp:
                generated_task = json.load(fp)
            original_task = [format_example(example) for example in original_task['train'] + original_task['test']]
            generated_task = [format_example(example) for example in generated_task[:10*n]]
            difficulties = metadata[key]['pso_difficulties'][:9*n]
            generated_task = [ex for ex, diff in sorted(zip(generated_task, difficulties), key=lambda item: item[1])]
            easy = generated_task[1*n:2*n]
            hard = generated_task[8*n:9*n]
            print(key)
            print('original:')
            plot_task(original_task)
            print('generated (easy):')
            plot_task(easy)
            print('generated (hard):')
            plot_task(hard)

# ORIGINAL DATA ################################################################

def evaluate_verifiers_on_original_tasks() -> None:
    """
    runs the verifiers on the original ARC training tasks
    """
    verifiers = get_verifiers()
    dataset = dict()
    for key in verifiers.keys():
        with open(f'arc_original/training/{key}.json', 'r') as fp:
            task = json.load(fp)
        dataset[key] = format_task(task)
    fix_bugs(dataset)
    failed_on = set()
    for key, verifier in verifiers.items():
        task = dataset[key]
        try:
            for example in task['train'] + task['test']:
                assert verifier(example['input']) == example['output']
        except:
            failed_on.add(key)
    n = len(dataset)
    k = len(failed_on)
    print(f'verification programs work for all examples for {n-k}/{n} tasks')
    print(f'verification fails (on one example) for tasks {failed_on}')

# MAIN #########################################################################

# if __name__ == '__main__':
#     # create the dirs
#     os.makedirs(PATH, exist_ok=True)
#     # setup the env
#     random.seed(SEED)
#     # default
#     generate_dataset()
