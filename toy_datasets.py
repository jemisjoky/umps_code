#!/usr/bin/env python3
import os
import re
import sys
import random
import warnings
from collections import OrderedDict
from itertools import product
from functools import reduce, lru_cache

import jax
import numpy as np
import jax.numpy as jnp

from utils import disk_cache
from train_tools import init_strset

def splitter(rng_key, str_list, data_split):
    """
    Randomly extract a collection of disjoint datasets from list of strings
    """
    # Process input, deal with variability in data_split
    single_split = not hasattr(data_split, '__len__')
    if single_split: data_split = [data_split]
    frac_split = all([n < 1 for n in data_split])
    num_out, num_in = sum(data_split), len(str_list)
    if frac_split:
        data_split = [round(num_in * f) for f in data_split]

    # Check input and desired dataset sizes
    if num_out > num_in:
        raise ValueError(f"The number of requested strings ({num_out}) is "
                         "more than the available Tomita strings in the "
                         f"requested length range ({num_in})")

    # Shuffle our data before pulling out different splits
    rand_indices = jax.random.shuffle(rng_key, np.arange(len(str_list)))
    str_list = [str_list[i] for i in rand_indices]

    ind, datasets = 0, []
    for split in data_split:
        datasets.append(str_list[ind: ind+split])
        ind += split

    return datasets

### TOMITA GRAMMARS ###
@disk_cache('tomita.data')
def tomita_dataset(rng_key, data_split, max_len, tomita_num, min_len=1,
                   add_ends=False, as_strset=False):
    """
    Get dataset of strings from a Tomita grammar with desired length(s)
    """
    assert max_len >= min_len > 0
    assert 7 >= tomita_num >= 1
    assert isinstance(tomita_num, int)
    tomita_fun = globals()[f"tomita_{tomita_num}"]
    core_alph = ['0', '1']
    full_alph = ['0', '1', '^', '$'] if add_ends else core_alph

    # Enumerate over all binary strings, take only the ones in the grammar
    str_list = []
    for n in range(min_len, max_len+1):
        for s in product(core_alph, repeat=n):
            s = "".join(s)
            if tomita_fun(s):
                str_list.append(s)

    # Add BOS and EOS chars if needed
    if add_ends: str_list = [f"^{s}$" for s in str_list]

    # Split the big list of strings into a collection of datasets, which
    # are either lists of strings or StrSet objects
    datasets = splitter(rng_key, str_list, data_split)
    if as_strset:
        datasets = [init_strset(ds, full_alph) for ds in datasets]
    return datasets

def score_tomita(trial_strs, tomita_num):
    """
    Calculate fraction of trial strings which belong in brackets language

    Args:
        trial_strs: List of strings we're testing for proper bracketing
    """
    num_strs = len(trial_strs)
    tomita_fun = globals()[f"tomita_{tomita_num}"]
    has_ends = any('^' in s or '$' in s for s in trial_strs)

    # Deal with strings which have beginning/end of sequence tokens
    if has_ends:
        trial_strs = [s[1:-1] for s in trial_strs
                      if s[0]=='^' and s[-1]=='$']
    num_correct = sum(1 for s in trial_strs if tomita_fun(s))
    correct_frac = num_correct / num_strs

    return correct_frac

@disk_cache('tomita_size.data')
def tomita_size(frac_split, min_len, max_len, tomita_num):
    """
    Get the absolute size of a given fractional split of a Tomita grammar
    """
    if hasattr(frac_split, '__len__'):
        assert all(fs < 1 for fs in frac_split)
        single_split = False
    else:
        assert frac_split < 1
        single_split = True
    rng_key = jax.random.PRNGKey(0)
    dataset = tomita_dataset(rng_key, data_split=frac_split, max_len=max_len, 
                             tomita_num=tomita_num, min_len=min_len)

    return len(dataset[0]) if single_split else tuple(len(ds) for ds in dataset)

# The following code comes from https://github.com/tech-srl/lstar_extraction
# 1*
def tomita_1(word):
    return not "0" in word
# (10)*
def tomita_2(word):
    return word=="10"*(int(len(word)/2))
import re
_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$") 
# *not* tomita 3: words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
# tomita 3: opposite of that
# Odd number of consecutive 1's followed by even number 
def tomita_3(w): 
    return None is _not_tomita_3.match(w) #complement of _not_tomita_3
# 000 not in word
def tomita_4(word):
    return not "000" in word
# Even number of 0's, even number of 1's
def tomita_5(word):
    return (word.count("0")%2 == 0) and (word.count("1")%2 == 0)
# Difference between number of 0's and number of 1's is multiple of 3
def tomita_6(word):
    return ((word.count("0")-word.count("1"))%3) == 0
# 0*1*0*1*
def tomita_7(word):
    return word.count("10") <= 1

### BRACKETS DATASET ###

@disk_cache('brackets.data')
def brackets_dataset(rng_key, data_split, max_len, min_len=1, 
                     add_ends=False, _prob_params=None):
    """
    Get disjoint datasets of fixed len star and matched parenthesis strings

    For max_len=5 this yields strings like '*(*)()', '(()*)', '**()*', etc.

    Args:
        rng_key:    JAX PRNGKey object for randomly shuffling our dataset
        data_split: Tuple giving the number of strings in each of our 
                    separate datasets. Alternately, we can specify the 
                    fraction of the total number of strings in each split,
                    although this only for works for strings of length < 18
                    (too many strings to practically enumerate)
        max_len:    Length of all of the strings our dataset
        add_ends:   Whether or not to BOS and EOS chars at start and end
        _prob_params: Custom parameters for the probabilistic generator

    Returns:
        datasets:   List of datasets, with each dataset a list of strings
    """
    assert max_len >= 0

    # Parse our input and pull out fractional split and size-specific info
    single_split = isinstance(data_split, (int, float))
    if single_split:
        data_split = [data_split]
    num_strs = sum(data_split)

    assert all([n >= 1 for n in data_split]) or \
           all([n < 1 for n in data_split])
    frac_split = all([n < 1 for n in data_split])
    short_string = max_len < 18
    assert short_string or min_len == 1, "Non-unit min_len needs max_len < 18"
    use_exact = short_string and _prob_params is None

    if frac_split and not short_string:
        raise ValueError("Fractional dataset splits only available for"
                         "short strings s, with len(s) < 18")

    # For short strings, generate everything and then prune it
    if use_exact:
        datasets = _brackets_exact(rng_key, min_len, max_len, 
                                   data_split, frac_split)

    # For longer strings, use a probabilistic generator
    else:
        # Probabilistic generator doesn't check for uniqueness, so call it
        # however long is needed to generate them all
        datasets = [[]] * len(data_split)
        all_strs = {}
        while any(len(ds) != spl for ds, spl in zip(datasets, data_split)):
            remaining = [spl - len(ds) for ds, spl in zip(datasets, data_split)]
            dsets = _brackets_rand(rng_key, max_len, remaining, _prob_params)

            # Remove any non-unique entries from dsets
            dsets = [list(OrderedDict((s, None) for s in ds).keys())
                                                    for ds in dsets]
            datasets = [ds + d for ds, d in zip(datasets, dsets)]
            print(f"Brackets dataset lens: {[len(ds) for ds in datasets]}")

    # Add BOS/EOS chars
    if add_ends: datasets = [[f"^{s}$" for s in ds] for ds in datasets]
    return datasets[0] if single_split else datasets

def _brackets_exact(rng_key, min_len, max_len, data_split, frac_split):
    """Enumerate all bracket strings then select some using data_split"""
    @lru_cache()
    def valid_strs(n):
        """Recursively iterate over all valid strings of length n"""
        assert n >= 0
        if n < 2:
            return ['*'] if n == 1 else ['']
        else:
            # Start with all strings that are parenthesized smaller strings
            out_strs = ['('+p+')' for p in valid_strs(n-2)]

            # Build up everything else as concatenations of smaller strings
            other_strs = set()
            for k in range(1,n):
                these_strs = set(l+r for l,r in product(valid_strs(k),
                                                        valid_strs(n-k)))
                other_strs.update(these_strs)
            out_strs.extend(list(other_strs))

            return out_strs

    # Get all strings with lengths between min_len and max_len, inclusive
    all_strs = []
    for n in range(min_len, max_len + 1):
        all_strs += [s for s in valid_strs(n)]
    num_strs = len(all_strs)

    # Get info about dataset sizes
    if frac_split:
        data_split = [round(num_strs * f) for f in data_split]
    assert sum(data_split) <= num_strs

    # Shuffle our data before pulling out different splits
    rand_indices = jax.random.shuffle(rng_key, np.arange(len(all_strs)))
    all_strs = [all_strs[i] for i in rand_indices]

    ind, datasets = 0, []
    for split in data_split:
        datasets.append(all_strs[ind: ind+split])
        ind += split

    return datasets

def _brackets_rand(rng_key, str_len, data_split, _prob_params=None):
    """Generate random strings using probabilistic tree automaton"""
    # Probabilities for different branching options
    if _prob_params is None:
        p_fork = 0.5
        p_star = 0.53 * (1 - p_fork)
        p_nest = 0.47 * (1 - p_fork)
        _prob_params = (p_fork, p_star, p_nest)
    assert all(p >= 0 for p in _prob_params)
    _prob_params = np.array(_prob_params)

    _short_params = _prob_params[:2] / np.sum(_prob_params[:2])
    assert jnp.isclose(np.sum(_prob_params), 1)
    assert jnp.isclose(np.sum(_short_params), 1)

    # We want to enumerate over all_strs, each time growing the strings.
    # To do that, we first define a few helper functions
    real_len = lambda s: len(s.replace('@', ''))
    fill_len = lambda s: len(s) - real_len(s)
    stop_cond = lambda s_list: all([real_len(s) == str_len for s in s_list])

    # Find a uniformly random '#' character and expand it randomly
    def grow_str(rng_key, s, no_nest=False):
        rng_key, k1, k2, k3 = jax.random.split(rng_key, num=4)
        num_fill = fill_len(s)
        all_inds = [m.start() for m in re.finditer(r'@', s)]
        assert len(all_inds) == num_fill
        rand_ind = all_inds[jax.random.randint(k1, (), 0, num_fill)]

        # Pick a replacement method for target '@' character at rand_ind
        cum_probs = jnp.cumsum(_short_params if no_nest else _prob_params)
        rand_u = jax.random.uniform(k2)
        rep_method = int(np.argmax(cum_probs > rand_u))
        if rep_method == 0:
            rep_str = '@@'
        elif rep_method == 1:
            go_left = bool(jax.random.bernoulli(k3))
            rep_str = '@*' if go_left else '*@'
        elif rep_method == 2:
            rep_str = '(@)'

        return s[:rand_ind] + rep_str + s[rand_ind+1:]

    def cond_grow(key, s):
        r_len = real_len(s)
        assert r_len <= str_len
        no_nest = r_len == str_len - 1

        return s if real_len(s) == str_len else grow_str(key, s, no_nest)

    # Time to do the iterated growing
    num_strs = sum(data_split)
    all_strs = ['@'] * num_strs
    while not stop_cond(all_strs):
        keys = jax.random.split(rng_key, num=num_strs+1)
        rng_key = keys[0]
        all_strs = [cond_grow(k, s) for k, s in zip(keys[1:], all_strs)]

    all_strs = [s.replace('@', '') for s in all_strs]

    # Shuffle our data before pulling out different splits
    rand_indices = jax.random.shuffle(rng_key, np.arange(len(all_strs)))
    all_strs = [all_strs[i] for i in rand_indices]

    ind, datasets = 0, []
    for split in data_split:
        datasets.append(all_strs[ind: ind+split])
        ind += split

    return datasets

@disk_cache('brackets_size.data')
def brackets_size(frac_split, min_len, max_len):
    """
    Get the absolute size of a fractional split of the brackets dataset
    """
    if hasattr(frac_split, '__len__'):
        assert all(fs < 1 for fs in frac_split)
        single_split = False
    else:
        assert frac_split < 1
        single_split = True
    rng_key = jax.random.PRNGKey(0)
    dataset = brackets(rng_key, data_split=frac_split, max_len=max_len, 
                       min_len=min_len)
    
    return len(dataset[0]) if single_split else tuple(len(ds) for ds in dataset)


def score_brackets(trial_strs):
    """
    Calculate fraction of trial strings which belong in brackets language

    Args:
        trial_strs: List of strings we're testing for proper bracketing
    """
    char_set = reduce((lambda curr_chars, s: curr_chars.union(set(s))), 
                      trial_strs, set())
    bracket_alphabet = {'(', ')', '*'}
    assert char_set.issubset(bracket_alphabet)
    num_strs = len(trial_strs)

    height = {'(': 1, '*': 0, ')': -1}
    def good_brack(my_str):
        h = 0
        for c in my_str:
            h += height[c] 
            if h < 0:
                return False
        if h == 0:
            return True
        else:
            return False

    good_brackets = list(map(good_brack, trial_strs))
    assert len(good_brackets) == num_strs
    correct_frac = len([gb for gb in good_brackets if gb]) / num_strs

    return correct_frac


### DEMARCATED PARITY DATASET ###

def parity_dataset(data_split: tuple, str_len: int, num_f=2, parity=0, 
                   as_tensors=True, disjoint=True, seed=None):
    """
    Get disjoint datasets of fixed-len regularly-demaracted parity strings

    Args:
        data_split: Tuple giving the number of strings in each of our 
                    separate datasets
        str_len:    Length of the strings in each of our datasets
        num_f:      The number of regularly spaced divider tokens in each 
                    parity substring
        parity:     Fixed parity of all parity substrings, must be 0 or 1
        as_tensors: Whether to return dataset as onehot-encoded tensors,
                    or as lists of strings
        seed:       A seed for our dataset generation

    Returns:
        datasets:   Tuple containing the onehot-encoded batch tensors for
                    each of our datasets. If as_tensors is True, datasets
                    is instead a tuple of lists of strings.
    """
    assert num_f >= 2
    assert str_len >= 3 * num_f - 2
    if isinstance(data_split, int):
        data_split = (data_split,)
    num_ds = len(data_split)
    num_ps = num_f - 1
    p_chars = str_len - num_f
    p_len, num_long = divmod(p_chars, num_ps)

    # Make sure we'll have enough unique parity strings
    total_strs = 2**(num_ps * (p_len-1) + num_long)
    fill_fraction = sum(data_split) / total_strs
    assert fill_fraction <= 1, (
           f"Only {total_strs} unique parity strings, and you're asking "
           f"for {sum(data_split)} of them")
    if fill_fraction > 0.9:
        warnings.warn(f"Your filling fraction is {fill_fraction}, "
                       "which will slow down dataset generation")

    # Hash set letting us maintain uniqueness of sampled substrings
    unique_strs = set()

    def get_par_string(this_len):
        """Get a single definite-parity string of length this_len"""
        assert this_len >= 2
        random_part = np.random.randint(2, size=(this_len-1,))
        needed_parity = (parity + np.sum(random_part)) % 2

        initial = "".join([str(k) for k in random_part])
        return initial + str(needed_parity)

    if seed is not None:
        np.random.seed(seed)

    # Generate a template for how long each of the substrings should be
    lens = []
    float_lens = [(str_len - num_f) / (num_f - 1)] * (num_f - 1)
    for i in range(num_f-1):
        new_len = round(float_lens[i])
        lens.append(new_len)
        if i < num_f - 2:
            float_lens[i+1] += float_lens[i] - new_len
    assert sum(lens) == str_len - num_f

    # Build up our regularly-demarcated parity strings one at a time
    datasets = ()
    for num_strs in data_split:
        sample_list = []
        while len(sample_list) < num_strs:
            next_str = 'f'
            for this_len in lens:
                next_str += get_par_string(this_len) + 'f'
            assert len(next_str) == str_len

            # Check for uniqueness of our generated string
            if next_str in unique_strs:
                continue
            unique_strs.add(next_str)

            # Convert this_str to a one-hot sequence matrix
            num_repr = [int(c) if c != 'f' else 2 for c in next_str]
            ones_index = (list(range(str_len)), num_repr)

            next_mat = np.zeros((str_len, 3))
            next_mat[ones_index] = 1

            # Add sequence matrix to our running list
            sample_list.append(next_mat if as_tensors else next_str)

        datasets = datasets + (np.stack(sample_list) if as_tensors
                                   else sample_list,)

    return datasets


### RANDOM PARITY DATASET ###

def rand_parity_dataset(num_strs, str_len, mean_len=8, parity=0, 
                        no_null=False, as_tensor=True, seed=None):
    """
    Return dataset of randomly demaracted parity strings of length str_len

    Args:
        num_strs: [Int] The number of strings in our dataset
        str_len: [Int] The length of each string in our dataset
        mean_len: [Num] The average length of parity substrings
        parity: [0 or 1] The parity for each of our parity substrings
        no_null: [Bool] Whether empty parity substrings are allowed
        as_tensors: [Bool] Whether to return a Numpy array with one-hot
                    encodings or a list with the strings themselves
        seed: [Int] A seed for our dataset generation

    Returns:
        dataset: Numpy array or list containing our dataset
    """
    assert parity in [0, 1]
    par_iter = parity_iter(mean_len=mean_len, parity=parity, 
                           no_null=no_null, seed=seed)

    sample_list = []
    for _ in range(num_strs):
        # Sample str_len characters and convert to one-hot sequence matrix
        next_str = "".join([next(par_iter) for _ in range(str_len)])
        num_repr = [int(c) if c != 'f' else 2 for c in next_str]
        ones_index = (list(range(str_len)), num_repr)

        next_mat = np.zeros((str_len, 3))
        next_mat[ones_index] = 1

        # Add it to our running list
        sample_list.append(next_mat if as_tensor else next_str)

    dataset = np.stack(sample_list) if as_tensor else sample_list

    return dataset

def parity_iter(mean_len=8, parity=0, no_null=False, seed=None):
    """
    Iterator over a single demarcated parity stream

    Args: mean_len: [Float] Mean length for the geometric distribution
                    determining parity string length within output stream

          even_parity: [Bool] All binary substrings between boundary tokens
                         will have identical parity, namely int(even_parity)

          no_null: [Bool] Whether empty parity substrings are allowed

          seed: [Int] Seed for the random number generator that generates
                  streams of demarcated parity strings

    Returns: [Iterator] Returns digits in parity string, punctuated by
             boundary character 'f'
    """
    assert mean_len > 0
    assert parity in [0, 1]
    if seed is not None:
        np.random.seed(seed)
    needed_parity = parity

    # Initialize parameters
    running_parity, f_seen, p_seen = 0, False, False

    # Set the probability defining our geometric length distribution
    f_prob = 1 / (1 + mean_len / 2)

    # Start sampling characters and don't stop
    while True:

        # 'f' isn't sampled if parity and null constraints aren't satisfied
        if f_seen and (running_parity != needed_parity or not p_seen):
            new_digit = np.random.randint(0, 2)
            running_parity = (running_parity + new_digit) % 2
            p_seen = True
            
            yield str(new_digit)

        # Decide if we want a 'f', and if not then sample another digit
        else:
            if np.random.uniform() < f_prob:
                running_parity, f_seen, p_seen = 0, True, False
                yield 'f'

            else:
                new_digit = np.random.randint(0, 2)
                running_parity = (running_parity + new_digit) % 2
                
                yield str(new_digit)


def random_dataset(num_strs, str_len, as_tensor=True, seed=None):
    """
    Return dataset of fully random strings of length str_len

    Args:
        num_strs: [Int] The number of strings in our dataset
        str_len: [Int] The length of each string in our dataset
        as_tensors: [Bool] Whether to return a Numpy array with one-hot
                    encodings or a list with the strings themselves
        seed: [Int] A seed for our dataset generation

    Returns:
        dataset: Numpy array or list containing our dataset
    """
    if seed is not None:
        np.random.seed(seed)

    sample_list = []
    for _ in range(num_strs):
        # Sample str_len characters and convert to one-hot sequence matrix
        rand_str = np.random.randint(3, size=(str_len,))
        ones_index = (list(range(str_len)), rand_str)

        next_mat = np.zeros((str_len, 3))
        next_mat[ones_index] = 1

        # Add it to our running list
        sample_list.append(next_mat if as_tensor else next_str)

    dataset = np.stack(sample_list) if as_tensor else sample_list

    return dataset

def parse_parity(par_string, give_substrings=False, discard_empties=False):
    """
    Given demarcated parity string, get parity and lengths of substrings
    
    Args:
        par_string:      Input string (or list of strings) composed of 
                         characters '0', '1', and 'f'
        give_substrings: If True, returns list of parity substrings

    Returns:
            par_list: Input string is converted to list of Booleans,
                         with each entry saying if the corresponding binary
                      substring has correct parity
            len_list: Each entry gives the length of the corresponding 
                      binary substring
            

    """
    # Convert par_string into a list
    no_batch = False
    if isinstance(par_string, np.ndarray):
        par_list = vecs_to_strings(par_string)
    elif isinstance(par_string, str):
        no_batch = True
        par_list = [par_string]
    else:
        par_list = list(par_string)

    assert isinstance(par_list, list)
    assert all([set(p).issubset({'0', '1', 'f'}) for p in par_list])

    # Parse par_list into lists of substrings
    substrings = [p.split('f') for p in par_list]
    if discard_empties:
        substrings = [[s for s in ss if len(s) > 0] for ss in substrings]
    par_list = [[sum([int(c) for c in s]) % 2 for s in ss] 
                                                   for ss in substrings]
    len_list = [[len(s) for s in ss] for ss in substrings]

    if no_batch:
        par_list = par_list[0]
        len_list = len_list[0]
        substrings = substrings[0]

    if give_substrings:
        return par_list, len_list, substrings
    else:
        return par_list, len_list

def score_parity(par_strings, correct_par=0):
    """
    Assess the fraction of parity substrings with the correct parity
    """
    assert correct_par in [0, 1]
    assert isinstance(par_strings, list)
    num_samps = len(par_strings)

    # Get the parity and length of all the parity substrings
    par_list, len_list = parse_parity(par_strings)
    assert len(par_list) == len(len_list) == num_samps

    # Calculate statistics of the output strings
    num_pars, corr_pars, good_samps = 0, 0, 0
    for pl, ll in zip(par_list, len_list):
        num_pars += len(pl)
        corr_pars += len([p for p in pl if p == correct_par])

        # Check that the whole string is valid demarcated parity string,
        # meaning it has an f on each end, all parity substrings are valid
        if ll[0]==ll[-1]==0 and all([p == correct_par for p in pl[1:-1]]):
            good_samps += 1
    corr_par_frac = corr_pars / num_pars
    corr_samp_frac = good_samps / num_samps
    par_per_samp = num_pars / num_samps

    return corr_par_frac, corr_samp_frac, par_per_samp

def vecs_to_strings(tensor):
    """
    Given one-hot encoded parity data, convert to equivalent strings

    Args: 
        tensor: NP array of shape (batch_size, str_len, 3) or (str_len, 3)
    """
    shape = tensor.shape
    assert shape[-1] == 3 and len(shape) in {2, 3}

    # If tensor doesn't have batch dimension, add it
    str_len = shape[-2]
    if len(tensor.shape) == 3:
        batch_size = shape
        no_batch = False
    else:
        batch_size, shape = 1, (1, str_len, 3)
        tensor = np.broadcast_to(tensor, shape)
        no_batch = True
    assert np.allclose(np.sum(tensor ** 2, 2), 1.)

    # Get the strings corresponding to each sequence matrix
    nums = [np.argmax(mat, axis=1) for mat in tensor]
    strings = [['f' if n == 2 else str(n) for n in s] for s in nums]
    strings = ["".join(s) for s in strings]

    if no_batch:
        assert len(strings) == 1
        return strings[0]
    else:
        return strings

if __name__ == '__main__':
    max_len    = 15
    data_split = 0.99999999
    rng_key = jax.random.PRNGKey(0)

    for tomita_num in range(1,8):
        dataset = tomita_dataset(rng_key, data_split, max_len, 
                                 tomita_num=tomita_num)[0]
        # dataset = [f"^{s}$" for s in dataset]
        dset_len = len(dataset)
        assert score_tomita(dataset + ['01110001100'], tomita_num) < \
               score_tomita(dataset, tomita_num) == 1.
        assert tomita_size(data_split, max_len, tomita_num)[0] == dset_len
        print(f"Tomita {tomita_num} with max_len={max_len} has "
              f"{dset_len} strings")