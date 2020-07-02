#!/usr/bin/env python3
import os
import sys
import pickle
from functools import partial
from string import ascii_lowercase

import jax
import torch

sys.path.append('..')
from ti_mps import TI_MPS
from train_tools import init_strset, to_string

samp_lens = [1, 16, 50]                 # What lengths we want to sample at
samp_size = 1000                        # Number of samples to draw
comp_exp  = False                       # Whether to run completion or
                                        # sampling task
dataset   = 'brackets'                  # Dataset models were trained on
save_name = ".motzkin_exp.record"       # Where the record is saved

ALPHABET = {'brackets':    ['(', ')', '*'],
            'tomita':      ['0', '1'],
            'bos_eos':     ['^', '$'],
            }
alph_noends = ALPHABET[dataset]
alph_ends = alph_noends + ALPHABET['bos_eos']

if dataset == 'brackets':
    from toy_datasets import brackets_dataset
    from toy_datasets import score_brackets as score_fun
elif dataset == 'tomita':
    from toy_datasets import score_tomita as tom_score

def is_lstm(model):
    assert isinstance(model, (torch.nn.Module, TI_MPS))
    return isinstance(model, torch.nn.Module)

def mps_sample_fun(rng_key, mps, target_lens, score_fun, ref_sets=None):
    """Draw samples from MPS model within JAX"""
    from sampler import draw_samples, fill_in_blanks
    bi_exp = ref_sets is not None

    examp_samps = {}
    if bi_exp:
        corr_frac = {}
        for samp_l in target_lens:
            ref_s = ref_sets[samp_l]
            ref_strs = to_string(ref_s, alph_noends)
            rng_key, key = jax.random.split(rng_key)
            samp_chars = fill_in_blanks(key, mps, alphabet=alph_noends, 
                                        ref_strset=ref_s)
            # TODO: Fold this code into fill_in_blanks
            # Generate validation strings with each character replaced by
            # suggested character from samp_chars
            samples = [s[:i] + c + s[i+1:] for s, cs in zip(ref_strs, samp_chars)
                        for i, c in enumerate(cs)]
            corr_frac[samp_l] = 100 * score_fun(samples)
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {corr_frac[samp_l]:.1f}%")
            print(f"Replacement examples: {samples[:10]}\n")
    else:
        corr_frac = {}
        for samp_l in target_lens:
            rng_key, key = jax.random.split(rng_key)
            samples = draw_samples(key, mps, alphabet=alph_noends, 
                           num_samps=samp_size, samp_len=samp_l)
            score = score_fun(samples)
            corr_frac[samp_l] = 100 * score
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {100 * score:.1f}%")
            print(f"Example samples: {samples[:10]}\n")

    return corr_frac

def lstm_sample_fun(rng_key, lstm, target_lens, score_fun, ref_sets=None):
    """Draw samples from LSTM model within Pytorch"""
    samp_mode = 'fixed'
    bi_exp = lstm.bi_dir
    lstm = lstm.eval()
    examp_samps = {}
    if bi_exp:
        corr_frac = {}
        for samp_l in target_lens:
            ref_s = ref_sets[samp_l]
            rng_key, key = jax.random.split(rng_key)

            # TODO: Finish up better bidirectional sampling code, including
            #       (a) deal with BOS/EOS, (b) properly put samp_chars in
            #       ref_set strings
            ref_strs = [s[1:-1] for s in to_string(ref_s, alph_ends)]
            samples = lstm.sample(key, alph_ends, 
                                samp_mode='completion', ref_strset=ref_s)

            # BOS and EOS should never be sampled, so replace those with
            # incorrect strings
            assert not any(('^' in s or '$' in s) for s in samples)
            # samples = [')(' if ('^' in s or '$' in s) else s for s in samples]
            corr_frac[samp_l] = 100 * score_fun(samples)
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {corr_frac[samp_l]:.1f}%")
            print(f"Replacement examples:{examp_samps[samp_l]}\n")

    else:
        corr_frac = {}
        for samp_l in target_lens:
            rng_key, key = jax.random.split(rng_key)
            samples = lstm.sample(key, alph_ends, samp_mode=samp_mode,
                                     num_samps=samp_size, samp_len=samp_l)
            score = score_fun(samples)
            corr_frac[samp_l] = 100 * score
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {100 * score:.1f}%")
            print(f"Example samples: {examp_samps[samp_l]}\n")

    return corr_frac

cf_form = "corr_frac_bi_{}" if comp_exp else "corr_frac_{}"
rng_key = jax.random.PRNGKey(0)

# Load the data record we're interested in
full_record = pickle.load(open(save_name, 'rb'))

# Get a StrSet containing brackets of interest
if comp_exp:
    ref_sets_ends = {}
    ref_sets_noends = {}
    for samp_l in samp_lens:
        rng_key, key = jax.random.split(rng_key)
        min_l = samp_l if samp_l < 18 else 1
        try:
            ref_se = brackets_dataset(rng_key=key, 
                                     data_split=samp_size,
                                     max_len=samp_l, 
                                     min_len=min_l,
                                     add_ends=True)
            ref_sne = brackets_dataset(rng_key=key, 
                                     data_split=samp_size,
                                     max_len=samp_l, 
                                     min_len=min_l,
                                     add_ends=False)
        except:
            assert samp_l == 1
            ref_se = brackets_dataset(rng_key=key, 
                                     data_split=0.99999,
                                     max_len=samp_l, 
                                     min_len=min_l,
                                     add_ends=True)
            ref_sne = brackets_dataset(rng_key=key, 
                                     data_split=0.99999,
                                     max_len=samp_l, 
                                     min_len=min_l,
                                     add_ends=False)
            ref_se, ref_sne = ref_se * samp_size, ref_sne * samp_size

        ref_sets_ends[samp_l] = init_strset(ref_se, alph_ends)
        ref_sets_noends[samp_l] = init_strset(ref_sne, alph_noends)
else:
    ref_sets_ends = None
    ref_sets_noends = None

# Go through each experimental setting and resample with trained model
for setting, global_rec in full_record.items():
    # Get relevant data for this experimental setting
    print(setting)
    _, _, model = setting[:3]
    assert model in ['mps', 'lstm']
    assert len(setting) in [3, 4]
    samp_fun = lstm_sample_fun if model == 'lstm' else mps_sample_fun
    best_model = global_rec['best_model']
    best_epoch = global_rec['best_epoch']
    local_rec = global_rec['local_recs'][best_epoch]

    # Figure out which lengths haven't been sampled yet
    these_lens = [l for l in samp_lens if cf_form.format(l) not in local_rec]
    if these_lens == []: continue

    # Perform the resampling and add results to local_rec
    rng_key, key = jax.random.split(rng_key)
    corr_frac = samp_fun(key, best_model, these_lens, score_fun, ref_sets=(
                         ref_sets_ends if model=='lstm' else ref_sets_noends))
    for s_len, score in corr_frac.items():
        lookup = cf_form.format(s_len)
        if lookup in local_rec:
            print(f"Already have samples from len {s_len}")
            continue
        local_rec[lookup] = score

    # Put this back in full_record and save
    global_rec['local_recs'][best_epoch] = local_rec
    full_record[setting] = global_rec
    pickle.dump(full_record, open(save_name, 'wb'))