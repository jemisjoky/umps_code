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

samp_lens = [16, 30]                    # What lengths we want to sample at
samp_size = 1000                        # Number of samples to draw
dataset   = 'tomita'                    # Dataset models were trained on
save_name = ".tomita_exp.record"        # Where the record is saved

ALPHABET = {'brackets':    ['(', ')', '*'],
            'tomita':      ['0', '1'],
            'bos_eos':     ['^', '$'],
            }
alph = ALPHABET[dataset]

if dataset == 'brackets':
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
            ref_sets = to_string(ref_s, alph)
            samp_chars = fill_in_blanks(key, mps, alphabet=alph, 
                                        ref_strset=ref_s)
            # TODO: Fold this code into fill_in_blanks
            # Generate validation strings with each character replaced by
            # suggested character from samp_chars
            samples = [s[:i] + c + s[i+1:] for s, cs in zip(ref_sets, samp_chars)
                        for i, c in enumerate(cs)]
            corr_frac[samp_l] = 100 * score_fun(samples)
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {corr_frac[samp_l]:.1f}%")
            print(f"Replacement examples: {samples[:10]}\n")
    else:
        corr_frac = {}
        for samp_l in target_lens:
            rng_key, key = jax.random.split(rng_key)
            samples = draw_samples(key, mps, alphabet=alph, 
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
    this_alph = alph + ALPHABET['bos_eos']
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
            raise NotImplementedError
            ref_sets = [s[1:-1] for s in to_string(ref_s, this_alph)]
            samp_chars = lstm.sample(key, alph, 
                                samp_mode='completion', ref_strset=ref_s)

            # BOS and EOS should never be sampled, so replace those with
            # incorrect strings
            samples = [')(' if ('^' in s or '$' in s) else s for s in samples]
            corr_frac[samp_l] = 100 * score_fun(samples)
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {corr_frac[samp_l]:.1f}%")
            print(f"Replacement examples:{examp_samps[samp_l]}\n")

    else:
        corr_frac = {}
        for samp_l in target_lens:
            rng_key, key = jax.random.split(rng_key)
            samples = lstm.sample(key, this_alph, samp_mode=samp_mode,
                                     num_samps=samp_size, samp_len=samp_l)
            score = score_fun(samples)
            corr_frac[samp_l] = 100 * score
            examp_samps[samp_l] = samples[:10]
            print(f"Correct frac len={samp_l}:  {100 * score:.1f}%")
            print(f"Example samples: {examp_samps[samp_l]}\n")

    return corr_frac

rng_key = jax.random.PRNGKey(0)

# Load the data record we're interested in
full_record = pickle.load(open(save_name, 'rb'))

# Go through each experimental setting and resample with trained model
for setting, global_rec in full_record.items():
    # Get relevant data for this experimental setting
    print(setting)
    tom_num, _, _, model = setting[:4]
    assert model in ['mps', 'lstm']
    assert len(setting) in [4, 5]
    score_fun = partial(tom_score, tomita_num=tom_num)
    samp_fun = lstm_sample_fun if model == 'lstm' else mps_sample_fun
    best_model = global_rec['best_model']
    best_epoch = global_rec['best_epoch']
    local_rec = global_rec['local_recs'][best_epoch]

    # Figure out which lengths haven't been sampled yet
    these_lens = [l for l in samp_lens if f"corr_frac_{l}" not in local_rec]
    if these_lens == []: continue

    # Perform the resampling and add results to local_rec
    rng_key, key = jax.random.split(rng_key)
    corr_frac = samp_fun(key, best_model, these_lens, score_fun)
    for s_len, score in corr_frac.items():
        lookup = f"corr_frac_{s_len}"
        if lookup in local_rec:
            print(f"Already have samples from len {s_len}")
            continue
        local_rec[lookup] = score
    print

    # Put this back in full_record and save
    global_rec['local_recs'][best_epoch] = local_rec
    full_record[setting] = global_rec
    pickle.dump(full_record, open(save_name, 'wb'))