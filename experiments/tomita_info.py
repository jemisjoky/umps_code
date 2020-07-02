#!/usr/bin/env python3
import os
import sys
import pickle
sys.path.append('..')

# Important variable parameters for the data printing
# local_metrics = ['corr_frac_16']
local_metrics = ['corr_frac_16', 'corr_frac_30']
global_metrics = ['best_epoch', 'best_loss']
key_format = ['Tomita number', 'Train size',   # Human-friendly names for
              'Bond dim', 'Model']             # entries of record keys
sort_order = [0, 1, 3, 2]                      # Order to sort record keys
save_name = ".tomita_exp.record"               # Where the record is saved

# Derived parameters
sort_key = lambda xs: [xs[i] for i in sort_order]
max_len = max(len(s) for s in local_metrics + global_metrics + key_format)

# Printing function
def nice_print(keys, values, upper=False):
    """Nicely print a list of values associated with a list of keys"""
    template_string = "{0:<{ml}}: {1:{form}}{suff}"
    f_format = ".3f"
    time_suf = " sec"
    for key, val in zip(keys, values):
        if upper:
            key = key.upper()
            if isinstance(val, str): val = val.upper()
        form = f_format if isinstance(val, float) else ''
        suff = time_suf if 'time' in key else ''
        print(template_string.format(key, val, ml=max_len, form=form, 
                                     suff=suff))

# Load the data record we're interested in and sort it
full_record = pickle.load(open(save_name, 'rb'))
all_keys = sorted(full_record.keys(), key=sort_key)
full_record = {k: full_record[k] for k in all_keys}

# Sort then print the desired info
for settings in full_record:
    # Load all of the data for this experimental key/setting
    assert len(settings) in [4, 5]
    if len(settings) == 5: continue
    global_rec = full_record[settings]
    best_model = global_rec['best_model']
    best_epoch = global_rec['best_epoch']
    best_rec = global_rec['local_recs'][best_epoch]
    try:
        best_globals = [global_rec[met] for met in global_metrics]
        best_locals = [best_rec[met] for met in local_metrics]
    except KeyError:
        continue

    # Print the values of interest
    nice_print(key_format, settings, upper=True)
    nice_print(local_metrics, best_locals)
    nice_print(global_metrics, best_globals)
    print()