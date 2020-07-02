#!/usr/bin/env python3
import os
import sys
import pickle
from itertools import product

sys.path.append('..')
from universal import run_experiment

### DEFAULTS ###

exp_args = {'comet_log':   False,
            'group_name':  '',
            'config_name': '',

            ### TOMITA ONLY ###
            'tomita_num':  3,

            'model':       'lstm',
            'dataset':     'brackets',
            'bond_dim':    20,

            'num_epochs':  30,
            # 'train_size':  10,  # Tomita 1
            # 'other_size':  2,   # Tomita 1
            # 'train_size':  1000,
            # 'other_size':  1000,
            'max_len':     15,
            'min_len':     1,
            'use_val':     True,
            'use_test':    False,

            'bi_exp':      True,
            'samp_size':   1000,
            'samp_lens':   [15],
            # 'bos_eos':     False,
            # 'input_dim':   3,

            'optimizer':   'adam',
            'learn_rate':  1e-2,
            'lr_sched':    ['const', 10, 1e-5],  # [scheduler, time_const, min_lr],
            # 'lr_sched':    ['const'],
            # 'mass':        1e-1,
            'mass':        None,
            'batch_size':  50,

            'early_stop':  True,
            'save_record': False,
            'verbose':     True,
            'rand_seed':   0,
            'fixed_dset':  True,

            ### MPS ONLY ###
            'init_method': 'eye',
            'bound_cond':  'open',
            'noise':       1e-6,
            'contract':    'parallel',

            ### LSTM ONLY ###
            'dropout':     0,
            'pos_enc':     False,
            'pe_dim':      6,
            'samp_mode':   'fixed',
            }
from toy_datasets import brackets_size

def default_args(args):
    # Incorporate some defaults for ease of flipping settings
    if args['model'] == 'lstm':
        args['bos_eos'] = True  # Needed for unidirectionality of LSTM
        args['optimizer'] = args['optimizer'].title()
    elif args['model'] == 'mps':
        args['bos_eos'] = False # Model is already bidirectional
        args['optimizer'] = args['optimizer'].lower()
    if args['dataset'] == 'brackets':
        args['input_dim'] = 3
    elif args['dataset'] == 'tomita':
        args['input_dim'] = 2
    # Add 2 to the input dimension if we're using BOS/EOS
    if args['bos_eos']:
        args['input_dim'] += 2
    return args



### START OF THE REAL SCRIPT ###



# Important variable parameters for experiment
comet_log   = False                  # Whether to log data with comet.ml
group_name  = 'motzkin_exp'          # Name of comet.ml experiment folder
num_trials  = 5                      # Number of random seeds to try
min_len     = 15                     # Minimum length of Motzkin strings
max_len     = 15                     # Maximum length of Motzkin strings
samp_lens   = [16]                   # Lengths of strings we will sample
epochs      = 100                    # Number of epochs to train for
save_name   = ".motzkin_exp.record"  # File name for record
train_sizes = [1000, 10000]          # List in increasing order
models      = ['mps', 'lstm']        # Which models we're trying out
bond_dims   = [20, 50]               # Sizes of hidden state spaces
other_size  = 1000                   # Size of validation/test sets
comp_exp    = False                  # Whether to carry run a completion
                                     # experiment or a sampling experiment

# Derived parameters for experiment
exp_args['min_len']    = min_len
exp_args['max_len']    = max_len
exp_args['samp_lens']  = samp_lens
exp_args['comet_log']  = comet_log
exp_args['group_name'] = group_name
exp_args['num_epochs'] = epochs
exp_args['other_size'] = other_size
exp_args['bi_exp']     = comp_exp

# Objects used in the experiment
full_record = {}
saturated_sizes = set()
config_format = "motzkin_{}_{}_{}_{}"
get_config = lambda m, bd, sz, sd: config_format.format(m, bd, sz, sd)

# Load the current experimental record if it exists
if os.path.exists(save_name):
    print("Loading previous experimental record from disk")
    full_record = pickle.load(open(save_name, 'rb'))

# Run the experiment many times and record best version for each config
for train_size in train_sizes:
    exp_args['train_size'] = train_size
    print(f"MOTZKIN ({exp_args['train_size']} train, "
                            f"{exp_args['other_size']} val)\n")

    for bond_dim, model in product(bond_dims, models):
        for seed in range(num_trials):
            print(f"{model.upper()}, BOND DIM {bond_dim}, SEED {seed}")
            exp_key = (train_size, bond_dim, model, seed)
            exp_args['model'] = model
            exp_args['bond_dim'] = bond_dim
            exp_args['rand_seed'] = seed
            exp_args['config_name'] = get_config(model, bond_dim, 
                                                 train_size, seed)

            # Apply default arguments
            exp_args = default_args(exp_args)

            # Run the experiment
            if exp_key not in full_record:
                record = run_experiment(exp_args)
            else:
                record = full_record[exp_key]

            # Record and print loss
            full_record[exp_key] = record
            best_loss = record['best_loss']
            samp_lens = exp_args['samp_lens']
            best_epoch = record['best_epoch']
            best_lrec = record['local_recs'][best_epoch]
            best_samp = best_lrec[f'corr_frac_{samp_lens[0]}']
            print(f"  Best val:  {best_loss:.3f}")
            print(f"  Best samp: {best_samp:.3f}")

        # Identify the best performing model among the different trials
        these_trials = {s: full_record[(train_size, bond_dim, model, s)] 
                        for s in range(num_trials)}
        best_seed, best_record = min(these_trials.items(), 
                                     key=lambda x: x[1]['best_loss'])
        best_epoch = best_record['best_epoch']
        best_loss = best_record['best_loss']
        best_lrec = best_record['local_recs'][best_epoch]
        best_samp = best_lrec[f'corr_frac_{samp_lens[0]}']
        print(f"BRACKETS, {model.upper()}, BOND DIM {bond_dim}")
        print(f"SEED:       {seed:.3f}")
        print(f"BEST VAL:   {best_loss:.3f}")
        print(f"BEST SAMP:  {best_samp:.3f}")
        print(f"BEST EPOCH: {best_epoch}")
        print()

        # Remove the non-optimal training runs to save memory
        for seed in range(num_trials):
            if seed == best_seed:
                full_record[(train_size, bond_dim, model)] = \
                    full_record[(train_size, bond_dim, model, seed)]
            # del full_record[(train_size, bond_dim, model, seed)]

        # Experimental record is stored as a dictionary indexed by triples 
        # (tom_num, bond_dim, model), with each value a complete record of 
        # that configuration, which is in turn a dictionary with detailed 
        # info in entry 'local_recs'
        pickle.dump(full_record, open(save_name, 'wb'))