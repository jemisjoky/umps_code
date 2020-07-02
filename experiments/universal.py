#!/usr/bin/env python3
import os
import sys
import time
from functools import partial
from types import SimpleNamespace
from string import ascii_lowercase

import jax
import jax.numpy as jnp
from comet_ml import Experiment
import torch

sys.path.append('..')
from train_tools import init_strset, shuffle, minibatches, to_string
from lstm_model import ProbLSTM, jax2tor
from sampler import fill_in_blanks

DEFAULTS = {'comet_log':  False,
            'group_name':  '',
            'config_name': '',

            # 'train_size': None,
            # 'str_len':     None,
            'min_len':     1,
            # 'bond_dim':    None,
            # 'learn_rate':  None,
            # 'batch_size':  None,
            # 'num_epochs':  None,
            'early_stop':  False,
            # 'input_dim':   None,
            'bound_cond':  'open',
            'init_method': 'eye',
            # 'optimizer':   None,
            'early_stop':  False,
            'save_record': False,
            # 'other_size':  None,
            # 'samp_len':    None,
            'rand_seed':   0,
            'noise':       1e-6,
            'dropout':     0,
            # 'pos_enc':     None,
            # 'pe_dim':      None,
            # 'samp_mode':   None,
            'verbose':     True,
            'fixed_dset':  True,
            }

ALLOWED =  {'bond_cond':   ['open'],
            'init_method': ['eye', 'rand'],
            'contract':    ['sequential', 'parallel'],
            'samp_mode':   ['variable', 'fixed', 'completion'],
            'bound_cond':  ['white_noise', 'infinite', 'positive', 'open'],
            'dataset':     ['brackets', 'tomita'],
            'tomita_num':  list(range(1, 8)),
            }

ALPHABET = {'brackets':    ['(', ')', '*'],
            'tomita':      ['0', '1'],
            'bos_eos':     ['^', '$'],
            }

def mps_sample_fun(rng_key, mps, score_fun, alph, ref_sets=None):
    """Draw samples from MPS model within JAX"""
    from sampler import draw_samples
    samp_lens = EXP_ARGS['samp_lens']
    samp_size = EXP_ARGS['samp_size']
    bi_exp    = EXP_ARGS['bi_exp']

    examp_samps = {}
    if bi_exp:
        corr_frac = {}
        for samp_l in samp_lens:
            ref_s = ref_sets[samp_l]
            ref_sets = to_string(ref_s, alph)
            rng_key, key = jax.random.split(rng_key)
            samp_chars = fill_in_blanks(key, mps, alphabet=alph, 
                                        ref_strset=ref_s)
            # TODO: Fold this code into fill_in_blanks
            # Generate validation strings with each character replaced by
            # suggested character from samp_chars
            samples = [s[:i] + c + s[i+1:] for s, cs in zip(ref_sets, samp_chars)
                        for i, c in enumerate(cs)]
            corr_frac[samp_l] = 100 * score_fun(samples)
            examp_samps[samp_l] = samples[:10]
            m_print(f"Correct frac len={samp_l}:  {corr_frac[samp_l]:.1f}%")
            m_print(f"Replacement examples: {samples[:10]}\n")
    else:
        corr_frac = {}
        for samp_l in samp_lens:
            rng_key, key = jax.random.split(rng_key)
            samples = draw_samples(key, mps, alphabet=alph, 
                           num_samps=samp_size, samp_len=samp_l)
            score = score_fun(samples)
            corr_frac[samp_l] = 100 * score
            examp_samps[samp_l] = samples[:10]
            m_print(f"Correct frac len={samp_l}:  {100 * score:.1f}%")
            m_print(f"Example samples: {samples[:10]}\n")

    return corr_frac, examp_samps

def mps_eval_fun(mps, dataset):
    """Evaluate MPS model within JAX over dataset"""
    from ti_mps import get_log_probs
    contract = EXP_ARGS['contract']
    batch_size = EXP_ARGS['batch_size']
    n, eval_loss, eval_ppl = 0, 0., 0.
    for batch in minibatches(dataset, batch_size, keep_end=True):
        eval_log_probs, log2_norms = get_log_probs(mps, batch, 
                                                   contract, True)
        eval_loss += -jnp.mean(eval_log_probs)
        eval_ppl += ppl_calc(eval_log_probs, batch.str_lens)
        n += 1
        # Get a log norm and rescale our core tensor using that
        ref_log = jnp.floor(jnp.median(log2_norms / batch.str_lens))
        if jnp.abs(ref_log) > 1:
            mps = mps._replace(core_tensor=(mps.core_tensor/2**ref_log))
            m_print(f"Rescaling by 2**{-ref_log}")
    eval_loss, eval_ppl = eval_loss / n, eval_ppl / n

    return mps, eval_loss, eval_ppl

def mps_train_fun(mps, dataset, epoch, optim_obj=None):
    """Train MPS model within JAX for 1 epoch"""
    from ti_mps import get_loss_grad
    contract = EXP_ARGS['contract']
    batch_size = EXP_ARGS['batch_size']
    opt_state, update_fun, get_model = optim_obj
    n, train_loss, train_ppl = 0, 0., 0.
    for batch in minibatches(dataset, batch_size, keep_end=True):
        grad_mps, train_log_probs, nll_loss = get_loss_grad(mps, batch, 
                                                    contraction=contract)
        opt_state = update_fun(epoch, grad_mps, opt_state)
        mps = get_model(opt_state)

        train_loss += nll_loss
        train_ppl += ppl_calc(train_log_probs, batch.str_lens)
        n += 1
    train_loss, train_ppl = train_loss / n, train_ppl / n

    return opt_state, train_loss, train_ppl

def lstm_sample_fun(rng_key, lstm, score_fun, alph, ref_sets=None):
    """Draw samples from LSTM model within Pytorch"""
    samp_lens = EXP_ARGS['samp_lens']
    samp_size = EXP_ARGS['samp_size']
    samp_mode = EXP_ARGS['samp_mode']
    bi_exp = lstm.bi_dir
    lstm = lstm.eval()
    examp_samps = {}
    if bi_exp:
        corr_frac = {}
        for samp_l in samp_lens:
            ref_s = ref_sets[samp_l]
            rng_key, key = jax.random.split(rng_key)

            # ref_strs = [s[1:-1] for s in to_string(ref_s, alph)]
            samples = lstm.sample(key, alph, 
                                samp_mode='completion', ref_strset=ref_s)

            # BOS and EOS should never be sampled
            # assert not any(('^' in s or '$' in s) for s in samples)
            samples = [')(' if ('^' in s or '$' in s) else s for s in samples]
            corr_frac[samp_l] = 100 * score_fun(samples)
            examp_samps[samp_l] = samples[:10]
            m_print(f"Correct frac len={samp_l}:  {corr_frac[samp_l]:.1f}%")
            m_print(f"Replacement examples:{examp_samps[samp_l]}\n")

    else:
        corr_frac = {}
        for samp_l in samp_lens:
            rng_key, key = jax.random.split(rng_key)
            samples = lstm.sample(key, alph, samp_mode=samp_mode,
                                     num_samps=samp_size, samp_len=samp_l)
            score = score_fun(samples)
            corr_frac[samp_l] = 100 * score
            examp_samps[samp_l] = samples[:10]
            m_print(f"Correct frac len={samp_l}:  {100 * score:.1f}%")
            m_print(f"Example samples: {examp_samps[samp_l]}\n")

    return corr_frac, examp_samps

def lstm_eval_fun(lstm, dataset):
    """Evaluate LSTM model within Pytorch over dataset"""
    batch_size = EXP_ARGS['batch_size']
    n, eval_loss, eval_ppl = 0, 0., 0.
    for batch in minibatches(dataset, batch_size, keep_end=True):
        with torch.no_grad():
            val_log_probs = lstm.get_loss(batch, rescale_probs=True)
        eval_loss += torch.mean(val_log_probs)
        eval_ppl += ppl_calc(val_log_probs, jax2tor(batch.str_lens))
        n += 1
    eval_loss, eval_ppl = eval_loss / n, eval_ppl / n

    return lstm, eval_loss.detach(), eval_ppl.detach()

def lstm_train_fun(lstm, dataset, epoch, optim_obj=None):
    """Train LSTM model within Pytorch for 1 epoch"""
    batch_size = EXP_ARGS['batch_size']
    n, train_loss, train_ppl = 0, 0., 0.
    lstm = lstm.train()
    for batch in minibatches(dataset, batch_size, keep_end=True):
        train_log_probs = lstm.get_loss(batch, rescale_probs=False)
        nll_loss = torch.mean(train_log_probs)
        
        train_loss += nll_loss
        train_ppl += ppl_calc(train_log_probs, jax2tor(batch.str_lens))
        n += 1

        optim_obj.zero_grad()
        nll_loss.backward()
        optim_obj.step()
    train_loss, train_ppl = train_loss / n, train_ppl / n

    return lstm, train_loss.detach(), train_ppl.detach()

def jax_reset_optim(new_lr, jaxro_input):
    """Reinitialize Jax optimizer with different learning rate"""
    # TODO: Clean up the treatment of optimizer parameters and schedulers
    lr_sched, optimizer, schedule_maker, mps_optimizer = jaxro_input
    opt_dict = {'step_size': schedule_maker(lr_sched, new_lr)}
    # if 'mass' in exp_args and exp_args['mass'] is not None:
    #     opt_dict['mass'] = l.mass
    optim = getattr(jax.experimental.optimizers, optimizer)(**opt_dict)
    return mps_optimizer(optim)

def torch_reset_optim(new_lr, lstm, optimizer):
    """Reinitialize Pytorch optimizer with different learning rate"""
    return getattr(torch.optim, optimizer)(lstm.parameters(), lr=new_lr)

def early_stop_fun(init_lr, warmup=5, patience=5, dec_factor=0.1, 
                            num_plats=3):
    """Build function which checks for early stopping condition"""
    loss_rec = []
    curr_lr = init_lr
    best_val, bad_eps = 1e10, 0
    plat_num = 1

    def stop_early(val_loss):
        nonlocal curr_lr, best_val, plat_num, bad_eps
        loss_rec.append(val_loss)
        if len(loss_rec) < warmup:
            return False, False, curr_lr
        drop_lr, e_stop = False, False

        # Compare validation loss to best for this plateau
        if val_loss < best_val:
            best_val, bad_eps = val_loss, 0
        else:
            bad_eps += 1

        # Drop the lr if we've had too many bad epochs
        if bad_eps >= patience:
            plat_num += 1
            drop_lr = True
            curr_lr *= dec_factor
            best_val, bad_eps = val_loss, 0
            e_stop = plat_num > num_plats

        return drop_lr, e_stop, curr_lr

    return stop_early

def run_experiment(exp_args):
    """
    Run an experiment parameterized by a dictionary of arguments

    Args:
        exp_args: Dictionary containing experimental settings, including:

          comet_log:   Whether or not to log data to Comet.ml
          group_name:  Name of our general group of experiments
          config_name: Name of this particular parameter configuration
          num_epochs:  Number of epochs to train for

          model:       String specifying the model used for experiment
          input_dim:   Input dimension of our model
          bond_dim:    Bond dim / hidden state dim of learning model

          dataset:     String specifying the dataset used for experiment
          train_size:  Number of strings to use in training dataset
          other_size:  Number of strings to use in val and test datasets
          max_len:     Maximum length of training, val, and test strings
          min_len:     Minimum length of training, val, and test strings
          use_val:     Whether to use validation dataset
          use_test:    Whether to use test dataset
          
          bi_exp:      Whether to use a bidirectional model
          samp_size:   Number of strings to generate in sampling
          samp_lens:   List of lengths for samples
          bos_eos:     Whether to enclose strings with BOS and EOS chars

          optimizer:   Optimizer used to update model weights
          learn_rate:  Learning rate of experimental optimizer
          lr_sched:    Specificiation of learning rate scheduler
          batch_size:  Size of minibatches used in training
          early_stop:  Whether to decrease lr on plateau and stop early
          save_record: Whether to save the full experimental record
          
          verbose:     Whether to print status info during training
          rand_seed:   Random seed used to set model randomness
          fixed_dset:  Whether to make dataset independent of seed

          ### MPS ONLY ###
          init_method: Method used to initialize model weights
          bound_cond:  Boundary condition for the model
          noise:       Initial noise used for near-identity init
          contract:    Type of contraction used to evaluate MPS
          mass:        Hyperparameter for some Jax optimizers

          ### LSTM ONLY ###
          dropout:     Amount of dropout to use
          pos_enc:     Whether to add positional encoding to inputs
          pe_dim:      Dimension of positional encoder
          samp_mode:   String specifying the type of sampling being used

          ### TOMITA ONLY ###
          tomita_num:  Which Tomita grammar we wish to use

    """
    start_time = time.time()
    
    # Initialize unspecified entries of exp_args with defaults
    for key in set(DEFAULTS) - set(exp_args):
        exp_args[key] = DEFAULTS[key]

    # Check that input parameters are allowed
    for key in set(ALLOWED) & set(exp_args):
        assert exp_args[key] in ALLOWED[key]

    # Load parameters in exp_args as variables for ease of reference
    assert set(locals()) & set(exp_args) == set()
    l = SimpleNamespace(**exp_args)

    # Misc initialization
    m_print = partial(print, flush=True) if l.verbose else lambda s: None
    stop_early = early_stop_fun(l.learn_rate)
    rng_key = jax.random.PRNGKey(l.rand_seed)
    globals()['EXP_ARGS'] = exp_args
    globals()['m_print'] = m_print

    # Get alphabet and all datasets involved in experiment
    alph = ALPHABET[l.dataset]
    if l.bos_eos: alph = alph + ALPHABET['bos_eos']
    data_split = [l.train_size]
    if l.use_val: data_split.append(l.other_size)
    if l.use_test: data_split.append(l.other_size)
    if l.fixed_dset:
        key = jax.random.PRNGKey(-1)
    else:
        key, rng_key = jax.random.split(rng_key)
    if l.dataset == 'brackets':
        from toy_datasets import brackets_dataset
        from toy_datasets import score_brackets as score_fun
        all_data = brackets_dataset(rng_key=key, data_split=data_split, 
                                    max_len=l.max_len, min_len=l.min_len,
                                    add_ends=l.bos_eos)
        # Build up reference strings for fill-in experiment (bi-dir only)
        if l.bi_exp:
            ref_sets = {}
            for samp_l in l.samp_lens:
                rng_key, key = jax.random.split(rng_key)
                try:
                    ref_s = brackets_dataset(rng_key=key, 
                                             data_split=l.samp_size,
                                             max_len=samp_l, 
                                             min_len=samp_l,
                                             add_ends=l.bos_eos)
                except:
                    # If there aren't enough strings, then repeat all 
                    # strings several times
                    ref_s = brackets_dataset(rng_key=key, 
                                             data_split=0.9999999,
                                             max_len=samp_l, 
                                             min_len=samp_l,
                                             add_ends=l.bos_eos)
                    ref_s *= l.samp_size // len(ref_s)
                # if l.bos_eos: ref_s = [f"^{s}$" for s in ref_s]
                ref_sets[samp_l] = init_strset(ref_s, alph)
                # TODO: Fix the following line, this is hacky
                if samp_l == l.max_len: ref_sets[samp_l] = val_set
        else:
            ref_sets = None

    elif l.dataset == 'tomita':
        from toy_datasets import tomita_dataset
        from toy_datasets import score_tomita as score_fun
        score_fun = partial(score_fun, tomita_num=l.tomita_num)
        all_data = tomita_dataset(rng_key=key, data_split=data_split,
                                  max_len=l.max_len, tomita_num=l.tomita_num, 
                                  add_ends=l.bos_eos)

    # Convert datasets to StrSet objects
    all_data = [init_strset(ds, alph) for ds in all_data]
    if l.use_test: test_set = all_data.pop()
    if l.use_val: val_set = all_data.pop()
    train_set = all_data.pop()
    assert len(all_data) == 0

    # Model specific-initialization, including optimizers
    if l.model == 'mps':
        import ti_mps as umps
        from utils import ppl_calc
        from train_tools import mps_optimizer, schedule_maker
        from sampler import draw_samples, fill_in_blanks
        from train_tools import mps_optimizer, schedule_maker
        
        key, rng_key = jax.random.split(rng_key)
        my_model = umps.init_ti_mps(l.bond_dim, l.input_dim, 
                                    bound_cond=l.bound_cond, 
                                    init_method=l.init_method, 
                                    rng_key=key, noise=l.noise)
        train_fun = mps_train_fun
        samp_fun = mps_sample_fun
        eval_fun = mps_eval_fun
        schedule = schedule_maker(l.lr_sched, l.learn_rate)
        opt_dict = {'step_size': schedule}
        if 'mass' in exp_args and exp_args['mass'] is not None:
            opt_dict['mass'] = l.mass
        jaxro_input = l.lr_sched, l.optimizer, schedule_maker, mps_optimizer
        optim = getattr(jax.experimental.optimizers, l.optimizer)(**opt_dict)
        init_opt, update_fun, get_model = mps_optimizer(optim)
        opt_state = init_opt(my_model)

    elif l.model == 'lstm':
        from lstm_model import ProbLSTM

        torch.manual_seed(l.rand_seed)
        ppl_calc = lambda log_p, seq_l: torch.exp(-torch.mean(log_p /
                                                            seq_l.float()))
        my_model = ProbLSTM(l.input_dim, l.bond_dim, num_layers=1, 
                            bi_dir=l.bi_exp, dropout=l.dropout, 
                            pos_enc=l.pos_enc, pe_dim=l.pe_dim)
        train_fun = lstm_train_fun
        samp_fun = lstm_sample_fun
        eval_fun = lstm_eval_fun
        optim = getattr(torch.optim, l.optimizer)(my_model.parameters(), 
                                                lr=l.learn_rate)
    globals()['ppl_calc'] = ppl_calc

    # Setup logging and print initialization
    if l.comet_log:
        experiment = Experiment(project_name=l.group_name)
        experiment.log_parameters(exp_args)
        experiment.set_name(l.config_name)
    global_rec = {'best_loss':  1e10,   # Best validation loss
                  'best_model': None,
                  'best_epoch':   -1,
                  'local_recs':   {},
                  'plat_epochs': [1],}
    init_time = time.time() - start_time
    global_rec['init_time'] = init_time
    m_print(f"Experiment:     {l.config_name} ({l.group_name})")
    m_print(f"Model:          {l.model}")
    m_print(f"Dataset:        {l.dataset}")
    m_print(f"Hidden dim:     {l.bond_dim}")
    m_print(f"Train set size: {l.train_size}")
    m_print(f"Val/test size:  {l.other_size}")
    m_print(f"Maximum length: {l.max_len}")
    m_print(f"Minimum length: {l.min_len}")
    m_print(f"Total epochs:   {l.num_epochs}")
    m_print(f"Num of samples: {l.samp_size}")
    m_print(f"Sample lengths: {l.samp_lens}")
    m_print(f"Learning rate:  {l.learn_rate}")
    m_print(f"Schedule specs: {l.lr_sched}")
    m_print(f"Optimizer:      {l.optimizer}")
    m_print(f"Batch size:     {l.batch_size}")
    m_print(f"Use BOS/EOS:    {l.bos_eos}")
    m_print(f"Init. Time:     {init_time:.2} sec\n")

    # Train and validate for all the epochs, ending via early stopping
    for epoch in range(1, l.num_epochs+1):
        local_rec = {'epoch': epoch}
        epoch_start = time.time()

        # Shuffle the training set
        key, rng_key = jax.random.split(rng_key)
        train_set = shuffle(train_set, key)

        # Draw samples from our model
        samp_start = time.time()
        corr_frac, examp_samps = samp_fun(key, my_model, score_fun, alph)
        samp_time = time.time() - samp_start
        for samp_l in corr_frac:
            local_rec[f'corr_frac_{samp_l}'] = corr_frac[samp_l]
        local_rec['samp_time'] = samp_time
        local_rec['samp_time'] = samp_time
        # local_rec['examp_samps'] = examp_samps

        # Evaluate our model on the validation set
        if l.use_val:
            eval_start = time.time()
            my_model, val_loss, val_ppl = eval_fun(my_model, val_set)
            eval_time = time.time() - eval_start
            if val_loss < global_rec['best_loss'] or epoch == 1:
                global_rec['best_model'] = my_model
                global_rec['best_epoch'] = epoch
                global_rec['best_loss'] = val_loss
                if epoch == 1: global_rec['init_loss'] = val_loss
        else:
             val_loss, val_ppl, eval_time = None, None, 0.
        local_rec['eval_time'] = eval_time
        local_rec['val_loss'] = float(val_loss)
        local_rec['val_ppl'] = float(val_ppl)

        # Check to see if early stopping condition is met
        if l.early_stop and l.use_val:
            drop_lr, e_stop, new_lr = stop_early(val_loss)
            if e_stop:
                m_print(f"### Stopping Early, at Epoch {epoch} ###\n")
                break
            else:
                if drop_lr: m_print(f"### Decreasing LR to {new_lr} ###\n")
        else:
            drop_lr = False

        # Train over all minibatches of training data
        train_start = time.time()
        if l.model == 'mps':
            if drop_lr:
                init_opt, update_fun, get_model = jax_reset_optim(new_lr, 
                                                            jaxro_input)
                opt_state = init_opt(my_model)
            optim_obj = opt_state, update_fun, get_model
            opt_state, train_loss, train_ppl = train_fun(my_model, 
                                                      train_set, epoch, 
                                                      optim_obj=optim_obj)
            my_model = get_model(opt_state)
        elif l.model == 'lstm':
            if drop_lr:
                optim = torch_reset_optim(new_lr, my_model, l.optimizer)
            my_model, train_loss, train_ppl = train_fun(my_model, 
                                                        train_set, epoch, 
                                                        optim_obj=optim)
        train_time = time.time() - train_start
        local_rec['train_time'] = train_time
        local_rec['train_loss'] = float(train_loss)
        local_rec['train_ppl'] = float(train_ppl)

        # Log and print data
        epoch_time = time.time() - epoch_start
        local_rec['epoch_time'] = epoch_time
        global_rec['local_recs'][epoch] = local_rec
        m_print(f"# Epoch:    {epoch}")
        m_print(f"Train Loss: {train_loss:.5f}")
        m_print(f"Val Loss:   {val_loss:.5f}")
        m_print(f"Train PPL:  {train_ppl:.5f}")
        m_print(f"Val PPL:    {val_ppl:.5f}")
        m_print(f"Epoch time: {int(epoch_time)} sec")
        m_print(f"Total time: {int(time.time() - start_time)} sec\n")
        if l.comet_log: experiment.log_metrics(local_rec, step=epoch)

    # Evaluate on test dataset
    if l.use_test:
        my_model = global_rec['best_model']
        _, test_loss, test_ppl = eval_fun(my_model, test_set)
        test_loss, test_ppl = float(test_loss), float(test_ppl)
    else:
        test_loss, test_ppl = None, None
    global_rec['test_loss'] = test_loss
    global_rec['test_ppl'] = test_ppl
    global_rec['total_time'] = time.time() - start_time

    # Do final logging and printing
    if l.save_record:
        import pickle
        save_name = f".{l.group_name}-{l.config_name}.record"
        with open(save_name, 'wb') as f:
            pickle.dump(global_rec, f)
    if l.comet_log:
        gr_copy = {k: v for (k, v) in global_rec.items() 
                            if k not in ['local_recs', 'best_model']}
        experiment.log_metrics(gr_copy)
    m_print(f"Total Epochs:  {epoch}")
    m_print(f"Best epoch:    {global_rec['best_epoch']}")
    m_print(f"Best val loss: {global_rec['best_loss']:.3f}\n")

    return global_rec
