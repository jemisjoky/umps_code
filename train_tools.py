import re
from collections import namedtuple
from functools import partial, reduce, lru_cache

import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import optimizers as jopt

import ti_mps

StrSet = namedtuple('StrSet', 'index_mat, str_lens')
StrSet.__doc__ = """\
Stores a batch matrix of encoded strings, along with their lengths
"""

def init_strset(str_list, alphabet=None):
    """
    Use a collection of strings or indices to initialize a StrSet object

    Args:
        str_list: List of strings or Numpy matrix of indices, which are 
                  stored in the output StrSet object
        alphabet: List containing the characters in our model's language,
                  only required when str_list is list of strings

    Returns:
        str_set:  The StrSet object containing an encoding of str_list
    """
    # Handle string inputs differently from matrix inputs
    if isinstance(str_list, list) and isinstance(str_list[0], str):
        # Check inputs and get str lengths
        assert alphabet is not None
        assert all(isinstance(s, str) for s in str_list)
        str_lens = jnp.array([len(s) for s in str_list])
        max_len = jnp.max(str_lens)

        # Get char-to-index lookup dictionary
        char_set = reduce((lambda ac, s: ac.union(set(s))), str_list, set())
        assert char_set.issubset(alphabet)
        char2ind = {c: str(i) for i, c in enumerate(alphabet)}

        # Pad strings to constant lengths
        pad_char = alphabet[0]
        str_list = [s + pad_char*(max_len-len(s)) for s in str_list]

        # Convert list of strings into list of index strings
        regex = re.compile(".")
        repl_fun = lambda m: f"{char2ind[m.group(0)]},"
        str_list = [regex.sub(repl_fun, s)[:-1] for s in str_list]

        # Convert index_str to JAX Numpy matrix
        str_vecs = [np.fromstring(s, dtype='int16', sep=',') for s in str_list]
        index_mat = jnp.stack(str_vecs)

    else:
        index_mat = jnp.array(str_list, dtype='int16')
        num_strs, max_len = index_mat.shape
        str_lens = jnp.full(num_strs, max_len)

    return StrSet(index_mat=index_mat, str_lens=str_lens)

def shuffle(str_set, rng_key):
    """
    Randomly shuffle the strings in a StrSet and return this new version

    Args:
        str_set:    StrSet containing the strings we want to shuffle
        rng_key:    JAX PRNG key to set the randomness used

    Returns:
        new_strset: StrSet containing the same data as the input, but with
                    everything randomly shuffled
    """
    # Unpack input
    index_mat = str_set.index_mat
    str_lens = str_set.str_lens
    max_len = index_mat.shape[1]
    batch_dim = len(index_mat)

    # Build random permutation vector, use it to jointly shuffle objects
    perm = jax.random.shuffle(rng_key, jnp.arange(batch_dim))
    index_mat = index_mat[perm]
    str_lens = str_lens[perm]

    return StrSet(index_mat=index_mat, str_lens=str_lens)

def minibatches(str_set, mini_size, keep_end=False):
    """
    Convert StrSet object into iterator over StrSet's of size `mini_size`

    Args:
        str_set:    StrSet object, which in practice will be a large one
                    holding an entire dataset
        mini_size:  Size of the minibatches we want to yield
        keep_end:   Whether to use the last part of our dataset when 
                    mini_size doesn't evenly divide the number of strings 
                    in str_set (Default: False)

    Returns:
        batch_iter: An iterator over minibatches of size mini_size, each
                    of which is itself a StrSet instance
    """
    index_mat, str_lens = str_set.index_mat, str_set.str_lens
    num_batches, tiny_batch = divmod(len(index_mat), mini_size)
    if keep_end and tiny_batch > 0:
        num_batches += 1

    for ind in range(num_batches):
        # Pull out part of index_mat and str_lens, truncate former so that
        # it doesn't contain unnecessary padding
        ind_mat = index_mat[ind*mini_size: (ind+1)*mini_size]
        s_lens = str_lens[ind*mini_size: (ind+1)*mini_size]
        m_len = jnp.max(s_lens)
        assert jnp.all(ind_mat[:, m_len:] == 0)
        ind_mat = ind_mat[:, :m_len]

        yield StrSet(index_mat=ind_mat, str_lens=s_lens)

def to_string(str_set, alphabet):
    """
    Convert StrSet object into a list of strings

    If str_set was created by init_strset, then using the same alphabet
    will produce the same list of strings that was used to create it

    Args:
        str_set:  StrSet object
        alphabet: List containing the characters used to define str_set

    Returns:
        str_list: A list of strings represented by str_set
    """
    index_mat, str_lens = str_set.index_mat, str_set.str_lens
    ind2char = {i: c for i, c in enumerate(alphabet)}

    # First create all the strings encoded by index_mat, then use str_lens
    # to truncate them to their proper length
    str_list = [''.join(ind2char[i] for i in row) for row in index_mat]
    str_list = [string[:s_len] for string, s_len in zip(str_list, str_lens)]

    return str_list

def mps_optimizer(jax_optimizer):
    """
    Wrap a JAX optimizer into an optimizer that acts on MPS models

    JAX optimizers are function triples (init_fun, update_fun, get_params),
    and mps_optimizer takes in one such triple and outputs another. The
    optimizer returned here is aware of the MPS structure, including the
    trainabilities of boundary operators, and accepts MPS objects (either
    TI-MPS or TTI-MPS) as the input to init_fun and the output of 
    get_params.

    The optimizer state in every case is a tuple, (mps, jax_state), where 
    jax_state is the state associated with the input jax optimizer

    Args:
        jax_optimizer: Arbitrary optimizer triple (as in jax.experimental)

    Returns:
        mps_optimizer: Optimizer triple wrapping the input, applying
                       updates to the relevant MPS parameters, and 
                       yielding the current MPS model
    """
    jax_init, jax_update, get_params = jax_optimizer

    def mps_init(mps):
        """Return an initial optimizer state"""
        params = trainable_params(mps)
        return (mps, jax_init(params))

    def mps_update(step, grad_mps, opt_state):
        """Update our MPS model using a gradient object"""
        # Unpack inputs
        old_mps, opt_state = opt_state
        grad = trainable_params(grad_mps)

        # Call Jax optimizer update on trainable parameters
        new_state = jax_update(step, grad, opt_state)

        # Update MPS params using new_params
        new_params = get_params(new_state)
        update_dict = {}
        update_dict['core_tensor'] = new_params[0]

        if trainable_bounds(old_mps):
            update_dict['bound_obj'] = new_params[1]
        
        return (old_mps._replace(**update_dict), new_state)

    def get_mps(opt_state):
        return opt_state[0]

    return mps_init, mps_update, get_mps

def schedule_maker(schedule_tuple, learn_rate):
    """
    Return a scheduler function given a tuple of the form:
        (sched_name, decay_steps, min_lr)

    This just wraps existing JAX schedulers, but using simplified syntax
    """
    sched_type = schedule_tuple[0]
    assert learn_rate >= 0
    assert sched_type in ['const', 'exp', 'poly', 'piecewise']

    if sched_type == 'const':
        # Constant learning rate
        sched_fun = jopt.constant(learn_rate)
    elif sched_type == 'exp':
        # Exponentially decaying learning rate
        sched_fun = jopt.exponential_decay(learn_rate, schedule_tuple[1], 
                                           0.5)
    elif sched_type == 'poly':
        # Harmonically decaying stepped learning rate
        sched_fun = jopt.inverse_time_decay(learn_rate, schedule_tuple[1], 
                                            5, staircase=True)
    elif sched_type == 'piecewise':
        # Piecewise constant learning rate, drops by factor of 10 each time
        step_len = schedule_tuple[1]
        assert step_len > 0
        bounds = [step_len * i for i in range(1, 10)]
        values = [learn_rate * 10**(-i) for i in range(10)]
        sched_fun = jopt.piecewise_constant(bounds, values)

    def my_sched_fun(epoch):
        lr = sched_fun(epoch)
        if len(schedule_tuple) <= 2:
            return lr
        else:
            return jnp.maximum(lr, schedule_tuple[2])

    return my_sched_fun

def trainable_bounds(mps):
    """Assesses if the boundary conditions for our MPS are trainable"""
    bound_cond = ti_mps.get_bound_cond(mps)
    assert bound_cond in ['white_noise', 'infinite', 'positive', 'open']

    if bound_cond in ['white_noise', 'infinite']:
        return False
    elif bound_cond in ['positive', 'open']:
        return True

def trainable_params(mps):
    """Extract the trainable parameters from our MPS"""
    first_param = mps.core_tensor

    if trainable_bounds(mps):
        return (first_param, mps.bound_obj)
    else:
        return (first_param,)
