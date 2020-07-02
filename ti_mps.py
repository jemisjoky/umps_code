from functools import partial, lru_cache
from collections import namedtuple

import jax
import numpy as onp
import jax.numpy as np

import utils
from spectral_tools import build_t_op, transfer_eigs_power
from parallel_ops import parallel_contract

TI_MPS = namedtuple('TI_MPS', 'core_tensor, bound_obj, state')
TI_MPS.__doc__ = """\
Translationally-invariant MPS model defining family of probability 
distributions over sequences via the Born rule

The model is parameterized by a single core tensor, a choice of boundary
conditions, and a state dictionary with other parameters
"""

def init_ti_mps(bond_dim, input_dim, bound_cond='positive', 
                init_method='eye', rng_key=None, **kwargs):
    """
    Initialize a TI-MPS for language modeling

    Args:
        bond_dim:    The bond dimension of our TI-MPS model, which is the
                     main parameter that sets its capacity
        input_dim:   The size of the alphabet the model's core tensor 
                     accepts as input
    Optional:
        bound_cond:  The choice of boundary conditions for our model, with
                     'white_noise', 'infinite', 'positive', and 'open' as 
                     allowed options. Each case places positive operators 
                     at the left and right boundaries, with 'white_noise' 
                     using fixed identity matrices, 'infinite' using
                     fixed-points of the MPS transfer operator, 'positive'
                     using arbitrary trainable positive matrices, and 
                     'open' using rank-1 trainable positive matrices. Both
                     'positive' and 'open' initialize with random matrices.
                     (Default: 'positive')
        init_method: The manner in which we initialize our MPS core.
                     Current options are:
                     'eye'   -> Initialize near the identity
                     'rand'  -> Initialize to have random normal entries
                     'given' -> Initialize with user-specified core_tensor
                     (Default: 'eye')
        rng_key:     Jax PRNGKey to control randomness in initialization
        noise:       Only used in 'eye' initialization, this specifies the
                     amount of noise we add to initial identity matrices
                     (Default: 1e-6)
        core_tensor: If init_method is set to 'given', core_tensor must be
                     a tensor of shape (bond_dim, bond_dim, input_dim)
                     which will be used as the model initialization

    Returns:
        ti_mps:      The initialized TI-MPS model
    """
    # Check and grab inputs
    assert init_method in ['eye', 'rand', 'given']
    assert bound_cond in ['white_noise', 'infinite', 'positive', 'open']
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    noise = kwargs['noise'] if 'noise' in kwargs else 1e-6
    core_tensor = kwargs['core_tensor'] if 'core_tensor' in kwargs \
                  else None

    # Initialize the core tensor if it isn't given
    if init_method is not 'given':
        # Initialize a Gaussian random tensor to use for initialization
        rng_key, key = jax.random.split(rng_key)
        rand_t = jax.random.normal(key, (bond_dim, bond_dim, input_dim))

        if init_method == 'eye':
            # Rescale random tensor and add an identity tensor
            core_tensor = noise * rand_t + utils.eye_tensor(bond_dim, 
                                                            input_dim)
        elif init_method == 'rand':
            core_tensor = rand_t / np.sqrt(bond_dim)

    # Initialize the boundary object
    rng_key, key = jax.random.split(rng_key)
    if bound_cond == 'positive':
        bound_obj = jax.random.normal(key, (2, bond_dim, bond_dim))
        bound_obj = bound_obj / np.sqrt(bond_dim)
    elif bound_cond == 'open':
        bound_obj = jax.random.normal(key, (2, bond_dim))
    # For non-trainable boundaries, bound_obj.shape specifies bound_cond 
    elif bound_cond == 'white_noise':
        bound_obj = np.empty(())
    elif bound_cond == 'infinite':
        bound_obj = np.empty((0,))

    # The state dictionary, which holds cached information about the 
    # spectrum of the MPS transfer operator
    state = {}
    state['eig_mats'] = None
    state['eig_val'] = None

    # Create MPS from initialization data
    my_mps = TI_MPS(core_tensor, bound_obj, state)

    # If needed, calculate fixed points of left/right transfer operators
    if bound_cond == 'infinite':
        my_mps = update_boundaries(my_mps)

    return my_mps

def get_log_norms(ti_mps, str_set):
    """
    Compute log normalization for sampling distribution on input strings

    Args:
        ti_mps:    TI_MPS whose log norms we are computing
        str_set:   StrSet object which determines the lengths at which we
                   compute the partition function for. Alternately, a 
                   number or vector can be used, setting lengths directly

    Returns:
        log_norms: Logarithms of the *exact* normalization factors for the
                   sampling distribution, one at the length of each string
                   contained in str_set
    """
    # If our object isn't a StrSet, make it one
    if hasattr(str_set, 'str_lens'):
        scalar_input = False
    else:
        if not hasattr(str_set, '__len__'):
            str_lens, scalar_input = [str_set], True
        else:
            str_lens, scalar_input = str_set, False

        # Initialize a StrSet with lengths given by str_lens
        from train_tools import init_strset
        str_set = init_strset(['a' * l for l in str_lens], ['a'])

    # Call our implementation
    log_norms = _get_log_norms(ti_mps, str_set)

    if scalar_input:
        assert len(log_norms) == 1
        return log_norms[0]
    else:
        return log_norms

@jax.jit
def _get_log_norms(ti_mps, str_set):
    """
    Implementation of get_log_norms which doesn't accept keyword inputs
    """
    core_tensor, boundaries = ti_mps.core_tensor, get_bound_mats(ti_mps)
    str_lens, max_len = str_set.str_lens, str_set.index_mat.shape[1]
    
    # Unpack boundary matrices and define rightward transfer operator
    left_mat, right_mat = boundaries
    t_op = build_t_op(core_tensor, direction='right', jitted=True)

    # Function implementing a single step of computing log_norm
    def scan_iter(iter_state, _):
        # scan_iter: c -> a -> (c, b)
        # iter_state holds a running log_norm and a density operator
        old_norm, old_density = iter_state

        # Apply transfer_op and normalize output
        new_density = t_op(old_density)
        this_norm = np.linalg.norm(new_density)
        new_density = new_density / this_norm

        new_norm = old_norm + np.log(this_norm)
        out_norm = new_norm + np.log(utils.hs_dot(right_mat, new_density))

        return (new_norm, new_density), out_norm

    # Apply transfer operator many times (starting with left_mat), and 
    # generate list of log_norms
    iter_init = (0., left_mat)
    _, proper_log_norms = jax.lax.scan(scan_iter, iter_init, 
                                       np.empty(max_len))
    assert len(proper_log_norms) == max_len

    # Add norm for the empty distribution and pick out lengths of interest
    empty_log_norm = np.log(utils.hs_dot(left_mat, right_mat))[None]
    all_log_norms = np.concatenate((empty_log_norm, proper_log_norms))
    log_norms = all_log_norms[str_lens]

    return log_norms

def get_log_probs(ti_mps, str_set, contraction='sequential', log_logs=False):
    """
    Compute the log of probabilities of inputs within an MPS distribution

    Args:
        ti_mps:      TI_MPS whose core tensor and boundary matrices 
                     encode a probability distribution
        str_set:     Encoded batch of input sentences whose log_probs 
                     we are calculating
        contraction: Sets the contraction algorithm used to perform the 
                     contraction of core tensors. Allowed options are 
                     'parallel' and 'sequential'. The former is more
                     expensive overall, but parallelizes much better than
                     'sequential', and should be preferred in the presence
                     of hardware accelerators, such as GPU's.
        log_logs:    Whether to return log2 norms for the input sequences

    Returns:
        log_probs:   Vector of properly normalized log probabilities for 
                     each sentence in our input
        log2_norms:  The rounded log2 norms of the input sequences
    """
    # Sets whether we use parallel or sequential evaluation
    assert contraction in ['sequential', 'parallel']

    log_probs, log2_norms = _get_log_probs(ti_mps, str_set, contraction)

    if log_logs:
        return log_probs, log2_norms
    else:
        return log_probs

@partial(jax.jit, static_argnums=(2,))
def _get_log_probs(ti_mps, str_set, contraction):
    """
    Implementation of get_log_probs which doesn't accept positional arguments
    """

    # Get log probabilities through one of these contraction methods
    if contraction == 'parallel':
        # Unpack inputs
        core_tensor, bound_mats = ti_mps.core_tensor, get_bound_mats(ti_mps)
        str_lens, max_len = str_set.str_lens, str_set.index_mat.shape[1]
        left_mat, right_mat = bound_mats
        bd = core_tensor.shape[0]

        # Get batch of core tensor slices, which form transition mats
        batch_mats = feed_timps(ti_mps, str_set)

        # Contract all sequences of transition mats in parallel
        prod_mats, log2_norms = parallel_contract(batch_mats, True)

        # Use transfer operator eigenmatrices as boundary conditions to get
        # unnormalized log probs for our input sequences. The ein_sum
        # string gives the trace of the left/right boundary mats with the 
        # product matrix and its adjoint, parallelized over the batch index
        log_probs = np.log(np.einsum('tu,buv,vw,btw->b', left_mat, 
                                                        prod_mats, 
                                                       right_mat, 
                                                      np.conj(prod_mats)))
        log_probs = log_probs + 2 * np.log(2)*log2_norms
    else:
        # Get prob amplitudes and use Born rule to convert into log probs
        prob_amps, log2_norms = sequential_eval(ti_mps, str_set, True)
        log_probs = np.log(np.abs(prob_amps)**2) + 2*np.log(2)*log2_norms

    # Subtracting log_norms ensures our log probs are properly normalized
    log_probs = log_probs - get_log_norms(ti_mps, str_set)

    return log_probs, log2_norms

@partial(jax.jit, static_argnums=(2,))
def sequential_eval(ti_mps, str_set, give_lognorm=False):
    """
    Get TI-MPS probability amplitude from StrSet via sequential evaluation

    Args:
        ti_mps:       TI-MPS used to obtain a probability amplitude, whose
                      boundary conditions must be open
        str_set:      StrSet specifying a batch of input strings
        give_lognorm: If true, the log normalization of the output values 
                      will be given as a second output, an option which 
                      avoids overflow.
    
    Returns:
        prod_values:  Vector giving all the probability amplitudes
        log2_norms:   Power of 2 by which values of prod_values were reduced
    """
    # Sequential evaluations requires open boundary conditions to work
    bound_cond = get_bound_cond(ti_mps)
    if bound_cond != 'open':
        raise ValueError("Sequential evaluation requires open boundary"
                f" conditions, but model has '{bound_cond}' boundaries")
    
    # Unpack inputs
    core_tensor, bound_vecs = ti_mps.core_tensor, ti_mps.bound_obj
    str_lens, index_mat = str_set.str_lens, str_set.index_mat
    batch_size, length = index_mat.shape
    bd, in_dim = core_tensor.shape[1:3]
    eye_mat = np.eye(bd)

    # Expand non-batched bound_vecs to batched ones
    exp_vecs = np.broadcast_to(bound_vecs[:,None,:], (2, batch_size, bd))
    left_vecs, right_vecs = exp_vecs

    # Convert all padded indices to have values of -1
    padded = np.arange(length)[None] >= str_lens[:, None]
    index_mat = np.where(padded, -1, index_mat)

    # Put spatial axis of index_mat first, for the scan operation later
    index_mat = index_mat.T

    # Function to evolve hidden vecs via vec-mat multiplication
    @jax.jit
    def inner_loop(h_vecs, inds):    # c -> a -> (c, b)
        # Use inds and core_tensor to get transition matrices
        naive_mats = core_tensor[:, :, inds]
        mats = np.where(inds < 0, eye_mat[:, :, None], naive_mats)
        h_vecs = np.einsum('bl,lrb->br', h_vecs, mats)

        # Divide v by a power of 2 to get its norm close to 1
        norms = np.linalg.norm(h_vecs, axis=1, keepdims=True)
        two_pows = np.floor(np.log2(norms))
        h_vecs = h_vecs / (2 ** two_pows)
        return h_vecs, two_pows

    out_vecs, log2_norms = jax.lax.scan(inner_loop, left_vecs, index_mat)
    log2_norms = np.sum(log2_norms, axis=0)
    prod_values = np.einsum('bv,bv->b', out_vecs, right_vecs)

    # Handle the log norms according to value of give_lognorm
    if give_lognorm:
        return prod_values, np.squeeze(log2_norms, axis=1)
    else:
        return v * 2**log2_norms

def get_loss_grad(ti_mps, str_set, probs_and_loss=True, 
                  loss_fun=(lambda lp: -np.mean(lp)), 
                  contraction='sequential'):
    """
    Use TI-MPS and input data to get gradients relative to a loss function

    The returned gradient is a TI-MPS object whose entries are the 
    gradients with respect to the corresponding tensors, along with the 
    log probs and loss for this batch of inputs
    
    Args:
        ti_mps:         TI_MPS whose probability distribution we are 
                        using to obtain the core tensor gradient
        str_set:        Encoded batch of input sentences we are using to 
                        compute the loss gradients
        probs_and_loss: If True, a batch of log probs and the overall loss
                        will be given as additional outputs
        loss_fun:       Used to specify custom loss functions, which are
                        required to be JAX functions from the *log* 
                        probabilities (base e) to loss. 
                        (Default: negative log likelihood loss)

    Returns:
        grad_mps:       TI_MPS object whose entries are all gradients
                        with respect to loss_fun for our input str_set
        log_probs:      Returned when probs_and_loss is True
        loss:           Returned when probs_and_loss is True
    """
    # Get our loss, prob, gradient function
    assert contraction in ['sequential', 'parallel']
    lpg_fun = build_lpg_fun(loss_fun, contraction=contraction)

    # Call this function on our inputs
    (loss, log_probs), grad_mps = lpg_fun(ti_mps, str_set)

    # Output auxiliary data when probs_and_loss is True
    if probs_and_loss:
        return grad_mps, log_probs, loss
    else:
        return grad_mps

@lru_cache()
def build_lpg_fun(loss_fun, jitted=True, contraction='sequential'):
    """
    Build a function which computes the loss, log_probs, and gradients
    for a TI-MPS model
    """
    # Our forward function which gets input str_set and returns the value 
    # of loss_fun on that input, along with the log probabilities
    def lp_fun(ti_mps, str_set):
        log_probs = get_log_probs(ti_mps, str_set, contraction, False)
        loss = loss_fun(log_probs)
        return loss, log_probs

    lpg_fun = jax.value_and_grad(lp_fun, has_aux=True)
    lpg_fun = jax.jit(lpg_fun) if jitted else lpg_fun

    return lpg_fun

@jax.jit
def feed_timps(ti_mps, str_set):
    """
    Use a batch of strings to build a batch of closed matrix images

    Args:
        ti_mps:     TI-MPS object
        str_set:    Batch of strings stored in StrSet object. When str_set
                    contains strings of different lengths, identity mats
                    are used for padding to maintain batched form.

    Returns:
        batch_mats: A doubly batched collection of matrices. The first 
                    index maps over the strings in str_set, while the 
                    second maps over the characters in each string.
    """
    # Unpack inputs
    core_tensor, index_mat = ti_mps.core_tensor, str_set.index_mat
    str_lens, max_len = str_set.str_lens, index_mat.shape[1]
    bond_dim, input_dim = core_tensor.shape[1:3]
    eye_mat = np.eye(bond_dim)

    # Feed indices into core_tensor, then pad with eye_mat's
    pad_cond = np.arange(max_len)[None] < str_lens[:, None]
    naive_batch_mats = np.einsum('ijbs->bsij', core_tensor[:, :, index_mat])
    batch_mats = np.where(pad_cond[..., None, None], 
                          naive_batch_mats, eye_mat)
    return batch_mats

def get_bound_cond(ti_mps):
    """
    Gets the boundary condition defining our TI-MPS

    Args:
        ti_mps:     TI-MPS language model

    Returns:
        bound_cond: A string from one of the following options:
                    'positive'    -> Trainable positive mats at edges
                    'open'        -> Trainable rank-1 mats at edges
                    'white_noise' -> Identity matrices at edges
                    'infinite'    -> Transfer op fixed-points at edges
    """
    shape_len = len(ti_mps.bound_obj.shape)
    if shape_len == 3:
        return 'positive'
    elif shape_len == 2:
        return 'open'
    elif shape_len == 1:
        return 'infinite'
    elif shape_len == 0:
        return 'white_noise'
    
def get_bound_mats(ti_mps):
    """
    Get the pair of matrices defining our MPS's boundary conditions

    Args:
        ti_mps:     Our TI-MPS model

    Returns:
        boundaries: A pair of matrices holding the MPS boundary conditions
    """
    bound_cond, bound_obj = get_bound_cond(ti_mps), ti_mps.bound_obj
    bond_dim = ti_mps.core_tensor.shape[0]

    # Use boundary-dependent mapping to convert bound_obj to positive mats
    if bound_cond == 'white_noise':
        return np.broadcast_to(np.eye(bond_dim), (2, bond_dim, bond_dim))
    elif bound_cond == 'infinite':
        if state['eig_mats'] is None:
            ti_mps = update_boundaries(ti_mps)
        return state['eig_mats']
    elif bound_cond == 'positive':
        return np.einsum('bik,bjk->bij', bound_obj, np.conj(bound_obj))
    elif bound_cond == 'open':
        return np.einsum('bi,bj->bij', bound_obj, np.conj(bound_obj))
    
def update_boundaries(ti_mps, new_ppl=None):
    """
    Calculates the dominant eigenvalue and spectral gap of the MPS model's
    transfer operator, and populates the matching attributes of the TI_MPS

    By feeding in new_ppl, this method also has the option of rescaling the
    norm of the core tensor to maintain numerical stability

    Args:
        ti_mps:         A TI-MPS model whose spectral properties we want
        new_ppl:        If set, the core tensor will be rescaled so that
                        the transfer operator's spectral norm is new_ppl

    Returns:
        updated_ti_mps: The same TI-MPS with up-to-date eigenmatrices and
                        core_tensor
    """
    # Unpack attributes from TI_MPS
    core_tensor, state = ti_mps.core_tensor, ti_mps.state
    old_eig_mats, old_eig_val = state['eig_mats'], state['eig_val']
    bd = core_tensor.shape[0]

    # Get initial guess values to feed into eigensolver
    if old_eig_mats is None:
        guess_mats = np.broadcast_to(np.eye(bd), (2, bd, bd))
    else:
        guess_mats = old_eig_mats

    # Call eigensolver
    eig_val, eig_mats = transfer_eigs_power(core_tensor, guess_mats)

    # Rescale core_tensor
    if new_ppl is not None:
        scale_factor = np.sqrt(new_ppl / eig_val)
        core_tensor = scale_factor * core_tensor

    # Update our TI_MPS info
    state['eig_mats'], state['eig_val'] = eig_mats, eig_val

    return ti_mps._replace(core_tensor=core_tensor, state=state)
