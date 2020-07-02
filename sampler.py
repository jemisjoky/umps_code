import string

import jax
import numpy as onp
import jax.numpy as np

from ti_mps import get_bound_mats
from spectral_tools import build_t_op

def draw_samples(rng_key, ti_mps, alphabet, num_samps=5, samp_len=10):
    """
    Use a TI-MPS model to sample fixed-length sequences from an alphabet

    Args:
        rng_key:   Random key used within Jax's PNRG
        ti_mps:    The TI-MPS model defining our sampling distribution
        alphabet:  The alphabet defining a basis in our sampling space
        num_samps: Number of samples to draw
        samp_len:  Length of all of the output samples

    Returns:
        samp_list: List of output sampled strings
    """
    # Sampling empty strings is easy
    assert not hasattr(samp_len, '__len__')
    if samp_len == 0:
        return [""] * num_samps

    # Unpack input args
    core_tensor = ti_mps.core_tensor
    in_dim = core_tensor.shape[2]
    assert len(alphabet) == in_dim
    left_b, right_b = get_bound_mats(ti_mps)

    # Calculate right boundaries for all our samples
    t_op = build_t_op(core_tensor, direction='left')
    rd_list = [right_b]     # 'rd' := 'reduced density'
    for _ in range(samp_len - 1):
        new_rd = t_op(rd_list[-1])
        rd_list.append(new_rd / np.linalg.norm(new_rd))
    assert len(rd_list) == samp_len
    rd_list = rd_list[::-1]

    # Generate random numbers for use in sampling
    rand_fvecs = jax.random.uniform(rng_key, shape=(samp_len, num_samps))

    # Use _sampler_inner_loop to get all samples in numerical form
    lds = np.broadcast_to(left_b, shape=(num_samps,)+left_b.shape)
    all_samps = []
    for rf, rd in zip(rand_fvecs, rd_list):
        lds, samp_vec = _sampler_inner_loop(lds, rf, core_tensor, rd)
        all_samps.append(samp_vec)
    all_samps = np.stack(all_samps).T   # Make batch dim into lead index

    # Convert numerical representation into actual strings
    samp_list = ["".join([alphabet[c] for c in cs]) for cs in all_samps]
    assert len(samp_list) == num_samps

    return samp_list

@jax.jit
def _sampler_inner_loop(l_densities, rand_floats, core_tensor, r_density):
    """Take in left density op, output sample and evolve density op"""

    # Get unnormalized probabilities and normalize
    probs = np.einsum('Bab,bdi,cd,aci->Bi', l_densities, np.conj(core_tensor), 
                                            r_density, core_tensor)
    norm_factor = np.sum(probs, axis=1)
    probs = probs / norm_factor[:, None]
    cum_probs = np.cumsum(probs, axis=1)
    # The following are good checks, but can't run them due to JIT
    # assert np.all(probs > 0)
    # assert np.allclose(cum_probs[:, -1], np.ones(num_samps))

    # Sample from cum_probs (argmax finds first cp with cp > rand_float)
    samp_ints = np.argmax(cum_probs > rand_floats[:, None], axis=1)
    samp_mats = core_tensor[:, :, samp_ints]

    # Conditionally evolve to get new left densities
    new_densities = np.einsum('acB,Bab,bdB->Bcd', np.conj(samp_mats), 
                                                  l_densities, samp_mats)
    norms = np.trace(new_densities, axis1=1, axis2=2)[:, None, None]
    new_densities = new_densities / norms

    return new_densities, samp_ints

def fill_in_blanks(rng_key, ti_mps, alphabet, ref_strset):
    """
    Use a TI-MPS model to fill in the blanks of reference strings

    Args:
        rng_key:    Random key used within Jax's PNRG
        ti_mps:     The TI-MPS model defining our sampling distribution
        alphabet:   The alphabet defining a basis in our sampling space
        ref_strset: StrSet object containing reference strings

    Returns:
        char_list:  List of (lists of) character replacements, where the
                    character replacements for a string s are what the MPS
                    predicts would fill in the missing gap for each site
                    in s. There is one list for each s, which contains one
                    string for each site
    """
    # Unpack input args
    core_tensor = ti_mps.core_tensor
    bd, in_dim = core_tensor.shape[1:]
    assert len(alphabet) == in_dim
    left_b, right_b = get_bound_mats(ti_mps)
    str_inds = ref_strset.index_mat.T
    str_len, batch_size = str_inds.shape

    # Get transition matrices for each character in ref_strset
    trans_mats = core_tensor[:, :, str_inds]
    trans_mats = np.transpose(trans_mats, (2, 3, 0, 1))

    # Calculate left environments for all our samples
    lenv_list = [np.broadcast_to(left_b, (batch_size, bd, bd))]
    for i in range(str_len-1):
        new_lenv = np.einsum('Bab,Bbd,Bac->Bcd', lenv_list[-1], 
                              np.conj(trans_mats[i]), trans_mats[i])

        lenv_list.append(new_lenv / np.linalg.norm(new_lenv, axis=(1,2), 
                                                   keepdims=True))
    assert len(lenv_list) == str_len
    lenv_list = np.array(lenv_list)

    # Calculate right environments for all our samples
    renv_list = [np.broadcast_to(right_b, (batch_size, bd, bd))]
    for i in range(str_len-1, 0, -1):
        new_renv = np.einsum('Bab,Bdb,Bca->Bcd', renv_list[-1], 
                              np.conj(trans_mats[i]), trans_mats[i])

        renv_list.append(new_renv / np.linalg.norm(new_renv, axis=(1,2), 
                                                   keepdims=True))
    assert len(renv_list) == str_len
    renv_list = np.array(renv_list[::-1])

    # Generate random numbers for use in sampling
    rand_floats = jax.random.uniform(rng_key, shape=(str_len, batch_size))

    # Get unnormalized probabilities and normalize
    # obs = np.einsum('Bab,bdi,cd,aci->Bi', l_densities, np.conj(core_tensor), 
    probs = np.einsum('SBab,bdi,SBcd,aci->SBi', lenv_list, 
                      np.conj(core_tensor), renv_list, core_tensor)
    probs = probs / np.sum(probs, axis=2, keepdims=True)
    cum_probs = np.cumsum(probs, axis=2)

    # Sample from cum_probs (argmax finds first cp with cp > rand_float)
    samp_ints = np.argmax(cum_probs > rand_floats[..., None], axis=2).T

    # Convert to list of character lists
    ind2char = {i: c for i, c in enumerate(alphabet)}
    char_list = [[ind2char[i] for i in inds] for inds in samp_ints]
    
    return char_list