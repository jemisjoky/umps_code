from functools import partial

import jax
import jax.numpy as np

@partial(jax.jit, static_argnums=(1,))
def parallel_contract(matrices, give_lognorm=False):
    """
    Reduce a batch of sequences of square matrices in parallel

    Args:
        matrices:     Tensor with shape (batch_size, length, bd, bd), 
                      for bd the matrix dimension and length the sequence 
                      length.
        give_lognorm: If true, the log normalization of the output matrix 
                      (or matrices) will be given as a second output, an
                      option which avoids overflow.

    
    Returns:
        prod_matrix:  Matrix of shape (batch_size, bd, bd)
        log2_norms:   Power of 2 by which values of prod_matrix were reduced
    """
    # Unpack input
    shape = matrices.shape
    assert len(shape) == 4
    assert shape[2] == shape[3]
    batch_size, length, bd = shape[:3]
    mats = np.swapaxes(matrices, 0, 1)

    # Define quantities for normalization
    def two_normalize(m):
        # Divide m by a power of 2 to get its norm close to 1
        norm = np.linalg.norm(m, axis=(2,3), 
                                 keepdims=True)
        two_pow = np.floor(np.log2(norm))
        stable_m = m / (2 ** two_pow)

        return stable_m, np.sum(two_pow, axis=0)

    # Iteratively contract half of our matrices in parallel with the other
    # half, until we've reduced each matrix sequence to a single matrix
    mats, log2_norms = two_normalize(mats)
    while mats.shape[0] > 1:
        # For odd sizes, set aside the last matrix as a "carry"
        half_size = mats.shape[0] // 2
        even_size = 2 * half_size
        carry_mats = mats[even_size:]

        # Contract all pairs of neighboring matrices and get norm factor
        mats = np.einsum('sblu,sbur->sblr', mats[0:even_size:2], 
                                            mats[1:even_size:2])
        mats, two_pow = two_normalize(mats)
        mats = np.concatenate((mats, carry_mats))

        # Update the running exponent offsets
        log2_norms = log2_norms + two_pow

    # Handle the log norms according to value of give_lognorm
    mat = np.squeeze(mats, axis=0)
    if give_lognorm:
        return mat, np.squeeze(log2_norms, axis=(1,2))
    else:
        return mat * 2**log2_norms
