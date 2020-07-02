import os
import string
import pickle
from functools import lru_cache

import jax
import numpy as onp
import jax.numpy as jnp

@lru_cache()
def batch_einstring(ein_str, num_batch_dims=0):
    """
    Convert einsum string into new string with extra batch dims prepended

    Args:
        ein_str:        Einsum string of the form 's_1[,s_i][->s_out]',
                        where the lefthand side describes any positive 
                        number of input tensors and the righthand side 
                        describes any number of output tensors
        num_batch_dims: The number of batch dimensions to prepend to all 
                        entries of the einsum string

    Returns:
        batch_str:      Einsum string describing the same operation as 
                        ein_str, but with extra batch dimensions prepended
    """
    # Check input and get collection of free characters
    char_set = set(ein_str)
    free_chars = string.ascii_letters
    assert len(char_set.difference(free_chars + ' ,->()')) == 0    
    free_chars = ''.join(list(set(free_chars).difference(list(char_set))))

    # Split ein_str into constituent pieces
    ein_str = ein_str.replace(' ', '')  # Strip whitespace
    has_rhs = '->' in ein_str
    if has_rhs:
        lhs, rhs = ein_str.split('->')
        rhs = rhs.split(',')
    else:
        lhs = ein_str
    lhs = lhs.split(',')
    assert len(lhs) >= 1

    # Add batch dimensions
    prefix = free_chars[:num_batch_dims]
    new_lhs = [prefix + term for term in lhs]
    batch_str = ','.join(new_lhs)
    if has_rhs:
        new_rhs = [prefix + term for term in rhs]
        batch_str += '->' + ','.join(new_rhs)

    return batch_str

def einmerge(merge_str, tensor):
    """
    Merge indices of a tensor using an einsum-style syntax

    ein_str must describe a single tensor, where closed groups of 
    parentheses describe adjacent indices to be merged. For example, the 
    string 'a(bc)' takes a third-order tensor and merges the last two 
    indices into a composite index, returning a second-order tensor. The
    string 'a(bc)(efg)' similarly takes in a sixth-order tensor and 
    returns a third-order tensor.

    Args:
        merge_str:     The string specifying the merging to apply
        tensor:        The tensor whose indices we want to merge

    Returns:
        merged_tensor: Identical to tensor, but where some of the indices
                       have been merged together
    """
    letters = string.ascii_letters
    raw_shape, merge_shape = tensor.shape, []
    assert len(set(merge_str).difference(list(letters + '()'))) == 0

    # Parse merge_str and build shape of merged_tensor
    paren_seen, ind_count, running_dim = False, 0, 1
    for c in merge_str:
        if c in letters:
            if not paren_seen:
                merge_shape.append(raw_shape[ind_count])
            else:
                running_dim *= raw_shape[ind_count]
            ind_count += 1
        elif c == '(':
            assert not paren_seen
            paren_seen = True
        elif c == ')':
            merge_shape.append(running_dim)
            assert paren_seen
            paren_seen, running_dim = False, 1
    assert ind_count == len(raw_shape)

    return jnp.reshape(tensor, merge_shape)

def two_normalize(tensor, axis=None):
    """
    Reduce the norm of tensor to near one by rescaling using power of two

    Args:
        tensor:     The tensor we wish to two-normalize. When axis is 
                    specified, the remaining axes are treated as batch dims
        axis:       Int or tuple of ints specifying the axes in which 
                    two-normalization occurs. When axis=None, the entire 
                    tensor is two-normalized
    
    Returns:
        out_tensor: Same as tensor, but where appropriate norms have been
                    two-normalized to be between 1 and 2
        two_pow:    The power of two by which out_tensor was rescaled
    """
    two_pow = jnp.floor(jnp.log2(jnp.linalg.norm(tensor, axis=axis, 
                                                         keepdims=True)))
    tensor = tensor / 2**two_pow

    return tensor, jnp.squeeze(two_pow, axis=axis)

def core_direct_sum(core_list, mat_axes=(-2, -1)):
    """
    Arrange multiple core tensors in a block diagonal along specified axes

    Args:
        core_list: Sequence of core tensors whose shapes are identical in
                   all axes besides those in mat_axes (no broadcasting)
        mat_axes:  Tuple of two axes along which block diagonalization 
                   (direct sum) is performed

    Return:
        sum_core:  Single core tensor whose dimensions in general axes are 
                   equal to those of each input core, and are sums of the 
                   corresponding dimensions in entries of mat_axes
    """
    # Get shapes of all axes not in mat_axes
    shapes = [list(c.shape) for c in core_list]
    for s in shapes: s[mat_axes[0]] = s[mat_axes[1]] = -1
    assert len(set(tuple(s) for s in shapes)) == 1  # Equal non-mat axes
    gen_shape = tuple(si for si in shapes[0] if si != -1)

    # Swap all mat_axes into last two axis positions
    swap_op = lambda c: jnp.swapaxes(jnp.swapaxes(c, mat_axes[0], -2), 
                                                     mat_axes[1], -1)
    core_list = [swap_op(c) for c in core_list]
    mat_dims = [[c.shape[i] for c in core_list] for i in (-2, -1)]
    sum_dims = tuple(sum(md) for md in mat_dims)

    # Form all-zero mat and fill block diagonal entries
    i, j = 0, 0
    sum_core = jnp.zeros(gen_shape + sum_dims)
    for core, m_dims in zip(core_list, zip(*mat_dims)):
        l_d, r_d = m_dims
        # Equivalent to -> sum_core[..., i:i+l_d, j:j+r_d] = core
        sum_core = jax.ops.index_update(sum_core, 
                            jax.ops.index[..., i:i+l_d, j:j+r_d], core)
        i, j = i+l_d, j+r_d

    # Swap mat_axes into original axis positions
    sum_core = swap_op(sum_core)

    return sum_core


def dagger(tensor):
    """Hermitian conjugate on all matrices in batch of matrices"""
    return jnp.conj(jnp.swapaxes(tensor, -2, -1))

def pad_to(tensor, pad_shape, **kwargs):
    """
    Pad an array into a specific shape. Wrapper for numpy.pad

    Args:
        tensor:    The tensor to be padded
        pad_shape: The shape which `tensor` will be padded to. The length
                   of shape must agree with the number of axes of `tensor`
                   (no broadcasting), and any axes for which 
                   dim_i(tensor) < pad_shape_i will be padded to have the
                   correct shape. Whenever dim_i(tensor) >= pad_shape_i, 
                   no padding occurs, and the original axis length is kept
        kwargs:    Any keyword arguments are passed to numpy.pad
    """
    # Determine the amount of padding for each axis
    old_shape = tensor.shape
    assert len(old_shape) == len(pad_shape)
    pad_width = tuple((0, max(s_o, s_p) - s_o) 
                          for s_o, s_p in zip(old_shape, pad_shape))

    return jnp.pad(tensor, pad_width, **kwargs)

@jax.custom_transforms
def stable_svd(a):
    """
    Singular Value Decomposition with more stable autodiff rule
    """
    return jnp.linalg.svd(a, full_matrices=False, compute_uv=True)

def stable_svd_jvp(primals, tangents):
    """Copied from the JAX source code and slightly tweaked for stability"""
    # Deformation parameter which yields regular SVD JVP rule when set to 0
    eps = 1e-10
    A, = primals
    dA, = tangents
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False, compute_uv=True)

    _T = lambda x: jnp.swapaxes(x, -1, -2)
    _H = lambda x: jnp.conj(_T(x))
    k = s.shape[-1]
    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = jnp.matmul(jnp.matmul(Ut, dA), V)
    ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

    # Deformation by eps avoids getting NaN's when SV's are degenerate
    f = jnp.square(s_dim) - jnp.square(_T(s_dim)) + jnp.eye(k)
    f = f + eps / f  # eps controls stability
    F = 1 / f - jnp.eye(k) / (1 + eps)
    
    dSS = s_dim * dS
    SdS = _T(s_dim) * dS
    dU = jnp.matmul(U, F * (dSS + _T(dSS)))
    dV = jnp.matmul(V, F * (SdS + _T(SdS)))

    m, n = A.shape[-2], A.shape[-1]
    if m > n:
        dU = dU + jnp.matmul(jnp.eye(m) - jnp.matmul(U, Ut), 
                                        jnp.matmul(dA, V)) / s_dim
    if n > m:
        dV = dV + jnp.matmul(jnp.eye(n) - jnp.matmul(V, Vt), 
                                        jnp.matmul(_H(dA), U)) / s_dim
    return (U, s, Vt), (dU, ds, _T(dV))

# Override the default JVP rule with the stable one defined above
jax.defjvp_all(stable_svd, stable_svd_jvp)

def svd_flex(ein_string, tensor, svd_thresh, max_D=0, 
             sv_treatment='symmetric', sv_out=True, backend='jax'):
    """
    Partition modes of an input tensor into two pieces via a flexible SVD

    Args:
        ein_string:   An einsum-style string of the form 
                      'input->left_out,right_out', where input labels the 
                      indices of tensor, and left_out/right_out describe 
                      those of the left and right output tensors, along 
                      with a new index joining the two. For example, an 
                      ein_string for the standard matrix SVD is 'ij->ik,kj'

                      Reversing the components of ein_string to the left 
                      and right of '->' gives an ein_string which 
                      multiplies the two outputs into a low rank 
                      approximation of the input tensor

        tensor:       Numpy (or Jax Numpy) tensor with two or more indices

        svd_thresh:   A parameter which truncates any SVD components with
                      singular values below the threshold. 
                      Setting this to 0. yields an exact SVD

        max_D:        A maximum allowed value for the new bond. When 
                      max_D = 0, this yields an exact SVD

        sv_treatment: A string describing what we want to do with the 
                      singular values. Our options are:

                      'symmetric': Multiply left and right outputs by the
                                   square root of the SV vector
                      'left':      Multiply left output by the SV vector
                      'right':     Multiply right output by the SV vector
                      'none':      Don't multiply SV vector into outputs

        sv_out:       Whether to return full singular value vector

        backend:      Specifies whether we want to use Jax ('jax') or 
                      classic Numpy ('numpy').

    Returns:
        left_tensor,
        right_tensor: Numpy tensors representing the left and right 
                      partitions of the input tensor, whose indices are
                      described in the righthand part of ein_string

        sv_vec:       Numpy vector containing singular values, only output
                      when sv_out = True
    """
    assert sv_treatment in ['symmetric', 'left', 'right', 'none']
    assert backend in ['jax', 'numpy']

    # Convert tensor to the appropriate format
    if backend == 'numpy':
        back = onp
    else:
        back = jnp

    tensor = back.asarray(tensor)

    # Check that our ein_string has correct formatting
    assert is_valid_svd_string(ein_string)

    # Parse ein_string into init_str, left_str, and right_str
    ein_string = ein_string.replace(' ', '')
    init_str, post_str = ein_string.split('->')
    left_str, right_str = post_str.split(',')

    # Get free indices and the new bond character
    bond_char = set(left_str).intersection(set(right_str)).pop()
    left_free = left_str.replace(bond_char, '')
    right_free = right_str.replace(bond_char, '')

    # Permute the indices of tensor into something closer to the output
    tensor = back.einsum(f"{init_str}->{left_free+right_free}", tensor)

    # Flatten both sides of our tensor to give a single matrix
    left_shape = tensor.shape[:len(left_free)]
    right_shape = tensor.shape[len(left_free):]
    left_size, right_size = back.prod(left_shape), back.prod(right_shape)
    matrix = tensor.reshape((left_size, right_size))

    # TODO: Add special handling for case when one side of partition is
    #       all singleton dimensions (i.e. min(left_size, right_size) == 1)

    # Get SVD and format so that left_mat @ diag(svs) @ right_mat = matrix
    left_mat, sv_vec, right_mat = back.linalg.svd(matrix, compute_uv=True, 
                                                  full_matrices=False)
    assert len(sv_vec) == left_mat.shape[1] == right_mat.shape[0]
    assert back.all(back.sort(sv_vec)[::-1] == sv_vec)
    assert len(sv_vec.shape) == 1

    # Get the truncation point arising from svd_thresh and max_D
    cutoff = len(sv_vec)
    if svd_thresh > 0:
        for i, s in enumerate(sv_vec):
            if s < svd_thresh:
                cutoff = i
                break
    if max_D > 0:
        cutoff = min(cutoff, max_D)

    # I don't want to handle cutoff being 0, it isn't useful
    if cutoff == 0:
        print("Warning: svd_thresh is big enough to truncate **all** SV's"
              ", overriding this behavior and including one SV")
        cutoff = 1
    assert cutoff == len(sv_vec) or \
           sv_vec[cutoff-1] >= svd_thresh > sv_vec[cutoff]

    # Truncate the left and right outputs of our SVD
    left_mat, right_mat = left_mat[:, :cutoff], right_mat[:cutoff]
    big_sv_vec = sv_vec[:cutoff]

    # Fold the singular values into the left and right SVD outputs
    if sv_treatment == 'symmetric':
        sqrt_sv_vec = back.sqrt(big_sv_vec)
        left_mat = back.einsum('ij,j->ij', left_mat, sqrt_sv_vec)
        right_mat = back.einsum('j,jk->jk', sqrt_sv_vec, right_mat)
    elif sv_treatment == 'left':
        left_mat = back.einsum('ij,j->ij', left_mat, big_sv_vec)
    elif sv_treatment == 'right':
        right_mat = back.einsum('j,jk->jk', big_sv_vec, right_mat)

    # Reshape the matrices to make them proper tensors
    left_tensor = left_mat.reshape(left_shape+(cutoff,))
    right_tensor = right_mat.reshape((cutoff,)+right_shape)

    # Move the new bond indices into the correct order
    left_tensor = back.einsum(f"{left_free+bond_char}->{left_str}",
                            left_tensor)
    right_tensor = back.einsum(f"{bond_char+right_free}->{right_str}",
                            right_tensor)

    if sv_out:
        return left_tensor, right_tensor, sv_vec
    else:
        return left_tensor, right_tensor

@lru_cache()
def build_svd_fun(ein_string, sv_treatment='symmetric', sv_out=True, 
                  jitted=True):
    """
    Returns a function which implements minimal version of svd_flex, for
    one particular einstring

    The function returned by build_svd_fun is designed to be JAX-friendly,
    so that it can be backpropagated, JIT-compiled, vectorized, etc. The
    resultant function takes in only one argument, a tensor whose indices
    must agree with the first entry of ein_string. For simplicity, there
    is currently no support for truncation.

    Args:
        ein_string:   An einsum-style string of the form 
                      'input->left_out,right_out', where input labels the 
                      indices of tensor, and left_out/right_out describe 
                      those of the left and right output tensors, along 
                      with a new index joining the two. For example, an 
                      ein_string for the standard matrix SVD is 'ij->ik,kj'

                      Reversing the components of ein_string to the left 
                      and right of '->' gives an ein_string which 
                      multiplies the two outputs into a low rank 
                      approximation of the input tensor

        sv_treatment: A string describing what we want to do with the 
                      singular values. Our options are:

                      'symmetric': Multiply left and right outputs by the
                                   square root of the SV vector
                      'left':      Multiply left output by the SV vector
                      'right':     Multiply right output by the SV vector
                      'none':      Don't multiply SV vector into outputs

        sv_out:       Whether to return full singular value vector

        jitted:       Whether to make svd_fun JIT compiled or not

    Returns:
        svd_fun:      A function which takes in a single tensor and returns
                      either two or three tensors (depending on sv_out).
                      The behavior of this function is set by the options 
                      fed to build_svd_fun.
    """
    assert sv_treatment in ['symmetric', 'left', 'right', 'none']

    # Check that our ein_string has correct formatting
    assert is_valid_svd_string(ein_string)

    # Parse ein_string into init_str, left_str, and right_str
    ein_string = ein_string.replace(' ', '')
    init_str, post_str = ein_string.split('->')
    left_str, right_str = post_str.split(',')

    # Get free indices and the new bond character
    bond_char = set(left_str).intersection(set(right_str)).pop()
    left_free = left_str.replace(bond_char, '')
    right_free = right_str.replace(bond_char, '')

    # Define function for fold singular values into left/right SVD outputs
    if sv_treatment == 'symmetric':
        def apply_sv(lm, rm, sv):
            sqrt_sv_vec = jnp.sqrt(sv)
            lm = jnp.einsum('ij,j->ij', lm, sqrt_sv_vec)
            rm = jnp.einsum('j,jk->jk', sqrt_sv_vec, rm)
            return lm, rm
    elif sv_treatment == 'left':
        apply_sv = lambda lm, rm, sv: (jnp.einsum('ij,j->ij', lm, sv), rm)
    elif sv_treatment == 'right':
        apply_sv = lambda lm, rm, sv: (lm, jnp.einsum('j,jk->jk', sv, rm))
    elif sv_treatment == 'none':
        apply_sv = lambda lm, rm, sv: (lm, rm)

    # Define function for returning output
    if sv_out:
        out_fun = lambda lt, rt, sv: (lt, rt, sv)
    else:
        out_fun = lambda lt, rt, sv: (lt, rt)

    def svd_fun(tensor):
        # Permute the indices of tensor into something closer to the output
        tensor = jnp.einsum(f"{init_str}->{left_free+right_free}", tensor)

        # Flatten both sides of our tensor to give a single matrix
        left_shape = tensor.shape[:len(left_free)]
        right_shape = tensor.shape[len(left_free):]
        left_size = jnp.prod(left_shape)
        right_size = jnp.prod(right_shape)
        matrix = tensor.reshape((left_size, right_size))

        # Get SVD and format so that left_mat@diag(svs)@right_mat = matrix
        left_mat, sv_vec, right_mat = stable_svd(matrix)

        # Fold singular values into left/right matrices
        left_mat, right_mat = apply_sv(left_mat, right_mat, sv_vec)

        # Reshape the matrices to make them proper tensors
        left_tensor = left_mat.reshape(left_shape + sv_vec.shape)
        right_tensor = right_mat.reshape(sv_vec.shape + right_shape)

        # Move the new bond indices into the correct order
        left_tensor = jnp.einsum(f"{left_free+bond_char}->{left_str}",
                                left_tensor)
        right_tensor = jnp.einsum(f"{bond_char+right_free}->{right_str}",
                                right_tensor)

        return out_fun(left_tensor, right_tensor, sv_vec)

    return jax.jit(svd_fun) if jitted else svd_fun

def is_valid_svd_string(ein_string: str) -> bool:
    """
    Check the input ein_string is properly formatted for use in svd_flex

    Args:
        ein_string: See svd_flex for formatting restrictions

    Returns:
        is_valid:   True if the format is valid, false otherwise
    """
    # Parse ein_string into init_str, left_str, and right_str
    ein_string = ein_string.replace(' ', '')
    try:
        init_str, post_str = ein_string.split('->')
        left_str, right_str = post_str.split(',')
    except ValueError:
        return False

    str_list = [init_str, left_str, right_str]

    # Check that each string component has correct formatting, meaning:
    # (1) Only alphabetic characters,
    # (2) No duplicate characters), 
    # (3) Each component must have at least one free index, and 
    # (4) The characters in left_str and right_str are those in init_str 
    #     plus one extra character for the new bond
    cond_1 = (init_str + left_str + right_str).isalpha()
    cond_2 = all(len(set(s)) == len(s) for s in str_list)
    cond_3 = all(len(s) >= 2 for s in str_list)
    cond_4a = len(left_str + right_str) == len(init_str) + 2
    cond_4b = len(set(left_str + right_str) - set(init_str)) == 1

    return cond_1 and cond_2 and cond_3 and cond_4a and cond_4b

def disk_cache(cachefile):
    """
    Decorator that will save function results in 'cachefile'

    NOTE: Based on "https://datascience.blog.wzb.eu/2016/08/12/a-tip-for-
            the-impatient-simple-caching-with-python-pickle-and-decorators"
    """
    # Avoid clutter from cachefiles
    cachefile = '.' + cachefile
    
    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args, **kwargs):   # define a wrapper that will finally call "fn" with all arguments
            def hashify(x):
                # If any args or kwargs aren't hashable, repr them
                try:
                    hash(x)
                    return x
                except TypeError:
                    return repr(x)

            # Convert all unhashable args to hashable ones
            h_args = tuple(hashify(a) for a in args)
            h_kwargs = tuple((k, hashify(v)) for k, v in kwargs.items())

            # if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as cachehandle:
                    cache_dict = pickle.load(cachehandle)

                if (h_args, h_kwargs) in cache_dict:
                    print("NOTE: Using saved result from '%s'" % cachefile)
                    return cache_dict[(h_args, h_kwargs)]

            else:
                cache_dict = {}

            # No record, so execute the function with all arguments passed
            res = fn(*args, **kwargs)
            cache_dict[(h_args, h_kwargs)] = res

            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                pickle.dump(cache_dict, cachehandle)

            return res

        return wrapped

    return decorator   # return this "customized" decorator that uses "cachefile"

def transfer_op_eigs(transfer_ops, bond_dim, guesses=None, precision=1e-7, 
                     max_iter=1e6):
    """
    Given a transfer operator and its adjoint, calculate their dominant 
    eigenvalue, as well as their dominant eigenmatrices, using the power 
    method.

    Loosely based on TMeigs_power_method function from the TensorNetwork
    library

    Args:
        transfer_ops: Function which takes in a batch matrix of size 
                      (2, bond_dim, bond_dim) and applies a left and right
                      transfer operator to both matrices
        bond_dim: Integer giving the size of the bond space where input/
                  output matrices for the transfer operators live
        guesses: Batch matrix of size (2, bond_dim, bond_dim) giving
                 our guess for the left and right eigenmatrices. If
                 None, the identity is used as a default guess.
        precision: The desired precision for our eigenvalue
        max_iter: The maximum number of iterations for our eigensolver 

    Returns:
        eig_val: Dominant eigenvalue of the transfer operator
        eig_mats: Dominant left and right eigenmatrices of the transfer op
    """
    assert precision > 0

    # If no guess was given, use the identity
    if guesses is None:
        guesses = jnp.broadcast_to(jnp.eye(bond_dim), (2,bond_dim,bond_dim))

    # Normalize our initial guesses
    mats = jnp.stack([m / jnp.linalg.norm(m) for m in guesses])

    # Apply our transfer operator and its adjoint to our matrices
    # until we reach convergence
    loop_num = 0
    error = 2 * precision
    while error > precision and loop_num < max_iter:
        loop_num += 1

        # Compute transfer-operated versions of our matrices, and get the
        # amount they are rescaled by
        new_mats = transfer_ops(mats)
        eigs = [jnp.linalg.norm(nm) for nm in new_mats]
        new_mats = jnp.stack([m / jnp.linalg.norm(m) for m in new_mats])
    
        # Get the difference of our rescaled output matrices and the 
        # respective input matrices
        diff_mats = new_mats - mats
        diff_norms = [jnp.linalg.norm(dm) for dm in diff_mats]
        error = max(diff_norms)

    # Now that everything has converged, get the eigenvalue as the average
    # of the left and right dominant eigenvalues (should be the same)
    #   TODO: Allow for arbitrary eigenvalues, not just positive ones
    assert max(eigs) - min(eigs) < precision
    eig_val = sum(eigs) / 2

    return jnp.asarray(eig_val), jnp.asarray(new_mats)

def flat_t_op_fun(core_tensor, direction='left'):
    """
    Same as t_op_fun, but returns a function which acts on flattened matrices
    """
    mat_shape = core_tensor.shape[:2]
    flat_shape = (jnp.prod(mat_shape),)
    t_op = t_op_fun(core_tensor, direction=direction, jitted=True)

    @jax.jit
    def flat_t_op(flat_mat):
        mat = flat_mat.reshape(mat_shape)
        out_mat = t_op(mat)
        return out_mat.reshape(flat_shape)

    return flat_t_op

@jax.jit
def hs_dot(mat1, mat2):
    """
    Compute the Hilbert-Schmidt inner-product between two matrices, 
    <M1, M2> = trace(M1.T @ M2)

    Args:
        mat1, mat2: Numpy matrices of equal size (D1, D2), or two arrays
                    of such matrices with sizes (batch_size, D1, D2)

    Returns:
        innner_prod: A scalar or batch of scalars 
    """
    assert all(isinstance(m, jnp.ndarray) and len(m.shape) in (2, 3) 
               for m in [mat1, mat2])
    assert mat1.shape == mat2.shape
    batched = len(mat1.shape) == 3
    return jnp.einsum('blr,blr->b' if batched else 'lr,lr', mat1, mat2)

def norm(tensor):
    """
    Gives the vector norm of the flattened tensor
    """
    return jnp.linalg.norm(tensor.reshape((-1,)))

def eye_tensor(bond_dim, in_dim):
    """
    Generate a core tensor whose matrix slices are all identity matrices

    Args:
        bond_dim: Dimension of the identity matrices
        in_dim:   Input dimension of the tensor

    Returns:
        tensor:   Jax Numpy tensor with shape (bond_dim, bond_dim, in_dim)
    """
    shape = (bond_dim, bond_dim, in_dim)
    return jnp.broadcast_to(jnp.eye(bond_dim)[..., None], shape)

def parity_init(bond_dim, parity=0, other_init='eye'):
    """
    Initialize an MPS that already enforces the parity constraint
    """
    assert bond_dim % 2 == 0
    assert parity in [0, 1]
    assert other_init in ['eye', 'rand', 'steye', 'strand']
    is_par = other_init in ['eye', 'rand']
    is_eye = other_init in ['eye', 'steye']
    half_dim = bond_dim // 2
    in_dim = 3

    # Build the parity sub-tensor
    par_tensor = onp.empty((in_dim, 2, 2))
    if is_par:
        # Parity-aware automaton
        par_tensor[0] = onp.eye(2)
        par_tensor[1] = onp.asarray([[0, 1], [1, 0]])
    else:
        # Stochastic automaton
        par_tensor[:2] = onp.ones((2, 2, 2)) / 2
    alpha = onp.asarray([1, 0])
    omega = alpha if parity == 0 else par_tensor[1].dot(alpha)
    par_tensor[2] = 0.2 * onp.outer(omega, alpha)
    par_tensor /= onp.sqrt(2)

    # Build the other sub-tensor
    rand_t = onp.random.standard_normal((in_dim, half_dim, half_dim))
    if is_eye:
        other_tensor = onp.eye(half_dim)[None] + 1e-6 * rand_t

    else:
        other_tensor = rand_t / onp.sqrt(half_dim)

    # Put the pieces together using the Kronecker product
    core_tensor = onp.empty((bond_dim, bond_dim, in_dim))
    for i in range(in_dim):
        core_tensor[:, :, i] = onp.kron(par_tensor[i], other_tensor[i])

    return core_tensor

def weighted_nll_loss(batch_probs, input_lens=None):
    """
    Get the negative log likelihood of batch of probabilities, normalized
    by lengths of original sentences

    Args:
        batch_probs: Numpy array with unnormalized probability of sentences
                     relative to MPS probability distribution
        input_lens: Numpy array of lengths of original sentences. If None
                    is given, lengths are identically set to 1

    Returns:
        loss: Sum of negative log likelihood over all input sentences
    """
    if input_lens is None:
        input_lens = jnp.ones((len(batch_probs),))

    # Convert to log probabilities, and normalize by length to ensure
    # each sentence contributes roughly equal gradients
    loss = jnp.sum(-jnp.log(batch_probs) / input_lens)

    return loss

def calculate_ppl(probs, input_lens, norm_per_sym):
    """
    Calculate the per-symbol perplexity of a collection of input sequences 
    with varying lengths

    Args:
        probs:        Unnormalized probabilities of collection of sequences
        input_lens:   Lengths of the same sequences
        per_sym_norm: To convert unnormalized probs to normalized ones, we
                      assume the log_norm of each length n sampling 
                      distribution has the form n * log(per_sym_norm),
                      which is true for infinite boundary conditions
    """
    weighted_cross_entropy = onp.mean(-onp.log(probs) / input_lens)
    unnormalized_ppl = onp.exp(weighted_cross_entropy)

    # Account for the known normalization factor in input probabilities
    ppl = per_sym_norm * unnormalized_ppl

    return float(ppl)

def ppl_calc(log_probs, seq_lens):
    """Calc the PPL for a bunch of log_probs"""
    ppl = onp.exp(-onp.mean(log_probs / seq_lens))
    return ppl

def stable_mean(vector):
    """Calculate the mean of a vector, with all non-finite values dropped"""
    assert len(vector.shape) == 1
    finite_vals = [v for v in vector if jnp.isfinite(v)]

    return sum(finite_vals) / len(finite_vals)

def no_error(val_foo, give_val=False):
    """Run val_foo with no inputs, test whether it raises an error"""
    try:
        val = val_foo()
        success = True
    except:
        val = None
        success = False

    if give_val:
        return success, val
    else:
        return success
