import warnings

import jax
import numpy as onp
import jax.numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from utils import flat_t_op_fun, hs_dot

def build_t_op(core_tensor, direction, jitted=True):
    """
    Get the transfer operator for a TI-MPS, which acts on an input matrix

    Args:
        core_tensor: MPS core tensor of shape (bond_dim, bond_dim, in_dim)
                     which defines the output transfer operator
        direction:   Either 'left', 'right', or 'both', specifying the 
                     direction in which output transfer operator propagates
                     its input. The last case gives a bidirectional t_op
                     which acts on a pair of density operators, described 
                     as a matrix with extra batch index of dim 2
        jitted:      Whether we want the output transfer operator function 
                     to be passed through Jax's `jit` function
    """
    assert direction in ['left', 'right', 'both']

    if direction == 'left':
        t_op = lambda mat: np.einsum('cai,ab,dbi->cd', 
                                     core_tensor, mat, core_tensor)
    elif direction == 'right':
        t_op = lambda mat: np.einsum('aci,ab,bdi->cd', 
                                     core_tensor, mat, core_tensor)
    elif direction == 'both':
        core_tensors = np.stack([core_tensor, 
                                 np.swapaxes(core_tensor, 0, 1)])
        t_op = lambda mat: np.einsum('Baci,Bab,Bbdi->Bcd', 
                                     core_tensors, mat, core_tensors)

    return jax.jit(t_op) if jitted else t_op

def transfer_eigs_power(core_tensor, boundaries, precision=1e-7, 
                        max_iter=1000):
    """
    Estimate the dominant eigenvalue of our MPS's transfer operator, along
    with its dominant left and right eigenmatrices, using the power method.

    Args:
        core_tensor: Numpy core tensor defining our distribution
        boundaries:  Numpy tensor giving initial guesses for the left and 
                     right eigenmatrices
        precision:   The desired precision for our eigenvalue
        max_iter:    The maximum number of transfer operator iterations to
                     apply to our initial boundary operators

    Returns:
        eig_val:  Dominant eigenvalue of the transfer operator
        eig_mats: Dominant left and right eigenmatrices of the transfer op,
                  each normalized to have unit Frobenius norm
    """
    bd, bdd, d = core_tensor.shape
    assert bd == bdd
    assert precision > 0
    assert boundaries.shape == (2, bd, bd)

    # Build a bidirectional transfer operator function from our core tensor
    bi_t_op = build_t_op(core_tensor, direction='both', jitted=True)

    # `mats` will always remain properly normalized in the following
    mats = np.stack([m / np.linalg.norm(m) for m in boundaries])

    # Apply our transfer operator and its adjoint to our matrices
    # until we reach convergence
    error = 2 * precision
    for loop_num in range(max_iter):
        if error < precision:
            break

        # Compute transfer-operated versions of our matrices, get overlaps
        # with original mats, and scale to make them unit norm
        new_mats = bi_t_op(mats)
        eigs = [hs_dot(n_m, m) for n_m, m in zip(new_mats, mats)]

        norms = np.linalg.norm(new_mats, axis=(1,2))
        new_mats = new_mats / norms[:, None, None]
    
        # Choose our error to be the average per-element difference between
        # our new matrix and old matrix
        diffs = np.linalg.norm(new_mats-mats, axis=(1,2))
        mats = new_mats
        error = sum(diffs) / 2

    # Now that everything has converged, get the dominant eigenvalue as the 
    # average of the left, right eigenvalues (should be positive and equal)
    eig_val = sum(eigs) / 2
    assert eig_val > 0, eig_val
    if not np.isclose(max(eigs), min(eigs), rtol=1e-3):
        warnings.warn("Left/right transfer ops getting different eigs, "
                     f"{eigs[0]} and {eigs[1]}, respectively")

    # Convert to default precision before returning
    return eig_val, new_mats

def transfer_eigs_scipy(core_tensor, guess_mats, guess_eig=None, 
                       precision=1e-7, check_gap=False):
    """
    Calculate the dominant eigenvalue and dominant left/right eigenmatrices 
    of our MPS transfer operator using the SciPy sparse eigensolver

    Args:
        core_tensor: Numpy core tensor defining our distribution
        guess_mats: Numpy tensor giving initial guesses for the left and 
                    right eigenmatrices
        guess_eig: Initial guess for the eigenvalue
        precision: The desired precision for our eigenvalue
        check_gap: Whether we calculate the spectral gap between the 
                   absolute value of the two largest eigenvalues, in 
                   which case this gap is given as an output

    Returns:
        eigenvalue: Dominant eigenvalue of the transfer operator
        eig_mats: Dominant left and right eigenmatrices of the transfer op,
                  each normalized to have unit Frobenius norm
        spec_gap: The gap between the largest and second largest 
                  eigenvalues. Only returned when check_gap is True
    """
    bd, bdd, d = core_tensor.shape
    mat_shape = guess_mats.shape[1:]
    mat_size = guess_mats.size // 2
    assert precision > 0
    assert guess_mats.shape[0] == 2
    assert mat_shape == (bd, bdd) == (bd, bd)

    # Convert everything to 64 bit precision
    # core_tensor = np.array(core_tensor, dtype='float64')
    # guess_mats = np.array(guess_mats, dtype='float64')

    # Get LinearOperators (Scipy) implementing our left and right transfer 
    # operators on a flattened representation of our edge eigenmatrices
    flat_t_ops = [flat_t_op_fun(core_tensor, d) for d in ['left', 'right']]
    lin_ops = [LinearOperator(shape=(mat_size, mat_size), matvec=f_t_op) 
                              for f_t_op in flat_t_ops]

    # Unpack and flatten our original guess matrices
    guess_vecs = [g_mat.reshape((mat_size,)) for g_mat in guess_mats]

    # Use scipy.sparse.linalg.eigs to get the dominant eigenquantities
    num_eigs = 2 if check_gap else 1
    eig_output = [eigs(lin_op, k=num_eigs, sigma=guess_eig, v0=gv, 
                       which='LR', maxiter=1e4, tol=precision) 
                             for lin_op, gv in zip(lin_ops, guess_vecs)]
    all_eigs, all_vecs = [[tup[i] for tup in eig_output] for i in range(2)]

    # Normalize eigenvectors and convert everything to real data
    all_eigs, all_vecs = [onp.real_if_close(a) for a in [all_eigs, all_vecs]]
    all_vecs = [av.T / onp.linalg.norm(av) for av in all_vecs]

    # Check that eigenvalues of left and right transfer operators align
    assert onp.allclose(all_eigs[0], all_eigs[1]), f"all_eigs={all_eigs}"
    all_eigs = (all_eigs[0] + all_eigs[1]) / 2

    # Check that all the eigenvectors are actually eigenvectors
    assert all([[onp.isclose(onp.dot(vecs[i], lin_op(vecs[i])), e) 
                for vecs, lin_op in zip(all_vecs, lin_ops)] 
                for i, e in enumerate(all_eigs)])
    
    # Sort eigenvectors if they aren't properly sorted
    if abs(all_eigs[0]) < abs(all_eigs[1]):
        all_eigs = all_eigs[::-1]
        all_vecs = all_vecs[::-1]

    # Unpack top eigenvalue and eigenmatrices, make realness checks
    eig_val = onp.real_if_close(all_eigs[0])
    eig_mats = onp.stack([vecs[0].reshape(mat_shape) for vecs in all_vecs])
    assert onp.isrealobj(eig_val)
    if not onp.isrealobj(eig_mats):
        i_norms = [onp.linalg.norm(mat.imag) for mat in eig_mats]
        norms = [onp.linalg.norm(mat) for mat in eig_mats]
        percents = [round(100 * i_n/n) for i_n, n in zip(i_norms, norms)]
        warnings.warn("Eigenmatrices have imaginary components giving "
                     f"{percents[0]}% (left) and {percents[1]}% (right) "
                      "of the norm, which are getting truncated")
    eig_mats = np.array(eig_mats.real)

    # Check top eigenmatrices are Hermitian, top eigenvalue is positive
    assert all([onp.allclose(mat, mat.T) for mat in eig_mats])
    assert eig_val > 0

    if check_gap:
        eig_two = abs(all_eigs[1])
        assert eig_val >= eig_two
        spec_gap = (eig_val - eig_two) / eig_val

        return eig_val, eig_mats, spec_gap
    else:
        return eig_val, eig_mats
