import jax.numpy as jnp
import jax


def unfold(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    """
    Mode-n matricization (unfolding). Kolda & Bader Section 2.4.
    X_(n): shape (I_n, I_1 * ... * I_{n-1} * I_{n+1} * ... * I_N)
    Column ordering: indices of other modes in original order, last cycles fastest.
    """
    ndim = tensor.ndim
    order = [mode] + [i for i in range(ndim) if i != mode]
    return jnp.transpose(tensor, order).reshape(tensor.shape[mode], -1)


def fold(matrix: jnp.ndarray, mode: int, shape: tuple) -> jnp.ndarray:
    """Inverse of unfold: reconstruct tensor from mode-n unfolding."""
    ndim = len(shape)
    order = [mode] + [i for i in range(ndim) if i != mode]
    unfolded_shape = tuple(shape[i] for i in order)
    tensor = matrix.reshape(unfolded_shape)
    inv_order = [0] * ndim
    for new_pos, old_pos in enumerate(order):
        inv_order[old_pos] = new_pos
    return jnp.transpose(tensor, inv_order)


def mode_n_product(tensor: jnp.ndarray, matrix: jnp.ndarray, mode: int) -> jnp.ndarray:
    """
    n-mode matrix product: Y = X ×_n U. Kolda & Bader Section 2.5.
    tensor: (I_1, ..., I_N), matrix: (J, I_n)
    result: (..., I_{n-1}, J, I_{n+1}, ...)
    """
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(matrix @ unfold(tensor, mode), mode, tuple(new_shape))


def multi_mode_product(tensor: jnp.ndarray, matrices: list) -> jnp.ndarray:
    """
    Tucker reconstruction: G ×_1 A ×_2 B ×_3 C ...
    matrices[n] has shape (J_n, I_n), applied to mode n.
    """
    result = tensor
    for mode, matrix in enumerate(matrices):
        result = mode_n_product(result, matrix, mode)
    return result


def khatri_rao(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Khatri-Rao product (column-wise Kronecker). Kolda & Bader Section 2.6.
    A: (I, R), B: (J, R) → result: (I*J, R)
    result[i*J + j, r] = A[i, r] * B[j, r]
    """
    I, R = A.shape
    J = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(I * J, R)
