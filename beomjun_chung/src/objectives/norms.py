"""Loss norms for gradient-based Tucker decomposition."""
import jax.numpy as jnp


def frobenius_loss(original: jnp.ndarray, reconstructed: jnp.ndarray) -> jnp.ndarray:
    """Frobenius (L2) loss: ||X - X_hat||_F^2"""
    diff = original - reconstructed
    return jnp.sum(diff * diff)


def l1_loss(original: jnp.ndarray, reconstructed: jnp.ndarray) -> jnp.ndarray:
    """L1 loss: ||X - X_hat||_1  (sum of absolute values)"""
    return jnp.sum(jnp.abs(original - reconstructed))


def huber_loss(original: jnp.ndarray, reconstructed: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss: L2 for small residuals, L1 for large residuals."""
    diff = jnp.abs(original - reconstructed)
    return jnp.sum(jnp.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta)))
