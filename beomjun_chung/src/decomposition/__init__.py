from typing import NamedTuple, List
import jax.numpy as jnp


class TuckerResult(NamedTuple):
    core: jnp.ndarray
    factors: List[jnp.ndarray]


class CPResult(NamedTuple):
    weights: jnp.ndarray
    factors: List[jnp.ndarray]
