# ruff: noqa: F821, UP037
"""Quadrature utilities for HawkesJAX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Array, Float


def leggauss_nodes_weights(
    num_quad: int,
    dtype: jnp.dtype,
) -> tuple[Float[Array, "Q"], Float[Array, "Q"]]:
    """Return Gauss-Legendre nodes and weights as JAX arrays.

    Parameters
    ----------
    num_quad : int
        Number of quadrature points.
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    nodes : Array
        Quadrature nodes of shape ``(Q,)``.
    weights : Array
        Quadrature weights of shape ``(Q,)``.

    """
    nodes, weights = np.polynomial.legendre.leggauss(num_quad)
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype)
