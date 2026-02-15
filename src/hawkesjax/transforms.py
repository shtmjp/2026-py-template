# ruff: noqa: F821, UP037
"""Parameter transforms for HawkesJAX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from hawkesjax.types import Params, RawParams

if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from hawkesjax.config import HawkesSpec


def log_softplus_stable(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute log(softplus(x)) with a stable approximation for large negatives.

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    Array
        Elementwise ``log(softplus(x))`` with a stable tail approximation.

    """
    threshold = jnp.asarray(-20.0, dtype=x.dtype)
    softplus_val = jax.nn.softplus(x)
    approx = x
    return jnp.where(x < threshold, approx, jnp.log(softplus_val))


def softplus_inverse(y: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute the inverse of softplus with numerical stability.

    Parameters
    ----------
    y : Array
        Positive input array.

    Returns
    -------
    Array
        Values ``x`` such that ``softplus(x) = y``.

    """
    tiny = jnp.finfo(y.dtype).tiny
    y_safe = jnp.maximum(y, tiny)
    return y_safe + jnp.log(-jnp.expm1(-y_safe))


def ordered_beta_from_delta_raw(
    beta_delta_raw: Float[Array, "K"],
    beta_min: float,
    beta_eps: float,
) -> Float[Array, "K"]:
    """Map unconstrained deltas to ordered beta values.

    Parameters
    ----------
    beta_delta_raw : Array
        Unconstrained deltas of shape ``(K,)``.
    beta_min : float
        Lower bound for beta values.
    beta_eps : float
        Positive offset added to softplus deltas.

    Returns
    -------
    Array
        Ordered beta values of shape ``(K,)``.

    """
    delta = jax.nn.softplus(beta_delta_raw) + beta_eps
    return jnp.asarray(beta_min, dtype=beta_delta_raw.dtype) + jnp.cumsum(delta)


def beta_to_beta_delta_raw(
    beta: Float[Array, "K"],
    beta_min: float,
    beta_eps: float,
) -> Float[Array, "K"]:
    """Convert ordered beta values back to unconstrained deltas.

    Parameters
    ----------
    beta : Array
        Ordered beta values of shape ``(K,)``.
    beta_min : float
        Lower bound used in the forward transform.
    beta_eps : float
        Positive offset used in the forward transform.

    Returns
    -------
    Array
        Unconstrained deltas of shape ``(K,)``.

    """
    beta_min_val = jnp.asarray(beta_min, dtype=beta.dtype)
    delta_first = beta[:1] - beta_min_val
    delta_rest = beta[1:] - beta[:-1]
    delta = jnp.concatenate([delta_first, delta_rest], axis=0)
    return softplus_inverse(delta - beta_eps)


def constrain(raw_params: RawParams, spec: HawkesSpec) -> Params:
    """Convert raw parameters to constrained parameters.

    Parameters
    ----------
    raw_params : RawParams
        Unconstrained parameters.
    spec : HawkesSpec
        Configuration containing beta constraints.

    Returns
    -------
    Params
        Constrained parameters with ordered beta values.

    """
    beta = ordered_beta_from_delta_raw(
        raw_params.beta_delta_raw, beta_min=spec.beta_min, beta_eps=spec.beta_eps
    )
    return Params(mu=raw_params.mu, alpha=raw_params.alpha, beta=beta)
