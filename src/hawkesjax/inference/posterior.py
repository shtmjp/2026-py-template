# ruff: noqa: F821, F722, UP037
"""Posterior utilities for HawkesJAX."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from hawkesjax.likelihood import make_loglik_raw
from hawkesjax.transforms import constrain
from hawkesjax.types import EventStream, Params, RawParams

if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from hawkesjax.config import HawkesSpec

    LogPriorFn = Callable[[Params], Array]
else:
    LogPriorFn = Callable[[Params], jax.Array]


def make_logposterior_flat(
    spec: HawkesSpec,
    events: EventStream,
    logprior_fn: LogPriorFn | None = None,
    bad_value: float = -1e30,
) -> Callable[[Float[Array, "P"]], Float[Array, ""]]:
    """Create a log-posterior function over a flat parameter vector.

    Parameters
    ----------
    spec : HawkesSpec
        Static configuration for likelihood evaluation.
    events : EventStream
        Marked event stream.
    logprior_fn : Callable or None, optional
        Function mapping constrained parameters to a log-prior value.
    bad_value : float, optional
        Finite value used when the log-posterior is not finite.

    Returns
    -------
    Callable
        Function ``logposterior(theta) -> scalar`` for flat parameters.

    """
    loglik_raw = make_loglik_raw(spec)

    raw_template = RawParams(
        mu=jnp.zeros((spec.num_types,), dtype=spec.dtype),
        alpha=jnp.zeros((spec.num_types, spec.num_types, spec.num_mixtures), dtype=spec.dtype),
        beta_delta_raw=jnp.zeros((spec.num_mixtures,), dtype=spec.dtype),
    )
    _, unravel = ravel_pytree(raw_template)

    def logposterior(theta: Float[Array, "P"]) -> Float[Array, ""]:
        raw_params = unravel(theta)
        loglik = loglik_raw(raw_params, events)
        if logprior_fn is None:
            logprior = jnp.zeros((), dtype=loglik.dtype)
        else:
            params = constrain(raw_params, spec)
            logprior = logprior_fn(params)
        value = loglik + logprior
        return jnp.where(jnp.isfinite(value), value, jnp.asarray(bad_value, dtype=value.dtype))

    return logposterior
