# ruff: noqa: F821, F722, UP037
"""Likelihood computations for HawkesJAX."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from hawkesjax.quadrature import leggauss_nodes_weights
from hawkesjax.transforms import constrain, log_softplus_stable

if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from hawkesjax.config import HawkesSpec
    from hawkesjax.types import EventStream, Params, RawParams

    PhiFn = Callable[[Array], Array]
else:
    PhiFn = Callable[[jax.Array], jax.Array]


def integrate_interval_gauss_legendre(
    mu: Float[Array, "D"],
    h0: Float[Array, "D K"],
    beta: Float[Array, "K"],
    dt: Float[Array, ""],
    nodes: Float[Array, "Q"],
    weights: Float[Array, "Q"],
    phi: PhiFn,
) -> Float[Array, ""]:
    """Integrate total intensity over a single interval using Gauss-Legendre.

    Parameters
    ----------
    mu : Array
        Baseline parameters of shape ``(D,)``.
    h0 : Array
        History at the left endpoint, shape ``(D, K)``.
    beta : Array
        Mixture decay rates of shape ``(K,)``.
    dt : Array
        Interval length.
    nodes : Array
        Quadrature nodes of shape ``(Q,)``.
    weights : Array
        Quadrature weights of shape ``(Q,)``.
    phi : Callable
        Link function applied to the linear drive.

    Returns
    -------
    Array
        Scalar integral of total intensity over the interval.

    """
    half = dt * 0.5
    nodes_scaled = half * (nodes + 1.0)
    decay = jnp.exp(-nodes_scaled[:, None] * beta[None, :])
    h_at = h0[None, :, :] * decay[:, None, :]
    drive = mu[None, :] + jnp.sum(h_at, axis=-1)
    intensity = phi(drive)
    total_intensity = jnp.sum(intensity, axis=-1)
    return half * jnp.sum(weights * total_intensity)


def loglik_constrained_scan(
    params: Params,
    events: EventStream,
    phi: PhiFn,
    log_phi: PhiFn,
    nodes: Float[Array, "Q"],
    weights: Float[Array, "Q"],
) -> Float[Array, ""]:
    """Compute log-likelihood using a streaming scan backend.

    Parameters
    ----------
    params : Params
        Constrained Hawkes parameters (alpha is a branching ratio).
    events : EventStream
        Marked event stream.
    phi : Callable
        Link function applied to the linear drive.
    log_phi : Callable
        Log of the link function.
    nodes : Array
        Quadrature nodes of shape ``(Q,)``.
    weights : Array
        Quadrature weights of shape ``(Q,)``.

    Returns
    -------
    Array
        Scalar log-likelihood.

    """
    mu = params.mu
    alpha = params.alpha
    beta = params.beta

    num_types = alpha.shape[0]
    num_mixtures = alpha.shape[2]
    h_plus0 = jnp.zeros((num_types, num_mixtures), dtype=mu.dtype)
    ll0 = jnp.zeros((), dtype=mu.dtype)
    alpha_scaled = alpha * beta[None, None, :]

    def step(
        carry: tuple[Float[Array, ""], Float[Array, "D K"], Float[Array, ""]],
        inputs: tuple[Float[Array, ""], Array],
    ) -> tuple[tuple[Float[Array, ""], Float[Array, "D K"], Float[Array, ""]], None]:
        ll, h_plus, t_prev = carry
        t_i, m_i = inputs
        dt = t_i - t_prev
        integral = integrate_interval_gauss_legendre(mu, h_plus, beta, dt, nodes, weights, phi)
        h_minus = h_plus * jnp.exp(-beta * dt)
        log_intensity = log_phi(mu[m_i] + jnp.sum(h_minus[m_i, :]))
        ll = ll + log_intensity - integral
        h_plus = h_minus + alpha_scaled[:, m_i, :]
        return (ll, h_plus, t_i), None

    (ll, h_plus_last, t_last), _ = jax.lax.scan(
        step, (ll0, h_plus0, events.t_start), (events.t_events, events.marks)
    )
    tail_dt = events.t_end - t_last
    tail_integral = integrate_interval_gauss_legendre(
        mu, h_plus_last, beta, tail_dt, nodes, weights, phi
    )
    return ll - tail_integral


def loglik_constrained_associative(
    params: Params,
    events: EventStream,
    phi: PhiFn,
    log_phi: PhiFn,
    nodes: Float[Array, "Q"],
    weights: Float[Array, "Q"],
) -> Float[Array, ""]:
    """Compute log-likelihood using an associative_scan backend.

    Parameters
    ----------
    params : Params
        Constrained Hawkes parameters (alpha is a branching ratio).
    events : EventStream
        Marked event stream.
    phi : Callable
        Link function applied to the linear drive.
    log_phi : Callable
        Log of the link function.
    nodes : Array
        Quadrature nodes of shape ``(Q,)``.
    weights : Array
        Quadrature weights of shape ``(Q,)``.

    Returns
    -------
    Array
        Scalar log-likelihood.

    """
    mu = params.mu
    alpha = params.alpha
    beta = params.beta

    if events.t_events.size == 0:
        h0 = jnp.zeros((alpha.shape[0], alpha.shape[2]), dtype=mu.dtype)
        total = integrate_interval_gauss_legendre(
            mu, h0, beta, events.t_end - events.t_start, nodes, weights, phi
        )
        return -total

    t_prev = jnp.concatenate([events.t_start[None], events.t_events[:-1]], axis=0)
    dt = events.t_events - t_prev
    decay = jnp.exp(-dt[:, None] * beta[None, :])

    alpha_scaled = alpha * beta[None, None, :]
    a_by_mark = jnp.take(alpha_scaled, events.marks, axis=1)
    a_by_mark = jnp.swapaxes(a_by_mark, 0, 1)

    def combine(
        left: tuple[Float[Array, "D K"], Float[Array, "K"]],
        right: tuple[Float[Array, "D K"], Float[Array, "K"]],
    ) -> tuple[Float[Array, "D K"], Float[Array, "K"]]:
        a_left, b_left = left
        a_right, b_right = right
        b_right_broadcast = b_right[..., None, :] if b_right.ndim == a_left.ndim - 1 else b_right
        return a_right + b_right_broadcast * a_left, b_right * b_left

    h_plus, _ = jax.lax.associative_scan(combine, (a_by_mark, decay), axis=0)
    h_minus = h_plus - a_by_mark

    indices = jnp.arange(events.marks.shape[0])
    h_minus_mark = h_minus[indices, events.marks, :]
    log_intensity = log_phi(mu[events.marks] + jnp.sum(h_minus_mark, axis=-1))

    h_start = jnp.concatenate(
        [
            jnp.zeros((1, alpha.shape[0], alpha.shape[2]), dtype=mu.dtype),
            h_plus[:-1],
        ],
        axis=0,
    )

    def integrate_single(h0: Float[Array, "D K"], dt_i: Float[Array, ""]) -> Float[Array, ""]:
        return integrate_interval_gauss_legendre(mu, h0, beta, dt_i, nodes, weights, phi)

    interval_integrals = jax.vmap(integrate_single)(h_start, dt)

    tail_dt = events.t_end - events.t_events[-1]
    tail_integral = integrate_interval_gauss_legendre(
        mu, h_plus[-1], beta, tail_dt, nodes, weights, phi
    )

    return jnp.sum(log_intensity) - jnp.sum(interval_integrals) - tail_integral


def make_loglik_raw(
    spec: HawkesSpec,
) -> Callable[[RawParams, EventStream], Float[Array, ""]]:
    """Create a jittable log-likelihood function from a static spec.

    Parameters
    ----------
    spec : HawkesSpec
        Static configuration captured by the returned closure.

    Returns
    -------
    Callable
        Function ``loglik_raw(raw_params, events) -> scalar``.

    """
    nodes, weights = leggauss_nodes_weights(spec.num_quad, spec.dtype)
    phi = spec.phi
    if spec.log_phi is None:
        if spec.phi is jax.nn.softplus:
            log_phi = log_softplus_stable
        else:

            def log_phi(x: Array) -> Array:
                return jnp.log(phi(x))

    else:
        log_phi = spec.log_phi

    if spec.backend == "scan":
        loglik_fn = loglik_constrained_scan
    else:
        loglik_fn = loglik_constrained_associative

    def loglik_raw(raw_params: RawParams, events: EventStream) -> Float[Array, ""]:
        params = constrain(raw_params, spec)
        return loglik_fn(params, events, phi, log_phi, nodes, weights)

    return loglik_raw
