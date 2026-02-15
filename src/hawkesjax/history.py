# ruff: noqa: F821, F722, UP037
"""History computations for HawkesJAX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int


def history_marked_scan(
    t_events: Float[Array, "N"],
    marks: Int[Array, "N"],
    alpha: Float[Array, "D D K"],
    beta: Float[Array, "K"],
    t_start: Float[Array, ""],
) -> tuple[Float[Array, "N D K"], Float[Array, "N D K"]]:
    """Compute history tensors using ``jax.lax.scan``.

    Parameters
    ----------
    t_events : Array
        Event times of shape ``(N,)``.
    marks : Array
        Event marks of shape ``(N,)``.
    alpha : Array
        Branching ratios of shape ``(D, D, K)`` for
        ``g(t)=alpha*beta*exp(-beta t)``.
    beta : Array
        Mixture decay rates of shape ``(K,)``.
    t_start : Array
        Observation window start time.

    Returns
    -------
    h_minus : Array
        Histories just before each event, shape ``(N, D, K)``.
    h_plus : Array
        Histories just after each event, shape ``(N, D, K)``.

    """
    num_types = alpha.shape[0]
    num_mixtures = alpha.shape[2]
    h_plus0 = jnp.zeros((num_types, num_mixtures), dtype=alpha.dtype)
    alpha_scaled = alpha * beta[None, None, :]

    def step(
        carry: tuple[Float[Array, "D K"], Float[Array, ""]],
        inputs: tuple[Float[Array, ""], Int[Array, ""]],
    ) -> tuple[
        tuple[Float[Array, "D K"], Float[Array, ""]],
        tuple[Float[Array, "D K"], Float[Array, "D K"]],
    ]:
        h_plus_prev, t_prev = carry
        t_i, m_i = inputs
        dt = t_i - t_prev
        decay = jnp.exp(-beta * dt)
        h_minus = h_plus_prev * decay
        h_plus = h_minus + alpha_scaled[:, m_i, :]
        return (h_plus, t_i), (h_minus, h_plus)

    (_, _), (h_minus, h_plus) = jax.lax.scan(step, (h_plus0, t_start), (t_events, marks))
    return h_minus, h_plus


def history_marked_associative(
    t_events: Float[Array, "N"],
    marks: Int[Array, "N"],
    alpha: Float[Array, "D D K"],
    beta: Float[Array, "K"],
    t_start: Float[Array, ""],
) -> tuple[Float[Array, "N D K"], Float[Array, "N D K"]]:
    """Compute history tensors using ``jax.lax.associative_scan``.

    Parameters
    ----------
    t_events : Array
        Event times of shape ``(N,)``.
    marks : Array
        Event marks of shape ``(N,)``.
    alpha : Array
        Branching ratios of shape ``(D, D, K)`` for
        ``g(t)=alpha*beta*exp(-beta t)``.
    beta : Array
        Mixture decay rates of shape ``(K,)``.
    t_start : Array
        Observation window start time.

    Returns
    -------
    h_minus : Array
        Histories just before each event, shape ``(N, D, K)``.
    h_plus : Array
        Histories just after each event, shape ``(N, D, K)``.

    """
    if t_events.size == 0:
        num_types = alpha.shape[0]
        num_mixtures = alpha.shape[2]
        empty = jnp.zeros((0, num_types, num_mixtures), dtype=alpha.dtype)
        return empty, empty

    t_prev = jnp.concatenate([t_start[None], t_events[:-1]], axis=0)
    dt = t_events - t_prev
    decay = jnp.exp(-dt[:, None] * beta[None, :])

    alpha_scaled = alpha * beta[None, None, :]
    a_by_mark = jnp.take(alpha_scaled, marks, axis=1)
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
    return h_minus, h_plus
