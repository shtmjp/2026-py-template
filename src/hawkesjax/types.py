# ruff: noqa: F821, F722, UP037
"""PyTree dataclasses for HawkesJAX inputs and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import jax

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class EventStream:
    """Marked event stream for a single observation window.

    Attributes
    ----------
    t_events : Array
        Event times of shape ``(N,)``.
    marks : Array
        Integer marks of shape ``(N,)`` in ``[0, D)``.
    t_start : Array
        Observation window start time.
    t_end : Array
        Observation window end time.

    """

    t_events: Float[Array, "N"]
    marks: Int[Array, "N"]
    t_start: Float[Array, ""]
    t_end: Float[Array, ""]

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        """Flatten the event stream into PyTree children.

        Returns
        -------
        children : tuple[Array, ...]
            PyTree children arrays.
        aux_data : None
            No auxiliary data is used.

        """
        return (self.t_events, self.marks, self.t_start, self.t_end), None

    @classmethod
    def tree_unflatten(cls: type[Self], _aux_data: None, children: tuple[Array, ...]) -> Self:
        """Reconstruct an event stream from PyTree children.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : tuple[Array, ...]
            PyTree children arrays.

        Returns
        -------
        EventStream
            Reconstructed event stream.

        """
        t_events, marks, t_start, t_end = children
        return cls(t_events=t_events, marks=marks, t_start=t_start, t_end=t_end)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class RawParams:
    """Unconstrained parameters for optimization or sampling.

    Attributes
    ----------
    mu : Array
        Baseline parameters of shape ``(D,)``.
    alpha : Array
        Branching ratios of shape ``(D, D, K)`` for the exponential kernel.
    beta_delta_raw : Array
        Unconstrained deltas of shape ``(K,)`` for ordered betas.

    """

    mu: Float[Array, "D"]
    alpha: Float[Array, "D D K"]
    beta_delta_raw: Float[Array, "K"]

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        """Flatten raw parameters into PyTree children.

        Returns
        -------
        children : tuple[Array, ...]
            PyTree children arrays.
        aux_data : None
            No auxiliary data is used.

        """
        return (self.mu, self.alpha, self.beta_delta_raw), None

    @classmethod
    def tree_unflatten(cls: type[Self], _aux_data: None, children: tuple[Array, ...]) -> Self:
        """Reconstruct raw parameters from PyTree children.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : tuple[Array, ...]
            PyTree children arrays.

        Returns
        -------
        RawParams
            Reconstructed raw parameters.

        """
        mu, alpha, beta_delta_raw = children
        return cls(mu=mu, alpha=alpha, beta_delta_raw=beta_delta_raw)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Params:
    """Constrained parameters with ordered mixture rates.

    Attributes
    ----------
    mu : Array
        Baseline parameters of shape ``(D,)``.
    alpha : Array
        Branching ratios of shape ``(D, D, K)`` for the exponential kernel.
    beta : Array
        Ordered beta values of shape ``(K,)``.

    """

    mu: Float[Array, "D"]
    alpha: Float[Array, "D D K"]
    beta: Float[Array, "K"]

    def tree_flatten(self) -> tuple[tuple[Array, ...], None]:
        """Flatten constrained parameters into PyTree children.

        Returns
        -------
        children : tuple[Array, ...]
            PyTree children arrays.
        aux_data : None
            No auxiliary data is used.

        """
        return (self.mu, self.alpha, self.beta), None

    @classmethod
    def tree_unflatten(cls: type[Self], _aux_data: None, children: tuple[Array, ...]) -> Self:
        """Reconstruct constrained parameters from PyTree children.

        Parameters
        ----------
        _aux_data : None
            Unused auxiliary data.
        children : tuple[Array, ...]
            PyTree children arrays.

        Returns
        -------
        Params
            Reconstructed constrained parameters.

        """
        mu, alpha, beta = children
        return cls(mu=mu, alpha=alpha, beta=beta)
