"""Configuration objects for HawkesJAX."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array

    PhiFn = Callable[[Array], Array]
else:
    PhiFn = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True, slots=True)
class HawkesSpec:
    """Static configuration for building Hawkes likelihood functions.

    Parameters
    ----------
    num_types : int
        Number of event types ``D``.
    num_mixtures : int
        Number of exponential mixture components ``K``.
    num_quad : int, optional
        Number of Gauss-Legendre quadrature points ``Q``.
    beta_min : float, optional
        Lower bound added to ordered beta values.
    beta_eps : float, optional
        Positive offset added to softplus deltas to keep betas increasing.
    phi : Callable
        Link function applied to the linear drive.
    log_phi : Callable or None, optional
        Log of the link function. If ``None``, a default is used.
    backend : {"scan", "associative"}, optional
        Backend used by the log-likelihood evaluation.
    dtype : jnp.dtype, optional
        Floating dtype used for created constants.

    """

    num_types: int
    num_mixtures: int
    num_quad: int = 16
    beta_min: float = 1e-3
    beta_eps: float = 1e-6
    phi: PhiFn = jax.nn.softplus
    log_phi: PhiFn | None = None
    backend: Literal["scan", "associative"] = "scan"
    dtype: jnp.dtype = jnp.float32

    def __post_init__(self) -> None:
        """Validate configuration values for consistency.

        Raises
        ------
        ValueError
            If any parameter value is invalid.

        """
        if self.num_types <= 0:
            raise ValueError
        if self.num_mixtures <= 0:
            raise ValueError
        if self.num_quad <= 0:
            raise ValueError
        if self.beta_min <= 0:
            raise ValueError
        if self.beta_eps <= 0:
            raise ValueError
        if self.backend not in ("scan", "associative"):
            raise ValueError
