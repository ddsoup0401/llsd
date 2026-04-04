"""Inference-time steering logic and utilities."""

import torch
import torch.nn as nn

from llsd.hooks import SteeringInjector


class SteeringController:
    """High-level controller for managing multiple steering vectors.

    This class handles combining multiple steering directions and managing
    their injection hooks.

    Attributes:
        model: The language model
        vectors: Dictionary of named steering vector sets
        active_injectors: Currently active injector instances
    """

    def __init__(self, model: nn.Module):
        """Initialize steering controller.

        Args:
            model: The transformer model to steer
        """
        self.model = model
        self.vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.active_injectors: list[SteeringInjector] = []

    def add_vectors(self, name: str, vectors: dict[int, torch.Tensor]):
        """Register a set of steering vectors.

        Args:
            name: Identifier for this vector set
            vectors: Dict mapping layer indices to vectors
        """
        self.vectors[name] = vectors

    def activate_single(self, name: str, alpha: float):
        """Activate a single steering vector.

        Args:
            name: Name of vector set to activate
            alpha: Steering strength
        """
        # TODO: Implement single vector activation
        raise NotImplementedError("Single activation will be implemented in Phase 1")

    def activate_multiple(self, alphas: dict[str, float]):
        """Activate multiple steering vectors with different strengths.

        Args:
            alphas: Dict mapping vector names to strengths
        """
        # TODO: Implement multi-vector activation and combination
        raise NotImplementedError("Multi-vector activation will be implemented in Phase 2")

    def deactivate_all(self):
        """Remove all active steering."""
        for injector in self.active_injectors:
            injector.remove_hooks()
        self.active_injectors.clear()


def combine_vectors(
    vectors: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Combine multiple steering vectors with weighted sum.

    Args:
        vectors: Dict mapping names to vectors [hidden_dim]
        weights: Dict mapping names to weight coefficients

    Returns:
        Combined vector [hidden_dim]

    Example:
        >>> vectors = {"v1": torch.randn(4096), "v2": torch.randn(4096)}
        >>> weights = {"v1": 1.5, "v2": -0.5}
        >>> combined = combine_vectors(vectors, weights)
    """
    if not vectors:
        raise ValueError("Must provide at least one vector")

    if set(vectors.keys()) != set(weights.keys()):
        raise ValueError("Vector names and weight names must match")

    # Initialize with zeros
    result = torch.zeros_like(next(iter(vectors.values())))

    # Weighted sum
    for name, vector in vectors.items():
        result += weights[name] * vector

    return result


def interpolate_alpha(
    start_alpha: float,
    end_alpha: float,
    steps: int,
) -> list[float]:
    """Generate interpolated alpha values for gradual steering adjustment.

    Useful for exploring the alpha parameter space or creating smooth transitions.

    Args:
        start_alpha: Starting alpha value
        end_alpha: Ending alpha value
        steps: Number of interpolation steps

    Returns:
        List of interpolated alpha values

    Example:
        >>> alphas = interpolate_alpha(0.0, 3.0, steps=7)
        >>> alphas
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    """
    return [start_alpha + (end_alpha - start_alpha) * i / (steps - 1) for i in range(steps)]
