"""PyTorch hooks for activation capture and steering injection."""

from typing import Callable

import torch
import torch.nn as nn


class ActivationCapture:
    """Capture activations from specific layers during forward pass.

    This class registers forward hooks on transformer layers to record their
    outputs during model execution. Useful for extracting steering vectors.

    Attributes:
        activations: Dictionary mapping layer indices to captured tensors
        hooks: List of registered hook handles
    """

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        """Initialize activation capture on specified layers.

        Args:
            model: The transformer model (e.g., LLaMAForCausalLM)
            layer_indices: List of layer indices to capture (0-indexed)

        Example:
            >>> capture = ActivationCapture(model, layers=[12, 16, 20])
            >>> output = model(**inputs)
            >>> acts = capture.activations[16]  # Get layer 16 activations
        """
        self.model = model
        self.layer_indices = layer_indices
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        # TODO: Implement hook registration for model.model.layers[i]
        raise NotImplementedError("Hook registration will be implemented in Phase 1")

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook function for a specific layer.

        Args:
            layer_idx: The layer index this hook is for

        Returns:
            Hook function with signature (module, input, output) -> None
        """

        def hook(module, input, output):
            # Extract hidden states from output tuple
            # output[0] is typically [batch, seq_len, hidden_dim]
            # TODO: Implement activation extraction and storage
            pass

        return hook

    def clear(self):
        """Clear captured activations to free memory."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_last_token_activations(self, layer_idx: int) -> torch.Tensor:
        """Get activations for the last token position.

        Args:
            layer_idx: Layer to get activations from

        Returns:
            Tensor of shape [batch, hidden_dim]
        """
        # TODO: Implement last token extraction
        raise NotImplementedError("Token position selection will be implemented in Phase 1")


class SteeringInjector:
    """Inject steering vectors into model activations at inference time.

    This class modifies the forward pass to add steering vectors to hidden states,
    enabling controlled divergent thinking without changing model weights.

    Attributes:
        alpha: Current steering strength
        steering_vectors: Vectors to inject at each layer
        hooks: List of registered hook handles
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vectors: dict[int, torch.Tensor],
        alpha: float = 1.0,
        injection_mode: str = "all_tokens",
    ):
        """Initialize steering injection.

        Args:
            model: The transformer model
            steering_vectors: Dict mapping layer indices to steering vectors [hidden_dim]
            alpha: Initial steering strength
            injection_mode: Where to inject ("all_tokens", "last_token", "prefill_only")

        Example:
            >>> vectors = {16: torch.load("vector_layer16.pt")}
            >>> injector = SteeringInjector(model, vectors, alpha=1.5)
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.alpha = alpha
        self.injection_mode = injection_mode
        self.hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for steering injection."""
        # TODO: Implement injection hooks
        raise NotImplementedError("Injection hooks will be implemented in Phase 1")

    def _make_steering_hook(self, vector: torch.Tensor) -> Callable:
        """Create a hook that adds a steering vector to activations.

        Args:
            vector: Steering vector of shape [hidden_dim]

        Returns:
            Hook function that modifies outputs
        """

        def hook(module, input, output):
            # Modify hidden states: h' = h + alpha * vector
            # TODO: Implement steering injection
            # Handle different injection modes (all_tokens vs last_token)
            pass

        return hook

    def set_alpha(self, alpha: float):
        """Update steering strength without re-registering hooks.

        Args:
            alpha: New steering strength
        """
        self.alpha = alpha

    def remove_hooks(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def get_layer_from_model(model: nn.Module, layer_idx: int) -> nn.Module:
    """Get a specific transformer layer from a model.

    Args:
        model: The model (e.g., LlamaForCausalLM)
        layer_idx: Layer index (0-indexed)

    Returns:
        The transformer layer module

    Example:
        >>> layer = get_layer_from_model(model, 16)
    """
    # TODO: Implement layer access (e.g., model.model.layers[layer_idx])
    raise NotImplementedError("Layer access will be implemented in Phase 1")


def normalize_vector(vector: torch.Tensor, norm_type: str = "l2") -> torch.Tensor:
    """Normalize a steering vector.

    Args:
        vector: Vector to normalize [hidden_dim]
        norm_type: Normalization type ("l2", "unit", "none")

    Returns:
        Normalized vector
    """
    if norm_type == "l2":
        return vector / (torch.norm(vector) + 1e-8)
    elif norm_type == "unit":
        return vector / vector.abs().max()
    elif norm_type == "none":
        return vector
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
