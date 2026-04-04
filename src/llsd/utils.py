"""Utility functions for LLSD."""

from typing import Optional

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        torch.device instance
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_info(model) -> dict:
    """Extract useful information about a model.

    Args:
        model: HuggingFace model

    Returns:
        Dict with keys: n_layers, hidden_dim, device, dtype
    """
    return {
        "n_layers": len(model.model.layers),
        "hidden_dim": model.config.hidden_size,
        "device": next(model.parameters()).device,
        "dtype": next(model.parameters()).dtype,
    }


def estimate_vram_usage(
    model_size_b: float,
    quantization: Optional[str] = None,
) -> float:
    """Estimate VRAM usage for a model.

    Args:
        model_size_b: Model size in billions of parameters
        quantization: "8bit", "4bit", or None

    Returns:
        Estimated VRAM in GB

    Example:
        >>> estimate_vram_usage(8.0, quantization="8bit")
        8.5
    """
    if quantization == "8bit":
        return model_size_b * 1.0 + 0.5  # ~1GB per billion params + overhead
    elif quantization == "4bit":
        return model_size_b * 0.6 + 0.5
    else:
        return model_size_b * 2.0 + 1.0  # fp16: 2 bytes per param


def format_generation_output(
    output: str,
    prompt: str,
    remove_prompt: bool = True,
) -> str:
    """Format model generation output.

    Args:
        output: Raw model output
        prompt: Original prompt
        remove_prompt: Whether to remove prompt from output

    Returns:
        Formatted output string
    """
    if remove_prompt and output.startswith(prompt):
        return output[len(prompt) :].strip()
    return output.strip()
