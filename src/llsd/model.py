"""Model loading and steering interface."""

from typing import Dict, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class SteeringModel:
    """Wrapper around a language model with activation steering capabilities.
    
    This class provides a high-level interface for loading models with quantization
    support and applying steering vectors at inference time.
    
    Attributes:
        model: The underlying HuggingFace model
        tokenizer: The tokenizer for the model
        device: Device the model is loaded on
        steering_vectors: Dictionary of loaded steering vectors by name
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: Optional[str] = None,
    ):
        """Initialize a SteeringModel.
        
        Args:
            model: Pre-loaded HuggingFace model
            tokenizer: Tokenizer for the model
            device: Device to use (defaults to model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.steering_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        self._injector = None
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        **kwargs,
    ) -> "SteeringModel":
        """Load a model with optional quantization.
        
        Args:
            model_name: HuggingFace model identifier
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            device_map: Device mapping strategy
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
            
        Returns:
            SteeringModel instance with loaded model
            
        Example:
            >>> model = SteeringModel.from_pretrained(
            ...     "meta-llama/Meta-Llama-3-8B-Instruct",
            ...     load_in_8bit=True
            ... )
        """
        # TODO: Implement model loading with quantization config
        raise NotImplementedError("Model loading will be implemented in Phase 1")
        
    def load_vectors(
        self,
        vectors: Union[str, Dict[str, str], Dict[int, torch.Tensor]],
        name: str = "default",
    ):
        """Load steering vectors from file or dictionary.
        
        Args:
            vectors: Path to .pt file, dict of paths, or dict of tensors
            name: Name to store vectors under
            
        Example:
            >>> model.load_vectors("vectors/divergent.pt", name="divergent")
            >>> model.load_vectors({
            ...     "divergent": "vectors/divergent.pt",
            ...     "poetic": "vectors/poetic.pt"
            ... })
        """
        # TODO: Implement vector loading
        raise NotImplementedError("Vector loading will be implemented in Phase 1")
        
    def set_divergence(self, alpha: float):
        """Set divergence strength for the default steering vector.
        
        Args:
            alpha: Steering strength (0=normal, 1-2=creative, 3+=surreal)
            
        Example:
            >>> model.set_divergence(1.5)
            >>> output = model.generate("Explain recursion")
        """
        # TODO: Implement single-vector steering
        raise NotImplementedError("Steering will be implemented in Phase 1")
        
    def set_multi_steering(self, alphas: Dict[str, float]):
        """Set multiple steering vectors simultaneously.
        
        Args:
            alphas: Dictionary mapping vector names to strengths
            
        Example:
            >>> model.set_multi_steering({
            ...     "divergent": 1.5,
            ...     "poetic": 0.8,
            ...     "technical": -0.5
            ... })
        """
        # TODO: Implement multi-vector steering
        raise NotImplementedError("Multi-vector steering will be implemented in Phase 2")
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text with steering applied.
        
        Args:
            prompt: Input text to complete
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text (prompt + completion)
            
        Example:
            >>> output = model.generate(
            ...     "What is consciousness?",
            ...     max_new_tokens=200
            ... )
        """
        # TODO: Implement generation with steering
        raise NotImplementedError("Generation will be implemented in Phase 1")
        
    def remove_steering(self):
        """Remove all steering hooks and reset to normal generation."""
        # TODO: Implement hook removal
        raise NotImplementedError("Hook management will be implemented in Phase 1")


def load_model_with_quantization(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model with optional quantization.
    
    This is a lower-level function for loading models. Most users should use
    SteeringModel.from_pretrained() instead.
    
    Args:
        model_name: HuggingFace model identifier
        load_in_8bit: Use 8-bit quantization (~8GB VRAM for 8B models)
        load_in_4bit: Use 4-bit quantization (~5GB VRAM for 8B models)
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        >>> model, tokenizer = load_model_with_quantization(
        ...     "meta-llama/Meta-Llama-3-8B-Instruct",
        ...     load_in_8bit=True
        ... )
    """
    # TODO: Implement quantization config and model loading
    raise NotImplementedError("Model loading will be implemented in Phase 1")
