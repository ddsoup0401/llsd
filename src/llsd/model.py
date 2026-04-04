"""Model loading and steering interface."""

from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llsd.hooks import SteeringInjector


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
        self.steering_vectors: dict[str, dict[int, torch.Tensor]] = {}
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
        model, tokenizer = load_model_with_quantization(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        return cls(model=model, tokenizer=tokenizer)

    def load_vectors(
        self,
        vectors: Union[str, dict[str, str], dict[int, torch.Tensor]],
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
        if isinstance(vectors, str):
            # Single file path
            loaded = torch.load(vectors, map_location="cpu")
            self.steering_vectors[name] = loaded
        elif isinstance(vectors, dict):
            # Check if it's a dict of paths or dict of tensors
            first_value = next(iter(vectors.values()))
            if isinstance(first_value, str):
                # Dict of paths
                for vec_name, path in vectors.items():
                    loaded = torch.load(path, map_location="cpu")
                    self.steering_vectors[vec_name] = loaded
            else:
                # Dict of tensors (already loaded)
                self.steering_vectors[name] = vectors
        else:
            raise ValueError("vectors must be a path string or dict")

        # Move vectors to device
        for vec_name in self.steering_vectors:
            if isinstance(self.steering_vectors[vec_name], dict):
                for layer_idx in self.steering_vectors[vec_name]:
                    self.steering_vectors[vec_name][layer_idx] = self.steering_vectors[vec_name][
                        layer_idx
                    ].to(self.device)

    def set_divergence(self, alpha: float):
        """Set divergence strength for the default steering vector.

        Args:
            alpha: Steering strength (0=normal, 1-2=creative, 3+=surreal)

        Example:
            >>> model.set_divergence(1.5)
            >>> output = model.generate("Explain recursion")
        """
        # Remove existing steering if any
        if self._injector is not None:
            self._injector.remove_hooks()

        # If alpha is 0, just remove steering
        if alpha == 0:
            self._injector = None
            return

        # Get default vectors
        if "default" not in self.steering_vectors:
            raise ValueError("No steering vectors loaded. Call load_vectors() first.")

        # Create new injector with specified alpha
        self._injector = SteeringInjector(
            model=self.model,
            steering_vectors=self.steering_vectors["default"],
            alpha=alpha,
        )

    def set_multi_steering(self, alphas: dict[str, float]):
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
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with steering hooks active
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def remove_steering(self):
        """Remove all steering hooks and reset to normal generation."""
        if self._injector is not None:
            self._injector.remove_hooks()
            self._injector = None


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
    # Configure quantization
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    elif load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16 if not (load_in_8bit or load_in_4bit) else None,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
