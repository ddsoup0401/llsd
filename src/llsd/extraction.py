"""Steering vector extraction from contrastive datasets."""

from typing import Dict, List, Tuple, Literal
import torch
from sklearn.decomposition import PCA
import numpy as np


def extract_steering_vectors(
    model_name: str,
    pairs: List[Dict[str, str]],
    layers: List[int],
    method: Literal["mean_diff", "pca"] = "mean_diff",
    n_components: int = 1,
    load_in_8bit: bool = True,
    **model_kwargs,
) -> Dict[int, torch.Tensor]:
    """Extract steering vectors from contrastive prompt pairs.
    
    This is the main entry point for steering vector extraction. It loads a model,
    runs contrastive prompts, captures activations, and computes steering directions.
    
    Args:
        model_name: HuggingFace model identifier
        pairs: List of dicts with "rigid" and "divergent" prompt keys
        layers: Layer indices to extract vectors from
        method: Extraction method ("mean_diff" or "pca")
        n_components: Number of PCA components (only for method="pca")
        load_in_8bit: Use 8-bit quantization
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        Dictionary mapping layer indices to steering vectors [hidden_dim]
        
    Example:
        >>> pairs = [
        ...     {"rigid": "Explain recursion technically",
        ...      "divergent": "Explain recursion as mirrors dreaming"}
        ... ]
        >>> vectors = extract_steering_vectors(
        ...     "meta-llama/Meta-Llama-3-8B-Instruct",
        ...     pairs=pairs,
        ...     layers=[12, 16, 20]
        ... )
    """
    # TODO: Implement full extraction pipeline:
    # 1. Load model
    # 2. Setup ActivationCapture
    # 3. Run rigid and divergent prompts
    # 4. Compute steering vectors
    # 5. Return per-layer vectors
    raise NotImplementedError("Extraction pipeline will be implemented in Phase 1")


def compute_mean_diff(
    rigid_activations: torch.Tensor,
    divergent_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute steering vector as mean difference.
    
    V_steering = mean(divergent_acts) - mean(rigid_acts)
    
    Args:
        rigid_activations: Activations from rigid prompts [n_samples, hidden_dim]
        divergent_activations: Activations from divergent prompts [n_samples, hidden_dim]
        
    Returns:
        Steering vector [hidden_dim]
        
    Example:
        >>> rigid = torch.randn(50, 4096)
        >>> divergent = torch.randn(50, 4096)
        >>> vector = compute_mean_diff(rigid, divergent)
        >>> vector.shape
        torch.Size([4096])
    """
    if rigid_activations.shape != divergent_activations.shape:
        raise ValueError("Rigid and divergent activations must have same shape")
    
    return divergent_activations.mean(dim=0) - rigid_activations.mean(dim=0)


def compute_pca_direction(
    rigid_activations: torch.Tensor,
    divergent_activations: torch.Tensor,
    n_components: int = 1,
) -> torch.Tensor:
    """Compute steering direction using PCA on activation differences.
    
    This method finds the principal component of the difference vectors,
    which can be more robust than mean difference.
    
    Args:
        rigid_activations: Activations from rigid prompts [n_samples, hidden_dim]
        divergent_activations: Activations from divergent prompts [n_samples, hidden_dim]
        n_components: Number of principal components (typically 1)
        
    Returns:
        Steering vector (first principal component) [hidden_dim]
        
    Example:
        >>> rigid = torch.randn(50, 4096)
        >>> divergent = torch.randn(50, 4096)
        >>> vector = compute_pca_direction(rigid, divergent)
    """
    if rigid_activations.shape != divergent_activations.shape:
        raise ValueError("Rigid and divergent activations must have same shape")
    
    # Compute differences
    diffs = (divergent_activations - rigid_activations).cpu().numpy()
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(diffs)
    
    # Return first component
    return torch.tensor(pca.components_[0], dtype=torch.float32)


def capture_activations_for_prompts(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    token_position: Literal["last", "mean"] = "last",
) -> Dict[int, torch.Tensor]:
    """Capture activations from a list of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to run
        layers: Layer indices to capture from
        token_position: Which token position to use ("last" or "mean")
        
    Returns:
        Dict mapping layer indices to activation tensors [n_prompts, hidden_dim]
        
    Example:
        >>> prompts = ["Prompt 1", "Prompt 2"]
        >>> acts = capture_activations_for_prompts(
        ...     model, tokenizer, prompts, layers=[16]
        ... )
    """
    # TODO: Implement prompt batching and activation capture
    raise NotImplementedError("Activation capture will be implemented in Phase 1")


def save_vectors(vectors: Dict[int, torch.Tensor], path: str):
    """Save steering vectors to disk.
    
    Args:
        vectors: Dict mapping layer indices to vectors
        path: Path to save .pt file
        
    Example:
        >>> save_vectors(vectors, "my_vectors.pt")
    """
    torch.save(vectors, path)


def load_vectors(path: str) -> Dict[int, torch.Tensor]:
    """Load steering vectors from disk.
    
    Args:
        path: Path to .pt file
        
    Returns:
        Dict mapping layer indices to vectors
        
    Example:
        >>> vectors = load_vectors("my_vectors.pt")
    """
    return torch.load(path, map_location="cpu")


def analyze_vector_quality(
    vector: torch.Tensor,
    rigid_acts: torch.Tensor,
    divergent_acts: torch.Tensor,
) -> Dict[str, float]:
    """Analyze the quality and separability of a steering vector.
    
    Computes metrics like projection magnitudes and separation to assess
    whether the vector captures a meaningful direction.
    
    Args:
        vector: Steering vector [hidden_dim]
        rigid_acts: Rigid activations [n_samples, hidden_dim]
        divergent_acts: Divergent activations [n_samples, hidden_dim]
        
    Returns:
        Dict with quality metrics:
            - mean_rigid_proj: Mean projection of rigid acts onto vector
            - mean_divergent_proj: Mean projection of divergent acts
            - separation: Difference between projections
            - vector_norm: L2 norm of the vector
    """
    # TODO: Implement quality analysis
    raise NotImplementedError("Vector analysis will be implemented in Phase 1")
