# LLSD Project Structure

## Overview

LLSD (LLM on LSD) is a research project for inducing controlled divergent thinking in large language models via activation steering. This document provides an overview of the codebase structure.

## Repository Layout

```
llsd/
├── src/llsd/              # Main package source
├── scripts/               # Command-line utilities
├── data/                  # Contrastive prompt datasets
├── vectors/               # Steering vector storage
├── examples/              # Jupyter notebook demos
├── tests/                 # Unit tests (to be added)
├── .github/workflows/     # CI/CD configuration
└── docs/                  # Documentation (to be added)
```

## Core Modules

### `src/llsd/model.py`

High-level API for loading models with steering capabilities.

Key classes:
- `SteeringModel`: Main interface for steered generation
- `load_model_with_quantization()`: Model loading with memory optimization

### `src/llsd/hooks.py`

PyTorch hook infrastructure for activation capture and injection.

Key classes:
- `ActivationCapture`: Records activations during forward pass
- `SteeringInjector`: Modifies activations at inference time

### `src/llsd/extraction.py`

Steering vector extraction from contrastive datasets.

Key functions:
- `extract_steering_vectors()`: Full extraction pipeline
- `compute_mean_diff()`: Mean difference method
- `compute_pca_direction()`: PCA-based extraction

### `src/llsd/steering.py`

Inference-time steering logic and utilities.

Key classes:
- `SteeringController`: Manages multiple steering vectors
- `combine_vectors()`: Multi-vector combination

### `src/llsd/dataset.py`

Dataset utilities for contrastive prompt pairs.

Key functions:
- `load_contrastive_pairs()`: Load JSONL datasets
- `create_example_pairs()`: Generate example pairs
- `generate_prompt_template()`: Template-based generation

### `src/llsd/utils.py`

General utility functions.

## Implementation Status

**Current Status: Baseline Structure Complete**

All module files have been created with:
- Complete docstrings
- Type hints
- Method signatures
- Implementation marked as `NotImplementedError`

**Next Steps (Phase 1 Implementation):**

1. Implement `load_model_with_quantization()` in `model.py`
2. Implement hook registration in `hooks.py`
3. Implement extraction pipeline in `extraction.py`
4. Implement steering injection in `steering.py`
5. Create comprehensive test suite
6. Generate example steering vectors

## Testing Strategy

Tests will be organized as:

```
tests/
├── unit/
│   ├── test_hooks.py
│   ├── test_extraction.py
│   └── test_steering.py
├── integration/
│   ├── test_extraction_pipeline.py
│   └── test_steered_generation.py
└── fixtures/
    └── sample_vectors.pt
```

## Development Workflow

1. **Feature Development**: Work in feature branches
2. **Testing**: Ensure all tests pass (`pytest`)
3. **Linting**: Run `black . && ruff check .`
4. **Documentation**: Update docstrings and README
5. **PR**: Submit for review

## Model Architecture Support

Currently targeting:
- LLaMA-3-8B (primary)
- Future: Mistral-7B, Qwen, others

## Hardware Requirements

- **Development**: RTX 3080+ (8GB+ VRAM)
- **Production**: Any GPU with 8GB+ VRAM (8-bit quantization)
- **Minimum**: CPU-only (slow but functional)

## Performance Targets

- **Vector Extraction**: ~30 minutes for 50 pairs on consumer GPU
- **Inference Overhead**: <5% latency increase
- **Memory Overhead**: <500MB per steering vector set

## Open Questions

1. Optimal layer selection strategy across models
2. Best normalization approach for vectors
3. Transferability of vectors between model sizes
4. Effect of quantization on steering quality

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.
