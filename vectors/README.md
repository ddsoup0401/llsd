# Steering Vectors

This directory contains pre-computed steering vectors for different models and thinking styles.

## Directory Structure

```
vectors/
├── llama3-8b/
│   ├── divergent_layer_16.pt
│   ├── divergent_layer_20.pt
│   └── metadata.json
└── mistral-7b/
    └── ...
```

## Vector Format

Steering vectors are saved as PyTorch tensors (`.pt` files) with shape `[hidden_dim]`.

To load:

```python
import torch
vector = torch.load("divergent_layer_16.pt")
```

## Metadata

Each vector set should include a `metadata.json` with:

```json
{
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "extraction_method": "mean_diff",
  "layers": [16, 20],
  "num_pairs": 50,
  "optimal_alpha_range": [1.0, 2.5],
  "description": "Divergent thinking direction"
}
```

## Creating Your Own Vectors

See the main README for instructions on extracting custom steering vectors.
