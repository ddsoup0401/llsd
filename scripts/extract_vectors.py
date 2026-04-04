"""Extract steering vectors from contrastive prompt pairs.

Usage:
    python scripts/extract_vectors.py --model meta-llama/Meta-Llama-3-8B-Instruct --pairs data/contrastive_pairs.jsonl --output vectors/
"""

import argparse
from pathlib import Path

import torch

from llsd import extract_steering_vectors
from llsd.dataset import load_contrastive_pairs


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors from contrastive pairs")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model identifier")
    parser.add_argument(
        "--pairs",
        type=str,
        default="data/contrastive_pairs.jsonl",
        help="Path to contrastive pairs JSONL file",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24, 28],
        help="Layer indices to extract from",
    )
    parser.add_argument(
        "--method", choices=["mean_diff", "pca"], default="mean_diff", help="Extraction method"
    )
    parser.add_argument(
        "--output", type=str, default="vectors/", help="Output directory for vectors"
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")

    args = parser.parse_args()

    # Load pairs
    print(f"Loading contrastive pairs from {args.pairs}")
    pairs = load_contrastive_pairs(args.pairs)
    print(f"Loaded {len(pairs)} pairs")

    # Extract vectors
    print(f"Extracting vectors from {args.model}")
    print(f"Target layers: {args.layers}")
    print(f"Method: {args.method}")

    vectors = extract_steering_vectors(
        model_name=args.model,
        pairs=pairs,
        layers=args.layers,
        method=args.method,
        load_in_8bit=args.load_in_8bit,
    )

    # Save vectors
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, vector in vectors.items():
        output_path = output_dir / f"layer_{layer_idx}_{args.method}.pt"
        torch.save(vector, output_path)
        print(f"Saved layer {layer_idx} vector to {output_path}")

    print("Extraction complete!")


if __name__ == "__main__":
    main()
