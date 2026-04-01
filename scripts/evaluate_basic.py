"""Basic evaluation and sanity checks for steering vectors.

Usage:
    python scripts/evaluate_basic.py --model meta-llama/Meta-Llama-3-8B-Instruct --vectors vectors/
"""

import argparse
from pathlib import Path
import torch
from llsd import SteeringModel


TEST_PROMPTS = [
    "What is recursion?",
    "Explain quantum entanglement.",
    "How does memory work?",
    "What is consciousness?",
]


def main():
    parser = argparse.ArgumentParser(description="Basic steering evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vectors", type=str, required=True)
    parser.add_argument("--load-in-8bit", action="store_true")
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = SteeringModel.from_pretrained(
        args.model,
        load_in_8bit=args.load_in_8bit
    )
    
    print("Loading vectors...")
    model.load_vectors(args.vectors)
    
    print("\nRunning validation tests...\n")
    
    # Test 1: Alpha = 0 should match unsteered
    print("=" * 60)
    print("TEST 1: Alpha=0 matches unsteered output")
    print("=" * 60)
    model.set_divergence(0.0)
    for prompt in TEST_PROMPTS[:2]:
        output = model.generate(prompt, max_new_tokens=50)
        print(f"Prompt: {prompt}")
        print(f"Output: {output[:100]}...")
        print()
    
    # Test 2: Different alphas produce different outputs
    print("=" * 60)
    print("TEST 2: Varying alpha produces different outputs")
    print("=" * 60)
    for alpha in [0.0, 1.0, 2.0, 3.0]:
        model.set_divergence(alpha)
        output = model.generate(TEST_PROMPTS[0], max_new_tokens=50)
        print(f"Alpha={alpha}: {output[:100]}...")
        print()
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
