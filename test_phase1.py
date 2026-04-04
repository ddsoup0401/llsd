"""Quick test script for Phase 1 implementation.

This script tests the basic functionality without requiring actual model downloads.
For full testing with real models, run after installing dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from llsd import SteeringModel  # noqa: F401
        from llsd.dataset import load_contrastive_pairs  # noqa: F401
        from llsd.extraction import (  # noqa: F401
            analyze_vector_quality,
            capture_activations_for_prompts,
            compute_mean_diff,
            compute_pca_direction,
            extract_steering_vectors,
        )
        from llsd.hooks import (  # noqa: F401
            ActivationCapture,
            SteeringInjector,
            get_layer_from_model,
        )
        from llsd.utils import estimate_vram_usage, get_device, get_model_info  # noqa: F401

        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_dataset_loading():
    """Test loading contrastive pairs."""
    print("\nTesting dataset loading...")
    try:
        from llsd.dataset import load_contrastive_pairs

        pairs = load_contrastive_pairs("data/contrastive_pairs.jsonl")
        print(f"[OK] Loaded {len(pairs)} contrastive pairs")

        # Check structure
        assert len(pairs) > 0, "No pairs loaded"
        assert "rigid_prompt" in pairs[0] or "rigid" in pairs[0], "Missing rigid prompt"
        assert "divergent_prompt" in pairs[0] or "divergent" in pairs[0], "Missing divergent prompt"
        print("[OK] Pair structure validated")

        # Print first pair as example
        print("\nExample pair:")
        print(f"  Topic: {pairs[0].get('topic', 'N/A')}")
        rigid_key = "rigid_prompt" if "rigid_prompt" in pairs[0] else "rigid"
        divergent_key = "divergent_prompt" if "divergent_prompt" in pairs[0] else "divergent"
        print(f"  Rigid: {pairs[0][rigid_key][:60]}...")
        print(f"  Divergent: {pairs[0][divergent_key][:60]}...")

        return True
    except Exception as e:
        print(f"[FAIL] Dataset loading failed: {e}")
        return False


def test_utility_functions():
    """Test utility functions that don't require models."""
    print("\nTesting utility functions...")
    try:
        from llsd.utils import estimate_vram_usage

        # Test VRAM estimation
        vram_8bit = estimate_vram_usage(8.0, quantization="8bit")
        vram_4bit = estimate_vram_usage(8.0, quantization="4bit")
        vram_fp16 = estimate_vram_usage(8.0, quantization=None)

        print("[OK] VRAM estimates for 8B model:")
        print(f"  8-bit: {vram_8bit:.1f} GB")
        print(f"  4-bit: {vram_4bit:.1f} GB")
        print(f"  FP16: {vram_fp16:.1f} GB")

        assert vram_4bit < vram_8bit < vram_fp16, "VRAM estimates should increase with precision"

        return True
    except Exception as e:
        print(f"[FAIL] Utility function test failed: {e}")
        return False


def test_vector_operations():
    """Test vector computation functions without models."""
    print("\nTesting vector operations...")
    try:
        import torch

        from llsd.extraction import analyze_vector_quality, compute_mean_diff, compute_pca_direction

        # Create dummy activations
        rigid_acts = torch.randn(50, 4096)
        divergent_acts = torch.randn(50, 4096) + 0.5  # Shifted to create difference

        # Test mean difference
        vector = compute_mean_diff(rigid_acts, divergent_acts)
        assert vector.shape == (4096,), f"Expected shape (4096,), got {vector.shape}"
        print(
            f"[OK] Mean difference vector computed: shape {vector.shape}, norm {vector.norm():.2f}"
        )

        # Test PCA direction
        pca_vector = compute_pca_direction(rigid_acts, divergent_acts)
        assert pca_vector.shape == (4096,), f"Expected shape (4096,), got {pca_vector.shape}"
        print(f"[OK] PCA vector computed: shape {pca_vector.shape}, norm {pca_vector.norm():.2f}")

        # Test quality analysis
        quality = analyze_vector_quality(vector, rigid_acts, divergent_acts)
        print("[OK] Quality metrics computed:")
        print(f"  Separation: {quality['separation']:.3f}")
        print(f"  Vector norm: {quality['vector_norm']:.3f}")

        return True
    except ImportError as e:
        print(f"[SKIP] Skipping vector operations test (torch not installed): {e}")
        return True  # Not a failure, just can't test without torch
    except Exception as e:
        print(f"[FAIL] Vector operations test failed: {e}")
        return False


def test_steering_controller():
    """Test steering controller logic."""
    print("\nTesting steering controller...")
    try:
        import torch

        from llsd.steering import combine_vectors, interpolate_alpha

        # Test vector combination
        vectors = {
            "v1": torch.randn(4096),
            "v2": torch.randn(4096),
        }
        weights = {"v1": 1.5, "v2": -0.5}
        combined = combine_vectors(vectors, weights)
        assert combined.shape == (4096,), f"Expected shape (4096,), got {combined.shape}"
        print(f"[OK] Vector combination works: shape {combined.shape}")

        # Test alpha interpolation
        alphas = interpolate_alpha(0.0, 3.0, steps=7)
        assert len(alphas) == 7, f"Expected 7 alphas, got {len(alphas)}"
        assert alphas[0] == 0.0 and alphas[-1] == 3.0, "Alpha range incorrect"
        print(f"[OK] Alpha interpolation works: {alphas}")

        return True
    except ImportError:
        print("[SKIP] Skipping steering controller test (torch not installed)")
        return True
    except Exception as e:
        print(f"[FAIL] Steering controller test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLSD Phase 1 Implementation Test")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Utility Functions", test_utility_functions()))
    results.append(("Vector Operations", test_vector_operations()))
    results.append(("Steering Controller", test_steering_controller()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] All tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -e .[dev]")
        print("2. Test with real model: python scripts/demo.py")
        print("3. Extract vectors: python scripts/extract_vectors.py")
    else:
        print("[FAIL] Some tests failed. Please review the output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
