"""LLSD: Controlled Divergent Thinking via Activation Steering.

Induce controlled divergent thinking in LLMs via activation steering at inference time.
"""

from llsd.extraction import extract_steering_vectors
from llsd.hooks import ActivationCapture, SteeringInjector
from llsd.model import SteeringModel

__version__ = "0.1.0"
__all__ = [
    "SteeringModel",
    "extract_steering_vectors",
    "ActivationCapture",
    "SteeringInjector",
]
