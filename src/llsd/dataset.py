"""Dataset utilities for contrastive prompt pairs."""

from typing import List, Dict, Optional
import json
from pathlib import Path


def load_contrastive_pairs(path: str) -> List[Dict[str, str]]:
    """Load contrastive prompt pairs from JSONL file.
    
    Args:
        path: Path to .jsonl file containing prompt pairs
        
    Returns:
        List of dicts with keys: "id", "topic", "rigid_prompt", "divergent_prompt"
        
    Example:
        >>> pairs = load_contrastive_pairs("data/contrastive_pairs.jsonl")
        >>> pairs[0]["rigid_prompt"]
        'Explain recursion in precise technical terms.'
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def save_contrastive_pairs(pairs: List[Dict[str, str]], path: str):
    """Save contrastive prompt pairs to JSONL file.
    
    Args:
        pairs: List of prompt pair dictionaries
        path: Path to save .jsonl file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')


def create_example_pairs() -> List[Dict[str, str]]:
    """Generate example contrastive prompt pairs for testing.
    
    Returns:
        List of example prompt pairs covering different categories
        
    Example:
        >>> pairs = create_example_pairs()
        >>> len(pairs)
        20
    """
    examples = [
        {
            "id": "recursion_01",
            "topic": "recursion",
            "rigid_prompt": "Explain recursion in precise technical terms.",
            "divergent_prompt": "Explain recursion as if mirrors were teaching themselves to dream.",
            "category": "cs_concepts"
        },
        {
            "id": "sorting_01",
            "topic": "sorting",
            "rigid_prompt": "Describe how sorting algorithms work.",
            "divergent_prompt": "Describe sorting as if scattered thoughts were finding their way home.",
            "category": "cs_concepts"
        },
        {
            "id": "memory_01",
            "topic": "computer_memory",
            "rigid_prompt": "Explain computer memory in technical terms.",
            "divergent_prompt": "Explain memory as if the computer were a city and RAM were its collective consciousness.",
            "category": "cs_concepts"
        },
        # Add more examples here
    ]
    return examples


def validate_pairs(pairs: List[Dict[str, str]]) -> List[str]:
    """Validate that prompt pairs have required fields.
    
    Args:
        pairs: List of prompt pair dicts to validate
        
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    required_fields = {"id", "rigid_prompt", "divergent_prompt"}
    
    for i, pair in enumerate(pairs):
        missing = required_fields - set(pair.keys())
        if missing:
            errors.append(f"Pair {i}: Missing fields {missing}")
            
    return errors


def generate_prompt_template(
    topic: str,
    style: str = "rigid",
    template_type: str = "explanation",
) -> str:
    """Generate a prompt from a topic using templates.
    
    Args:
        topic: The concept to explain
        style: "rigid" or "divergent"
        template_type: Type of prompt ("explanation", "comparison", "analogy")
        
    Returns:
        Generated prompt string
        
    Example:
        >>> generate_prompt_template("quantum entanglement", style="divergent")
        'Explain quantum entanglement using unexpected metaphors and associations.'
    """
    templates = {
        "rigid": {
            "explanation": f"Explain {topic} in precise technical terms.",
            "comparison": f"Compare and contrast {topic} using formal definitions.",
            "analogy": f"Describe {topic} using literal, direct language.",
        },
        "divergent": {
            "explanation": f"Explain {topic} using unexpected metaphors and associations.",
            "comparison": f"Compare {topic} as if they were characters in different dreams.",
            "analogy": f"Describe {topic} through the lens of sensory experience and imagination.",
        }
    }
    
    return templates[style][template_type]
