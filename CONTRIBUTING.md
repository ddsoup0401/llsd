# Contributing to LLSD

Thank you for your interest in contributing to LLSD! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/llsd.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install in dev mode: `pip install -e ".[dev]"`
6. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run linters: `black . && ruff check .`
5. Commit your changes: `git commit -m "Add feature X"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a pull request

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Follow PEP 8 naming conventions
- Add type hints where possible
- Write docstrings for all public functions/classes (Google style)

## Testing

- Write tests for new features in `tests/`
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage for new code

## Areas for Contribution

We especially welcome contributions in these areas:

### 1. New Steering Vector Types

Create new steering directions beyond divergent thinking:

- Humor vs. serious
- Formal vs. casual
- Technical vs. accessible
- Optimistic vs. pessimistic

To contribute:
- Add new contrastive pairs to `data/`
- Extract vectors and validate quality
- Submit with example outputs

### 2. Model Support

Extend LLSD to new architectures:

- Mistral, Mixtral
- Qwen, Yi
- Gemma, Phi

Requirements:
- Update `hooks.py` with layer access logic
- Test vector extraction and injection
- Document any architecture-specific quirks

### 3. Evaluation Metrics

Improve measurement of creativity/coherence:

- Implement lexical diversity metrics
- Add embedding-based novelty measures
- Create human evaluation frameworks

### 4. Performance Optimization

- Triton kernels for faster injection
- Batch processing for vector extraction
- Memory-efficient activation caching

### 5. Documentation

- Jupyter notebook tutorials
- Blog posts explaining the technique
- Video demos

## Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Include tests for new functionality
- Update documentation as needed
- Reference any related issues
- Ensure CI passes before requesting review

## Steering Vector Contribution Format

When contributing new steering vectors, please:

1. Include the contrastive pairs dataset (`.jsonl`)
2. Provide extraction metadata:
   - Model used
   - Layers targeted
   - Extraction method
   - Number of pairs used
3. Show 3-5 example outputs at different alpha values
4. Document optimal alpha range

Example directory structure:

```
vectors/humor/
├── pairs.jsonl
├── layer_16_mean_diff.pt
├── layer_20_mean_diff.pt
├── metadata.json
└── examples.md
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Tag maintainers for urgent matters

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the work, not the person
- Welcome newcomers

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for helping make LLSD better!
