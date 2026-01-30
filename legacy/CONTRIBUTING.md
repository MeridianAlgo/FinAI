# Contributing to Fin.AI

Thank you for your interest in contributing to Fin.AI! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Enhancements

We welcome feature requests! Please open an issue with:

- A clear description of the enhancement
- Why this would be useful
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

#### Pull Request Guidelines

- Keep changes focused and atomic
- Write clear commit messages
- Add tests for new features
- Update README.md if needed
- Ensure all tests pass
- Follow the existing code style

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/FinAI.git
cd FinAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Formatting

We use `black` for code formatting:

```bash
black fin_ai/ train.py generate.py
```

### Linting

Run linting before submitting:

```bash
flake8 fin_ai/ train.py generate.py
```

## Testing

Run tests before submitting:

```bash
pytest tests/
```

Add tests for new features:

```python
# tests/test_new_feature.py
def test_new_feature():
    # Your test here
    assert True
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update configuration examples if needed

## Commit Messages

Write clear, descriptive commit messages:

```
Add feature: Brief description

- Detailed point 1
- Detailed point 2
```

Good examples:
- `Add support for custom tokenizers`
- `Fix memory leak in data loader`
- `Update README with new examples`

Bad examples:
- `fix bug`
- `update`
- `changes`

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] GPU support for faster training
- [ ] Model quantization for inference
- [ ] Additional dataset integrations
- [ ] Performance optimizations

### Medium Priority
- [ ] Web UI for generation
- [ ] Fine-tuning utilities
- [ ] Model evaluation metrics
- [ ] Better error handling

### Documentation
- [ ] More usage examples
- [ ] Tutorial notebooks
- [ ] API documentation
- [ ] Video tutorials

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behavior includes:**
- Being respectful and inclusive
- Accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behavior includes:**
- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Enforcement

Violations may result in temporary or permanent ban from the project.

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Fin.AI!
