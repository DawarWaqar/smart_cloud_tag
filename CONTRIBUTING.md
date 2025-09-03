# Contributing to Smart Cloud Tag

Thank you for your interest in contributing to Smart Cloud Tag! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/smart_cloud_tag.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e .`

## Development Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart_cloud_tag.git
cd smart_cloud_tag

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/smart_cloud_tag
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom prompt templates
fix: resolve issue with Azure Blob Storage authentication
docs: update README with new installation instructions
```

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Update CHANGELOG.md if applicable
5. Submit a pull request with a clear description

### Testing

- Write tests for all new functionality
- Ensure existing tests continue to pass
- Aim for high test coverage
- Test with multiple cloud providers when possible

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Feature Requests

For feature requests, please:

- Check existing issues first
- Provide a clear description of the feature
- Explain the use case and benefits
- Consider implementation complexity

## Questions?

Feel free to open an issue for questions.
