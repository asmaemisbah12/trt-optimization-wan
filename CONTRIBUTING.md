# Contributing to Wan TensorRT Project

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Set up your development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

Format code with Black:
```bash
black src/ scripts/
```

### Testing

Run tests before submitting:
```bash
pytest tests/
```

Add tests for new features in the `tests/` directory.

### Commit Messages

Use clear, descriptive commit messages:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Example:
```
feat: add CUDA graph support for TRT runtime

- Implement graph capture in TRTInference class
- Add warmup and replay methods
- Update benchmarking to test graph performance

Closes #42
```

## Areas for Contribution

### High Priority

1. **Full TRT Pipeline Integration**: Complete the integration of TRT engines into the diffusion pipeline
2. **INT8/FP8 Quantization**: Implement calibration and quantization workflows
3. **Multi-GPU Support**: Add data parallel and model parallel support
4. **Video I/O**: Robust video saving and loading utilities

### Medium Priority

1. **Documentation**: Expand tutorials and API documentation
2. **Testing**: Increase test coverage
3. **Optimizations**: Profile and optimize bottlenecks
4. **CLI Tools**: Enhance command-line interfaces

### Good First Issues

- Add more quality metrics (FVD, IS, etc.)
- Improve logging and progress bars
- Add configuration validation
- Write integration tests

## Pull Request Process

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, logical commits

3. **Update documentation** as needed (README, docstrings, etc.)

4. **Add tests** for new functionality

5. **Run the test suite**:
   ```bash
   pytest tests/
   black src/ scripts/ --check
   ```

6. **Push to your fork** and submit a pull request

7. **Describe your changes** in the PR description:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
   - Related issues

## Code Review

- Be patient and responsive to feedback
- All PRs require at least one approval
- Address review comments promptly
- Keep PRs focused and reasonably sized

## Reporting Issues

When reporting bugs:
- Use a clear, descriptive title
- Describe steps to reproduce
- Include error messages and logs
- Specify your environment (OS, Python version, CUDA version, etc.)
- Include minimal code to reproduce the issue

## Feature Requests

When requesting features:
- Explain the use case and motivation
- Describe the desired behavior
- Provide examples if possible
- Consider implementation complexity

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Give constructive feedback
- Focus on what is best for the project

## Questions?

- Open an issue for general questions
- Use discussions for broader topics
- Check existing issues before creating new ones

Thank you for contributing! 🚀

