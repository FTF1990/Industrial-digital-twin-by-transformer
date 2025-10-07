# Contributing to Industrial Digital Twin by Transformer

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in [GitHub Issues](https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)

### Submitting Changes

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
   cd Industrial-digital-twin-by-transformer
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test Your Changes**
   ```bash
   # Run the quick start example
   python examples/quick_start.py

   # Test with notebooks
   jupyter notebook notebooks/train_and_inference.ipynb
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add feature X" # or "fix: resolve issue Y"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a PR on GitHub with a clear description.

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Maximum line length: 100 characters
- Use type hints where appropriate

```python
def process_sensor_data(
    data: np.ndarray,
    sensor_names: List[str],
    window_size: int = 5
) -> np.ndarray:
    """
    Process sensor data with windowing.

    Args:
        data: Input sensor measurements
        sensor_names: List of sensor identifiers
        window_size: Size of smoothing window

    Returns:
        Processed sensor data
    """
    # Implementation
    pass
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md for new features
- Add examples for new functionality

### Git Commit Messages

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example:
```
feat: add LSTM baseline model for comparison

- Implement LSTM encoder-decoder architecture
- Add training script in examples/
- Update documentation with performance comparison
```

## Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# (Optional) Install development tools
pip install pytest black flake8 mypy
```

## Areas for Contribution

We welcome contributions in:

### 🚀 Features

- [ ] Additional model architectures (LSTM, GRU baselines)
- [ ] Attention visualization tools
- [ ] Real-time streaming data support
- [ ] Model compression and optimization
- [ ] Multi-GPU training support

### 📊 Data & Examples

- [ ] Example datasets with different industrial scenarios
- [ ] Data augmentation techniques
- [ ] Preprocessing pipelines
- [ ] Synthetic data generation

### 📚 Documentation

- [ ] Additional tutorials
- [ ] API reference documentation
- [ ] Video tutorials
- [ ] Deployment guides

### 🧪 Testing

- [ ] Unit tests for models
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Edge case handling

### 🛠️ Tools

- [ ] Hyperparameter tuning scripts
- [ ] Model comparison tools
- [ ] Deployment utilities
- [ ] Monitoring dashboards

## Questions?

Feel free to:
- Open an issue for discussion
- Join our community (if available)
- Contact the maintainers

## Code of Conduct

Be respectful and constructive in all interactions. We're building this together!

---

Thank you for contributing to Industrial Digital Twin by Transformer! 🙏
