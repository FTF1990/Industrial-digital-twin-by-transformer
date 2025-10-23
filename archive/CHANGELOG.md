# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-06

### Added
- **V1 Static Transformer Model**: Lightweight architecture for static sensor-to-sensor mapping
- **V4 Hybrid Transformer Model**: Advanced hybrid temporal + static architecture
- **Modular Project Structure**: Organized code into models/, src/, notebooks/, examples/
- **Data Loading Pipeline**: Flexible SensorDataLoader class with preprocessing
- **Training Framework**: Complete ModelTrainer with early stopping and LR scheduling
- **Inference Engine**: ModelInference class with evaluation and visualization
- **Gradio Web Interface**: Interactive UI for training and inference
- **Jupyter Notebooks**: Comprehensive tutorials and examples
- **Documentation**: Detailed README, contributing guide, and API docs
- **Examples**: Quick start script for easy onboarding
- **Utility Functions**:
  - Temporal context data creation
  - IFD smoothing filters
  - Signal validation
  - Duplicate column handling

### Features
- GPU acceleration support
- Real-time training progress monitoring
- Model checkpoint saving/loading
- Configuration import/export (JSON)
- Multiple evaluation metrics (R², RMSE, MAE)
- Comprehensive visualization tools
- Batch processing support

### Documentation
- Complete README with architecture diagrams
- Installation instructions for Colab and local
- Quick start guide
- API documentation
- Contributing guidelines
- Data format specifications
- MIT License

### Technical Specifications
- Python 3.8+ support
- PyTorch 2.0+ backend
- Scikit-learn integration
- Gradio 4.0+ web interface
- Matplotlib/Seaborn visualization

## [Unreleased]

### Planned Features
- LSTM baseline models
- Attention visualization
- Real-time streaming support
- Docker containerization
- REST API for deployment
- Additional example datasets
- Hyperparameter optimization tools
- Multi-GPU training
- Model quantization
- ONNX export support

---

## Version History

### Version Numbering

- **Major version** (X.0.0): Breaking changes or major new features
- **Minor version** (1.X.0): New features, backward compatible
- **Patch version** (1.0.X): Bug fixes, backward compatible

### Development Timeline

- **2025-01-06**: Initial release (v1.0.0)
- Future releases will be documented here

---

For more details, see the [GitHub Releases](https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer/releases) page.
