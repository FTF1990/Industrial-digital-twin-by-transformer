# Project Structure

This document provides a detailed overview of the project organization.

## Directory Tree

```
Industrial-digital-twin-by-transformer/
│
├── models/                          # Core model implementations
│   ├── __init__.py                  # Package initialization
│   ├── static_transformer.py        # SST (StaticSensorTransformer) model
│   ├── hybrid_transformer.py        # HST (HybridSensorTransformer) model
│   ├── utils.py                     # Utility functions for models
│   └── saved/                       # Directory for saved model checkpoints
│       └── .gitkeep
│
├── src/                             # Source code for training and inference
│   ├── __init__.py                  # Package initialization
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── trainer.py                   # Training pipeline and utilities
│   └── inference.py                 # Inference engine and evaluation
│
├── notebooks/                       # Jupyter notebooks for tutorials
│   ├── train_and_inference.ipynb    # Main tutorial notebook
│   └── gradio_interface.ipynb       # Gradio interface (to be created)
│
├── examples/                        # Example scripts
│   └── quick_start.py               # Quick start example
│
├── data/                            # Data storage
│   ├── raw/                         # Raw sensor data (CSV files)
│   │   └── .gitkeep
│   ├── processed/                   # Processed data (optional)
│   │   └── .gitkeep
│   └── README.md                    # Data format guide
│
├── configs/                         # Configuration files
│   ├── example_sst_config.json       # Example SST configuration
│   └── example_hst_config.json       # Example HST configuration
│
├── docs/                            # Additional documentation
│   └── GRADIO_FULL.md              # Full Gradio interface guide
│
├── tests/                           # Unit tests (to be implemented)
│   └── .gitkeep
│
├── gradio_app.py                    # Gradio web application
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation script
├── README.md                        # Main project documentation
├── CONTRIBUTING.md                  # Contribution guidelines
├── CHANGELOG.md                     # Version history
├── LICENSE                          # MIT License
└── .gitignore                       # Git ignore rules
```

## Module Descriptions

### `models/`

Contains the core Transformer model implementations.

- **`static_transformer.py`**: StaticSensorTransformer (SST) class
  - Lightweight architecture for static sensor mapping
  - Uses positional encoding for sensor locations
  - Best for systems with stable sensor relationships

- **`hybrid_transformer.py`**: HybridSensorTransformer (HST) class
  - Dual-branch architecture (temporal + static)
  - Handles time-dependent sensor behaviors
  - Fusion layer combines both approaches

- **`utils.py`**: Helper functions
  - `create_temporal_context_data()`: Creates time-series windows
  - `apply_ifd_smoothing()`: Applies signal smoothing
  - `validate_signal_exclusivity_v1/v4()`: Input validation
  - `handle_duplicate_columns()`: Data preprocessing

### `src/`

Source code for the training and inference pipeline.

- **`data_loader.py`**: SensorDataLoader class
  - Loads CSV files or DataFrames
  - Handles data splitting and scaling
  - Creates PyTorch DataLoaders
  - Methods:
    - `load_data()`: Load from CSV
    - `prepare_data()`: Split and scale
    - `create_dataloaders()`: Create PyTorch loaders

- **`trainer.py`**: ModelTrainer class
  - Unified training loop for SST and HST
  - Early stopping and LR scheduling
  - Model checkpoint management
  - Methods:
    - `train()`: Complete training loop
    - `train_epoch()`: Single epoch
    - `validate()`: Validation step
    - `save_model()` / `load_model()`: Persistence

- **`inference.py`**: ModelInference class
  - Model evaluation and predictions
  - Visualization tools
  - Metrics calculation
  - Methods:
    - `predict()`: Make predictions
    - `evaluate()`: Calculate metrics
    - `plot_predictions()`: Visualize results

### `notebooks/`

Interactive Jupyter notebooks for learning and experimentation.

- **`train_and_inference.ipynb`**: Complete tutorial
  - Step-by-step guide
  - SST and HST model training
  - Evaluation and comparison
  - Visualization examples

### `examples/`

Standalone Python scripts demonstrating usage.

- **`quick_start.py`**: Minimal working example
  - Complete end-to-end workflow
  - Synthetic data generation
  - Model training and evaluation

### `data/`

Data storage with clear organization.

- **`raw/`**: Original CSV files (not tracked by git)
- **`processed/`**: Preprocessed data (optional)
- **`README.md`**: Data format specifications

### `configs/`

JSON configuration files for reproducibility.

- **`example_sst_config.json`**: SST model configuration
- **`example_hst_config.json`**: HST model configuration

Contains all model parameters, training settings, and signal selections.

### `docs/`

Additional documentation beyond README.

- **`GRADIO_FULL.md`**: Guide for using complete Gradio interface

## File Naming Conventions

- **Python modules**: lowercase_with_underscores.py
- **Classes**: CamelCase (e.g., `SensorDataLoader`)
- **Functions**: lowercase_with_underscores (e.g., `create_temporal_context_data`)
- **Constants**: UPPERCASE_WITH_UNDERSCORES (e.g., `DEFAULT_BATCH_SIZE`)
- **Configuration files**: example_*.json or config_*.json

## Import Structure

### Importing Models

```python
from models.static_transformer import StaticSensorTransformer
from models.hybrid_transformer import HybridSensorTransformer
```

### Importing Utilities

```python
from models.utils import create_temporal_context_data, apply_ifd_smoothing
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer
from src.inference import ModelInference
```

## Adding New Features

### Adding a New Model

1. Create `models/v5_your_model.py`
2. Implement as `nn.Module` subclass
3. Add to `models/__init__.py`
4. Create example in `examples/`
5. Update documentation

### Adding New Utilities

1. Add function to `models/utils.py` or `src/`
2. Add docstring with type hints
3. Add example usage in notebooks
4. Update CHANGELOG.md

### Adding Tests

1. Create test file in `tests/`
2. Name it `test_*.py`
3. Use pytest framework
4. Run with `pytest tests/`

## Configuration Management

### Using Configuration Files

```python
import json

# Load configuration
with open('configs/example_sst_config.json', 'r') as f:
    config = json.load(f)

# Use in training
model = StaticSensorTransformer(
    num_boundary_sensors=len(config['signals']['boundary']),
    num_target_sensors=len(config['signals']['target']),
    **config['model_architecture']
)

trainer = ModelTrainer(model, config=config['training'])
```

## Best Practices

1. **Keep models/ pure**: Only model definitions, no training code
2. **Use src/ for pipelines**: Training, data loading, inference
3. **Document everything**: Docstrings, comments, README updates
4. **Version configurations**: Save configs with trained models
5. **Track experiments**: Use CHANGELOG.md and git tags
6. **Test before committing**: Run examples and notebooks

## Dependencies Between Modules

```
gradio_app.py
    ↓
┌───────────────┐
│  models/      │ ←─── Core ML models (no dependencies)
└───────────────┘
        ↓
┌───────────────┐
│  src/         │ ←─── Uses models/, independent utilities
└───────────────┘
        ↓
┌───────────────┐
│  notebooks/   │ ←─── Uses models/ and src/
│  examples/    │
└───────────────┘
```

## Questions?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.
