# Getting Started with Industrial Digital Twin by Transformer

Welcome! This guide will help you get started with the project in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- Basic understanding of machine learning
- Industrial sensor data in CSV format (or use our synthetic data generator)

## Installation

### Option 1: Google Colab (Fastest)

Perfect for quick experimentation without local setup.

```python
# In a Colab notebook
!git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
%cd Industrial-digital-twin-by-transformer
!pip install -r requirements.txt

# You're ready to go!
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (5 Minutes)

### 1. Prepare Your Data

Place your CSV file in `data/raw/`:

```
data/raw/my_sensor_data.csv
```

Your CSV should look like:
```
timestamp,temp_1,pressure_1,flow_1,quality_1,...
2025-01-01 00:00:00,23.5,101.3,45.2,0.95,...
2025-01-01 00:00:01,23.6,101.4,45.1,0.94,...
```

### 2. Run Quick Start Example

```bash
python examples/quick_start.py
```

This script will:
- Load your data (or create synthetic data)
- Train an SST model
- Evaluate performance
- Save the trained model

### 3. Explore with Jupyter

```bash
jupyter notebook notebooks/train_and_inference.ipynb
```

Follow the step-by-step tutorial to:
- Load and visualize your data
- Train SST and HST models
- Compare performance
- Export results

### 4. Use Web Interface (Optional)

```bash
python gradio_app.py
```

Access at `http://127.0.0.1:7860` for an interactive experience.

## Common Usage Patterns

### Pattern 1: Basic Prediction

```python
from models.static_transformer import StaticSensorTransformer
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer

# Load data
loader = SensorDataLoader(data_path='data/raw/my_data.csv')

# Define sensors
boundary_signals = ['temp_1', 'pressure_1', 'flow_1']
target_signals = ['quality_1', 'efficiency_1']

# Prepare data
data = loader.prepare_data(boundary_signals, target_signals)

# Create and train model
model = StaticSensorTransformer(
    num_boundary_sensors=3,
    num_target_sensors=2
)

trainer = ModelTrainer(model)
train_loader, val_loader = loader.create_dataloaders(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val']
)

history = trainer.train(train_loader, val_loader)
```

### Pattern 2: Time-Dependent Predictions

```python
from models.hybrid_transformer import HybridSensorTransformer
from models.utils import create_temporal_context_data

# Create temporal context
X_train_ctx, y_train_ctx, _ = create_temporal_context_data(
    data['X_train'],
    data['y_train'],
    context_window=5
)

# Train HST model
model = HybridSensorTransformer(
    num_boundary_sensors=3,
    num_target_sensors=2,
    use_temporal=True,
    context_window=5
)

# Rest of training is the same
```

### Pattern 3: Inference and Evaluation

```python
from src.inference import ModelInference

# Create inference engine
inference = ModelInference(
    model=model,
    scaler_X=data['scaler_X'],
    scaler_y=data['scaler_y']
)

# Make predictions
predictions = inference.predict(new_sensor_data)

# Evaluate
metrics = inference.evaluate(X_test, y_test, target_signals)
inference.print_metrics(metrics)

# Visualize
fig = inference.plot_predictions(X_test, y_test, signal_indices=[0, 1])
```

## Choosing Between SST and HST

### Use SST (StaticSensorTransformer) when:
- ✅ Sensor relationships are stable over time
- ✅ You want faster training and inference
- ✅ You have limited computational resources
- ✅ System operates in steady-state

### Use HST (HybridSensorTransformer) when:
- ✅ Sensors show temporal dependencies
- ✅ System behavior changes over time
- ✅ You need higher accuracy
- ✅ Some sensors have dynamic patterns

### Not sure? Start with SST, then try HST if:
- SST R² score < 0.85
- Residuals show temporal patterns
- Target sensors have known time-dependencies

## Configuration Files

Save your setup for reproducibility:

```python
import json

config = {
    "model_type": "SST",
    "signals": {
        "boundary": boundary_signals,
        "target": target_signals
    },
    "training": {
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.001
    }
}

with open('my_experiment.json', 'w') as f:
    json.dump(config, f, indent=2)
```

Load it later:

```python
with open('my_experiment.json', 'r') as f:
    config = json.load(f)
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or model dimensions
```python
config = {
    'batch_size': 32,  # Reduce from 64
    'd_model': 64      # Reduce from 128
}
```

### Issue: Poor model performance (R² < 0.7)

**Solutions**:
1. Check data quality (missing values, outliers)
2. Increase model capacity (more layers, larger d_model)
3. Try HST if data has temporal patterns
4. Collect more training data
5. Verify sensor selection (boundary vs target)

### Issue: Training is slow

**Solutions**:
1. Enable GPU: Verify `torch.cuda.is_available()`
2. Increase batch size (if memory allows)
3. Reduce model complexity
4. Use SST instead of HST

### Issue: "Signal exclusivity error"

**Solution**: Ensure boundary and target signals don't overlap
```python
# Wrong:
boundary = ['temp_1', 'pressure_1']
target = ['temp_1', 'flow_1']  # temp_1 appears in both!

# Correct:
boundary = ['temp_1', 'pressure_1']
target = ['flow_1', 'quality_1']
```

## Next Steps

1. **Experiment**: Try different signal combinations
2. **Tune**: Adjust hyperparameters in configs/
3. **Compare**: Train both SST and HST, compare metrics
4. **Deploy**: Save best model for production use
5. **Contribute**: Share improvements with the community!

## Learning Resources

- **Notebooks**: `notebooks/train_and_inference.ipynb` - Complete tutorial
- **Examples**: `examples/quick_start.py` - Working code
- **Documentation**: `README.md` - Full project overview
- **API Reference**: Docstrings in all modules
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`

## Getting Help

- 📖 Read the [README](../README.md)
- 💬 Open an [Issue](https://github.com/YOUR_USERNAME/Industrial-digital-twin-by-transformer/issues)
- 📧 Contact maintainers (see README)

## Example: Complete Workflow

Here's a complete example from data to predictions:

```python
import torch
from models.static_transformer import StaticSensorTransformer
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer
from src.inference import ModelInference

# 1. Load data
loader = SensorDataLoader(data_path='data/raw/sensors.csv')

# 2. Configure
boundary = ['temp_1', 'temp_2', 'pressure_1']
target = ['quality', 'efficiency']

# 3. Prepare
data = loader.prepare_data(boundary, target)

# 4. Create model
model = StaticSensorTransformer(
    num_boundary_sensors=len(boundary),
    num_target_sensors=len(target)
)

# 5. Train
trainer = ModelTrainer(model)
train_loader, val_loader = loader.create_dataloaders(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val']
)
history = trainer.train(train_loader, val_loader)

# 6. Evaluate
inference = ModelInference(
    model, data['scaler_X'], data['scaler_y']
)
X_test = data['scaler_X'].inverse_transform(data['X_test'])
y_test = data['scaler_y'].inverse_transform(data['y_test'])
metrics = inference.evaluate(X_test, y_test, target)

# 7. Save
trainer.save_model('models/saved/my_model.pth')

print("Done! Model performance:")
print(f"Average R²: {metrics['Overall']['R2']:.4f}")
```

---

**Ready to build your digital twin? Let's go! 🚀**
