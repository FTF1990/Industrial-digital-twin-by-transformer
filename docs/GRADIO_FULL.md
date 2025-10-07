# Gradio Interface - Full Version

This folder would contain the complete Gradio interface from your original Cell 3.

## Using the Full Gradio Interface

You have two options to access the full-featured Gradio interface:

### Option 1: Use the Simplified Gradio App (Recommended for Quick Start)

```bash
python gradio_app.py
```

This provides core functionality for:
- Data loading
- V1 model training
- Basic inference and visualization

### Option 2: Use the Complete Original Interface

To use the full interface with all advanced features from Cell 3:

1. **Copy your original Cell 3 code** to a new file named `gradio_full.py` in the project root

2. **Update the imports** at the top of the file to use the modular structure:

```python
# Replace the original Cell 1 and Cell 2 imports with:
from models.v1_transformer import CompactSensorTransformer
from models.v4_hybrid_transformer import HybridTemporalTransformer
from models.utils import (
    create_temporal_context_data,
    apply_ifd_smoothing,
    handle_duplicate_columns,
    get_available_signals,
    validate_signal_exclusivity_v1,
    validate_signal_exclusivity_v4
)
```

3. **Run the full interface**:

```bash
python gradio_full.py
```

## Features in Full Interface

The complete Gradio interface (Cell 3) includes:

### ✅ Data Loading Tab
- Upload CSV files or use pre-loaded DataFrame
- Automatic duplicate column handling
- Complete signal list display
- Data validation and statistics

### ✅ V1 Model Training Tab
- Full configuration options
- Signal selection with validation
- Real-time training progress
- Configuration import/export (JSON)
- Training history visualization

### ✅ V4 Model Training Tab
- Temporal context configuration
- Hybrid branch selection
- IFD smoothing options
- Advanced hyperparameter tuning
- Configuration management

### ✅ Inference Tab
- Multi-model selection
- Custom data range selection
- Multi-signal visualization
- Comprehensive metrics display
- Interactive plots

## Converting Original Code

If you want to integrate your original Cell 3 code into this project structure:

1. The model definitions (Cell 1 & 2) are already in `models/`
2. Training functions can use `src/trainer.py`
3. Keep the Gradio UI code from Cell 3 as-is
4. Just update the imports to use the modular structure

## Example: Minimal Changes Required

```python
# OLD (Cell 3):
# All code in one cell with Cell 1 and Cell 2 models defined above

# NEW (modular):
from models.v1_transformer import CompactSensorTransformer
from models.v4_hybrid_transformer import HybridTemporalTransformer
from models.utils import *
# ... rest of your Gradio code stays the same
```

---

**Need Help?**
- Check `examples/quick_start.py` for a complete working example
- See `notebooks/train_and_inference.ipynb` for detailed tutorials
- Refer to the main README.md for full documentation
