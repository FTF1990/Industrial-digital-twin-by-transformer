# Gradio Interface Notebook

## Option 1: Use the Provided Gradio App (Simplest)

For a complete, ready-to-use Gradio interface, simply run:

```bash
python gradio_app.py
```

This provides the core functionality with a clean, simplified interface.

## Option 2: Full Cell 3 Interface (Advanced)

If you want to use your **original Cell 3 code with ALL features** from the `说明.txt` file, follow these steps:

### Step 1: Extract Cell 3 Code

From your original `说明.txt` file (lines 360-3013), copy the entire Cell 3 section.

### Step 2: Create Full Gradio Script

Create a new file `gradio_full_interface.py` in the project root:

```python
# At the top of the file, replace the inline model definitions with imports:

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

# Then paste the rest of your Cell 3 code below
# (starting from the global_state definition)
```

### Step 3: Use in Jupyter Notebook

To use the full interface in a Jupyter notebook:

1. **Create a new cell** in `notebooks/gradio_interface.ipynb`
2. **Copy your Cell 3 code**
3. **Update the imports** at the top:

```python
# Cell 1: Imports (update these lines)
from models.v1_transformer import CompactSensorTransformer
from models.v4_hybrid_transformer import HybridTemporalTransformer
from models.utils import *

# Cell 2: Rest of your Cell 3 code (unchanged)
# ... paste the rest of your original Cell 3 code here ...
```

## What's the Difference?

### Current `gradio_app.py` (Simplified)
- ✅ Easy to understand and modify
- ✅ Core functionality for V1 training
- ✅ Basic inference and visualization
- ✅ ~300 lines of code

### Original Cell 3 (Full-Featured)
- ✅ Complete V1 and V4 training
- ✅ Advanced configuration management
- ✅ Signal validation and error handling
- ✅ Real-time training progress
- ✅ Configuration import/export
- ✅ Comprehensive inference options
- ✅ ~2500 lines of code

## Quick Conversion Guide

Here's exactly how to convert your Cell 3 to use the modular structure:

### Before (Original Cell 3):
```python
# Cell 1: V1 model definition
class CompactSensorTransformer(nn.Module):
    # ... model code ...

# Cell 2: V4 model definition
class HybridTemporalTransformer(nn.Module):
    # ... model code ...

# Cell 3: Gradio interface
# ... all the gradio code ...
```

### After (Modular):
```python
# Just import the models instead!
from models.v1_transformer import CompactSensorTransformer
from models.v4_hybrid_transformer import HybridTemporalTransformer
from models.utils import *

# Your Cell 3 code works exactly the same!
# ... all the gradio code (unchanged) ...
```

## Pre-built Notebook

The `notebooks/gradio_interface.ipynb` file provides:

1. **Setup cells** - Imports and initialization
2. **Training functions** - Ready to use
3. **Configuration functions** - Import/export
4. **Callback functions** - Data loading and training
5. **Gradio interface** - Can be expanded with your full Cell 3

### To Add Your Full Cell 3:

1. Open `notebooks/gradio_interface.ipynb`
2. The imports are already set up correctly
3. Add your full Gradio interface code in the last cells
4. Run all cells to launch

## Alternative: Copy Original Code Directly

If you prefer to keep Cell 3 exactly as-is:

1. **Create new notebook**: `notebooks/gradio_original.ipynb`
2. **Cell 1**: Import models
   ```python
   from models.v1_transformer import CompactSensorTransformer
   from models.v4_hybrid_transformer import HybridTemporalTransformer
   from models.utils import *
   import torch, pandas, numpy, gradio, etc...
   ```
3. **Cell 2**: Paste your entire original Cell 3 code
4. **Done!** - Run and it works exactly the same

## File Locations

- **Simple version**: `gradio_app.py` (root directory)
- **Notebook template**: `notebooks/gradio_interface.ipynb`
- **Original code**: Your `说明.txt` lines 360-3013

## Need Help?

The full Cell 3 code from `说明.txt` is **100% compatible** with this modular structure. The only change needed is replacing Cell 1 & 2 model definitions with the import statements shown above.

---

**Recommendation**: Start with `gradio_app.py` to understand the structure, then add your full Cell 3 code when you need all advanced features.
