# Quick Start Guide - Inverse Control Optimization

**Get started with inverse optimization in 5 minutes!**

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Navigate to project directory
cd Industrial-digital-twin-by-transformer

# Install all dependencies
pip install -r requirements.txt
```

**New dependencies for v2.0:**
- `filterpy>=1.4.5` - Kalman filtering
- `plotly>=5.0.0` - Interactive visualization

### 2. Verify Installation

```bash
python test_imports.py
```

Expected output:
```
✅ All tests passed successfully!
```

---

## 🚀 Three Ways to Use Inverse Optimization

### Option 1: Quick Demo (Recommended for First Time)

Run the built-in demonstration with synthetic data:

```bash
python quick_start_inverse_opt.py
```

This will show:
- ✅ Single-objective optimization example
- ✅ Multi-objective Pareto frontier
- ✅ Constraint management demo

**Time**: ~30 seconds

---

### Option 2: Interactive Web Interface

Launch the Gradio web application:

```bash
# Method 1: Direct command
python gradio_apps/inverse_control_interface.py

# Method 2: Use launch script
./launch_inverse_opt_ui.sh
```

Access at: **http://localhost:7861**

#### Using the Web Interface:

**Step 1: Load Model (Tab 0)**
1. Enter Stage1 model path (e.g., `saved_models/stage1_model.pth`)
2. Click "Load Model"
3. Enter data file path (e.g., `data/data.csv`)
4. Click "Load Data"
5. Run basic inference to verify

**Step 2: Optimize (Tab 1)**
1. **Select targets** from dropdown (e.g., "combustor_acceleration")
2. Click "Create Targets Table"
3. **Edit target bias** (e.g., -10 for -10% reduction)
4. **Select variable inputs** (e.g., "fuel_flow", "air_flow", "temperature")
5. Click "Create Inputs Table"
6. **Edit constraints** (Min, Max, Max_Change_%)
7. Adjust optimization parameters (learning rate, epochs)
8. Click "Run Optimization" 🚀

**Step 3: View Results**
- Loss convergence plot
- Input changes table
- Objectives comparison chart

---

### Option 3: Python Code

Use directly in your Python scripts:

```python
from optimization import (
    InverseOptimizer,
    ConstraintManager,
    InputConstraint,
    OptimizationConfig
)

# 1. Load your trained model
model, scaler_X, scaler_y = load_your_model()

# 2. Create optimizer
optimizer = InverseOptimizer(
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    device='cuda'  # or 'cpu'
)

# 3. Define constraints
constraints = [
    InputConstraint(
        name='fuel_flow',
        min_value=50.0,
        max_value=150.0,
        baseline_value=100.0,
        max_change_rate=0.20,  # ±20%
        is_fixed=False
    ),
    InputConstraint(
        name='air_flow',
        min_value=200.0,
        max_value=400.0,
        baseline_value=300.0,
        max_change_rate=0.15,  # ±15%
        is_fixed=False
    ),
    # Add other inputs (set is_fixed=True for fixed inputs)
]

constraint_manager = ConstraintManager(constraints)

# 4. Define optimization target
# Example: Reduce combustor acceleration by 10%
targets = {
    5: {  # Index of target signal
        'bias': -0.10,  # -10% reduction
        'weight': 1.0
    }
}

# 5. Run optimization
result = optimizer.optimize(
    targets=targets,
    constraint_manager=constraint_manager,
    config=OptimizationConfig(
        learning_rate=0.01,
        max_epochs=500
    )
)

# 6. Get results
print(f"Optimized inputs: {result['optimized_inputs']}")
print(f"Predictions: {result['predictions']}")
print(f"Time: {result['elapsed_time']:.2f}s")
```

---

## 📖 Examples and Documentation

### Jupyter Notebook Tutorial

Complete walkthrough with real examples:

```bash
jupyter notebook examples/inverse_optimization_example.ipynb
```

Covers:
- Loading trained models
- Single-objective optimization
- Multi-objective Pareto frontier
- Kalman filter correction
- Visualization and analysis

### Full Documentation

```bash
# Read optimization module docs
cat optimization/README.md

# Or open in browser
# optimization/README.md
```

---

## 🎯 Real-World Usage Example

### Scenario: Reduce NOx Emissions

**Goal**: Reduce NOx emissions by 15% while keeping other parameters within limits

**Steps**:

1. **Identify signals**:
   - Target: `nox_emissions` (signal index 12)
   - Variable inputs: `fuel_flow`, `air_ratio`, `temperature`
   - Fixed inputs: All others

2. **Set constraints**:
   ```python
   # Fuel flow: 80-120 kg/h, baseline 100, max change ±15%
   # Air ratio: 1.1-1.5, baseline 1.3, max change ±10%
   # Temperature: 450-550°C, baseline 500, max change ±5%
   ```

3. **Run optimization**:
   ```python
   targets = {12: {'bias': -0.15, 'weight': 1.0}}  # -15%
   result = optimizer.optimize(targets, constraint_manager)
   ```

4. **Expected result**:
   - Optimization time: ~1-2 seconds
   - NOx reduction achieved: ~14-15%
   - Suggested fuel_flow: 92.3 kg/h (↓7.7%)
   - Suggested air_ratio: 1.38 (↑6.2%)
   - Suggested temperature: 485°C (↓3%)

---

## ⚡ Performance Tips

### GPU vs CPU

```python
# Use GPU for faster optimization (5-10x speedup)
optimizer = InverseOptimizer(model, scaler_X, scaler_y, device='cuda')

# CPU works too (slightly slower)
optimizer = InverseOptimizer(model, scaler_X, scaler_y, device='cpu')
```

### Convergence Speed

**Faster convergence:**
- Fewer variable inputs (3-5 optimal)
- Looser constraints
- Higher learning rate (0.01-0.05)
- Single objective

**Better accuracy:**
- More epochs (500-1000)
- Smaller learning rate (0.001-0.01)
- Use learning rate scheduler
- Enable gradient clipping

---

## 🔧 Troubleshooting

### Problem: "Optimization not converging"

**Solution 1**: Relax constraints
```python
# Increase max_change_rate
InputConstraint(..., max_change_rate=0.30)  # was 0.20
```

**Solution 2**: Adjust learning rate
```python
OptimizationConfig(
    learning_rate=0.001,  # Reduce from 0.01
    lr_scheduler=True
)
```

**Solution 3**: Check if target is feasible
```python
# Try smaller bias first
targets = {5: {'bias': -0.05}}  # Try -5% instead of -10%
```

### Problem: "Module not found"

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python test_imports.py
```

### Problem: "CUDA out of memory"

```python
# Use CPU instead
optimizer = InverseOptimizer(model, scaler_X, scaler_y, device='cpu')
```

---

## 📊 What to Expect

### Typical Performance

| Scenario | Inputs | Targets | Time (GPU) | Time (CPU) |
|----------|--------|---------|------------|------------|
| Simple | 1-3 | 1 | 0.5-1s | 1-2s |
| **Your case** | **3** | **1** | **~1s** | **~2-3s** |
| Medium | 5-10 | 1-2 | 1-2s | 3-5s |
| Complex | 10-20 | 2-5 | 2-4s | 6-10s |

### Success Indicators

✅ **Good optimization:**
- Loss decreases steadily
- Converges in 200-500 epochs
- Final loss < 0.01
- Target achievement > 90%

⚠️ **May need tuning:**
- Loss oscillates
- Takes > 1000 epochs
- Final loss > 0.1
- Target achievement < 70%

---

## 🎓 Next Steps

1. ✅ Run `python test_imports.py` to verify setup
2. ✅ Try `python quick_start_inverse_opt.py` for demo
3. ✅ Launch UI with `python gradio_apps/inverse_control_interface.py`
4. ✅ Load your trained model and real data
5. ✅ Start optimizing!

---

## 📞 Need Help?

- **Documentation**: `optimization/README.md`
- **Examples**: `examples/inverse_optimization_example.ipynb`
- **Issues**: [GitHub Issues](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- **Email**: shvichenko11@gmail.com

---

**Happy Optimizing! 🚀**
