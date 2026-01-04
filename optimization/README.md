# Inverse Control Optimization Module

**Gradient-based inverse optimization for industrial digital twin models**

## 🎯 Overview

This module enables **inverse optimization** of control parameters using trained digital twin models. Unlike traditional optimization that tunes model parameters, this approach **freezes the model** (treating it as a digital twin) and optimizes **input boundary conditions** to achieve desired outputs.

### Core Principle

```
Traditional Training:    Fixed Inputs → Optimize Model → Outputs
Inverse Optimization:    Fixed Model → Optimize Inputs → Target Outputs
```

**Key Innovation**: Use gradient descent on inputs (not model parameters) to find control strategies that achieve specified targets.

---

## 🚀 Features

### 1. **Gradient-Based Inverse Optimizer**
- Optimize input parameters to achieve target outputs
- Support for multiple objectives with weights
- Constraint handling (hard bounds + change rate limits)
- Fast convergence (~0.5-2 seconds typical)
- GPU accelerated

### 2. **Multi-Objective Optimization**
- Pareto frontier generation for conflicting objectives
- Weight scanning to explore trade-offs
- Interactive visualization with Plotly
- Solution comparison and selection

### 3. **Kalman Filter Real-Time Correction**
- Unscented Kalman Filter for nonlinear models
- Real-time correction based on sensor feedback
- Handles measurement noise robustly
- Performance improvement tracking

### 4. **Constraint Management**
- Hard constraints (min/max bounds)
- Soft constraints (maximum change rate)
- Fixed vs optimizable inputs
- Automatic constraint projection

---

## 📦 Installation

```bash
# Install additional dependencies for optimization
pip install filterpy plotly
```

Or use the updated requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 🔧 Quick Start

### Basic Inverse Optimization

```python
from optimization import (
    InverseOptimizer,
    ConstraintManager,
    InputConstraint,
    OptimizationConfig
)

# 1. Load trained model (Stage1 SST)
model = load_your_trained_model()
scaler_X, scaler_y = load_scalers()

# 2. Create optimizer
optimizer = InverseOptimizer(
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    device='cuda'
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
    # ... other inputs (fixed or optimizable)
]

constraint_manager = ConstraintManager(constraints)

# 4. Define optimization target
# Example: Reduce NOx emissions by 10%
targets = {
    5: {  # Index of NOx signal
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

# 6. Get optimized inputs
print(f"Optimized fuel_flow: {result['optimized_inputs'][0]:.2f}")
print(f"Optimized air_flow: {result['optimized_inputs'][1]:.2f}")
print(f"Achieved NOx reduction: {result['predictions'][5]:.2f}")
```

---

## 📊 Multi-Objective Optimization

```python
from optimization import MultiObjectiveOptimizer, MultiObjectiveConfig

# Create multi-objective optimizer
mo_optimizer = MultiObjectiveOptimizer(
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y
)

# Define conflicting objectives
objectives = [
    {'signal_idx': 5, 'bias': -0.10},  # Reduce NOx by 10%
    {'signal_idx': 8, 'bias': 0.05}    # Increase efficiency by 5%
]

# Generate Pareto frontier
pareto_results = mo_optimizer.generate_pareto_frontier(
    objectives=objectives,
    constraint_manager=constraint_manager,
    config=MultiObjectiveConfig(n_pareto_points=20)
)

# Visualize
fig = mo_optimizer.plot_pareto_frontier(pareto_results)
fig.show()

# Select a solution
solution = mo_optimizer.select_solution(pareto_results, index=10)
print(f"Selected solution: {solution}")
```

---

## 🔧 Kalman Filter Correction

```python
from optimization import KalmanCorrector, KalmanConfig

# Create Kalman corrector
kf = KalmanCorrector(
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    optimizable_input_indices=[0, 1, 2],  # Indices of optimizable inputs
    target_output_indices=[5],  # Indices of target outputs
    fixed_input_values=baseline_inputs
)

# Run simulation with noisy measurements
sim_results = kf.run_simulation(
    initial_state=initial_state,
    measurements=noisy_sensor_data,
    config=KalmanConfig(
        process_noise=0.01,
        measurement_noise=0.1
    )
)

# Get corrected inputs
corrected_inputs = sim_results['corrected_inputs']

# Compute metrics
metrics = kf.compute_correction_metrics(sim_results, uncorrected_predictions)
print(f"RMSE improvement: {metrics['RMSE_Improvement_%'].mean():.2f}%")
```

---

## 🎨 Gradio Web Interface

Launch the interactive web application:

```bash
python gradio_apps/inverse_control_interface.py
```

The interface provides:
- **Tab 0**: Model loading and basic inference
- **Tab 1**: Gradient-based inverse optimization
- **Tab 2**: Multi-objective Pareto frontier
- **Tab 3**: Kalman filter real-time correction

---

## 📚 API Reference

### InverseOptimizer

```python
class InverseOptimizer:
    def __init__(self, model, scaler_X, scaler_y, device='cuda'):
        """Initialize inverse optimizer"""

    def optimize(self, targets, constraint_manager, initial_inputs=None,
                 config=None, callback=None):
        """
        Run inverse optimization

        Args:
            targets: Dict of {signal_idx: {'bias': float, 'weight': float}}
            constraint_manager: ConstraintManager instance
            initial_inputs: Initial input values (optional)
            config: OptimizationConfig (optional)
            callback: Callback function(epoch, loss, inputs) (optional)

        Returns:
            Dict with 'optimized_inputs', 'predictions', 'loss_history', etc.
        """
```

### ConstraintManager

```python
class ConstraintManager:
    def __init__(self, constraints: List[InputConstraint]):
        """Initialize constraint manager"""

    def project(self, values: torch.Tensor) -> torch.Tensor:
        """Project values to satisfy constraints"""

    def validate_values(self, values: np.ndarray) -> bool:
        """Check if values satisfy constraints"""
```

### OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    learning_rate: float = 0.01
    max_epochs: int = 500
    optimizer_type: str = "adam"  # 'adam' or 'sgd'
    gradient_clip: float = 1.0
    patience: int = 50
    loss_type: str = "mse"  # 'mse', 'mae', 'huber'
```

---

## 📖 Examples

### Example 1: Single Objective, Single Variable

Reduce combustion chamber acceleration by 10% by optimizing fuel flow:

```python
targets = {0: {'bias': -0.10, 'weight': 1.0}}  # Target signal index 0
# Optimize only fuel_flow, keep others fixed
result = optimizer.optimize(targets, constraint_manager)
```

**Expected time**: ~0.5-1 second (GPU)

### Example 2: Single Objective, Multiple Variables

Reduce NOx by 15% by optimizing fuel_flow, air_flow, and temperature:

```python
targets = {5: {'bias': -0.15, 'weight': 1.0}}
# Allow fuel_flow, air_flow, temperature to vary
result = optimizer.optimize(targets, constraint_manager)
```

**Expected time**: ~1-2 seconds (GPU)

### Example 3: Multiple Objectives

Reduce NOx by 10% AND increase efficiency by 5%:

```python
targets = {
    5: {'bias': -0.10, 'weight': 1.0},  # NOx
    8: {'bias': 0.05, 'weight': 1.0}    # Efficiency
}
result = optimizer.optimize(targets, constraint_manager)
```

**Expected time**: ~1-2 seconds (GPU)

---

## ⚡ Performance

### Typical Performance Metrics

| Scenario | Variables | Targets | Epochs | Time (GPU) | Time (CPU) |
|----------|-----------|---------|--------|------------|------------|
| Simple | 1-3 | 1 | 150-300 | **0.5-1s** | 1-2s |
| Medium | 4-10 | 1-2 | 300-500 | **1-2s** | 3-5s |
| Complex | 10-30 | 2-5 | 500-1000 | **2-4s** | 6-10s |

### Optimization Factors

**Faster convergence:**
- Fewer optimizable variables
- Looser constraints
- Single objective
- Higher learning rate (with caution)

**Slower convergence:**
- Many variables (>20)
- Tight constraints
- Conflicting objectives
- Complex target requirements

---

## 🔬 Technical Details

### How It Works

1. **Freeze Model**: Set `requires_grad=False` for all model parameters
2. **Optimizable Inputs**: Create input tensor with `requires_grad=True`
3. **Forward Pass**: Model predicts outputs from current inputs
4. **Loss Computation**: Compare predictions with targets
5. **Backward Pass**: Compute gradients w.r.t. inputs (not model!)
6. **Update Inputs**: Use optimizer (Adam/SGD) to adjust inputs
7. **Constraint Projection**: Clip inputs to satisfy constraints
8. **Repeat**: Until convergence or max epochs

### Mathematical Formulation

Minimize:
```
L = Σ w_i * ||f(x) - y_target_i||²
```

Subject to:
```
x_min ≤ x ≤ x_max  (hard constraints)
|x - x_baseline| ≤ r * |x_baseline|  (change rate)
```

Where:
- `f(·)`: Trained neural network (frozen)
- `x`: Input boundary conditions (optimizable)
- `y_target`: Desired outputs
- `w_i`: Objective weights

---

## 🛠️ Troubleshooting

### Optimization Not Converging

**Possible causes:**
- Learning rate too high/low
- Constraints too tight (infeasible)
- Conflicting objectives with incompatible weights
- Poor initialization

**Solutions:**
```python
# Try adaptive learning rate
config = OptimizationConfig(
    lr_scheduler=True,
    lr_scheduler_patience=20
)

# Relax constraints
constraint = InputConstraint(
    max_change_rate=0.30  # Increase from 0.20
)

# Reduce number of epochs for preliminary testing
config.max_epochs = 200
```

### Slow Optimization

**Solutions:**
- Use GPU: `device='cuda'`
- Reduce `max_epochs`
- Use early stopping: `patience=30`
- Simplify model (if possible)

### Numerical Instability

**Solutions:**
```python
config = OptimizationConfig(
    gradient_clip=0.5,  # Reduce from 1.0
    use_gradient_normalization=True,
    loss_type='huber'  # More robust than MSE
)
```

---

## 📄 Citation

If you use this inverse optimization module, please cite:

```bibtex
@software{industrial_digital_twin_inverse_opt,
  author = {FTF1990},
  title = {Inverse Control Optimization for Industrial Digital Twin},
  year = {2025},
  url = {https://github.com/FTF1990/Industrial-digital-twin-by-transformer}
}
```

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- **Email**: shvichenko11@gmail.com
- **Examples**: See `examples/inverse_optimization_example.ipynb`

---

**Made with ❤️ for Industrial Control Optimization**
