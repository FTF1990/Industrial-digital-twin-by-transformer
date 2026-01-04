# Industrial Digital Twin by Transformer

**[English](README.md)** | **[中文](README_CN.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **An innovative Transformer-based framework for industrial digital twin modeling using sequential sensor outputs from complex systems with advanced residual boost training.**

This project introduces Transformer architectures and residual boost training methodology specifically designed for predicting sensor outputs in industrial digital twin applications. Unlike traditional approaches, our models leverage the **sequential nature of multi-sensor systems** in complex industrial environments to achieve improved prediction accuracy through multi-stage refinement.

---

**If you find this project helpful, please consider giving it a ⭐ star! Your support helps others discover this work and motivates continued development.**

---

## 🌟 Key Innovation

**Sequential Sensor Prediction using Transformers**: This framework applies Transformer architecture to the problem of predicting sequential sensor outputs in industrial digital twins. The model treats multiple sensors as a sequence, capturing both spatial relationships between sensors and temporal dependencies in their measurements.

### Why This Matters

In complex industrial systems (manufacturing plants, chemical processes, power generation, etc.), sensors don't operate in isolation. Their outputs are:
- **Spatially correlated**: Physical proximity and process flow create dependencies
- **Temporally dependent**: Historical measurements influence current and future readings
- **Hierarchically structured**: Some sensors measure boundary conditions while others measure internal states

Traditional machine learning approaches treat sensors independently or use simple time-series models. Our Transformer-based approach **captures the full complexity of sensor interrelationships**.

## 🚀 Features

### Model Architecture

#### **StaticSensorTransformer (SST)**
- **Purpose**: Maps boundary condition sensors to target sensor predictions
- **Architecture**: Sensor sequence Transformer with learned positional encodings
- **Innovation**: Treats fixed sensor arrays as sequences (replacing NLP token sequences)
- **Use Case**: Industrial systems with complex sensor inter-dependencies
- **Advantages**:
  - Captures spatial sensor relationships through attention mechanism
  - Fast training and inference
  - Learns physical causality between sensors
  - Excellent for industrial digital twin applications

### 🆕 Enhanced Residual Boost Training System (v1.0)

#### **Stage2 Boost Training** 🚀
- Train secondary models on residuals from SST predictions
- Further refine predictions for improved accuracy
- Configurable architecture and training parameters
- Automatic model saving and versioning

#### **Intelligent Delta R² Threshold Selection** 🎯
- Calculate Delta R² (R²_ensemble - R²_stage1) for each signal
- Selectively apply Stage2 corrections based on Delta R² threshold
- Generate ensemble models combining SST + Stage2
- Optimized performance/efficiency balance
- Only use Stage2 for signals where it provides significant improvement

#### **Comprehensive Inference Comparison** 📊
- Compare ensemble model vs. pure SST model
- Visualize performance improvements for all output signals
- Detailed per-signal metrics analysis (MAE, RMSE, R²)
- CSV export with predictions and R² scores
- Interactive index range selection

#### **All-Signal Visualization** 📈
- Individual prediction vs actual comparison for every output signal
- Dynamic layout adapting to number of signals
- R² scores displayed for each signal
- Easy identification of model improvements

### ⚡ Lightweight & Edge-Ready Architecture

#### **Ultra-Lightweight Transformer Design**
Despite being Transformer-based, our models are designed as **ultra-lightweight variants** that maintain exceptional performance while minimizing computational requirements:

- **Edge Device Optimized**: Train and deploy on resource-constrained hardware
- **Fast Inference**: Real-time predictions with minimal latency
- **Low Memory Footprint**: Efficient model architecture for embedded systems
- **Rapid Training**: Quick model convergence even on limited compute

#### **Digital Twin Anything: Universal Edge Deployment** 🌐

Our design philosophy enables **personalized digital twins for individual assets**:

- **Per-Vehicle Digital Twins**: Dedicated models for each car or vehicle
- **Per-Engine Monitoring**: Individual engine-specific predictive models
- **Device-Level Customization**: Any system with sufficient testbench sensor data can have its own lightweight digital twin
- **Automated Edge Pipeline**: Complete training and inference pipeline deployable on edge devices

**Vision**: Create an automated, lightweight digital twin for **anything** - from individual machines to entire production lines, all running on edge hardware with continuous learning capabilities.

#### **Future Potential: Simulation Model Surrogate** 🔬

**Envisioned application for computational efficiency**:

The lightweight nature of our Transformer architecture opens an exciting future possibility:
- Treat each simulation mesh region as a virtual "sensor"
- Potentially use lightweight Transformers to learn complex simulation behaviors
- **Could reverse-engineer expensive simulations** with orders of magnitude less computational cost
- May maintain high accuracy while enabling real-time simulation surrogate models
- Promising for CFD, FEA, and other computationally intensive simulations

This approach could unlock new possibilities:
- Real-time simulation during design iterations
- Democratizing access to high-fidelity simulations
- Embedding complex physics models in edge devices
- Accelerating digital twin development cycles

*Note: This represents a theoretical framework and future research direction that has not yet been fully validated in production environments.*

### Additional Features

- ✅ **Modular Design**: Easy to extend and customize
- ✅ **Comprehensive Training Pipeline**: Built-in data preprocessing, training, and evaluation
- ✅ **Interactive Gradio Interface**: User-friendly web interface for all training stages
- ✅ **Jupyter Notebooks**: Complete tutorials and examples
- ✅ **Production Ready**: Exportable models for deployment
- ✅ **Extensive Documentation**: Clear API documentation and usage examples
- ✅ **Automated Model Management**: Intelligent model saving and loading with configurations

## 📊 Use Cases

This framework is ideal for:

- **Manufacturing Digital Twins**: Predict equipment states from sensor arrays
- **Chemical Process Monitoring**: Model complex sensor interactions in reactors
- **Power Plant Optimization**: Forecast turbine and generator conditions
- **HVAC Systems**: Predict temperature and pressure distributions
- **Predictive Maintenance**: Early detection of anomalies from sensor patterns
- **Quality Control**: Predict product quality from process sensors

## 🏗️ Architecture Overview
```
Please contact the author for detail information
```

## 🔧 Installation

### Quick Start with Google Colab

```bash
# Clone the repository
!git clone https://github.com/FTF1990/Industrial-digital-twin-by-transformer.git
%cd Industrial-digital-twin-by-transformer

# Install dependencies
!pip install -r requirements.txt
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/FTF1990/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📚 Quick Start

### 1. Prepare Your Data

Place your CSV sensor data file in the `data/raw/` folder. Your CSV should have:
- Each row represents a timestep
- Each column represents a sensor measurement
- (Optional) First column can be a timestamp

Example CSV structure:
```csv
timestamp,sensor_1,sensor_2,sensor_3,...,sensor_n
2025-01-01 00:00:00,23.5,101.3,45.2,...,78.9
2025-01-01 00:00:01,23.6,101.4,45.1,...,79.0
...
```

### 2. Train Stage1 Model Using Jupyter Notebook (Basic Training)

This section demonstrates **basic Stage1 (SST) model training** for learning sensor prediction fundamentals.

**Note**: The notebook provides a foundation for understanding the SST architecture and basic training process. For the complete Stage2 Boost training and ensemble model generation, please use the enhanced Gradio interface (Section 3).

**Available Notebooks**:
- `notebooks/Train and run model with demo data and your own data with gradio interface.ipynb` - Quick start tutorial for beginners
- `notebooks/transformer_boost_Leap_final.ipynb` - Advanced example: Complete Stage1 + Stage2 training on LEAP dataset (Author's testing file, comments in Chinese)

**Basic Training Example** (for your own data):

```python
from models.static_transformer import StaticSensorTransformer
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer

# Load data
data_loader = SensorDataLoader(data_path='data/raw/your_data.csv')

# Configure signals
boundary_signals = ['sensor_1', 'sensor_2', 'sensor_3']  # Inputs
target_signals = ['sensor_4', 'sensor_5']  # Outputs to predict

# Prepare data
data_splits = data_loader.prepare_data(boundary_signals, target_signals)

# Create and train Stage1 SST model
model = StaticSensorTransformer(
    num_boundary_sensors=len(boundary_signals),
    num_target_sensors=len(target_signals)
)

trainer = ModelTrainer(model, device='cuda')
history = trainer.train(train_loader, val_loader)

# Save trained model
torch.save(model.state_dict(), 'saved_models/my_sst_model.pth')
```

**What you'll learn in Stage1**:
- Loading and preprocessing sensor data
- Configuring boundary and target sensors
- Training the Static Sensor Transformer (SST)
- Basic model evaluation and prediction

**For complete functionality** (Stage2 Boost + Ensemble Models), proceed to Section 3.

### 3. Use Enhanced Gradio Interface (Complete Stage1 + Stage2 Training)

**Gradio UI Demo Video**: Coming soon

#### **Getting Started with Jupyter Notebook Tutorial**

For a step-by-step guide, see:
- `notebooks/Train and run model with demo data and your own data with gradio interface.ipynb`

This notebook demonstrates:
- Downloading demo data from Kaggle (power-gen-machine dataset)
- Setting up the Gradio interface
- Training with demo data or your own custom data

Simply follow the notebook steps to get started with the complete workflow.

#### **Complete Workflow**

The enhanced interface provides the **complete end-to-end workflow**:
- 📊 **Tab 1: Data Loading** - Refresh and select demo data (`data.csv`) or upload your own CSV
- 🎯 **Tab 2: Signal Configuration & Stage1 Training** - Refresh, load signal configuration, select parameters, and train base SST models
- 🔬 **Tab 3: Residual Extraction** - Extract and analyze prediction errors from Stage1 models
- 🚀 **Tab 4: Stage2 Boost Training** - Train secondary models on residuals for error correction
- 🎯 **Tab 5: Ensemble Model Generation** - Intelligent Delta R² threshold-based model combination
- 📊 **Tab 6: Inference Comparison** - Compare Stage1 SST vs. ensemble model performance with visualizations
- 💾 **Tab 7: Export** - Automatic model saving with complete configurations

**This is the recommended way to experience the full capabilities of the framework**, including:
- Automated multi-stage training pipeline using demo data
- Intelligent signal-wise Stage2 selection
- Comprehensive performance metrics and visualizations
- Production-ready ensemble model generation

**Using Your Own Data**:
Simply place your CSV file in the `data/` folder, refresh in Tab 1, and select your file. Ensure your CSV follows the same format as the demo data (timesteps as rows, sensors as columns). Then configure your own input/output signals in Tab 2.

**Quick Start Guide**: See `docs/QUICKSTART.md` for a 5-minute tutorial

## 📖 Documentation

```
Please contact the author for detail information
```

## 🎯 Performance

### Benchmark Results

#### 🏭 Industrial Rotating Machinery Case Study

**Dataset**: [Power Generation Machine Sensor Data](https://www.kaggle.com/datasets/tianffan/power-gen-machine)

**Application Domain**: Real-world advanced rotating machinery for power generation
- Multi-sensor system monitoring for complex industrial equipment
- High-frequency operational data from production environment
- Representative of industrial digital twin applications

**Dataset Characteristics**:
- **Source**: Real industrial equipment sensor array
- **Complexity**: Multi-sensor interdependencies in high-performance rotating systems
- **Scale**: Full operational sensor suite covering critical parameters
- **Quality**: Production-grade sensor measurements

**Performance Results** (Test Set):

| Metric | Stage1 (SST) | Stage1+Stage2 Ensemble | Improvement |
|--------|--------------|------------------------|-------------|
| **R²** | 0.8101 | **0.9014** | +11.3% |
| **MAE** | 1.56 | **1.24** | -20.2% |
| **RMSE** | 3.89 | **3.57** | -8.3% |

**Configuration**:
- **Dataset**: 89 target signals, 217K samples
- **Stage1**: 50 epochs, default hyperparameters
- **Stage2**: Selective boost on 36/89 signals (Delta R² threshold: 0.03)
- **Hardware**: Single NVIDIA A100 GPU
- **Training**: No data augmentation, no special tuning

**Training Recommendations** (Based on Practical Experience):

The above results were achieved with default hyperparameters. However, **better performance can typically be obtained** with the following parameter tuning strategy:
- 📉 **Lower learning rate**: Smaller learning rates (e.g., 0.00003 vs. default 0.0001) often lead to better convergence
- ⏱️ **Higher scheduler patience**: Increased learning rate scheduler patience (e.g., 8 vs. default 3) allows more stable training
- 📊 **Higher decay factor**: Higher learning rate decay factors reduce aggressive learning rate reductions
- 🔄 **More epochs**: Training for more epochs with the above settings generally improves final performance

These adjustments help achieve smoother convergence and better generalization, especially for complex industrial sensor systems.

**Stage2 Intelligent Selection**:
- **36 signals** selected for Stage2 correction (significant improvement observed)
- **53 signals** kept Stage1-only predictions (already performing well)
- Adaptive strategy balances performance gains with computational efficiency

**Example Signal Improvements** (Stage1 → Ensemble):
- Vibration sensors: R² -0.13 → 0.26, -0.55 → 0.47 (challenging signals)
- Temperature sensors: R² 0.35 → 0.59, 0.68 → 0.93 (moderate improvements)
- Pressure sensors: R² 0.08 → 0.47, 0.42 → 0.63 (significant gains)

<details>
<summary><b>📊 Click to View Full Results Visualization (All Signals Prediction Performance)</b></summary>

<br>

The following image shows the prediction performance of all 89 target signals on the test set after Stage1 + Stage2 Boost:

![All Signals Prediction Results Demo](saved_models/result_demo.webp)

**Figure Description**:
- Blue line: Ground Truth
- Orange line: Model Prediction
- Each subplot represents the prediction performance of one sensor signal
- Most signals show predictions closely matching ground truth values

</details>

**Practical Insights**:
- ✅ **Strong out-of-box baseline**: Stage1 achieves R² = 0.81 with default settings
- ✅ **Refinement when needed**: Stage2 boost provides targeted improvements for challenging signals
- ✅ **Real-world sensor data**: Demonstrates effectiveness on production equipment measurements
- ✅ **Efficient training**: Both stages train quickly on standard hardware

**Trained Models**: [Available on Kaggle Models](https://www.kaggle.com/models/tianffan/industrial-digital-twin-by-transformer)

**Model File Locations**:
- **Stage1 Models**: Three files (`.pth`, `_config.json`, `_scaler.pkl`) are located in `saved_models/`
- **Stage2 Models**: Located in `saved_models/stage2_boost/`

**Note on Benchmarks**:
These results are provided as reference examples on specific datasets. This project prioritizes **practical applicability and ease of deployment** over competitive benchmark scores. Performance will vary based on your specific industrial application, sensor characteristics, and data quality. We encourage users to evaluate the framework on their own use cases.

---

#### 🌍 Atmospheric Physics Simulation Benchmark

**Dataset**: LEAP atmospheric physics simulation dataset

**Performance Results**:
- **Hardware**: Single NVIDIA A100 GPU (Google Colab)
- **Signals**: 164 output signals (excluding ptend_q family)
- **Stage1 (SST)**: R² ≈ 0.56
- **Stage2 Boost**: R² ≈ 0.58
- **Training**: No data augmentation applied

**Testing Notebook**: See `notebooks/transformer_boost_Leap_final.ipynb` (Author's testing file with comments in Chinese)

---

### 📌 Performance Notes

**Variability Factors**:
Results may vary based on:
- Dataset characteristics (sensor correlation patterns, noise levels, signal complexity)
- Physical system properties (sensor spatial relationships, temporal dynamics)
- Model configuration (architecture size, training parameters)
- Application domain (manufacturing, energy, chemical processes, etc.)

**Best Results Observed**:
- **Highly correlated sensor systems**: R² > 0.80 (e.g., rotating machinery)
- **Complex multi-physics systems**: R² 0.55-0.65 (e.g., atmospheric simulation)

The framework shows particularly strong performance when sensor outputs have **clear physical interdependencies and spatial relationships**, which aligns with its core design philosophy.

---

### 🤝 Community Contributions Welcome

We warmly encourage users to share their benchmark results! If you have applied this framework to your domain, please contribute:
- **Anonymized/desensitized datasets** from your industrial applications
- **Performance metrics** (R², MAE, RMSE, etc.) and visualizations
- **Use case descriptions** and domain insights

Your contributions help build understanding of the framework's capabilities across diverse industrial scenarios. Please open an [issue](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues) or submit a pull request!

## 🤝 Contributing

Thank you for your interest in this project! We truly value community engagement and feedback.

**Ways to Support This Project**:
- ⭐ **Give us a star!** It helps others discover this work and motivates continued development
- 🐛 **Bug reports or suggestions?** Please feel free to open an [issue](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- 💬 **Ideas or questions?** We welcome discussions in issues or comments
- 📊 **Performance results?** Share your anonymized data and results - these are especially valuable!

**Current Status**: Due to time constraints, the author may not be able to immediately review and merge external pull requests. We sincerely appreciate your understanding.

**For major changes**: We kindly ask that you open an issue first to discuss your proposed changes before investing significant effort.

⏱️ **Response time**: The author will respond as time permits. Your patience is greatly appreciated.

Your understanding, patience, and contributions are greatly appreciated! 🙏

### Development Setup

```bash
# Clone repository
git clone https://github.com/FTF1990/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- Inspired by digital twin applications in industrial automation
- Built with PyTorch, Gradio, and the amazing open-source community

## 📞 Contact

For questions, issues, or collaborations:
- **GitHub Issues**: [Create an issue](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- **Email**: shvichenko11@gmail.com

## 🔗 Citation

If you use this work in your research, please cite:

```bibtex
@software{industrial_digital_twin_transformer,
  author = {FTF1990},
  title = {Industrial Digital Twin by Transformer},
  year = {2025},
  url = {https://github.com/FTF1990/Industrial-digital-twin-by-transformer}
}
```

## 🗺️ Roadmap

### v1.0 (Current) ✅
- [x] Stage2 Boost training system
- [x] Intelligent R² threshold selection
- [x] Ensemble model generation
- [x] Inference comparison tools
- [x] Enhanced Gradio interface

### v2.0 (Current) ✅

#### **Inverse Control Optimization System** 🎯
Gradient-based inverse optimization for control parameter tuning:

- **Gradient-Based Inverse Optimizer**:
  - Freeze trained model parameters (model as digital twin)
  - Optimize input boundary conditions using gradient descent
  - Achieve target outputs by adjusting controllable inputs
  - Fast convergence (typically 0.5-2 seconds on GPU)
  - Support for multiple objectives with customizable weights

- **Constraint Management**:
  - Hard constraints (min/max bounds)
  - Soft constraints (maximum change rate limits)
  - Automatic constraint projection during optimization
  - Fixed vs optimizable input specification

- **Multi-Objective Optimization**:
  - Pareto frontier generation for conflicting objectives
  - Weight scanning to explore trade-off solutions
  - Interactive visualization with Plotly
  - Solution selection and comparison tools

- **Kalman Filter Real-Time Correction**:
  - Unscented Kalman Filter for nonlinear model handling
  - Real-time control correction based on sensor feedback
  - Robust handling of measurement noise
  - Performance improvement tracking

- **Interactive Gradio Interface**:
  - Tab 0: Model loading and basic inference
  - Tab 1: Gradient-based inverse optimization
  - Tab 2: Multi-objective Pareto frontier
  - Tab 3: Kalman filter real-time correction

**Use Cases**:
- Optimize fuel flow to reduce NOx emissions by 10%
- Balance efficiency vs emissions trade-offs
- Real-time control adjustment based on sensor feedback
- Find optimal operating points for multiple objectives

**Quick Start**:
```bash
# Launch inverse optimization interface
python gradio_apps/inverse_control_interface.py

# Or use in code
from optimization import InverseOptimizer, ConstraintManager
optimizer = InverseOptimizer(model, scaler_X, scaler_y)
result = optimizer.optimize(targets, constraint_manager)
```

See [`optimization/README.md`](optimization/README.md) for detailed documentation.

---

### v3.0 (Planned) 🚀

#### **Stage3 Temporal Oscillation Enhancement System** 🕐
The next evolution targeting temporal oscillation signal reconstruction:

- **Stage3 Temporal Oscillation Feature Extraction**:
  - Focus on signals with temporal oscillation characteristics (high-frequency pulsations, vibrations, etc.)
  - Current spatial-sequence Transformers can only capture mean features of temporal oscillations, unable to reconstruct oscillation patterns
  - Use temporal ML models or temporal Transformers for pure time-series feature extraction
  - Enhance and restore temporal oscillation characteristics inherent to the signals themselves

- **Final Residual Future Prediction**:
  - After Stage1 + Stage2 + Stage3, the final residuals are primarily devoid of spatial features
  - Enable pure time-series forecasting on final residuals for future timestep prediction
  - Suitable for applications requiring forward prediction capabilities

- **Signal Relationship Mask Editing** (Planned):
  - Maximize Transformer flexibility with input-output signal relationship masks
  - Apply engineering knowledge to mask non-directly-related factors
  - Better reconstruct real system behaviors by incorporating domain expertise
  - Enhance model accuracy through expert-guided feature relationships

- **Complete Spatial-Temporal Decomposition Architecture**:
  - **Stage1 (SST)**: Spatial sensor relationships and cross-sensor dependencies
  - **Stage2 (Boost)**: Spatial residual correction and secondary spatial patterns
  - **Stage3 (Temporal)**: Pure temporal oscillation features and time-series dynamics
  - **Final Goal**: Separate spatial and temporal features into hierarchical layers, capturing all predictable patterns except irreducible noise for universal digital twin applications

- **Hierarchical Feature Extraction Philosophy**:
  - Layer 1: Primary spatial sensor correlations (SST)
  - Layer 2: Residual spatial patterns (Stage2 Boost)
  - Layer 3: Temporal oscillation characteristics (Stage3 Temporal)
  - Final Residual: Irreducible stochastic noise + optional future prediction

This design aims to achieve **universal digital twin modeling** by systematically decomposing and capturing all predictable features across different domains.

---

**Made with ❤️ for the Industrial AI Community**
