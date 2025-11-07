<div align="center">

## 📖 Language / 语言选择

[![English](https://img.shields.io/badge/🇺🇸_English-Click_Here-0078D4?style=for-the-badge)](#english)
[![简体中文](https://img.shields.io/badge/🇨🇳_简体中文-点击这里-FF0000?style=for-the-badge)](#中文)

</div>

---

<a name="english"></a>

# Industrial Digital Twin by Transformer

**[English](#english)** | **[中文](#中文)**

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

### 🔑 Core Innovation: Sensors as Sequence Elements

**Traditional NLP Transformers vs. SST (Our Innovation)**

```
┌─────────────────────────────────────────────────────────────────┐
│                  NLP Transformer (Traditional)                  │
├─────────────────────────────────────────────────────────────────┤
│ Input:  [The, cat, sits, on, the, mat]  ← Words as tokens      │
│ Embed:  [E₁,  E₂,  E₃,   E₄,  E₅,  E₆]  ← Word embeddings      │
│ Pos:    [P₁,  P₂,  P₃,   P₄,  P₅,  P₆]  ← Temporal order       │
│ Attn:   Semantic relationships between words                     │
└─────────────────────────────────────────────────────────────────┘

                              ⬇️  INNOVATION  ⬇️

┌─────────────────────────────────────────────────────────────────┐
│              SST - Sensor Sequence Transformer (Ours)           │
├─────────────────────────────────────────────────────────────────┤
│ Input:  [S₁,  S₂,  S₃, ..., Sₙ]  ← Fixed sensor array          │
│         (Temp, Pressure, Flow, ...)                             │
│ Embed:  [E₁,  E₂,  E₃, ..., Eₙ]  ← Sensor value embeddings     │
│ Pos:    [P₁,  P₂,  P₃, ..., Pₙ]  ← SPATIAL locations           │
│ Attn:   Physical causality & sensor inter-dependencies          │
│                                                                  │
│ Key Differences:                                                 │
│ • Fixed sequence length (N sensors predetermined)               │
│ • Position = Sensor location, NOT temporal order                │
│ • Attention learns cross-sensor physical relationships          │
│ • Domain-specific for industrial systems                        │
└─────────────────────────────────────────────────────────────────┘
```

### 🎯 SST Architecture Deep Dive

```
Physical Sensor Array: [Sensor₁, Sensor₂, ..., Sensorₙ]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Sensor Embedding Layer                        │
│  • Projects each scalar sensor reading → d_model dimensions     │
│  • Each sensor gets its own embedding transformation            │
│  • Input: (batch, N_sensors) → Output: (batch, N_sensors, d_model)│
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│               Learnable Position Encoding                        │
│  • Unlike NLP: Encodes SPATIAL sensor positions                 │
│  • Learns sensor location importance (e.g., inlet vs outlet)    │
│  • Shape: (N_sensors, d_model) - one per sensor                │
│  • Added to embeddings: Embed + PosEncode                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Head Self-Attention Mechanism                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Head 1: Learns temperature-pressure relationships        │  │
│  │ Head 2: Learns flow-velocity correlations               │  │
│  │ Head 3: Learns spatial proximity effects                │  │
│  │ ...                                                      │  │
│  │ Head N: Learns system-wide dependencies                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│  • Captures complex, non-linear sensor interactions             │
│  • Attention weights reveal sensor importance                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Transformer Encoder Stack                      │
│  Layer 1: Attention + FFN + Residual                            │
│  Layer 2: Attention + FFN + Residual                            │
│  ...                                                             │
│  Layer L: Attention + FFN + Residual                            │
│  • Each layer refines sensor relationship understanding         │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              Global Pooling (Sequence Aggregation)               │
│  • Adaptive average pooling over sensor sequence                │
│  • Aggregates information from all sensors                      │
│  • Output: (batch, d_model) - fixed-size representation        │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Output Projection Layer                       │
│  • Projects aggregated representation → target sensor values    │
│  • Linear transformation: d_model → N_target_sensors           │
│  • Final predictions: (batch, N_target_sensors)                │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
              Target Sensor Predictions
```

### 📊 Stage2 Residual Boost System

Built on top of SST, the Stage2 system further refines predictions:

```
Step 1: Base SST Model
   Boundary Sensors → [SST] → Predictions + Residuals

Step 2: Stage2 Residual Model
   Boundary Sensors → [SST₂] → Residual Corrections

Step 3: Intelligent Delta R² Selection
   For each target signal:
     Delta R² = R²_ensemble - R²_stage1
     if Delta R² > threshold: Apply Stage2 correction
     else: Use base SST prediction

Step 4: Final Ensemble Model
   Predictions = Stage1 predictions + selective Stage2 corrections
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

### Project Structure

```
Industrial-digital-twin-by-transformer/
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── static_transformer.py    # SST (StaticSensorTransformer)
│   ├── utils.py                # Utility functions
│   └── saved/                  # Saved model checkpoints
├── saved_models/               # Trained models with configs
│   ├── StaticSensorTransformer_*.pth   # SST models
│   ├── stage2_boost/           # Stage2 residual models
│   ├── ensemble/               # Ensemble model configs
│   └── tft_models/            # TFT models (if used)
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── trainer.py             # Training pipeline
│   └── inference.py           # Inference engine
├── docs/                       # Documentation
│   ├── ENHANCED_VERSION_README.md  # Enhanced features guide
│   ├── UPDATE_NOTES.md        # Detailed update notes
│   ├── QUICKSTART.md          # 5-minute quick start
│   └── FILE_MANIFEST.md       # File structure guide
├── notebooks/                  # Jupyter notebooks
│   ├── transformer_boost_Leap_final.ipynb  # Author's testing file on LEAP dataset (comments in Chinese)
│   └── Train and run model with demo data and your own data with gradio interface.ipynb  # Quick start tutorial
├── data/                      # Data folder
│   ├── raw/                   # Place your CSV files here
│   └── residuals_*.csv       # Extracted residuals
├── examples/                  # Example scripts
│   └── quick_start.py        # Quick start example
├── configs/                   # Configuration files
├── archive/                   # Archived old files
│   ├── gradio_app.py         # Old simple interface
│   ├── gradio_full_interface.py  # Old full interface
│   └── hybrid_transformer.py  # Deprecated HST model
├── gradio_sensor_transformer_app.py # 🆕 Enhanced Gradio application
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── LICENSE                   # MIT License
└── README.md                # This file
```

### Model APIs

#### StaticSensorTransformer (SST)

```python
from models.static_transformer import StaticSensorTransformer

model = StaticSensorTransformer(
    num_boundary_sensors=10,    # Number of input sensors
    num_target_sensors=5,       # Number of output sensors
    d_model=128,                # Model dimension
    nhead=8,                    # Number of attention heads
    num_layers=3,               # Number of transformer layers
    dropout=0.1                 # Dropout rate
)

# Forward pass
predictions = model(boundary_conditions)  # Shape: (batch_size, num_target_sensors)
```

#### Stage2 Residual Boost Training

```python
# Step 1: Train base SST model
base_model = StaticSensorTransformer(...)
# ... train base model ...

# Step 2: Extract residuals
residuals = true_values - base_model_predictions

# Step 3: Train Stage2 model on residuals
stage2_model = StaticSensorTransformer(...)
# ... train stage2 on residuals ...

# Step 4: Generate ensemble with intelligent Delta R² selection
for signal_idx in range(num_signals):
    r2_base = calculate_r2(true_values[:, signal_idx], base_predictions[:, signal_idx])
    r2_ensemble = calculate_r2(true_values[:, signal_idx], base_pred[:, signal_idx] + stage2_pred[:, signal_idx])
    delta_r2 = r2_ensemble - r2_base

    if delta_r2 > threshold:  # e.g., threshold=0.05 (5% improvement)
        # Use Stage2 correction (significant improvement)
        ensemble_pred[:, signal_idx] = base_pred[:, signal_idx] + stage2_pred[:, signal_idx]
    else:
        # Keep base prediction (no significant improvement)
        ensemble_pred[:, signal_idx] = base_pred[:, signal_idx]
```

**Note**: The enhanced Gradio interface (`gradio_sensor_transformer_app.py`) automates this entire workflow.

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

### v2.0 (Upcoming) 🚀

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

---
---
---

<a name="中文"></a>

<div align="center">

## 📖 Language / 语言选择

[![English](https://img.shields.io/badge/🇺🇸_English-Click_Here-0078D4?style=for-the-badge)](#english)
[![简体中文](https://img.shields.io/badge/🇨🇳_简体中文-点击这里-FF0000?style=for-the-badge)](#中文)

</div>

---

# Industrial Digital Twin by Transformer (基于 Transformer 的工业数字孪生)

**[English](#english)** | **[中文](#中文)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **一个创新的基于 Transformer 的框架，专为复杂系统中的工业数字孪生建模设计，使用序列传感器输出和先进的残差提升训练方法。**

本项目引入了 Transformer 架构和残差提升训练方法，专门设计用于预测工业数字孪生应用中的传感器输出。与传统方法不同，我们的模型利用复杂工业环境中**多传感器系统的序列特性**，通过多阶段优化实现更好的预测精度。

---

**如果您觉得这个项目有帮助，请考虑给它一个 ⭐ star！您的支持帮助更多人发现这项工作，并激励项目持续发展。**

---

## 🌟 核心创新

**使用 Transformer 进行序列传感器预测**：这个框架将 Transformer 架构应用于工业数字孪生中序列传感器输出预测问题的框架。该模型将多个传感器视为一个序列，捕获传感器之间的空间关系及其测量值的时间依赖性。

### 为什么这很重要

在复杂的工业系统（制造工厂、化工过程、发电等）中，传感器不是孤立运行的。它们的输出具有以下特征：
- **空间相关性**：物理邻近性和工艺流程创建了依赖关系
- **时间依赖性**：历史测量值影响当前和未来的读数
- **层次结构**：一些传感器测量边界条件，而另一些测量内部状态

传统的机器学习方法独立对待传感器或使用简单的时间序列模型。我们基于 Transformer 的方法**捕获传感器相互关系的全部复杂性**。

## 🚀 功能特性

### 模型架构

#### **StaticSensorTransformer (SST)**
- **用途**：将边界条件传感器映射到目标传感器预测
- **架构**：具有学习位置编码的传感器序列 Transformer
- **创新点**：将固定传感器阵列视为序列（替代 NLP 中的词元序列）
- **应用场景**：具有复杂传感器相互依赖关系的工业系统
- **优势**：
  - 通过注意力机制捕获空间传感器关系
  - 快速训练和推理
  - 学习传感器之间的物理因果关系
  - 非常适合工业数字孪生应用

### 🆕 增强型残差提升训练系统 (v1.0)

#### **Stage2 提升训练** 🚀
- 在 SST 预测残差上训练第二阶段模型
- 进一步优化预测以提高准确性
- 可配置的架构和训练参数
- 自动模型保存和版本控制

#### **智能 Delta R² 阈值选择** 🎯
- 计算每个信号的 Delta R² (R²_ensemble - R²_stage1)
- 基于 Delta R² 阈值选择性地应用 Stage2 修正
- 生成结合 SST + Stage2 的集成模型
- 优化的性能/效率平衡
- 仅对有显著改进的信号使用 Stage2

#### **全面的推理对比** 📊
- 比较集成模型与纯 SST 模型
- 可视化所有输出信号的性能改进
- 详细的逐信号指标分析（MAE、RMSE、R²）
- CSV 导出包含预测值和 R² 分数
- 交互式索引范围选择

#### **全信号可视化** 📈
- 每个输出信号的独立预测 vs 实际值对比
- 动态布局适应信号数量
- 每个信号显示 R² 分数
- 轻松识别模型改进

### ⚡ 轻量化与边缘就绪架构

#### **超轻量化 Transformer 设计**
尽管基于 Transformer 架构，我们的模型被设计为**超轻量化变体**，在最小化计算需求的同时保持良好性能：

- **边缘设备优化**：在资源受限的硬件上训练和部署
- **快速推理**：实时预测，延迟极低
- **低内存占用**：适用于嵌入式系统的高效模型架构
- **快速训练**：即使在有限算力下也能快速收敛

#### **Digital Twin Anything：通用边缘部署** 🌐

我们的设计理念实现了**个性化的单体资产数字孪生**：

- **单车数字孪生**：为每辆汽车建立专属模型
- **单机监控**：为每台发动机建立个性化预测模型
- **设备级定制**：任何在测试台架下有足够传感器数据的设备系统都可以拥有专属的轻量级数字孪生
- **自动化边缘流程**：完整的训练和推理流程可部署在边缘设备上

**愿景**：为**任何事物**创建自动化的轻量级数字孪生 - 从单个机器到整条生产线，全部运行在边缘硬件上并具备持续学习能力。

#### **未来潜力：仿真模型代理** 🔬

**面向计算效率的前瞻性应用展望**：

我们轻量化 Transformer 架构的特性开启了一个令人兴奋的未来可能性：
- 将仿真中的每个网格区域视为虚拟"传感器"
- 有潜力使用轻量级 Transformer 学习复杂的仿真行为
- **可能以极低算力逆向构建昂贵的仿真模型**，计算成本有望降低数个数量级
- 有望在保持高精度的同时实现实时仿真代理模型
- 对 CFD、FEA 等计算密集型仿真具有应用前景

这一方法可能带来新的应用场景：
- 设计迭代过程中的实时仿真
- 普及高保真仿真的使用
- 在边缘设备中嵌入复杂物理模型
- 加速数字孪生开发周期

*注：这代表了一个理论框架和未来研究方向，尚未在生产环境中得到充分验证。*

### 附加功能

- ✅ **模块化设计**：易于扩展和定制
- ✅ **全面的训练流程**：内置数据预处理、训练和评估
- ✅ **交互式 Gradio 界面**：适用于所有训练阶段的用户友好型 Web 界面
- ✅ **Jupyter Notebooks**：完整的教程和示例
- ✅ **生产就绪**：可导出模型用于部署
- ✅ **详尽的文档**：清晰的 API 文档和使用示例
- ✅ **自动化模型管理**：智能模型保存和加载（含配置）

## 📊 使用场景

本框架非常适合：

- **制造业数字孪生**：从传感器阵列预测设备状态
- **化工过程监控**：建模反应器中的复杂传感器交互
- **发电厂优化**：预测涡轮机和发电机状况
- **HVAC 系统**：预测温度和压力分布
- **预测性维护**：从传感器模式中早期检测异常
- **质量控制**：从工艺传感器预测产品质量

## 🏗️ 架构概述

### 🔑 核心创新：传感器作为序列元素

**传统 NLP Transformer vs. SST（我们的创新）**

```
┌─────────────────────────────────────────────────────────────────┐
│                  NLP Transformer（传统）                        │
├─────────────────────────────────────────────────────────────────┤
│ 输入:  [The, cat, sits, on, the, mat]  ← 单词作为词元          │
│ 嵌入:  [E₁,  E₂,  E₃,   E₄,  E₅,  E₆]  ← 词嵌入                │
│ 位置:  [P₁,  P₂,  P₃,   P₄,  P₅,  P₆]  ← 时间顺序              │
│ 注意力: 单词之间的语义关系                                      │
└─────────────────────────────────────────────────────────────────┘

                              ⬇️  创新点  ⬇️

┌─────────────────────────────────────────────────────────────────┐
│              SST - 传感器序列 Transformer（我们的）             │
├─────────────────────────────────────────────────────────────────┤
│ 输入:  [S₁,  S₂,  S₃, ..., Sₙ]  ← 固定传感器阵列               │
│         (温度, 压力, 流量, ...)                                 │
│ 嵌入:  [E₁,  E₂,  E₃, ..., Eₙ]  ← 传感器值嵌入                 │
│ 位置:  [P₁,  P₂,  P₃, ..., Pₙ]  ← 空间位置                     │
│ 注意力: 物理因果关系和传感器相互依赖关系                        │
│                                                                  │
│ 关键差异：                                                       │
│ • 固定序列长度（N 个传感器预先确定）                            │
│ • 位置 = 传感器位置，而非时间顺序                               │
│ • 注意力学习跨传感器物理关系                                    │
│ • 针对工业系统的领域专用设计                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 🎯 SST 架构深入解析

```
物理传感器阵列: [Sensor₁, Sensor₂, ..., Sensorₙ]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    传感器嵌入层                                  │
│  • 将每个标量传感器读数投影到 d_model 维度                      │
│  • 每个传感器获得自己的嵌入变换                                  │
│  • 输入: (batch, N_sensors) → 输出: (batch, N_sensors, d_model) │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│               可学习位置编码                                     │
│  • 与 NLP 不同：编码空间传感器位置                              │
│  • 学习传感器位置重要性（例如，进口 vs 出口）                   │
│  • 形状: (N_sensors, d_model) - 每个传感器一个                 │
│  • 添加到嵌入中: Embed + PosEncode                             │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              多头自注意力机制                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 头 1: 学习温度-压力关系                                  │  │
│  │ 头 2: 学习流量-速度相关性                                │  │
│  │ 头 3: 学习空间邻近效应                                   │  │
│  │ ...                                                      │  │
│  │ 头 N: 学习系统级依赖关系                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│  • 捕获复杂的非线性传感器交互                                   │
│  • 注意力权重揭示传感器重要性                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Transformer 编码器堆栈                         │
│  层 1: 注意力 + FFN + 残差                                      │
│  层 2: 注意力 + FFN + 残差                                      │
│  ...                                                             │
│  层 L: 注意力 + FFN + 残差                                      │
│  • 每一层优化传感器关系理解                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              全局池化（序列聚合）                                │
│  • 对传感器序列进行自适应平均池化                               │
│  • 聚合来自所有传感器的信息                                     │
│  • 输出: (batch, d_model) - 固定大小表示                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    输出投影层                                    │
│  • 将聚合表示投影到目标传感器值                                 │
│  • 线性变换: d_model → N_target_sensors                        │
│  • 最终预测: (batch, N_target_sensors)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
              目标传感器预测
```

### 📊 Stage2 残差提升系统

建立在 SST 之上，Stage2 系统进一步优化预测：

```
步骤 1: 基础 SST 模型
   边界传感器 → [SST] → 预测 + 残差

步骤 2: Stage2 残差模型
   边界传感器 → [SST₂] → 残差修正

步骤 3: 智能 Delta R² 选择
   对于每个目标信号:
     Delta R² = R²_ensemble - R²_stage1
     if Delta R² > 阈值: 应用 Stage2 修正
     else: 使用基础 SST 预测

步骤 4: 最终集成模型
   预测 = Stage1 预测 + 选择性 Stage2 修正

```

## 🔧 安装

### 使用 Google Colab 快速开始

```bash
# 克隆仓库
!git clone https://github.com/FTF1990/Industrial-digital-twin-by-transformer.git
%cd Industrial-digital-twin-by-transformer

# 安装依赖
!pip install -r requirements.txt
```

### 本地安装

```bash
# 克隆仓库
git clone https://github.com/FTF1990/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows 系统: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 📚 快速入门

### 1. 准备数据

将您的 CSV 传感器数据文件放在 `data/raw/` 文件夹中。您的 CSV 应该具有：
- 每行代表一个时间步
- 每列代表一个传感器测量值
- （可选）第一列可以是时间戳

CSV 结构示例：
```csv
timestamp,sensor_1,sensor_2,sensor_3,...,sensor_n
2025-01-01 00:00:00,23.5,101.3,45.2,...,78.9
2025-01-01 00:00:01,23.6,101.4,45.1,...,79.0
...
```

### 2. 使用 Jupyter Notebook 训练 Stage1 模型（基础训练）

本节演示**基础 Stage1 (SST) 模型训练**，用于学习传感器预测建模的基础知识。

**注意**：Notebook 提供了理解 SST 架构和基础训练过程的基础。如需完整的 Stage2 提升训练和集成模型生成功能，请使用增强型 Gradio 界面（第3节）。

**可用的 Notebooks**：
- `notebooks/Train and run model with demo data and your own data with gradio interface.ipynb` - 初学者快速入门教程
- `notebooks/transformer_boost_Leap_final.ipynb` - 高级示例：在 LEAP 数据集上的完整 Stage1 + Stage2 训练（作者测试文件，注释为中文）

**基础训练示例**（用于您自己的数据）：

```python
from models.static_transformer import StaticSensorTransformer
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer

# 加载数据
data_loader = SensorDataLoader(data_path='data/raw/your_data.csv')

# 配置信号
boundary_signals = ['sensor_1', 'sensor_2', 'sensor_3']  # 输入
target_signals = ['sensor_4', 'sensor_5']  # 要预测的输出

# 准备数据
data_splits = data_loader.prepare_data(boundary_signals, target_signals)

# 创建和训练 Stage1 SST 模型
model = StaticSensorTransformer(
    num_boundary_sensors=len(boundary_signals),
    num_target_sensors=len(target_signals)
)

trainer = ModelTrainer(model, device='cuda')
history = trainer.train(train_loader, val_loader)

# 保存训练好的模型
torch.save(model.state_dict(), 'saved_models/my_sst_model.pth')
```

**在 Stage1 中您将学到**：
- 加载和预处理传感器数据
- 配置边界传感器和目标传感器
- 训练静态传感器 Transformer (SST)
- 基础模型评估和预测

**如需完整功能**（Stage2 提升 + 集成模型），请继续第3节。

### 3. 使用增强型 Gradio 界面（完整 Stage1 + Stage2 训练）

**Gradio UI 演示视频**：即将推出

#### **Jupyter Notebook 入门教程**

有关分步指南，请参阅：
- `notebooks/Train and run model with demo data and your own data with gradio interface.ipynb`

该 notebook 演示了：
- 从 Kaggle 下载演示数据（power-gen-machine 数据集）
- 设置 Gradio 界面
- 使用演示数据或您自己的自定义数据进行训练

只需按照 notebook 步骤操作即可开始使用完整工作流程。

#### **完整工作流程**

增强型界面提供**完整的端到端工作流程**：
- 📊 **Tab 1: 数据加载** - 刷新并选择演示数据（`data.csv`）或上传您自己的 CSV
- 🎯 **Tab 2: 信号配置与 Stage1 训练** - 刷新，加载信号配置，选择参数，训练基础 SST 模型
- 🔬 **Tab 3: 残差提取** - 从 Stage1 模型中提取和分析预测误差
- 🚀 **Tab 4: Stage2 提升训练** - 在残差上训练第二阶段模型进行误差修正
- 🎯 **Tab 5: 集成模型生成** - 基于智能 Delta R² 阈值的模型组合
- 📊 **Tab 6: 推理对比** - 比较 Stage1 SST vs. 集成模型性能并可视化
- 💾 **Tab 7: 导出** - 自动模型保存（含完整配置）

**这是体验框架完整功能的推荐方式**，包括：
- 使用演示数据的自动化多阶段训练流程
- 智能的逐信号 Stage2 选择
- 全面的性能指标和可视化
- 生产就绪的集成模型生成

**使用您自己的数据**：
只需将您的 CSV 文件放在 `data/` 文件夹中，在 Tab 1 中刷新并选择您的文件。确保您的 CSV 遵循与演示数据相同的格式（时间步作为行，传感器作为列）。然后在 Tab 2 中配置您自己的输入/输出信号。

**快速入门指南**：参见 `docs/QUICKSTART.md` 获取 5 分钟教程

## 📖 文档

### 项目结构

```
Industrial-digital-twin-by-transformer/
├── models/                      # 模型实现
│   ├── __init__.py
│   ├── static_transformer.py    # SST (StaticSensorTransformer)
│   ├── utils.py                # 工具函数
│   └── saved/                  # 保存的模型检查点
├── saved_models/               # 训练好的模型（含配置）
│   ├── StaticSensorTransformer_*.pth   # SST 模型
│   ├── stage2_boost/           # Stage2 残差模型
│   ├── ensemble/               # 集成模型配置
│   └── tft_models/            # TFT 模型（如果使用）
├── src/                        # 源代码
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载和预处理
│   ├── trainer.py             # 训练流程
│   └── inference.py           # 推理引擎
├── docs/                       # 文档
│   ├── ENHANCED_VERSION_README.md  # 增强功能指南
│   ├── UPDATE_NOTES.md        # 详细更新说明
│   ├── QUICKSTART.md          # 5 分钟快速入门
│   └── FILE_MANIFEST.md       # 文件结构指南
├── notebooks/                  # Jupyter notebooks
│   ├── transformer_boost_Leap_final.ipynb  # 作者在 LEAP 数据集上的测试文件（注释为中文）
│   └── Train and run model with demo data and your own data with gradio interface.ipynb  # 快速入门教程
├── data/                      # 数据文件夹
│   ├── raw/                   # 将您的 CSV 文件放在这里
│   └── residuals_*.csv       # 提取的残差
├── examples/                  # 示例脚本
│   └── quick_start.py        # 快速入门示例
├── configs/                   # 配置文件
├── archive/                   # 归档的旧文件
│   ├── gradio_app.py         # 旧的简单界面
│   ├── gradio_full_interface.py  # 旧的完整界面
│   └── hybrid_transformer.py  # 已弃用的 HST 模型
├── gradio_sensor_transformer_app.py # 🆕 增强型 Gradio 应用
├── requirements.txt          # Python 依赖
├── setup.py                  # 包设置
├── LICENSE                   # MIT 许可证
└── README.md                # 英文说明文件
```

### 模型 API

#### StaticSensorTransformer (SST)

```python
from models.static_transformer import StaticSensorTransformer

model = StaticSensorTransformer(
    num_boundary_sensors=10,    # 输入传感器数量
    num_target_sensors=5,       # 输出传感器数量
    d_model=128,                # 模型维度
    nhead=8,                    # 注意力头数量
    num_layers=3,               # Transformer 层数
    dropout=0.1                 # Dropout 率
)

# 前向传播
predictions = model(boundary_conditions)  # 形状: (batch_size, num_target_sensors)
```

#### Stage2 残差提升训练

```python
# 步骤 1: 训练基础 SST 模型
base_model = StaticSensorTransformer(...)
# ... 训练基础模型 ...

# 步骤 2: 提取残差
residuals = true_values - base_model_predictions

# 步骤 3: 在残差上训练 Stage2 模型
stage2_model = StaticSensorTransformer(...)
# ... 在残差上训练 stage2 ...

# 步骤 4: 使用智能 Delta R² 选择生成集成
for signal_idx in range(num_signals):
    r2_base = calculate_r2(true_values[:, signal_idx], base_predictions[:, signal_idx])
    r2_ensemble = calculate_r2(true_values[:, signal_idx], base_pred[:, signal_idx] + stage2_pred[:, signal_idx])
    delta_r2 = r2_ensemble - r2_base

    if delta_r2 > threshold:  # 例如, threshold=0.05 (5% 改进)
        # 使用 Stage2 修正（显著改进）
        ensemble_pred[:, signal_idx] = base_pred[:, signal_idx] + stage2_pred[:, signal_idx]
    else:
        # 保持基础预测（无显著改进）
        ensemble_pred[:, signal_idx] = base_pred[:, signal_idx]
```

**注意**：增强型 Gradio 界面（`gradio_sensor_transformer_app.py`）自动化了整个工作流程。

## 🎯 性能

### 基准测试结果

#### 🏭 工业旋转机械案例研究

**数据集**：[发电机械传感器数据](https://www.kaggle.com/datasets/tianffan/power-gen-machine)

**应用领域**：真实世界的尖端发电旋转机械
- 复杂工业设备的多传感器系统监测
- 生产环境的高频操作数据
- 工业数字孪生应用的代表性案例

**数据集特征**：
- **来源**：真实工业设备传感器阵列
- **复杂度**：高性能旋转系统中的多传感器相互依赖关系
- **规模**：覆盖关键参数的完整传感器套件
- **质量**：生产级传感器测量数据

**性能结果**（测试集）：

| 指标 | Stage1 (SST) | Stage1+Stage2 集成 | 改进幅度 |
|------|--------------|---------------------|----------|
| **R²** | 0.8101 | **0.9014** | +11.3% |
| **MAE** | 1.56 | **1.24** | -20.2% |
| **RMSE** | 3.89 | **3.57** | -8.3% |

**配置**：
- **数据集**：89 个目标信号，21.7 万样本
- **Stage1**：50 epochs，默认超参数
- **Stage2**：选择性增强 36/89 个信号（Delta R² 阈值：0.03）
- **硬件**：单卡 NVIDIA A100 GPU
- **训练**：无数据增强，无特殊调参

**训练推荐**（基于实践经验）：

以上结果使用默认超参数获得。然而，通过以下参数调优策略**通常可以获得更好的性能**：
- 📉 **更低的学习率**：较小的学习率（例如 0.00003 vs. 默认 0.0001）通常能带来更好的收敛
- ⏱️ **更高的调度器耐心值**：增加学习率调度器耐心值（例如 8 vs. 默认 3）允许更稳定的训练
- 📊 **更高的衰减因子**：更高的学习率衰减因子可减少激进的学习率下降
- 🔄 **更多的训练轮数**：使用上述设置训练更多轮次通常能提高最终性能

这些调整有助于实现更平滑的收敛和更好的泛化能力，特别是对于复杂的工业传感器系统。

**Stage2 智能选择**：
- **36 个信号** 选择 Stage2 校正（观察到显著改进）
- **53 个信号** 保持 Stage1 预测（已表现良好）
- 自适应策略平衡性能提升与计算效率

**信号改进示例**（Stage1 → 集成）：
- 振动传感器：R² -0.13 → 0.26，-0.55 → 0.47（挑战性信号）
- 温度传感器：R² 0.35 → 0.59，0.68 → 0.93（中等改进）
- 压力传感器：R² 0.08 → 0.47，0.42 → 0.63（显著提升）

<details>
<summary><b>📊 点击查看完整效果演示图（所有信号预测效果可视化）</b></summary>

<br>

下图展示了经过 Stage1 + Stage2 Boost 后，所有 89 个目标信号在测试集上的预测效果：

![所有信号预测效果演示](saved_models/result_demo.webp)

**图片说明**：
- 蓝色线条：真实值（Ground Truth）
- 橙色线条：模型预测值（Prediction）
- 每个子图代表一个传感器信号的预测效果
- 可以看到大部分信号的预测曲线与真实值高度吻合

</details>

**实用见解**：
- ✅ **强劲的开箱即用基线**：Stage1 使用默认设置达到 R² = 0.81
- ✅ **按需精炼**：Stage2 增强为挑战性信号提供针对性改进
- ✅ **真实传感器数据**：在生产设备测量数据上展示有效性
- ✅ **高效训练**：两个阶段都能在标准硬件上快速训练

**训练模型**：[Kaggle Models 提供](https://www.kaggle.com/models/tianffan/industrial-digital-twin-by-transformer)

**模型文件位置**：
- **Stage1 模型**：三个文件（`.pth`、`_config.json`、`_scaler.pkl`）位于 `saved_models/` 目录下
- **Stage2 模型**：位于 `saved_models/stage2_boost/` 目录下

**关于基准测试的说明**：
这些结果作为特定数据集上的参考示例提供。本项目优先考虑**实用性和易部署性**，而非竞争性基准分数。性能将根据您的具体工业应用、传感器特性和数据质量而变化。我们鼓励用户在自己的应用场景中评估本框架。

---

#### 🌍 大气物理仿真基准测试

**数据集**：LEAP 大气物理仿真数据集

**性能结果**：
- **硬件**：单卡 NVIDIA A100 GPU（Google Colab）
- **信号**：164 个输出信号（不包括 ptend_q 系列）
- **Stage1 (SST)**：R² ≈ 0.56
- **Stage2 Boost**：R² ≈ 0.58
- **训练**：未应用数据增强

**测试 Notebook**：参见 `notebooks/transformer_boost_Leap_final.ipynb`（作者测试文件，注释为中文）

---

### 📌 性能说明

**变异因素**：
结果可能因以下因素而变化：
- 数据集特征（传感器相关模式、噪声水平、信号复杂度）
- 物理系统属性（传感器空间关系、时间动态）
- 模型配置（架构大小、训练参数）
- 应用领域（制造业、能源、化工过程等）

**观察到的最佳结果**：
- **高度相关的传感器系统**：R² > 0.80（如旋转机械）
- **复杂多物理系统**：R² 0.55-0.65（如大气仿真）

当传感器输出具有**明确的物理相互依赖关系和空间关系**时，该框架表现出特别强的性能，这与其核心设计理念一致。

---

### 🤝 欢迎社区贡献

我们热烈鼓励用户分享基准测试结果！如果您已将此框架应用于您的领域，请贡献：
- 您工业应用中的**脱敏数据集**
- **性能指标**（R²、MAE、RMSE 等）和可视化
- **应用案例描述**和领域见解

您的贡献有助于建立对框架在不同工业场景下能力的理解。请开启 [issue](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues) 或提交 pull request！

## 🤝 贡献

感谢您对本项目的关注！我们非常重视社区的参与和反馈。

**支持本项目的方式**：
- ⭐ **给我们一个 star！** 这有助于更多人发现这项工作，并激励项目持续发展
- 🐛 **Bug 报告或建议？** 欢迎开启 [issue](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- 💬 **想法或问题？** 欢迎在 issue 或评论中讨论
- 📊 **性能结果？** 分享您的脱敏数据和结果 - 这些特别有价值！

**当前状态**：由于时间限制，作者可能无法立即审查和合并外部的 Pull Request。衷心感谢您的理解。

**对于重大更改**：恳请您先开启 issue 讨论您的提议，然后再投入大量精力。

⏱️ **回复时间**：作者会在时间允许的情况下回复。非常感谢您的耐心。

非常感谢您的理解、耐心和贡献！🙏

### 开发设置

```bash
# 克隆仓库
git clone https://github.com/FTF1990/Industrial-digital-twin-by-transformer.git
cd Industrial-digital-twin-by-transformer

# 以开发模式安装
pip install -e .

# 运行测试（如果可用）
python -m pytest tests/
```

## 📄 许可证

本项目根据 MIT 许可证授权 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- Transformer 架构基于 "Attention Is All You Need"（Vaswani et al., 2017）
- 灵感来自工业自动化中的数字孪生应用
- 使用 PyTorch、Gradio 和出色的开源社区构建

## 📞 联系方式

如有问题、议题或合作：
- **GitHub Issues**：[创建 issue](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- **电子邮件**：shvichenko11@gmail.com

## 🔗 引用

如果您在研究中使用此工作，请引用：

```bibtex
@software{industrial_digital_twin_transformer,
  author = {FTF1990},
  title = {Industrial Digital Twin by Transformer},
  year = {2025},
  url = {https://github.com/FTF1990/Industrial-digital-twin-by-transformer}
}
```

## 🗺️ 路线图

### v1.0（当前）✅
- [x] Stage2 提升训练系统
- [x] 智能 R² 阈值选择
- [x] 集成模型生成
- [x] 推理对比工具
- [x] 增强型 Gradio 界面

### v2.0（即将推出）🚀

#### **Stage3 时序震荡增强系统** 🕐
下一代演进目标：时序震荡信号重构

- **Stage3 时序震荡特征提取**：
  - 针对具有时序震荡特性的信号（高频脉动、振动等）
  - 当前的空间序列 Transformer 对时序高频震荡信号只能提取均值特征，无法还原时序震荡特征
  - 采用时序 ML 模型或时序 Transformer 进行纯时序特征提取
  - 增强并还原信号本身固有的时序震荡特征

- **最终残差未来预测**：
  - 经过 Stage1 + Stage2 + Stage3 后，最终残差基本已不包含空间特征
  - 可对最终残差进行纯时序预测，实现未来时间步预测
  - 适用于需要前向预测能力的应用场景

- **信号关联掩码编辑功能**（计划推出）：
  - 最大限度利用 Transformer 的灵活性，编辑输入输出信号关联掩码
  - 运用真实工程经验对不直接关联的要素之间施加掩码屏蔽
  - 更好地还原真实系统行为，融入领域专家知识
  - 通过专家引导的特征关系提高模型准确性

- **完整的空间-时间分解架构**：
  - **Stage1 (SST)**：空间传感器关系和跨传感器依赖性
  - **Stage2 (Boost)**：空间残差修正和次级空间模式
  - **Stage3 (Temporal)**：纯时序震荡特征和时间序列动态
  - **最终目标**：将空间和时间特征完全剥离并分层预测，除不可预测的噪音特征外，捕捉所有可预测模式，实现场景泛用化的数字孪生

- **分层特征提取哲学**：
  - 第一层：主要空间传感器相关性（SST）
  - 第二层：残差空间模式（Stage2 提升）
  - 第三层：时序震荡特征（Stage3 时序）
  - 最终残差：不可约随机噪声 + 可选的未来预测

此设计旨在通过系统性地分解和捕获不同领域的所有可预测特征，实现**通用数字孪生建模**。
---

**为工业 AI 社区精心打造 ❤️**
