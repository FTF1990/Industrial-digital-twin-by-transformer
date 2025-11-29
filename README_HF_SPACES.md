---
title: Industrial Digital Twin by Transformer
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: mit
---

# Industrial Digital Twin by Transformer 🏭

**An innovative Transformer-based framework for industrial digital twin modeling using sequential sensor outputs.**

## 🌟 Features

- **Stage1 Training**: Train Static Sensor Transformer (SST) models
- **Residual Extraction**: Analyze prediction errors
- **Stage2 Boost Training**: Residual correction models
- **Ensemble Models**: Intelligent Delta R² threshold selection
- **Visualization**: Compare predictions vs actual values
- **Export**: Save trained models and results

## 🚀 Quick Start

### 1. Upload Your Data

Go to **Tab 1: Data Loading** and upload your CSV file with sensor data.

**CSV Format Requirements:**
- Rows: timesteps
- Columns: sensor measurements
- Optional: first column can be timestamp

Example:
```csv
timestamp,sensor_1,sensor_2,sensor_3,...
2025-01-01 00:00:00,23.5,101.3,45.2,...
2025-01-01 00:00:01,23.6,101.4,45.1,...
```

### 2. Configure Signals

In **Tab 2: Signal Configuration**, select:
- **Boundary Signals**: Input sensors (features)
- **Target Signals**: Output sensors to predict

### 3. Train Models

- **Stage1**: Train base SST model
- **Stage2**: (Optional) Train residual boost models for higher accuracy

### 4. Inference & Export

- Compare model performance
- Visualize predictions
- Export trained models

## 📊 Demo Data

If you don't have your own data, you can use the provided demo data to explore the interface.

## 🔧 Architecture

**StaticSensorTransformer (SST)**: Novel Transformer architecture treating sensor arrays as sequences, capturing spatial relationships between sensors.

**Stage2 Residual Boost**: Secondary model trained on residuals to refine predictions.

## 📖 Documentation

Full documentation: [GitHub Repository](https://github.com/FTF1990/Industrial-digital-twin-by-transformer)

## 📄 License

MIT License - See [LICENSE](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/blob/main/LICENSE)

## 📞 Contact

- **GitHub Issues**: [Report Issues](https://github.com/FTF1990/Industrial-digital-twin-by-transformer/issues)
- **Email**: shvichenko11@gmail.com

---

**Made with ❤️ for the Industrial AI Community**
