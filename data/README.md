# Data Folder

This folder contains the datasets for training and evaluating the digital twin models.

## Structure

```
data/
├── raw/              # Place your raw CSV sensor data here
└── README.md        # This file
```

## Data Format

Your CSV file should follow this format:

### Example CSV Structure

```csv
timestamp,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,...,sensor_n
2025-01-01 00:00:00,23.5,101.3,45.2,78.9,12.3,...,67.8
2025-01-01 00:00:01,23.6,101.4,45.1,79.0,12.4,...,67.9
2025-01-01 00:00:02,23.7,101.5,45.3,79.1,12.5,...,68.0
...
```

### Requirements

1. **Format**: CSV (Comma-Separated Values)
2. **Encoding**: UTF-8
3. **Headers**: First row must contain sensor names
4. **Timestamp** (Optional): First column can be a timestamp
   - If present, it will be automatically excluded from training
5. **Sensor Columns**: All other columns should contain numeric sensor measurements
6. **Missing Values**: Handle missing values before uploading (use interpolation or fill methods)

### Recommended Data Characteristics

- **Minimum Samples**: At least 1000 timesteps for meaningful training
- **Sensor Count**:
  - Boundary sensors: 3-50 sensors
  - Target sensors: 1-30 sensors
- **Sampling Rate**: Consistent sampling intervals
- **Data Quality**:
  - Remove outliers if necessary
  - Normalize extreme values
  - Handle sensor failures appropriately

### Example Dataset Structure

For a manufacturing process:

| Sensor Type | Examples |
|-------------|----------|
| **Boundary Conditions** (Inputs) | Temperature setpoints, Flow rates, Pressure inputs, Motor speeds |
| **Target Sensors** (Outputs) | Internal temperatures, Product quality metrics, Vibration levels, Energy consumption |

## Placing Your Data

1. Save your CSV file in the `data/raw/` folder
2. Use a descriptive filename (e.g., `manufacturing_sensors_2025.csv`)
3. Update the path in your training code:

```python
data_path = 'data/raw/manufacturing_sensors_2025.csv'
```

## Dataset Examples (To Be Added)

We will provide example datasets in future releases. Until then, you can:

1. Use your own industrial sensor data
2. Generate synthetic data for testing
3. Contact us for sample datasets

## Data Privacy

**Important**: Do not commit sensitive or proprietary data to version control!

- The `.gitignore` file excludes `*.csv` files in `data/raw/`
- Always anonymize data before sharing
- Remove confidential information from sensor names

## Getting Help

If you have questions about data format or preparation:
1. Check the example notebooks in `notebooks/`
2. Read the documentation in the main README.md
3. Open an issue on GitHub

---

**Note**: This folder is set up to store datasets locally. Large datasets are not tracked by Git.
