"""
Quick Start Example - Industrial Digital Twin by Transformer

This script demonstrates a minimal example of training and using
the StaticSensorTransformer (SST) model for sensor prediction.
"""

import torch
import pandas as pd
import numpy as np

# Import our modules
from models.static_transformer import StaticSensorTransformer
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer
from src.inference import ModelInference


def main():
    print("=" * 80)
    print("Industrial Digital Twin by Transformer - Quick Start")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ========================================
    # 1. Load Data
    # ========================================
    print("\n" + "=" * 80)
    print("Step 1: Loading Data")
    print("=" * 80)

    # Replace with your actual data path
    data_path = 'data/raw/your_sensor_data.csv'

    # Or use a DataFrame if you already have one loaded
    # df = pd.read_csv(data_path)
    # data_loader = SensorDataLoader(df=df)

    try:
        data_loader = SensorDataLoader(data_path=data_path)
        print("✅ Data loaded successfully!")
        print(data_loader.get_data_info())
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        print("\nPlease:")
        print("1. Place your CSV file in 'data/raw/' folder")
        print("2. Update the data_path variable in this script")
        print("\nFor now, creating synthetic data for demonstration...")

        # Create synthetic data for demonstration
        n_samples = 5000
        n_boundary = 5
        n_target = 3

        synthetic_data = {
            f'boundary_{i}': np.random.randn(n_samples) for i in range(n_boundary)
        }
        synthetic_data.update({
            f'target_{i}': np.random.randn(n_samples) for i in range(n_target)
        })

        df = pd.DataFrame(synthetic_data)
        data_loader = SensorDataLoader(df=df)
        print("✅ Synthetic data created for demonstration")

    # ========================================
    # 2. Configure Signals
    # ========================================
    print("\n" + "=" * 80)
    print("Step 2: Configuring Sensors")
    print("=" * 80)

    available_signals = data_loader.get_available_signals()

    # Select first 5 as boundary, next 3 as targets
    # Adjust based on your data
    boundary_signals = available_signals[:5]
    target_signals = available_signals[5:8] if len(available_signals) > 5 else available_signals[:3]

    print(f"\nBoundary Sensors ({len(boundary_signals)}):")
    for sig in boundary_signals:
        print(f"  • {sig}")

    print(f"\nTarget Sensors ({len(target_signals)}):")
    for sig in target_signals:
        print(f"  • {sig}")

    # ========================================
    # 3. Prepare Data
    # ========================================
    print("\n" + "=" * 80)
    print("Step 3: Preparing Data")
    print("=" * 80)

    data_splits = data_loader.prepare_data(
        boundary_signals=boundary_signals,
        target_signals=target_signals,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )

    print(f"\nData Split:")
    print(f"  Training:   {len(data_splits['X_train'])} samples")
    print(f"  Validation: {len(data_splits['X_val'])} samples")
    print(f"  Test:       {len(data_splits['X_test'])} samples")

    # ========================================
    # 4. Create Model
    # ========================================
    print("\n" + "=" * 80)
    print("Step 4: Creating StaticSensorTransformer (SST) Model")
    print("=" * 80)

    model = StaticSensorTransformer(
        num_boundary_sensors=len(boundary_signals),
        num_target_sensors=len(target_signals),
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1
    )

    print(f"\n✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================
    # 5. Train Model
    # ========================================
    print("\n" + "=" * 80)
    print("Step 5: Training Model")
    print("=" * 80)

    # Create data loaders
    train_loader, val_loader = data_loader.create_dataloaders(
        data_splits['X_train'],
        data_splits['y_train'],
        data_splits['X_val'],
        data_splits['y_val'],
        batch_size=64
    )

    # Configure training
    config = {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'epochs': 10,  # Use 100+ for real training
        'batch_size': 64,
        'grad_clip': 1.0,
        'early_stop_patience': 25,
        'scheduler_patience': 10,
        'scheduler_factor': 0.5
    }

    # Train
    trainer = ModelTrainer(model, device=str(device), config=config)
    history = trainer.train(train_loader, val_loader, verbose=True)

    print(f"\n✅ Training completed!")
    print(f"   Best validation loss: {history['best_val_loss']:.6f}")

    # ========================================
    # 6. Evaluate Model
    # ========================================
    print("\n" + "=" * 80)
    print("Step 6: Evaluating Model")
    print("=" * 80)

    # Prepare test data
    X_test_original = data_splits['scaler_X'].inverse_transform(data_splits['X_test'])
    y_test_original = data_splits['scaler_y'].inverse_transform(data_splits['y_test'])

    # Create inference engine
    inference = ModelInference(
        model=model,
        scaler_X=data_splits['scaler_X'],
        scaler_y=data_splits['scaler_y'],
        device=str(device)
    )

    # Evaluate
    metrics = inference.evaluate(X_test_original, y_test_original, target_signals)
    inference.print_metrics(metrics)

    # ========================================
    # 7. Save Model
    # ========================================
    print("\n" + "=" * 80)
    print("Step 7: Saving Model")
    print("=" * 80)

    import os
    os.makedirs('models/saved', exist_ok=True)

    save_path = 'models/saved/quickstart_sst_model.pth'
    trainer.save_model(save_path)
    print(f"\n✅ Model saved to: {save_path}")

    print("\n" + "=" * 80)
    print("Quick Start Completed Successfully!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Try the full tutorial in notebooks/train_and_inference.ipynb")
    print("2. Experiment with V4 Hybrid Transformer for temporal data")
    print("3. Use the Gradio interface: python gradio_app.py")
    print("4. Customize model architecture and hyperparameters")
    print("=" * 80)


if __name__ == "__main__":
    main()
