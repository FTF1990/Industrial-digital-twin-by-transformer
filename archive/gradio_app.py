"""
Gradio Web Interface for Industrial Digital Twin Transformer

This application provides an interactive web interface for:
- Loading and exploring sensor datasets
- Training SST and HST Transformer models
- Running inference and visualizing results
- Exporting models and configurations

For the full-featured Gradio interface with all advanced features,
please refer to Cell 3 in the original notebook or run:
    jupyter notebook notebooks/gradio_interface.ipynb

This simplified version demonstrates the core functionality.
"""

import gradio as gr
import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime

from models.static_transformer import StaticSensorTransformer
from models.hybrid_transformer import HybridSensorTransformer
from models.utils import (
    create_temporal_context_data,
    apply_ifd_smoothing,
    validate_signal_exclusivity_v1,
    validate_signal_exclusivity_v4
)
from src.data_loader import SensorDataLoader
from src.trainer import ModelTrainer
from src.inference import ModelInference

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global state to store data and models
global_state = {
    'df': None,
    'data_loader': None,
    'trained_models': {},
    'current_model': None
}


def load_dataset(file):
    """Load dataset from uploaded CSV file"""
    try:
        if file is None:
            return "❌ Please upload a CSV file", gr.update(choices=[])

        df = pd.read_csv(file.name)
        global_state['df'] = df
        global_state['data_loader'] = SensorDataLoader(df=df)

        signals = global_state['data_loader'].get_available_signals()

        msg = f"✅ Dataset loaded successfully!\n"
        msg += f"Shape: {df.shape}\n"
        msg += f"Available signals: {len(signals)}"

        return msg, gr.update(choices=signals)
    except Exception as e:
        return f"❌ Error loading data: {str(e)}", gr.update(choices=[])


def train_sst_model(boundary_sigs, target_sigs, epochs, batch_size, lr):
    """Train SST (StaticSensorTransformer) model"""
    try:
        if global_state['data_loader'] is None:
            return "❌ Please load dataset first!"

        # Validate signals
        is_valid, error_msg = validate_signal_exclusivity_v1(boundary_sigs, target_sigs)
        if not is_valid:
            return error_msg

        if not boundary_sigs or not target_sigs:
            return "❌ Please select both boundary and target signals!"

        # Prepare data
        data_splits = global_state['data_loader'].prepare_data(
            boundary_signals=boundary_sigs,
            target_signals=target_sigs,
            test_size=0.2,
            val_size=0.2
        )

        # Create model
        model = StaticSensorTransformer(
            num_boundary_sensors=len(boundary_sigs),
            num_target_sensors=len(target_sigs),
            d_model=128,
            nhead=8,
            num_layers=3,
            dropout=0.1
        )

        # Create data loaders
        train_loader, val_loader = global_state['data_loader'].create_dataloaders(
            data_splits['X_train'],
            data_splits['y_train'],
            data_splits['X_val'],
            data_splits['y_val'],
            batch_size=batch_size
        )

        # Train
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'early_stop_patience': 25
        }

        trainer = ModelTrainer(model, device=str(device), config=config)
        history = trainer.train(train_loader, val_loader, verbose=False)

        # Save model info
        model_name = f"SST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        global_state['trained_models'][model_name] = {
            'model': model,
            'trainer': trainer,
            'data_splits': data_splits,
            'boundary_signals': boundary_sigs,
            'target_signals': target_sigs,
            'type': 'SST'
        }

        msg = f"✅ SST Model training completed!\n"
        msg += f"Model name: {model_name}\n"
        msg += f"Best validation loss: {history['best_val_loss']:.6f}\n"
        msg += f"Epochs trained: {len(history['train_losses'])}"

        return msg

    except Exception as e:
        return f"❌ Training failed: {str(e)}"


def run_inference(model_name, start_idx, end_idx):
    """Run inference on selected model"""
    try:
        if model_name not in global_state['trained_models']:
            return "❌ Model not found!", None

        model_info = global_state['trained_models'][model_name]
        model = model_info['model']
        data_splits = model_info['data_splits']

        # Create inference engine
        inference = ModelInference(
            model=model,
            scaler_X=data_splits['scaler_X'],
            scaler_y=data_splits['scaler_y'],
            device=str(device)
        )

        # Get test data slice
        X_test = data_splits['scaler_X'].inverse_transform(data_splits['X_test'])
        y_test = data_splits['scaler_y'].inverse_transform(data_splits['y_test'])

        X_slice = X_test[start_idx:end_idx]
        y_slice = y_test[start_idx:end_idx]

        # Evaluate
        metrics = inference.evaluate(X_slice, y_slice, model_info['target_signals'])

        # Plot
        fig = inference.plot_predictions(
            X_slice,
            y_slice,
            signal_indices=[0, 1, 2] if len(model_info['target_signals']) >= 3 else [0],
            target_signal_names=model_info['target_signals']
        )

        # Format metrics
        metrics_text = "Evaluation Metrics:\n" + "="*60 + "\n"
        for signal, metric in metrics.items():
            if signal != 'Overall':
                metrics_text += f"\n{signal}:\n"
                metrics_text += f"  R²: {metric['R2']:.4f}\n"
                metrics_text += f"  RMSE: {metric['RMSE']:.4f}\n"
                metrics_text += f"  MAE: {metric['MAE']:.4f}\n"

        return metrics_text, fig

    except Exception as e:
        return f"❌ Inference failed: {str(e)}", None


# Create Gradio Interface
with gr.Blocks(title="Industrial Digital Twin by Transformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Industrial Digital Twin by Transformer")
    gr.Markdown("### Train and deploy Transformer models for industrial sensor prediction")

    with gr.Tabs():
        # Data Loading Tab
        with gr.Tab("📊 Data Loading"):
            gr.Markdown("## Upload Your Sensor Data")
            with gr.Row():
                with gr.Column():
                    data_file = gr.File(label="Upload CSV File", file_types=[".csv"])
                    load_btn = gr.Button("Load Data", variant="primary")
                with gr.Column():
                    data_status = gr.Textbox(label="Status", lines=5)

            available_signals = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Available Signals (will populate after loading)",
                visible=False
            )

            load_btn.click(
                fn=load_dataset,
                inputs=[data_file],
                outputs=[data_status, available_signals]
            )

        # SST Training Tab
        with gr.Tab("🎯 SST Model Training"):
            gr.Markdown("## Train SST (StaticSensorTransformer)")
            gr.Markdown("Maps boundary sensors to target sensors without temporal dependencies")

            with gr.Row():
                with gr.Column():
                    boundary_signals_sst = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="Boundary Signals (Inputs)"
                    )
                    target_signals_sst = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="Target Signals (Outputs)"
                    )

                    with gr.Row():
                        epochs_sst = gr.Slider(1, 100, value=50, step=1, label="Epochs")
                        batch_size_sst = gr.Slider(16, 256, value=64, step=16, label="Batch Size")
                        lr_sst = gr.Number(value=0.001, label="Learning Rate")

                    train_btn_sst = gr.Button("Train SST Model", variant="primary")

                with gr.Column():
                    training_log_sst = gr.Textbox(label="Training Log", lines=15)

            # Update dropdowns when data is loaded
            available_signals.change(
                fn=lambda x: (gr.update(choices=x), gr.update(choices=x)),
                inputs=[available_signals],
                outputs=[boundary_signals_sst, target_signals_sst]
            )

            train_btn_sst.click(
                fn=train_sst_model,
                inputs=[boundary_signals_sst, target_signals_sst, epochs_sst, batch_size_sst, lr_sst],
                outputs=[training_log_sst]
            )

        # Inference Tab
        with gr.Tab("🔮 Inference"):
            gr.Markdown("## Run Model Inference")

            with gr.Row():
                with gr.Column():
                    model_selector = gr.Dropdown(
                        choices=[],
                        label="Select Trained Model"
                    )
                    refresh_btn = gr.Button("Refresh Model List")

                    with gr.Row():
                        start_idx = gr.Number(value=0, label="Start Index", precision=0)
                        end_idx = gr.Number(value=500, label="End Index", precision=0)

                    inference_btn = gr.Button("Run Inference", variant="primary")

                with gr.Column():
                    metrics_output = gr.Textbox(label="Metrics", lines=15)

            inference_plot = gr.Plot(label="Predictions")

            refresh_btn.click(
                fn=lambda: gr.update(choices=list(global_state['trained_models'].keys())),
                outputs=[model_selector]
            )

            inference_btn.click(
                fn=run_inference,
                inputs=[model_selector, start_idx, end_idx],
                outputs=[metrics_output, inference_plot]
            )

    gr.Markdown("""
    ---
    ### 📖 Quick Guide
    1. **Upload Data**: Load your CSV sensor data
    2. **Select Signals**: Choose boundary (input) and target (output) sensors
    3. **Train Model**: Configure and train SST or HST model
    4. **Run Inference**: Evaluate model performance and visualize predictions

    **Note**: For advanced features and HST model training, use the full Gradio interface
    in `notebooks/gradio_interface.ipynb` or refer to Cell 3 in the original code.
    """)

if __name__ == "__main__":
    print("="*80)
    print("Starting Gradio Interface for Industrial Digital Twin")
    print("="*80)
    print(f"Device: {device}")
    print("\nLaunching interface...")
    print("Access the interface at: http://127.0.0.1:7860")
    print("="*80)

    demo.launch(share=False, debug=True)
