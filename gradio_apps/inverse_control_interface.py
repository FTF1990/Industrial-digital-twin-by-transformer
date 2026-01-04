#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse Control Optimization Interface - Gradio Web Application

This application provides an interactive web interface for inverse optimization
of industrial digital twin models, including:
- Tab 0: Model Loading & Basic Inference
- Tab 1: Gradient-Based Inverse Optimization
- Tab 2: Multi-Objective Optimization (Pareto Frontier)
- Tab 3: Kalman Filter Real-Time Correction
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.static_transformer import StaticSensorTransformer
from models.residual_tft import GroupedMultiTargetTFT
from optimization import (
    InverseOptimizer,
    MultiObjectiveOptimizer,
    KalmanCorrector,
    ConstraintManager,
    InputConstraint,
    OptimizationConfig,
    MultiObjectiveConfig,
    KalmanConfig
)

# ============================================================================
# Global State Management
# ============================================================================

class GlobalState:
    """Global state for sharing data between tabs"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage1_model = None
        self.stage2_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.boundary_signals = []
        self.target_signals = []
        self.data_df = None
        self.inference_results = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimization_result = None
        self.pareto_results = None

STATE = GlobalState()


# ============================================================================
# Tab 0: Model Loading & Basic Inference
# ============================================================================

def load_stage1_model(model_path):
    """Load Stage1 SST model"""
    try:
        if not os.path.exists(model_path):
            return None, None, None, f"❌ Model file not found: {model_path}"

        # Load config
        config_path = model_path.replace('.pth', '_config.json')
        if not os.path.exists(config_path):
            return None, None, None, f"❌ Config file not found: {config_path}"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load scaler
        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        if not os.path.exists(scaler_path):
            return None, None, None, f"❌ Scaler file not found: {scaler_path}"

        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        # Create model
        model = StaticSensorTransformer(
            num_boundary_sensors=len(config['boundary_signals']),
            num_target_sensors=len(config['target_signals']),
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1)
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=STATE.device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.to(STATE.device)
        model.eval()

        # Update global state
        STATE.stage1_model = model
        STATE.scaler_X = scalers['scaler_X']
        STATE.scaler_y = scalers['scaler_y']
        STATE.boundary_signals = config['boundary_signals']
        STATE.target_signals = config['target_signals']

        msg = f"✅ Stage1 model loaded successfully!\n"
        msg += f"   Boundary signals: {len(STATE.boundary_signals)}\n"
        msg += f"   Target signals: {len(STATE.target_signals)}\n"
        msg += f"   Device: {STATE.device}"

        return (
            gr.update(choices=STATE.boundary_signals, value=None),
            gr.update(choices=STATE.target_signals, value=None),
            gr.update(choices=STATE.target_signals, value=None),
            msg
        )

    except Exception as e:
        return None, None, None, f"❌ Error loading model: {str(e)}"


def load_data_file(file_path):
    """Load CSV data file"""
    try:
        if not os.path.exists(file_path):
            return None, f"❌ File not found: {file_path}"

        df = pd.read_csv(file_path)
        STATE.data_df = df

        msg = f"✅ Data loaded successfully!\n"
        msg += f"   Shape: {df.shape}\n"
        msg += f"   Columns: {df.shape[1]}"

        return df.head(10), msg

    except Exception as e:
        return None, f"❌ Error loading data: {str(e)}"


def run_basic_inference(start_idx, end_idx, use_ensemble):
    """Run basic model inference"""
    try:
        if STATE.stage1_model is None:
            return None, None, None, "❌ Please load Stage1 model first"

        if STATE.data_df is None:
            return None, None, None, "❌ Please load data file first"

        # Validate indices
        max_idx = len(STATE.data_df)
        if start_idx < 0 or end_idx > max_idx or start_idx >= end_idx:
            return None, None, None, f"❌ Invalid index range. Data has {max_idx} rows."

        # Extract data slice
        data_slice = STATE.data_df.iloc[start_idx:end_idx]

        # Prepare inputs
        X = data_slice[STATE.boundary_signals].values
        y_true = data_slice[STATE.target_signals].values

        # Scale inputs
        X_scaled = STATE.scaler_X.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(STATE.device)

        # Predict
        with torch.no_grad():
            y_pred_scaled = STATE.stage1_model(X_tensor)
            y_pred = STATE.scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())

        # Compute metrics
        r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
        mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))

        # Store results
        STATE.inference_results = {
            'X': X,
            'y_true': y_true,
            'y_pred': y_pred,
            'start_idx': start_idx,
            'end_idx': end_idx
        }

        # Create results dataframe
        results_df = pd.DataFrame({
            'Index': range(start_idx, end_idx)
        })

        for i, signal in enumerate(STATE.target_signals):
            results_df[f'{signal}_true'] = y_true[:, i]
            results_df[f'{signal}_pred'] = y_pred[:, i]

        # Create plot
        fig, axes = plt.subplots(min(3, len(STATE.target_signals)), 1, figsize=(12, 8))
        if len(STATE.target_signals) == 1:
            axes = [axes]

        for i in range(min(3, len(STATE.target_signals))):
            axes[i].plot(y_true[:, i], label='True', linewidth=2)
            axes[i].plot(y_pred[:, i], label='Predicted', linewidth=2, alpha=0.7)
            axes[i].set_title(f'{STATE.target_signals[i]}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        metrics_text = f"📊 Inference Metrics:\n"
        metrics_text += f"   R² Score: {r2:.4f}\n"
        metrics_text += f"   MAE: {mae:.4f}\n"
        metrics_text += f"   RMSE: {rmse:.4f}\n"
        metrics_text += f"   Samples: {end_idx - start_idx}"

        return results_df, fig, metrics_text, "✅ Inference completed successfully!"

    except Exception as e:
        import traceback
        return None, None, None, f"❌ Error during inference:\n{traceback.format_exc()}"


# ============================================================================
# Tab 1: Gradient-Based Inverse Optimization
# ============================================================================

def create_target_settings_table(selected_targets):
    """Create editable target settings table"""
    if not selected_targets or STATE.inference_results is None:
        return pd.DataFrame()

    # Get current values from inference results
    y_pred = STATE.inference_results['y_pred']
    current_values = y_pred[0]  # Use first sample as baseline

    data = []
    for target_name in selected_targets:
        if target_name in STATE.target_signals:
            idx = STATE.target_signals.index(target_name)
            data.append({
                'Signal': target_name,
                'Current_Value': f"{current_values[idx]:.4f}",
                'Target_Bias_%': '0.0',  # User will edit this
                'Weight': '1.0'
            })

    return pd.DataFrame(data)


def create_variable_inputs_table(selected_inputs):
    """Create editable variable inputs table"""
    if not selected_inputs or STATE.inference_results is None:
        return pd.DataFrame()

    # Get current values from inference results
    X = STATE.inference_results['X']
    current_values = X[0]  # Use first sample as baseline

    data = []
    for input_name in selected_inputs:
        if input_name in STATE.boundary_signals:
            idx = STATE.boundary_signals.index(input_name)
            current_val = current_values[idx]

            # Suggest reasonable bounds (±50% by default)
            min_val = current_val * 0.5
            max_val = current_val * 1.5

            data.append({
                'Input': input_name,
                'Baseline': f"{current_val:.4f}",
                'Min': f"{min_val:.4f}",
                'Max': f"{max_val:.4f}",
                'Max_Change_%': '20.0'  # ±20% by default
            })

    return pd.DataFrame(data)


def run_inverse_optimization(
    targets_df,
    variables_df,
    learning_rate,
    max_epochs,
    optimizer_type,
    use_grad_clip,
    progress=gr.Progress()
):
    """Run gradient-based inverse optimization"""
    try:
        if STATE.stage1_model is None:
            return None, None, None, None, "❌ Please load model first"

        if targets_df is None or len(targets_df) == 0:
            return None, None, None, None, "❌ Please configure targets"

        if variables_df is None or len(variables_df) == 0:
            return None, None, None, None, "❌ Please configure variable inputs"

        # Parse targets
        targets = {}
        for _, row in targets_df.iterrows():
            signal_name = row['Signal']
            if signal_name in STATE.target_signals:
                idx = STATE.target_signals.index(signal_name)
                targets[idx] = {
                    'bias': float(row['Target_Bias_%']) / 100.0,
                    'weight': float(row['Weight'])
                }

        # Parse variable inputs and build constraints
        constraints_list = []
        optimizable_indices = []

        for _, row in variables_df.iterrows():
            input_name = row['Input']
            if input_name in STATE.boundary_signals:
                idx = STATE.boundary_signals.index(input_name)
                optimizable_indices.append(idx)

                constraints_list.append(InputConstraint(
                    name=input_name,
                    min_value=float(row['Min']),
                    max_value=float(row['Max']),
                    baseline_value=float(row['Baseline']),
                    max_change_rate=float(row['Max_Change_%']) / 100.0,
                    is_fixed=False
                ))

        # Add fixed constraints for other inputs
        X = STATE.inference_results['X']
        baseline_inputs = X[0]

        for i, signal_name in enumerate(STATE.boundary_signals):
            if i not in optimizable_indices:
                constraints_list.append(InputConstraint(
                    name=signal_name,
                    min_value=baseline_inputs[i],
                    max_value=baseline_inputs[i],
                    baseline_value=baseline_inputs[i],
                    is_fixed=True
                ))

        constraint_manager = ConstraintManager(constraints_list)

        # Create optimizer
        optimizer = InverseOptimizer(
            model=STATE.stage1_model,
            scaler_X=STATE.scaler_X,
            scaler_y=STATE.scaler_y,
            device=STATE.device
        )

        # Optimization config
        config = OptimizationConfig(
            learning_rate=learning_rate,
            max_epochs=int(max_epochs),
            optimizer_type=optimizer_type.lower(),
            gradient_clip=1.0 if use_grad_clip else 0.0,
            verbose=True
        )

        # Run optimization
        def callback(epoch, loss, inputs):
            if epoch % 10 == 0:
                progress((epoch, config.max_epochs), desc=f"Epoch {epoch}, Loss: {loss:.6f}")

        print("\n" + "="*80)
        print("Starting Inverse Optimization...")
        print("="*80)

        result = optimizer.optimize(
            targets=targets,
            constraint_manager=constraint_manager,
            initial_inputs=baseline_inputs,
            config=config,
            callback=callback
        )

        # Store result
        STATE.optimization_result = result

        # Create comparison table
        comparison_data = []
        for i, signal_name in enumerate(STATE.boundary_signals):
            if i in optimizable_indices:
                comparison_data.append({
                    'Input': signal_name,
                    'Baseline': f"{baseline_inputs[i]:.4f}",
                    'Optimized': f"{result['optimized_inputs'][i]:.4f}",
                    'Change_%': f"{(result['optimized_inputs'][i] - baseline_inputs[i]) / baseline_inputs[i] * 100:.2f}"
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Create loss plot
        fig_loss, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result['loss_history'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Optimization Loss Convergence')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()

        # Create objective comparison plot
        fig_obj, ax = plt.subplots(figsize=(10, 5))

        target_names = []
        current_values = []
        optimized_values = []
        target_values = []

        for _, row in targets_df.iterrows():
            signal_name = row['Signal']
            if signal_name in STATE.target_signals:
                idx = STATE.target_signals.index(signal_name)
                target_names.append(signal_name)

                # Current value
                current_values.append(float(row['Current_Value']))

                # Optimized value
                optimized_values.append(result['predictions'][idx])

                # Target value
                bias = float(row['Target_Bias_%']) / 100.0
                target_values.append(float(row['Current_Value']) * (1 + bias))

        x = np.arange(len(target_names))
        width = 0.25

        ax.bar(x - width, current_values, width, label='Current', alpha=0.7)
        ax.bar(x, optimized_values, width, label='Optimized', alpha=0.7)
        ax.bar(x + width, target_values, width, label='Target', alpha=0.7)

        ax.set_xlabel('Target Signal')
        ax.set_ylabel('Value')
        ax.set_title('Optimization Objectives Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        status_msg = f"✅ Optimization completed!\n"
        status_msg += f"   Epochs: {result['num_epochs']}\n"
        status_msg += f"   Final loss: {result['final_loss']:.6f}\n"
        status_msg += f"   Elapsed time: {result['elapsed_time']:.2f}s\n"
        status_msg += f"   Converged: {'Yes' if result['converged'] else 'No'}"

        return comparison_df, fig_loss, fig_obj, status_msg

    except Exception as e:
        import traceback
        return None, None, None, f"❌ Error:\n{traceback.format_exc()}"


# ============================================================================
# Tab 2: Multi-Objective Optimization
# ============================================================================

def run_pareto_generation(
    obj1_signal,
    obj1_bias,
    obj2_signal,
    obj2_bias,
    n_points,
    variables_df,
    progress=gr.Progress()
):
    """Generate Pareto frontier"""
    try:
        if STATE.stage1_model is None:
            return None, None, "❌ Please load model first"

        if variables_df is None or len(variables_df) == 0:
            return None, None, "❌ Please configure variable inputs"

        # Build objectives
        objectives = []
        for signal_name, bias in [(obj1_signal, obj1_bias), (obj2_signal, obj2_bias)]:
            if signal_name in STATE.target_signals:
                idx = STATE.target_signals.index(signal_name)
                objectives.append({
                    'signal_idx': idx,
                    'bias': bias / 100.0
                })

        # Build constraints (reuse from variables_df)
        constraints_list = []
        optimizable_indices = []

        for _, row in variables_df.iterrows():
            input_name = row['Input']
            if input_name in STATE.boundary_signals:
                idx = STATE.boundary_signals.index(input_name)
                optimizable_indices.append(idx)

                constraints_list.append(InputConstraint(
                    name=input_name,
                    min_value=float(row['Min']),
                    max_value=float(row['Max']),
                    baseline_value=float(row['Baseline']),
                    max_change_rate=float(row['Max_Change_%']) / 100.0,
                    is_fixed=False
                ))

        X = STATE.inference_results['X']
        baseline_inputs = X[0]

        for i, signal_name in enumerate(STATE.boundary_signals):
            if i not in optimizable_indices:
                constraints_list.append(InputConstraint(
                    name=signal_name,
                    min_value=baseline_inputs[i],
                    max_value=baseline_inputs[i],
                    baseline_value=baseline_inputs[i],
                    is_fixed=True
                ))

        constraint_manager = ConstraintManager(constraints_list)

        # Create multi-objective optimizer
        mo_optimizer = MultiObjectiveOptimizer(
            model=STATE.stage1_model,
            scaler_X=STATE.scaler_X,
            scaler_y=STATE.scaler_y,
            device=STATE.device
        )

        # Configuration
        mo_config = MultiObjectiveConfig(
            n_pareto_points=int(n_points),
            base_config=OptimizationConfig(
                max_epochs=300,
                verbose=False
            )
        )

        # Generate Pareto frontier
        pareto_results = mo_optimizer.generate_pareto_frontier(
            objectives=objectives,
            constraint_manager=constraint_manager,
            initial_inputs=baseline_inputs,
            config=mo_config
        )

        STATE.pareto_results = pareto_results

        # Create Pareto plot
        try:
            fig = mo_optimizer.plot_pareto_frontier(
                pareto_results,
                objective_names=[obj1_signal, obj2_signal],
                use_plotly=True
            )
        except:
            # Fallback to matplotlib
            fig = mo_optimizer.plot_pareto_frontier(
                pareto_results,
                objective_names=[obj1_signal, obj2_signal],
                use_plotly=False
            )

        # Create solutions table
        solutions_data = []
        for i, point in enumerate(pareto_results['pareto_points']):
            solutions_data.append({
                'Solution_ID': i,
                'Weight_1': f"{point['weight_1']:.2f}",
                'Weight_2': f"{point['weight_2']:.2f}",
                f'{obj1_signal}': f"{point['objective_1']:.4f}",
                f'{obj2_signal}': f"{point['objective_2']:.4f}",
                'Converged': point['converged']
            })

        solutions_df = pd.DataFrame(solutions_data)

        status_msg = f"✅ Pareto frontier generated!\n"
        status_msg += f"   Points: {len(pareto_results['pareto_points'])}\n"
        status_msg += f"   Objectives: {obj1_signal} vs {obj2_signal}"

        return fig, solutions_df, status_msg

    except Exception as e:
        import traceback
        return None, None, f"❌ Error:\n{traceback.format_exc()}"


# ============================================================================
# Tab 3: Kalman Filter Correction
# ============================================================================

def run_kalman_simulation(
    process_noise,
    measurement_noise,
    n_steps,
    variables_df
):
    """Run Kalman filter simulation"""
    try:
        if STATE.stage1_model is None:
            return None, None, "❌ Please load model first"

        if STATE.optimization_result is None:
            return None, None, "❌ Please run optimization first (Tab 1)"

        if variables_df is None or len(variables_df) == 0:
            return None, None, "❌ Please configure variable inputs"

        # Get optimizable indices
        optimizable_indices = []
        for _, row in variables_df.iterrows():
            input_name = row['Input']
            if input_name in STATE.boundary_signals:
                idx = STATE.boundary_signals.index(input_name)
                optimizable_indices.append(idx)

        # Get target indices (from optimization result)
        target_indices = list(STATE.optimization_result['targets'].keys())

        # Get fixed input values
        optimized_inputs = STATE.optimization_result['optimized_inputs']

        # Create Kalman corrector
        kf = KalmanCorrector(
            model=STATE.stage1_model,
            scaler_X=STATE.scaler_X,
            scaler_y=STATE.scaler_y,
            optimizable_input_indices=optimizable_indices,
            target_output_indices=target_indices,
            fixed_input_values=optimized_inputs,
            device=STATE.device
        )

        # Generate synthetic measurements with noise
        predictions_clean = STATE.optimization_result['predictions'][target_indices]

        np.random.seed(42)
        measurements = []
        for _ in range(n_steps):
            noise = np.random.normal(0, measurement_noise, len(target_indices))
            measurements.append(predictions_clean + noise * predictions_clean)
        measurements = np.array(measurements)

        # Initial state
        initial_inputs = optimized_inputs[optimizable_indices]
        initial_outputs = predictions_clean
        initial_state = np.concatenate([initial_inputs, initial_outputs])

        # Run simulation
        kf_config = KalmanConfig(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )

        sim_results = kf.run_simulation(
            initial_state=initial_state,
            measurements=measurements,
            config=kf_config
        )

        # Uncorrected predictions (constant)
        uncorrected = np.tile(predictions_clean, (n_steps, 1))

        # Create plot
        fig = KalmanCorrector.plot_correction_results(
            simulation_results=sim_results,
            uncorrected_predictions=uncorrected,
            target_names=[STATE.target_signals[i] for i in target_indices]
        )

        # Compute metrics
        metrics_df = kf.compute_correction_metrics(sim_results, uncorrected)

        status_msg = f"✅ Kalman simulation completed!\n"
        status_msg += f"   Steps: {n_steps}\n"
        status_msg += f"   Avg RMSE improvement: {metrics_df['RMSE_Improvement_%'].mean():.2f}%"

        return fig, metrics_df, status_msg

    except Exception as e:
        import traceback
        return None, None, f"❌ Error:\n{traceback.format_exc()}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="Inverse Control Optimization System", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🎯 Inverse Control Optimization System

        **Gradient-based inverse optimization for industrial digital twin models**

        Use this interface to:
        - Load trained models and run basic inference
        - Optimize input parameters to achieve target outputs
        - Explore multi-objective trade-offs (Pareto frontier)
        - Apply Kalman filtering for real-time correction
        """)

        with gr.Tabs():

            # ================================================================
            # Tab 0: Model Loading & Basic Inference
            # ================================================================

            with gr.Tab("📂 Model Loading & Inference"):
                gr.Markdown("### Load trained model and run basic inference")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model Configuration")

                        stage1_model_path = gr.Textbox(
                            label="Stage1 Model Path",
                            value="saved_models/stage1_model.pth",
                            placeholder="Path to .pth file"
                        )

                        load_model_btn = gr.Button("🔄 Load Model", variant="primary")

                        model_status = gr.Textbox(
                            label="Model Status",
                            lines=5,
                            interactive=False
                        )

                        gr.Markdown("#### Data Configuration")

                        data_file_path = gr.Textbox(
                            label="Data File Path",
                            value="data/data.csv",
                            placeholder="Path to CSV file"
                        )

                        load_data_btn = gr.Button("📁 Load Data", variant="primary")

                        data_status = gr.Textbox(
                            label="Data Status",
                            lines=3,
                            interactive=False
                        )

                        data_preview = gr.Dataframe(
                            label="Data Preview (first 10 rows)",
                            interactive=False
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Inference Configuration")

                        with gr.Row():
                            start_idx = gr.Number(
                                label="Start Index",
                                value=0,
                                precision=0
                            )
                            end_idx = gr.Number(
                                label="End Index",
                                value=100,
                                precision=0
                            )

                        use_ensemble = gr.Checkbox(
                            label="Use Ensemble Model (Stage1+Stage2)",
                            value=False
                        )

                        run_inference_btn = gr.Button("▶️ Run Inference", variant="primary")

                        inference_status = gr.Textbox(
                            label="Inference Status",
                            lines=3,
                            interactive=False
                        )

                        metrics_display = gr.Textbox(
                            label="Performance Metrics",
                            lines=6,
                            interactive=False
                        )

                        inference_plot = gr.Plot(label="Prediction vs Actual")

                        inference_results = gr.Dataframe(
                            label="Inference Results",
                            interactive=False
                        )

                # Connect Tab 0 buttons
                load_model_btn.click(
                    fn=load_stage1_model,
                    inputs=[stage1_model_path],
                    outputs=[
                        gr.Dropdown(visible=False),  # Placeholder for boundary signals
                        gr.Dropdown(visible=False),  # Placeholder for target signals (Tab1)
                        gr.Dropdown(visible=False),  # Placeholder for target signals (Tab2)
                        model_status
                    ]
                )

                load_data_btn.click(
                    fn=load_data_file,
                    inputs=[data_file_path],
                    outputs=[data_preview, data_status]
                )

                run_inference_btn.click(
                    fn=run_basic_inference,
                    inputs=[start_idx, end_idx, use_ensemble],
                    outputs=[inference_results, inference_plot, metrics_display, inference_status]
                )

            # ================================================================
            # Tab 1: Gradient-Based Inverse Optimization
            # ================================================================

            with gr.Tab("🎯 Inverse Optimization"):
                gr.Markdown("### Optimize input parameters to achieve target outputs")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Target Configuration")

                        selected_targets = gr.Dropdown(
                            label="Select Target Signals",
                            choices=[],
                            multiselect=True
                        )

                        create_targets_btn = gr.Button("Create Targets Table")

                        targets_table = gr.Dataframe(
                            label="Target Settings (Editable)",
                            interactive=True,
                            headers=['Signal', 'Current_Value', 'Target_Bias_%', 'Weight'],
                            col_count=(4, "fixed")
                        )

                        gr.Markdown("""
                        **Instructions:**
                        - `Target_Bias_%`: Desired change (e.g., -10 for -10% reduction)
                        - `Weight`: Relative importance (higher = more important)
                        """)

                        gr.Markdown("#### Variable Inputs Configuration")

                        selected_inputs = gr.Dropdown(
                            label="Select Variable Inputs",
                            choices=[],
                            multiselect=True
                        )

                        create_inputs_btn = gr.Button("Create Inputs Table")

                        variables_table = gr.Dataframe(
                            label="Variable Inputs Settings (Editable)",
                            interactive=True,
                            headers=['Input', 'Baseline', 'Min', 'Max', 'Max_Change_%'],
                            col_count=(5, "fixed")
                        )

                        gr.Markdown("""
                        **Instructions:**
                        - `Min/Max`: Absolute bounds
                        - `Max_Change_%`: Maximum relative change from baseline
                        """)

                        gr.Markdown("#### Optimization Parameters")

                        learning_rate = gr.Slider(
                            label="Learning Rate",
                            minimum=0.001,
                            maximum=0.1,
                            value=0.01,
                            step=0.001
                        )

                        max_epochs = gr.Slider(
                            label="Max Epochs",
                            minimum=100,
                            maximum=2000,
                            value=500,
                            step=50
                        )

                        optimizer_type = gr.Radio(
                            label="Optimizer Type",
                            choices=["Adam", "SGD"],
                            value="Adam"
                        )

                        use_grad_clip = gr.Checkbox(
                            label="Use Gradient Clipping",
                            value=True
                        )

                        run_opt_btn = gr.Button("🚀 Run Optimization", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Optimization Results")

                        opt_status = gr.Textbox(
                            label="Optimization Status",
                            lines=6,
                            interactive=False
                        )

                        loss_plot = gr.Plot(label="Loss Convergence")

                        comparison_table = gr.Dataframe(
                            label="Input Changes",
                            interactive=False
                        )

                        objectives_plot = gr.Plot(label="Objectives Comparison")

                # Connect Tab 1 buttons
                # Note: We need to store selected_targets and selected_inputs choices from Tab 0
                # This will be handled when model loads

                create_targets_btn.click(
                    fn=create_target_settings_table,
                    inputs=[selected_targets],
                    outputs=[targets_table]
                )

                create_inputs_btn.click(
                    fn=create_variable_inputs_table,
                    inputs=[selected_inputs],
                    outputs=[variables_table]
                )

                run_opt_btn.click(
                    fn=run_inverse_optimization,
                    inputs=[
                        targets_table,
                        variables_table,
                        learning_rate,
                        max_epochs,
                        optimizer_type,
                        use_grad_clip
                    ],
                    outputs=[comparison_table, loss_plot, objectives_plot, opt_status]
                )

            # ================================================================
            # Tab 2: Multi-Objective Optimization
            # ================================================================

            with gr.Tab("📊 Multi-Objective Trade-offs"):
                gr.Markdown("### Explore Pareto frontier for conflicting objectives")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Objective 1")

                        obj1_signal = gr.Dropdown(
                            label="Target Signal 1",
                            choices=[]
                        )

                        obj1_bias = gr.Number(
                            label="Target Bias % (e.g., -10 for -10%)",
                            value=-10.0
                        )

                        gr.Markdown("#### Objective 2")

                        obj2_signal = gr.Dropdown(
                            label="Target Signal 2",
                            choices=[]
                        )

                        obj2_bias = gr.Number(
                            label="Target Bias % (e.g., +5 for +5%)",
                            value=5.0
                        )

                        gr.Markdown("#### Pareto Configuration")

                        n_pareto_points = gr.Slider(
                            label="Number of Pareto Points",
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=5
                        )

                        gr.Markdown("**Note:** Uses variable inputs from Tab 1")

                        run_pareto_btn = gr.Button("🔬 Generate Pareto Frontier", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Pareto Frontier Results")

                        pareto_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False
                        )

                        pareto_plot = gr.Plot(label="Pareto Frontier")

                        pareto_solutions = gr.Dataframe(
                            label="Pareto Solutions",
                            interactive=False
                        )

                # Connect Tab 2 buttons
                run_pareto_btn.click(
                    fn=run_pareto_generation,
                    inputs=[
                        obj1_signal,
                        obj1_bias,
                        obj2_signal,
                        obj2_bias,
                        n_pareto_points,
                        variables_table  # Reuse from Tab 1
                    ],
                    outputs=[pareto_plot, pareto_solutions, pareto_status]
                )

            # ================================================================
            # Tab 3: Kalman Filter Correction
            # ================================================================

            with gr.Tab("🔧 Kalman Correction"):
                gr.Markdown("### Real-time correction with Kalman filtering")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Kalman Filter Configuration")

                        process_noise = gr.Slider(
                            label="Process Noise (Q)",
                            minimum=0.001,
                            maximum=0.1,
                            value=0.01,
                            step=0.001
                        )

                        measurement_noise = gr.Slider(
                            label="Measurement Noise (R)",
                            minimum=0.01,
                            maximum=1.0,
                            value=0.1,
                            step=0.01
                        )

                        n_steps = gr.Slider(
                            label="Simulation Steps",
                            minimum=10,
                            maximum=200,
                            value=50,
                            step=10
                        )

                        gr.Markdown("**Note:** Uses optimized plan from Tab 1")

                        run_kalman_btn = gr.Button("▶️ Run Kalman Simulation", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Correction Results")

                        kalman_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False
                        )

                        kalman_plot = gr.Plot(label="Real-time Correction")

                        kalman_metrics = gr.Dataframe(
                            label="Correction Metrics",
                            interactive=False
                        )

                # Connect Tab 3 buttons
                run_kalman_btn.click(
                    fn=run_kalman_simulation,
                    inputs=[
                        process_noise,
                        measurement_noise,
                        n_steps,
                        variables_table  # Reuse from Tab 1
                    ],
                    outputs=[kalman_plot, kalman_metrics, kalman_status]
                )

        # Update dropdowns when model is loaded
        # This is a workaround to propagate choices to Tab 1 and Tab 2
        load_model_btn.click(
            fn=load_stage1_model,
            inputs=[stage1_model_path],
            outputs=[selected_inputs, selected_targets, obj1_signal, model_status]
        ).then(
            fn=lambda: STATE.target_signals,
            outputs=[obj2_signal]
        )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
