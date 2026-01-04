#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Script for Inverse Optimization System

This script provides a simple demonstration of the inverse optimization
functionality with synthetic data.
"""

import torch
import numpy as np
from optimization import (
    InverseOptimizer,
    ConstraintManager,
    InputConstraint,
    OptimizationConfig
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def create_synthetic_model():
    """Create a simple synthetic model for demonstration"""
    from models.static_transformer import StaticSensorTransformer
    from sklearn.preprocessing import StandardScaler

    # Create small model
    model = StaticSensorTransformer(
        num_boundary_sensors=5,
        num_target_sensors=3,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    # Create synthetic scalers
    scaler_X = StandardScaler()
    scaler_X.mean_ = np.array([100, 50, 75, 200, 150])
    scaler_X.scale_ = np.array([10, 5, 8, 20, 15])
    scaler_X.n_features_in_ = 5

    scaler_y = StandardScaler()
    scaler_y.mean_ = np.array([300, 400, 500])
    scaler_y.scale_ = np.array([30, 40, 50])
    scaler_y.n_features_in_ = 3

    return model, scaler_X, scaler_y


def demo_single_objective_optimization():
    """Demonstrate single-objective optimization"""
    print("\n" + "="*80)
    print("DEMO 1: Single-Objective Optimization")
    print("="*80)
    print("Objective: Reduce Target Signal 0 by 10%")
    print("Variable Inputs: Input 0, 1, 2 (others fixed)")
    print("-"*80)

    # Create synthetic model
    model, scaler_X, scaler_y = create_synthetic_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Create optimizer
    optimizer = InverseOptimizer(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        device=device
    )

    # Define baseline inputs
    baseline_inputs = np.array([100.0, 50.0, 75.0, 200.0, 150.0])

    # Define constraints
    constraints = []
    for i in range(5):
        if i < 3:  # First 3 are optimizable
            constraints.append(InputConstraint(
                name=f'Input_{i}',
                min_value=baseline_inputs[i] * 0.5,
                max_value=baseline_inputs[i] * 1.5,
                baseline_value=baseline_inputs[i],
                max_change_rate=0.20,
                is_fixed=False
            ))
        else:  # Rest are fixed
            constraints.append(InputConstraint(
                name=f'Input_{i}',
                min_value=baseline_inputs[i],
                max_value=baseline_inputs[i],
                baseline_value=baseline_inputs[i],
                is_fixed=True
            ))

    constraint_manager = ConstraintManager(constraints)

    # Define target: reduce target signal 0 by 10%
    targets = {
        0: {
            'bias': -0.10,  # -10% reduction
            'weight': 1.0
        }
    }

    # Optimization configuration
    config = OptimizationConfig(
        learning_rate=0.01,
        max_epochs=300,
        optimizer_type='adam',
        patience=30,
        verbose=True
    )

    # Run optimization
    print("\n🚀 Starting optimization...")
    result = optimizer.optimize(
        targets=targets,
        constraint_manager=constraint_manager,
        initial_inputs=baseline_inputs,
        config=config
    )

    # Display results
    print("\n" + "="*80)
    print("✅ OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Converged: {result['converged']}")
    print(f"Epochs: {result['num_epochs']}")
    print(f"Final Loss: {result['final_loss']:.6f}")
    print(f"Elapsed Time: {result['elapsed_time']:.2f}s")

    print("\n📊 Input Changes:")
    print("-"*80)
    for i in range(3):  # Show only optimizable inputs
        baseline_val = baseline_inputs[i]
        optimized_val = result['optimized_inputs'][i]
        change_pct = (optimized_val - baseline_val) / baseline_val * 100
        print(f"  Input_{i}: {baseline_val:.2f} → {optimized_val:.2f} ({change_pct:+.2f}%)")

    print("\n🎯 Target Achievement:")
    print("-"*80)
    # Get baseline prediction
    baseline_pred = optimizer.predict(baseline_inputs)
    optimized_pred = result['predictions']

    target_val = baseline_pred[0] * 0.9  # -10%
    achievement = (baseline_pred[0] - optimized_pred[0]) / (baseline_pred[0] - target_val) * 100

    print(f"  Target Signal 0:")
    print(f"    Baseline:  {baseline_pred[0]:.2f}")
    print(f"    Optimized: {optimized_pred[0]:.2f}")
    print(f"    Target:    {target_val:.2f}")
    print(f"    Achievement: {achievement:.1f}%")

    return result


def demo_multi_objective_optimization():
    """Demonstrate multi-objective optimization"""
    print("\n" + "="*80)
    print("DEMO 2: Multi-Objective Optimization (Pareto Frontier)")
    print("="*80)
    print("Objective 1: Reduce Target Signal 0 by 10%")
    print("Objective 2: Increase Target Signal 1 by 5%")
    print("-"*80)

    from optimization import MultiObjectiveOptimizer, MultiObjectiveConfig

    # Create synthetic model
    model, scaler_X, scaler_y = create_synthetic_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Define baseline inputs
    baseline_inputs = np.array([100.0, 50.0, 75.0, 200.0, 150.0])

    # Define constraints (same as demo 1)
    constraints = []
    for i in range(5):
        if i < 3:
            constraints.append(InputConstraint(
                name=f'Input_{i}',
                min_value=baseline_inputs[i] * 0.5,
                max_value=baseline_inputs[i] * 1.5,
                baseline_value=baseline_inputs[i],
                max_change_rate=0.20,
                is_fixed=False
            ))
        else:
            constraints.append(InputConstraint(
                name=f'Input_{i}',
                min_value=baseline_inputs[i],
                max_value=baseline_inputs[i],
                baseline_value=baseline_inputs[i],
                is_fixed=True
            ))

    constraint_manager = ConstraintManager(constraints)

    # Create multi-objective optimizer
    mo_optimizer = MultiObjectiveOptimizer(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        device=device
    )

    # Define objectives
    objectives = [
        {'signal_idx': 0, 'bias': -0.10},  # Reduce by 10%
        {'signal_idx': 1, 'bias': 0.05}    # Increase by 5%
    ]

    # Configuration
    mo_config = MultiObjectiveConfig(
        n_pareto_points=10,  # Small number for demo
        base_config=OptimizationConfig(
            max_epochs=200,
            verbose=False
        )
    )

    print("\n🚀 Generating Pareto frontier (10 points)...")
    pareto_results = mo_optimizer.generate_pareto_frontier(
        objectives=objectives,
        constraint_manager=constraint_manager,
        initial_inputs=baseline_inputs,
        config=mo_config
    )

    print("\n" + "="*80)
    print("✅ PARETO FRONTIER GENERATED")
    print("="*80)
    print(f"Total solutions: {len(pareto_results['pareto_points'])}")

    print("\n📊 Sample Solutions:")
    print("-"*80)
    print(f"{'ID':<4} {'Weight1':<8} {'Weight2':<8} {'Obj1':<12} {'Obj2':<12} {'Conv':<6}")
    print("-"*80)

    for i in range(min(5, len(pareto_results['pareto_points']))):
        point = pareto_results['pareto_points'][i]
        print(f"{i:<4} {point['weight_1']:<8.2f} {point['weight_2']:<8.2f} "
              f"{point['objective_1']:<12.2f} {point['objective_2']:<12.2f} "
              f"{'Yes' if point['converged'] else 'No':<6}")

    return pareto_results


def demo_constraint_validation():
    """Demonstrate constraint management"""
    print("\n" + "="*80)
    print("DEMO 3: Constraint Management")
    print("="*80)

    # Create constraints
    constraints = [
        InputConstraint(
            name='Fuel_Flow',
            min_value=50.0,
            max_value=150.0,
            baseline_value=100.0,
            max_change_rate=0.20,
            is_fixed=False
        ),
        InputConstraint(
            name='Air_Flow',
            min_value=200.0,
            max_value=400.0,
            baseline_value=300.0,
            max_change_rate=0.15,
            is_fixed=False
        ),
        InputConstraint(
            name='Temperature',
            min_value=500.0,
            max_value=500.0,
            baseline_value=500.0,
            is_fixed=True
        )
    ]

    constraint_manager = ConstraintManager(constraints)

    print(constraint_manager.get_constraint_summary())

    # Test constraint projection
    print("\n🔧 Testing Constraint Projection:")
    print("-"*80)

    test_values = np.array([130.0, 350.0, 550.0])  # Some violate constraints
    print(f"Input values (before projection): {test_values}")

    projected = constraint_manager.project(torch.from_numpy(test_values).float()).numpy()
    print(f"Projected values (after):         {projected}")

    is_valid = constraint_manager.validate_values(projected)
    print(f"Valid after projection: {is_valid}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("🎯 INVERSE OPTIMIZATION SYSTEM - QUICK START DEMO")
    print("="*80)
    print("\nThis script demonstrates the inverse optimization capabilities:")
    print("  1. Single-objective optimization")
    print("  2. Multi-objective Pareto frontier")
    print("  3. Constraint management")
    print("\nNote: Using synthetic model for demonstration purposes.")

    try:
        # Demo 1: Single-objective
        demo_single_objective_optimization()

        # Demo 2: Multi-objective
        demo_multi_objective_optimization()

        # Demo 3: Constraints
        demo_constraint_validation()

        print("\n" + "="*80)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📚 Next Steps:")
        print("  1. Try with your own trained model")
        print("  2. Launch Gradio interface: python gradio_apps/inverse_control_interface.py")
        print("  3. Check examples/inverse_optimization_example.ipynb")
        print("  4. Read optimization/README.md for full documentation")
        print("\n" + "="*80)

    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
