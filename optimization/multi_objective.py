"""
Multi-objective optimization and Pareto frontier generation

This module extends inverse optimization to handle multiple conflicting objectives,
generating Pareto-optimal solutions through weighted optimization.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .inverse_optimizer import InverseOptimizer
from .config import OptimizationConfig, MultiObjectiveConfig
from .constraints import ConstraintManager


class MultiObjectiveOptimizer(InverseOptimizer):
    """
    Multi-objective inverse optimizer

    Extends InverseOptimizer to handle multiple conflicting objectives by:
    1. Scanning weight space (w1, w2, ..., wn) where Σwi = 1
    2. Running inverse optimization for each weight combination
    3. Collecting Pareto-optimal solutions
    4. Visualizing Pareto frontier
    """

    def generate_pareto_frontier(
        self,
        objectives: List[Dict[str, Any]],
        constraint_manager: ConstraintManager,
        initial_inputs: Optional[np.ndarray] = None,
        config: Optional[MultiObjectiveConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate Pareto frontier for two-objective optimization

        Args:
            objectives: List of objective specifications, each containing:
                       - 'signal_idx': Index of target signal
                       - 'value': Target value (optional, mutually exclusive with bias)
                       - 'bias': Relative change (optional, mutually exclusive with value)
            constraint_manager: Constraint manager for inputs
            initial_inputs: Initial input values (original scale), optional
            config: Multi-objective configuration, optional

        Returns:
            Dictionary containing:
                - 'pareto_points': List of solution dictionaries
                - 'weights': Weight combinations used
                - 'objectives': Objective values for each solution
                - 'inputs': Input values for each solution
        """
        if config is None:
            config = MultiObjectiveConfig()

        if len(objectives) != 2:
            raise ValueError("Currently only 2-objective optimization is supported")

        # Generate weight combinations
        weights = np.linspace(config.weight_min, config.weight_max, config.n_pareto_points)

        # Storage for results
        pareto_points = []
        all_weights = []
        all_objectives = []
        all_inputs = []

        print(f"\nGenerating Pareto frontier with {config.n_pareto_points} points...")
        print("=" * 80)

        # Run optimization for each weight combination
        for i, w1 in enumerate(weights):
            w2 = 1.0 - w1

            # Prepare targets with weights
            targets = {}
            for obj_idx, obj_spec in enumerate(objectives):
                weight = w1 if obj_idx == 0 else w2

                target_dict = {
                    'weight': weight
                }

                if 'value' in obj_spec:
                    target_dict['value'] = obj_spec['value']
                elif 'bias' in obj_spec:
                    target_dict['bias'] = obj_spec['bias']
                else:
                    raise ValueError(f"Objective {obj_idx} must have 'value' or 'bias'")

                targets[obj_spec['signal_idx']] = target_dict

            # Run inverse optimization
            result = self.optimize(
                targets=targets,
                constraint_manager=constraint_manager,
                initial_inputs=initial_inputs,
                config=config.base_config,
                callback=None
            )

            # Extract objective values
            obj_values = [
                result['predictions'][obj['signal_idx']]
                for obj in objectives
            ]

            # Store results
            pareto_points.append({
                'weight_1': w1,
                'weight_2': w2,
                'objective_1': obj_values[0],
                'objective_2': obj_values[1],
                'inputs': result['optimized_inputs'].copy(),
                'loss': result['final_loss'],
                'converged': result['converged']
            })

            all_weights.append([w1, w2])
            all_objectives.append(obj_values)
            all_inputs.append(result['optimized_inputs'])

            # Print progress
            print(f"Point {i+1:3d}/{config.n_pareto_points} | "
                  f"Weights: [{w1:.2f}, {w2:.2f}] | "
                  f"Obj1: {obj_values[0]:8.3f} | "
                  f"Obj2: {obj_values[1]:8.3f} | "
                  f"Loss: {result['final_loss']:.6f}")

        print("=" * 80)
        print("Pareto frontier generation completed!\n")

        return {
            'pareto_points': pareto_points,
            'weights': np.array(all_weights),
            'objectives': np.array(all_objectives),
            'inputs': np.array(all_inputs),
            'objective_specs': objectives,
            'config': config
        }

    def plot_pareto_frontier(
        self,
        pareto_results: Dict[str, Any],
        objective_names: Optional[List[str]] = None,
        use_plotly: bool = True
    ) -> Any:
        """
        Visualize Pareto frontier

        Args:
            pareto_results: Results from generate_pareto_frontier()
            objective_names: Names for objectives (optional)
            use_plotly: Use interactive Plotly plot (default: True)

        Returns:
            Plotly figure if use_plotly=True, else Matplotlib figure
        """
        objectives = pareto_results['objectives']
        weights = pareto_results['weights']
        pareto_points = pareto_results['pareto_points']

        if objective_names is None:
            objective_names = ['Objective 1', 'Objective 2']

        if use_plotly and PLOTLY_AVAILABLE:
            return self._plot_pareto_plotly(
                objectives, weights, pareto_points, objective_names
            )
        else:
            return self._plot_pareto_matplotlib(
                objectives, weights, pareto_points, objective_names
            )

    def _plot_pareto_plotly(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
        pareto_points: List[Dict],
        objective_names: List[str]
    ):
        """Create interactive Plotly scatter plot"""
        # Prepare hover text
        hover_texts = []
        for point in pareto_points:
            text = (
                f"Weight 1: {point['weight_1']:.2f}<br>"
                f"Weight 2: {point['weight_2']:.2f}<br>"
                f"{objective_names[0]}: {point['objective_1']:.3f}<br>"
                f"{objective_names[1]}: {point['objective_2']:.3f}<br>"
                f"Converged: {point['converged']}"
            )
            hover_texts.append(text)

        # Create scatter plot
        fig = go.Figure()

        # Add Pareto points
        fig.add_trace(go.Scatter(
            x=objectives[:, 0],
            y=objectives[:, 1],
            mode='markers+lines',
            marker=dict(
                size=10,
                color=weights[:, 0],  # Color by weight_1
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Weight 1"),
                line=dict(width=1, color='white')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Pareto Frontier'
        ))

        # Update layout
        fig.update_layout(
            title='Pareto Frontier - Multi-Objective Optimization',
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            hovermode='closest',
            width=800,
            height=600,
            template='plotly_white'
        )

        return fig

    def _plot_pareto_matplotlib(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
        pareto_points: List[Dict],
        objective_names: List[str]
    ):
        """Create static Matplotlib plot"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Create scatter plot
        scatter = ax.scatter(
            objectives[:, 0],
            objectives[:, 1],
            c=weights[:, 0],
            cmap='viridis',
            s=100,
            edgecolors='white',
            linewidth=1.5,
            alpha=0.8
        )

        # Add line connecting points
        ax.plot(
            objectives[:, 0],
            objectives[:, 1],
            'k--',
            alpha=0.3,
            linewidth=1
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Weight 1', rotation=270, labelpad=20)

        # Labels and title
        ax.set_xlabel(objective_names[0], fontsize=12)
        ax.set_ylabel(objective_names[1], fontsize=12)
        ax.set_title('Pareto Frontier - Multi-Objective Optimization', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def export_pareto_solutions(
        self,
        pareto_results: Dict[str, Any],
        filepath: str,
        input_names: Optional[List[str]] = None
    ):
        """
        Export Pareto solutions to CSV

        Args:
            pareto_results: Results from generate_pareto_frontier()
            filepath: Output CSV file path
            input_names: Names for input variables (optional)
        """
        pareto_points = pareto_results['pareto_points']
        n_inputs = pareto_results['inputs'].shape[1]

        if input_names is None:
            input_names = [f'Input_{i}' for i in range(n_inputs)]

        # Build dataframe
        data = []
        for point in pareto_points:
            row = {
                'Weight_1': point['weight_1'],
                'Weight_2': point['weight_2'],
                'Objective_1': point['objective_1'],
                'Objective_2': point['objective_2'],
                'Loss': point['loss'],
                'Converged': point['converged']
            }

            # Add input values
            for i, name in enumerate(input_names):
                row[name] = point['inputs'][i]

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        print(f"Pareto solutions exported to: {filepath}")

    def select_solution(
        self,
        pareto_results: Dict[str, Any],
        index: int
    ) -> Dict[str, Any]:
        """
        Select a specific solution from Pareto frontier

        Args:
            pareto_results: Results from generate_pareto_frontier()
            index: Index of solution to select

        Returns:
            Selected solution dictionary
        """
        if not (0 <= index < len(pareto_results['pareto_points'])):
            raise ValueError(f"Index {index} out of range [0, {len(pareto_results['pareto_points'])-1}]")

        return pareto_results['pareto_points'][index]

    def compare_solutions(
        self,
        pareto_results: Dict[str, Any],
        indices: List[int],
        input_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple Pareto solutions

        Args:
            pareto_results: Results from generate_pareto_frontier()
            indices: List of solution indices to compare
            input_names: Names for input variables (optional)

        Returns:
            Comparison dataframe
        """
        n_inputs = pareto_results['inputs'].shape[1]

        if input_names is None:
            input_names = [f'Input_{i}' for i in range(n_inputs)]

        # Extract solutions
        solutions = [pareto_results['pareto_points'][i] for i in indices]

        # Build comparison table
        data = {
            'Solution': [f'Solution_{i}' for i in indices],
            'Weight_1': [s['weight_1'] for s in solutions],
            'Weight_2': [s['weight_2'] for s in solutions],
            'Objective_1': [s['objective_1'] for s in solutions],
            'Objective_2': [s['objective_2'] for s in solutions],
        }

        # Add input values
        for i, name in enumerate(input_names):
            data[name] = [s['inputs'][i] for s in solutions]

        df = pd.DataFrame(data)
        return df
