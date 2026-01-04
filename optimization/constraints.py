"""
Constraint management for inverse optimization
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class InputConstraint:
    """
    Constraint specification for a single input variable

    Attributes:
        name: Input variable name
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        baseline_value: Baseline/reference value for change rate calculation
        max_change_rate: Maximum change rate relative to baseline (e.g., 0.2 for ±20%)
        is_fixed: Whether this input is fixed (not optimizable)
    """

    name: str
    min_value: float
    max_value: float
    baseline_value: Optional[float] = None
    max_change_rate: Optional[float] = None
    is_fixed: bool = False

    def __post_init__(self):
        """Validate constraint parameters"""
        if self.min_value >= self.max_value:
            raise ValueError(f"{self.name}: min_value must be less than max_value")

        if self.baseline_value is not None:
            if not (self.min_value <= self.baseline_value <= self.max_value):
                raise ValueError(
                    f"{self.name}: baseline_value must be within [min_value, max_value]"
                )

        if self.max_change_rate is not None:
            if self.max_change_rate <= 0:
                raise ValueError(f"{self.name}: max_change_rate must be positive")
            if self.baseline_value is None:
                raise ValueError(
                    f"{self.name}: baseline_value required when max_change_rate is specified"
                )

    def project_value(self, value: float) -> float:
        """
        Project a single value to satisfy this constraint

        Args:
            value: Input value to project

        Returns:
            Projected value within constraints
        """
        # Hard constraints
        projected = max(self.min_value, min(self.max_value, value))

        # Change rate constraints
        if self.max_change_rate is not None and self.baseline_value is not None:
            max_deviation = abs(self.baseline_value * self.max_change_rate)
            lower_bound = self.baseline_value - max_deviation
            upper_bound = self.baseline_value + max_deviation

            # Combine with hard constraints
            lower_bound = max(lower_bound, self.min_value)
            upper_bound = min(upper_bound, self.max_value)

            projected = max(lower_bound, min(upper_bound, projected))

        return projected

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'baseline_value': self.baseline_value,
            'max_change_rate': self.max_change_rate,
            'is_fixed': self.is_fixed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'InputConstraint':
        """Create from dictionary"""
        return cls(**data)


class ConstraintManager:
    """
    Manager for multiple input constraints

    This class handles constraint projection for multiple input variables,
    supporting both hard constraints (min/max) and soft constraints (change rates).
    """

    def __init__(self, constraints: List[InputConstraint]):
        """
        Initialize constraint manager

        Args:
            constraints: List of InputConstraint objects
        """
        self.constraints = constraints
        self.n_inputs = len(constraints)

        # Build constraint lookup
        self.constraint_dict = {c.name: c for c in constraints}

        # Identify optimizable vs fixed inputs
        self.optimizable_indices = [
            i for i, c in enumerate(constraints) if not c.is_fixed
        ]
        self.fixed_indices = [i for i, c in enumerate(constraints) if c.is_fixed]

        self.n_optimizable = len(self.optimizable_indices)
        self.n_fixed = len(self.fixed_indices)

    def project(self, values: torch.Tensor) -> torch.Tensor:
        """
        Project values to satisfy all constraints

        Args:
            values: Tensor of input values to project (shape: [n_inputs] or [batch, n_inputs])

        Returns:
            Projected values (same shape as input)
        """
        # Handle both single sample and batch
        original_shape = values.shape
        is_batched = len(original_shape) > 1

        if is_batched:
            batch_size = original_shape[0]
            values = values.view(-1, self.n_inputs)
        else:
            values = values.view(1, -1)

        # Project each input
        projected = values.clone()

        for i, constraint in enumerate(self.constraints):
            if constraint.is_fixed:
                # Keep fixed values unchanged
                continue

            # Extract values for this input
            input_values = projected[:, i]

            # Hard constraints
            input_values = torch.clamp(
                input_values, min=constraint.min_value, max=constraint.max_value
            )

            # Change rate constraints
            if (
                constraint.max_change_rate is not None
                and constraint.baseline_value is not None
            ):
                max_deviation = abs(constraint.baseline_value * constraint.max_change_rate)
                lower_bound = constraint.baseline_value - max_deviation
                upper_bound = constraint.baseline_value + max_deviation

                # Combine with hard constraints
                lower_bound = max(lower_bound, constraint.min_value)
                upper_bound = min(upper_bound, constraint.max_value)

                input_values = torch.clamp(
                    input_values, min=lower_bound, max=upper_bound
                )

            # Update projected values
            projected[:, i] = input_values

        # Restore original shape
        if not is_batched:
            projected = projected.squeeze(0)

        return projected

    def get_initial_values(self) -> np.ndarray:
        """
        Get initial values (baseline or midpoint) for optimization

        Returns:
            Array of initial values
        """
        initial = np.zeros(self.n_inputs)

        for i, constraint in enumerate(self.constraints):
            if constraint.baseline_value is not None:
                initial[i] = constraint.baseline_value
            else:
                # Use midpoint if no baseline
                initial[i] = (constraint.min_value + constraint.max_value) / 2

        return initial

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get constraint bounds for all inputs

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array([c.min_value for c in self.constraints])
        upper = np.array([c.max_value for c in self.constraints])
        return lower, upper

    def validate_values(self, values: np.ndarray) -> bool:
        """
        Check if values satisfy all constraints

        Args:
            values: Input values to validate

        Returns:
            True if all constraints are satisfied
        """
        if len(values) != self.n_inputs:
            return False

        for i, (constraint, value) in enumerate(zip(self.constraints, values)):
            # Check hard constraints
            if not (constraint.min_value <= value <= constraint.max_value):
                return False

            # Check change rate constraints
            if (
                constraint.max_change_rate is not None
                and constraint.baseline_value is not None
            ):
                max_deviation = abs(constraint.baseline_value * constraint.max_change_rate)
                lower_bound = constraint.baseline_value - max_deviation
                upper_bound = constraint.baseline_value + max_deviation

                lower_bound = max(lower_bound, constraint.min_value)
                upper_bound = min(upper_bound, constraint.max_value)

                if not (lower_bound <= value <= upper_bound):
                    return False

        return True

    def get_constraint_summary(self) -> str:
        """
        Get human-readable constraint summary

        Returns:
            Formatted string describing all constraints
        """
        lines = ["Constraint Summary:", "=" * 80]

        for constraint in self.constraints:
            status = "FIXED" if constraint.is_fixed else "OPTIMIZABLE"
            line = f"{constraint.name:20s} [{status:12s}] "
            line += f"Range: [{constraint.min_value:8.2f}, {constraint.max_value:8.2f}]"

            if constraint.baseline_value is not None:
                line += f" | Baseline: {constraint.baseline_value:8.2f}"

            if constraint.max_change_rate is not None:
                line += f" | Max Δ: ±{constraint.max_change_rate*100:.1f}%"

            lines.append(line)

        lines.append("=" * 80)
        lines.append(f"Total inputs: {self.n_inputs}")
        lines.append(f"Optimizable: {self.n_optimizable}")
        lines.append(f"Fixed: {self.n_fixed}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, constraints_dict: Dict[str, Dict]) -> 'ConstraintManager':
        """
        Create ConstraintManager from dictionary

        Args:
            constraints_dict: Dict mapping input names to constraint specs

        Returns:
            ConstraintManager instance
        """
        constraints = []
        for name, spec in constraints_dict.items():
            spec['name'] = name
            constraints.append(InputConstraint.from_dict(spec))

        return cls(constraints)

    def to_dict(self) -> Dict[str, Dict]:
        """
        Convert to dictionary

        Returns:
            Dict mapping input names to constraint specs
        """
        return {c.name: c.to_dict() for c in self.constraints}
