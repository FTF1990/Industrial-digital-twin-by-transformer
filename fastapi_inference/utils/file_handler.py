"""
File Handling Utilities
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def read_csv_data(file_path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read CSV data with validation

    Args:
        file_path: Path to CSV file
        required_columns: Optional list of required column names

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    print(f" Loaded CSV: {file_path} - Shape: {df.shape}")
    return df


def save_predictions_csv(
    predictions: np.ndarray,
    signal_names: List[str],
    output_path: str,
    metadata: Optional[Dict] = None,
    input_data: Optional[pd.DataFrame] = None
) -> str:
    """
    Save predictions to CSV with metadata

    Args:
        predictions: Prediction array (N_samples, N_signals)
        signal_names: List of target signal names
        output_path: Output file path
        metadata: Optional metadata to include in header
        input_data: Optional input data to include in output

    Returns:
        str: Path to saved file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    pred_df = pd.DataFrame(predictions, columns=signal_names)

    # Add input data if provided
    if input_data is not None:
        # Reset index to align
        input_data_reset = input_data.reset_index(drop=True)
        pred_df = pd.concat([input_data_reset, pred_df], axis=1)

    # Save to CSV
    pred_df.to_csv(output_path, index=False)

    # Save metadata if provided
    if metadata:
        metadata_path = output_path.replace('.csv', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Prediction Metadata\n")
            f.write("=" * 80 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"=Ä Metadata saved: {metadata_path}")

    print(f" Predictions saved: {output_path}")
    return output_path


def validate_data_format(
    df: pd.DataFrame,
    boundary_signals: List[str],
    target_signals: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate data format for inference or evaluation

    Args:
        df: Input DataFrame
        boundary_signals: List of required boundary signal names
        target_signals: Optional list of target signal names (for evaluation)

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check boundary signals
    missing_boundary = set(boundary_signals) - set(df.columns)
    if missing_boundary:
        return False, f"Missing boundary signals: {missing_boundary}"

    # Check target signals if provided
    if target_signals:
        missing_target = set(target_signals) - set(df.columns)
        if missing_target:
            return False, f"Missing target signals: {missing_target}"

    # Check for NaN values
    if df[boundary_signals].isnull().any().any():
        return False, "Boundary signals contain NaN values"

    if target_signals and df[target_signals].isnull().any().any():
        return False, "Target signals contain NaN values"

    return True, "Data format valid"


def generate_output_filename(
    ensemble_name: str,
    output_dir: str,
    prefix: str = "predictions"
) -> str:
    """
    Generate timestamped output filename

    Args:
        ensemble_name: Name of ensemble model
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        str: Full output file path
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{ensemble_name}_{timestamp}.csv"
    return os.path.join(output_dir, filename)
