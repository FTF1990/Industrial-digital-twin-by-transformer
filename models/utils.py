"""
Utility functions for data processing and model evaluation
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d


def create_temporal_context_data(X, y, context_window=5):
    """
    Create temporal context data

    Creates temporal context windows for time-series sensor data. Each sample
    is expanded to include surrounding timesteps for temporal analysis.

    Args:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples, n_targets)
        context_window (int): Number of timesteps to include before and after each sample

    Returns:
        tuple: (X_context, y_context, valid_indices)
            - X_context: Array of shape (valid_samples, context_size, n_features)
            - y_context: Array of shape (valid_samples, n_targets)
            - valid_indices: List of original indices that have valid context windows
    """
    n_samples, n_features = X.shape
    context_size = 2 * context_window + 1

    valid_start = context_window
    valid_end = n_samples - context_window
    valid_samples = valid_end - valid_start

    if valid_samples <= 0:
        raise ValueError(f"Insufficient data, need at least {2*context_window+1} samples")

    X_context = np.zeros((valid_samples, context_size, n_features))
    y_context = np.zeros((valid_samples, y.shape[1]))
    valid_indices = []

    for i in range(valid_samples):
        original_idx = valid_start + i
        start_idx = original_idx - context_window
        end_idx = original_idx + context_window + 1

        X_context[i] = X[start_idx:end_idx]
        y_context[i] = y[original_idx]
        valid_indices.append(original_idx)

    return X_context, y_context, valid_indices


def apply_ifd_smoothing(y_data, target_sensors, ifd_sensor_names,
                        window_length=15, polyorder=3):
    """
    Apply smoothing filter to specified IFD sensors

    Applies Savitzky-Golay smoothing filter to specified sensors to reduce noise
    while preserving peak features. Particularly useful for IFD (Industrial Fault Detection)
    sensors with noisy signals.

    Args:
        y_data (np.ndarray): Target sensor data of shape (n_samples, n_sensors)
        target_sensors (list): List of all target sensor names
        ifd_sensor_names (list): List of sensor names to apply smoothing to
        window_length (int): Length of the filter window. Default: 15
        polyorder (int): Order of the polynomial used for filtering. Default: 3

    Returns:
        np.ndarray: Smoothed sensor data with same shape as input
    """
    y_smoothed = y_data.copy()

    for sensor in ifd_sensor_names:
        if sensor in target_sensors:
            idx = target_sensors.index(sensor)
            original_signal = y_data[:, idx]

            # Adjust window length for short signals
            window_len = min(window_length, len(original_signal) // 4)
            if window_len % 2 == 0:
                window_len += 1

            if window_len >= 3:
                # Apply Savitzky-Golay filter
                smoothed_signal = savgol_filter(original_signal, window_len, polyorder)

                # Peak enhancement to preserve important features
                peaks = maximum_filter1d(original_signal, size=window_len//3)
                is_peak = (original_signal == peaks) & (original_signal > np.percentile(original_signal, 75))

                enhanced_signal = smoothed_signal.copy()
                enhanced_signal[is_peak] = smoothed_signal[is_peak] * 0.8 + original_signal[is_peak] * 1.2

                y_smoothed[:, idx] = enhanced_signal

    return y_smoothed


def handle_duplicate_columns(df):
    """
    Handle duplicate column names in DataFrame by adding numbered suffixes

    Handles duplicate column names in a DataFrame by appending numeric suffixes
    to duplicated columns while preserving the original column order.

    Args:
        df (pd.DataFrame): Input DataFrame that may contain duplicate column names

    Returns:
        tuple: (df, duplicates)
            - df: DataFrame with unique column names
            - duplicates: Dictionary mapping original column names to duplicate counts
    """
    cols = df.columns.tolist()
    new_cols = []
    col_counts = {}

    for col in cols:
        if col not in col_counts:
            col_counts[col] = 0
            new_cols.append(col)
        else:
            col_counts[col] += 1
            new_cols.append(f"{col}_#{col_counts[col] + 1}")

    df.columns = new_cols

    # Return statistics of duplicates
    duplicates = {k: v for k, v in col_counts.items() if v > 0}
    return df, duplicates


def get_available_signals(df):
    """
    Get all available signals

    Extracts available sensor signal names from a DataFrame, excluding timestamp columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing sensor data

    Returns:
        list: List of available signal names
    """
    if df is None:
        return []

    cols = df.columns.tolist()

    # Remove timestamp columns (assuming first column might be timestamp)
    if cols and (cols[0].startswith('2025') or
                 'timestamp' in cols[0].lower() or
                 'time' in cols[0].lower()):
        cols = cols[1:]

    return cols


def validate_signal_exclusivity_v1(boundary_signals, target_signals):
    """
    Validate signal exclusivity for V1 model

    Validates that boundary and target signals don't overlap for V1 model.

    Args:
        boundary_signals (list): List of boundary condition signal names
        target_signals (list): List of target signal names

    Returns:
        tuple: (is_valid, error_msg)
            - is_valid: Boolean indicating if validation passed
            - error_msg: Error message if validation failed, empty string otherwise
    """
    if not boundary_signals or not target_signals:
        return True, ""

    boundary_set = set(boundary_signals)
    target_set = set(target_signals)
    overlap = boundary_set & target_set

    if overlap:
        overlap_list = list(overlap)
        error_msg = f"❌ Signal exclusivity error!\n\nThe following signals appear in both boundary and target:\n"

        for i, sig in enumerate(overlap_list[:10], 1):
            error_msg += f"  {i}. {sig}\n"

        if len(overlap_list) > 10:
            error_msg += f"  ... and {len(overlap_list)-10} more duplicate signals\n"

        error_msg += f"\nPlease remove these signals from one of the positions!"
        return False, error_msg

    return True, ""


def validate_signal_exclusivity_v4(boundary_signals, target_signals, temporal_signals):
    """
    Validate signal exclusivity for V4 model

    Validates signal selections for V4 model:
    1. Boundary and target signals must not overlap
    2. Temporal signals must be a subset of target signals

    Args:
        boundary_signals (list): List of boundary condition signal names
        target_signals (list): List of target signal names
        temporal_signals (list): List of temporal signal names

    Returns:
        tuple: (is_valid, error_msg)
            - is_valid: Boolean indicating if validation passed
            - error_msg: Error message if validation failed, empty string otherwise
    """
    if not boundary_signals or not target_signals:
        return True, ""

    errors = []

    # Check boundary-target overlap
    boundary_set = set(boundary_signals)
    target_set = set(target_signals)
    overlap_bt = boundary_set & target_set

    if overlap_bt:
        overlap_list = list(overlap_bt)
        error_msg = f"Boundary and target signals overlap ({len(overlap_list)} signals):\n"
        for i, sig in enumerate(overlap_list[:5], 1):
            error_msg += f"  {i}. {sig}\n"
        if len(overlap_list) > 5:
            error_msg += f"  ... and {len(overlap_list)-5} more\n"
        errors.append(error_msg)

    # Check temporal signals are subset of target signals
    if temporal_signals:
        temporal_set = set(temporal_signals)
        invalid_temporal = temporal_set - target_set

        if invalid_temporal:
            invalid_list = list(invalid_temporal)
            error_msg = f"Temporal signals must be in target signals ({len(invalid_list)} invalid):\n"
            for i, sig in enumerate(invalid_list[:5], 1):
                error_msg += f"  {i}. {sig}\n"
            if len(invalid_list) > 5:
                error_msg += f"  ... and {len(invalid_list)-5} more\n"
            errors.append(error_msg)

    if errors:
        full_error = "❌ Signal selection error!\n\n" + "\n".join(errors) + "\n Please fix before training!"
        return False, full_error

    return True, ""
