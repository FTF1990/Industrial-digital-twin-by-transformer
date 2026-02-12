"""
Signal Mapping Utilities

Generates mask matrices from signal mapping configuration.
Used by training scripts to create the appropriate mask for MaskedSST.

Config format:
{
    "boundary": ["SP", "AP", "T1", "P2_P1"],
    "target": ["VP", "PV", "MV", "DP", "AP"],
    "signal_mapping": {
        "enabled": true,
        "auto_exclude_self": true,
        "forced_exclusions": [
            {"input": "SP", "output": "MV"}
        ],
        "forced_inclusions": []
    }
}
"""

import json
import torch


def load_signal_config(config_path):
    """Load signal configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def is_mapping_enabled(config):
    """Check if signal mapping is enabled in the config."""
    mapping = config.get('signal_mapping', {})
    return mapping.get('enabled', False)


def generate_mask_matrix(input_signals, output_signals, mapping_config):
    """
    Generate a binary mask matrix from signal mapping configuration.

    Args:
        input_signals (list[str]): List of input (boundary) signal names
        output_signals (list[str]): List of output (target) signal names
        mapping_config (dict): Signal mapping configuration with keys:
            - auto_exclude_self (bool): Block self-prediction for overlapping signals
            - forced_exclusions (list): List of {"input": ..., "output": ...} to block
            - forced_inclusions (list): List of {"input": ..., "output": ...} to allow

    Returns:
        torch.Tensor: Binary mask of shape (num_output, num_input)
            1 = allow attention, 0 = block attention
    """
    num_in = len(input_signals)
    num_out = len(output_signals)

    # Start with all connections allowed
    mask = torch.ones(num_out, num_in)

    # Auto-exclude self-prediction: block signal from predicting itself
    if mapping_config.get('auto_exclude_self', True):
        for i, out_sig in enumerate(output_signals):
            for j, in_sig in enumerate(input_signals):
                if out_sig == in_sig:
                    mask[i, j] = 0

    # Apply forced exclusions
    for excl in mapping_config.get('forced_exclusions', []):
        in_sig = excl.get('input', '')
        out_sig = excl.get('output', '')
        if in_sig in input_signals and out_sig in output_signals:
            j = input_signals.index(in_sig)
            i = output_signals.index(out_sig)
            mask[i, j] = 0

    # Apply forced inclusions (can override auto-exclude or forced exclusions)
    for incl in mapping_config.get('forced_inclusions', []):
        in_sig = incl.get('input', '')
        out_sig = incl.get('output', '')
        if in_sig in input_signals and out_sig in output_signals:
            j = input_signals.index(in_sig)
            i = output_signals.index(out_sig)
            mask[i, j] = 1

    return mask


def generate_full_mask(input_signals, output_signals):
    """Generate an all-ones mask (no blocking). Used for stage2_mask_mode='none'."""
    return torch.ones(len(output_signals), len(input_signals))


def describe_mask(mask, input_signals, output_signals):
    """
    Generate a human-readable description of the mask.

    Returns:
        dict with keys: total_connections, blocked_connections, blocked_pairs
    """
    total = mask.numel()
    blocked = int((mask == 0).sum().item())
    blocked_pairs = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                blocked_pairs.append({
                    'input': input_signals[j],
                    'output': output_signals[i]
                })

    return {
        'total_connections': total,
        'blocked_connections': blocked,
        'allowed_connections': total - blocked,
        'blocked_pairs': blocked_pairs,
    }
