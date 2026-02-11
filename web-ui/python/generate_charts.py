#!/usr/bin/env python3
"""
Chart Generation Script

Generates per-signal comparison charts (Actual vs Stage1 vs Ensemble).

Usage:
    python generate_charts.py --predictions ./results/predictions.csv \
        --results ./results/inference_results.json --output ./results/charts/
"""

import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def emit(msg_type, **kwargs):
    msg = {"type": msg_type, **kwargs}
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Generate visualization charts')
    parser.add_argument('--predictions', required=True, help='Predictions CSV')
    parser.add_argument('--results', required=True, help='Inference results JSON')
    parser.add_argument('--output', default='./charts', help='Output directory for PNG files')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max samples to plot (for readability)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pred_df = pd.read_csv(args.predictions)
    with open(args.results, 'r') as f:
        results = json.load(f)

    target_signals = results.get('target_signals', [])
    has_actual = any(f"{sig}_actual" in pred_df.columns for sig in target_signals)

    n_samples = len(pred_df)
    if n_samples > args.max_samples:
        step = n_samples // args.max_samples
        pred_df = pred_df.iloc[::step].reset_index(drop=True)
    x_axis = np.arange(len(pred_df))

    chart_files = []

    # ── Per-signal comparison charts ──
    for i, sig in enumerate(target_signals):
        emit("progress", signal=sig, index=i + 1, total=len(target_signals))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Top: Predictions comparison
        ax = axes[0]
        stage1_col = f"{sig}_stage1"
        ensemble_col = f"{sig}_ensemble"
        actual_col = f"{sig}_actual"

        if has_actual and actual_col in pred_df.columns:
            ax.plot(x_axis, pred_df[actual_col], label='Actual', color='#2196F3',
                    linewidth=1, alpha=0.8)

        ax.plot(x_axis, pred_df[stage1_col], label='Stage1', color='#FF9800',
                linewidth=0.8, alpha=0.7, linestyle='--')
        ax.plot(x_axis, pred_df[ensemble_col], label='Ensemble', color='#4CAF50',
                linewidth=0.8, alpha=0.7)

        # Add metrics text
        s1_info = results.get('stage1_metrics', {}).get(sig, {})
        en_info = results.get('ensemble_metrics', {}).get(sig, {})
        boost_info = results.get('boosting_info', {}).get(sig, {})

        title = f'{sig}'
        if s1_info:
            title += f'  |  Stage1 R²={s1_info.get("r2", "N/A")}'
        if en_info:
            title += f'  |  Ensemble R²={en_info.get("r2", "N/A")}'
            delta = en_info.get("delta_r2", 0)
            title += f'  |  ΔR²={delta:+.4f}'
        if boost_info.get('boosted'):
            title += '  [BOOSTED]'

        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Bottom: Residual plot
        ax2 = axes[1]
        if has_actual and actual_col in pred_df.columns:
            residual_s1 = pred_df[actual_col] - pred_df[stage1_col]
            residual_en = pred_df[actual_col] - pred_df[ensemble_col]
            ax2.fill_between(x_axis, residual_s1, alpha=0.3, color='#FF9800', label='Stage1 Error')
            ax2.fill_between(x_axis, residual_en, alpha=0.3, color='#4CAF50', label='Ensemble Error')
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.legend(loc='upper right', fontsize=8)
        else:
            diff = pred_df[ensemble_col] - pred_df[stage1_col]
            ax2.fill_between(x_axis, diff, alpha=0.4, color='#9C27B0', label='Ensemble - Stage1')
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.legend(loc='upper right', fontsize=8)

        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(args.output, f"{sig}.png")
        fig.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        chart_files.append(chart_path)

    # ── Overview chart: R² comparison bar chart ──
    if results.get('stage1_metrics') and results.get('ensemble_metrics'):
        fig, ax = plt.subplots(figsize=(max(10, len(target_signals) * 0.8), 6))

        x = np.arange(len(target_signals))
        width = 0.35

        s1_r2 = [results['stage1_metrics'].get(s, {}).get('r2', 0) for s in target_signals]
        en_r2 = [results['ensemble_metrics'].get(s, {}).get('r2', 0) for s in target_signals]

        bars1 = ax.bar(x - width/2, s1_r2, width, label='Stage1', color='#FF9800', alpha=0.8)
        bars2 = ax.bar(x + width/2, en_r2, width, label='Ensemble', color='#4CAF50', alpha=0.8)

        # Mark boosted signals
        for j, sig in enumerate(target_signals):
            if results.get('boosting_info', {}).get(sig, {}).get('boosted'):
                ax.annotate('*', (x[j], max(s1_r2[j], en_r2[j]) + 0.02),
                           ha='center', fontsize=14, color='red')

        ax.axhline(y=results.get('overall_stage1_r2', 0), color='#FF9800',
                   linestyle='--', alpha=0.5, label=f'Stage1 Avg={results.get("overall_stage1_r2", 0)}')
        ax.axhline(y=results.get('overall_ensemble_r2', 0), color='#4CAF50',
                   linestyle='--', alpha=0.5, label=f'Ensemble Avg={results.get("overall_ensemble_r2", 0)}')

        ax.set_ylabel('R² Score')
        ax.set_title('Per-Signal R² Comparison: Stage1 vs Ensemble (* = boosted)')
        ax.set_xticks(x)
        short_labels = [s[:20] + '..' if len(s) > 22 else s for s in target_signals]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=min(0, min(s1_r2 + en_r2) - 0.05))

        plt.tight_layout()
        overview_path = os.path.join(args.output, "_overview_r2_comparison.png")
        fig.savefig(overview_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        chart_files.append(overview_path)

    # ── Training loss chart (if history available in results) ──

    emit("complete", charts=chart_files, total=len(chart_files))


if __name__ == '__main__':
    main()
