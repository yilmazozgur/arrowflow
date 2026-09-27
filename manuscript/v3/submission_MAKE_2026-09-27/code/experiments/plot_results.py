"""
Phase 4: Publication Figures
=============================
Generate publication-quality plots from experiment CSV logs.

Run:
    python -m test.experiments.plot_results --results-dir test/experiments/results/

Generates:
    - Figure 1: Learning curves (accuracy vs training size) per dataset
    - Figure 2: MoE disentanglement (ArrowFlow MoE vs MLP MoE)
    - Table 1: Final accuracy comparison across all datasets
    - Table 2: Ablation results
"""

import os
import sys
import glob
import csv
import argparse
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Will output tables only.")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir=None):
    """Load all CSV results from the results directory."""
    if results_dir is None:
        results_dir = RESULTS_DIR

    all_rows = []
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    for filepath in csv_files:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in ['test_error_mean', 'test_error_std', 'test_error_min',
                            'test_error_max', 'test_error_stderr', 'val_error_mean',
                            'val_error_std', 'time_mean', 'n_train', 'n_test',
                            'n_classes', 'n_sims']:
                    if key in row and row[key]:
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            pass
                row['_source'] = os.path.basename(filepath)
                all_rows.append(row)

    print(f"Loaded {len(all_rows)} result rows from {len(csv_files)} CSV files")
    return all_rows


def filter_results(rows, **kwargs):
    """Filter rows by field values."""
    filtered = rows
    for key, value in kwargs.items():
        if isinstance(value, list):
            filtered = [r for r in filtered if r.get(key) in value]
        else:
            filtered = [r for r in filtered if r.get(key) == value]
    return filtered


# ---------------------------------------------------------------------------
# Figure 1: Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(rows, datasets=None, output_prefix='fig1_learning_curve'):
    """
    Plot test accuracy vs training set size for ArrowFlow + baselines.
    One subplot per dataset.
    """
    if not HAS_MPL:
        print("Skipping plot (matplotlib not available)")
        return

    if datasets is None:
        datasets = list(set(r.get('dataset', '') for r in rows if r.get('dataset')))
        datasets = [d for d in datasets if d]

    n_datasets = len(datasets)
    if n_datasets == 0:
        print("No data to plot")
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)

    colors = {
        'ArrowFlow': '#e74c3c',
        'Random Forest': '#2ecc71',
        'SVM-RBF': '#3498db',
        'MLP': '#9b59b6',
        'MLP [64]': '#9b59b6',
        'MLP [128]': '#8e44ad',
        'XGBoost': '#f39c12',
        'KNN': '#1abc9c',
        'KNN-5': '#1abc9c',
        'Logistic Regression': '#95a5a6',
    }
    markers = {
        'ArrowFlow': 'o',
        'Random Forest': 's',
        'SVM-RBF': '^',
        'MLP': 'D',
        'MLP [64]': 'D',
        'MLP [128]': 'D',
        'XGBoost': 'v',
        'KNN': 'p',
        'KNN-5': 'p',
        'Logistic Regression': 'x',
    }

    for idx, dataset in enumerate(datasets):
        ax = axes[0][idx]
        ds_rows = filter_results(rows, dataset=dataset)

        # Group by method
        method_data = defaultdict(lambda: {'n_train': [], 'error': [], 'stderr': []})
        for r in ds_rows:
            method = r.get('method', 'Unknown')
            n_train = r.get('n_train', 0)
            error = r.get('test_error_mean', 0)
            stderr = r.get('test_error_stderr', 0)
            if isinstance(n_train, (int, float)) and isinstance(error, (int, float)):
                method_data[method]['n_train'].append(float(n_train))
                method_data[method]['error'].append(100 * float(error))
                method_data[method]['stderr'].append(100 * float(stderr) if stderr else 0)

        for method, data in sorted(method_data.items()):
            # Sort by n_train
            sorted_idx = np.argsort(data['n_train'])
            x = np.array(data['n_train'])[sorted_idx]
            y = np.array(data['error'])[sorted_idx]
            yerr = np.array(data['stderr'])[sorted_idx]

            color = colors.get(method, '#7f8c8d')
            marker = markers.get(method, 'o')

            ax.errorbar(x, y, yerr=yerr, label=method, color=color, marker=marker,
                        linewidth=2, markersize=6, capsize=3)

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Test Error (%)')
        ax.set_title(f'{dataset}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, f'{output_prefix}.pdf')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: MoE Disentanglement
# ---------------------------------------------------------------------------

def plot_moe_disentanglement(rows, datasets=None, output_prefix='fig2_moe_control'):
    """
    Bar chart comparing: ArrowFlow MoE=7, MLP Ensemble=7, ArrowFlow single, MLP single.
    """
    if not HAS_MPL:
        print("Skipping plot (matplotlib not available)")
        return

    # Filter for ablation results with MoE control labels
    moe_rows = [r for r in rows if r.get('ablation') in
                ['arrowflow_moe_7', 'mlp_ensemble_7', 'arrowflow_single', 'mlp_single',
                 'MLP Ensemble (n=7)', 'MLP']]

    if not moe_rows:
        print("No MoE control data found")
        return

    if datasets is None:
        datasets = list(set(r.get('dataset', '') for r in moe_rows if r.get('dataset')))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False)

    bar_labels = ['ArrowFlow\nMoE=7', 'MLP\nEnsemble=7', 'ArrowFlow\nSingle', 'MLP\nSingle']
    bar_keys = ['arrowflow_moe_7', 'mlp_ensemble_7', 'arrowflow_single', 'mlp_single']
    bar_colors = ['#e74c3c', '#9b59b6', '#e74c3c', '#9b59b6']
    bar_alpha = [1.0, 1.0, 0.5, 0.5]

    for idx, dataset in enumerate(datasets):
        ax = axes[0][idx]
        ds_rows = filter_results(moe_rows, dataset=dataset)

        values = []
        errors = []
        for key in bar_keys:
            matching = [r for r in ds_rows if r.get('ablation') == key]
            if matching:
                values.append(100 * float(matching[0].get('test_error_mean', 0)))
                errors.append(100 * float(matching[0].get('test_error_stderr', 0)))
            else:
                values.append(0)
                errors.append(0)

        x = np.arange(len(bar_labels))
        bars = ax.bar(x, values, yerr=errors, color=bar_colors, alpha=bar_alpha,
                      capsize=5, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel('Test Error (%)')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, f'{output_prefix}.pdf')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


# ---------------------------------------------------------------------------
# Table 1: Final accuracy comparison
# ---------------------------------------------------------------------------

def print_accuracy_table(rows):
    """Print a markdown table of final accuracies across all datasets."""
    # Group by dataset and method, take the result with largest n_train
    table = defaultdict(dict)
    for r in rows:
        dataset = r.get('dataset', '')
        method = r.get('method', '')
        error = r.get('test_error_mean', '')
        if dataset and method and error:
            n_train = float(r.get('n_train', 0))
            existing = table[dataset].get(method)
            if existing is None or n_train > existing['n_train']:
                table[dataset][method] = {
                    'error': float(error),
                    'stderr': float(r.get('test_error_stderr', 0)),
                    'n_train': n_train,
                }

    if not table:
        print("No data for accuracy table")
        return

    # Collect all methods
    all_methods = set()
    for methods in table.values():
        all_methods.update(methods.keys())
    all_methods = sorted(all_methods)

    # Print table
    print("\n## Table 1: Final Test Error (%) at Full Training Size\n")
    header = "| Dataset | " + " | ".join(all_methods) + " |"
    separator = "|" + "|".join(["---"] * (len(all_methods) + 1)) + "|"
    print(header)
    print(separator)

    for dataset in sorted(table.keys()):
        row_parts = [dataset]
        best_error = min(
            (table[dataset][m]['error'] for m in table[dataset]),
            default=1.0
        )
        for method in all_methods:
            if method in table[dataset]:
                e = table[dataset][method]['error']
                se = table[dataset][method]['stderr']
                cell = f"{100*e:.1f}"
                if se > 0:
                    cell += f" +/- {100*se:.1f}"
                if e == best_error:
                    cell = f"**{cell}**"
                row_parts.append(cell)
            else:
                row_parts.append("--")
        print("| " + " | ".join(row_parts) + " |")

    print()


# ---------------------------------------------------------------------------
# Table 2: Ablation results
# ---------------------------------------------------------------------------

def print_ablation_table(rows):
    """Print ablation results as a markdown table."""
    abl_rows = [r for r in rows if r.get('ablation')]
    if not abl_rows:
        print("No ablation data found")
        return

    # Group by dataset
    datasets = sorted(set(r.get('dataset', '') for r in abl_rows if r.get('dataset')))

    print("\n## Table 2: Ablation Study Results\n")

    for dataset in datasets:
        print(f"\n### {dataset}\n")
        print("| Ablation | Test Error (%) | Std Err | Time (s) |")
        print("|----------|---------------|---------|----------|")

        ds_rows = sorted(
            filter_results(abl_rows, dataset=dataset),
            key=lambda r: float(r.get('test_error_mean', 1.0))
        )
        for r in ds_rows:
            name = r.get('ablation', '')
            error = 100 * float(r.get('test_error_mean', 0))
            stderr = 100 * float(r.get('test_error_stderr', 0))
            time_s = float(r.get('time_mean', 0))
            print(f"| {name} | {error:.1f} | {stderr:.1f} | {time_s:.1f} |")

    print()


# ---------------------------------------------------------------------------
# IMDB crossover analysis
# ---------------------------------------------------------------------------

def print_imdb_crossover(rows):
    """Analyze and print the IMDB crossover point."""
    imdb_rows = filter_results(rows, dataset='imdb')
    if not imdb_rows:
        print("No IMDB data found")
        return

    print("\n## IMDB Scaling Analysis\n")
    print("| N_train | ArrowFlow Error | MLP Error | Winner |")
    print("|---------|----------------|-----------|--------|")

    # Group by n_train
    by_ntrain = defaultdict(dict)
    for r in imdb_rows:
        method = r.get('method', '')
        n_train = r.get('n_train', 0)
        error = r.get('test_error_mean', 0)
        if isinstance(n_train, (int, float)) and isinstance(error, (int, float)):
            by_ntrain[int(n_train)][method] = float(error)

    for n_train in sorted(by_ntrain.keys()):
        methods = by_ntrain[n_train]
        af_err = methods.get('ArrowFlow', None)
        mlp_err = methods.get('MLP [64]', methods.get('MLP', None))

        af_str = f"{100*af_err:.1f}%" if af_err is not None else "--"
        mlp_str = f"{100*mlp_err:.1f}%" if mlp_err is not None else "--"

        if af_err is not None and mlp_err is not None:
            winner = "ArrowFlow" if af_err < mlp_err else "MLP"
        else:
            winner = "--"

        print(f"| {n_train:,} | {af_str} | {mlp_str} | {winner} |")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot ArrowFlow experiment results')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory containing result CSV files')
    args = parser.parse_args()

    rows = load_all_results(args.results_dir)
    if not rows:
        print("No results found. Run experiments first.")
        return

    # Generate all outputs
    print_accuracy_table(rows)
    print_ablation_table(rows)
    print_imdb_crossover(rows)

    if HAS_MPL:
        plot_learning_curves(rows)
        plot_moe_disentanglement(rows)
    else:
        print("\nInstall matplotlib for figures: pip install matplotlib")

    print("\nDone. Figures saved to:", FIGURES_DIR)


if __name__ == '__main__':
    main()
