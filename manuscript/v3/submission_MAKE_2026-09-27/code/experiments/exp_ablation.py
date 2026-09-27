"""
Phase 3: Ablation Study
========================
Systematic ablation to understand what each ArrowFlow component contributes.
Run on 2-3 datasets (best and worst from Phase 2).

Run:
    python -m test.experiments.exp_ablation [--quick] [--dataset iris]

Results are logged to test/experiments/results/ablation_*.csv
"""

import os
import sys
import copy
import csv
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from arrowflow.benchmark import (
    ArrowFlowConfig, run_arrowflow_experiment, run_baseline,
    load_dataset, log_results
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Base configuration (the "default" from which ablations deviate)
# ---------------------------------------------------------------------------

BASE_CONFIG = ArrowFlowConfig(
    no_of_filters=[128],
    layer_types=['sort', 'sort'],
    no_of_iters=200,
    moe_no_of_networks=5,
    batch_size=32,
    learning_rate=0.1,
    distance_computation_metric='l1',
    pol_deg=3,
    no_of_embedding_dim=16,
    last_layer_update=True,
    poly_expansion=False,
    verbose=0,
)

# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

def make_ablations():
    """Generate all ablation configs. Each is (name, config)."""
    ablations = []

    # 0. Baseline (default config)
    ablations.append(('base', copy.deepcopy(BASE_CONFIG)))

    # --- Architecture ablations ---

    # 1. Sort-only vs Hybrid (tensor-sort)
    c = copy.deepcopy(BASE_CONFIG)
    c.layer_types = ['tensor', 'sort']
    c.pol_deg = 0  # tensor first layer doesn't need polynomial expansion
    ablations.append(('hybrid_tensor_sort', c))

    # 2. MoE sweep
    for moe_n in [1, 3, 5, 7, 11]:
        c = copy.deepcopy(BASE_CONFIG)
        c.moe_no_of_networks = moe_n
        ablations.append((f'moe_{moe_n}', c))

    # 3. Filter count sweep
    for n_filters in [32, 64, 128, 256, 512]:
        c = copy.deepcopy(BASE_CONFIG)
        c.no_of_filters = [n_filters]
        c.layer_types = ['sort', 'sort']
        ablations.append((f'filters_{n_filters}', c))

    # 4. Depth sweep
    c = copy.deepcopy(BASE_CONFIG)
    c.no_of_filters = [128, 64]
    c.layer_types = ['sort', 'sort', 'sort']
    c.filter_rfs = [None, None, None]
    c.metric_flag = [None, None, None]
    ablations.append(('depth_2_layers', c))

    c = copy.deepcopy(BASE_CONFIG)
    c.no_of_filters = [128, 64, 128]
    c.layer_types = ['sort', 'sort', 'sort', 'sort']
    c.filter_rfs = [None, None, None, None]
    c.metric_flag = [None, None, None, None]
    ablations.append(('depth_3_layers', c))

    # --- Distance metric ablations ---

    # 5. Distance metrics
    for metric in ['l1', 'l2', 'l0']:
        c = copy.deepcopy(BASE_CONFIG)
        c.distance_computation_metric = metric
        ablations.append((f'dist_{metric}', c))

    # --- Training ablations ---

    # 6. Learning rate sweep
    for lr in [0.01, 0.05, 0.1, 0.5]:
        c = copy.deepcopy(BASE_CONFIG)
        c.learning_rate = lr
        ablations.append((f'lr_{lr}', c))

    # 7. Last layer update on/off
    c = copy.deepcopy(BASE_CONFIG)
    c.last_layer_update = False
    ablations.append(('no_last_layer_update', c))

    c = copy.deepcopy(BASE_CONFIG)
    c.last_layer_update = True
    ablations.append(('with_last_layer_update', c))

    # 8. Motion averaging method
    for method in ['mean', 'median']:
        c = copy.deepcopy(BASE_CONFIG)
        c.average_method_motion = method
        ablations.append((f'motion_avg_{method}', c))

    # --- Embedding ablations ---

    # 9. Polynomial expansion on/off
    c = copy.deepcopy(BASE_CONFIG)
    c.poly_expansion = True
    c.pol_deg = 3
    ablations.append(('poly_expansion_on', c))

    c = copy.deepcopy(BASE_CONFIG)
    c.poly_expansion = False
    c.pol_deg = 3
    ablations.append(('poly_expansion_off', c))

    # 10. Embedding dimension
    for dim in [8, 16, 32, 64]:
        c = copy.deepcopy(BASE_CONFIG)
        c.no_of_embedding_dim = dim
        ablations.append((f'embed_dim_{dim}', c))

    # 11. Polynomial degree
    for deg in [0, 2, 3, 5]:
        c = copy.deepcopy(BASE_CONFIG)
        c.pol_deg = deg
        ablations.append((f'pol_deg_{deg}', c))

    # --- Initialization ablations ---

    # 12. Init from data vs random
    c = copy.deepcopy(BASE_CONFIG)
    c.initial_filter_with_data = True
    ablations.append(('init_from_data', c))

    c = copy.deepcopy(BASE_CONFIG)
    c.initial_filter_with_data = False
    ablations.append(('init_random', c))

    # --- Backprop ratio ---

    # 13. Ratio of data used in backprop
    for ratio in [0.25, 0.5, 0.75, 1.0]:
        c = copy.deepcopy(BASE_CONFIG)
        c.ratio_data_backprop = ratio
        ablations.append((f'backprop_ratio_{ratio}', c))

    return ablations


# ---------------------------------------------------------------------------
# MoE control experiment: MLP ensemble vs ArrowFlow MoE
# ---------------------------------------------------------------------------

def run_moe_control(dataset_name, n_sims=5, random_state=42):
    """
    Critical control: compare MoE-ArrowFlow vs MoE-MLP
    to disentangle the ensembling effect from the sort-layer effect.
    """
    print(f"\n{'='*60}")
    print(f"MoE Control Experiment: {dataset_name}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test, n_classes = load_dataset(
        dataset_name, random_state=random_state
    )

    results = []

    # MLP ensemble (7 networks)
    print("\n--- MLP Ensemble (7 networks) ---")
    bl_result = run_baseline('mlp_ensemble', X_train, y_train, X_test, y_test,
                              random_state=random_state)
    bl_result['dataset'] = dataset_name
    bl_result['ablation'] = 'mlp_ensemble_7'
    print(f"  Test Error: {100*bl_result['test_error_mean']:.1f}%")
    results.append(bl_result)

    # ArrowFlow MoE=7
    print("\n--- ArrowFlow MoE=7 ---")
    af_config = copy.deepcopy(BASE_CONFIG)
    af_config.moe_no_of_networks = 7
    af_result = run_arrowflow_experiment(dataset_name, af_config, n_sims=n_sims,
                                         random_state=random_state)
    af_result['ablation'] = 'arrowflow_moe_7'
    print(f"  Test Error: {100*af_result['test_error_mean']:.1f}%")
    results.append(af_result)

    # ArrowFlow single
    print("\n--- ArrowFlow Single ---")
    af_config_single = copy.deepcopy(BASE_CONFIG)
    af_config_single.moe_no_of_networks = 1
    af_result_single = run_arrowflow_experiment(dataset_name, af_config_single, n_sims=n_sims,
                                                 random_state=random_state)
    af_result_single['ablation'] = 'arrowflow_single'
    print(f"  Test Error: {100*af_result_single['test_error_mean']:.1f}%")
    results.append(af_result_single)

    # Single MLP
    print("\n--- Single MLP ---")
    bl_single = run_baseline('mlp', X_train, y_train, X_test, y_test,
                              random_state=random_state)
    bl_single['dataset'] = dataset_name
    bl_single['ablation'] = 'mlp_single'
    print(f"  Test Error: {100*bl_single['test_error_mean']:.1f}%")
    results.append(bl_single)

    return results


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation_study(datasets=None, quick=False):
    """Run all ablations on specified datasets."""
    if datasets is None:
        datasets = ['iris', 'digits'] if quick else ['iris', 'wine', 'digits', 'segment']

    n_sims = 3 if quick else 5
    ablations = make_ablations()

    if quick:
        # Only run a subset of ablations
        quick_names = {'base', 'moe_1', 'moe_5', 'moe_7',
                       'filters_64', 'filters_128', 'filters_256',
                       'hybrid_tensor_sort', 'dist_l1', 'dist_l2',
                       'lr_0.05', 'lr_0.1', 'lr_0.5'}
        ablations = [(name, config) for name, config in ablations if name in quick_names]

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# ABLATION: {dataset_name}")
        print(f"{'#'*60}")

        for abl_name, abl_config in ablations:
            print(f"\n--- {abl_name} ---")
            try:
                result = run_arrowflow_experiment(
                    dataset_name, abl_config, n_sims=n_sims
                )
                result['ablation'] = abl_name
                print(f"  Test Error: {100*result['test_error_mean']:.1f}% "
                      f"+/- {100*result['test_error_stderr']:.1f}%")
                all_results.append(result)
            except Exception as e:
                print(f"  FAILED: {e}")

        # MoE control
        try:
            moe_results = run_moe_control(dataset_name, n_sims=n_sims)
            all_results.extend(moe_results)
        except Exception as e:
            print(f"  MoE control FAILED: {e}")

    # Save results
    filename = os.path.join(RESULTS_DIR,
                            f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    fieldnames = [
        'timestamp', 'dataset', 'method', 'ablation', 'n_train', 'n_test', 'n_classes',
        'n_sims', 'test_error_mean', 'test_error_std', 'test_error_min', 'test_error_max',
        'test_error_stderr', 'val_error_mean', 'val_error_std', 'time_mean', 'config'
    ]

    if all_results:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                row = {k: r.get(k, '') for k in fieldnames}
                row['timestamp'] = datetime.now().isoformat()
                row['config'] = str(r.get('config', ''))
                writer.writerow(row)
        print(f"\nResults saved to {filename}")

    return all_results


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    dataset_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset' and i + 1 < len(sys.argv):
            dataset_arg = [sys.argv[i + 1]]

    run_ablation_study(datasets=dataset_arg, quick=quick)
