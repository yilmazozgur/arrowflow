"""Descriptive E05 diagnostics from complete, already validated job results.

The runner validates predictions, model artifacts and diagnostics first. This
module additionally enforces the aggregation schedule. It adds no hypothesis
tests; undefined correlations remain missing with their actual denominators.
"""
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np


PREFIX_METRICS = ('first_view_error', 'mean_single_view_error', 'ensemble_error',
                  'change_from_first', 'change_from_mean_single')
PAIR_METRICS = ('disagreement', 'double_fault', 'error_correlation')
LAYER_METRICS = ('unique_count', 'duplicate_fraction', 'zero_distance_pair_fraction',
                 'normalized_distance_mean', 'normalized_distance_min', 'normalized_distance_max',
                 'query_any_response_tie_fraction', 'query_repeated_response_fraction')


def _indexed(rows, key, expected, description):
    rows = list(rows)
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows) or indexed.keys() != set(expected):
        raise ValueError(f'Incomplete or duplicate {description} schedule')
    return indexed


def _mean(values):
    values = [float(value) for value in values if value is not None]
    if not all(np.isfinite(values)):
        raise ValueError('Nonfinite diagnostic value')
    return float(np.mean(values)) if values else None


def _nested_stats(rows, metric, p):
    folds = [(repeat, fold) for repeat in range(p['outer_repeats']) for fold in range(p['outer_folds'])]
    expected = {(repeat, fold, seed) for repeat, fold in folds for seed in p['fit_seeds']}
    indexed = _indexed(rows, lambda row: (row['outer_repeat'], row['outer_fold'], row['model_seed']),
                       expected, 'fold/seed diagnostic')
    fold_records, spreads = [], []
    for repeat, fold in folds:
        values = [indexed[repeat, fold, seed][metric] for seed in p['fit_seeds']]
        defined = [float(value) for value in values if value is not None]
        fold_records.append(dict(outer_repeat=repeat, outer_fold=fold, mean=_mean(values),
                                 contributing_seeds=len(defined), expected_seeds=len(values)))
        if len(defined) > 1:
            spreads.append(float(np.std(defined, ddof=1)))
    means = [row['mean'] for row in fold_records if row['mean'] is not None]
    return dict(mean=_mean(means), outer_fold_sd=float(np.std(means, ddof=1)) if len(means)>1 else None,
                mean_within_fold_seed_sd=_mean(spreads), n_expected_folds=len(folds),
                n_contributing_folds=len(means), expected_seed_cells=len(expected),
                contributing_seed_cells=sum(row['contributing_seeds'] for row in fold_records),
                fold_means=fold_records)


def _aggregate(rows, grouping, metrics, p):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in grouping)].append(row)
    return [dict(zip(grouping, key), metrics={metric: _nested_stats(values, metric, p) for metric in metrics})
            for key, values in groups.items()]


def summarize_view_diagnostics(results, p):
    """Average views/pairs, then fitting seeds, then equally weighted folds.

    Undefined pair correlations are excluded only from correlation means. Every
    expected cell and its missing/defined pair counts is retained. Means over
    available seed/fold cells expose their reduced denominators explicitly.
    """
    expected_jobs = {(name, repeat, fold) for name in p['panels']['e05']
                     for repeat in range(p['outer_repeats']) for fold in range(p['outer_folds'])}
    if not expected_jobs:
        raise ValueError('An E05 panel must be declared')
    indexed = _indexed(results, lambda result: tuple(result['provenance'][key]
                       for key in ('dataset_id', 'outer_repeat', 'outer_fold')), expected_jobs, 'E05 job')
    groups = {(scheme, seed, head) for scheme in p['e05_schemes'] for seed in p['fit_seeds'] for head in p['head_conditions']}
    all_pairs = set(combinations(range(p['e05_views']), 2))
    widths = p['architectures'][p['primary_architecture_index']]
    prefix_records, pair_records, layer_records = [], [], []
    for (name, repeat, fold), result in indexed.items():
        if result['status'] != 'ok' or result['provenance']['family'] != 'e05':
            raise ValueError('Only successful verified E05 jobs can be summarized')
        diagnostics = _indexed(result['diagnostics'], lambda row: (row['scheme'],row['model_seed'],row['head_update']),
                               groups, 'view diagnostic')
        fits = _indexed(result['fits'], lambda row: (row['scheme'],row['model_seed'],row['head_update'],row['view']),
                        {(*key, view) for key in groups for view in range(p['e05_views'])}, 'view fit')
        for (scheme, seed, head), diagnostic in diagnostics.items():
            base = dict(dataset_id=name, outer_repeat=repeat, outer_fold=fold,
                        scheme=scheme, model_seed=seed, head_update=head)
            prefixes = _indexed(diagnostic['prefixes'], lambda row: row['prefix'], p['e05_prefixes'], 'prefix')
            pairs = _indexed(diagnostic['pairs'], lambda row: (row['view_a'],row['view_b']), all_pairs, 'view pair')
            layers = {}
            for view in range(p['e05_views']):
                fit = fits[scheme, seed, head, view]
                if fit['status'] != 'ok':
                    raise ValueError('Failed view cannot enter diagnostics')
                layers[view] = _indexed(fit['layers'], lambda row: row['depth'], range(1,len(widths)+1), 'hidden layer')
                if any(layers[view][depth]['prototype_count'] != count for depth,count in enumerate(widths,1)):
                    raise ValueError('Hidden prototype count differs from architecture')
            for prefix, values in prefixes.items():
                prefix_records.append(dict(base, prefix=prefix, **{metric:values[metric] for metric in PREFIX_METRICS}))
                selected_pairs = [pair for (a,b), pair in pairs.items() if b<prefix]
                missing = Counter(pair['correlation_reason'] for pair in selected_pairs if pair['error_correlation'] is None)
                pair_records.append(dict(base, prefix=prefix, total_pairs=len(selected_pairs),
                    defined_correlation_pairs=sum(pair['error_correlation'] is not None for pair in selected_pairs),
                    correlation_missing_reasons=dict(missing), no_pair_reason='one-view prefix has no pair' if prefix==1 else None,
                    **{metric:_mean(pair[metric] for pair in selected_pairs) for metric in PAIR_METRICS}))
                for depth,count in enumerate(widths,1):
                    layer_records.append(dict(base, prefix=prefix, depth=depth, prototype_count=count,
                        **{metric:_mean(layers[view][depth][metric] for view in range(prefix)) for metric in LAYER_METRICS}))
    grouping = ('dataset_id','scheme','head_update','prefix')
    pairs = _aggregate(pair_records, grouping, PAIR_METRICS, p)
    for summary in pairs:
        selected = [row for row in pair_records if all(row[key]==summary[key] for key in grouping)]
        missing = Counter()
        for row in selected:
            missing.update(row['correlation_missing_reasons'])
        summary.update(total_pairs=sum(row['total_pairs'] for row in selected),
                       defined_correlation_pairs=sum(row['defined_correlation_pairs'] for row in selected),
                       correlation_missing_reasons=dict(missing))
    return dict(prefix_summaries=_aggregate(prefix_records, grouping, PREFIX_METRICS, p),
                pair_summaries=pairs,
                layer_summaries=_aggregate(layer_records, grouping+('depth','prototype_count'), LAYER_METRICS, p),
                prefix_seed_fold_records=prefix_records, pair_seed_fold_records=pair_records,
                layer_seed_fold_records=layer_records,
                aggregation='Mean across pairs/views within a replicate; mean fitting seeds within a fold; equal-weight mean folds.',
                missing_correlation_policy='Available replicate means only, with every expected cell, defined pair count, missing reason and contributing fold/seed count retained.',
                inferential_significance_claims=False)
