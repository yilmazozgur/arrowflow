"""Task 10b: arrowflow_full_knn (knn run) versus arrowflow_full (bridge run) across two nested runs.

The fixture runs mimic the saved layout of a complete run: protocol, environment, candidates, planned jobs,
per-dataset data/manifest/splits, one result file and fit log per planned job, and summary.json. Every job carries the
complete inner selection history the harness writes (evaluation.select_model: every candidate screened on every inner
fold with the first fitting seed; for stochastic families the three best screened candidates rescored with the other
two seeds), with scripted inner scores, so reporting.validate_result_records accepts it. The field layouts were copied
read-only from runs/2026-09-12-bridge (results/iris__svc_rbf__r0f0.json, summary.json, environment.json);
test_fixture_layout_mirrors_the_real_bridge_run pins the key sets and the history shape whenever that run is on this
machine. The panels are two tiny synthetic datasets with 15 outer folds of synthetic predictions. One valid pair and
its comparison are built once per session (`baseline`); tests that alter records work on a private copy (`runs`).
"""
import csv
import functools
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import stats
from experiments.make_revision import compare_runs as cr
from experiments.make_revision.evaluation import (ModelSpec, canonical_json, config_id, dataset_fingerprint,
                                                  expected_schedule, holm_adjust, make_splits, metric_values,
                                                  paired_corrected_interval, summarize_outer)
from experiments.make_revision.run_revision import planned_jobs, write_json

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'/'2026-09-12'
REAL_BRIDGE = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'/'2026-09-12-bridge'
METRICS = ('accuracy', 'error', 'balanced_accuracy', 'macro_f1')
OUTPUTS = ('knn_vs_full_contrasts.csv', 'knn_vs_full_summary.json', 'main_table.json')
ANALYSIS_SOURCES = ('compare_runs.py', 'evaluation.py', 'reporting.py', 'run_revision.py')
CSV_COLUMNS = ['dataset', 'model_a', 'model_b', 'mean_difference', 'standard_error', 'ci_low', 'ci_high',
               'p_approximate', 'holm_p_approximate', 'n_folds', 'df']
DATASETS = ('alpha', 'beta')
CLASSES = {'alpha': 2, 'beta': 3}
PER_CLASS = {'alpha': 10, 'beta': 8}
SEEDS = [8129, 19391, 39019]
FOLDS = [(repeat, fold) for repeat in range(3) for fold in range(5)]
BRIDGE_MODELS = ('arrowflow_full', 'dummy', 'svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting')
KNN_MODELS = ('arrowflow_full_knn',) + BRIDGE_MODELS[1:]
COMPARATORS = BRIDGE_MODELS[1:]
STOCHASTIC = {'arrowflow_full': True, 'arrowflow_full_knn': True, 'dummy': False, 'svc_rbf': False,
              'random_forest': True, 'mlp': True, 'numeric_knn': False, 'gradient_boosting': True}
ERROR_RATE = {('alpha', 'arrowflow_full'): .45, ('alpha', 'arrowflow_full_knn'): .1,
              ('beta', 'arrowflow_full'): .3, ('beta', 'arrowflow_full_knn'): .25}      # comparators: .2
BRIDGE_REVISION = '67073eb47dd61cc412600b5efb36304be0a6e3e7'
KNN_REVISION = '70fb9bf31092cb64e2bd349403ad090699a3494d'
ARROWFLOW_CANDIDATES = [{'aggregation': 'majority', 'batch_size': 32, 'degree_offset': offset, 'embed_scale': 1,
                         'iterations': 200, 'learning_rate': .1, 'n_views': 7, 'strategy': 'diverse',
                         'validation_ratio': .1, 'widths': [128]} for offset in (0, -1)]
CANDIDATES = {'arrowflow_full': ARROWFLOW_CANDIDATES, 'arrowflow_full_knn': ARROWFLOW_CANDIDATES, 'dummy': [{}],
              'svc_rbf': [{'C': 1, 'gamma': 'scale'}, {'C': 10, 'gamma': .1}],
              'random_forest': [{'max_depth': None, 'n_estimators': 100}, {'max_depth': 10, 'n_estimators': 300}],
              'mlp': [{'alpha': .01, 'hidden_layer_sizes': [64]}, {'alpha': .0001, 'hidden_layer_sizes': [128]}],
              'numeric_knn': [{'n_neighbors': 3, 'p': 2, 'weights': 'distance'}, {'n_neighbors': 5, 'p': 1, 'weights': 'uniform'}],
              'gradient_boosting': [{'learning_rate': .1, 'max_depth': 2}, {'learning_rate': .03, 'max_depth': 3}]}
ENVIRONMENT = {'code_revision': None, 'numeric_threads': 1, 'numpy': '1.26.4',
               'platform': 'Linux-6.8.0-138-generic-x86_64-with-glibc2.35',
               'python': '3.12.7 (main, Oct  1 2024, 08:52:12) [GCC 11.4.0]', 'registry': None, 'scipy': '1.14.1',
               'sklearn': '1.5.2', 'source_hashes': {'experiments/make_revision/evaluation.py':
                                                     '14f49187be28572c783c79bcdb54e1a90232d1212716e760f4a09c06f9ab7870'},
               'torch': '2.6.0+cu124'}
TIMING = {'classifier_fit_seconds': 0.0005362420006349566, 'encoding_seconds': 0.0005882360001123743,
          'fit_seconds': 0.0011365010004737996, 'fit_warnings': [], 'inference_seconds': 0.00011610399997152854,
          'inference_state_array_bytes': None, 'peak_process_rss_kib': 547296, 'predict_seconds': 0.0002331220002815826,
          'preprocessing_settings': "TimedPipeline(steps=[('imputer', NumericImputer()), ('classifier', SVC())])",
          'query_encoding_seconds': 0.00011701800031005405, 'representation_metadata': None, 'training_tie_rate': None}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs(models, candidates=None):
    candidates = {**CANDIDATES, **(candidates or {})}
    return {model: ModelSpec(model, None, candidates[model], STOCHASTIC[model]) for model in models}


def synthetic_dataset(name, shift=0.):
    y = np.repeat(np.arange(CLASSES[name]), PER_CLASS[name])
    X = np.random.RandomState(len(name)).normal(size=(len(y), 3)) + y[:, None] + shift
    return X, y


def synthetic_labels(name, model, repeat, fold, seed, test, y):
    digest = hashlib.sha256(canonical_json([name, model, repeat, fold, seed]).encode()).digest()
    wrong = np.random.RandomState(int.from_bytes(digest[:4], 'big')).random_sample(len(test)) < ERROR_RATE.get((name, model), .2)
    return [int((y[s] + 1) % CLASSES[name]) if w else int(y[s]) for s, w in zip(test, wrong)]


@functools.lru_cache(maxsize=None)
def cached_metrics(truth, labels):
    return metric_values(np.asarray(truth), list(labels))


def scripted_inner_score(rank, seed_index, inner_fold):
    """Inner accuracy that falls with the candidate's rank at every fitting seed and inner fold."""
    return round(.9 - .1*rank - .02*inner_fold - .01*seed_index, 6)


def selection_record(spec, split, seeds, chosen=0, history='complete'):
    """The selection record evaluation.select_model writes, under scripted scores where candidate `chosen` wins.

    history='first_inner_fit_only' keeps only the winner's first inner fit (the round-0 fixture of this file).
    """
    ids = [config_id(config) for config in spec.candidates]
    ranked = [ids[chosen]] + [cid for cid in ids if cid != ids[chosen]]
    fits = []

    def fit(cid, seed):
        for inner_fold, inner in enumerate(split['inner']):
            fits.append(dict(TIMING, config=spec.candidates[ids.index(cid)], config_id=cid, fit_rows=inner['train'],
                             inner_fold=inner_fold, model_id=spec.model_id, model_seed=seed,
                             outer_fold=split['outer_fold'], outer_repeat=split['outer_repeat'],
                             score=scripted_inner_score(ranked.index(cid), seeds.index(seed), inner_fold),
                             stage='inner', status='ok', training_sample_count=len(inner['train']),
                             validation_rows=inner['validation']))
    for cid in ids:                                              # screening: every candidate, first fitting seed
        fit(cid, seeds[0])
    finalists = ranked[:3] if spec.stochastic else ranked
    if spec.stochastic:                                          # reranking: the finalists, the other fitting seeds
        for cid in finalists:
            for seed in seeds[1:]:
                fit(cid, seed)
    active = seeds if spec.stochastic else seeds[:1]
    score = float(np.mean([scripted_inner_score(0, seeds.index(seed), inner_fold)
                           for seed in active for inner_fold in range(len(split['inner']))]))
    if history == 'first_inner_fit_only':
        fits, finalists = [row for row in fits if row['config_id'] == ranked[0]][:1], ranked[:1]
    return {'config': spec.candidates[chosen], 'config_id': ids[chosen], 'finalist_ids': finalists, 'fits': fits,
            'inner_score': score}


def write_log(path, result):
    Path(path).write_text(''.join(canonical_json(row) + '\n' for row in result['selection']['fits'] + result['models']))


def summary_record(revision, rows, schedule):
    models = list(schedule['expected_seeds'])
    return {'code_revision': revision, 'hypothesis_tests': [], 'model_rows': rows,
            'purpose': 'complete_nested_benchmark_metrics_recomputed_from_predictions',
            'selection_audit': 'Reconstructed from all recorded inner scores; inner predictions were not saved.',
            'summaries': {name: [summarize_outer(model_rows, model, metric, expected_folds=schedule['expected_folds'],
                                                 expected_seeds=schedule['expected_seeds'][model])
                                 for model in models for metric in METRICS] for name, model_rows in rows.items()}}


def build_run(root, *, protocol, models, revision, registry, candidates=None, split_seeds=None, data_shift=None,
              flips=None, config_choice=None, history='complete'):
    """A complete, internally consistent run directory; returns its outer model rows per dataset."""
    registry_specs = specs(models, candidates)
    write_json(root/'protocol.json', protocol)
    write_json(root/'environment.json', dict(ENVIRONMENT, code_revision=revision, registry=registry))
    write_json(root/'candidates.json', {m: {'stochastic': s.stochastic, 'candidates': s.candidates,
                                            'config_ids': [config_id(c) for c in s.candidates]}
                                        for m, s in registry_specs.items()})
    jobs = planned_jobs(protocol['datasets'], protocol, registry_specs)
    write_json(root/'planned_jobs.json', jobs)
    prepared = {}
    for name in protocol['datasets']:
        X, y = synthetic_dataset(name, (data_shift or {}).get(name, 0.))
        features, labels = ['x0', 'x1', 'x2'], [f'class{k}' for k in range(CLASSES[name])]
        splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'],
                             (split_seeds or {}).get(name, protocol['split_seed']))
        manifest = {'class_counts': np.bincount(y).tolist(), 'dataset_hash': dataset_fingerprint(X, y, features, labels),
                    'dataset_id': name, 'feature_names': features, 'label_map': labels,
                    'sample_order': 'source row order; zero-based sample_id', 'shape': list(X.shape),
                    'source': 'synthetic test fixture', 'splits_hash': config_id(splits)}
        write_json(root/name/'manifest.json', manifest)
        write_json(root/name/'splits.json', splits)
        np.savez_compressed(root/name/'data.npz', X=X, y=y)
        prepared[name] = (y, manifest, splits)
    rows = {name: [] for name in protocol['datasets']}
    for job in jobs:
        name, model, repeat, fold = job['dataset_id'], job['model_id'], job['outer_repeat'], job['outer_fold']
        y, manifest, splits = prepared[name]
        split = splits[repeat*protocol['outer_folds'] + fold]
        selection = selection_record(registry_specs[model], split, protocol['fit_seeds'],
                                     (config_choice or {}).get((name, model, repeat, fold), 0), history)
        common = {'code_revision': revision, 'condition': 'clean', 'config_id': selection['config_id'],
                  'dataset_hash': manifest['dataset_hash'], 'dataset_id': name, 'model_id': model, 'outer_fold': fold,
                  'outer_repeat': repeat, 'perturbation_seed': None, 'view_id': 'ensemble'}
        predictions, outer = [], []
        for seed in job['model_seeds']:
            labels = synthetic_labels(name, model, repeat, fold, seed, split['test'], y)
            for position in (flips or {}).get((name, model, repeat, fold, seed), ()):
                labels[position] = (labels[position] + 1) % CLASSES[name]
            predictions += [dict(common, model_seed=seed, sample_id=s, y_true=int(y[s]), y_pred=p)
                            for s, p in zip(split['test'], labels)]
            outer.append(dict(common, **TIMING, config=selection['config'], fit_rows=split['train'], test_rows=split['test'],
                              model_seed=seed, stage='outer', status='ok', training_sample_count=len(split['train']),
                              **cached_metrics(tuple(int(v) for v in y[split['test']]), tuple(labels))))
        result = {'models': outer, 'predictions': predictions, 'selection': selection, 'status': 'ok'}
        write_json(root/job['result_file'], result)
        write_log(root/job['log_file'], result)
        rows[name].extend(outer)
    write_json(root/'summary.json', summary_record(revision, rows, expected_schedule(protocol, registry_specs)))
    return rows


def build_pair(root, *, bridge_protocol=None, knn_protocol=None, bridge=None, knn=None, both=None):
    """A bridge run, then a knn run whose reference block pins it; bridge/knn/both are build_run options."""
    both = both or {}
    bridge_p = json.loads((PROTOCOLS/'bridge.json').read_text())
    bridge_p.update(datasets=list(DATASETS), **(bridge_protocol or {}))
    bridge_rows = build_run(root/'bridge', protocol=bridge_p, revision=BRIDGE_REVISION,
                            registry='experiments.make_revision.bridge:bridge_registry',
                            **{'models': BRIDGE_MODELS, **both, **(bridge or {})})
    knn_p = json.loads((PROTOCOLS/'bridge_knn.json').read_text())
    knn_p.update(datasets=list(DATASETS), primary_family_size=len(DATASETS), **(knn_protocol or {}))
    knn_p['knn_readout']['reference'].update(code_revision=BRIDGE_REVISION, protocol_sha256=sha256(root/'bridge'/'protocol.json'),
                                             summary_sha256=sha256(root/'bridge'/'summary.json'))
    knn_rows = build_run(root/'knn', protocol=knn_p, revision=KNN_REVISION,
                         registry='experiments.make_revision.bridge:bridge_knn_registry',
                         **{'models': KNN_MODELS, **both, **(knn or {})})
    return SimpleNamespace(bridge=root/'bridge', knn=root/'knn', bridge_rows=bridge_rows, knn_rows=knn_rows)


def recorded_comparison(runs, output):
    """compare_knn with every call of the two harness validators recorded."""
    calls = {'collect_confirmatory_results': [], 'validate_result_records': []}
    collect, validate = cr.collect_confirmatory_results, cr.validate_result_records

    def recording_collect(path, *args):
        calls['collect_confirmatory_results'].append(Path(path).name)
        return collect(path, *args)

    def recording_validate(result, job, *args):
        calls['validate_result_records'].append(job['result_file'])
        return validate(result, job, *args)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cr, 'collect_confirmatory_results', recording_collect)
        patch.setattr(cr, 'validate_result_records', recording_validate)
        return cr.compare_knn(runs.knn, runs.bridge, output), calls


@pytest.fixture(scope='session')
def baseline(tmp_path_factory):
    """One complete, valid pair (never altered) and its comparison."""
    root = tmp_path_factory.mktemp('baseline')
    runs = build_pair(root)
    result, calls = recorded_comparison(runs, root/'out')
    return SimpleNamespace(runs=runs, result=result, calls=calls, out=root/'out')


def private_copy(baseline, root):
    for label in ('bridge', 'knn'):
        shutil.copytree(getattr(baseline.runs, label), root/label)
    return SimpleNamespace(bridge=root/'bridge', knn=root/'knn', bridge_rows=baseline.runs.bridge_rows,
                           knn_rows=baseline.runs.knn_rows)


@pytest.fixture
def runs(baseline, tmp_path):
    return private_copy(baseline, tmp_path/'runs')


def fold_means(rows, model, metric):
    groups = {}
    for row in rows:
        if row['model_id'] == model:
            groups.setdefault((row['outer_repeat'], row['outer_fold']), []).append(row[metric])
    return groups


def assert_refused(runs, tmp_path, message):
    with pytest.raises(cr.RunComparisonError) as refused:
        cr.compare_knn(runs.knn, runs.bridge, tmp_path/'out')
    assert message in str(refused.value), str(refused.value)
    assert not (tmp_path/'out').exists()


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def edit_reference(runs, **values):
    rewrite(runs.knn/'protocol.json', lambda protocol: protocol['knn_readout']['reference'].update(values))


def job_file(root, index, key):
    return root/json.loads((root/'planned_jobs.json').read_text())[index][key]


def rewrite_result(root, index, change):
    """Change one saved result and rewrite its fit log to match, as a consistent edit would."""
    job = json.loads((root/'planned_jobs.json').read_text())[index]
    rewrite(root/job['result_file'], change)
    write_log(root/job['log_file'], json.loads((root/job['result_file']).read_text()))


def rewrite_knn_summary_rows(runs, change):
    """Change the knn run's summary.json model_rows and rebuild its summaries from them (a self-consistent summary)."""
    path = runs.knn/'summary.json'
    protocol = json.loads((runs.knn/'protocol.json').read_text())
    rows = json.loads(path.read_text())['model_rows']
    change(rows)
    record = summary_record(KNN_REVISION, rows, expected_schedule(protocol, specs(KNN_MODELS)))
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


# ----------------------------------------------------------------------------- the primary contrast family

def test_contrast_is_the_seed_averaged_corrected_t_of_knn_minus_full_with_holm_across_datasets(baseline):
    runs, out, contrasts = baseline.runs, baseline.out, baseline.result['contrasts']
    assert [row['dataset'] for row in contrasts] == list(DATASETS)
    t_critical = stats.t.ppf(.975, 14)
    for row, name in zip(contrasts, DATASETS):
        combined = ([r for r in runs.knn_rows[name] if r['model_id'] == 'arrowflow_full_knn']
                    + [r for r in runs.bridge_rows[name] if r['model_id'] == 'arrowflow_full'])
        direct = paired_corrected_interval(combined, 'arrowflow_full_knn', 'arrowflow_full', metric='accuracy', q=.25,
                                           confidence=.95, expected_folds=FOLDS,
                                           expected_seeds={'arrowflow_full_knn': SEEDS, 'arrowflow_full': SEEDS})
        assert (row['model_a'], row['model_b'], row['n_folds'], row['df']) == ('arrowflow_full_knn', 'arrowflow_full', 15, 14)
        for key in ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_approximate'):
            assert row[key] == direct[key]
        # by hand: fitting seeds averaged within each outer fold, then the corrected resampled t over 15 folds
        knn, full = fold_means(combined, 'arrowflow_full_knn', 'accuracy'), fold_means(combined, 'arrowflow_full', 'accuracy')
        assert all(len(knn[k]) == len(full[k]) == 3 for k in FOLDS)
        differences = np.array([np.mean(knn[k]) - np.mean(full[k]) for k in FOLDS])
        se = np.sqrt((1/15 + .25) * differences.var(ddof=1))
        assert row['mean_difference'] == pytest.approx(differences.mean(), abs=1e-12)
        assert row['standard_error'] == pytest.approx(se, abs=1e-12)
        assert (row['ci_low'], row['ci_high']) == pytest.approx((differences.mean() - t_critical*se, differences.mean() + t_critical*se), abs=1e-12)
        assert row['p_approximate'] == pytest.approx(2*stats.t.sf(abs(differences.mean()/se), 14), abs=1e-12)
    assert contrasts[0]['mean_difference'] > 0 and contrasts[0]['p_approximate'] < .05       # the fixture's kNN errs less on alpha
    p = [row['p_approximate'] for row in contrasts]
    holm = [row['holm_p_approximate'] for row in contrasts]
    assert holm == holm_adjust(p)
    low, high = int(np.argmin(p)), int(np.argmax(p))
    assert holm[low] == pytest.approx(min(1., 2*p[low])) and holm[high] == pytest.approx(max(holm[low], p[high]))
    with (out/'knn_vs_full_contrasts.csv').open() as stream:
        saved = list(csv.reader(stream))
    assert saved[0] == CSV_COLUMNS and len(saved) == 1 + len(DATASETS)
    for line, row in zip(saved[1:], contrasts):
        record = dict(zip(CSV_COLUMNS, line))
        assert [record[k] for k in ('dataset', 'model_a', 'model_b')] == [row['dataset'], 'arrowflow_full_knn', 'arrowflow_full']
        assert all(float(record[k]) == row[k] for k in CSV_COLUMNS[3:9]) and (int(record['n_folds']), int(record['df'])) == (15, 14)
    summary = json.loads((out/'knn_vs_full_summary.json').read_text())
    assert summary == json.loads(json.dumps(baseline.result['summary']))
    assert summary['contrasts'] == json.loads(json.dumps(contrasts))
    assert summary['comparators_reproduced'] is True and summary['reproduction']['comparators_reproduced'] is True
    assert summary['reproduction']['mismatches'] == [] and summary['reproduction']['cells_compared'] == len(DATASETS) * 15 * 12
    assert summary['reproduction']['comparators'] == list(COMPARATORS)
    for label, root, revision in (('bridge', runs.bridge, BRIDGE_REVISION), ('knn', runs.knn, KNN_REVISION)):
        source = summary['provenance'][label]
        assert source['code_revision'] == revision and source['summary_sha256'] == sha256(root/'summary.json')
        assert source['protocol_sha256'] == sha256(root/'protocol.json')
    pairing = summary['provenance']['pairing']
    assert pairing['datasets'] == list(DATASETS) and pairing['fit_seeds'] == SEEDS
    assert (pairing['outer_folds'], pairing['outer_repeats']) == (5, 3)
    assert pairing['dataset_hash'] == {n: json.loads((runs.bridge/n/'manifest.json').read_text())['dataset_hash'] for n in DATASETS}
    assert pairing['splits_hash'] == {n: json.loads((runs.knn/n/'manifest.json').read_text())['splits_hash'] for n in DATASETS}
    assert summary['family']['size'] == len(DATASETS) and summary['family']['multiplicity'] == 'holm'


def test_summary_records_the_reproduction_criterion_the_csv_sha256_and_the_analysis_sources(baseline):
    summary = json.loads((baseline.out/'knn_vs_full_summary.json').read_text())
    table = json.loads((baseline.out/'main_table.json').read_text())
    criterion = summary['definitions']['comparators_reproduced']
    assert criterion == summary['reproduction']['criterion'] == cr.REPRODUCTION_CRITERION
    for phrase in ('candidate definitions are identical in both runs', 'stochastic flag, candidate list and config_ids',
                   'the same config_id', 'identical per-example predictions', 'the same test sample IDs'):
        assert phrase in criterion
    assert 'definitions.comparators_reproduced' in table['definitions']['reproduced_bridge_exactly']
    assert summary['outputs'] == {'knn_vs_full_contrasts.csv': {'sha256': sha256(baseline.out/'knn_vs_full_contrasts.csv'),
                                                                'rows': len(DATASETS)}}
    sources = {f'experiments/make_revision/{name}': sha256(REPO/'experiments'/'make_revision'/name) for name in ANALYSIS_SOURCES}
    assert table['analysis_sources'] == summary['provenance']['analysis_sources'] == sources


# ----------------------------------------------------------------------------- full re-verification of both runs

def test_a_fully_valid_pair_is_accepted_after_every_saved_record_of_both_runs_is_reverified(baseline):
    runs, calls = baseline.runs, baseline.calls
    jobs = {label: json.loads((getattr(runs, label)/'planned_jobs.json').read_text()) for label in ('knn', 'bridge')}
    assert calls['collect_confirmatory_results'] == ['knn', 'bridge']
    assert calls['validate_result_records'] == [job['result_file'] for label in ('knn', 'bridge') for job in jobs[label]]
    verification = baseline.result['summary']['provenance']['verification']
    assert verification['validators'] == list(cr.VALIDATORS)
    for label in ('knn', 'bridge'):
        rows = getattr(runs, f'{label}_rows')
        assert verification[label] == {'jobs_verified': len(jobs[label]), 'model_rows_verified': sum(map(len, rows.values()))}
    assert len(jobs['knn']) == len(jobs['bridge']) == len(DATASETS) * 15 * 7


@pytest.mark.parametrize('case', ['one_inner_fit_per_job', 'one_screening_fit_missing'])
def test_a_pair_with_an_incomplete_screening_history_is_refused(baseline, tmp_path, case):
    if case == 'one_inner_fit_per_job':                     # the round-0 fixture, which compare_runs used to accept
        runs = build_pair(tmp_path/'runs', both={'history': 'first_inner_fit_only'})
        where = 'knn run results/alpha__arrowflow_full_knn__r0f0.json'
    else:
        runs = private_copy(baseline, tmp_path/'runs')

        def drop_last_screening_fit(result):
            fits = result['selection']['fits']
            fits.remove([row for row in fits if row['model_seed'] == SEEDS[0]][-1])
        rewrite_result(runs.bridge, 0, drop_last_screening_fit)
        where = 'bridge run results/alpha__arrowflow_full__r0f0.json'
    assert_refused(runs, tmp_path, f'{where} fails reporting.validate_result_records: Incomplete candidate screening history')


# ----------------------------------------------------------------------------- one behavioural test per guard

def test_guard_error_summary_error_must_equal_the_error_reverified_from_predictions(runs, tmp_path):
    """The only link from main_table mean_error to the saved predictions."""
    def change(rows):
        row = rows['beta'][1]
        assert (row['model_id'], row['outer_repeat'], row['outer_fold'], row['model_seed']) == ('arrowflow_full_knn', 0, 0, 19391)
        row['error'] += .05
    rewrite_knn_summary_rows(runs, change)
    assert_refused(runs, tmp_path, 'knn run summary.json model_rows for beta differ from the model rows re-verified from '
                                   'the saved records (row 1, model_id arrowflow_full_knn outer_repeat 0 outer_fold 0 '
                                   'model_seed 19391: error)')


def test_guard_definitions_a_comparator_with_other_candidate_definitions_is_not_reproduced(tmp_path):
    changed = [CANDIDATES['svc_rbf'][0], {'C': 100, 'gamma': .01}]          # the selected first candidate is unchanged
    runs = build_pair(tmp_path/'runs', knn={'candidates': {'svc_rbf': changed}})
    record = cr.compare_knn(runs.knn, runs.bridge, tmp_path/'out')['summary']['reproduction']
    assert record['mismatches'] == [] and record['cells_mismatched'] == 0     # every prediction and selected config agrees
    assert record['definition_mismatches'] == [{'model_id': 'svc_rbf', 'field': 'candidates'},
                                               {'model_id': 'svc_rbf', 'field': 'config_ids'}]
    assert record['comparators_reproduced'] is False
    assert record['reproduced_by_dataset']['svc_rbf'] == {'alpha': False, 'beta': False}
    assert all(all(flags.values()) for model, flags in record['reproduced_by_dataset'].items() if model != 'svc_rbf')


@pytest.mark.parametrize('declared', ['missing', 3])
def test_guard_primary_family_size_is_required_and_must_equal_the_panel(runs, tmp_path, declared):
    def change(protocol):
        if declared == 'missing':
            del protocol['primary_family_size']
        else:
            protocol['primary_family_size'] = declared
    rewrite(runs.knn/'protocol.json', change)
    assert_refused(runs, tmp_path, 'The knn protocol declares no primary_family_size (the Holm family size is required)'
                   if declared == 'missing' else 'The knn protocol primary_family_size 3 differs from its panel of 2 datasets')


def test_guard_truth_a_prediction_whose_truth_differs_from_the_prepared_labels_is_refused(runs, tmp_path):
    def change(result):
        row = result['predictions'][0]
        row['y_true'] = (row['y_true'] + 1) % CLASSES['alpha']
    rewrite_result(runs.knn, 0, change)
    assert_refused(runs, tmp_path, 'knn run results/alpha__arrowflow_full_knn__r0f0.json fails '
                                   'reporting.validate_result_records: Prediction truth or label mismatch')


def test_guard_status_a_result_not_marked_ok_is_refused(runs, tmp_path):
    rewrite_result(runs.knn, 0, lambda result: result.update(status='failed_outer'))
    assert_refused(runs, tmp_path, 'knn run fails run_revision.collect_confirmatory_results: Incomplete planned evidence: '
                                   'failed_outer: results/alpha__arrowflow_full_knn__r0f0.json')


def test_guard_splits_a_pair_run_on_other_than_the_declared_nested_splits_is_refused(tmp_path):
    runs = build_pair(tmp_path/'runs', both={'split_seeds': {'beta': 99}})    # the runs agree with each other only
    assert_refused(runs, tmp_path, 'knn run beta/splits.json is not the nested splits the protocol declares '
                                   '(outer_folds, outer_repeats, inner_folds, split_seed)')


@pytest.mark.parametrize('field, bridge_value', [('protocol_id', 'arrowflow-v3-bridge-1'), ('model_id', 'arrowflow_full')])
def test_guard_reference_identity_protocol_and_model_ids_must_match_the_bridge_run(runs, tmp_path, field, bridge_value):
    edit_reference(runs, **{field: 'something-else'})
    assert_refused(runs, tmp_path, 'The knn protocol reference block does not match the bridge run: '
                                   f"{field}: declared 'something-else', bridge run has '{bridge_value}'")


def test_guard_families_runs_with_different_comparator_families_are_refused(tmp_path):
    runs = build_pair(tmp_path/'runs', knn={'models': KNN_MODELS[:-1]})       # gradient_boosting not refitted
    assert_refused(runs, tmp_path, "The runs hold different comparator families (knn run only: [], bridge run only: "
                                   "['gradient_boosting'])")


@pytest.mark.parametrize('declared', [['arrowflow_full_knn_vs_dummy'], 'arrowflow_full_knn_vs_arrowflow_full'])
def test_guard_primary_contrast_the_knn_protocol_must_list_the_contrast(runs, tmp_path, declared):
    rewrite(runs.knn/'protocol.json', lambda protocol: protocol.update(primary_contrasts=declared))
    assert_refused(runs, tmp_path, 'The knn protocol primary_contrasts do not declare arrowflow_full_knn_vs_arrowflow_full')


@pytest.mark.parametrize('field', ['dataset_hash', 'code_revision'])
def test_guard_identity_summary_model_rows_must_carry_the_reverified_identity(runs, tmp_path, field):
    rows = runs.knn_rows['alpha']
    index = len(rows) - 1
    assert (rows[index]['model_id'], rows[index]['outer_repeat'], rows[index]['outer_fold'], rows[index]['model_seed']) == (
        'gradient_boosting', 2, 4, 39019)
    rewrite_knn_summary_rows(runs, lambda published: published['alpha'][index].update({field: 'f'*len(rows[index][field])}))
    assert_refused(runs, tmp_path, f'knn run summary.json model_rows for alpha differ from the model rows re-verified from '
                                   f'the saved records (row {index}, model_id gradient_boosting outer_repeat 2 outer_fold 4 '
                                   f'model_seed 39019: {field})')


# ----------------------------------------------------------------------------- other refusals

def tamper_prediction_metric(runs):
    """A self-consistent summary.json whose model row no longer matches the saved per-example predictions."""
    def change(rows):
        row = rows['alpha'][0]
        assert row['model_id'] == 'arrowflow_full_knn'
        accuracy = 0. if row['accuracy'] > 0 else 1.
        row.update(accuracy=accuracy, error=1 - accuracy)
    rewrite_knn_summary_rows(runs, change)


def tamper_summary_mean(runs):
    def change(summary):
        row = summary['summaries']['beta'][5]
        assert (row['model_id'], row['metric']) == ('dummy', 'error')
        row['mean'] += .01
    rewrite(runs.knn/'summary.json', change)


REFUSALS = {        # name: (build_pair options, or None for a private copy of the baseline pair; edit; message)
    'knn_summary_missing': (None, lambda runs: (runs.knn/'summary.json').unlink(), 'incomplete'),
    'bridge_result_missing': (None, lambda runs: job_file(runs.bridge, 0, 'result_file').unlink(), 'incomplete'),
    'knn_fit_log_missing': (None, lambda runs: job_file(runs.knn, -1, 'log_file').unlink(), 'incomplete'),
    'truncated_planned_jobs': (None, lambda runs: rewrite(runs.knn/'planned_jobs.json', lambda jobs: jobs.pop()), 'planned_jobs'),
    'unfrozen_bridge': (None, lambda runs: rewrite(runs.bridge/'protocol.json', lambda p: p.update(frozen=False)), 'not frozen'),
    'reference_summary_sha256': (None, lambda runs: edit_reference(runs, summary_sha256='0'*64), 'reference'),
    'reference_code_revision': (None, lambda runs: edit_reference(runs, code_revision='1'*40), 'reference'),
    'reference_protocol_sha256': (None, lambda runs: edit_reference(runs, protocol_sha256='2'*64), 'reference'),
    'bridge_summary_is_not_the_pinned_file': (None, lambda runs: (runs.bridge/'summary.json').write_text(
        (runs.bridge/'summary.json').read_text() + '\n'), 'reference'),
    'dataset_hash': ({'knn': {'data_shift': {'alpha': .5}}}, None, 'dataset hash'),
    'split_hash': ({'knn': {'split_seeds': {'beta': 99}}}, None, 'splits hash'),
    'fit_seeds': ({'knn_protocol': {'fit_seeds': [8129, 19391, 40009]}}, None, 'fit_seeds'),
    'outer_schedule': ({'knn_protocol': {'outer_repeats': 2}}, None, 'outer_repeats'),
    'candidates': ({'knn': {'candidates': {'arrowflow_full_knn': ARROWFLOW_CANDIDATES[:1]}}}, None, 'candidates'),
    'summary_accuracy_disagrees_with_predictions': (None, tamper_prediction_metric, r're-verified .*: accuracy, error\)'),
    'summaries_disagree_with_model_rows': (None, tamper_summary_mean, 'model_rows'),
}


@pytest.mark.parametrize('change', sorted(REFUSALS))
def test_refuses_incomplete_unverified_or_unpaired_runs_before_writing_anything(baseline, tmp_path, change):
    options, edit, message = REFUSALS[change]
    runs = private_copy(baseline, tmp_path/'runs') if options is None else build_pair(tmp_path/'runs', **options)
    if edit is not None:
        edit(runs)
    with pytest.raises(cr.RunComparisonError, match=message):
        cr.compare_knn(runs.knn, runs.bridge, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('case', ['truncated_data_npz', 'knn_readout_not_an_object', 'output_file_is_a_directory',
                                  'output_is_a_file'])
def test_failures_that_used_to_escape_exit_2_with_a_clear_message(runs, tmp_path, capsys, case):
    output = tmp_path/'out'
    if case == 'truncated_data_npz':
        path = runs.knn/'beta'/'data.npz'
        path.write_bytes(path.read_bytes()[:path.stat().st_size // 2])
        message = 'knn run prepared data for beta is unreadable or does not match its manifest (BadZipFile: File is not a zip file)'
    elif case == 'knn_readout_not_an_object':
        rewrite(runs.knn/'protocol.json', lambda protocol: protocol.update(knn_readout=['reference']))
        message = 'The knn protocol knn_readout must be an object holding the reference block, not list'
    elif case == 'output_file_is_a_directory':
        (output/'main_table.json').mkdir(parents=True)
        message = f'Output {output/"main_table.json"} exists and is not a regular file'
    else:
        output.write_text('not a directory\n')
        message = f'Output {output} exists and is not a directory'
    with pytest.raises(SystemExit) as refused:
        cr.main(['knn', '--knn-source', str(runs.knn), '--bridge-source', str(runs.bridge), '--output', str(output)])
    assert refused.value.code == 2
    assert f'compare_runs knn refused: {message}\n' in capsys.readouterr().err
    if case == 'output_file_is_a_directory':
        assert [path.name for path in output.iterdir()] == ['main_table.json'] and not any((output/'main_table.json').iterdir())
    elif case == 'output_is_a_file':
        assert output.read_text() == 'not a directory\n'
    else:
        assert not output.exists()


# ----------------------------------------------------------------------------- comparator reproduction and the main table

@pytest.mark.parametrize('kind', ['prediction', 'selected_configuration'])
def test_comparator_reproduction_lists_every_mismatch_without_raising(baseline, tmp_path, kind):
    cell = ('beta', 'random_forest', 1, 3)
    options = {'flips': {cell + (19391,): (0, 2)}} if kind == 'prediction' else {'config_choice': {cell: 1}}
    runs = build_pair(tmp_path/'runs', knn=options)
    result = cr.compare_knn(runs.knn, runs.bridge, tmp_path/'out')
    record = result['summary']['reproduction']
    assert result['summary']['comparators_reproduced'] is False and record['comparators_reproduced'] is False
    assert record['cells_compared'] == len(DATASETS) * 15 * 12 and record['definition_mismatches'] == []
    test = json.loads((runs.knn/'beta'/'splits.json').read_text())[1*5 + 3]['test']
    first, second = (config_id(c) for c in CANDIDATES['random_forest'])
    identity = {'dataset_id': 'beta', 'model_id': 'random_forest', 'outer_repeat': 1, 'outer_fold': 3, 'n_test': len(test)}
    if kind == 'prediction':
        assert record['mismatches'] == [dict(identity, model_seed=19391, n_differing=2, sample_ids=[test[0], test[2]],
                                             config_id_knn=first, config_id_bridge=first)]
    else:
        assert record['mismatches'] == [dict(identity, model_seed=seed, n_differing=0, sample_ids=[],
                                             config_id_knn=second, config_id_bridge=first) for seed in SEEDS]
    assert record['cells_mismatched'] == len(record['mismatches'])
    flags = {(name, row['model_id']): row['reproduced_bridge_exactly']
             for name in DATASETS for row in result['main_table']['rows'][name]}
    assert flags.pop(('beta', 'random_forest')) is False
    assert all(flags[(name, model)] is True for name in DATASETS for model in COMPARATORS if (name, model) in flags)
    assert all(flags[(name, model)] is None for name in DATASETS for model in ('arrowflow_full', 'arrowflow_full_knn'))
    assert result['contrasts'] == baseline.result['contrasts']


def test_main_table_reads_each_family_from_its_own_verified_summary(tmp_path):
    runs = build_pair(tmp_path/'runs', knn={'flips': {('alpha', 'mlp', 2, 4, 8129): (1,)}})
    result = cr.compare_knn(runs.knn, runs.bridge, tmp_path/'out')
    table = json.loads((tmp_path/'out'/'main_table.json').read_text())
    assert table == json.loads(json.dumps(result['main_table']))
    assert table['metric'] == 'error' and table['datasets'] == list(DATASETS) and table['comparators_reproduced'] is False
    assert table['models'] == ['arrowflow_full', 'arrowflow_full_knn', *COMPARATORS]
    summaries = {label: json.loads((root/'summary.json').read_text())['summaries']
                 for label, root in (('bridge', runs.bridge), ('knn', runs.knn))}
    for name in DATASETS:
        assert [row['model_id'] for row in table['rows'][name]] == table['models']
        for row in table['rows'][name]:
            source = 'bridge' if row['model_id'] == 'arrowflow_full' else 'knn'
            published = next(s for s in summaries[source][name] if (s['model_id'], s['metric']) == (row['model_id'], 'error'))
            assert row['source_run'] == source
            assert ([row[k] for k in ('mean_error', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')]
                    == [published[k] for k in ('mean', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')])
    groups = fold_means(runs.bridge_rows['alpha'], 'arrowflow_full', 'error')               # independent arithmetic
    means = [np.mean(groups[k]) for k in FOLDS]
    full = table['rows']['alpha'][0]
    assert full['mean_error'] == pytest.approx(np.mean(means), abs=1e-12)
    assert full['outer_fold_sd'] == pytest.approx(np.std(means, ddof=1), abs=1e-12)
    assert full['mean_within_fold_seed_sd'] == pytest.approx(np.mean([np.std(groups[k], ddof=1) for k in FOLDS]), abs=1e-12)
    assert (full['n_folds'], full['seeds_per_fold']) == (15, 3)
    assert table['rows']['alpha'][2]['mean_within_fold_seed_sd'] is None and table['rows']['alpha'][2]['seeds_per_fold'] == 1
    mlp = next(row for row in table['rows']['alpha'] if row['model_id'] == 'mlp')
    bridge_mlp = next(s for s in summaries['bridge']['alpha'] if (s['model_id'], s['metric']) == ('mlp', 'error'))
    assert mlp['reproduced_bridge_exactly'] is False and mlp['mean_error'] != bridge_mlp['mean']   # the knn run's own refit
    assert all(row['reproduced_bridge_exactly'] is True for name in DATASETS for row in table['rows'][name]
               if row['model_id'] in COMPARATORS and (name, row['model_id']) != ('alpha', 'mlp'))


# ----------------------------------------------------------------------------- the command

def test_knn_command_end_to_end(baseline, runs, tmp_path, capsys):
    arguments = ['knn', '--knn-source', str(baseline.runs.knn), '--bridge-source', str(baseline.runs.bridge), '--output']
    completed = subprocess.run([sys.executable, '-m', 'experiments.make_revision.compare_runs', *arguments, str(tmp_path/'cli')],
                               cwd=REPO, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in (tmp_path/'cli').iterdir()) == sorted(OUTPUTS)
    assert 'alpha' in completed.stdout and 'comparators_reproduced: True' in completed.stdout
    assert all((tmp_path/'cli'/name).read_bytes() == (baseline.out/name).read_bytes() for name in OUTPUTS)
    cr.main([*arguments, str(tmp_path/'cli')])                      # the same evidence again: identical files, no error
    (tmp_path/'cli'/'main_table.json').write_text('{}\n')
    with pytest.raises(SystemExit) as refused:
        cr.main([*arguments, str(tmp_path/'cli')])
    assert refused.value.code == 2 and 'overwrite' in capsys.readouterr().err
    assert (tmp_path/'cli'/'main_table.json').read_text() == '{}\n'
    (runs.bridge/'summary.json').unlink()
    with pytest.raises(SystemExit) as refused:
        cr.main(['knn', '--knn-source', str(runs.knn), '--bridge-source', str(runs.bridge), '--output', str(tmp_path/'cli-incomplete')])
    assert refused.value.code == 2 and 'incomplete' in capsys.readouterr().err
    assert not (tmp_path/'cli-incomplete').exists()


# ----------------------------------------------------------------------------- the real frozen bridge run (read-only)

needs_real_bridge = pytest.mark.skipif(not REAL_BRIDGE.is_dir(), reason='the frozen bridge run is not on this machine')


def history_shape(root, dataset, model):
    """(complete fit count, screening order, finalist count) of one saved selection history."""
    result = json.loads((root/'results'/f'{dataset}__{model}__r0f0.json').read_text())
    declared = json.loads((root/'candidates.json').read_text())[model]
    n, stochastic, fits = len(declared['candidates']), declared['stochastic'], result['selection']['fits']
    screening = [(row['config_id'], row['model_seed'], row['inner_fold']) for row in fits[:3*n]]
    return (len(fits) == 3*n + (min(3, n)*2*3 if stochastic else 0),
            screening == [(cid, SEEDS[0], fold) for cid in declared['config_ids'] for fold in range(3)],
            len(result['selection']['finalist_ids']) == (min(3, n) if stochastic else n))


@needs_real_bridge
def test_fixture_layout_mirrors_the_real_bridge_run(baseline):
    def load(path):
        return json.loads(Path(path).read_text())
    runs = baseline.runs
    real, mine = load(REAL_BRIDGE/'results'/'iris__svc_rbf__r0f0.json'), load(runs.bridge/'results'/'alpha__svc_rbf__r0f0.json')
    assert set(mine) == set(real) and set(mine['selection']) == set(real['selection'])
    for key in ('models', 'predictions'):
        assert set(mine[key][0]) == set(real[key][0])
    assert set(mine['selection']['fits'][0]) == set(real['selection']['fits'][0])
    for model in BRIDGE_MODELS:
        assert history_shape(REAL_BRIDGE, 'iris', model) == history_shape(runs.bridge, 'alpha', model) == (True, True, True)
    real_summary, mine_summary = load(REAL_BRIDGE/'summary.json'), load(runs.bridge/'summary.json')
    assert set(mine_summary) == set(real_summary)
    assert set(mine_summary['summaries']['alpha'][0]) == set(real_summary['summaries']['iris'][0])
    assert set(mine_summary['model_rows']['alpha'][0]) == set(real_summary['model_rows']['iris'][0])
    assert [(r['model_id'], r['metric']) for r in mine_summary['summaries']['alpha']] == [
        (r['model_id'], r['metric']) for r in real_summary['summaries']['iris']]
    assert set(load(runs.bridge/'environment.json')) == set(load(REAL_BRIDGE/'environment.json'))
    assert set(load(runs.bridge/'alpha'/'manifest.json')) == set(load(REAL_BRIDGE/'iris'/'manifest.json'))
    real_jobs, mine_jobs = load(REAL_BRIDGE/'planned_jobs.json'), load(runs.bridge/'planned_jobs.json')
    assert set(mine_jobs[0]) == set(real_jobs[0]) and [j['model_id'] for j in mine_jobs[:7]] == [j['model_id'] for j in real_jobs[:7]]
    real_candidates, mine_candidates = load(REAL_BRIDGE/'candidates.json'), load(runs.bridge/'candidates.json')
    assert set(mine_candidates) == set(real_candidates)
    assert all(set(mine_candidates[m]) == set(real_candidates[m]) and mine_candidates[m]['stochastic'] == real_candidates[m]['stochastic']
               for m in real_candidates)
    assert sorted(p.name for p in (runs.bridge/'alpha').iterdir()) == sorted(p.name for p in (REAL_BRIDGE/'iris').iterdir())
    assert sorted(p.name for p in runs.bridge.iterdir() if p.is_file()) == sorted(p.name for p in REAL_BRIDGE.iterdir() if p.is_file())


@needs_real_bridge
def test_real_bridge_run_is_complete_and_is_the_reference_the_knn_protocol_pins():
    run = cr.load_run(REAL_BRIDGE, 'bridge')
    assert run.models[0] == 'arrowflow_full' and len(run.jobs) == 735
    record = cr.check_reference(json.loads((PROTOCOLS/'bridge_knn.json').read_text()), run)
    assert record['summary_sha256'] == run.summary_sha256 == sha256(REAL_BRIDGE/'summary.json')
    assert record['code_revision'] == run.environment['code_revision'] and record['protocol_sha256'] == sha256(REAL_BRIDGE/'protocol.json')


@needs_real_bridge
def test_real_bridge_run_passes_the_full_reverification():
    """The validator path on all 735 saved jobs of the frozen bridge run (about 20 s)."""
    run = cr.load_run(REAL_BRIDGE, 'bridge')
    rows, cells = cr.verify_run(run)
    assert sum(map(len, rows.values())) == len(cells) == 1575
    assert canonical_json(rows) == canonical_json(run.summary['model_rows'])
    assert {key[1] for key in cells} == set(run.models) and {key[0] for key in cells} == set(run.protocol['datasets'])


# ============================================================================= Task 20A: compare_runs training

from experiments.make_revision import knn_controls as kc

CONTROLS = ('arrowflow_knn_untrained', 'input_footrule_knn')
TRAINING_OUTPUTS = ('training_contrasts.csv', 'training_contrasts.json', 'training_error_table.json', 'training_depth_split.json')
TRAINING_REVISION = 'bfb9d8139' + '0' * 31
TRAINING_KNN_CANDIDATES = [{'aggregation': 'majority', 'batch_size': 32, 'degree_offset': offset, 'embed_scale': 1,
                            'iterations': 200, 'learning_rate': .1, 'n_views': 7, 'strategy': 'diverse',
                            'validation_ratio': .1, 'widths': widths} for widths in ([128], [64, 128]) for offset in (0, -1)]
TRAINING_CANDIDATES = {'arrowflow_full_knn': TRAINING_KNN_CANDIDATES,
                       **{model: kc.project_candidates(TRAINING_KNN_CANDIDATES, kc.CANDIDATE_KEYS[model]) for model in CONTROLS}}
STOCHASTIC.update({model: True for model in CONTROLS})
ERROR_RATE.update({('alpha', 'arrowflow_knn_untrained'): .35, ('alpha', 'input_footrule_knn'): .3,
                   ('beta', 'arrowflow_knn_untrained'): .3, ('beta', 'input_footrule_knn'): .2})
SHARED_HASHES = {source: hashlib.sha256(source.encode()).hexdigest() for source in cr.SHARED_SOURCES}
DEEP_FOLDS = [(repeat, fold) for repeat, fold in FOLDS if fold % 2 == 0]       # the fixture's ArrowFlow-kNN picks [64, 128]


def build_training_pair(root):
    """A bridge_knn run (ArrowFlow-kNN and the dummy) and a knn_training run whose reference block pins it."""
    knn_p = json.loads((PROTOCOLS/'bridge_knn.json').read_text())
    knn_p.update(datasets=list(DATASETS), primary_family_size=len(DATASETS))
    deep = next(i for i, c in enumerate(TRAINING_KNN_CANDIDATES) if c['widths'] == [64, 128])
    knn_rows = build_run(root/'knn', protocol=knn_p, models=('arrowflow_full_knn', 'dummy'), revision=KNN_REVISION,
                         registry='experiments.make_revision.bridge:bridge_knn_registry', candidates=TRAINING_CANDIDATES,
                         config_choice={(name, 'arrowflow_full_knn', repeat, fold): deep if (repeat, fold) in DEEP_FOLDS else 0
                                        for name in DATASETS for repeat, fold in FOLDS})
    rewrite(root/'knn'/'environment.json', lambda e: e.update(source_hashes=dict(SHARED_HASHES)))
    training_p = json.loads((PROTOCOLS/'knn_training.json').read_text())
    training_p.update(datasets=list(DATASETS), primary_family_size=2 * len(DATASETS), frozen=True,
                      frozen_at_utc='2026-09-13T12:00:00+00:00')     # a completed run's protocol is frozen
    training_p['training_controls']['reference'].update(code_revision=KNN_REVISION, protocol_sha256=sha256(root/'knn'/'protocol.json'),
                                                        summary_sha256=sha256(root/'knn'/'summary.json'))
    shallow = next(i for i, c in enumerate(TRAINING_CANDIDATES[CONTROLS[0]]) if c['widths'] == [128])
    training_rows = build_run(root/'training', protocol=training_p, models=CONTROLS, revision=TRAINING_REVISION,
                              registry='experiments.make_revision.knn_controls:knn_training_registry',
                              candidates=TRAINING_CANDIDATES,
                              config_choice={(name, CONTROLS[0], repeat, fold): shallow for name in DATASETS for repeat, fold in FOLDS})
    rewrite(root/'training'/'environment.json',
            lambda e: e.update(source_hashes={**SHARED_HASHES, 'experiments/make_revision/knn_controls.py': '1' * 64}))
    return SimpleNamespace(knn=root/'knn', training=root/'training', knn_rows=knn_rows, training_rows=training_rows)


@pytest.fixture(scope='session')
def training_baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp('training_baseline')
    runs = build_training_pair(root)
    return SimpleNamespace(runs=runs, result=cr.compare_training(runs.training, runs.knn, root/'out'), out=root/'out')


@pytest.fixture
def training_runs(training_baseline, tmp_path):
    for label in ('knn', 'training'):
        shutil.copytree(getattr(training_baseline.runs, label), tmp_path/'runs'/label)
    return SimpleNamespace(knn=tmp_path/'runs'/'knn', training=tmp_path/'runs'/'training')


def test_training_family_is_trained_minus_each_control_seed_averaged_with_holm_across_every_member(training_baseline):
    runs, contrasts = training_baseline.runs, training_baseline.result['contrasts']
    assert [(r['family_index'], r['dataset'], r['model_b']) for r in contrasts] == [
        (1, 'alpha', CONTROLS[0]), (2, 'alpha', CONTROLS[1]), (3, 'beta', CONTROLS[0]), (4, 'beta', CONTROLS[1])]
    assert all(r['model_a'] == 'arrowflow_full_knn' and (r['run_a'], r['run_b']) == ('knn', 'training')
               and (r['n_folds'], r['df']) == (15, 14) for r in contrasts)
    for row in contrasts:
        trained = fold_means(runs.knn_rows[row['dataset']], 'arrowflow_full_knn', 'accuracy')
        control = fold_means(runs.training_rows[row['dataset']], row['model_b'], 'accuracy')
        differences = np.array([np.mean(trained[key]) - np.mean(control[key]) for key in sorted(trained)])
        se = np.sqrt((1 / 15 + .25) * np.var(differences, ddof=1))
        assert np.isclose(row['mean_difference'], differences.mean(), rtol=0, atol=1e-12)
        assert np.isclose(row['standard_error'], se, rtol=0, atol=1e-12)
        assert np.isclose(row['p_approximate'], 2 * stats.t.sf(abs(differences.mean() / se), 14), rtol=0, atol=1e-12)
    assert [r['holm_p_approximate'] for r in contrasts] == holm_adjust([r['p_approximate'] for r in contrasts])
    assert contrasts[0]['mean_difference'] > 0                  # the fixture's untrained control errs more on alpha
    text = (training_baseline.out/'training_contrasts.csv').read_text()
    assert text.splitlines()[0].split(',') == list(cr.TRAINING_CONTRAST_COLUMNS) and len(text.splitlines()) == 5
    summary = json.loads((training_baseline.out/'training_contrasts.json').read_text())
    assert summary['outputs']['training_contrasts.csv']['sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert summary['family']['size'] == 4 and summary['contrasts_declared'] == kc.PRIMARY_CONTRASTS
    assert summary['provenance']['verification']['training']['jobs_verified'] == 2 * 2 * 15
    assert summary['provenance']['pairing']['shared_sources'] == SHARED_HASHES
    assert set(summary['provenance']['analysis_sources']) == {f'experiments/make_revision/{name}'
                                                             for name in (*ANALYSIS_SOURCES, 'knn_controls.py')}


def test_training_error_table_holds_the_three_models_from_their_verified_summaries(training_baseline):
    table, runs = training_baseline.result['error_table'], training_baseline.runs
    assert table['models'] == ['arrowflow_full_knn', *CONTROLS]
    for name in DATASETS:
        rows = table['rows'][name]
        assert [(r['model_id'], r['source_run']) for r in rows] == [('arrowflow_full_knn', 'knn'), (CONTROLS[0], 'training'),
                                                                     (CONTROLS[1], 'training')]
        for row in rows:
            source = runs.knn_rows if row['source_run'] == 'knn' else runs.training_rows
            expected = summarize_outer(source[name], row['model_id'], 'error', expected_folds=FOLDS, expected_seeds=SEEDS)
            assert np.isclose(row['mean_error'], expected['mean'], rtol=0, atol=1e-12)
            assert (row['n_folds'], row['seeds_per_fold']) == (15, 3)
    assert json.loads((training_baseline.out/'training_error_table.json').read_text()) == table


def test_training_depth_split_follows_the_widths_arrowflow_knn_selected_in_each_outer_fold(training_baseline):
    depth, runs = training_baseline.result['depth_split'], training_baseline.runs
    assert depth['depths'] == [[128], [64, 128]] and 'outside the Holm family' in depth['status']
    for name in DATASETS:
        shallow, deep = depth['by_dataset'][name]
        assert deep['folds'] == [list(f) for f in DEEP_FOLDS] and shallow['folds'] == [list(f) for f in FOLDS if f not in DEEP_FOLDS]
        trained = fold_means(runs.knn_rows[name], 'arrowflow_full_knn', 'accuracy')
        untrained = fold_means(runs.training_rows[name], CONTROLS[0], 'accuracy')
        for entry in (shallow, deep):
            expected = [np.mean(trained[tuple(f)]) - np.mean(untrained[tuple(f)]) for f in entry['folds']]
            assert np.allclose(entry['fold_differences'], expected, rtol=0, atol=1e-12)
            assert 'p_approximate' not in entry['interval'] and entry['interval']['n_folds'] == entry['n_folds']
        assert (shallow['untrained_selected_the_same_widths'], deep['untrained_selected_the_same_widths']) == (6, 0)
    assert [(p['widths'], p['n_dataset_folds']) for p in depth['pooled']] == [([128], 12), ([64, 128], 18)]
    assert json.loads((training_baseline.out/'training_depth_split.json').read_text()) == depth


TRAINING_REFUSALS = {
    'reference_pin': (lambda r: rewrite(r.training/'protocol.json', lambda p: p['training_controls']['reference'].update(
        summary_sha256='0' * 64)), 'reference block does not match the knn run: summary_sha256'),
    'family_size': (lambda r: rewrite(r.training/'protocol.json', lambda p: p.update(primary_family_size=2)), 'primary_family_size'),
    'contrast_order': (lambda r: rewrite(r.training/'protocol.json', lambda p: p.update(primary_contrasts=p['primary_contrasts'][::-1])),
                       'primary_contrasts must be'),
    'split_seed': (lambda r: rewrite(r.training/'protocol.json', lambda p: p.update(split_seed=1)), 'Protocol split_seed differs'),
    'candidates': (lambda r: rewrite(r.training/'candidates.json', lambda c: c[CONTROLS[1]].update(
        candidates=c[CONTROLS[1]]['candidates'][:1], config_ids=c[CONTROLS[1]]['config_ids'][:1])), 'not the projection'),
    'dataset_hash': (lambda r: rewrite(r.training/'beta'/'manifest.json', lambda m: m.update(dataset_hash='0' * 64)),
                     'dataset hash of beta differs'),
    'shared_source_differs': (lambda r: rewrite(r.training/'environment.json', lambda e: e['source_hashes'].update(
        {'experiments/make_revision/models.py': '2' * 64})), 'Sources sealed by both runs differ: experiments/make_revision/models.py'),
    'shared_source_absent': (lambda r: rewrite(r.knn/'environment.json', lambda e: e['source_hashes'].pop('arrowflow/arrowflow.py')),
                             'must seal arrowflow/arrowflow.py'),
    'depths': (lambda r: rewrite(r.training/'protocol.json', lambda p: p['training_controls']['depth_split'].update(depths=[[128], [128]])),
               'depths'),
    'incomplete': (lambda r: (r.training/'summary.json').unlink(), 'incomplete or unverified'),
    'tampered_prediction': (lambda r: rewrite_result(r.training, 0, lambda result: result['predictions'][0].update(
        y_pred=(result['predictions'][0]['y_pred'] + 1) % 2)), 'fails reporting.validate_result_records'),
}


@pytest.mark.parametrize('change', sorted(TRAINING_REFUSALS))
def test_training_refuses_unpinned_unpaired_or_unverified_runs_before_writing_anything(training_runs, tmp_path, change):
    alter, message = TRAINING_REFUSALS[change]
    alter(training_runs)
    with pytest.raises(cr.RunComparisonError) as refused:
        cr.compare_training(training_runs.training, training_runs.knn, tmp_path/'out')
    assert message in str(refused.value), str(refused.value)
    assert not (tmp_path/'out').exists()


def test_training_commands_end_to_end_with_the_prepared_pairing_check(training_baseline, training_runs, tmp_path, capsys):
    runs = training_baseline.runs
    arguments = ['training', '--training-source', str(runs.training), '--knn-source', str(runs.knn), '--output']
    completed = subprocess.run([sys.executable, '-m', 'experiments.make_revision.compare_runs', *arguments, str(tmp_path/'cli')],
                               cwd=REPO, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in (tmp_path/'cli').iterdir()) == sorted(TRAINING_OUTPUTS)
    assert 'alpha: arrowflow_full_knn - arrowflow_knn_untrained accuracy' in completed.stdout and 'widths [64, 128]' in completed.stdout
    assert all((tmp_path/'cli'/name).read_bytes() == (training_baseline.out/name).read_bytes() for name in TRAINING_OUTPUTS)
    cr.main([*arguments, str(tmp_path/'cli')])                      # the same evidence again: identical files, no error
    (tmp_path/'cli'/'training_depth_split.json').write_text('{}\n')
    with pytest.raises(SystemExit) as refused:
        cr.main([*arguments, str(tmp_path/'cli')])
    assert refused.value.code == 2 and 'overwrite' in capsys.readouterr().err
    prepared = tmp_path/'prepared'                                  # a knn_training directory whose jobs have not run
    for name in ('protocol.json', 'environment.json', 'candidates.json'):
        (prepared/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runs.training/name, prepared/name)
    for name in DATASETS:
        (prepared/name).mkdir()
        shutil.copyfile(runs.training/name/'manifest.json', prepared/name/'manifest.json')
    cr.main(['training-pairing', '--training-source', str(prepared), '--knn-source', str(runs.knn)])
    assert 'paired with arrowflow-v3-bridge-knn-1' in capsys.readouterr().out
    rewrite(prepared/'protocol.json', lambda p: p.update(primary_family_size=3))
    with pytest.raises(SystemExit) as refused:
        cr.main(['training-pairing', '--training-source', str(prepared), '--knn-source', str(runs.knn)])
    assert refused.value.code == 2 and 'primary_family_size' in capsys.readouterr().err


REAL_KNN = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'/'2026-09-12-bridge-knn'


@pytest.mark.skipif(not REAL_KNN.is_dir(), reason='the bridge_knn run is not on this machine')
def test_real_knn_run_is_the_training_reference_and_its_shared_sources_are_unchanged_in_this_tree():
    knn = cr.load_run(REAL_KNN, 'knn')
    record = cr.check_training_reference(json.loads((PROTOCOLS/'knn_training.json').read_text()), knn)
    assert record['summary_sha256'] == sha256(REAL_KNN/'summary.json') and record['model_id'] == 'arrowflow_full_knn'
    for model in CONTROLS:
        assert kc.project_candidates(knn.candidates['arrowflow_full_knn']['candidates'], kc.CANDIDATE_KEYS[model]) == kc.control_candidates(model)
    for source in cr.SHARED_SOURCES:
        assert knn.environment['source_hashes'][source] == sha256(REPO/source), source
