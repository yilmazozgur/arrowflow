"""G4 follow-up experiments (a) and (c): the ten newdata models on Wine quality and Segment without exact duplicate rows
(family dedup) and on the artificial ranking-pattern datasets (family artificial).

Design (progress.md "RULING (G4 follow-up, 2026-09-14)", items 3 and 5): the nested design of newdata_batch1.json (5 x 3
outer folds, 3 inner folds, split seed 27183, fitting seeds 8129, 19391 and 39019, the candidate budget, seed and tie rule,
the ten models of newdata.newdata_registry with their candidates, three stochastic finalists, internal validation ratio 0.1,
the adaptive encoding and augmentation rules, report metrics, failure policy and freeze requirement), copied unchanged and
recorded in design_source with the template sha256; the component ablation design of newdata_ablation.json in the ablation
block; and the prespecified analysis in the analysis block, written before any outer score exists. One frozen protocol per
family: protocols/2026-09-14/dedup.json (arrowflow-v3-dedup-1; cap 9 h at 16 workers) and artificial.json
(arrowflow-v3-artificial-1; cap 4 h at 8 workers). Every dataset is loaded through its pinned extra_data loader.

python -m experiments.make_revision.extra_runs draft --family F --output P
    the unfrozen stage protocol of family F
python -m experiments.make_revision.extra_runs prepare --protocol P --output O
    run_revision's prepared layout (protocol, candidates, environment, manifest, splits, data) for the pinned datasets
python -m experiments.make_revision.extra_runs smoke --family F --output O [--workers 3]
    synthetic, never evidence: a synthetic family run through run_revision's worker and reporting, its component ablation
    (extra_ablation prepare, run and summary) and compare_extra analyse
python -m experiments.make_revision.extra_runs pilot --protocol P --output O
    training-only runtime on the first outer training partition of every dataset: three evenly spaced candidates of every
    model (as run_revision.runtime_pilot) and every ablation variant at the middle ArrowFlow-kNN candidate; no score
python -m experiments.make_revision.extra_runs project --protocol P --pilot O/pilot.json --output PROJECTION.json
    the calibrated projection of the run and its ablation at the family's workers, against the cap
python -m experiments.make_revision.extra_runs freeze --draft P --projection PROJECTION.json --stages STAGES.json [--output F]
    the frozen family protocol, only if the calibrated projection is within the cap
python -m experiments.make_revision.extra_runs run --protocol F --output O [--workers N]
    run_revision's run stage for the frozen family protocol (the same checks, worker, lock and verification)
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'  # as run_revision: spawned workers import this -m module before any numeric library
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import extra_data as ed
from . import newdata as nd
from .evaluation import ModelSpec, canonical_json, config_id, dataset_fingerprint, make_splits
from .extra_data import DatasetIdentityError
from .holistic import COMPONENT_VARIANTS, MAIN_MODELS, RULES, RUN_LABELS, SOURCES as HOLISTIC_SOURCES
from .knn_controls import CANDIDATE_KEYS as CONTROL_CANDIDATE_KEYS, INPUT_MODEL, TRAINED_MODEL, UNTRAINED_MODEL, project_candidates
from .projected_knn import PROJECTED_MODEL

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core: every module newdata
# sealed for the ten models, newdata itself (the registry and the design) and the pinned loaders.
SOURCE_MODULES = nd.SOURCE_MODULES + ['experiments.make_revision.newdata', 'experiments.make_revision.extra_data']

FAMILIES = ('dedup', 'artificial')
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-14'
PROTOCOL_FILES = {family: PROTOCOLS/f'{family}.json' for family in FAMILIES}
PROTOCOL_IDS = {'dedup': 'arrowflow-v3-dedup-1', 'artificial': 'arrowflow-v3-artificial-1'}
TEMPLATE = nd.PROTOCOLS/'newdata_batch1.json'
ABLATION_TEMPLATE = nd.PROTOCOLS/'newdata_ablation.json'
REGISTRY = 'experiments.make_revision.extra_runs:extra_registry'
SMOKE_REGISTRY = 'experiments.make_revision.extra_runs:smoke_extra_registry'
CAP_HOURS = {'dedup': 9, 'artificial': 4}
WORKERS = {'dedup': 16, 'artificial': 8}
MAX_WORKERS = 16
ROLES = ('family', 'descriptive')
TEMPLATE_KEYS = (*nd.DESIGN_COPY_KEYS, 'model_order', 'models', 'primary_contrasts', 'secondary_contrasts')
ABLATION_KEYS = ('aggregation', 'depth_split', 'dropped_variants', 'failure_policy', 'fit_reuse', 'knn_readout', 'max_workers',
                 'parallelism', 'reporting', 'variant_definitions', 'variants')
ABLATION_SHARED_KEYS = ('confidence', 'fit_seeds', 'historical_results', 'inner_folds', 'numeric_threads_per_worker', 'outer_folds',
                        'outer_repeats', 'report_metrics', 'selection_metric', 'split_seed', 'test_train_ratio')
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'projection')
SMOKE_FIELDS = (*FREEZE_FIELDS, 'protocol_id', 'purpose', 'registry', 'datasets', 'panel', 'primary_family_size',
                'secondary_family_size', 'multiplicity', 'analysis', 'outer_folds', 'outer_repeats', 'inner_folds')
DRAFT_STATUS = 'drafted_awaiting_prepare_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_confirmatory_scoring'
PRIMARY_CONTRAST, SECONDARY_CONTRAST = nd.PRIMARY_CONTRAST, nd.SECONDARY_CONTRAST
DESCRIPTIVE = 'descriptive; no multiplicity adjustment'
KINDS = {'central': 'pooled', 'upper': 'max'}
STAGE_OVERHEAD_HOURS = .25
ITEMS = {'dedup': 'experiment (a)', 'artificial': 'experiment (c)'}
SELECTION_RULINGS = {
    'dedup': 'progress.md "RULING (G4 follow-up, 2026-09-14)" item (3), recorded before any fit on these datasets: Wine quality '
             'and Segment rerun without exact duplicate rows (the first occurrence kept in loaded order; no group carries two '
             'labels) instead of duplicate-grouped folds; the ten newdata_registry models under the newdata batch design; the '
             'component ablation at the reconstructed selections; the training effect over the untrained ArrowFlow and over '
             'input kNN, Holm over the two datasets each; the comparison with the registered full-data results descriptive '
             'and unpaired; a sensitivity analysis beside the unchanged registered seventeen-dataset analysis',
    'artificial': 'progress.md "RULING (G4 follow-up, 2026-09-14)" item (5): ranks8 and ranks16 (45 rows per class) and '
                  'ranks8_original (the 77 original rows, descriptive only) from artificial_ranks.py; the design and models of '
                  'experiment (a) plus the component ablation; families over ranks8 and ranks16; not pooled into the '
                  'seventeen-dataset Friedman analysis; pinned and frozen only after the data module is committed and the '
                  'controller has approved the data (artificial-data-approved.md)'}
NOTES = {
    'dedup': {
        'sensitivity': 'a sensitivity analysis of the registered seventeen-dataset analysis, which stays unchanged',
        'why_duplicate_free': 'in wine_quality 23% of outer test rows (segment 15%) have an exact copy in the training partition '
                              '(runs/2026-09-13-referee-analyses/duplicates). Grouped folds were rejected: inside each fit the '
                              'readout selection of ArrowFlow-kNN, of both training controls and of projected kNN uses '
                              'stratified splits of training rows that would still hold duplicate pairs, while numeric kNN '
                              'selects through the harness inner folds; removing every exact duplicate row reaches every split '
                              'and every selection mechanism alike',
        'low_power': 'each family holds two datasets, so Holm adjusts over two'},
    'artificial': {
        'descriptive_only': 'ranks8_original (77 rows) is reported descriptively and belongs to no family',
        'not_pooled': 'the artificial datasets are not pooled into the seventeen-dataset Friedman analysis',
        'features': 'each item\'s relative position, NaN for a deleted item; every model imputes missing cells fold-locally',
        'low_power': 'each family holds two datasets, so Holm adjusts over two'},
}


def _plain(value):
    """The JSON form of a declaration (tuples become lists), so it compares equal to a protocol read from disk."""
    return json.loads(canonical_json(value))


# ----------------------------------------------------------------------------- panels and declarations

def panel_declaration(family):
    if family == 'dedup':
        return _plain(list(ed.DEDUP_PINS))
    pins = ed.read_artificial_pins(ed.ARTIFICIAL_PINS_FILE)
    return _plain([ed.artificial_entry(name, None if pins is None else pins[name]) for name in ed.ARTIFICIAL_DATASETS])


def panel_by_name(protocol):
    return {entry['name']: entry for entry in protocol['panel']}


def family_members(panel):
    return [entry['name'] for entry in panel if entry['role'] == 'family']


def multiplicity(size):
    return f'Holm_across_the_{size}_family_datasets_within_each_family; the primary and secondary families are adjusted separately'


def expected_duplicates(family, entry):
    return dict.fromkeys(ed.IDENTITY_COUNTS, 0) if family == 'dedup' else entry.get('duplicates')


def analysis_declaration(family, panel):
    names, members = [entry['name'] for entry in panel], family_members(panel)
    others = [name for name in names if name not in members]
    families = {}
    for label, contrast, model in (('primary', PRIMARY_CONTRAST, UNTRAINED_MODEL), ('secondary', SECONDARY_CONTRAST, INPUT_MODEL)):
        families[label] = {
            'contrast': contrast, 'model_a': TRAINED_MODEL, 'model_b': model, 'size': len(members), 'datasets': members,
            'definition': f'per family dataset, {TRAINED_MODEL} minus {model} accuracy' + (' (the training effect)' if model == UNTRAINED_MODEL else ''),
            'interval': nd.INTERVAL_RULE, 'alpha': .05,
            'multiplicity': f'Holm across the {len(members)} family datasets (evaluation.holm_adjust)'
                            + ('' if label == 'primary' else ', adjusted separately from the primary family')}
    block = {
        'status': 'prespecified before any outer score on these datasets exists; computed only by compare_extra analyse, which '
                  'refuses (exit 2, nothing written) until the run and its component ablation are complete, then re-verifies '
                  'both before any score is read',
        'command': f'python -m experiments.make_revision.compare_extra analyse --family {family} --run <run> --ablation <ablation> '
                   '--output <directory>' + (' --runs <directory holding the registered runs>' if family == 'dedup' else ''),
        'requires_run_and_ablation_complete': True, 'metric': 'accuracy', 'datasets': names, 'family_datasets': members,
        'descriptive_only_datasets': others, 'primary_family': families['primary'], 'secondary_family': families['secondary'],
        'main_table': {'models': list(MAIN_MODELS), 'status': 'descriptive',
                       'definition': 'per dataset, mean outer error (the mean over outer folds of the fitting-seed-averaged '
                                     'error), outer-fold SD (ddof 1) and mean within-fold seed SD of ArrowFlow-kNN, the five '
                                     'tuned classical models and the majority class, from the verified summary.json'},
        'comparator_intervals': {'models': list(nd.COMPARATORS), 'status': DESCRIPTIVE,
                                 'definition': f'per dataset and comparator, {TRAINED_MODEL} minus the comparator accuracy: '
                                               f'{nd.INTERVAL_RULE}, unadjusted (compare_newdata.comparator_rows); the comparator '
                                               'with the lowest mean outer error is flagged'},
        'competitiveness': {'status': 'descriptive; fixed rules', 'rules': RULES,
                            'definition': 'holistic.competitiveness on the mean outer errors of ArrowFlow-kNN, the five tuned '
                                          'classical models and the majority class: the best tuned model, the gap, within three '
                                          'points, best-on, ceiling, near-majority and no-learning'},
        'ladder': [{'rung': rung, 'model_id': model} for rung, model in nd.LADDER],
        'ladder_table': 'per dataset, mean outer error, outer-fold SD (ddof 1) and mean within-fold seed SD of every rung '
                        '(compare_newdata.ladder_rows)',
        'complete_metrics': {'models': list(nd.MODEL_ORDER), 'metrics': ['error', 'balanced_accuracy', 'macro_f1', 'accuracy'],
                             'definition': 'per dataset and model, each metric\'s mean, outer-fold SD and within-fold seed SD '
                                           '(holistic.complete_metric_rows)'},
        'selected_widths': {'status': 'descriptive',
                            'definition': f'the hidden widths of the configuration {TRAINED_MODEL} selected on the inner folds '
                                          'of each outer fold (knn_controls.selected_widths on the verified outer model rows)'},
        'duplicate_audit': {'status': 'descriptive',
                            'definition': 'exact duplicate feature rows of the prepared data (extra_data.duplicate_groups: float64 '
                                          'equality of every feature, NaN equal to NaN; equal to referee_analyses.duplicate_groups '
                                          'on data without NaN): rows, distinct rows, duplicate rows, duplicate groups, '
                                          'label-conflicting groups and the largest group; per outer fold the test rows with an '
                                          'exact duplicate in the training partition',
                            'expected': {entry['name']: expected_duplicates(family, entry) for entry in panel}},
        'components': {'variants': list(COMPONENT_VARIANTS), 'status': 'descriptive; no p values; no multiplicity adjustment',
                       'definition': 'per dataset and variant, ArrowFlow (views7, which reproduces the run\'s outer predictions '
                                     'exactly) minus the variant, accuracy, in points beside the fraction: the ablation '
                                     'summary\'s change_from_views7 accuracy interval negated with its bounds swapped '
                                     '(holistic.component_rows), with the outer folds in which the variant is identical to '
                                     'views7 by construction'},
        'notes': NOTES[family]}
    if family == 'dedup':
        block['full_data_comparison'] = {
            'status': 'unpaired; descriptive; no interval and no test', 'models': list(nd.MODEL_ORDER),
            'sources': {entry['name']: entry['source'] for entry in panel},
            'registered_runs': {label: HOLISTIC_SOURCES[label] for label in RUN_LABELS},
            'definition': 'per deduplicated dataset and model, its mean outer error beside the mean outer error of the same model '
                          'on the registered full source dataset, loaded as holistic.py loads it (holistic.load_panel: every '
                          'registered run re-verified with compare_runs.load_run and verify_run and the registered pairings '
                          're-checked; holistic.Panel.summary); the difference deduplicated minus full in points; different rows '
                          'and different splits, so neither an interval nor a test'}
    return _plain(block)


def ablation_declaration(family, template):
    block = {key: template[key] for key in ABLATION_KEYS}
    summary_json, summary_csv = f'{family}_ablation_summary.json', f'{family}_ablation_summary.csv'
    block['knn_readout'] = {**template['knn_readout'],
                            'source': f'identical to arrowflow_full_knn ({family}.json models.arrowflow_full_knn.readout)'}
    block['reporting'] = {**template['reporting'], 'outputs': f'{summary_json} (metrics, intervals, depth split, reproduction '
                                                              f'counts, model rows) and {summary_csv} (metrics per dataset, '
                                                              'variant and metric)'}
    block.update(
        reference=f'the {family} run of this protocol (arrowflow_full_knn): its protocol.json must be byte-identical to the committed '
                  'frozen protocol and its summary.json complete; the run\'s protocol sha256, summary sha256 and code revision are '
                  'sealed in the ablation manifest at prepare and re-checked at summary',
        reproduction='views7 must reproduce the outer predictions of the run exactly for every dataset, outer fold and fitting seed '
                     '(the same labels in the sealed test-sample order); checked inside every job, again for every record by the '
                     'summary, and every sealed selection is re-derived from the run at summary time',
        selected_configuration='results/<dataset>__arrowflow_full_knn__r<repeat>f<fold>.json selection block of the run, '
                               'reconstructed from the complete inner fit history at prepare time (reporting.validate_result_records) '
                               'and resolved to embed_dim, degree and augment from the outer training partition shape '
                               '(bridge.resolve_selected); sealed in reference_selected_configurations.csv and reference_selections.json '
                               'with the reference per-example predictions',
        dataset_loading='the prepared data are copied from the run; in production every dataset is reloaded through its pinned '
                        'extra_data loader and must equal the run\'s prepared features (NaN equal to NaN) and labels, with the '
                        'pinned dataset and splits hashes',
        shared_design='confidence, fit_seeds, historical_results, inner_folds, numeric_threads_per_worker, outer_folds, '
                      'outer_repeats, report_metrics, selection_metric, split_seed and test_train_ratio are the protocol\'s top-level '
                      'values (equal in both templates)',
        commands='python -m experiments.make_revision.extra_ablation prepare|ablation --protocol <frozen protocol> --reference <run> '
                 '--output <ablation>; python -m experiments.make_revision.extra_ablation summary --output <ablation>')
    return block


def design_source(family):
    return {'template': 'protocols/2026-09-12/newdata_batch1.json (sha256 in source_template_sha256)',
            'identical_to_template': ', '.join(TEMPLATE_KEYS),
            'ablation_template': 'protocols/2026-09-12/newdata_ablation.json (sha256 in ablation_template_sha256)',
            'ablation_identical_to_template': ', '.join(key for key in ABLATION_KEYS if key not in ('knn_readout', 'reporting')),
            'ablation_changed': 'knn_readout.source, reporting.outputs',
            'ablation_shared_with_the_run_design': ', '.join(ABLATION_SHARED_KEYS),
            'changed': 'protocol_id, production_family, registry, datasets, panel, primary_family_size, secondary_family_size, '
                       'multiplicity, analysis, selection_ruling, wallclock_cap_hours, frozen, status, resource_decision',
            'removed_from_template': {'batch, batches, batch_projection': 'one run per family',
                                      'model_template_sha256': 'the model declarations are copied from the template itself',
                                      'frozen_at_utc': 'set at the freeze'},
            'added': 'ablation, workers, projection, ablation_template_sha256',
            'family': family}


def draft_protocol(family):
    """The unfrozen stage protocol of one family: newdata_batch1.json's nested design and models, newdata_ablation.json's
    component ablation design, the family panel with its pins and the prespecified analysis."""
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {", ".join(FAMILIES)}')
    template, ablation_template = json.loads(TEMPLATE.read_text()), json.loads(ABLATION_TEMPLATE.read_text())
    protocol = {key: template[key] for key in TEMPLATE_KEYS}
    if protocol['models'] != nd.models_declaration() or protocol['model_order'] != list(nd.MODEL_ORDER):
        raise ValueError('newdata_batch1.json no longer declares the ten newdata_registry models')
    if [key for key, value in nd.DESIGN.items() if protocol[key] != value]:
        raise ValueError('newdata_batch1.json no longer holds the newdata nested design')
    unequal = [key for key in ABLATION_SHARED_KEYS if canonical_json(ablation_template[key]) != canonical_json(template[key])]
    if unequal:
        raise ValueError(f'The run and ablation templates disagree on {", ".join(unequal)}')
    panel = panel_declaration(family)
    members = family_members(panel)
    protocol.update(
        protocol_id=PROTOCOL_IDS[family], production_family=family, registry=REGISTRY, datasets=[entry['name'] for entry in panel],
        panel=panel, primary_family_size=len(members), secondary_family_size=len(members), multiplicity=multiplicity(len(members)),
        analysis=analysis_declaration(family, panel), ablation=ablation_declaration(family, ablation_template),
        wallclock_cap_hours=CAP_HOURS[family], workers=WORKERS[family], design_source=design_source(family),
        source_template_sha256=nd.sha256_file(TEMPLATE), ablation_template_sha256=nd.sha256_file(ABLATION_TEMPLATE),
        selection_ruling=SELECTION_RULINGS[family], frozen=False, status=DRAFT_STATUS,
        resource_decision='pending: prepare, the synthetic smoke (run, reporting, component ablation and analysis) and the '
                          'training-only pilot of every dataset, then the calibrated projection against the cap',
        projection=None)
    return _plain(protocol)


def _validate_smoke(p, family):
    panel = p.get('panel')
    if (not isinstance(panel, list) or not panel or any(not isinstance(entry, dict) for entry in panel)
            or len({entry.get('name') for entry in panel}) != len(panel) or any(entry.get('role') not in ROLES for entry in panel)
            or not family_members(panel) or (family == 'dedup' and any(not entry.get('source') for entry in panel))):
        raise ValueError('A synthetic smoke panel holds uniquely named datasets with the role family or descriptive, at least one '
                         'family member, and a source for every dedup entry')
    members = family_members(panel)
    if (p.get('registry') != SMOKE_REGISTRY or p.get('datasets') != [entry['name'] for entry in panel]
            or p.get('primary_family_size') != len(members) or p.get('secondary_family_size') != len(members)
            or p.get('multiplicity') != multiplicity(len(members)) or p.get('analysis') != analysis_declaration(family, panel)
            or p.get('frozen') is not True or not str(p.get('protocol_id', '')).endswith('-synthetic-smoke')
            or any(type(p.get(key)) is not int or p[key] < 2 for key in ('outer_folds', 'inner_folds'))
            or type(p.get('outer_repeats')) is not int or p['outer_repeats'] < 1):
        raise ValueError('A synthetic smoke protocol is frozen, names the smoke registry and declares its datasets, family sizes, '
                         'multiplicity and analysis from its panel')
    return p


def validate_extra_protocol(p):
    """A production protocol is the family draft, or the draft with exactly the freeze fields set by `freeze`; a synthetic
    smoke protocol is the draft with its synthetic panel, design and analysis."""
    family = p.get('production_family') if isinstance(p, dict) else None
    if family not in FAMILIES:
        raise ValueError(f'production_family must be one of {", ".join(FAMILIES)}')
    draft = draft_protocol(family)
    smoke = p.get('purpose') == 'synthetic_smoke_only'
    ignored = SMOKE_FIELDS if smoke else FREEZE_FIELDS
    differing = sorted(key for key in set(p) | set(draft)
                       if key not in ignored and canonical_json(p.get(key)) != canonical_json(draft.get(key)))
    if differing:
        raise ValueError(f'The protocol differs from extra_runs.draft_protocol({family!r}) in {", ".join(differing)}')
    if smoke:
        return _validate_smoke(p, family)
    if not p.get('frozen'):
        if canonical_json(p) != canonical_json(draft):
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('projection') or {}
    hours = projection.get('decision_hours')
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != CAP_HOURS[family] or projection.get('workers') != WORKERS[family]
            or isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= CAP_HOURS[family]):
        raise ValueError(f'A frozen protocol records its freeze and a calibrated projection within the {CAP_HOURS[family]} h cap '
                         f'at {WORKERS[family]} workers')
    if any(not entry.get('pinned', True) for entry in p['panel']):
        raise ValueError('A frozen protocol pins every dataset')
    return p


# ----------------------------------------------------------------------------- registries

def extra_registry(protocol):
    """The production registry of a family protocol: newdata.build_registry's ten models."""
    validate_extra_protocol(protocol)
    if protocol.get('registry') != REGISTRY:
        raise ValueError(f'extra_registry serves protocols declaring {REGISTRY}')
    return nd.build_registry(protocol)


def smoke_extra_registry(protocol):
    """Synthetic smoke only: newdata.smoke_newdata_registry's ten models (ArrowFlow-kNN at two one-iteration candidates, the
    controls at their projections, every comparator at its first two candidates) for a family smoke protocol."""
    if protocol.get('purpose') != 'synthetic_smoke_only':
        raise ValueError('smoke_extra_registry serves synthetic smoke protocols only')
    validate_extra_protocol(protocol)
    real = nd.build_registry(protocol)
    trained = nd.SMOKE_TRAINED_CANDIDATES
    registry = {TRAINED_MODEL: ModelSpec(TRAINED_MODEL, real[TRAINED_MODEL].factory, trained, True)}
    for model in (UNTRAINED_MODEL, INPUT_MODEL, PROJECTED_MODEL):
        keys = CONTROL_CANDIDATE_KEYS[UNTRAINED_MODEL if model == UNTRAINED_MODEL else INPUT_MODEL]
        registry[model] = ModelSpec(model, real[model].factory, project_candidates(trained, keys), True)
    for model in nd.MODEL_ORDER[4:]:
        registry[model] = ModelSpec(model, real[model].factory, real[model].candidates[:2], real[model].stochastic)
    return registry


# ----------------------------------------------------------------------------- loading, prepare and run

def load(name, protocol, *, loaders=None):
    """(X, y, manifest) of one production dataset through its pinned extra_data loader; `loaders` maps a family to the
    source loader handed to it (tests use stand-ins)."""
    family = protocol['production_family']
    entry = panel_by_name(protocol).get(name)
    if entry is None:
        raise DatasetIdentityError(f'{name} is not in the {family} panel')
    loader = (loaders or {}).get(family)
    if family == 'dedup':
        return ed.load_dedup(name, loader=loader)
    return ed.load_artificial_pinned(name, entry, loader=loader)


def prepare(output, protocol, names=None, *, loaders=None):
    """run_revision.prepare for the pinned datasets of a family protocol: the same records and layout, each dataset loaded
    through its pinned loader and its dataset and splits hashes checked against the panel."""
    from .run_revision import environment_record, get_registry, write_json
    validate_extra_protocol(protocol)
    if protocol.get('purpose') is not None:
        raise ValueError('prepare serves production protocols; the smoke writes its own synthetic datasets')
    output = Path(output)
    names = list(protocol['datasets'] if names is None else names)
    outside = [name for name in names if name not in protocol['datasets']]
    if outside:
        raise ValueError(f'Datasets outside the protocol datasets: {outside}')
    registry = get_registry(protocol['registry'], protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', nd.candidate_record(registry))
    write_json(output/'environment.json', environment_record(protocol['registry']))
    entries = panel_by_name(protocol)
    for name in names:
        X, y, manifest = load(name, protocol, loaders=loaders)
        splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
        manifest['splits_hash'] = config_id(splits)
        if (manifest['dataset_hash'], manifest['splits_hash']) != (entries[name].get('dataset_hash'), entries[name].get('splits_hash')):
            raise DatasetIdentityError(f'{name}: the dataset or splits hash differs from the panel pins')
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        destination = output/name/'data.npz'
        if not destination.exists():
            np.savez_compressed(destination, X=X, y=y)
    return registry


def run(protocol_path, output, workers):
    """run_revision's run stage for a frozen family protocol, with the same checks in the same order."""
    from .reporting import collect_verified_results
    from .run_revision import _worker, environment_record, execution_lock, get_registry, load_prepared, planned_jobs, write_json
    output = Path(output)
    protocol = json.loads(Path(protocol_path).read_text())
    if not protocol.get('frozen'):
        raise ValueError('Confirmatory run requires a reviewed frozen protocol')
    if protocol.get('purpose') is not None:
        raise ValueError('A production run needs the frozen production protocol, not a synthetic smoke protocol')
    validate_extra_protocol(protocol)
    saved = json.loads((output/'protocol.json').read_text())
    if saved != protocol:
        raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
    registry_path = protocol['registry']
    registry = get_registry(registry_path, protocol)
    if canonical_json(json.loads((output/'candidates.json').read_text())) != canonical_json(nd.candidate_record(registry)):
        raise ValueError('Candidate registry changed after prepare')
    if json.loads((output/'environment.json').read_text()) != environment_record(registry_path):
        raise ValueError('Code revision, source, environment, or registry changed after prepare')
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError('Worker count must be between 1 and 16')
    names, entries = list(protocol['datasets']), panel_by_name(protocol)
    for name in names:
        manifest = load_prepared(output, name)[2]
        if (manifest['dataset_hash'], manifest['splits_hash']) != (entries[name]['dataset_hash'], entries[name]['splits_hash']):
            raise DatasetIdentityError(f'{name}: the prepared dataset or splits hash differs from the panel pins')
    write_json(output/'planned_jobs.json', planned_jobs(names, protocol, registry))
    jobs = [(str(output), name, index, model, registry_path) for name in names
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in pool.map(_worker, jobs):
            print(path, flush=True)
    collect_verified_results(output, names, protocol, registry)


# ----------------------------------------------------------------------------- training-only pilot, projection and freeze

def ablation_pilot(name, X, y, train, spec, protocol):
    """Training-only timing of every component-ablation variant at the middle ArrowFlow-kNN candidate (index len // 2, fixed
    before any score), resolved on the first outer training partition, one fitting seed; predictions on every fourth
    training row only; no score. The fits are run_knn_ablation's (fit_views7, the separate variants and the controls)."""
    from . import run_knn_ablation as base
    from .bridge import resolve_selected
    from .models import seed_fit
    from .multiview import MultiViewArrowFlowKNN
    config = spec.candidates[len(spec.candidates) // 2]
    seed, query = protocol['fit_seeds'][0], train[::4]
    record = {'dataset_id': name, 'config_id': config_id(config), 'config': config, 'model_seed': seed, 'train_rows': len(train),
              'query_rows': len(query)}
    try:
        selected = resolve_selected(config, X.shape[1], len(train))
        variants = base.knn_ablation_variants(selected)
        params, sources = dict(variants), base.fit_sources(variants)
        seconds, network_fits = {}, {}
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            _, _, knn_views, output_views = base.fit_views7(params['views7'], seed, X[train], y[train], X[query])
            base.derived_predictions(knn_views, output_views)
            seconds['views7'], network_fits['views7'] = time.perf_counter() - start, 7
            for variant in base.SEPARATE:
                if sources[variant] == 'identical_to_views7':
                    seconds[variant], network_fits[variant] = 0., 0
                    continue
                seed_fit(seed)
                start = time.perf_counter()
                MultiViewArrowFlowKNN(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, 7
            for variant, estimator in base.CONTROLS.items():
                seed_fit(seed)
                start = time.perf_counter()
                estimator(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, 0
        per_seed = sum(seconds.values())
        record.update(selected=selected, fit_sources=sources, seconds_by_variant=seconds, network_fits_by_variant=network_fits,
                      seconds_per_seed=per_seed, seconds_per_job_estimate=per_seed * len(protocol['fit_seeds']),
                      all_variants_to_views7_ratio=per_seed / seconds['views7'], status='ok')
    except Exception as exc:
        record.update(status='failed', exception=f'{type(exc).__name__}: {exc}')
    return record


def runtime_pilot(output, protocol, names=None, *, loaders=None):
    """run_revision.runtime_pilot for the pinned datasets of a family (three evenly spaced candidates of every model on the
    first outer training partition, predictions on training rows only; the same rows and estimates), plus the ablation
    variant timing of ablation_pilot; pilot.json records the protocol hash."""
    from .evaluation import _fit_predict
    from .run_revision import execution_lock, load_prepared, write_json
    output = Path(output)
    names = list(protocol['datasets'] if names is None else names)
    with execution_lock():
        registry = prepare(output, protocol, names, loaders=loaders)
        rows, ablation_rows, identity = [], [], {}
        for name in names:
            X, y, manifest, splits = load_prepared(output, name)
            identity[name] = {'dataset_hash': manifest['dataset_hash'], 'splits_hash': manifest['splits_hash']}
            train = splits[0]['train']  # never pass any outer-test sample or label
            for model, spec in registry.items():
                for index in sorted({0, len(spec.candidates) // 2, len(spec.candidates) - 1}):
                    config = spec.candidates[index]
                    start = time.perf_counter()
                    row = {'dataset_id': name, 'model_id': model, 'config': config, 'config_id': config_id(config),
                           'fit_rows': train, 'model_seed': protocol['fit_seeds'][0]}
                    try:
                        _, timing = _fit_predict(spec, config, protocol['fit_seeds'][0], X[train], y[train], X[train])
                        row.update(timing, status='ok', elapsed_seconds=time.perf_counter() - start)
                    except Exception as exc:
                        row.update(status='failed', exception=f'{type(exc).__name__}: {exc}',
                                   elapsed_seconds=time.perf_counter() - start)
                    rows.append(row)
            ablation_rows.append(ablation_pilot(name, X, y, train, registry[TRAINED_MODEL], protocol))
    estimates = {}
    for model, spec in registry.items():
        durations = [r['elapsed_seconds'] for r in rows if r['model_id'] == model and r['status'] == 'ok']
        per_outer = nd.fits_per_outer(len(spec.candidates), spec.stochastic, protocol['inner_folds'])
        total = per_outer * protocol['outer_folds'] * protocol['outer_repeats'] * len(protocol['datasets'])
        estimates[model] = {'fits_per_outer': per_outer, 'panel_fit_count': total,
                            'observed_seconds_min': min(durations) if durations else None,
                            'observed_seconds_max': max(durations) if durations else None,
                            'serial_panel_seconds_using_observed_max': total * max(durations) if durations else None}
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'family': protocol['production_family'], 'rows': rows,
              'ablation_rows': ablation_rows, 'workload_estimates': estimates,
              'estimate_limitations': 'Sampled configurations/datasets; max extrapolation is not a runtime bound; RSS is process '
                                      'lifetime high-water mark.',
              'candidate_reduction': False, 'datasets_piloted': names, 'dataset_identity': identity,
              'protocol_id': protocol['protocol_id'], 'protocol_hash': config_id(protocol)}
    write_json(output/'pilot.json', report)
    return report


def hours(durations, workers):
    return {'serial_hours': sum(durations) / 3600, 'serial_over_workers_hours': sum(durations) / 3600 / workers,
            'simulated_makespan_hours': nd.makespan(durations, workers) / 3600}


def realized_full_data_hours(protocol, runs_root=None):
    """dedup only, informational: the realized fit and predict hours of each source dataset's ten models in the registered
    runs (measured under their 16 workers); the sources hold more rows than the deduplicated datasets."""
    if protocol['production_family'] != 'dedup':
        return None
    runs_root = Path(runs_root or nd.WORKSPACE_RUNS)
    holders = {'knn': (TRAINED_MODEL, *nd.COMPARATORS), 'training': (UNTRAINED_MODEL, INPUT_MODEL), 'projected': (PROJECTED_MODEL,)}
    result = {}
    for entry in protocol['panel']:
        by_model = {}
        for label, models in holders.items():
            directory = runs_root/HOLISTIC_SOURCES[label]
            if not (directory/'planned_jobs.json').is_file():
                return None
            for job in json.loads((directory/'planned_jobs.json').read_text()):
                if job['dataset_id'] == entry['source'] and job['model_id'] in models:
                    by_model[job['model_id']] = by_model.get(job['model_id'], 0.) + nd.job_fit_seconds(directory/job['log_file']) / 3600
        result[entry['name']] = {'source': entry['source'], 'serial_hours': sum(by_model.values()),
                                 'serial_hours_by_model': {model: by_model.get(model) for model in nd.MODEL_ORDER}}
    return result


def projection(protocol, pilot_path, calibration=None, runs_root=None):
    """The calibrated projection of a family: every run job priced as newdata.projection prices a piloted dataset
    (fits_per_outer x mean pilot seconds x the calibration factor of the realized bridge_knn and knn_training runs, central
    pooled and upper max, plus the job overhead), every ablation job as the fitting seeds x the piloted per-seed seconds of
    all variants x the arrowflow_full_knn factor; the decision is the central simulated makespan of the run jobs plus that of
    the ablation jobs at the family's workers plus STAGE_OVERHEAD_HOURS."""
    family = protocol['production_family']
    workers, cap = WORKERS[family], CAP_HOURS[family]
    pilot_path = Path(pilot_path)
    pilot = json.loads(pilot_path.read_text())
    if pilot.get('protocol_hash') != config_id(protocol):
        raise ValueError('The pilot did not run with this protocol')
    calibration = calibration or nd.CALIBRATION
    factors = {}
    for label, models in nd.CALIBRATION_MODELS.items():
        factors.update(nd.calibration_factors(calibration[label]['run'], json.loads(Path(calibration[label]['pilot']).read_text()), models))
    names = list(protocol['datasets'])
    means = nd.pilot_means(pilot)
    registry = nd.build_registry(protocol)
    per_outer = {model: nd.fits_per_outer(len(spec.candidates), spec.stochastic, protocol['inner_folds']) for model, spec in registry.items()}
    missing = [f'{name} {model}' for name in names for model in nd.MODEL_ORDER if (name, model) not in means]
    if missing:
        raise ValueError(f'The pilot must time every model on every dataset (missing {missing})')
    jobs = {(name, model): {kind: factors[nd.CALIBRATION_SOURCE.get(model, model)][factor] * per_outer[model] * means[name, model]
                                  + nd.JOB_OVERHEAD_SECONDS for kind, factor in KINDS.items()}
            for name in names for model in nd.MODEL_ORDER}
    folds = protocol['outer_folds'] * protocol['outer_repeats']
    run_hours = {kind: nd.batch_hours(jobs, names, kind, folds, workers) for kind in KINDS}
    ablation_rows = {row['dataset_id']: row for row in pilot['ablation_rows']}
    failed = [f"{row['dataset_id']}: {row.get('exception')}" for row in pilot['ablation_rows'] if row['status'] != 'ok']
    if failed or sorted(ablation_rows) != sorted(names):
        raise ValueError(f'The pilot must time every ablation variant on every dataset (failed: {failed})')
    arrowflow = factors[TRAINED_MODEL]
    ablation_jobs = {name: {kind: arrowflow[factor] * ablation_rows[name]['seconds_per_job_estimate'] + nd.JOB_OVERHEAD_SECONDS
                            for kind, factor in KINDS.items()} for name in names}
    ablation_hours = {kind: hours([ablation_jobs[name][kind] for name in names for _ in range(folds)], workers) for kind in KINDS}
    total = {kind: run_hours[kind]['simulated_makespan_hours'] + ablation_hours[kind]['simulated_makespan_hours'] + STAGE_OVERHEAD_HOURS
             for kind in KINDS}
    harness = sum(pilot['workload_estimates'][model]['serial_panel_seconds_using_observed_max'] for model in nd.MODEL_ORDER) / 3600 / workers
    return _plain({
        'purpose': f'{family}_calibrated_projection', 'family': family, 'protocol_id': protocol['protocol_id'],
        'protocol_hash': config_id(protocol), 'cap_hours': cap, 'workers': workers,
        'decision': f'calibrated central simulated makespan at {workers} workers of the run jobs plus the ablation jobs plus '
                    f'{STAGE_OVERHEAD_HOURS} h of stage overhead (prepare, reporting, ablation prepare and summary, analysis)',
        'decision_hours': total['central'], 'upper_hours': total['upper'], 'within_cap': total['central'] <= cap,
        'run': run_hours, 'ablation': ablation_hours, 'stage_overhead_hours': STAGE_OVERHEAD_HOURS,
        'harness_max_based_run_hours': harness,
        'per_dataset': {name: {'central_serial_hours': folds * sum(jobs[name, model]['central'] for model in nd.MODEL_ORDER) / 3600,
                               'upper_serial_hours': folds * sum(jobs[name, model]['upper'] for model in nd.MODEL_ORDER) / 3600,
                               'arrowflow_job_minutes': {kind: jobs[name, TRAINED_MODEL][kind] / 60 for kind in KINDS},
                               'ablation_job_minutes': {kind: ablation_jobs[name][kind] / 60 for kind in KINDS},
                               'central_serial_hours_by_model': {model: folds * jobs[name, model]['central'] / 3600 for model in nd.MODEL_ORDER}}
                        for name in names},
        'calibration': factors, 'fits_per_outer': per_outer,
        'job_seconds': {f'{name}|{model}': jobs[name, model] for name in names for model in nd.MODEL_ORDER},
        'ablation_job_seconds': ablation_jobs,
        'ablation_pilot': {name: {key: ablation_rows[name][key] for key in ('config_id', 'fit_sources', 'all_variants_to_views7_ratio',
                                                                          'seconds_per_seed', 'seconds_per_job_estimate')}
                           for name in names},
        'realized_full_data_serial_hours': realized_full_data_hours(protocol, runs_root),
        'assumptions': 'the calibration factors were measured under 16 single-thread workers of the bridge_knn and knn_training '
                       'production runs and are applied at this family\'s workers; other families running at the same time add '
                       'contention the factors do not hold; the ablation jobs take their fold\'s three seeds in sequence',
        'sources': {'pilot': {'path': str(pilot_path), 'sha256': nd.sha256_file(pilot_path)},
                    **{label: {'run': str(entry['run']), 'pilot': str(entry['pilot']), 'pilot_sha256': nd.sha256_file(entry['pilot'])}
                       for label, entry in calibration.items()}}})


def freeze(draft_path, projection_path, stages_path, output_path=None, *, frozen_at_utc=None):
    """The frozen family protocol from the committed draft, only if every dataset is pinned and the calibrated projection of
    that draft's pilot is within the cap; an existing file with different content is never replaced."""
    from .run_revision import write_json
    draft = json.loads(Path(draft_path).read_text())
    family = draft.get('production_family')
    if family not in FAMILIES or canonical_json(draft) != canonical_json(draft_protocol(family)):
        raise ValueError('The draft differs from extra_runs.draft_protocol(family); the stages must have used the committed draft')
    unpinned = [entry['name'] for entry in draft['panel'] if not entry.get('pinned', True)]
    if unpinned:
        raise ValueError(f'Not frozen: {", ".join(unpinned)} not pinned')
    plan, stages = json.loads(Path(projection_path).read_text()), json.loads(Path(stages_path).read_text())
    if plan.get('protocol_hash') != config_id(draft) or plan.get('family') != family or plan.get('workers') != WORKERS[family]:
        raise ValueError('The projection was not computed from a pilot of this draft at the family workers')
    if not plan.get('within_cap') or not 0 < plan['decision_hours'] <= CAP_HOURS[family]:
        raise ValueError(f'Not frozen: the calibrated projection {plan["decision_hours"]:.2f} h exceeds the {CAP_HOURS[family]} h cap '
                         f'at {WORKERS[family]} workers')
    record = {'cap_hours': CAP_HOURS[family], 'workers': WORKERS[family], 'decision': plan['decision'],
              'decision_hours': plan['decision_hours'], 'upper_hours': plan['upper_hours'], 'run': plan['run'],
              'ablation': plan['ablation'], 'stage_overhead_hours': plan['stage_overhead_hours'],
              'harness_max_based_run_hours': plan['harness_max_based_run_hours'],
              'per_dataset': {name: {key: entry[key] for key in ('central_serial_hours', 'upper_serial_hours', 'arrowflow_job_minutes',
                                                                 'ablation_job_minutes')} for name, entry in plan['per_dataset'].items()},
              'calibration_factors': {model: {'pooled': f['pooled'], 'max': f['max']} for model, f in plan['calibration'].items()},
              'ablation_pilot': plan['ablation_pilot'], 'realized_full_data_serial_hours': plan['realized_full_data_serial_hours'],
              'assumptions': plan['assumptions'], 'projection_sha256': nd.sha256_file(projection_path), 'stages': stages}
    run_hours, ablation_hours = plan['run']['central']['simulated_makespan_hours'], plan['ablation']['central']['simulated_makespan_hours']
    text = (f"G4 follow-up {ITEMS[family]} (ruling 2026-09-14): drafted from newdata_batch1.json with the identical nested design "
            f"and ten models and from newdata_ablation.json's component ablation; {stages['summary']}; projected at {WORKERS[family]} "
            f"single-thread workers as the calibrated central simulated makespan of the run ({run_hours:.2f} h) plus the ablation "
            f"({ablation_hours:.2f} h) plus {plan['stage_overhead_hours']} h of stage overhead: {plan['decision_hours']:.2f} h (upper "
            f"{plan['upper_hours']:.2f} h; harness max-based run {plan['harness_max_based_run_hours']:.2f} h); cap {CAP_HOURS[family]} h; "
            'frozen after the pilot')
    protocol = validate_extra_protocol(_plain(dict(draft, frozen=True, frozen_at_utc=frozen_at_utc or datetime.now(timezone.utc).isoformat(),
                                                   status=FROZEN_STATUS, resource_decision=text, projection=record)))
    write_json(Path(output_path or PROTOCOL_FILES[family]), protocol)
    return protocol


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

SMOKE_PANELS = {
    'dedup': ({'name': 'syn_dedup_a', 'role': 'family', 'source': 'syn_dedup_a', 'samples': 240, 'missing': False},
              {'name': 'syn_dedup_b', 'role': 'family', 'source': 'syn_dedup_b', 'samples': 120, 'missing': False}),
    'artificial': ({'name': 'syn_ranks_a', 'role': 'family', 'samples': 240, 'missing': True},
                   {'name': 'syn_ranks_b', 'role': 'family', 'samples': 120, 'missing': True},
                   {'name': 'syn_ranks_c', 'role': 'descriptive', 'samples': 90, 'missing': True})}


def smoke_protocol(family, panel=None, design=nd.SMOKE_DESIGN):
    """A frozen synthetic smoke protocol of one family (frozen only to pass reporting's gate; purpose synthetic_smoke_only)."""
    panel = _plain(list(panel or SMOKE_PANELS[family]))
    members = family_members(panel)
    protocol = dict(draft_protocol(family), **design, protocol_id=f'{PROTOCOL_IDS[family]}-synthetic-smoke', purpose='synthetic_smoke_only',
                    registry=SMOKE_REGISTRY, datasets=[entry['name'] for entry in panel], panel=panel,
                    primary_family_size=len(members), secondary_family_size=len(members), multiplicity=multiplicity(len(members)),
                    analysis=analysis_declaration(family, panel), frozen=True, frozen_at_utc='2026-09-14T00:00:00+00:00',
                    status='synthetic_smoke_only_not_evidence', resource_decision='synthetic smoke only', projection=None)
    return validate_extra_protocol(_plain(protocol))


def write_smoke_dataset(directory, entry, protocol, index):
    """A three-class, four-feature synthetic dataset in run_revision's prepared layout. With missing cells, one tenth of the
    cells is NaN and the second row repeats the first (which holds a NaN) under another label, so the NaN-aware duplicate
    audit has one label-conflicting group to count."""
    from .run_revision import write_json
    rng = np.random.RandomState(61 + index)
    y = np.tile([0, 1, 2], entry['samples'] // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    if entry.get('missing'):
        X[rng.rand(*X.shape) < .1] = np.nan
        X[0, 0] = np.nan
        X[1] = X[0]
    features, labels = [f'x{i}' for i in range(4)], ['0', '1', '2']
    splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
    manifest = {'dataset_id': entry['name'], 'purpose': 'synthetic_smoke_only', 'source': 'synthetic smoke dataset',
                'feature_names': features, 'label_map': labels, 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                'sample_order': 'source row order; zero-based sample_id', 'dataset_hash': dataset_fingerprint(X, y, features, labels),
                'splits_hash': config_id(splits)}
    target = Path(directory)/entry['name']
    write_json(target/'manifest.json', manifest)
    write_json(target/'splits.json', splits)
    if not (target/'data.npz').exists():
        np.savez_compressed(target/'data.npz', X=X, y=y)


def run_smoke_family(output, protocol, workers=1):
    """run_revision's prepare, run and reporting stages for a synthetic family run, through the harness's worker and validators."""
    from .reporting import summarize_verified_results
    from .run_revision import _worker, environment_record, get_registry, planned_jobs, write_json
    output = Path(output)
    registry = get_registry(protocol['registry'], protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', nd.candidate_record(registry))
    write_json(output/'environment.json', environment_record(protocol['registry']))
    for index, entry in enumerate(protocol['panel']):
        write_smoke_dataset(output, entry, protocol, index)
    write_json(output/'planned_jobs.json', planned_jobs(protocol['datasets'], protocol, registry))
    jobs = [(str(output), name, index, model, protocol['registry']) for name in protocol['datasets']
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for _ in pool.map(_worker, jobs):
            pass
    write_json(output/'summary.json', summarize_verified_results(output))
    return output


def smoke(output, family, workers=3):
    """A synthetic family run, its component ablation (prepare, run, summary) and compare_extra analyse; never evidence."""
    from . import extra_ablation
    from .compare_extra import analyse
    from .run_revision import execution_lock, write_json
    output = Path(output)
    protocol = smoke_protocol(family)
    with execution_lock():
        run_dir = run_smoke_family(output/'run', protocol, workers)
        extra_ablation.prepare(output/'ablation', protocol, run_dir, allow_smoke=True, purpose='synthetic_smoke_only')
    extra_ablation.run(output/'ablation', workers, allow_smoke=True)
    ablation = extra_ablation.write_summary(output/'ablation', allow_smoke=True)
    result = analyse(family, run_dir, output/'ablation', output/'analysis', allow_smoke=True)
    record = {'purpose': 'synthetic_smoke_only_not_paper_evidence', 'family': family, 'run': str(run_dir),
              'ablation': str(output/'ablation'), 'analysis': str(output/'analysis'),
              'primary_family': result['primary_family'], 'secondary_family': result['secondary_family'],
              'views7_reproduces_reference': {name: entry['views7_reproduces_reference'] for name, entry in ablation['summaries'].items()},
              'duplicate_audit': {name: entry['counts'] for name, entry in result['duplicate_audit'].items()},
              'outputs': sorted(result['outputs'])}
    write_json(output/'smoke.json', record)
    return record


def main(argv=None):
    from .run_revision import write_json
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    drafted = commands.add_parser('draft', help='write the unfrozen stage protocol of a family')
    drafted.add_argument('--family', choices=FAMILIES, required=True)
    drafted.add_argument('--output', type=Path, required=True)
    prepared = commands.add_parser('prepare', help='prepare the pinned datasets in run_revision layout')
    prepared.add_argument('--protocol', type=Path, required=True)
    prepared.add_argument('--output', type=Path, required=True)
    smoked = commands.add_parser('smoke', help='synthetic run, ablation and analysis of a family (never evidence)')
    smoked.add_argument('--family', choices=FAMILIES, required=True)
    smoked.add_argument('--output', type=Path, required=True)
    smoked.add_argument('--workers', type=int, default=3)
    piloted = commands.add_parser('pilot', help='training-only runtime pilot of every dataset')
    piloted.add_argument('--protocol', type=Path, required=True)
    piloted.add_argument('--output', type=Path, required=True)
    projected = commands.add_parser('project', help='the calibrated projection against the cap')
    projected.add_argument('--protocol', type=Path, required=True)
    projected.add_argument('--pilot', type=Path, required=True)
    projected.add_argument('--output', type=Path, required=True)
    frozen = commands.add_parser('freeze', help='write the frozen family protocol if the projection is within the cap')
    frozen.add_argument('--draft', type=Path, required=True)
    frozen.add_argument('--projection', type=Path, required=True)
    frozen.add_argument('--stages', type=Path, required=True)
    frozen.add_argument('--output', type=Path)
    running = commands.add_parser('run', help='run the prepared frozen family protocol')
    running.add_argument('--protocol', type=Path, required=True)
    running.add_argument('--output', type=Path, required=True)
    running.add_argument('--workers', type=int)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        write_json(args.output, draft_protocol(args.family))
    elif args.command == 'prepare':
        prepare(args.output, json.loads(args.protocol.read_text()))
    elif args.command == 'smoke':
        if not 1 <= args.workers <= MAX_WORKERS:
            raise ValueError('Worker count must be between 1 and 16')
        record = smoke(args.output, args.family, args.workers)
        for row in record['primary_family'] + record['secondary_family']:
            print(f"{row['family']} {row['dataset']}: {row['model_a']} - {row['model_b']} {row['mean_difference']:+.4f} "
                  f"Holm p={row['holm_p_approximate']:.3g} (synthetic)")
        print('views7 reproduced: ' + ', '.join(f"{name} {entry['matching_fold_seeds']}/{entry['total_fold_seeds']}"
                                                 for name, entry in record['views7_reproduces_reference'].items()))
    elif args.command == 'pilot':
        report = runtime_pilot(args.output, json.loads(args.protocol.read_text()))
        print(json.dumps(report['workload_estimates'], indent=2))
        for row in report['ablation_rows']:
            print(f"ablation {row['dataset_id']}: {row['status']} per seed {row.get('seconds_per_seed', float('nan')):.1f} s")
    elif args.command == 'project':
        record = projection(json.loads(args.protocol.read_text()), args.pilot)
        write_json(args.output, record)
        for name, entry in record['per_dataset'].items():
            print(f"{name}: central {entry['central_serial_hours']:.2f} serial h, upper {entry['upper_serial_hours']:.2f}; ArrowFlow-kNN "
                  f"job {entry['arrowflow_job_minutes']['central']:.1f} min, ablation job {entry['ablation_job_minutes']['central']:.1f} min")
        print(f"run makespan {record['run']['central']['simulated_makespan_hours']:.2f} h (upper {record['run']['upper']['simulated_makespan_hours']:.2f}), "
              f"ablation {record['ablation']['central']['simulated_makespan_hours']:.2f} h (upper {record['ablation']['upper']['simulated_makespan_hours']:.2f}); "
              f"decision {record['decision_hours']:.2f} h (upper {record['upper_hours']:.2f}) at {record['workers']} workers; "
              f"within the {record['cap_hours']} h cap: {record['within_cap']}")
    elif args.command == 'freeze':
        protocol = freeze(args.draft, args.projection, args.stages, args.output)
        print(f"frozen {protocol['protocol_id']} at {protocol['frozen_at_utc']}: decision {protocol['projection']['decision_hours']:.2f} h")
    else:
        protocol = json.loads(args.protocol.read_text())
        workers = args.workers if args.workers is not None else protocol.get('workers', 1)
        run(args.protocol, args.output, workers)


if __name__ == '__main__':
    main()
