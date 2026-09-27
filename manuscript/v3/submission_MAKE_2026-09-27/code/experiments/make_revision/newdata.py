"""Task 23B: the training effect of ArrowFlow-kNN on ten additional official OpenML datasets (frozen selection).

The ten datasets are the eligible pool frozen in progress.md ("RULING (Task 23B selection frozen)"), with the identity,
strata and external kNN gaps of task-23-dataset-selection.md. They are pinned here, in a new module, because the dataset
tables of run_revision.py are sealed by earlier runs. Every dataset is loaded exactly as run_revision.load_dataset loads
an OpenML source (fetch_openml(data_id=..., as_frame=False, parser='auto') into sklearn's default data home; X as float;
labels from numpy.unique of the target, label_map their sorted strings; no load-time imputation, so any missing cell would
be imputed fold-locally by each model's NumericImputer) and refused unless every pin holds: OpenML data id, name,
version, file id, ARFF md5, default target, ignored attributes, shape, missing and infinite cells, label map, class
counts, the sha256 of the loaded float64 array and of the label strings, the sha256 of the feature names, the harness
dataset_fingerprint and the hash of the declared nested splits.

One combined registry (newdata_registry) holds ten models under the nested design of bridge_knn.json: arrowflow_full_knn
and the conventional comparators exactly as bridge.bridge_knn_registry builds them, the two training controls of
knn_controls.py and projected_numeric_knn of projected_knn.py. Production runs in two sealed batches of five datasets,
one frozen protocol per batch (protocols/2026-09-12/newdata_batch1.json and newdata_batch2.json); the prespecified
analysis in both protocols is computed only by compare_newdata.py after both batches are complete.

python -m experiments.make_revision.newdata draft --output P
    the unfrozen stage protocol (all ten datasets, no batch), used by the prepare, smoke and pilot stages
python -m experiments.make_revision.newdata prepare --protocol P --output O [--dataset NAME ...]
    run_revision's prepared layout (protocol, candidates, environment, manifest, splits, data) for pinned datasets
python -m experiments.make_revision.newdata smoke --output O [--workers 3] [--real-data --protocol P]
    synthetic two-batch run through run_revision's worker and reporting and compare_newdata analyse (never evidence);
    with --real-data also one fit of every model at its first candidate on the first outer training partition of every
    pinned dataset (network iterations reduced to 2; training rows only; no score recorded)
python -m experiments.make_revision.newdata pilot --protocol P --dataset NAME ... --output O
    run_revision.runtime_pilot for pinned datasets: three evenly spaced candidates of every model on the first outer
    training partition, predictions on training rows only
python -m experiments.make_revision.newdata project --pilot O/pilot.json --output PROJECTION.json
    calibrated per-dataset and per-batch projections at 16 workers and the batch split balanced by projected time
python -m experiments.make_revision.newdata freeze --draft P --projection PROJECTION.json --stages STAGES.json --output-dir D
    the two frozen batch protocols, only if every batch's calibrated projection is within the cap
python -m experiments.make_revision.newdata run --protocol B --output O --workers 16
    run_revision's run stage for one frozen batch protocol (the same checks, worker, lock and verification)
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'  # as run_revision: spawned workers import this -m module before any numeric library
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
from itertools import combinations, product
import json
from math import comb, factorial
import multiprocessing
from pathlib import Path
import time
import numpy as np
from sklearn import datasets
from .bridge import FIXED, bridge_candidates, bridge_knn_registry
from .comparisons import CONVENTIONAL_GRIDS, conventional_registry
from .evaluation import ModelSpec, canonical_json, config_id, dataset_fingerprint, make_splits
from .knn_controls import (CANDIDATE_KEYS as CONTROL_CANDIDATE_KEYS, INPUT_MODEL, KNN_READOUT_GRID, KNN_SELECTION_FOLDS,
                           TRAINED_MODEL, UNTRAINED_MODEL, control_candidates, input_factory, project_candidates,
                           untrained_factory)
from .projected_knn import NUMERIC_READOUT_GRID, PROJECTED_MODEL, RAW_MODEL, projected_candidates, projected_factory

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core: every module the
# bridge_knn, knn_training and knn_projected runs sealed for the models registered here.
SOURCE_MODULES = ['experiments.make_revision.bridge', 'experiments.make_revision.multiview',
                  'experiments.make_revision.comparisons', 'experiments.make_revision.datasets',
                  'experiments.make_revision.secondary_studies', 'experiments.make_revision.knn_controls',
                  'experiments.make_revision.projected_knn']

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-12'
WORKSPACE_RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'
REGISTRY = 'experiments.make_revision.newdata:newdata_registry'
SMOKE_REGISTRY = 'experiments.make_revision.newdata:smoke_newdata_registry'
PROTOCOL_FILES = {1: PROTOCOLS/'newdata_batch1.json', 2: PROTOCOLS/'newdata_batch2.json'}
PROTOCOL_IDS = {1: 'arrowflow-v3-newdata-batch1-1', 2: 'arrowflow-v3-newdata-batch2-1'}
DRAFT_PROTOCOL_ID = 'arrowflow-v3-newdata-stages-draft'
TEMPLATE = 'bridge_knn.json'
CAP_HOURS = 9
WORKERS = 16
BATCH_SIZE = 5
TIE_TOLERANCE = 1e-12

COMPARATORS = ('numeric_knn', 'svc_rbf', 'random_forest', 'mlp', 'gradient_boosting', 'dummy')
MODEL_ORDER = (TRAINED_MODEL, UNTRAINED_MODEL, INPUT_MODEL, PROJECTED_MODEL, 'dummy', 'svc_rbf', 'random_forest', 'mlp',
               'numeric_knn', 'gradient_boosting')
PRIMARY_CONTRAST = f'{TRAINED_MODEL}_vs_{UNTRAINED_MODEL}'
SECONDARY_CONTRAST = f'{TRAINED_MODEL}_vs_{INPUT_MODEL}'
LADDER = (('raw_numeric_knn', RAW_MODEL), ('unsorted_projected_knn', PROJECTED_MODEL), ('encoded_ranking_knn', INPUT_MODEL),
          ('untrained_arrowflow_knn', UNTRAINED_MODEL), ('arrowflow_knn', TRAINED_MODEL))

# The nested design keys copied unchanged from bridge_knn.json (the test pins this list against the template file).
DESIGN_COPY_KEYS = ('aggregation_integrity', 'augmentation', 'candidate_budget', 'candidate_seed', 'candidate_tie_rule',
                    'confidence', 'device', 'failure_policy', 'fit_seeds', 'freeze_requirement', 'full_method',
                    'historical_results', 'inner_folds', 'internal_validation_ratio', 'numeric_threads_per_worker',
                    'outer_folds', 'outer_repeats', 'paired_interval', 'parallelism', 'report_metrics', 'selection_metric',
                    'split_seed', 'stochastic_finalists', 'test_train_ratio')
DESIGN = {'split_seed': 27183, 'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3, 'fit_seeds': [8129, 19391, 39019],
          'candidate_budget': 24, 'candidate_seed': 41071, 'candidate_tie_rule': 'lowest_canonical_config_id',
          'stochastic_finalists': 3, 'selection_metric': 'accuracy', 'test_train_ratio': .25, 'confidence': .95}


# ----------------------------------------------------------------------------- the frozen panel and its pins

def _pin(name, data_id, openml_name, version, file_id, md5, target, shape, label_map, class_counts, sha_x, sha_y, sha_names,
         dataset_hash, splits_hash, duplicates, stratum, gap_sources, gap_points, ignore=None):
    return {'name': name, 'data_id': data_id, 'openml_name': openml_name, 'version': version, 'file_id': file_id,
            'md5_checksum': md5, 'default_target': target, 'ignore_attributes': ignore, 'shape': list(shape),
            'n_missing': 0, 'n_infinite': 0, 'label_map': list(label_map), 'class_counts': list(class_counts),
            'sha256_X': sha_x, 'sha256_y': sha_y, 'feature_names_sha256': sha_names, 'dataset_hash': dataset_hash,
            'splits_hash': splits_hash,
            'duplicates': dict(zip(('duplicate_rows', 'duplicate_groups', 'label_conflicting_groups'), duplicates)),
            'stratum': stratum, 'external_gap': {'sources': dict(gap_sources), 'points': gap_points}}


# Panel order: the order of the frozen ruling (stratum H by data id, then the other six by data id). Values transcribed from
# task-23-dataset-selection.md Section 2 (identity, class counts, sha256 of X and y, duplicates, Evidence column) and
# verified against fresh downloads on 2026-09-13; dataset_hash and splits_hash computed with the harness on those arrays.
# External gap: the best tuned RF, SVM or GB minus the best tuned kNN in points, per source as the selection document's
# Section 2 Evidence column prints it; points is the mean of the sources where two exist (non-informative OpenML sources
# included, as for climate, the only source of which is non-informative); HCV has no external evidence.
PINS = (
    _pin('balance_scale', 11, 'balance-scale', 1, '11', '76938608d472f620c170cef9c8c1fa65', 'class', (625, 4),
         ('B', 'L', 'R'), (49, 288, 288), 'd8d20b9bd4e5be6bff5a3ca6a6d81d4fb061084808008d5eac5abd31f8015cb4',
         '3b6759b01b84ce5b3470db3c17458cc2b2e54fe51d7daca8f2274c4f39a659c7',
         'dd07babe6ad083770e5ea00c9e95e2885ff09411c345ceb3bde9ae3b31079cf8',
         '498096feb1f6776ec20ae7b0a57cce39f12ad06c0afb1f62243c825423417e5f', '989fbe54fa5292b3', (0, 0, 0), 'H',
         {'openml_task_11': 9.9, 'fernandez_delgado_2014': 9.1}, 9.5),
    _pin('mfeat_zernike', 22, 'mfeat-zernike', 1, '22', '590fe11f6c0eeacd456609f22c4eaf6d', 'class', (2000, 47),
         ('1', '10', '2', '3', '4', '5', '6', '7', '8', '9'), (200,) * 10,
         '50fabead84e7c390f4f4e06957cb5c0f7fcc99e17b44fc6df9f59b0f7c300d66',
         'c67eb11583409b87ed18ed985fe8a480d983f641448c500226dfab7e93a1b952',
         'f9194d17c68d63f056514d1f9187a24bfc39d570e718c36e6888744d0b913b47',
         '5160d61c6e084dec230157fd910afb97e8d1fd0c7f4611f025040e2f0cd472d1', '3d403360286afa26', (6, 6, 0), 'H',
         {'openml_task_22': 8.7}, 8.7),
    _pin('ionosphere', 59, 'ionosphere', 1, '59', '23dd3c8b5693e2848901fd1a0248ef42', 'class', (351, 34), ('b', 'g'),
         (126, 225), '69468e18f3a97d2ed2709a0ad1fb4263f1dbd16f658b15500513ef06ccd7af71',
         'a62d1716a49fbef04a2c3de4da87f36baf1ca894776175eeeb715a3395f31c2f',
         '6722aa2893c657c079f3ef90a14cb988a2383cb78c48cd5f9df1fce7e43327c5',
         'dda052743b36e2698a87124a36c26b9c2c9ea58bcce6fa9fda1d3e5def492be5', '231ab272ec6a79b5', (1, 1, 0), 'H',
         {'openml_task_57': 6.3, 'fernandez_delgado_2014': 9.5}, 7.9),
    _pin('vertebra_column', 1523, 'vertebra-column', 1, '1593719', '78913b1453c43b7b9e0623b0d295b1cb', 'Class', (310, 6),
         ('1', '2', '3'), (60, 100, 150), 'a3ac0ae0d50c6b916be247218266a02a886acd1ab4d4930660a7d06ca6bf44ec',
         '0cbe50613e61df7cc4b00e7e54f4f5e05f77d4a40931e6a729c8981a0723835d',
         'eddf50c45b00547572581b2e5aff4c2fb3020ae18b0cfdb6e6c47a5ae5cfe491',
         '3770457475ec7d53e2368490358f146fdea80804b34305b33cd0a795e153bb4a', 'd1c530e96235c8c2', (0, 0, 0), 'H',
         {'fernandez_delgado_2014': 7.1, 'openml_task_9939': 7.4}, 7.25),
    _pin('diabetes', 37, 'diabetes', 1, '37', '3cbaa3e54586aa88cf6aacb4033e4470', 'class', (768, 8),
         ('tested_negative', 'tested_positive'), (500, 268),
         'db6367d8ed67a9f06f9276bd075617c0240bd36838c34d7b921f6b21df60036c',
         '4dc5a2259526cd11dde0ec7308b7c2f430574be817132ccaf98aa5390a9e6eb6',
         '3c97dbf919f1287d51e0899eaafca323df36f9eaf1a9798af92e6a8e475735ac',
         '26f50df6731663a5ad1f8c2a7181668f7e727637323927012c079a3f7b289a96', 'c791ebf248362e45', (0, 0, 0), 'C',
         {'openml_task_37': 3.5, 'fernandez_delgado_2014': 0.4}, 1.95),
    _pin('banknote_authentication', 1462, 'banknote-authentication', 1, '1586223', 'baa2dc5b745775a943ebeb9c276401f8',
         'Class', (1372, 4), ('1', '2'), (762, 610), 'f153ac742c3ec415cb4b672731908d9902f67c831d45f74941bf8538d8621642',
         'e6fb532213c8ead8829f17bb8bf24425040a3392b612e5e5f68983c174cb99ac',
         'b717eda3f7cc66dd82852dd00bb3eae70c3b39ecc454a7479c22f6abc3eb9984',
         '50f2701a1e1215b7ba70de3b0bb2adaaf03ac37aaec76a8f76252622ee389f7b', '7a0628abd290e42c', (24, 11, 0), 'C',
         {'openml_task_10093': 0.1}, 0.1),
    _pin('qsar_biodeg', 1494, 'qsar-biodeg', 1, '1592286', 'a2c189cd65511103fa540d7186155c24', 'Class', (1055, 41),
         ('1', '2'), (699, 356), '584e72d06483246d8ec69bb570035e491182e0144e40dd069e8d9ecda76187d0',
         'c9b656957085b059c1caffd2808deb3cabd906a181f6f7d7b18bf2be7b894d02',
         '5e039b594f67268c76ac43add2dc2f53d41fae97bcd5a53ec09a7cd5fc4acf7e',
         'd98d99ce1cb59785a3376398c16d9db78635f77cb6f707856d24dcc07db06c9f', '9c9cd4a11834c63b', (3, 3, 0), 'C',
         {'openml_task_9957': 2.3}, 2.3),
    _pin('steel_plates_fault', 40982, 'steel-plates-fault', 3, '18151921', '7ccdabeb01749cce9fa3b1d4a702fb8c', 'target',
         (1941, 27), ('Bumps', 'Dirtiness', 'K_Scratch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch'),
         (402, 55, 391, 673, 158, 72, 190), '92e432d6860deb12dd15b6b07ab9767e25b7812595d0745f5916288c19164c68',
         'bc4253c02101741b9cebe9329c507745ed71721d89ceedde22c94ebbe78c177c',
         '8bd0e3f66a0b19fee8df05634a3434512c47d9d907b051c4961ad6c33c2e6046',
         'f4408a76aa4aef6665368c99de713b9f7ba84bfbc0af24d2a458acd0e09ea924', '56e1c8a731816b1e', (0, 0, 0), 'C',
         {'fernandez_delgado_2014': 4.6, 'openml_task_146817': 7.4}, 6.0),
    _pin('climate_model_simulation_crashes', 40994, 'climate-model-simulation-crashes', 4, '18237248',
         'f7c55d9a11782a5ff980cee371787edd', 'outcome', (540, 18), ('0', '1'), (46, 494),
         'f7f6b2ae9a618cfc6a493cc351cbd5f5cb7ed5a1edab3bda524c0487e8de4e44',
         '8f74bf1bf64762681b5cffdec0892a17c663ec69fa2c51afe42bb2f32dc9fe9d',
         'c512fa59dc654b6bcb204b9524bfeb0bebf1cf6be7cdfd3bbe8558c5f2963343',
         '6c76b66184fc0e45b09e2b3a27fa5d94a94b77a934811eaa805ac295b56509bd', '9c9deb85963bf183', (0, 0, 0), 'C',
         {'openml_task_146819': 4.1}, 4.1, ignore=['Study', 'Run']),
    _pin('hcv_egyptian_patients', 46850, 'hepatitis_c_virus_hcv_for_egyptian_patients', 2, '22124413',
         '496acf3c6daf0ee7e124dc1802afb14f', 'Baselinehistological_staging', (1385, 28), ('1', '2', '3', '4'),
         (336, 332, 355, 362), '0174a199281d484d8ca60dcfae7e192a31c6ff6b951bb904a798cef2686d8335',
         '1aec6b85ddbdcc6dbca49d13c6af1313606a2b0b4159dae70e99419a6f3760f2',
         '7828d47b92465a3a4a0fd4f2467957920ef19b5878bb968c8d83ec0b9d380ea3',
         '284048565b76499ea5a35ce555bc937b165e1be197d9f520bfe64ddc9d4ebfd5', 'c6d05b187035a507', (0, 0, 0), 'C', {}, None),
)
PIN_BY_NAME = {pin['name']: pin for pin in PINS}
PANEL = tuple(pin['name'] for pin in PINS)
H_STRATUM = tuple(pin['name'] for pin in PINS if pin['stratum'] == 'H')
NOTES = {
    'ceiling_or_floor': {'banknote_authentication': 'near ceiling: every external model family reaches 99.9 to 100% accuracy',
                         'climate_model_simulation_crashes': 'majority class 91.5% (494 of 540 rows)',
                         'hcv_egyptian_patients': 'no external evidence; excluded from the Spearman correlation'},
    'low_power': 'ten datasets, four against six: the exact null distribution has 210 values, the smallest attainable p is '
                 '1/210, and p <= 0.05 needs the observed statistic among the ten largest; a dataset-level test this small '
                 'has low power, so a non-significant result is not evidence that the strata do not differ',
    'headroom': 'stratum H was selected for at least 5 points of external headroom above tuned kNN, so a stratum difference '
                'does not separate metric quality from headroom',
}
# Research estimate (task-23-dataset-selection.md Section 5): ArrowFlow job minutes per dataset, central (low, high), from
# the log-linear model of the realized bridge_knn fits; used only to project the datasets without a pilot.
RESEARCH_ARROWFLOW_JOB_MINUTES = {'balance_scale': (57, 40, 67), 'mfeat_zernike': (105, 73, 122), 'ionosphere': (66, 46, 77),
                                  'vertebra_column': (46, 32, 53), 'diabetes': (57, 40, 67),
                                  'banknote_authentication': (71, 49, 82), 'qsar_biodeg': (89, 62, 104),
                                  'steel_plates_fault': (86, 60, 100), 'climate_model_simulation_crashes': (62, 43, 73),
                                  'hcv_egyptian_patients': (78, 55, 91)}
RESEARCH_CALIBRATION = {'central': 1.01, 'low': .71, 'high': 1.18}


# ----------------------------------------------------------------------------- loading with refusal on any pin mismatch

class DatasetIdentityError(ValueError):
    """A loaded OpenML dataset differs from its pin."""


def array_sha256(X):
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def labels_sha256(target):
    return hashlib.sha256('\n'.join(str(value) for value in np.asarray(target).tolist()).encode('utf-8')).hexdigest()


def names_sha256(names):
    return hashlib.sha256('\n'.join(str(name) for name in names).encode('utf-8')).hexdigest()


def fetch_source(pin):
    """run_revision.load_dataset's OpenML call: sklearn's default data home is the persistent download cache."""
    return datasets.fetch_openml(data_id=pin['data_id'], as_frame=False, parser='auto')


def decode(data):
    """X, y, label_map and feature names exactly as run_revision.load_dataset derives them from an OpenML source."""
    X = np.asarray(data.data, dtype=float)
    labels, y = np.unique(data.target, return_inverse=True)
    names = [str(n) for n in data.get('feature_names', [f'x{i}' for i in range(X.shape[1])])]
    return X, np.asarray(y), [str(label) for label in labels], names


def identity_mismatches(pin, data):
    """[(field, pinned, loaded)] for every pinned identity field the loaded OpenML bunch fails, and its decoded arrays."""
    details = dict(getattr(data, 'details', None) or {})
    X, y, label_map, names = decode(data)
    loaded = {'data_id': details.get('id'), 'openml_name': details.get('name'), 'version': details.get('version'),
              'file_id': details.get('file_id'), 'md5_checksum': details.get('md5_checksum'),
              'default_target': details.get('default_target_attribute'), 'ignore_attributes': details.get('ignore_attribute'),
              'shape': list(X.shape), 'n_missing': int(np.isnan(X).sum()), 'n_infinite': int(np.isinf(X).sum()),
              'label_map': label_map, 'class_counts': np.bincount(y, minlength=len(label_map)).tolist(),
              'sha256_X': array_sha256(X), 'sha256_y': labels_sha256(data.target), 'feature_names_sha256': names_sha256(names),
              'dataset_hash': dataset_fingerprint(X, y, names, label_map)}
    pinned = {key: pin[key] for key in loaded}
    pinned.update(data_id=str(pin['data_id']), version=str(pin['version']))
    return [(key, pinned[key], loaded[key]) for key in loaded if loaded[key] != pinned[key]], (X, y, label_map, names)


def load_newdata(name, fetch=fetch_source):
    """(X, y, manifest) of one pinned dataset; DatasetIdentityError before anything is returned if any pin fails."""
    if name not in PIN_BY_NAME:
        raise DatasetIdentityError(f'{name} is not a pinned Task 23B dataset (pinned: {", ".join(PANEL)})')
    pin = PIN_BY_NAME[name]
    wrong, (X, y, label_map, names) = identity_mismatches(pin, fetch(pin))
    if wrong:
        raise DatasetIdentityError(f'{name}: dataset identity mismatch: '
                                   + '; '.join(f'{key} pinned {expected!r}, loaded {observed!r}' for key, expected, observed in wrong))
    manifest = {'dataset_id': name, 'source': f'OpenML data_id={pin["data_id"]}', 'shape': list(X.shape),
                'class_counts': np.bincount(y).tolist(), 'feature_names': names, 'label_map': label_map,
                'sample_order': 'source row order; zero-based sample_id', 'dataset_hash': pin['dataset_hash'],
                'openml': {key: pin[key] for key in ('data_id', 'openml_name', 'version', 'file_id', 'md5_checksum',
                                                     'default_target', 'ignore_attributes')},
                'identity': {key: pin[key] for key in ('sha256_X', 'sha256_y', 'feature_names_sha256', 'n_missing', 'n_infinite')},
                'stratum': pin['stratum'],
                'loader': 'newdata.load_newdata: run_revision.load_dataset for an OpenML source, with every pin checked'}
    return X, y, manifest


# ----------------------------------------------------------------------------- the combined registry

def build_registry(protocol):
    """The ten models: bridge_knn_registry's arrowflow_full_knn and comparators, both training controls and the projected
    control, in MODEL_ORDER."""
    knn = bridge_knn_registry(protocol)
    registry = {TRAINED_MODEL: knn[TRAINED_MODEL],
                UNTRAINED_MODEL: ModelSpec(UNTRAINED_MODEL, untrained_factory, control_candidates(UNTRAINED_MODEL), True),
                INPUT_MODEL: ModelSpec(INPUT_MODEL, input_factory, control_candidates(INPUT_MODEL), True),
                PROJECTED_MODEL: ModelSpec(PROJECTED_MODEL, projected_factory, projected_candidates(), True)}
    registry.update((model, spec) for model, spec in knn.items() if model != TRAINED_MODEL)
    if tuple(registry) != MODEL_ORDER:
        raise RuntimeError(f'Registry order {list(registry)} differs from MODEL_ORDER')
    return registry


def newdata_registry(protocol):
    """The production registry of a newdata protocol (stage draft or frozen batch)."""
    validate_newdata_protocol(protocol)
    if protocol.get('registry') != REGISTRY:
        raise ValueError(f'newdata_registry serves protocols declaring {REGISTRY}')
    return build_registry(protocol)


SMOKE_TRAINED_CANDIDATES = [dict(c, iterations=1) for c in bridge_candidates()
                            if c['learning_rate'] == .1 and c['widths'] == [128] and c['embed_scale'] == 1]


def smoke_newdata_registry(protocol):
    """Synthetic smoke only: the ten models with ArrowFlow-kNN at two one-iteration candidates, the controls at their
    projections and every comparator at its first two candidates."""
    if protocol.get('purpose') != 'synthetic_smoke_only':
        raise ValueError('smoke_newdata_registry serves synthetic smoke protocols only')
    validate_newdata_protocol(protocol)
    real = build_registry(protocol)
    trained = SMOKE_TRAINED_CANDIDATES
    registry = {TRAINED_MODEL: ModelSpec(TRAINED_MODEL, real[TRAINED_MODEL].factory, trained, True)}
    for model in (UNTRAINED_MODEL, INPUT_MODEL, PROJECTED_MODEL):
        keys = CONTROL_CANDIDATE_KEYS[UNTRAINED_MODEL if model == UNTRAINED_MODEL else INPUT_MODEL]
        registry[model] = ModelSpec(model, real[model].factory, project_candidates(trained, keys), True)
    for model in MODEL_ORDER[4:]:
        registry[model] = ModelSpec(model, real[model].factory, real[model].candidates[:2], real[model].stochastic)
    return registry


# ----------------------------------------------------------------------------- protocol declarations and validation

INTERVAL_RULE = ('fitting seeds averaged within each outer fold; corrected resampled t over the outer folds '
                 '(evaluation.paired_corrected_interval, q = test_train_ratio = 0.25, 95%, df = outer folds - 1 = 14); '
                 'two-sided p')


def _plain(value):
    """The JSON form of a declaration (tuples become lists), so it compares equal to a protocol read from disk."""
    return json.loads(canonical_json(value))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel_declaration():
    return _plain(list(PINS))


def models_declaration():
    registry = build_registry(DESIGN)
    readout = {'aggregation': 'majority', 'grid': KNN_READOUT_GRID, 'selection_folds': KNN_SELECTION_FOLDS,
               'selection': 'mean accuracy over StratifiedKFold(3, shuffle=True, random_state=derive_seed(view_seed, '
                            '"readout_selection")) splits of the training partition inside every fit; ties '
                            'lowest_canonical_config_id; refit on the full training representation',
               'representation': 'inverse positions of the final hidden ranking of each view'}
    models = {
        TRAINED_MODEL: {'estimator': 'experiments.make_revision.multiview:MultiViewArrowFlowKNN via bridge.AdaptiveMultiViewKNN',
                        'registry': 'bridge.bridge_knn_registry', 'candidate_source': 'bridge.bridge_candidates (full_method.candidate_grid)',
                        'candidates': len(registry[TRAINED_MODEL].candidates), 'stochastic': True, 'readout': readout,
                        'network': 'seven views trained with full_method.fixed (checkpoint validation_ratio 0.1) and the '
                                   'augmentation rule; each view reseeds from derive_seed(fit seed, "view", v)',
                        'template': 'bridge_knn.json knn_readout without its reference block'},
        UNTRAINED_MODEL: {'estimator': 'experiments.make_revision.knn_controls:UntrainedMultiViewArrowFlowKNN via AdaptiveUntrainedKNN',
                          'candidate_keys': list(CONTROL_CANDIDATE_KEYS[UNTRAINED_MODEL]),
                          'candidates': len(registry[UNTRAINED_MODEL].candidates), 'stochastic': True,
                          'template': 'knn_training.json training_controls.models.arrowflow_knn_untrained'},
        INPUT_MODEL: {'estimator': 'experiments.make_revision.knn_controls:MultiViewInputKNN via AdaptiveInputKNN',
                      'candidate_keys': list(CONTROL_CANDIDATE_KEYS[INPUT_MODEL]),
                      'candidates': len(registry[INPUT_MODEL].candidates), 'stochastic': True,
                      'template': 'knn_training.json training_controls.models.input_footrule_knn'},
        PROJECTED_MODEL: {'estimator': 'experiments.make_revision.projected_knn:MultiViewProjectedKNN via AdaptiveProjectedKNN',
                          'candidate_keys': list(CONTROL_CANDIDATE_KEYS[INPUT_MODEL]), 'readout_grid': NUMERIC_READOUT_GRID,
                          'candidates': len(registry[PROJECTED_MODEL].candidates), 'stochastic': True,
                          'template': 'knn_projected.json projected_control.model'},
        'dummy': {'estimator': 'sklearn DummyClassifier(strategy="most_frequent") (run_revision.dummy_factory)',
                  'candidates': 1, 'stochastic': False, 'template': 'bridge_knn.json registry'},
    }
    for family, grid in CONVENTIONAL_GRIDS.items():
        models[family] = {'estimator': 'comparisons.conventional_factory (NumericImputer, StandardScaler for svc_rbf, mlp '
                                       'and numeric_knn, classifier)', 'grid': grid,
                          'candidates': len(registry[family].candidates), 'stochastic': registry[family].stochastic,
                          'template': 'bridge_knn.json registry (comparisons.conventional_registry)'}
    return _plain(models)


def analysis_declaration(panel, notes=None):
    names = [d['name'] for d in panel]
    h = [d['name'] for d in panel if d['stratum'] == 'H']
    c = [d['name'] for d in panel if d['stratum'] != 'H']
    with_gap = [d['name'] for d in panel if d['external_gap']['points'] is not None]
    splits = comb(len(names), len(h))
    tie = f'a value within {TIE_TOLERANCE} of the observed counts as at least the observed'
    return _plain({
        'status': 'prespecified before any outer score on these datasets exists; computed only by compare_newdata analyse, '
                  'which refuses unless both batch runs are complete',
        'command': 'python -m experiments.make_revision.compare_newdata analyse --batch1 <batch 1 run> --batch2 '
                   '<batch 2 run> --output <directory>',
        'requires_both_batches_complete': True, 'metric': 'accuracy', 'datasets': names,
        'primary_family': {'contrast': PRIMARY_CONTRAST, 'model_a': TRAINED_MODEL, 'model_b': UNTRAINED_MODEL, 'size': len(names),
                           'definition': f'per dataset, {TRAINED_MODEL} minus {UNTRAINED_MODEL} accuracy (the training effect)',
                           'interval': INTERVAL_RULE, 'multiplicity': f'Holm across the {len(names)} datasets (evaluation.holm_adjust)',
                           'alpha': .05},
        'secondary_family': {'contrast': SECONDARY_CONTRAST, 'model_a': TRAINED_MODEL, 'model_b': INPUT_MODEL, 'size': len(names),
                             'definition': f'per dataset, {TRAINED_MODEL} minus {INPUT_MODEL} accuracy',
                             'interval': INTERVAL_RULE,
                             'multiplicity': f'Holm across the {len(names)} datasets, adjusted separately from the primary family',
                             'alpha': .05},
        'moderator_test': {'hypothesis': 'the dataset-level mean training effect is larger in stratum H than in the other datasets',
                           'effect': 'the primary family mean_difference of each dataset', 'strata': {'H': h, 'C': c},
                           'statistic': 'mean effect over H minus mean effect over C',
                           'null_distribution': f'all {splits} assignments of the {len(names)} datasets to a group of {len(h)} '
                                                f'and a group of {len(c)}, every dataset effect held fixed',
                           'splits': splits, 'sided': 'one-sided, H greater',
                           'p_value': f'the number of assignments whose statistic is at least the observed statistic (the '
                                      f'observed assignment included; {tie}) divided by {splits}',
                           'alpha': .05, 'smallest_attainable_p': 1 / splits,
                           'effect_sizes': 'per stratum: n, mean, SD (ddof 1), median, min and max of the dataset effects, '
                                           'with every dataset effect and its stratum'},
        'spearman': {'status': 'secondary; descriptive', 'gap': 'panel external_gap.points (frozen before any run)',
                     'effect': 'the primary family mean_difference of each dataset', 'datasets': with_gap,
                     'excluded': [name for name in names if name not in with_gap],
                     'rho': 'Pearson correlation of the average ranks of the gaps and of the effects (ties share their average rank)',
                     'p_value': f'exact over all {factorial(len(with_gap))} permutations of the effects against the fixed gaps: '
                                f'one-sided in the direction of the hypothesis (rho at least the observed) and two-sided '
                                f'(|rho| at least |observed|); {tie}'},
        'ladder': [{'rung': rung, 'model_id': model} for rung, model in LADDER],
        'ladder_table': 'per dataset, mean outer error, outer-fold SD (ddof 1) and mean within-fold seed SD of every rung, '
                        'from the verified summary.json of the batch run holding the dataset',
        'comparator_intervals': {'models': list(COMPARATORS), 'status': 'descriptive; no multiplicity adjustment',
                                 'definition': f'per dataset and comparator, {TRAINED_MODEL} minus the comparator accuracy: '
                                               f'{INTERVAL_RULE}, unadjusted; the comparator with the lowest mean outer '
                                               'error is flagged for display'},
        'duplicate_audit': {'status': 'descriptive',
                            'definition': 'exact duplicate raw feature rows of the prepared data (float64 equality of every '
                                          'feature; referee_analyses.duplicate_groups): rows, distinct rows, duplicate rows, '
                                          'duplicate groups, label-conflicting groups and the largest group; per outer fold '
                                          'the test rows with an exact duplicate in the training partition',
                            'expected': {d['name']: d.get('duplicates') for d in panel}},
        'notes': notes if notes is not None else {},
    })


def validate_batches(batches, names):
    """Two batches of equal size that partition the panel, each listed in panel order."""
    half = len(names) // 2
    if (not isinstance(batches, dict) or sorted(batches) != ['1', '2'] or len(names) % 2
            or any(not isinstance(batches[key], list) or len(batches[key]) != half for key in ('1', '2'))):
        raise ValueError(f'batches must map "1" and "2" to {half} datasets each')
    joined = batches['1'] + batches['2']
    if len(set(joined)) != len(joined) or set(joined) != set(names):
        raise ValueError('batches must partition the panel')
    if any(batches[key] != [name for name in names if name in batches[key]] for key in ('1', '2')):
        raise ValueError('each batch must list its datasets in panel order')
    return batches


def _validate_smoke_panel(panel):
    if (not isinstance(panel, list) or len(panel) < 4 or len(panel) % 2 or any(not isinstance(d, dict) for d in panel)
            or len({d.get('name') for d in panel}) != len(panel) or any(d.get('stratum') not in ('H', 'C') for d in panel)
            or {d['stratum'] for d in panel} != {'H', 'C'}
            or any(not isinstance(d.get('external_gap'), dict) or 'points' not in d['external_gap'] for d in panel)
            or sum(d['external_gap']['points'] is not None for d in panel) < 3):
        raise ValueError('a synthetic smoke panel holds an even number (at least four) of uniquely named datasets in strata H '
                         'and C, at least three of them with an external gap')


def validate_newdata_protocol(p):
    """Refuse a protocol whose panel, design, models, contrasts, analysis, batches or projection differ from this module."""
    smoke = p.get('purpose') == 'synthetic_smoke_only'
    panel = p.get('panel')
    if smoke:
        _validate_smoke_panel(panel)
    elif panel != panel_declaration():
        raise ValueError('panel must be the ten frozen datasets with their pins, strata and external gaps (newdata.PINS)')
    names = [d['name'] for d in panel]
    if p.get('production_family') != 'newdata':
        raise ValueError('production_family must be newdata')
    if p.get('registry') != (SMOKE_REGISTRY if smoke else REGISTRY):
        raise ValueError(f'registry must be {SMOKE_REGISTRY if smoke else REGISTRY}')
    if not smoke:
        wrong = sorted(key for key, value in DESIGN.items() if p.get(key) != value)
        if wrong:
            raise ValueError(f'the nested design differs from bridge_knn.json in {", ".join(wrong)}')
    if p.get('model_order') != list(MODEL_ORDER) or p.get('models') != models_declaration():
        raise ValueError('model_order and models must declare the ten registered models (newdata.models_declaration)')
    if p.get('primary_contrasts') != [PRIMARY_CONTRAST] or p.get('secondary_contrasts') != [SECONDARY_CONTRAST]:
        raise ValueError(f'primary_contrasts must be [{PRIMARY_CONTRAST}] and secondary_contrasts [{SECONDARY_CONTRAST}]')
    if p.get('primary_family_size') != len(names) or p.get('secondary_family_size') != len(names):
        raise ValueError('primary_family_size and secondary_family_size must equal the panel size')
    if p.get('analysis') != analysis_declaration(panel, {} if smoke else NOTES):
        raise ValueError('analysis must be the prespecified analysis of the panel (newdata.analysis_declaration)')
    batch = p.get('batch')
    if batch is None:
        if p.get('frozen') or p.get('batches') is not None or p.get('datasets') != names:
            raise ValueError('a protocol without a batch is the unfrozen stage draft of the whole panel')
    else:
        batches = validate_batches(p.get('batches'), names)
        if type(batch) is not int or batch not in (1, 2) or p.get('datasets') != batches[str(batch)]:
            raise ValueError('batch must be 1 or 2, and datasets the datasets of that batch')
        if not smoke and p.get('protocol_id') != PROTOCOL_IDS[batch]:
            raise ValueError(f'the batch {batch} protocol_id must be {PROTOCOL_IDS[batch]}')
    if p.get('frozen') and not smoke:
        projection = p.get('batch_projection') or {}
        decision = projection.get('decision_hours') or {}
        if (p.get('wallclock_cap_hours') != CAP_HOURS or projection.get('cap_hours') != CAP_HOURS
                or projection.get('batches') != p.get('batches') or sorted(decision) != ['1', '2']
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= CAP_HOURS for v in decision.values())):
            raise ValueError(f'a frozen batch protocol records the batch projection of both batches within the {CAP_HOURS} h cap')
    return p


def base_protocol():
    """The unfrozen stage draft: bridge_knn.json's nested design with the panel, models, contrasts and analysis."""
    template_path = PROTOCOLS/TEMPLATE
    template = json.loads(template_path.read_text())
    protocol = {key: template[key] for key in DESIGN_COPY_KEYS}
    protocol.update(
        protocol_id=DRAFT_PROTOCOL_ID, production_family='newdata', registry=REGISTRY, datasets=list(PANEL), batch=None,
        batches=None, panel=panel_declaration(), model_order=list(MODEL_ORDER), models=models_declaration(),
        primary_contrasts=[PRIMARY_CONTRAST], primary_family_size=len(PANEL), secondary_contrasts=[SECONDARY_CONTRAST],
        secondary_family_size=len(PANEL),
        multiplicity='Holm_across_the_10_datasets_within_each_family; the primary and secondary families are adjusted separately',
        analysis=analysis_declaration(panel_declaration(), NOTES), wallclock_cap_hours=CAP_HOURS,
        design_source={'template': 'protocols/2026-09-12/bridge_knn.json (sha256 in source_template_sha256)',
                       'identical_to_template': ', '.join(DESIGN_COPY_KEYS),
                       'removed_from_template': {
                           'knn_readout': 'its readout declaration moved to models.arrowflow_full_knn; its reference to the '
                                          'bridge run does not apply, because every model is fitted afresh on these datasets',
                           'secondary_studies': 'not applicable to this family'},
                       'model_templates': 'models.arrowflow_knn_untrained and models.input_footrule_knn as knn_training.json '
                                          'declares them, models.projected_numeric_knn as knn_projected.json declares it '
                                          '(sha256 in model_template_sha256)'},
        source_template_sha256=sha256_file(template_path),
        model_template_sha256={name: sha256_file(PROTOCOLS/name) for name in ('knn_training.json', 'knn_projected.json')},
        selection_ruling='progress.md "RULING (Task 23B selection frozen)", recorded 2026-09-13 before any ArrowFlow or '
                         'comparator fit on these datasets: all ten eligible datasets, reported whatever the result; strata, '
                         'identity and external gaps from task-23-dataset-selection.md',
        frozen=False, status='drafted_awaiting_prepare_smoke_and_training_only_pilot',
        resource_decision='pending: prepare, synthetic smoke with a real-data fit check, and the training-only pilot on '
                          'mfeat_zernike (largest) and vertebra_column (smallest)', batch_projection=None)
    return _plain(protocol)


def batch_protocol(draft, batch, batches, *, frozen_at_utc, resource_decision, batch_projection):
    """One frozen batch protocol derived from the stage draft; only batch fields and freeze provenance change."""
    protocol = _plain(dict(draft, protocol_id=PROTOCOL_IDS[batch], batch=batch, batches=batches,
                           datasets=list(batches[str(batch)]), frozen=True, frozen_at_utc=frozen_at_utc,
                           status='reviewed_and_piloted_before_confirmatory_scoring', resource_decision=resource_decision,
                           batch_projection=batch_projection))
    return validate_newdata_protocol(protocol)


# ----------------------------------------------------------------------------- stages: prepare, run, pilot

def candidate_record(registry):
    return {name: {'stochastic': spec.stochastic, 'candidates': spec.candidates,
                   'config_ids': [config_id(c) for c in spec.candidates]} for name, spec in registry.items()}


def prepare(output, names, protocol):
    """run_revision.prepare for pinned datasets: the same records and layout, with load_newdata as the loader and the
    splits hash checked against its pin."""
    from .run_revision import environment_record, get_registry, write_json
    output = Path(output)
    outside = [name for name in names if name not in protocol['datasets']]
    if outside:
        raise ValueError(f'Datasets outside the protocol datasets: {outside}')
    registry = get_registry(protocol['registry'], protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', candidate_record(registry))
    write_json(output/'environment.json', environment_record(protocol['registry']))
    for name in names:
        X, y, manifest = load_newdata(name)
        splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
        manifest['splits_hash'] = config_id(splits)
        if manifest['splits_hash'] != PIN_BY_NAME[name]['splits_hash']:
            raise DatasetIdentityError(f'{name}: splits hash {manifest["splits_hash"]} differs from the pin')
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        destination = output/name/'data.npz'
        if not destination.exists():
            np.savez_compressed(destination, X=X, y=y)
    return registry


def run(protocol_path, output, workers):
    """run_revision's run stage for one frozen batch protocol, with the same checks in the same order."""
    from .reporting import collect_verified_results
    from .run_revision import (_worker, environment_record, execution_lock, get_registry, load_prepared, planned_jobs,
                               write_json)
    output = Path(output)
    protocol = json.loads(Path(protocol_path).read_text())
    if not protocol.get('frozen'):
        raise ValueError('Confirmatory run requires a reviewed frozen protocol')
    if protocol.get('purpose') is not None or validate_newdata_protocol(protocol).get('batch') not in (1, 2):
        raise ValueError('A production run is one batch of a frozen newdata protocol')
    saved = json.loads((output/'protocol.json').read_text())
    if saved != protocol:
        raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
    registry_path = protocol['registry']
    registry = get_registry(registry_path, protocol)
    if canonical_json(json.loads((output/'candidates.json').read_text())) != canonical_json(candidate_record(registry)):
        raise ValueError('Candidate registry changed after prepare')
    if json.loads((output/'environment.json').read_text()) != environment_record(registry_path):
        raise ValueError('Code revision, source, environment, or registry changed after prepare')
    if not 1 <= workers <= WORKERS:
        raise ValueError('Worker count must be between 1 and 16')
    names = list(protocol['datasets'])
    for name in names:
        manifest = load_prepared(output, name)[2]
        if (manifest['dataset_hash'], manifest['splits_hash']) != (PIN_BY_NAME[name]['dataset_hash'], PIN_BY_NAME[name]['splits_hash']):
            raise DatasetIdentityError(f'{name}: the prepared dataset or splits hash differs from the pin')
    write_json(output/'planned_jobs.json', planned_jobs(names, protocol, registry))
    jobs = [(str(output), name, index, model, registry_path) for name in names
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in pool.map(_worker, jobs):
            print(path, flush=True)
    collect_verified_results(output, names, protocol, registry)


def fits_per_outer(n_candidates, stochastic, inner_folds):
    """run_revision.runtime_pilot's fit count of one outer-fold job."""
    return n_candidates * inner_folds + (min(3, n_candidates) * 2 * inner_folds + 3 if stochastic else 1)


def runtime_pilot(output, names, protocol):
    """run_revision.runtime_pilot for pinned datasets: three evenly spaced candidates of every model on the first outer
    training partition, predictions on training rows only; the same pilot.json layout and estimates."""
    from .evaluation import _fit_predict
    from .run_revision import execution_lock, load_prepared, write_json
    output = Path(output)
    with execution_lock():
        registry = prepare(output, names, protocol)
        rows = []
        for name in names:
            X, y, _, splits = load_prepared(output, name)
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
    estimates = {}
    for model, spec in registry.items():
        durations = [r['elapsed_seconds'] for r in rows if r['model_id'] == model and r['status'] == 'ok']
        per_outer = fits_per_outer(len(spec.candidates), spec.stochastic, protocol['inner_folds'])
        total = per_outer * protocol['outer_folds'] * protocol['outer_repeats'] * len(protocol['datasets'])
        estimates[model] = {'fits_per_outer': per_outer, 'panel_fit_count': total,
                            'observed_seconds_min': min(durations) if durations else None,
                            'observed_seconds_max': max(durations) if durations else None,
                            'serial_panel_seconds_using_observed_max': total * max(durations) if durations else None}
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'rows': rows, 'workload_estimates': estimates,
              'estimate_limitations': 'Sampled configurations/datasets; max extrapolation is not a runtime bound; RSS is process lifetime high-water mark.',
              'candidate_reduction': False, 'datasets_piloted': list(names), 'protocol_id': protocol['protocol_id']}
    write_json(output/'pilot.json', report)
    return report


# ----------------------------------------------------------------------------- projection and the balanced batch split

CALIBRATION = {'bridge_knn': {'run': WORKSPACE_RUNS/'2026-09-12-bridge-knn',
                              'pilot': WORKSPACE_RUNS/'2026-09-12-bridge-knn-pilot2'/'pilot.json'},
               'knn_training': {'run': WORKSPACE_RUNS/'2026-09-13-knn-training',
                                'pilot': WORKSPACE_RUNS/'2026-09-13-knn-training-pilot'/'pilot.json'}}
CALIBRATION_MODELS = {'bridge_knn': (TRAINED_MODEL, *COMPARATORS), 'knn_training': (UNTRAINED_MODEL, INPUT_MODEL)}
CALIBRATION_SOURCE = {PROJECTED_MODEL: INPUT_MODEL}      # same job structure and encoders; its own run is not read
JOB_OVERHEAD_SECONDS = 1.


def job_fit_seconds(log_path):
    """Fit plus predict seconds summed over every fit of one job's fit log."""
    return float(sum((row.get('fit_seconds') or 0.) + (row.get('predict_seconds') or 0.)
                     for row in map(json.loads, Path(log_path).read_text().splitlines())))


def pilot_means(pilot):
    """{(dataset, model): mean elapsed seconds of its pilot fits}; a failed pilot fit is refused."""
    groups = {}
    for row in pilot['rows']:
        if row['status'] != 'ok':
            raise ValueError(f"A pilot fit failed ({row['dataset_id']} {row['model_id']} {row['config_id']}): {row.get('exception')}")
        groups.setdefault((row['dataset_id'], row['model_id']), []).append(row['elapsed_seconds'])
    return {key: float(np.mean(values)) for key, values in groups.items()}


def calibration_factors(run_dir, pilot, models):
    """Per model and piloted dataset of a completed production run: realized mean job seconds (fit plus predict of every
    fit, under that run's 16 workers) over the pilot projection of the job (fits_per_outer times the mean idle pilot
    seconds); pooled (sum over sum) and max over the datasets."""
    run_dir = Path(run_dir)
    candidates = json.loads((run_dir/'candidates.json').read_text())
    inner = json.loads((run_dir/'protocol.json').read_text())['inner_folds']
    realized = {}
    for job in json.loads((run_dir/'planned_jobs.json').read_text()):
        if job['model_id'] in models:
            realized.setdefault((job['dataset_id'], job['model_id']), []).append(job_fit_seconds(run_dir/job['log_file']))
    means, factors = pilot_means(pilot), {}
    for model in models:
        per_outer = fits_per_outer(len(candidates[model]['candidates']), candidates[model]['stochastic'], inner)
        per = {dataset: {'jobs': len(realized[dataset, model]), 'realized_mean_job_seconds': float(np.mean(realized[dataset, model])),
                         'pilot_mean_fit_seconds': mean, 'pilot_projected_job_seconds': per_outer * mean}
               for (dataset, name), mean in sorted(means.items()) if name == model and (dataset, model) in realized}
        if not per:
            raise ValueError(f'No calibration dataset holds both pilot fits and realized jobs of {model}')
        for entry in per.values():
            entry['factor'] = entry['realized_mean_job_seconds'] / entry['pilot_projected_job_seconds']
        factors[model] = {'fits_per_outer': per_outer, 'datasets': per, 'max': max(e['factor'] for e in per.values()),
                          'pooled': sum(e['realized_mean_job_seconds'] for e in per.values())
                                    / sum(e['pilot_projected_job_seconds'] for e in per.values())}
    return factors


def job_projection(pilot, factors, panel=PANEL):
    """{(dataset, model): projected seconds of one outer-fold job under the 16-worker load, central and upper}.

    Piloted datasets: fits_per_outer x mean pilot seconds x the calibration factor (central pooled, upper max), plus
    JOB_OVERHEAD_SECONDS. Other datasets: ArrowFlow-kNN from the research estimate re-anchored on the piloted datasets
    (central: the mean ratio of the central projection to the research central; upper: the largest ratio of the upper
    projection, times the research model's high over central calibration), every other model at the slowest piloted
    dataset."""
    registry = build_registry(DESIGN)
    per_outer = {model: fits_per_outer(len(spec.candidates), spec.stochastic, DESIGN['inner_folds']) for model, spec in registry.items()}
    means = pilot_means(pilot)
    piloted = [name for name in panel if any(dataset == name for dataset, _ in means)]
    missing = [(name, model) for name in piloted for model in MODEL_ORDER if (name, model) not in means]
    if not piloted or missing:
        raise ValueError(f'The pilot must time every model on each piloted dataset (missing {missing})')
    jobs = {}
    for name in piloted:
        for model in MODEL_ORDER:
            factor, base = factors[CALIBRATION_SOURCE.get(model, model)], per_outer[model] * means[name, model]
            jobs[name, model] = {'central': factor['pooled'] * base + JOB_OVERHEAD_SECONDS,
                                 'upper': factor['max'] * base + JOB_OVERHEAD_SECONDS, 'basis': 'pilot'}
    research = {name: RESEARCH_ARROWFLOW_JOB_MINUTES[name][0] * 60. for name in panel}
    ratios = {name: {kind: (jobs[name, TRAINED_MODEL][kind] - JOB_OVERHEAD_SECONDS) / research[name] for kind in ('central', 'upper')}
              for name in piloted}
    anchor = {'central': float(np.mean([ratio['central'] for ratio in ratios.values()])),
              'upper': max(ratio['upper'] for ratio in ratios.values()) * RESEARCH_CALIBRATION['high'] / RESEARCH_CALIBRATION['central']}
    for name in panel:
        if name in piloted:
            continue
        jobs[name, TRAINED_MODEL] = {**{kind: research[name] * anchor[kind] + JOB_OVERHEAD_SECONDS for kind in ('central', 'upper')},
                                     'basis': 'research estimate re-anchored on the piloted datasets'}
        for model in MODEL_ORDER[1:]:
            jobs[name, model] = {**{kind: max(jobs[p, model][kind] for p in piloted) for kind in ('central', 'upper')},
                                 'basis': 'slowest piloted dataset'}
    return jobs, {'piloted': piloted, 'fits_per_outer': per_outer, 'research_ratios': ratios, 'anchor': anchor}


def makespan(durations, workers=WORKERS):
    """Wall time of jobs taken in order by the first free of `workers` workers (ProcessPoolExecutor.map's FIFO queue)."""
    free, end = [0.] * workers, 0.
    for duration in durations:
        finish = heapq.heappop(free) + duration
        end = max(end, finish)
        heapq.heappush(free, finish)
    return end


def batch_hours(jobs, names, kind, folds, workers=WORKERS):
    """Serial hours, serial hours over the workers and the simulated makespan of one batch in run's job order."""
    durations = [jobs[name, model][kind] for name in names for _ in range(folds) for model in MODEL_ORDER]
    return {'serial_hours': sum(durations) / 3600, 'serial_over_workers_hours': sum(durations) / 3600 / workers,
            'simulated_makespan_hours': makespan(durations, workers) / 3600}


def balanced_split(serial, panel=PANEL):
    """Two batches of equal size, the first panel dataset in batch 1, minimizing the absolute difference of their projected
    serial time; ties go to the lexicographically smallest panel indices of batch 1."""
    best = None
    for others in combinations(range(1, len(panel)), len(panel) // 2 - 1):
        indices = (0, *others)
        one = [panel[i] for i in indices]
        two = [name for name in panel if name not in one]
        key = (abs(sum(serial[name] for name in one) - sum(serial[name] for name in two)), indices)
        if best is None or key < best[0]:
            best = (key, one, two)
    return {'1': best[1], '2': best[2]}


def projection(pilot_path, calibration=CALIBRATION, workers=WORKERS):
    """The calibrated projection of every dataset and of the balanced batches, with the cap decision."""
    pilot_path = Path(pilot_path)
    pilot = json.loads(pilot_path.read_text())
    factors = {}
    for label, models in CALIBRATION_MODELS.items():
        factors.update(calibration_factors(calibration[label]['run'], json.loads(Path(calibration[label]['pilot']).read_text()), models))
    jobs, basis = job_projection(pilot, factors)
    folds = DESIGN['outer_folds'] * DESIGN['outer_repeats']
    per_dataset = {name: {'central_serial_hours': folds * sum(jobs[name, m]['central'] for m in MODEL_ORDER) / 3600,
                          'upper_serial_hours': folds * sum(jobs[name, m]['upper'] for m in MODEL_ORDER) / 3600,
                          'arrowflow_job_minutes': {kind: jobs[name, TRAINED_MODEL][kind] / 60 for kind in ('central', 'upper')},
                          'basis': 'pilot' if name in basis['piloted'] else 'research estimate re-anchored (ArrowFlow-kNN); slowest piloted dataset (other models)',
                          'central_serial_hours_by_model': {m: folds * jobs[name, m]['central'] / 3600 for m in MODEL_ORDER}}
                   for name in PANEL}
    batches = balanced_split({name: per_dataset[name]['central_serial_hours'] for name in PANEL})
    per_batch = {key: {kind: batch_hours(jobs, batches[key], kind, folds, workers) for kind in ('central', 'upper')} for key in batches}
    observed_max = {m: pilot['workload_estimates'][m]['serial_panel_seconds_using_observed_max'] / len(PANEL) for m in MODEL_ORDER}
    harness = {key: len(batches[key]) * sum(observed_max.values()) / 3600 / workers for key in batches}
    decision = {key: per_batch[key]['central']['simulated_makespan_hours'] for key in batches}
    calibration_jobs = [job_fit_seconds(Path(calibration['bridge_knn']['run'])/job['log_file'])
                        for job in json.loads((Path(calibration['bridge_knn']['run'])/'planned_jobs.json').read_text())]
    return _plain({
        'purpose': 'task23b_calibrated_projection_and_balanced_batch_split', 'cap_hours': CAP_HOURS, 'workers': workers,
        'decision': 'calibrated central simulated makespan of each batch at 16 workers', 'decision_hours': decision,
        'within_cap': all(value <= CAP_HOURS for value in decision.values()), 'batches': batches, 'per_batch': per_batch,
        'harness_max_based_hours': harness, 'per_dataset': per_dataset, 'calibration': factors, 'basis': basis,
        'job_seconds': {f'{name}|{model}': jobs[name, model] for name in PANEL for model in MODEL_ORDER},
        'job_overhead_seconds': JOB_OVERHEAD_SECONDS,
        'simulation_check': {'run': str(calibration['bridge_knn']['run']), 'jobs': len(calibration_jobs),
                             'serial_hours': sum(calibration_jobs) / 3600,
                             'simulated_makespan_hours': makespan(calibration_jobs, workers) / 3600},
        'sources': {'pilot': {'path': str(pilot_path), 'sha256': sha256_file(pilot_path), 'datasets': pilot.get('datasets_piloted')},
                    **{label: {'run': str(entry['run']), 'pilot': str(entry['pilot']), 'pilot_sha256': sha256_file(entry['pilot'])}
                       for label, entry in calibration.items()},
                    'research_estimate': 'task-23-dataset-selection.md Section 5 (RESEARCH_ARROWFLOW_JOB_MINUTES)'}})


def freeze(draft_path, projection_path, stages_path, output_dir, frozen_at_utc=None):
    """Write both frozen batch protocols from the stage draft, only if every batch projects within the cap."""
    from .run_revision import write_json
    draft = json.loads(Path(draft_path).read_text())
    if draft != base_protocol():
        raise ValueError('The stage draft differs from newdata.base_protocol(); the stages must have used the committed draft')
    plan, stages = json.loads(Path(projection_path).read_text()), json.loads(Path(stages_path).read_text())
    over = {key: value for key, value in plan['decision_hours'].items() if value > CAP_HOURS}
    if over:
        raise ValueError(f'Not frozen: the batch projections {over} exceed the {CAP_HOURS} h cap')
    batches = validate_batches(plan['batches'], list(PANEL))
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    record = {'cap_hours': CAP_HOURS, 'workers': plan['workers'], 'batches': batches, 'decision': plan['decision'],
              'decision_hours': plan['decision_hours'], 'per_batch': plan['per_batch'],
              'harness_max_based_hours': plan['harness_max_based_hours'],
              'per_dataset': {name: {key: plan['per_dataset'][name][key] for key in ('central_serial_hours', 'upper_serial_hours',
                                                                                   'arrowflow_job_minutes', 'basis')} for name in PANEL},
              'calibration_factors': {model: {'pooled': f['pooled'], 'max': f['max']} for model, f in plan['calibration'].items()},
              'research_anchor': plan['basis']['anchor'], 'simulation_check': plan['simulation_check'],
              'projection_sha256': sha256_file(projection_path), 'stages': stages}
    hours = ', '.join(f"batch {key} {plan['decision_hours'][key]:.2f} h (upper {plan['per_batch'][key]['upper']['simulated_makespan_hours']:.2f} h, "
                      f"harness max-based {plan['harness_max_based_hours'][key]:.2f} h)" for key in ('1', '2'))
    decision = (f"Task 23B (ruling 2026-09-13, selection frozen): drafted from bridge_knn.json with the identical nested design; "
                f"{stages['summary']}; projected at {plan['workers']} single-thread workers as the calibrated central simulated "
                f"makespan: {hours}; batches balanced by projected serial time; cap {CAP_HOURS} h per batch; frozen after the pilot")
    paths = []
    for batch in (1, 2):
        path = Path(output_dir)/f'newdata_batch{batch}.json'
        write_json(path, batch_protocol(draft, batch, batches, frozen_at_utc=frozen_at, resource_decision=decision,
                                        batch_projection=record))
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- synthetic smoke (never evidence) and real-data fit check

SMOKE_PANEL = _plain([{'name': 'syn_h1', 'stratum': 'H', 'external_gap': {'sources': {'synthetic': 9.0}, 'points': 9.0}},
                      {'name': 'syn_c1', 'stratum': 'C', 'external_gap': {'sources': {'synthetic': 1.0}, 'points': 1.0}},
                      {'name': 'syn_h2', 'stratum': 'H', 'external_gap': {'sources': {'synthetic': 6.0}, 'points': 6.0}},
                      {'name': 'syn_c2', 'stratum': 'C', 'external_gap': {'sources': {}, 'points': None}}])
SMOKE_BATCHES = {'1': ['syn_h1', 'syn_c1'], '2': ['syn_h2', 'syn_c2']}
SMOKE_DESIGN = {'outer_folds': 3, 'outer_repeats': 1, 'inner_folds': 2}


def smoke_protocol(batch, panel=SMOKE_PANEL, batches=SMOKE_BATCHES, design=SMOKE_DESIGN):
    """A frozen synthetic smoke batch protocol (frozen only to pass reporting's gate; purpose synthetic_smoke_only)."""
    names = [entry['name'] for entry in panel]
    protocol = _plain(dict(base_protocol(), **design, protocol_id=f'{PROTOCOL_IDS[batch]}-synthetic-smoke', registry=SMOKE_REGISTRY,
                           purpose='synthetic_smoke_only', panel=panel, analysis=analysis_declaration(panel, {}), batch=batch,
                           batches=batches, datasets=list(batches[str(batch)]), primary_family_size=len(names),
                           secondary_family_size=len(names), frozen=True, frozen_at_utc='2026-09-13T00:00:00+00:00',
                           status='synthetic_smoke_only_not_evidence', resource_decision='synthetic smoke only'))
    return validate_newdata_protocol(protocol)


def write_smoke_dataset(directory, name, protocol, samples=120):
    """A three-class, four-feature synthetic dataset in run_revision's prepared layout; the second panel dataset repeats its
    first row with another label, so the duplicate audit has one label-conflicting group to count."""
    from .run_revision import write_json
    index = [entry['name'] for entry in protocol['panel']].index(name)
    rng = np.random.RandomState(33 + index)
    y = np.tile([0, 1, 2], samples // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1. + .5 * index
    if index == 1:
        X[1] = X[0]
    features, labels = [f'x{i}' for i in range(4)], ['0', '1', '2']
    splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
    manifest = {'dataset_id': name, 'purpose': 'synthetic_smoke_only', 'source': 'synthetic smoke dataset', 'feature_names': features,
                'label_map': labels, 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                'sample_order': 'source row order; zero-based sample_id', 'dataset_hash': dataset_fingerprint(X, y, features, labels),
                'splits_hash': config_id(splits)}
    target = Path(directory)/name
    write_json(target/'manifest.json', manifest)
    write_json(target/'splits.json', splits)
    if not (target/'data.npz').exists():
        np.savez_compressed(target/'data.npz', X=X, y=y)


def run_smoke_batch(output, protocol, workers=1):
    """run_revision's prepare, run and reporting stages for one synthetic batch, through the harness's worker and validators."""
    from .reporting import summarize_verified_results
    from .run_revision import _worker, environment_record, get_registry, planned_jobs, write_json
    output = Path(output)
    registry = get_registry(protocol['registry'], protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', candidate_record(registry))
    write_json(output/'environment.json', environment_record(protocol['registry']))
    for name in protocol['datasets']:
        write_smoke_dataset(output, name, protocol)
    write_json(output/'planned_jobs.json', planned_jobs(protocol['datasets'], protocol, registry))
    jobs = [(str(output), name, index, model, protocol['registry']) for name in protocol['datasets']
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for _ in pool.map(_worker, jobs):
            pass
    write_json(output/'summary.json', summarize_verified_results(output))
    return output


def real_data_check(protocol, names=None, iterations=2):
    """One fit of every model at its first candidate on the first outer training partition of each pinned dataset,
    predicting those training rows (arrowflow_full_knn with `iterations` network iterations). Status, seconds and peak
    RSS only; no prediction is scored."""
    from .evaluation import _fit_predict
    registry = newdata_registry(protocol)
    names = list(names or protocol['datasets'])
    rows = []
    for name in names:
        X, y, _ = load_newdata(name)
        train = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])[0]['train']
        for model, spec in registry.items():
            config = dict(spec.candidates[0], **({'iterations': iterations} if model == TRAINED_MODEL else {}))
            row = {'dataset_id': name, 'model_id': model, 'config_id': config_id(spec.candidates[0]), 'fit_rows': len(train),
                   'iterations_override': iterations if model == TRAINED_MODEL else None}
            start = time.perf_counter()
            try:
                predictions, timing = _fit_predict(spec, config, protocol['fit_seeds'][0], X[train], y[train], X[train])
                row.update(status='ok', predictions=len(predictions), fit_seconds=timing['fit_seconds'],
                           predict_seconds=timing['predict_seconds'], peak_process_rss_kib=timing['peak_process_rss_kib'])
            except Exception as exc:
                row.update(status='failed', exception=f'{type(exc).__name__}: {exc}')
            row['elapsed_seconds'] = time.perf_counter() - start
            rows.append(row)
    return {'purpose': 'real_data_fit_check_training_rows_only_no_scores', 'datasets': names, 'rows': rows,
            'failed': [f"{row['dataset_id']} {row['model_id']}: {row['exception']}" for row in rows if row['status'] != 'ok']}


def smoke(output, workers=3, real_data_protocol=None):
    """Two synthetic batches through run_revision's worker and reporting, then compare_newdata analyse on them; optionally
    the real-data fit check."""
    from .compare_newdata import analyse
    from .run_revision import execution_lock, write_json
    output = Path(output)
    with execution_lock():
        batch1 = run_smoke_batch(output/'batch1', smoke_protocol(1), workers)
        batch2 = run_smoke_batch(output/'batch2', smoke_protocol(2), workers)
        result = analyse(batch1, batch2, output/'analysis',
                         frozen_protocols={1: sha256_file(batch1/'protocol.json'), 2: sha256_file(batch2/'protocol.json')})
        record = {'purpose': 'synthetic_smoke_only_not_paper_evidence', 'batch1': str(batch1), 'batch2': str(batch2),
                  'analysis': str(output/'analysis'), 'primary_family': result['primary_family'],
                  'secondary_family': result['secondary_family'],
                  'moderator_test': {key: value for key, value in result['moderator_test'].items() if key != 'null_distribution'},
                  'spearman': result['spearman'], 'duplicate_audit': result['duplicate_audit'],
                  'outputs': sorted(result['outputs'])}
        if real_data_protocol is not None:
            record['real_data_fit_check'] = real_data_check(real_data_protocol)
    write_json(output/'smoke.json', record)
    return record


def main(argv=None):
    from .run_revision import write_json
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    draft = commands.add_parser('draft', help='write the unfrozen stage protocol')
    draft.add_argument('--output', type=Path, required=True)
    prepared = commands.add_parser('prepare', help='prepare pinned datasets in run_revision layout')
    prepared.add_argument('--protocol', type=Path, required=True)
    prepared.add_argument('--output', type=Path, required=True)
    prepared.add_argument('--dataset', nargs='+', choices=PANEL)
    smoked = commands.add_parser('smoke', help='synthetic two-batch smoke and optional real-data fit check')
    smoked.add_argument('--output', type=Path, required=True)
    smoked.add_argument('--workers', type=int, default=3)
    smoked.add_argument('--real-data', action='store_true')
    smoked.add_argument('--protocol', type=Path)
    piloted = commands.add_parser('pilot', help='training-only runtime pilot')
    piloted.add_argument('--protocol', type=Path, required=True)
    piloted.add_argument('--dataset', nargs='+', choices=PANEL, required=True)
    piloted.add_argument('--output', type=Path, required=True)
    projected = commands.add_parser('project', help='calibrated projection and the balanced batch split')
    projected.add_argument('--pilot', type=Path, required=True)
    projected.add_argument('--output', type=Path, required=True)
    frozen = commands.add_parser('freeze', help='write the two frozen batch protocols if both project within the cap')
    frozen.add_argument('--draft', type=Path, required=True)
    frozen.add_argument('--projection', type=Path, required=True)
    frozen.add_argument('--stages', type=Path, required=True)
    frozen.add_argument('--output-dir', type=Path, default=PROTOCOLS)
    running = commands.add_parser('run', help='run one prepared frozen batch')
    running.add_argument('--protocol', type=Path, required=True)
    running.add_argument('--output', type=Path, required=True)
    running.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        write_json(args.output, base_protocol())
    elif args.command == 'prepare':
        protocol = json.loads(args.protocol.read_text())
        prepare(args.output, args.dataset or protocol['datasets'], protocol)
    elif args.command == 'smoke':
        if not 1 <= args.workers <= WORKERS:
            raise ValueError('Worker count must be between 1 and 16')
        if args.real_data and args.protocol is None:
            parser.error('--real-data needs --protocol')
        record = smoke(args.output, args.workers, json.loads(args.protocol.read_text()) if args.real_data else None)
        for row in record['primary_family'] + record['secondary_family']:
            print(f"{row['family']} {row['dataset']}: {row['model_a']} - {row['model_b']} {row['mean_difference']:+.4f} "
                  f"Holm p={row['holm_p_approximate']:.3g} (synthetic)")
        check = record.get('real_data_fit_check')
        if check is not None:
            print(f"real-data fit check: {len(check['rows'])} fits, {len(check['failed'])} failed")
            for failure in check['failed']:
                print(f'  FAILED {failure}')
    elif args.command == 'pilot':
        report = runtime_pilot(args.output, args.dataset, json.loads(args.protocol.read_text()))
        print(json.dumps(report['workload_estimates'], indent=2))
    elif args.command == 'project':
        record = projection(args.pilot)
        write_json(args.output, record)
        for name in PANEL:
            entry = record['per_dataset'][name]
            print(f"{name}: central {entry['central_serial_hours']:.2f} serial h, upper {entry['upper_serial_hours']:.2f} ({entry['basis']})")
        for key, names in record['batches'].items():
            hours = record['per_batch'][key]
            print(f"batch {key} {names}: central makespan {hours['central']['simulated_makespan_hours']:.2f} h, upper "
                  f"{hours['upper']['simulated_makespan_hours']:.2f} h, harness max-based {record['harness_max_based_hours'][key]:.2f} h")
        print(f"within the {CAP_HOURS} h cap: {record['within_cap']}")
    elif args.command == 'freeze':
        for path in freeze(args.draft, args.projection, args.stages, args.output_dir):
            print(path)
    else:
        run(args.protocol, args.output, args.workers)


if __name__ == '__main__':
    main()
