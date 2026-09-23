"""prepare/smoke/pilot/run entry point; run requires an explicitly frozen protocol.

python -m experiments.make_revision.run_revision prepare --dataset iris --output /tmp/revision
Task5 can supply --registry package.module:function returning {model_id: ModelSpec}.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
import sklearn
from sklearn import datasets
from sklearn.dummy import DummyClassifier
from .evaluation import (ModelSpec, canonical_json, candidate_grid, dataset_fingerprint,
                         config_id, make_splits, evaluate_fold, _fit_predict,
                         expected_schedule, validate_outer_schedule)
from .models import ArrowFlowEstimator
from .datasets import SUSHI_ID, load_sushi
from .gene import GENE_ID

SOURCE_MODULES = ['experiments.make_revision.datasets','experiments.make_revision.reporting']

PROTOCOL = Path(__file__).with_name('protocol.json')
DATASETS = {
    'iris': (datasets.load_iris, (150, 4), [50, 50, 50]),
    'wine': (datasets.load_wine, (178, 13), [59, 71, 48]),
    'breast_cancer': (datasets.load_breast_cancer, (569, 30), [212, 357]),
    'digits': (datasets.load_digits, (1797, 64), [178,182,177,183,181,182,181,179,174,180]),
    'wine_quality': (40691, (1599, 11), [744,638,217]),
    'vehicle': (54, (846, 18), [218,212,217,199]),
    'segment': (36, (2310, 19), [330]*7),
}


def arrowflow_factory(config, seed):
    return ArrowFlowEstimator(seed=seed, **config)


def dummy_factory(config, seed):
    return DummyClassifier(strategy='most_frequent')


def default_registry(protocol):
    # Prespecified modest single-view training envelope; Task5 adds comparators.
    candidates = candidate_grid({'widths': [[32], [64], [128], [64,64]],
                                 'embed_dim': [16,32], 'degree': [1,2],
                                 'strategy': ['random','target_aware','calibrated'],
                                 'iterations': [100,200], 'learning_rate': [.1,.5]},
                                protocol['candidate_budget'], protocol['candidate_seed'])
    return {'arrowflow': ModelSpec('arrowflow', arrowflow_factory, candidates, True),
            'dummy': ModelSpec('dummy', dummy_factory, [{}], False)}


def get_registry(path, protocol):
    module, function = path.split(':')
    registry = getattr(importlib.import_module(module), function)(protocol)
    for name, spec in registry.items():
        if name != spec.model_id or len(spec.candidates) > protocol['candidate_budget']:
            raise ValueError('Invalid registry IDs or candidate budget')
    return registry


from contextlib import contextmanager

@contextmanager
def execution_lock():
    """One benchmark command per source tree; all families share the worker cap."""
    import fcntl
    import tempfile
    key=hashlib.sha256(str(Path(__file__).resolve().parents[2]).encode()).hexdigest()[:16]
    with (Path(tempfile.gettempdir())/f'arrowflow-make-{key}.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('A benchmark command is already active for this source tree') from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def code_revision():
    return subprocess.check_output(['git','rev-parse','HEAD'], text=True, cwd=Path(__file__).resolve().parents[2]).strip()


def environment_record(registry_path):
    import torch
    module = importlib.import_module(registry_path.split(':')[0])
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name('evaluation.py'),
             Path(__file__).with_name('models.py'), Path(__file__).with_name('reporting.py'), root/'arrowflow'/'arrowflow.py',
             root/'arrowflow'/'ranking.py', root/'arrowflow'/'config.py', root/'arrowflow'/'benchmark.py', Path(module.__file__)]
    paths.extend(Path(importlib.import_module(name).__file__)
                 for name in getattr(module, 'SOURCE_MODULES', ()))
    hashes = {str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    return {'code_revision': code_revision(), 'python': sys.version, 'numpy': np.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'registry': registry_path, 'numeric_threads': 1,
            'source_hashes': hashes}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace a different prepared input or existing result.
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite {path}; use a new output directory')
        return
    with path.open('x') as stream:
        stream.write(content)


def load_dataset(name):
    if name == SUSHI_ID:
        return load_sushi()
    if name == GENE_ID:
        from .gene import load_tcga
        return load_tcga()
    source, shape, counts = DATASETS[name]
    data = (datasets.fetch_openml(data_id=source, as_frame=False, parser='auto')
            if isinstance(source, int) else source())
    X = np.asarray(data.data, dtype=float)
    if name == 'wine_quality':
        quality = np.asarray(data.target, dtype=float)
        y = np.where(quality <= 5, 0, np.where(quality == 6, 1, 2))
        label_map = ['quality<=5', 'quality==6', 'quality>=7']
    else:
        labels, y = np.unique(data.target, return_inverse=True)
        label_map = [str(label) for label in (data.target_names if not isinstance(source, int) and 'target_names' in data else labels)]
    if X.shape != shape or np.bincount(y).tolist() != counts or np.isinf(X).any():
        raise ValueError(f'{name}: dataset identity/shape/class-count mismatch')
    names = [str(n) for n in data.get('feature_names', [f'x{i}' for i in range(shape[1])])]
    manifest = {'dataset_id': name, 'source': f'OpenML data_id={source}' if isinstance(source,int) else f'sklearn {source.__name__}',
                'shape': list(shape), 'class_counts': counts, 'feature_names': names,
                'label_map': label_map, 'sample_order': 'source row order; zero-based sample_id',
                'dataset_hash': dataset_fingerprint(X, y, names, label_map)}
    return X, np.asarray(y), manifest


def prepare(output, names, protocol, registry_path):
    registry = get_registry(registry_path, protocol)
    candidates = {name: {'stochastic': spec.stochastic, 'candidates': spec.candidates,
                         'config_ids': [config_id(c) for c in spec.candidates]} for name,spec in registry.items()}
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', candidates)
    write_json(output/'environment.json', environment_record(registry_path))
    for name in names:
        X,y,manifest = load_dataset(name)
        splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'],
                             protocol['inner_folds'], protocol['split_seed'])
        manifest['splits_hash'] = config_id(splits)
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        destination = output/name/'data.npz'
        if not destination.exists():
            np.savez_compressed(destination, X=X, y=y)
    return registry


def load_prepared(output, name):
    manifest=json.loads((output/name/'manifest.json').read_text())
    with np.load(output/name/'data.npz', allow_pickle=False) as data:
        X,y=data['X'],data['y']
    if dataset_fingerprint(X,y,manifest['feature_names'],manifest['label_map']) != manifest['dataset_hash']:
        raise ValueError('Prepared dataset hash mismatch')
    splits=json.loads((output/name/'splits.json').read_text())
    if config_id(splits) != manifest['splits_hash']:
        raise ValueError('Prepared splits hash mismatch')
    return X,y,manifest,splits


def planned_jobs(names, protocol, registry):
    """Explicit expected outputs, saved before the first worker starts."""
    if len(set(names)) != len(names):
        raise ValueError('Dataset job names must be unique')
    schedule = expected_schedule(protocol, registry)
    jobs = []
    for name in names:
        for repeat, fold in schedule['expected_folds']:
            for model in registry:
                stem = f'{name}__{model}__r{repeat}f{fold}'
                jobs.append({'dataset_id': name, 'model_id': model, 'outer_repeat': repeat,
                             'outer_fold': fold, 'model_seeds': schedule['expected_seeds'][model],
                             'result_file': f'results/{stem}.json',
                             'log_file': f'results/{stem}.fits.jsonl'})
    return jobs


def collect_confirmatory_results(output, names, protocol, registry):
    """Reconcile every planned result and full fit log before returning model rows.

    Task5 must use this collector, then pass expected_schedule(protocol, registry)
    to the statistical helpers. Missing/interrupted/failed jobs are never omitted.
    """
    output = Path(output)
    expected = planned_jobs(names, protocol, registry)
    manifest = output/'planned_jobs.json'
    if not manifest.exists() or canonical_json(json.loads(manifest.read_text())) != canonical_json(expected):
        raise ValueError('Missing or changed planned job manifest')
    rows = {name: [] for name in names}
    issues = []
    for job in expected:
        result_path, log_path = output/job['result_file'], output/job['log_file']
        if not result_path.exists():
            issues.append(f'missing result: {job["result_file"]}')
        if not log_path.exists():
            issues.append(f'missing fit log: {job["log_file"]}')
        if not result_path.exists() or not log_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text())
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            models = result['models']
            if result['status'] != 'ok':
                issues.append(f'{result["status"]}: {job["result_file"]}')
            if canonical_json(events) != canonical_json(result['selection']['fits'] + models):
                issues.append(f'fit log/result disagreement: {job["result_file"]}')
            for row in models:
                if any(row[key] != job[key] for key in ('dataset_id', 'model_id', 'outer_repeat', 'outer_fold')):
                    issues.append(f'result job identity mismatch: {job["result_file"]}')
            rows[job['dataset_id']].extend(models)
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'invalid result or fit log {job["result_file"]}: {exc}')
    schedule = expected_schedule(protocol, registry)
    for name, model_rows in rows.items():
        try:
            validate_outer_schedule(model_rows, **schedule)
        except ValueError as exc:
            issues.append(f'{name}: {exc}')
    if issues:
        raise ValueError('Incomplete planned evidence: ' + '; '.join(issues))
    return rows


def _worker(job):
    output, name, fold_index, model_name, registry_path = job
    output=Path(output)
    protocol=json.loads((output/'protocol.json').read_text())
    registry=get_registry(registry_path, protocol)
    X,y,manifest,splits=load_prepared(output,name)
    split=splits[fold_index]
    destination=output/'results'/f'{name}__{model_name}__r{split["outer_repeat"]}f{split["outer_fold"]}.json'
    log=destination.with_suffix('.fits.jsonl')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or log.exists():
        raise FileExistsError(f'Existing fold output: {destination}')
    with log.open('x') as stream:
        def sink(row):
            stream.write(canonical_json(row)+'\n'); stream.flush()
        result=evaluate_fold(X,y,split,registry[model_name],protocol['fit_seeds'],
                             dataset_id=name,dataset_hash=manifest['dataset_hash'],
                             code_revision=code_revision(),score=protocol['selection_metric'],sink=sink)
    write_json(destination,result)
    return str(destination)


def runtime_pilot(output,names,protocol,registry_path):
    registry=prepare(output,names,protocol,registry_path)
    # A registry module may reset per-family process state (e.g. a preprocessing memo) so that every
    # family's first pilot fit measures its own preprocessing cost.
    reset=getattr(importlib.import_module(registry_path.split(':')[0]),'reset_pilot_family_state',None)
    rows=[]
    for name in names:
        X,y,manifest,splits=load_prepared(output,name)
        train=splits[0]['train']  # never pass any outer-test sample or label
        for model,spec in registry.items():
            if reset is not None:reset()
            # Runtime only: fixed three evenly spaced candidates, no accuracy ranking.
            candidate_indices=sorted(set([0,len(spec.candidates)//2,len(spec.candidates)-1]))
            for index in candidate_indices:
                config=spec.candidates[index]
                start=time.perf_counter()
                row={'dataset_id':name,'model_id':model,'config':config,'config_id':config_id(config),
                     'fit_rows':train,'model_seed':protocol['fit_seeds'][0]}
                try:
                    _,timing=_fit_predict(spec,config,protocol['fit_seeds'][0],X[train],y[train],X[train])
                    row.update(timing,status='ok',elapsed_seconds=time.perf_counter()-start)
                except Exception as exc:
                    row.update(status='failed',exception=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-start)
                rows.append(row)
    estimates={}
    for model,spec in registry.items():
        durations=[r['elapsed_seconds'] for r in rows if r['model_id']==model and r['status']=='ok']
        fits_per_outer=len(spec.candidates)*protocol['inner_folds']
        fits_per_outer += min(3,len(spec.candidates))*2*protocol['inner_folds']+3 if spec.stochastic else 1
        total_fits=fits_per_outer*protocol['outer_folds']*protocol['outer_repeats']*len(protocol['datasets'])
        estimates[model]={'fits_per_outer':fits_per_outer,'panel_fit_count':total_fits,
                          'observed_seconds_min':min(durations) if durations else None,
                          'observed_seconds_max':max(durations) if durations else None,
                          'serial_panel_seconds_using_observed_max':total_fits*max(durations) if durations else None}
    report={'purpose':'training_only_runtime_no_heldout_scores','rows':rows,'workload_estimates':estimates,
            'estimate_limitations':'Sampled configurations/datasets; max extrapolation is not a runtime bound; RSS is process lifetime high-water mark.',
            'candidate_reduction':False}
    write_json(output/'pilot.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','smoke','pilot','run'])
    parser.add_argument('--dataset',nargs='+',choices=[*DATASETS, SUSHI_ID, GENE_ID],default=['iris'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--registry',default=__package__+'.run_revision:default_registry')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--sushi-archive',type=Path,help='Audited local provider archive; alternatively ARROWFLOW_SUSHI_ARCHIVE')
    parser.add_argument('--gene-cache',type=Path,help='Download/cache directory of the audited UCI gene archive; alternatively ARROWFLOW_GENE_CACHE')
    args=parser.parse_args(argv)
    if args.sushi_archive is not None:
        os.environ['ARROWFLOW_SUSHI_ARCHIVE'] = str(args.sushi_archive)
    if args.gene_cache is not None:
        os.environ['ARROWFLOW_GENE_CACHE'] = str(args.gene_cache)
    protocol=json.loads(args.protocol.read_text())
    if args.command=='run':
        if not protocol.get('frozen'):
            raise ValueError('Confirmatory run requires a reviewed frozen protocol')
        saved=json.loads((args.output/'protocol.json').read_text())
        if saved!=protocol:
            raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
        registry=get_registry(args.registry,protocol)
        expected={name:{'stochastic':s.stochastic,'candidates':s.candidates,
                        'config_ids':[config_id(c) for c in s.candidates]} for name,s in registry.items()}
        if canonical_json(json.loads((args.output/'candidates.json').read_text())) != canonical_json(expected):
            raise ValueError('Candidate registry changed after prepare')
        environment=json.loads((args.output/'environment.json').read_text())
        if environment != environment_record(args.registry):
            raise ValueError('Code revision, source, environment, or registry changed after prepare')
        if not 1<=args.workers<=16:
            raise ValueError('Worker count must be between 1 and 16')
        write_json(args.output/'planned_jobs.json', planned_jobs(args.dataset,protocol,registry))
        jobs=[(str(args.output),name,i,model,args.registry) for name in args.dataset
              for i in range(protocol['outer_folds']*protocol['outer_repeats']) for model in registry]
        with execution_lock(), ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for path in pool.map(_worker,jobs):
                print(path,flush=True)
        from .reporting import collect_verified_results
        collect_verified_results(args.output,args.dataset,protocol,registry)
    elif args.command=='prepare':
        prepare(args.output,args.dataset,protocol,args.registry)
    elif args.command=='pilot':
        with execution_lock():report=runtime_pilot(args.output,args.dataset,protocol,args.registry)
        print(json.dumps(report['workload_estimates'],indent=2))
    else:
        # Explicitly tiny integration exercise; disjoint from paper evidence.
        X,y,manifest=load_dataset(args.dataset[0])
        split=make_splits(y,3,1,2,protocol['split_seed'])[0]
        if args.dataset[0] == GENE_ID:
            from .gene import smoke_spec
            spec=smoke_spec()
        elif args.dataset[0] == SUSHI_ID:
            from functools import partial
            from .comparisons import NativeArrowFlow, ordinal_factory
            spec=ModelSpec('smoke_native_arrowflow',partial(ordinal_factory,NativeArrowFlow,stochastic=True),
                           [{'widths':[6],'iterations':4}],True)
        else:
            spec=ModelSpec('smoke_arrowflow',arrowflow_factory,
                           [{'widths':[6],'embed_dim':8,'iterations':4}],True)
        result=evaluate_fold(X,y,split,spec,protocol['fit_seeds'],dataset_id=args.dataset[0],
                             dataset_hash=manifest['dataset_hash'],code_revision=code_revision())
        write_json(args.output/'smoke.json',{'purpose':'smoke_only_not_paper_evidence','split':split,'result':result})


if __name__=='__main__':
    main()
