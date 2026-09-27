"""Prepare, inspect and execute the sealed E01/E03/E04 matched study.

Confirmation is blocked until the reviewed study protocol is explicitly frozen.
Smoke uses synthetic data; pilot and shuffle never score an outer test partition.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import time
import numpy as np
from .datasets import SUSHI_ID
from .comparisons import derive_seed
from .evaluation import (canonical_json,config_id,dataset_fingerprint,make_splits,
                         validate_outer_schedule,summarize_outer,paired_corrected_interval,holm_adjust,metric_values)
from .models import array_hash
from .matched import (ArtifactWriter,architecture_id,model_seed_schedule,probe_candidates,encoder_settings,validate_encoder_table,
                      build_partition,score_inner_partition,evaluate_study_fold,choose_configurations)
from .run_revision import load_dataset,load_prepared,write_json,environment_record,execution_lock

STUDY_PROTOCOL=Path(__file__).with_name('study_protocol.json')
SOURCE_MODULES=['experiments.make_revision.matched','experiments.make_revision.models',
                'experiments.make_revision.comparisons','experiments.make_revision.datasets',
                'experiments.make_revision.evaluation','experiments.make_revision.run_revision']


def study_environment():
    env=environment_record(__package__+'.run_studies:study_environment')
    root=Path(__file__).resolve().parents[2]
    for relative in ['experiments/make_revision/study_protocol.json','experiments/make_revision/protocol.json',
                     'manuscript/MAKE/review/secondary_experiment_spec.md']:
        env['source_hashes'][relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    return env


def expected_jobs(names,p):
    return [{'dataset_id':name,'outer_repeat':repeat,'outer_fold':fold,
             'model_seeds':model_seed_schedule(p,name==SUSHI_ID),
             'stem':f'{name}__r{repeat}f{fold}'}
            for name in names for repeat in range(p['outer_repeats']) for fold in range(p['outer_folds'])]


def prepare_study(output,names,p,*,purpose='confirmatory',data_loader=None):
    output=Path(output);data_loader=data_loader or load_dataset
    if len(names)!=len(set(names)):raise ValueError('Dataset list contains duplicate jobs')
    validate_encoder_table(p)
    write_json(output/'protocol.json',p)
    write_json(output/'environment.json',study_environment())
    write_json(output/'run_manifest.json',{'purpose':purpose,'datasets':names,'protocol_hash':config_id(p),
        'probe_candidates':probe_candidates(p),'hdc_candidates':[{'dimension':d} for d in p['hdc_dimensions']]})
    write_json(output/'planned_jobs.json',expected_jobs(names,p))
    for name in names:
        X,y,manifest=data_loader(name)
        splits=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])
        manifest=dict(manifest,splits_hash=config_id(splits))
        write_json(output/name/'manifest.json',manifest);write_json(output/name/'splits.json',splits)
        data_path=output/name/'data.npz'
        if not data_path.exists():np.savez_compressed(data_path,X=X,y=y)


def verify_prepared(output,*,allow_smoke=False):
    output=Path(output);p=json.loads((output/'protocol.json').read_text());validate_encoder_table(p)
    manifest=json.loads((output/'run_manifest.json').read_text())
    if not p['frozen'] and not (allow_smoke and manifest['purpose']=='synthetic_smoke_only'):
        raise ValueError('Matched confirmation/summary requires a frozen reviewed protocol')
    if manifest['protocol_hash']!=config_id(p):raise ValueError('Prepared study protocol changed')
    if canonical_json(manifest['probe_candidates'])!=canonical_json(probe_candidates(p)):
        raise ValueError('Prepared probe candidates changed')
    if manifest['hdc_candidates']!=[{'dimension':d} for d in p['hdc_dimensions']]:
        raise ValueError('Prepared HDC candidates changed')
    saved=json.loads((output/'environment.json').read_text());current=study_environment()
    if allow_smoke:
        saved.pop('code_revision',None);current.pop('code_revision',None)
    if saved!=current:raise ValueError('Study source/environment seal changed')
    names=p['datasets']
    if not names or len(names)!=len(set(names)) or manifest['datasets']!=names:
        raise ValueError('Prepared datasets differ from complete protocol panel')
    for name in names:
        X,y,data,splits=load_prepared(output,name)
        generated=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])
        if splits!=generated:raise ValueError('Prepared splits differ from complete generated fold schedule')
    jobs=expected_jobs(names,p)
    if json.loads((output/'planned_jobs.json').read_text())!=jobs:raise ValueError('Planned study jobs changed')
    return p,manifest,jobs


def study_worker(job):
    output,name,repeat,fold=job;output=Path(output)
    p=json.loads((output/'protocol.json').read_text())
    X,y,manifest,splits=load_prepared(output,name)
    split=next(s for s in splits if (s['outer_repeat'],s['outer_fold'])==(repeat,fold))
    stem=f'{name}__r{repeat}f{fold}';result_path=output/'results'/f'{stem}.json'
    log_path=output/'logs'/f'{stem}.jsonl';artifact_root=output/'artifacts'/stem
    if result_path.exists() or log_path.exists() or artifact_root.exists():
        raise FileExistsError(f'Existing study job artifacts: {stem}')
    log_path.parent.mkdir(parents=True,exist_ok=True);artifact_root.mkdir(parents=True)
    with log_path.open('x') as stream:
        def sink(row):stream.write(canonical_json(row)+'\n');stream.flush()
        result=evaluate_study_fold(X,y,split,p,dataset_id=name,dataset_hash=manifest['dataset_hash'],
            code_revision=json.loads((output/'environment.json').read_text())['code_revision'],native=name==SUSHI_ID,
            artifact_dir=artifact_root,sink=sink)
    write_json(result_path,result)
    return str(result_path)


def reconcile_job_records(result,events,p,job,split,X,y,artifact_root,code_revision):
    """Bind separate scientific records to logged events and prepared inputs.

    This checks provenance without fitting anything. A failed candidate may be
    retained in an otherwise complete inner matrix; a failed job is never usable
    for confirmation and is handled by the collector before this success check.
    """
    def require(condition,message):
        if not condition:raise ValueError(message)
    def records(stage):return [{k:v for k,v in row.items() if k!='stage'} for row in events if row['stage']==stage]
    require(records('inner_score')==result['inner_scores'],'inner scores disagree with logged events')
    require(records('outer_model')==result['models'],'outer models disagree with logged events')
    require(records('encoding')==[dict(row,status='ok') for row in result['partitions']],
            'partition metadata disagrees with encoding events')
    require({r['stage'] for r in events}<={'encoding','network_pair','hdc_bundle','inner_score','outer_model'},
            'unexpected event stage in successful job')
    native=job['dataset_id']==SUSHI_ID
    architectures={architecture_id(w):w for w in p['native_architectures' if native else 'architectures']}
    partitions={f'inner{i}':(i,s['train'],s['validation']) for i,s in enumerate(split['inner'])}
    partitions['outer']=(-1,split['train'],split['test'])
    require(Counter(r['partition_id'] for r in result['partitions'])==Counter({key:1 for key in partitions}),
            'incomplete partition schedule')
    metadata={r['partition_id']:r for r in result['partitions']}
    settings=encoder_settings(p,job['dataset_id'])
    for name,(inner,train,query) in partitions.items():
        meta=metadata[name]
        require(meta['inner_fold']==inner and meta['train_ids']==train and meta['query_ids']==query,
                'partition row IDs disagree with saved split')
        require(meta['raw_train_hash']==array_hash(X[train]) and meta['raw_query_hash']==array_hash(X[query])
                and meta['training_labels_hash']==array_hash(y[train]),'partition raw data/label hashes disagree')
        expected_seed=None if native else derive_seed(p['encoder_seed'],job['dataset_id'],job['outer_repeat'],job['outer_fold'],inner,
                                                       settings['degree'],p['strategy'],p['view_id'])
        require(meta['encoder_seed']==expected_seed,'partition encoder seed disagrees')
        with np.load(artifact_root/name/'shared.npz',allow_pickle=False) as arrays:
            require(np.array_equal(arrays['train_ids'],train) and np.array_equal(arrays['query_ids'],query),
                    'shared artifact row IDs disagree with saved split')
            require(meta['source_train_hash']==array_hash(arrays['train']) and meta['source_query_hash']==array_hash(arrays['query']),
                    'partition encoded hashes disagree with shared artifact')
            require(meta['encoder_parameters']=={'degree':settings['degree'],'strategy':p['strategy'],
                    'embed_dim':10 if native else settings['embed_dim'],'native':native},'partition encoder settings disagree')
            if native:
                require(np.array_equal(arrays['train'],X[train]) and np.array_equal(arrays['query'],X[query]),
                        'native shared orders disagree with saved data')
            encoder_keys=set(arrays.files)-{'train','query','train_ids','query_ids'}
            require(meta['encoder_array_hashes']=={key:array_hash(arrays[key]) for key in encoder_keys},
                    'fitted encoder hashes disagree with shared artifact')
    expected_inner=[]
    def key(kind,inner,seed,state,cfg,architecture=None,depth=None):
        return (kind,inner,seed,state,config_id(cfg),architecture,depth)
    for inner in range(p['inner_folds']):
        for arch,widths in architectures.items():
            for seed in p['fit_seeds']:
                expected_inner.append(key('output',inner,seed,'trained',{},arch))
                for depth in range(1,len(widths)+1):
                    for state in ('trained','untrained'):
                        expected_inner.extend(key('probe',inner,seed,state,cfg,arch,depth) for cfg in probe_candidates(p))
        expected_inner.extend(key('input',inner,p['fit_seeds'][0],'input',cfg) for cfg in probe_candidates(p))
        expected_inner.append(key('borda',inner,p['fit_seeds'][0],'borda',{}))
        expected_inner.extend(key('hdc',inner,seed,'hdc',{'dimension':d}) for d in p['hdc_dimensions'] for seed in p['fit_seeds'])
    observed_inner=[]
    for row in result['inner_scores']:
        require(row['config_id']==config_id(row['config']),'inner canonical config ID disagrees')
        require(row['partition_id']==f"inner{row['inner_fold']}",'inner score partition identity disagrees')
        require(row['status'] in ('ok','failed'),'invalid inner status')
        if row['status']=='ok':require(np.isfinite(row['score']) and 0<=row['score']<=1,'invalid inner accuracy')
        else:require(bool(row.get('exception')),'failed inner record omits reason')
        observed_inner.append(key(row['kind'],row['inner_fold'],row['model_seed'],row['state'],row['config'],
                                  row.get('architecture'),row.get('depth')))
    require(Counter(observed_inner)==Counter(expected_inner),'incomplete inner score schedule (output/Borda/probe/HDC/input)')
    networks=[r for r in events if r['stage']=='network_pair'];hdc=[r for r in events if r['stage']=='hdc_bundle']
    expected_networks=[(part,arch,seed) for part in partitions for arch in architectures for seed in p['fit_seeds']]
    expected_hdc=[(part,d,seed) for part in partitions
                  for d in (p['hdc_dimensions'] if part!='outer' else [result['selected']['hdc']['config']['dimension']])
                  for seed in p['fit_seeds']]
    require(Counter((r['partition_id'],r['architecture'],r['model_seed']) for r in networks)==Counter(expected_networks),
            'incomplete network event schedule')
    require(Counter((r['partition_id'],r['dimension'],r['model_seed']) for r in hdc)==Counter(expected_hdc),
            'incomplete HDC event schedule')
    for row in networks+hdc:
        meta=metadata[row['partition_id']]
        require(row['status']=='ok' and all(row[k]==meta[k] for k in ('inner_fold','source_train_hash','source_query_hash')),
                'fit event disagrees with shared partition')
        stream=(job['dataset_id'],job['outer_repeat'],job['outer_fold'],row['inner_fold'])
        if row['stage']=='network_pair':
            widths=architectures[row['architecture']]
            require(row['widths']==widths and row['network_seed']==derive_seed(row['model_seed'],'network',*stream,widths),
                    'network event derived seed/architecture disagrees')
        else:
            require(row['item_seed']==derive_seed(row['model_seed'],'hdc_item',*stream,row['dimension'])
                    and row['position_seed']==derive_seed(row['model_seed'],'hdc_position',*stream,row['dimension']),
                    'HDC event derived seeds disagree')
    fixed_configs={arch+'_output':{'architecture':widths,'iterations':p['iterations']} for arch,widths in architectures.items()}
    expected_configs=dict(fixed_configs,borda={},**{name:value['config'] for name,value in result['selected'].items()})
    model_rows={}
    for row in result['models']:
        cfg=expected_configs[row['model_id']]
        require(canonical_json(row['config'])==canonical_json(cfg) and row['config_id']==config_id(cfg),
                'model config disagrees with selected/fixed procedure')
        require(row['code_revision']==code_revision,'model code revision disagrees with source seal')
        require(all(row[k]==metadata['outer'][k] for k in ('source_train_hash','source_query_hash')),
                'model source hashes disagree with outer partition')
        require(row['training_sample_count']==len(split['train']),'model training count disagrees')
        if row['model_id']=='hdc':
            expected_event=next(e for e in hdc if (e['partition_id'],e['model_seed'],e['dimension'])==('outer',row['model_seed'],cfg['dimension']))
            require(row.get('fit_event')==expected_event,'HDC model fit event disagrees with log')
        elif row['model_id'] not in ('borda','input_footrule'):
            arch=next(arch for arch in architectures if row['model_id']==arch+'_output'
                      or any(row['model_id']==f'{arch}_d{depth}_{state}' for depth in range(1,len(architectures[arch])+1) for state in ('trained','untrained')))
            expected_event=next(e for e in networks if (e['partition_id'],e['architecture'],e['model_seed'])==('outer',arch,row['model_seed']))
            require(row.get('fit_event')==expected_event,'network model fit event disagrees with architecture/seed log')
            state_key='initial_hash' if row['model_id'].endswith('_untrained') else 'final_state_hash'
            require(row['final_state_hash']==expected_event[state_key],'model state hash disagrees with fit event')
        model_rows[(row['model_id'],row['model_seed'])]=row
    for row in result['predictions']:
        model=model_rows[(row['model_id'],row['model_seed'])]
        require(row['config_id']==model['config_id'],'prediction config disagrees with selected model')
        require(row['code_revision']==code_revision,'prediction code revision disagrees with source seal')


def collect_study_results(output,*,allow_smoke=False):
    """Complete job/log/artifact/prediction and model/depth/fold/seed reconciliation."""
    output=Path(output);p,manifest,jobs=verify_prepared(output,allow_smoke=allow_smoke)
    by_dataset={name:[] for name in manifest['datasets']};issues=[]
    prepared={name:load_prepared(output,name) for name in manifest['datasets']}
    code_revision=json.loads((output/'environment.json').read_text())['code_revision']
    for job in jobs:
        name=job['dataset_id'];stem=job['stem'];path=output/'results'/f'{stem}.json';log=output/'logs'/f'{stem}.jsonl'
        if not path.exists():issues.append(f'missing result {stem}')
        if not log.exists():issues.append(f'missing log {stem}')
        if not path.exists() or not log.exists():continue
        try:
            result=json.loads(path.read_text());events=[json.loads(line) for line in log.read_text().splitlines()]
            if events!=result['events']:issues.append(f'log/result mismatch {stem}')
            if result['status']!='ok':
                issues.append(f'failed study job {stem}');continue
            expected_artifacts=set()
            for partition in [f'inner{i}' for i in range(p['inner_folds'])]+['outer']:
                expected_artifacts.add(partition+'/shared.npz')
                for widths in p['native_architectures' if name==SUSHI_ID else 'architectures']:
                    for seed in p['fit_seeds']:
                        for state in ('initial','trained','representations'):
                            expected_artifacts.add(f'{partition}/{architecture_id(widths)}_{seed}_{state}.npz')
                dimensions=p['hdc_dimensions'] if partition!='outer' else [result['selected']['hdc']['config']['dimension']]
                for dimension in dimensions:
                    for seed in p['fit_seeds']:expected_artifacts.add(f'{partition}/hdc_{dimension}_{seed}.npz')
            paths=[a['path'] for a in result['artifacts']]
            if set(paths)!=expected_artifacts or len(paths)!=len(expected_artifacts):
                issues.append(f'incomplete artifact schedule {stem}')
            if result['selected']!=choose_configurations(result['inner_scores'],p,name==SUSHI_ID):
                issues.append(f'selection/inner-score mismatch {stem}')
            for artifact in result['artifacts']:
                asset=output/'artifacts'/stem/artifact['path']
                if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest()!=artifact['sha256']:
                    issues.append(f'missing/changed artifact {stem}/{artifact["path"]}')
            X,y,data,splits=prepared[name]
            split=next(s for s in splits if (s['outer_repeat'],s['outer_fold'])==(job['outer_repeat'],job['outer_fold']))
            reconcile_job_records(result,events,p,job,split,X,y,output/'artifacts'/stem,code_revision)
            models=result['models'];by_dataset[name].extend(models)
            for row in models:
                if any(row[k]!=job[k] for k in ('dataset_id','outer_repeat','outer_fold')) or row['dataset_hash']!=data['dataset_hash']:
                    issues.append(f'model identity mismatch {stem}')
            expected={(model,seed,sample) for model,seeds in job['model_seeds'].items() for seed in seeds for sample in split['test']}
            observed=[]
            for row in result['predictions']:
                observed.append((row['model_id'],row['model_seed'],row['sample_id']))
                if (row['dataset_id']!=name or row['dataset_hash']!=data['dataset_hash']
                    or (row['outer_repeat'],row['outer_fold'])!=(job['outer_repeat'],job['outer_fold'])
                    or row['sample_id'] not in split['test'] or row['y_true']!=y[row['sample_id']]):
                    issues.append(f'prediction identity mismatch {stem}')
            if set(observed)!=expected or len(observed)!=len(expected):issues.append(f'incomplete prediction matrix {stem}')
            for row in models:
                preds=[r for r in result['predictions'] if (r['model_id'],r['model_seed'])==(row['model_id'],row['model_seed'])]
                if preds and row.get('status')=='ok':
                    actual=metric_values([r['y_true'] for r in preds],[r['y_pred'] for r in preds])
                    if any(not np.isclose(actual[k],row[k],rtol=0,atol=1e-12) for k in actual):issues.append(f'metric/prediction mismatch {stem}')
        except (KeyError,ValueError,TypeError,IndexError) as exc:issues.append(f'invalid job {stem}: {exc}')
    folds=[(r,f) for r in range(p['outer_repeats']) for f in range(p['outer_folds'])]
    for name,rows in by_dataset.items():
        try:
            schedule=model_seed_schedule(p,name==SUSHI_ID)
            if any(r['model_id'] not in schedule for r in rows):raise ValueError('Unexpected study model')
            validate_outer_schedule(rows,expected_folds=folds,expected_seeds=schedule)
        except ValueError as exc:issues.append(f'{name}: {exc}')
    if issues:raise ValueError('Incomplete matched evidence: '+'; '.join(issues))
    return by_dataset


def run_prepared(output,workers=1,*,allow_smoke=False):
    p,manifest,jobs=verify_prepared(output,allow_smoke=allow_smoke)
    if not 1<=workers<=p['max_workers']:raise ValueError('Worker count exceeds prespecified limit')
    work=[(str(output),j['dataset_id'],j['outer_repeat'],j['outer_fold']) for j in jobs]
    with execution_lock(), ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in pool.map(study_worker,work):print(path,flush=True)
    return collect_study_results(output,allow_smoke=allow_smoke)


def summarize_study(output,*,allow_smoke=False):
    output=Path(output);rows=collect_study_results(output,allow_smoke=allow_smoke)
    p=json.loads((output/'protocol.json').read_text());manifest=json.loads((output/'run_manifest.json').read_text())
    folds=[(r,f) for r in range(p['outer_repeats']) for f in range(p['outer_folds'])]
    summaries={};contrasts=[]
    primary=p.get('primary_datasets',['iris','wine','breast_cancer','wine_quality','vehicle','segment','digits'])
    for name,records in rows.items():
        schedule=model_seed_schedule(p,name==SUSHI_ID)
        summaries[name]=[summarize_outer(records,model,metric,expected_folds=folds,expected_seeds=seeds)
                         for model,seeds in schedule.items() for metric in ('accuracy','error','balanced_accuracy','macro_f1')]
        if name in primary:
            widths=p['architectures'][p['primary_architecture_index']];arch=architecture_id(widths)
            pairs=[(f'{arch}_d{len(widths)}_trained',f'{arch}_d{len(widths)}_untrained'),(arch+'_output','input_footrule')]
            for a,b in pairs:
                q=p['test_train_ratio']
                contrasts.append(dict(dataset_id=name,model_a=a,model_b=b,
                    **paired_corrected_interval(records,a,b,q=q,expected_folds=folds,expected_seeds={a:schedule[a],b:schedule[b]})))
    complete_primary=set(primary)<=rows.keys() and len(contrasts)==p['primary_tabular_comparisons']
    if complete_primary:
        for row,adjusted in zip(contrasts,holm_adjust([r['p_approximate'] for r in contrasts])):row['holm_p_approximate']=adjusted
    return {'purpose':manifest['purpose'],'summaries':summaries,'primary_contrasts':contrasts,
            'complete_primary_family':complete_primary,'holm_applied':complete_primary}


def shuffle_prerequisite(output,p):
    output=Path(output);X,y,manifest=load_dataset('iris')
    split=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])[0]
    indices=split['inner'][0];train=indices['train'];query=indices['validation']
    shuffled=np.random.RandomState(p['shuffle_seed']).permutation(y[train])
    fixed=dict(p,architectures=[p['architectures'][p['primary_architecture_index']]],fit_seeds=p['fit_seeds'][:1],hdc_dimensions=[])
    writer=ArtifactWriter(output/'shuffle_artifacts')
    cache=build_partition(X[train],shuffled,X[query],train,query,dataset_id='iris',outer_repeat=0,outer_fold=0,
                          inner_fold=0,protocol=fixed,writer=writer)
    if cache['failures']:raise RuntimeError('Shuffle prerequisite fitting failed')
    prediction=next(iter(cache['networks'].values()))['output']
    majority=np.unique(shuffled,return_counts=True);dummy=np.full(len(query),majority[0][np.argmax(majority[1])])
    records=[];metrics={}
    for model,pred in [('shuffle_arrowflow',prediction),('shuffle_dummy',dummy)]:
        metrics[model]=metric_values(y[query],pred)
        records.extend({'sample_id':int(sample),'y_true':int(y[sample]),'y_pred':int(label),'model_id':model} for sample,label in zip(query,pred))
    report={'purpose':'inner_label_shuffle_prerequisite_not_outer_evidence','dataset_hash':manifest['dataset_hash'],
            'split_hash':config_id(split),'train_ids':train,'query_ids':query,'shuffle_seed':p['shuffle_seed'],
            'model_seed':p['fit_seeds'][0],'original_labels_hash':array_hash(y[train]),'shuffled_labels_hash':array_hash(shuffled),
            'validation_labels_hash':array_hash(y[query]),'training_class_counts':np.unique(shuffled,return_counts=True)[1].tolist(),
            'prediction_hash':array_hash(prediction),'predictions':records,'metrics':metrics,
            'partition':cache['metadata'],'events':cache['events'],'artifacts':writer.files,'environment':study_environment()}
    write_json(output/'shuffle.json',report)
    return report


def runtime_pilot(output,names,p):
    output=Path(output);prepare_study(output,names,p,purpose='training_runtime_only')
    rows=[]
    for name in names:
        X,y,data,splits=load_prepared(output,name);train=splits[0]['train'];query=train[::4]
        start=time.perf_counter()
        cache=build_partition(X[train],y[train],X[query],train,query,dataset_id=name,outer_repeat=0,outer_fold=0,
            inner_fold=-1,protocol=p,native=name==SUSHI_ID,writer=ArtifactWriter(output/'pilot_artifacts'/name))
        cache_seconds=time.perf_counter()-start
        start=time.perf_counter();probes=score_inner_partition(cache,y[train],None,p)
        probe_seconds=time.perf_counter()-start
        row={'dataset_id':name,'dataset_hash':data['dataset_hash'],'partition':cache['metadata'],
             'cache_seconds':cache_seconds,'probe_grid_seconds':probe_seconds,'events':cache['events'],
             'probe_timings':probes,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'status':'failed' if cache['failures'] or any(r['status']!='ok' for r in probes) else 'ok'}
        rows.append(row);print(canonical_json({k:row[k] for k in ['dataset_id','cache_seconds','probe_grid_seconds','status']}),flush=True)
        del cache
    report={'purpose':'training_only_runtime_no_outer_scores','rows':rows,
            'network_fit_count_tabular':7*p['outer_folds']*p['outer_repeats']*(p['inner_folds']+1)*len(p['architectures'])*len(p['fit_seeds']),
            'network_fit_count_native':p['outer_folds']*p['outer_repeats']*(p['inner_folds']+1)*len(p['native_architectures'])*len(p['fit_seeds'])}
    write_json(output/'pilot.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','smoke','pilot','shuffle','run','summary'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--protocol',type=Path,default=STUDY_PROTOCOL)
    parser.add_argument('--dataset',nargs='+');parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--sushi-archive',type=Path)
    args=parser.parse_args(argv);p=json.loads(args.protocol.read_text())
    if args.sushi_archive:os.environ['ARROWFLOW_SUSHI_ARCHIVE']=str(args.sushi_archive)
    if args.command=='run':
        if not p['frozen']:raise ValueError('Confirmation requires a frozen reviewed study protocol')
        if json.loads((args.output/'protocol.json').read_text())!=p:raise ValueError('Prepared protocol differs')
        run_prepared(args.output,args.workers)
    elif args.command=='summary':write_json(args.output/'summary.json',summarize_study(args.output))
    elif args.command=='prepare':prepare_study(args.output,args.dataset or p['datasets'],p)
    elif args.command=='pilot':
        with execution_lock():runtime_pilot(args.output,args.dataset or ['iris','digits',SUSHI_ID],p)
    elif args.command=='shuffle':
        with execution_lock():shuffle_prerequisite(args.output,p)
    else:
        tiny=dict(p,outer_folds=3,outer_repeats=1,inner_folds=2,architectures=[[4],[4,3],[8]],
                  embed_dim=6,iterations=4,hdc_dimensions=[8,16],datasets=['synthetic'],frozen=False,
                  encoder_by_dataset={'synthetic':{'embed_dim':6,'degree':p['degree']}})  # the table must name the smoke panel
        rng=np.random.RandomState(33);X=rng.randn(60,4);y=np.tile([0,1,2],20)
        features=[f'x{i}' for i in range(4)];labels=['0','1','2']
        manifest={'dataset_id':'synthetic','feature_names':features,'label_map':labels,'shape':[60,4],
                  'class_counts':[20,20,20],'dataset_hash':dataset_fingerprint(X,y,features,labels)}
        prepare_study(args.output,['synthetic'],tiny,purpose='synthetic_smoke_only',data_loader=lambda name:(X,y,manifest))
        run_prepared(args.output,args.workers,allow_smoke=True)
        write_json(args.output/'smoke_summary.json',summarize_study(args.output,allow_smoke=True))


if __name__=='__main__':main()
