"""Sealed E05/E06/E07 execution and separate sequential E11 cost command."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import pickle
from pathlib import Path
import resource
import time
import numpy as np
from .evaluation import canonical_json,config_id,dataset_fingerprint,make_splits,metric_values,summarize_outer,paired_corrected_interval
from .models import array_hash
from .run_revision import load_dataset,load_prepared,write_json,environment_record,execution_lock
from .matched import ArtifactWriter,probe_candidates,select_symmetric_probe
from .secondary_inputs import degree_schedule
from . import secondary_studies as study

PROTOCOL=Path(__file__).with_name('secondary_protocol.json')
SOURCE_MODULES=['experiments.make_revision.secondary_studies','experiments.make_revision.secondary_inputs',
    'experiments.make_revision.run_studies','experiments.make_revision.run_revision','experiments.make_revision.matched',
    'experiments.make_revision.models','experiments.make_revision.comparisons','experiments.make_revision.evaluation',
    'experiments.make_revision.datasets','experiments.make_revision.view_reporting']


def environment():
    result=environment_record(__package__+'.run_secondary:environment');root=Path(__file__).resolve().parents[2]
    for path in ['experiments/make_revision/protocol.json','experiments/make_revision/study_protocol.json',
                 'experiments/make_revision/secondary_protocol.json','manuscript/MAKE/review/secondary_experiment_spec.md']:
        result['source_hashes'][path]=hashlib.sha256((root/path).read_bytes()).hexdigest()
    return result


def case_schedule(p,family):
    # Case IDs and stream semantics come directly from the reviewed bank primitive.
    cases=study.make_bank(np.zeros((2,2)),np.zeros((1,2)),p,'schedule',0,0).cases
    if family=='e07':cases=[c for c in cases if c.family=='clean' or c.family in p['crossed_families'] and c.severity in p['crossed_levels']]
    return [{'case_id':c.case_id,'corruption_family':c.family,'severity':c.severity,'corruption_seed':c.base_seed} for c in cases]


def record_schedule(family,p,n_features):
    rows=[]
    if family=='e05':
        for scheme in p['e05_schemes']:
            for seed in p['fit_seeds']:
                for head in p['head_conditions']:
                    for kind,values in [('view',range(p['e05_views'])),('prefix',p['e05_prefixes'])]:
                        for view in values:
                            model=f'{scheme}__h{int(head)}__{kind}{view}'
                            rows.append(dict(record_id=f'{model}__s{seed}',model_id=model,model_seed=seed,
                                scheme=scheme,head_update=head,output_kind=kind,view=view,case_id='clean',corruption_family='clean',severity=0.,corruption_seed=None))
    else:
        if family=='e06':
            seeds={name:p['fit_seeds'] if spec.stochastic else p['fit_seeds'][:1] for name,spec in study.e06_registry(p).items()}
            seeds.update(row_af=p['fit_seeds'],projected_af=p['fit_seeds'])
        elif family=='e07':
            seeds={}
            for d in degree_schedule(n_features,p['degrees'],p['degree_column_cap']):
                if d['eligible']:
                    seeds[f'd{d["degree"]}_af']=p['fit_seeds'];seeds[f'd{d["degree"]}_footrule']=p['fit_seeds'][:1]
        else:raise ValueError('Unknown secondary family')
        for model,values in seeds.items():
            for seed in values:
                for case in case_schedule(p,family):
                    rows.append(dict(record_id=f'{model}__s{seed}__{case["case_id"]}',model_id=model,model_seed=seed,**case))
    return rows


def protocol_datasets(p):
    families=p['families']
    if not families or len(families)!=len(set(families)) or any(f not in ('e05','e06','e07') for f in families):
        raise ValueError('Invalid secondary family schedule')
    for family in families:
        panel=p['panels'][family]
        if not panel or len(panel)!=len(set(panel)):raise ValueError('Empty or duplicate protocol panel')
    return list(dict.fromkeys(name for family in families for name in p['panels'][family]))


def prepare(output,p,*,purpose='confirmatory',loader=None):
    output=Path(output);loader=loader or load_dataset
    names=protocol_datasets(p)
    write_json(output/'protocol.json',p);write_json(output/'environment.json',environment())
    write_json(output/'manifest.json',{'purpose':purpose,'protocol_hash':config_id(p),'datasets':names,
        'e06_candidates':{k:v.candidates for k,v in study.e06_registry(p).items()},'e07_candidates':probe_candidates(p)})
    jobs=[]
    for name in names:
        X,y,data=loader(name);splits=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])
        write_json(output/name/'manifest.json',dict(data,splits_hash=config_id(splits)));write_json(output/name/'splits.json',splits)
        if not (output/name/'data.npz').exists():np.savez_compressed(output/name/'data.npz',X=X,y=y)
        for family in p['families']:
            if name not in p['panels'][family]:continue
            schedule=record_schedule(family,p,X.shape[1])
            for split in splits:
                jobs.append(dict(family=family,dataset_id=name,outer_repeat=split['outer_repeat'],outer_fold=split['outer_fold'],
                    stem=f'{family}__{name}__r{split["outer_repeat"]}f{split["outer_fold"]}',records=schedule))
    write_json(output/'planned_jobs.json',jobs)
    return jobs


def verify(output,allow_smoke=False):
    output=Path(output);p=json.loads((output/'protocol.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
    if not p['frozen'] and not (allow_smoke and manifest['purpose']=='synthetic_smoke_only'):raise ValueError('A frozen reviewed protocol is required')
    if manifest['protocol_hash']!=config_id(p):raise ValueError('Protocol seal changed')
    if canonical_json(manifest['e06_candidates'])!=canonical_json({k:v.candidates for k,v in study.e06_registry(p).items()}) or manifest['e07_candidates']!=probe_candidates(p):
        raise ValueError('Candidate seal changed')
    saved=json.loads((output/'environment.json').read_text());current=environment()
    if allow_smoke:saved.pop('code_revision',None);current.pop('code_revision',None)
    if saved!=current:raise ValueError('Source/environment seal changed')
    names=protocol_datasets(p)
    if manifest['datasets']!=names:raise ValueError('Prepared datasets differ from complete protocol panels')
    expected=[]
    for name in names:
        X,y,data,splits=load_prepared(output,name)
        generated=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])
        if splits!=generated:raise ValueError('Prepared splits differ from complete generated fold schedule')
        for family in p['families']:
            if name not in p['panels'][family]:continue
            for split in splits:
                expected.append(dict(family=family,dataset_id=name,outer_repeat=split['outer_repeat'],outer_fold=split['outer_fold'],
                    stem=f'{family}__{name}__r{split["outer_repeat"]}f{split["outer_fold"]}',records=record_schedule(family,p,X.shape[1])))
    jobs=json.loads((output/'planned_jobs.json').read_text())
    if jobs!=expected:raise ValueError('Complete planned job/condition schedule changed')
    return p,manifest,jobs


def evaluate_job(X,y,split,p,job,*,dataset_hash,code_revision,artifact_root,sink=None):
    writer=ArtifactWriter(artifact_root);family=job['family'];name=job['dataset_id'];train=split['train'];query=split['test']
    result={'status':'running','fits':[],'selections':{},'partitions':[],'rows':[],'predictions':[],'diagnostics':[],'events':[],
        'provenance':dict(dataset_id=name,dataset_hash=dataset_hash,code_revision=code_revision,split_hash=config_id(split),
            family=family,outer_repeat=split['outer_repeat'],outer_fold=split['outer_fold'],train_ids=train,query_ids=query,
            raw_train_hash=array_hash(X[train]),raw_query_hash=array_hash(X[query]),training_labels_hash=array_hash(y[train]))}
    def emit(stage,payload):
        event={'stage':stage,'record':payload};result['events'].append(event)
        if sink:sink(event)
    emit('provenance',result['provenance']);prediction_arrays={}
    try:
        if family=='e05':
            learned=study.fit_views(X[train],y[train],X[query],p,dataset_id=name,repeat=job['outer_repeat'],fold=job['outer_fold'],writer=writer)
            result['fits']=learned['fits']
            for fit in result['fits']:emit('fit',fit)
            if any(r['status']!='ok' for r in result['fits']):raise RuntimeError('View fitting failed')
            for scheme in p['e05_schemes']:
                for seed in p['fit_seeds']:
                    for head in p['head_conditions']:
                        matrix=np.stack([learned['outputs'][(scheme,seed,head,view)] for view in range(p['e05_views'])])
                        diag=dict(scheme=scheme,model_seed=seed,head_update=head,pairs=study.error_diagnostics(matrix,y[query]),prefixes=[])
                        for prefix in p['e05_prefixes']:
                            first=float(np.mean(matrix[0]!=y[query]));mean=float(np.mean(matrix[:prefix]!=y[query]));error=float(np.mean(study.majority(matrix[:prefix])!=y[query]))
                            diag['prefixes'].append(dict(prefix=prefix,first_view_error=first,mean_single_view_error=mean,ensemble_error=error,
                                change_from_first=error-first,change_from_mean_single=error-mean))
                        result['diagnostics'].append(diag);emit('diagnostic',diag)
            for spec in job['records']:
                key=(spec['scheme'],spec['model_seed'],spec['head_update']);view=spec['view']
                pred=learned['outputs'][(*key,view)] if spec['output_kind']=='view' else study.majority([learned['outputs'][(*key,v)] for v in range(view)])
                cfg=dict(scheme=spec['scheme'],head_update=spec['head_update'],kind=spec['output_kind'],view=view,
                         widths=p['architectures'][p['primary_architecture_index']],iterations=p['iterations'])
                prediction_arrays[spec['record_id']]=pred
                result['rows'].append(dict(spec,config=cfg,config_id=config_id(cfg),status='ok'))
        else:
            learned=(study.fit_corruption_models if family=='e06' else study.fit_degrees)(X,y,split,p,dataset_id=name,writer=writer)
            for field,stage in [('fits','fit'),('partitions','partition')]:
                result[field]=learned[field]
                for record in result[field]:emit(stage,record)
            result['selections']=learned['selections']
            for model,selected in result['selections'].items():emit('selection',{'model_id':model,'selection':selected})
            if family=='e07':result['degree_schedule']=learned['degree_schedule'];emit('degree_schedule',learned['degree_schedule'])
            if learned['failures']:raise RuntimeError(canonical_json(learned['failures']))
            bank=study.make_bank(X[train],X[query],p,name,job['outer_repeat'],job['outer_fold']);bank.save(Path(artifact_root)/'corruptions')
            for file in ('arrays.npz','manifest.json'):
                asset=Path(artifact_root)/'corruptions'/file;writer.files.append({'path':'corruptions/'+file,'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()})
            result['corruption_manifest']=bank.metadata();emit('corruptions',result['corruption_manifest'])
            cases={c.case_id:c for c in bank.cases};fit_map={(r['model_id'],r['model_seed']):r for r in result['fits']}
            states={key:study.fitted_hash(model) for key,model in learned['models'].items()}
            for spec in job['records']:
                case=cases[spec['case_id']];key=(spec['model_id'],spec['model_seed']);model=learned['models'][key];fit=fit_map[key]
                if array_hash(case.raw)!=case.raw_hash:raise ValueError('Shared corruption changed')
                start=time.perf_counter();pred=model.predict(case.raw);elapsed=time.perf_counter()-start
                if study.fitted_hash(model)!=states[key]:raise ValueError('Frozen model/encoder changed during query inference')
                prediction_arrays[spec['record_id']]=pred
                result['rows'].append(dict(spec,config=fit['config'],config_id=fit['config_id'],state_hash=states[key],
                    raw_query_hash=case.raw_hash,inference_seconds=elapsed,query_encoding_seconds=getattr(model,'last_encoding_seconds_',None),status='ok'))
            bank.assert_intact()
        for row in result['rows']:
            pred=np.asarray(prediction_arrays[row['record_id']]);row.update(dataset_id=name,dataset_hash=dataset_hash,code_revision=code_revision,
                outer_repeat=job['outer_repeat'],outer_fold=job['outer_fold'],prediction_hash=array_hash(pred),**metric_values(y[query],pred))
            prediction=dict(record_id=row['record_id'],model_id=row['model_id'],model_seed=row['model_seed'],config_id=row['config_id'],code_revision=code_revision,
                sample_ids=query,y_true=y[query].tolist(),y_pred=pred.tolist())
            result['predictions'].append(prediction);emit('row',row)
        writer.save('predictions.npz',dict(prediction_arrays,query_ids=np.asarray(query),truth=y[query]))
        result['status']='ok'
    except Exception as exc:
        result['status']='failed';result['exception']=f'{type(exc).__name__}: {exc}';emit('terminal_failure',{'exception':result['exception']})
        present={r['record_id'] for r in result['rows']}
        for spec in job['records']:
            if spec['record_id'] not in present:
                row=dict(spec,status='failed',exception=result['exception']);result['rows'].append(row);emit('row',row)
    result['artifacts']=writer.files
    return result


def worker(arguments):
    output,job=arguments;output=Path(output);stem=job['stem'];log=output/'logs'/(stem+'.jsonl');result_path=output/'results'/(stem+'.json');root=output/'artifacts'/stem
    if log.exists() or result_path.exists() or root.exists():raise FileExistsError('Existing secondary job '+stem)
    log.parent.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True)
    X,y,data,splits=load_prepared(output,job['dataset_id']);split=next(s for s in splits if (s['outer_repeat'],s['outer_fold'])==(job['outer_repeat'],job['outer_fold']))
    p=json.loads((output/'protocol.json').read_text());revision=json.loads((output/'environment.json').read_text())['code_revision']
    with log.open('x') as stream:
        def sink(event):stream.write(canonical_json(event)+'\n');stream.flush()
        result=evaluate_job(X,y,split,p,job,dataset_hash=data['dataset_hash'],code_revision=revision,artifact_root=root,sink=sink)
    write_json(result_path,result);return str(result_path)


def validate_selection(selected,spec,p,split):
    from .reporting import _selected_from_history
    _selected_from_history(selected,spec,split,p)


def validate_job(result,events,job,p,X,y,split,data,revision,root):
    def require(value,message):
        if not value:raise ValueError(message)
    def logged(stage):return [e['record'] for e in events if e['stage']==stage]
    require(events==result['events'],'events differ from log')
    require(result['status']=='ok','failed terminal job')
    provenance=dict(dataset_id=job['dataset_id'],dataset_hash=data['dataset_hash'],code_revision=revision,split_hash=config_id(split),
        family=job['family'],outer_repeat=job['outer_repeat'],outer_fold=job['outer_fold'],train_ids=split['train'],query_ids=split['test'],
        raw_train_hash=array_hash(X[split['train']]),raw_query_hash=array_hash(X[split['test']]),training_labels_hash=array_hash(y[split['train']]))
    require(result['provenance']==provenance and logged('provenance')==[provenance],'job provenance')
    for field,stage in [('fits','fit'),('partitions','partition'),('rows','row'),('diagnostics','diagnostic')]:
        require(result[field]==logged(stage),field+' differs from log')
    require(sorted(logged('selection'),key=lambda r:r['model_id'])==[{'model_id':name,'selection':s} for name,s in sorted(result['selections'].items())],'selection differs from log')
    require({e['stage'] for e in events}<={'provenance','fit','partition','row','diagnostic','selection','degree_schedule','corruptions'},'unexpected event stage')
    expected={r['record_id']:r for r in job['records']};rows=result['rows'];predictions=result['predictions']
    require(Counter(r['record_id'] for r in rows)==Counter({key:1 for key in expected}),'incomplete condition/view/degree/seed rows')
    require(Counter(r['record_id'] for r in predictions)==Counter({key:1 for key in expected}),'incomplete predictions')
    family=job['family'];expected_artifacts={'predictions.npz'};expected_fits=set();fit_map={};partition_specs={}
    if family=='e05':
        require(not result['selections'] and not result['partitions'],'unexpected E05 selection/partition')
        for scheme in p['e05_schemes']:
            for seed in p['fit_seeds']:
                for head in p['head_conditions']:
                    for view in range(p['e05_views']):
                        fit_id=f'{scheme}__s{seed}__h{int(head)}__v{view}';expected_fits.add(fit_id)
                        expected_artifacts.update(f'views/{fit_id}_{suffix}.npz' for suffix in ('initial','trained','encoding'))
        pairs={}
        for fit in result['fits']:
            stream=study.view_stream(p,job['dataset_id'],job['outer_repeat'],job['outer_fold'],fit['model_seed'],fit['scheme'],fit['view'])
            require(all(fit[k]==v for k,v in stream.items()),'view seed stream changed')
            require(fit['head_update'] or fit['head_before_hash']==fit['head_after_hash'],'frozen head changed')
            pair=(fit['scheme'],fit['model_seed'],fit['view'])
            if pair in pairs:require(pairs[pair]==fit['initial_state_hash'],'head initial states differ')
            pairs[pair]=fit['initial_state_hash']
            with np.load(root/f'views/{fit["fit_id"]}_encoding.npz',allow_pickle=False) as arrays:
                require(array_hash(arrays['train'])==fit['train_encoding_hash'] and array_hash(arrays['query'])==fit['query_encoding_hash'],'view encoding hash')
                require(study.fitted_hash(pickle.loads(arrays['serialized_encoder'].tobytes()))==fit['encoder_hash'],'view encoder state hash')
                encoder=pickle.loads(arrays['serialized_encoder'].tobytes())
                require(encoder.seed==fit['encoder_seed'] and encoder.strategy==fit['strategy'],'view encoder seed/settings')
                require(np.array_equal(encoder.transform(X[split['train']]),arrays['train']) and np.array_equal(encoder.transform(X[split['test']]),arrays['query']),'view arrays disagree with source rows')
            last=len(p['architectures'][p['primary_architecture_index']])
            for suffix,field in [('initial','initial_state_hash'),('trained','final_state_hash')]:
                with np.load(root/f'views/{fit["fit_id"]}_{suffix}.npz',allow_pickle=False) as arrays:
                    digest=hashlib.sha256()
                    for key in sorted(arrays.files):digest.update(key.encode());digest.update(array_hash(arrays[key]).encode())
                    expected_hash=digest.hexdigest() if suffix=='initial' else config_id({'network':digest.hexdigest(),'encoder':None})
                    require(fit[field]==expected_hash,'view snapshot state hash')
                    head_hash=config_id({key:array_hash(arrays[key]) for key in arrays.files if key.startswith(f'layer_{last}_')})
                    require(fit['head_before_hash' if suffix=='initial' else 'head_after_hash']==head_hash,'view output head snapshot hash')
                    if suffix=='trained':
                        with np.load(root/f'views/{fit["fit_id"]}_encoding.npz',allow_pickle=False) as shared:
                            require(fit['layers']==study.diagnostics_from_layers([arrays[f'layer_{i}_positions'] for i in range(last)],shared['query']),
                                    'prototype/response diagnostics differ from frozen arrays')

        for seed in p['fit_seeds']:
            for head in p['head_conditions']:
                first=[r for r in result['fits'] if r['model_seed']==seed and r['head_update']==head and r['view']==0]
                require(len({(r['initial_state_hash'],r['encoder_hash'],r['train_encoding_hash'],r['query_encoding_hash']) for r in first})==1,'first views differ')
    else:
        if family=='e06':
            registry=study.e06_registry(p);require(set(result['selections'])==set(registry),'missing conventional selections')
            for name,spec in registry.items():validate_selection(result['selections'][name],spec,p,split)
            partition_specs={name:(split['train'],split['test']) for name in ('row_af','projected_af')}
        else:
            degrees=degree_schedule(X.shape[1],p['degrees'],p['degree_column_cap'])
            require(result['degree_schedule']==degrees and logged('degree_schedule')==[degrees],'degree exclusion schedule')
            active=[r['degree'] for r in degrees if r['eligible']]
            require(set(result['selections'])=={f'd{d}_footrule' for d in active},'missing degree selections')
            for d in active:
                selected=result['selections'][f'd{d}_footrule'];actual=select_symmetric_probe(selected['fits'],probe_candidates(p),
                    inner_folds=p['inner_folds'],seeds=p['fit_seeds'][:1],states=('input',))
                require(all(r['config_id']==config_id(r['config']) and r['partition_id']==f'd{d}_inner{r["inner_fold"]}' for r in selected['fits']),'degree inner config/partition identity')
                require(all(selected[k]==v for k,v in actual.items()),'degree selection disagreement')
                partition_specs[f'd{d}_outer']=(split['train'],split['test'])
                for i,indices in enumerate(split['inner']):partition_specs[f'd{d}_inner{i}']=(indices['train'],indices['validation'])
        require(Counter(r['partition_id'] for r in result['partitions'])==Counter({name:1 for name in partition_specs}),'incomplete encoder partition schedule')
        for part in result['partitions']:
            train,query=partition_specs[part['partition_id']]
            require(part['train_ids']==train and part['query_ids']==query,'encoder partition row IDs')
            require(part['raw_train_hash']==array_hash(X[train]) and part['raw_query_hash']==array_hash(X[query])
                    and part['training_labels_hash']==array_hash(y[train]),'encoder partition raw hashes')
            path=f'encoders/{part["partition_id"]}.npz';expected_artifacts.add(path)
            with np.load(root/path,allow_pickle=False) as arrays:
                require(np.array_equal(arrays['train_ids'],train) and np.array_equal(arrays['query_ids'],query),'encoder artifact row IDs')
                require(array_hash(arrays['train'])==part['train_encoding_hash'] and array_hash(arrays['query'])==part['query_encoding_hash'],'encoder array hashes')
                require(study.fitted_hash(pickle.loads(arrays['serialized_encoder'].tobytes()))==part['encoder_hash'],'fitted encoder artifact hash')
                encoder=pickle.loads(arrays['serialized_encoder'].tobytes())
                if family=='e07':
                    degree=int(part['partition_id'].split('_')[0][1:]);inner=-1 if part['partition_id'].endswith('outer') else int(part['partition_id'].split('inner')[1])
                    expected_seed=study.encoder_seed(p,job['dataset_id'],split,inner)
                    require(part['degree']==degree and part['inner_fold']==inner and part['encoder_seed']==expected_seed
                            and encoder.seed==expected_seed and encoder.degree==degree,'degree partition/encoder seed identity')
                elif part['partition_id']=='projected_af':
                    require(encoder.seed==study.encoder_seed(p,job['dataset_id'],split) and encoder.strategy=='random' and encoder.degree==2,'projected AF encoder seed/settings')
                require(np.array_equal(encoder.transform(X[train]),arrays['train']) and np.array_equal(encoder.transform(X[query]),arrays['query']),'encoder arrays disagree with source rows')
        for spec in job['records']:
            expected_fits.add(f'{spec["model_id"]}__s{spec["model_seed"]}')
        for fit_id in expected_fits:
            expected_artifacts.add(f'models/{fit_id}.npz')
        bank=study.make_bank(X[split['train']],X[split['test']],p,job['dataset_id'],job['outer_repeat'],job['outer_fold'])
        metadata=bank.metadata()
        require(result['corruption_manifest']==metadata and logged('corruptions')==[metadata],'corruption bank provenance')
        require(json.loads((root/'corruptions/manifest.json').read_text())==metadata,'saved corruption metadata')
        with np.load(root/'corruptions/arrays.npz',allow_pickle=False) as arrays:
            for item in metadata['cases']:require(array_hash(arrays[item['array_key']])==item['raw_hash'],'raw corruption artifact hash')
            for key,value in metadata['state_hashes'].items():require(array_hash(arrays[key])==value,'corruption draw/statistic artifact hash')
        expected_artifacts.update(('corruptions/arrays.npz','corruptions/manifest.json'))
        cases={c.case_id:c for c in bank.cases}
    require(Counter(r['fit_id'] for r in result['fits'])==Counter({key:1 for key in expected_fits}),'incomplete fit seed/view/degree schedule')
    require(all(f['status']=='ok' for f in result['fits']),'failed fit')
    for fit in result['fits']:
        if family!='e05':
            model=fit['model_id'];cfg=fit['config']
            if model in result['selections']:require(cfg==result['selections'][model]['config'],'fit differs from selected configuration')
            elif family=='e06':require(cfg=={'widths':p['architectures'][p['primary_architecture_index']],'iterations':p['iterations']},'fixed AF configuration')
            else:require(cfg=={'degree':int(model.split('_')[0][1:]),'widths':p['architectures'][p['primary_architecture_index']],'iterations':p['iterations']},'fixed degree AF configuration')
            require(fit['config_id']==config_id(cfg),'fit canonical config ID');fit_map[(model,fit['model_seed'])]=fit
            with np.load(root/f'models/{fit["fit_id"]}.npz',allow_pickle=False) as arrays:
                if 'serialized_model' in arrays.files:
                    fitted=pickle.loads(arrays['serialized_model'].tobytes())
                    require(study.fitted_hash(fitted)==fit['state_hash'],'serialized fitted state differs from log')
                else:
                    digest=hashlib.sha256()
                    for key in sorted(arrays.files):digest.update(key.encode());digest.update(array_hash(arrays[key]).encode())
                    partition_id=model if family=='e06' else model.split('_')[0]+'_outer'
                    encoder=next(r['encoder_hash'] for r in result['partitions'] if r['partition_id']==partition_id)
                    require(config_id({'network':digest.hexdigest(),'encoder':encoder})==fit['state_hash'],'saved network/encoder state differs from log')
                    require(fit['network_seed']==study.model_seed(p,job['dataset_id'],split,fit['model_seed']),'fixed network derived seed differs')
    require(Counter(a['path'] for a in result['artifacts'])==Counter({path:1 for path in expected_artifacts}),'incomplete artifact schedule')
    for asset in result['artifacts']:
        path=root/asset['path'];require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256'],'changed artifact '+asset['path'])
    pred_map={r['record_id']:r for r in predictions}
    with np.load(root/'predictions.npz',allow_pickle=False) as arrays:
        require(set(arrays.files)==set(expected)|{'query_ids','truth'},'prediction artifact schedule')
        require(np.array_equal(arrays['query_ids'],split['test']) and np.array_equal(arrays['truth'],y[split['test']]),'prediction artifact row IDs/truth')
        for row in rows:
            require(all(row[k]==v for k,v in expected[row['record_id']].items()),'condition identity')
            require(row['status']=='ok' and row['code_revision']==revision and row['dataset_hash']==data['dataset_hash']
                    and row['dataset_id']==job['dataset_id'] and (row['outer_repeat'],row['outer_fold'])==(job['outer_repeat'],job['outer_fold']),'row source/fold identity')
            if family=='e05':
                cfg=dict(scheme=row['scheme'],head_update=row['head_update'],kind=row['output_kind'],view=row['view'],
                         widths=p['architectures'][p['primary_architecture_index']],iterations=p['iterations'])
            else:
                fit=fit_map[(row['model_id'],row['model_seed'])];cfg=fit['config']
                require(row['state_hash']==fit['state_hash'] and row['raw_query_hash']==cases[row['case_id']].raw_hash,'frozen fit/raw query identity')
            require(row['config']==cfg and row['config_id']==config_id(cfg),'row selected/fixed config identity')
            pred=pred_map[row['record_id']];actual=arrays[row['record_id']]
            if family=='e05' and row['output_kind']=='view':
                matches=[fit for fit in result['fits'] if (fit['scheme'],fit['model_seed'],fit['head_update'],fit['view'])
                         ==(row['scheme'],row['model_seed'],row['head_update'],row['view'])]
                require(len(matches)==1 and matches[0]['prediction_hash']==array_hash(actual),'view fit prediction hash differs from saved individual output')
            require(pred['sample_ids']==split['test'] and pred['y_true']==y[split['test']].tolist()
                    and np.array_equal(pred['y_pred'],actual) and row['prediction_hash']==array_hash(actual),'prediction array/row/hash identity')
            require(all(pred[k]==row[k] for k in ('model_id','model_seed','config_id','code_revision')),'prediction config/source identity')
            metrics=metric_values(y[split['test']],actual)
            require(all(np.isclose(row[k],v,rtol=0,atol=1e-12) for k,v in metrics.items()),'prediction/metric disagreement')
    if family=='e05':
        groups=[(scheme,seed,head) for scheme in p['e05_schemes'] for seed in p['fit_seeds'] for head in p['head_conditions']]
        require(Counter((d['scheme'],d['model_seed'],d['head_update']) for d in result['diagnostics'])==Counter(groups),'missing joint diagnostics')
        for diag in result['diagnostics']:
            scheme,seed,head=diag['scheme'],diag['model_seed'],diag['head_update']
            matrix=np.array([pred_map[f'{scheme}__h{int(head)}__view{view}__s{seed}']['y_pred'] for view in range(p['e05_views'])])
            require(diag['pairs']==study.error_diagnostics(matrix,y[split['test']]),'joint-error diagnostics disagree')
            require([d['prefix'] for d in diag['prefixes']]==p['e05_prefixes'],'prefix diagnostic schedule')
            for prefix in p['e05_prefixes']:
                pred=pred_map[f'{scheme}__h{int(head)}__prefix{prefix}__s{seed}']['y_pred']
                require(np.array_equal(pred,study.majority(matrix[:prefix])),'ensemble majority differs from saved individual views')
                first=float(np.mean(matrix[0]!=y[split['test']]));mean=float(np.mean(matrix[:prefix]!=y[split['test']]));error=float(np.mean(np.asarray(pred)!=y[split['test']]))
                actual=dict(prefix=prefix,first_view_error=first,mean_single_view_error=mean,ensemble_error=error,change_from_first=error-first,change_from_mean_single=error-mean)
                require(next(d for d in diag['prefixes'] if d['prefix']==prefix)==actual,'prefix error changes disagree with predictions')

    else:require(not result['diagnostics'],'unexpected secondary diagnostics')


def collect_results(output,*,allow_smoke=False,validated_jobs=None):
    output=Path(output);p,manifest,jobs=verify(output,allow_smoke);issues=[];collected=defaultdict(list);complete=[]
    revision=json.loads((output/'environment.json').read_text())['code_revision']
    for job in jobs:
        path=output/'results'/(job['stem']+'.json');log=output/'logs'/(job['stem']+'.jsonl')
        if not path.exists() or not log.exists():issues.append('missing job '+job['stem']);continue
        try:
            result=json.loads(path.read_text());events=[json.loads(line) for line in log.read_text().splitlines()]
            X,y,data,splits=load_prepared(output,job['dataset_id']);split=next(s for s in splits if (s['outer_repeat'],s['outer_fold'])==(job['outer_repeat'],job['outer_fold']))
            validate_job(result,events,job,p,X,y,split,data,revision,output/'artifacts'/job['stem'])
            collected[job['family']].extend(result['rows'])
            if validated_jobs is not None:complete.append(result)
        except (KeyError,ValueError,TypeError,IndexError,OSError) as exc:issues.append(job['stem']+': '+str(exc))
    if issues:raise ValueError('Incomplete secondary evidence: '+'; '.join(issues))
    if validated_jobs is not None:validated_jobs.extend(complete)
    return dict(collected)


def run(output,workers=1,*,allow_smoke=False):
    p,manifest,jobs=verify(output,allow_smoke)
    if not 1<=workers<=p['max_workers']:raise ValueError('Worker count exceeds shared limit')
    with execution_lock(), ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in pool.map(worker,[(str(output),job) for job in jobs]):print(path,flush=True)
    return collect_results(output,allow_smoke=allow_smoke)


def summary(output,*,allow_smoke=False):
    validated=[];rows=collect_results(output,allow_smoke=allow_smoke,validated_jobs=validated)
    p=json.loads((Path(output)/'protocol.json').read_text())
    result={};folds=[(r,f) for r in range(p['outer_repeats']) for f in range(p['outer_folds'])]
    for family,records in rows.items():
        groups=defaultdict(list)
        for r in records:groups[(r['dataset_id'],r['model_id'],r['corruption_family'],r['severity'])].append(r)
        output_rows=[]
        for key,values in groups.items():
            # Average noise draws first, then fitting seeds within outer folds.
            seedfold=defaultdict(list)
            for r in values:seedfold[(r['outer_repeat'],r['outer_fold'],r['model_seed'])].append(r)
            reduced=[dict(outer_repeat=r,outer_fold=f,model_seed=s,model_id=key[1],status='ok',
                **{metric:float(np.mean([row[metric] for row in v])) for metric in ('accuracy','error','balanced_accuracy','macro_f1')})
                for (r,f,s),v in seedfold.items()]
            seeds=sorted({r['model_seed'] for r in reduced})
            output_rows.append(dict(dataset_id=key[0],model_id=key[1],corruption_family=key[2],severity=key[3],
                metrics={metric:summarize_outer(reduced,key[1],metric,expected_folds=folds,expected_seeds=seeds)
                         for metric in ('accuracy','error','balanced_accuracy','macro_f1')}))
            if key[2]!='clean':
                clean=[dict(r,model_id='clean_reference',status='ok') for r in groups[(key[0],key[1],'clean',0.)]]
                changed=[dict(r,model_id='condition',dataset_id=key[0]) for r in reduced]
                output_rows[-1]['change_from_clean']=paired_corrected_interval(changed+clean,'condition','clean_reference',metric='error',
                    q=p['test_train_ratio'],expected_folds=folds,expected_seeds={'condition':seeds,'clean_reference':seeds})
        result[family]=output_rows
    report={'summaries':result,'noise_replicates':'averaged within fit seed, then fitting seeds within outer fold','inferential_significance_claims':False}
    if 'e05' in rows:
        from .view_reporting import summarize_view_diagnostics
        report['view_diagnostics']=summarize_view_diagnostics([job for job in validated if job['provenance']['family']=='e05'],p)
    return report


def cost_worker(arguments):
    """One independent cost job in a fresh process; no accuracy is computed."""
    output,job,p=arguments;output=Path(output);path=output/'cost'/f'{job["job_id"]}.json'
    if path.exists():raise FileExistsError('Existing cost job')
    rows=[]
    try:
        X,y,data,splits=load_prepared(Path(job['source_root']),job['dataset_id']);split=splits[0];train=split['train'];query=train[::4]
        Xtr,ytr,Xqu=X[train],y[train],X[query];native=job['dataset_id']==study.__dict__.get('SUSHI_ID','sushi_childhood_east_west_v2')
        if job['kind']=='ensemble':
            view_p=dict(p,fit_seeds=[p['cost_seed']],head_conditions=[True],e05_schemes=[job['scheme']])
            start=time.perf_counter();learned=study.fit_views(Xtr,ytr,Xqu,view_p,dataset_id=job['dataset_id'],repeat=0,fold=0,retain_models=True)
            total=time.perf_counter()-start
            if any(f['status']!='ok' for f in learned['fits']):raise RuntimeError('Cost view fitting failure')
            for prefix in p['e05_prefixes']:
                members=[learned['models'][(job['scheme'],p['cost_seed'],True,v)] for v in range(prefix)]
                fits=learned['fits'][:prefix];encoder_times={r['encoder_hash']:r['encoder_fit_seconds'] for r in fits}
                measured=study.measure_cost(study.ViewEnsemble(members),Xqu,p,
                    fit_seconds=sum(f['fit_seconds'] for f in fits)+sum(encoder_times.values()),
                    encoder_fit_seconds=sum(encoder_times.values()),training_samples=len(train))
                rows.append(dict(record_id=job['job_id']+f'__prefix{prefix}',prefix=prefix,unique_group_fit_wall_seconds=total,logical_group_fits=p['e05_views'],**measured))
        else:
            start=time.perf_counter();encoding=None
            if job['kind']=='selected':
                from .run_revision import get_registry
                registry=get_registry(job['registry'],job['source_protocol']);model=registry[job['model_id']].factory(job['config'],p['cost_seed'])
                model.fit(Xtr,ytr)
            else:
                from .comparisons import NativeArrowFlow,BordaClassifier,OrderedPositionHDC,StableFootruleKNN
                encoder_start=time.perf_counter()
                enc=None if native else study.OrdinalEncoder('random',p['embed_dim'],2,.3,study.encoder_seed(p,job['dataset_id'],split)).fit(Xtr)
                orders=Xtr if native else enc.transform(Xtr);encoding=time.perf_counter()-encoder_start
                if job['kind']=='fixed_af':
                    widths=job['config']['widths'];seed=study.derive_seed(p['cost_seed'],'network',job['dataset_id'],0,0,-1,widths)
                    net_p=dict(p,architectures=[widths],primary_architecture_index=0)
                    model=study.fixed_network(net_p,orders.shape[1],seed).fit_orders(orders,ytr)
                    if enc is not None:model.encoder_=enc
                    else:
                        # Native adapter supplies the raw-order predict boundary.
                        native_model=NativeArrowFlow(widths=widths,iterations=p['iterations'],seed=seed);native_model.__dict__.update(model.__dict__);model=native_model
                    model.encoding_seconds_=encoding
                else:
                    if job['kind']=='footrule':readout=StableFootruleKNN(**job['config']).fit(orders,ytr,sample_ids=train)
                    elif job['kind']=='borda':readout=BordaClassifier().fit(orders,ytr)
                    else:
                        dimension=job['config']['dimension'];seed=p['cost_seed']
                        readout=OrderedPositionHDC(dimension,seed=seed,
                            item_seed=study.derive_seed(seed,'hdc_item',job['dataset_id'],0,0,-1,dimension),
                            position_seed=study.derive_seed(seed,'hdc_position',job['dataset_id'],0,0,-1,dimension)).fit(orders,ytr)
                    encoding+=getattr(readout,'encoding_seconds_',0.)
                    model=readout if native else study.EncodedReadout(enc,readout)
                    model.encoding_seconds_=encoding;model.classifier_fit_seconds_=getattr(readout,'classifier_fit_seconds_',None)
            elapsed=time.perf_counter()-start
            rows.append(dict(record_id=job['job_id'],**study.measure_cost(model,Xqu,p,fit_seconds=elapsed,
                encoder_fit_seconds=encoding,training_samples=len(train))))
        report={'status':'ok','job':job,'train_ids':train,'query_ids':query,'raw_train_hash':array_hash(Xtr),'raw_query_hash':array_hash(Xqu),
                'dataset_hash':data['dataset_hash'],'rows':rows,'environment':environment()}
    except Exception as exc:report={'status':'failed','job':job,'rows':rows,'exception':f'{type(exc).__name__}: {exc}'}
    write_json(path,report);return str(path)


def cost(output,p,e02_source,native_source,matched_source):
    if not all((e02_source,native_source,matched_source)):raise ValueError('All completed E02/native/matched sources are required')
    output=Path(output)
    if not p['frozen']:raise ValueError('Cost requires a frozen reviewed protocol')
    if p!=json.loads((output/'protocol.json').read_text()):raise ValueError('Cost protocol differs from prepared study')
    with execution_lock():
        # Cost runs only after all secondary and source panels have completed.
        collect_results(output)
        from .run_revision import get_registry
        from .reporting import collect_verified_results
        from .run_studies import collect_study_results
        sources={};jobs=[]
        for source in (Path(e02_source),Path(native_source)):
            saved=json.loads((source/'environment.json').read_text());protocol=json.loads((source/'protocol.json').read_text())
            if not protocol['frozen'] or saved!=environment_record(saved['registry']):raise ValueError('Unfrozen or changed selected-model source')
            names=list(dict.fromkeys(j['dataset_id'] for j in json.loads((source/'planned_jobs.json').read_text())))
            registry=get_registry(saved['registry'],protocol);collect_verified_results(source,names,protocol,registry)
            sources[str(source)]={'environment':saved,'protocol_hash':config_id(protocol)}
            for name in p['cost_datasets']:
                if name not in names:continue
                for model in registry:
                    path=source/'results'/f'{name}__{model}__r0f0.json';record=json.loads(path.read_text())
                    jobs.append(dict(job_id=f'selected__{name}__{model}',kind='selected',dataset_id=name,model_id=model,
                        config=record['selection']['config'],source_root=str(source),source_record=str(path),
                        source_record_hash=hashlib.sha256(path.read_bytes()).hexdigest(),source_protocol=protocol,registry=saved['registry']))
        matched_source=Path(matched_source);collect_study_results(matched_source)
        sources[str(matched_source)]={'environment':json.loads((matched_source/'environment.json').read_text())}
        for name in p['cost_datasets']:
            path=matched_source/'results'/f'{name}__r0f0.json';record=json.loads(path.read_text())
            common=dict(dataset_id=name,source_root=str(matched_source),source_record=str(path),source_record_hash=hashlib.sha256(path.read_bytes()).hexdigest())
            widths=p['native_architectures'] if name=='sushi_childhood_east_west_v2' else p['architectures']
            for w in widths:jobs.append(dict(job_id=f'matched__{name}__af'+str(w).replace(' ',''),kind='fixed_af',config={'widths':w},**common))
            for kind,key in [('footrule','input_footrule'),('hdc','hdc'),('borda',None)]:
                jobs.append(dict(job_id=f'matched__{name}__{kind}',kind=kind,config=record['selected'][key]['config'] if key else {},**common))
            if name!='sushi_childhood_east_west_v2':
                for scheme in p['e05_schemes']:jobs.append(dict(job_id=f'views__{name}__{scheme}',kind='ensemble',scheme=scheme,config={},**common))
        write_json(output/'cost'/'sources.json',sources);write_json(output/'cost'/'planned_jobs.json',jobs)
        # One worker and one task per fresh process; never overlap benchmark jobs.
        with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            for path in pool.map(cost_worker,[(str(output),job,p) for job in jobs]):print(path,flush=True)
        reports=[]
        for job in jobs:
            report=json.loads((output/'cost'/f'{job["job_id"]}.json').read_text())
            expected=[job['job_id']+f'__prefix{k}' for k in p['e05_prefixes']] if job['kind']=='ensemble' else [job['job_id']]
            if report['status']!='ok' or report['job']!=job or [r['record_id'] for r in report['rows']]!=expected:raise ValueError('Incomplete cost evidence')
            if any(len(r['prediction_seconds'])!=p['cost_repetitions'] or r['warmups']!=p['cost_warmups'] for r in report['rows']):raise ValueError('Incomplete cost repetitions')
            reports.append(report)
        write_json(output/'cost'/'report.json',{'purpose':'sequential_training_queries_only_no_accuracy','jobs':reports})
        return reports


def runtime_pilot(output,p):
    """Fixed timing choices only: first outer training partition, no score call."""
    output=Path(output);write_json(output/'protocol.json',p);write_json(output/'environment.json',environment())
    records=[]
    with execution_lock():
        for name in ['iris','digits']:
            X,y,data=load_dataset(name);split=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])[0]
            train=split['train'];query=train[::4];Xtr,ytr,Xqu=X[train],y[train],X[query]
            common=dict(dataset_id=name,dataset_hash=data['dataset_hash'],train_ids=train,query_ids=query,split_hash=config_id(split),
                raw_train_hash=array_hash(Xtr),raw_query_hash=array_hash(Xqu))
            view_p=dict(p,fit_seeds=p['fit_seeds'][:1]);start=time.perf_counter()
            views=study.fit_views(Xtr,ytr,Xqu,view_p,dataset_id=name,repeat=0,fold=0,writer=ArtifactWriter(output/name/'views'))
            row=dict(common,family='e05',elapsed_seconds=time.perf_counter()-start,fits=views['fits'],logical_fits=len(views['fits']),
                unique_fits=len({(f.get('encoder_hash'),f.get('initial_state_hash'),f['head_update']) for f in views['fits']}),
                status='ok' if all(f['status']=='ok' for f in views['fits']) else 'failed')
            records.append(row);write_json(output/name/'e05.json',row);print(canonical_json({k:row[k] for k in ('dataset_id','family','elapsed_seconds','status')}),flush=True)
            if name=='digits':
                for model_id,spec in study.e06_registry(p).items():
                    # Prespecified timing extreme among canonical candidate list, without scoring.
                    def scale(c):return (c.get('n_estimators',0),c.get('C',0),-c.get('n_neighbors',1),canonical_json(c))
                    cfg=max(spec.candidates,key=scale);start=time.perf_counter();model=spec.factory(cfg,p['cost_seed']).fit(Xtr,ytr)
                    fitted=time.perf_counter()-start;measured=study.measure_cost(model,Xqu,p,fit_seconds=fitted,training_samples=len(train))
                    records.append(dict(common,family='e06',model_id=model_id,config=cfg,status='ok',**measured))
            if name=='digits':
                start=time.perf_counter();enc=study.RowRankingEncoder().fit(Xtr);orders=enc.transform(Xtr);encoding=time.perf_counter()-start
                start=time.perf_counter();model=study.fixed_network(p,orders.shape[1],study.model_seed(p,name,split,p['cost_seed'])).fit_orders(orders,ytr);model.encoder_=enc
                fitted=time.perf_counter()-start
                bank=study.make_bank(Xtr,Xqu,p,name,0,0);start=time.perf_counter();study.predict_cases({('row_af',p['cost_seed']):model},bank.cases)
                all_cases_seconds=time.perf_counter()-start
                records.append(dict(common,family='e06',model_id='row_af',status='ok',corruption_case_count=len(bank.cases),all_cases_inference_seconds=all_cases_seconds,
                    **study.measure_cost(model,Xqu,p,fit_seconds=fitted+encoding,encoder_fit_seconds=encoding,training_samples=len(train))))
            for d in degree_schedule(X.shape[1],p['degrees'],p['degree_column_cap']):
                if not d['eligible']:
                    records.append(dict(common,family='e07',degree=d['degree'],status='excluded',columns=d['columns']));continue
                start=time.perf_counter();enc=study.NestedDegreeEncoder(p['embed_dim'],d['degree'],study.encoder_seed(p,name,split),p['degree_column_cap']).fit(Xtr)
                orders=enc.transform(Xtr);encoding=time.perf_counter()-start
                start=time.perf_counter();model=study.fixed_network(p,p['embed_dim'],study.model_seed(p,name,split,p['cost_seed'])).fit_orders(orders,ytr);model.encoder_=enc
                fitted=time.perf_counter()-start
                records.append(dict(common,family='e07',degree=d['degree'],columns=d['columns'],status='ok',
                    **study.measure_cost(model,Xqu,p,fit_seconds=fitted+encoding,encoder_fit_seconds=encoding,training_samples=len(train))))
            write_json(output/name/'timings.json',[r for r in records if r['dataset_id']==name])
    report={'purpose':'training_only_fixed_runtime_no_scores','environment':environment(),'protocol_hash':config_id(p),
        'records':records,'workload_counts':study.workload_counts(p),'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(output/'pilot.json',report);return report



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','smoke','pilot','run','summary','cost'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--sushi-archive',type=Path);parser.add_argument('--e02-source',type=Path);parser.add_argument('--native-source',type=Path);parser.add_argument('--matched-source',type=Path)
    args=parser.parse_args(argv);p=json.loads(args.protocol.read_text())
    if args.sushi_archive:os.environ['ARROWFLOW_SUSHI_ARCHIVE']=str(args.sushi_archive)
    if args.command=='prepare':prepare(args.output,p)
    elif args.command=='run':
        if not p['frozen']:raise ValueError('Confirmation requires a frozen reviewed protocol')
        if p!=json.loads((args.output/'protocol.json').read_text()):raise ValueError('Prepared protocol differs')
        run(args.output,args.workers)
    elif args.command=='summary':write_json(args.output/'summary.json',summary(args.output))
    elif args.command=='smoke':
        tiny=dict(p,outer_folds=3,outer_repeats=1,inner_folds=2,architectures=[[4],[4,3],[8]],embed_dim=6,iterations=4,
            e05_views=3,e05_prefixes=[1,3],candidate_budget=1,degrees=[1,2],corruption_seeds=[104729],
            gaussian_levels=[0,.1],quantization_steps=[0,.1],masking_probabilities=[0,.1],crossed_levels=[.1],
            probe_neighbors=[1,3],probe_weights=['uniform','distance'],panels={family:['synthetic'] for family in p['families']},frozen=False)
        X=np.random.RandomState(43).randn(60,4);y=np.tile([0,1,2],20);names=['a','b','c','d'];labels=['0','1','2']
        data={'dataset_id':'synthetic','feature_names':names,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,names,labels)}
        prepare(args.output,tiny,purpose='synthetic_smoke_only',loader=lambda name:(X,y,data));run(args.output,args.workers,allow_smoke=True)
        write_json(args.output/'summary.json',summary(args.output,allow_smoke=True))
    elif args.command=='pilot':runtime_pilot(args.output,p)
    else:cost(args.output,p,args.e02_source,args.native_source,args.matched_source)


if __name__=='__main__':main()
