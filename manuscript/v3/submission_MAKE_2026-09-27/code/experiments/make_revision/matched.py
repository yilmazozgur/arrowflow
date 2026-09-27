"""E03/E04 matched learning/probe study. Query labels never enter cache fitting."""
import copy
import hashlib
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .comparisons import (derive_seed, StableFootruleKNN, BordaClassifier, OrderedPositionHDC)
from .datasets import SUSHI_ID
from .evaluation import (candidate_grid, config_id, canonical_json, metric_values,
                         validate_split, validate_outer_schedule)
from .models import ArrowFlowEstimator, OrdinalEncoder, SharedEncoding, array_hash
from arrowflow.ranking import inverse_positions

SOURCE_MODULES = ['experiments.make_revision.models','experiments.make_revision.comparisons',
                  'experiments.make_revision.datasets','experiments.make_revision.evaluation']


def architecture_id(widths): return 'af_h' + '_'.join(map(str, widths))


def model_seed_schedule(protocol, native=False):
    seeds = protocol['fit_seeds']
    result = {'input_footrule': seeds[:1], 'borda': seeds[:1], 'hdc': seeds}
    for widths in protocol['native_architectures' if native else 'architectures']:
        arch = architecture_id(widths)
        result[arch+'_output'] = seeds
        for depth in range(1,len(widths)+1):
            for state in ('trained','untrained'):
                result[f'{arch}_d{depth}_{state}'] = seeds
    return result


def probe_candidates(protocol):
    return candidate_grid({'n_neighbors':protocol['probe_neighbors'],'weights':protocol['probe_weights']},
                          budget=len(protocol['probe_neighbors'])*len(protocol['probe_weights']))


def encoder_settings(protocol, dataset_id):
    """Encoder width/degree for one dataset; the global fields apply outside `encoder_by_dataset`."""
    table = protocol.get('encoder_by_dataset')
    if table and dataset_id in table:
        return dict(table[dataset_id])
    return {'embed_dim': protocol['embed_dim'], 'degree': protocol['degree']}


def validate_encoder_table(protocol):
    """Every `encoder_by_dataset` key must name a protocol dataset and carry embed_dim and degree;
    a misspelt key would otherwise fall back silently to the global settings."""
    table = protocol.get('encoder_by_dataset') or {}
    unknown = sorted(set(table) - set(protocol['datasets']))
    if unknown:
        raise ValueError(f'encoder_by_dataset names datasets outside the protocol panel: {unknown}')
    for name, settings in table.items():
        if not isinstance(settings, dict) or set(settings) != {'embed_dim', 'degree'}:
            raise ValueError(f'encoder_by_dataset[{name!r}] must map exactly embed_dim and degree')
    return table


class ArtifactWriter:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else None
        self.files = []

    def save(self, name, arrays):
        if self.root is None: return
        path = self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
        self.files.append({'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


class BundleCache:
    """One in-memory bundle per exact rank hash/dimension/item+position seeds."""
    def __init__(self):
        self.arrays = {}
        self.computations = 0

    def get(self, model, orders):
        key = (array_hash(orders),model.dimension,model.item_seed_,model.position_seed_)
        if key not in self.arrays:
            value = model.transform(orders)
            value.setflags(write=False)
            self.arrays[key] = value
            self.computations += 1
        return self.arrays[key]


def build_partition(X_train,y_train,X_query,train_ids,query_ids,*,dataset_id,outer_repeat,
                    outer_fold,inner_fold,protocol,native=False,writer=None,hdc_dimensions=None):
    """Fit one shared encoder and one trained copy per architecture/seed.

    All representations/bundles are cached until this partition is scored, then
    callers release the cache. Disk artifacts retain shared orders, fitted encoder
    arrays, initial/final network snapshots and every hidden representation;
    large HDC bundles remain in memory and their hashes are persisted.
    """
    writer = writer or ArtifactWriter()
    p=protocol; train_ids=np.asarray(train_ids); query_ids=np.asarray(query_ids)
    if (len(train_ids)!=len(X_train) or len(query_ids)!=len(X_query)
        or len(y_train)!=len(X_train) or len(np.unique(train_ids))!=len(train_ids)
        or len(np.unique(query_ids))!=len(query_ids)):
        raise ValueError('Partition row IDs must align and be unique')
    if p['augmentation'] or not p['multistep_lr'] or p['strategy']!='random':
        raise ValueError('Matched study requires no augmentation and fixed random encoding/multistep training')
    partition_id = f'inner{inner_fold}' if inner_fold>=0 else 'outer'
    settings=encoder_settings(p,dataset_id)
    encoder_seed=derive_seed(p['encoder_seed'],dataset_id,outer_repeat,outer_fold,inner_fold,
                             settings['degree'],p['strategy'],p['view_id'])
    start=time.perf_counter()
    if native:
        inverse_positions(X_train);inverse_positions(X_query)
        train=np.asarray(X_train,dtype=np.int64).copy();query=np.asarray(X_query,dtype=np.int64).copy()
        train.setflags(write=False);query.setflags(write=False)
        shared=SharedEncoding(train,query,array_hash(train),array_hash(query),array_hash(np.asarray(X_train)),array_hash(np.asarray(X_query)))
        encoder_arrays={}
    else:
        encoder=OrdinalEncoder(strategy='random',degree=settings['degree'],embed_dim=settings['embed_dim'],seed=encoder_seed).fit(X_train)
        shared=SharedEncoding.create(encoder,X_train,X_query)
        encoder_arrays={'imputation_means':encoder.imputer_.means_, 'scaler_mean':encoder.scaler_.mean_,
                        'scaler_scale':encoder.scaler_.scale_, 'scaler_var':encoder.scaler_.var_,
                        'projection':encoder.projection_}
        if encoder.poly_ is not None: encoder_arrays['polynomial_powers']=encoder.poly_.powers_
    metadata={'partition_id':partition_id,'inner_fold':inner_fold,'train_ids':train_ids.tolist(),
              'query_ids':query_ids.tolist(),'encoder_seed':None if native else encoder_seed,
              'encoder_parameters':{'degree':settings['degree'],'strategy':p['strategy'],'embed_dim':shared.train.shape[1],'native':native},
              'source_train_hash':shared.train_hash,'source_query_hash':shared.test_hash,
              'raw_train_hash':shared.raw_train_hash,'raw_query_hash':shared.raw_test_hash,
              'training_labels_hash':array_hash(np.asarray(y_train)),
              'encoder_array_hashes':{k:array_hash(v) for k,v in encoder_arrays.items()},
              'encoding_seconds':time.perf_counter()-start}
    writer.save(partition_id+'/shared.npz',dict(train=shared.train,query=shared.test,train_ids=train_ids,query_ids=query_ids,**encoder_arrays))
    cache={'shared':shared,'metadata':metadata,'networks':{},'hdc':{},'events':[], 'failures':[]}
    cache['events'].append(dict(metadata,stage='encoding',status='ok'))
    common={'partition_id':partition_id,'inner_fold':inner_fold,
            'source_train_hash':shared.train_hash,'source_query_hash':shared.test_hash}
    with threadpool_limits(limits=1):
        for widths in p['native_architectures' if native else 'architectures']:
            arch=architecture_id(widths)
            for seed in p['fit_seeds']:
                network_seed=derive_seed(seed,'network',dataset_id,outer_repeat,outer_fold,inner_fold,widths)
                event=dict(common,stage='network_pair',architecture=arch,widths=widths,model_seed=seed,network_seed=network_seed)
                try:
                    shared.assert_identical(shared.train,shared.test)
                    start=time.perf_counter()
                    initial=ArrowFlowEstimator(embed_dim=shared.train.shape[1],widths=widths,
                        iterations=p['iterations'],learning_rate=p['learning_rate'],batch_size=p['batch_size'],
                        last_layer_update=p['last_layer_update'],ratio_data_backprop=p['ratio_data_backprop'],
                        motion_normalization_mult=p['motion_normalization_mult'],p_correct=p['p_correct'],
                        validation_ratio=p.get('validation_ratio',0),seed=network_seed)
                    initial.initialize_orders(shared.train,y_train)
                    before=initial.state_hash(); trained=copy.deepcopy(initial)
                    trained_start=trained.state_hash()
                    if before!=trained_start: raise ValueError('Initial model copies disagree')
                    event['initialization_seconds']=time.perf_counter()-start
                    writer.save(f'{partition_id}/{arch}_{seed}_initial.npz',initial.state_snapshot())
                    trained.train_initialized(shared.train,y_train)
                    after=initial.state_hash()
                    if before!=after: raise ValueError('Training mutated the initial control')
                    start=time.perf_counter()
                    representations={}
                    for state,model in [('untrained',initial),('trained',trained)]:
                        tr=model.transform_orders_by_depth(shared.train);qu=model.transform_orders_by_depth(shared.test)
                        for value in tr+qu: value.setflags(write=False)
                        representations[state]={'train':tr,'query':qu}
                    event['representation_seconds']=time.perf_counter()-start
                    start=time.perf_counter(); output=trained.predict_orders(shared.test)
                    event['output_inference_seconds']=time.perf_counter()-start
                    final_hash=trained.state_hash()
                    writer.save(f'{partition_id}/{arch}_{seed}_trained.npz',trained.state_snapshot())
                    arrays={f'{state}_{side}_d{d+1}':value for state,rep in representations.items()
                            for side,depths in rep.items() for d,value in enumerate(depths)}
                    writer.save(f'{partition_id}/{arch}_{seed}_representations.npz',arrays)
                    event.update(status='ok',initial_hash=before,trained_start_hash=trained_start,
                                 initial_after_hash=after,final_state_hash=final_hash,fit_seconds=trained.training_seconds_,
                                 validation_sample_count=trained.validation_sample_count_,
                                 network_training_sample_count=trained.training_sample_count_,
                                 representation_hashes={k:array_hash(v) for k,v in arrays.items()},
                                 prototype_entries=sum(layer.index_matrix.size for layer in trained.network_.graph.vertex_list.values()),
                                 peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    cache['networks'][(arch,seed)]={'representations':representations,'output':output,'event':event}
                except Exception as exc:
                    event.update(status='failed',exception=f'{type(exc).__name__}: {exc}')
                    cache['failures'].append(event)
                cache['events'].append(event)
        bundle_cache=BundleCache()
        for dimension in (p['hdc_dimensions'] if hdc_dimensions is None else hdc_dimensions):
            for seed in p['fit_seeds']:
                item_seed=derive_seed(seed,'hdc_item',dataset_id,outer_repeat,outer_fold,inner_fold,dimension)
                position_seed=derive_seed(seed,'hdc_position',dataset_id,outer_repeat,outer_fold,inner_fold,dimension)
                event=dict(common,stage='hdc_bundle',dimension=dimension,model_seed=seed,item_seed=item_seed,position_seed=position_seed)
                try:
                    shared.assert_identical(shared.train,shared.test)
                    model=OrderedPositionHDC(dimension=dimension,seed=seed,item_seed=item_seed,position_seed=position_seed).fit_encoder(shared.train)
                    start=time.perf_counter();tr=bundle_cache.get(model,shared.train);qu=bundle_cache.get(model,shared.test)
                    encoding_seconds=time.perf_counter()-start+model.encoder_setup_seconds_
                    start=time.perf_counter();model.fit_bundles(tr,y_train);fit_seconds=time.perf_counter()-start
                    start=time.perf_counter();prediction=model.predict_bundles(qu);inference_seconds=time.perf_counter()-start
                    event.update(status='ok',encoding_seconds=encoding_seconds,fit_seconds=fit_seconds,
                                 inference_seconds=inference_seconds,train_bundle_hash=array_hash(tr),query_bundle_hash=array_hash(qu),
                                 representation_metadata=model.representation_metadata_,
                                 peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    writer.save(f'{partition_id}/hdc_{dimension}_{seed}.npz',dict(item_keys=model.item_keys_,position_codes=model.position_codes_,class_prototypes=model.class_prototypes_))
                    cache['hdc'][(dimension,seed)]={'prediction':prediction,'event':event}
                except Exception as exc:
                    event.update(status='failed',exception=f'{type(exc).__name__}: {exc}');cache['failures'].append(event)
                cache['events'].append(event)
        cache['bundle_computations']=bundle_cache.computations
    return cache


def score_inner_partition(cache,y_train,y_query,protocol):
    """Only this scoring boundary receives validation labels; no learning refit."""
    p=protocol; meta=cache['metadata'];shared=cache['shared']; rows=[]
    def record(kind,cfg,seed,state,predict,**extra):
        row=dict(kind=kind,config=cfg,config_id=config_id(cfg),model_seed=seed,state=state,
                 inner_fold=meta['inner_fold'],partition_id=meta['partition_id'],**extra)
        try:
            start=time.perf_counter();pred=predict()
            if y_query is not None:
                if len(pred)!=len(y_query): raise ValueError('Prediction count mismatch')
                row['score']=float(np.mean(np.asarray(pred)==np.asarray(y_query)))
            row.update(status='ok',elapsed_seconds=time.perf_counter()-start,
                       prediction_hash=array_hash(np.asarray(pred)))
        except Exception as exc: row.update(status='failed',score=None,exception=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    def grid(kind,tr,qu,seed,state,input_kind,**extra):
        candidates=probe_candidates(p)
        try:
            start=time.perf_counter()
            model=StableFootruleKNN(n_neighbors=max(p['probe_neighbors']),input_kind=input_kind).fit(tr,y_train,sample_ids=meta['train_ids'])
            distances,indices=model.kneighbors(qu)
            elapsed=time.perf_counter()-start
            cache_id=config_id({'train':array_hash(tr),'query':array_hash(qu),'row_ids':meta['train_ids'],
                                'max_neighbors':max(p['probe_neighbors'])})
            for cfg in candidates:
                model.n_neighbors=cfg['n_neighbors'];model.weights=cfg['weights']
                record(kind,cfg,seed,state,lambda:model.predict_neighbors(distances,indices),
                       neighbor_cache_id=cache_id,neighbor_build_seconds=elapsed,**extra)
        except Exception as exc:
            for cfg in candidates:
                rows.append(dict(kind=kind,config=cfg,config_id=config_id(cfg),model_seed=seed,state=state,
                                 inner_fold=meta['inner_fold'],partition_id=meta['partition_id'],
                                 status='failed',score=None,exception=f'{type(exc).__name__}: {exc}',**extra))
    for (arch,seed),net in cache['networks'].items():
        for state,rep in net['representations'].items():
            for depth,(tr,qu) in enumerate(zip(rep['train'],rep['query']),1):
                grid('probe',tr,qu,seed,state,'positions',architecture=arch,depth=depth)
        record('output',{},seed,'trained',lambda net=net:net['output'],architecture=arch)
    shared.assert_identical(shared.train,shared.test)
    grid('input',shared.train,shared.test,p['fit_seeds'][0],'input','orders')
    record('borda',{},p['fit_seeds'][0],'borda',lambda:BordaClassifier().fit(shared.train,y_train).predict(shared.test))
    for (dimension,seed),hdc in cache['hdc'].items():
        record('hdc',{'dimension':dimension},seed,'hdc',lambda hdc=hdc:hdc['prediction'])
    return rows


def select_symmetric_probe(rows,candidates,*,inner_folds,seeds,states=('trained','untrained')):
    expected={(fold,seed,state) for fold in range(inner_folds) for seed in seeds for state in states}
    ranked=[]
    for cfg in candidates:
        selected=[r for r in rows if r['config_id']==config_id(cfg)]
        keys=[(r['inner_fold'],r['model_seed'],r['state']) for r in selected]
        if set(keys)!=expected or len(keys)!=len(expected):
            raise ValueError('Selection needs a complete declared inner-fold/seed/state matrix')
        if all(r['status']=='ok' for r in selected):
            ranked.append((-float(np.mean([r['score'] for r in selected])),config_id(cfg),cfg))
    if not ranked: raise ValueError('No complete successful candidate for paired selection')
    score,cid,cfg=min(ranked,key=lambda item:item[:2])
    return {'config':cfg,'config_id':cid,'inner_score':-score}


def choose_configurations(scores,protocol,native=False):
    p=protocol; selected={}; candidates=probe_candidates(p)
    for widths in p['native_architectures' if native else 'architectures']:
        arch=architecture_id(widths)
        for depth in range(1,len(widths)+1):
            rows=[r for r in scores if r['kind']=='probe' and r['architecture']==arch and r['depth']==depth]
            chosen=select_symmetric_probe(rows,candidates,inner_folds=p['inner_folds'],seeds=p['fit_seeds'])
            for state in ('trained','untrained'):selected[f'{arch}_d{depth}_{state}']=chosen
    selected['input_footrule']=select_symmetric_probe([r for r in scores if r['kind']=='input'],candidates,
        inner_folds=p['inner_folds'],seeds=p['fit_seeds'][:1],states=('input',))
    selected['hdc']=select_symmetric_probe([r for r in scores if r['kind']=='hdc'],[{'dimension':d} for d in p['hdc_dimensions']],
        inner_folds=p['inner_folds'],seeds=p['fit_seeds'],states=('hdc',))
    return selected


def evaluate_study_fold(X,y,split,protocol,*,dataset_id,dataset_hash,code_revision,native=False,artifact_dir=None,sink=None):
    """Fit/cache inside every inner partition, then fit selected procedures once."""
    X=np.asarray(X);y=np.asarray(y);p=protocol;validate_split(split,len(y))
    writer=ArtifactWriter(artifact_dir);events=[];partitions=[];scores=[];selected={};models=[];predictions=[]
    common={'dataset_id':dataset_id,'dataset_hash':dataset_hash,'outer_repeat':split['outer_repeat'],
            'outer_fold':split['outer_fold'],'code_revision':code_revision,'view_id':0,'condition':'clean','perturbation_seed':None}
    def emit(row):
        events.append(row)
        if sink is not None:sink(row)
    def build(train,query,inner,dimensions=None):
        cache=build_partition(X[train],y[train],X[query],train,query,dataset_id=dataset_id,
            outer_repeat=split['outer_repeat'],outer_fold=split['outer_fold'],inner_fold=inner,
            protocol=p,native=native,writer=writer,hdc_dimensions=dimensions)
        partitions.append(cache['metadata'])
        for event in cache['events']:emit(event)
        if cache['failures']:raise RuntimeError('Partition fitting failed; inspect retained events')
        return cache
    try:
        for inner,indices in enumerate(split['inner']):
            cache=build(indices['train'],indices['validation'],inner)
            rows=score_inner_partition(cache,y[indices['train']],y[indices['validation']],p)
            scores.extend(rows)
            for row in rows:emit(dict(row,stage='inner_score'))
            del cache
        selected=choose_configurations(scores,p,native)
        cache=build(split['train'],split['test'],-1,[selected['hdc']['config']['dimension']])
        shared=cache['shared'];meta=cache['metadata'];train_y=y[split['train']]
        def final(model_id,seed,cfg,predict,event=None,**extra):
            row=dict(common,model_id=model_id,model_seed=seed,config=cfg,config_id=config_id(cfg),
                     training_sample_count=len(train_y),source_train_hash=shared.train_hash,
                     source_query_hash=shared.test_hash,**extra)
            try:
                start=time.perf_counter();pred=np.asarray(predict());elapsed=time.perf_counter()-start
                row.update(status='ok',readout_fit_predict_seconds=elapsed,**metric_values(y[split['test']],pred))
                if event:row['fit_event']=event
                predictions.extend(dict(common,model_id=model_id,model_seed=seed,config_id=row['config_id'],
                    sample_id=int(sample),y_true=y[sample].item(),y_pred=np.asarray(label).item()) for sample,label in zip(split['test'],pred))
            except Exception as exc:row.update(status='failed',exception=f'{type(exc).__name__}: {exc}')
            models.append(row);emit(dict(row,stage='outer_model'))
        for (arch,seed),net in cache['networks'].items():
            for state,rep in net['representations'].items():
                for depth,(tr,qu) in enumerate(zip(rep['train'],rep['query']),1):
                    name=f'{arch}_d{depth}_{state}';cfg=selected[name]['config']
                    final(name,seed,cfg,lambda tr=tr,qu=qu,cfg=cfg:StableFootruleKNN(**cfg,input_kind='positions').fit(tr,train_y,sample_ids=split['train']).predict(qu),
                          event=net['event'],final_state_hash=net['event']['final_state_hash' if state=='trained' else 'initial_hash'])
            final(arch+'_output',seed,{'architecture':net['event']['widths'],'iterations':p['iterations']},lambda net=net:net['output'],event=net['event'],final_state_hash=net['event']['final_state_hash'])
        shared.assert_identical(shared.train,shared.test)
        cfg=selected['input_footrule']['config']
        final('input_footrule',p['fit_seeds'][0],cfg,lambda:StableFootruleKNN(**cfg).fit(shared.train,train_y,sample_ids=split['train']).predict(shared.test))
        final('borda',p['fit_seeds'][0],{},lambda:BordaClassifier().fit(shared.train,train_y).predict(shared.test))
        for (dimension,seed),hdc in cache['hdc'].items():final('hdc',seed,{'dimension':dimension},lambda hdc=hdc:hdc['prediction'],event=hdc['event'])
        validate_outer_schedule(models,expected_folds=[(split['outer_repeat'],split['outer_fold'])],expected_seeds=model_seed_schedule(p,native))
        status='ok'
    except Exception as exc:
        emit({'stage':'terminal_failure','status':'failed','exception':f'{type(exc).__name__}: {exc}'})
        present={(r['model_id'],r['model_seed']) for r in models}
        for model_id,seeds in model_seed_schedule(p,native).items():
            for seed in seeds:
                if (model_id,seed) not in present:
                    row=dict(common,model_id=model_id,model_seed=seed,status='failed',config_id=None,exception=f'{type(exc).__name__}: {exc}')
                    models.append(row);emit(dict(row,stage='outer_model'))
        status='failed'
    return {'status':status,'selected':selected,'models':models,'predictions':predictions,
            'inner_scores':scores,'partitions':partitions,'events':events,'artifacts':writer.files}
