"""Prespecified E05/E06/E07 ablations and label-free E11 timing helpers."""
import copy
from functools import partial
import hashlib
from itertools import combinations
import pickle
import resource
import time
import warnings
import numpy as np
from scipy.spatial.distance import cdist,pdist
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from arrowflow.ranking import score_order,inverse_positions
from .models import OrdinalEncoder,NumericImputer,ArrowFlowEstimator,array_hash
from .comparisons import derive_seed,conventional_registry,conventional_factory,StableFootruleKNN
from .evaluation import ModelSpec,config_id,canonical_json,select_model,metric_values,validate_split
from .matched import ArtifactWriter,probe_candidates,select_symmetric_probe
from .secondary_inputs import CorruptionBank,NestedDegreeEncoder,degree_schedule

SOURCE_MODULES=['experiments.make_revision.secondary_inputs','experiments.make_revision.models',
                'experiments.make_revision.comparisons','experiments.make_revision.evaluation',
                'experiments.make_revision.matched']


class RowRankingEncoder(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None):
        self.imputer_=NumericImputer().fit(X)
        self.scaler_=StandardScaler().fit(self.imputer_.transform(X))
        return self
    def transform(self,X):return score_order(self.scaler_.transform(self.imputer_.transform(X)))


def row_factory(family,config,seed):
    model=conventional_factory(family,config,seed,native=True)
    model.steps.insert(0,('row_encoder',RowRankingEncoder()))
    return model


def e06_registry(p):
    result={}
    for family,spec in conventional_registry(p).items():
        if family not in ('svc_rbf','random_forest','numeric_knn'):continue
        result['raw_'+family]=ModelSpec('raw_'+family,spec.factory,spec.candidates,spec.stochastic)
        result['row_'+family]=ModelSpec('row_'+family,partial(row_factory,family),spec.candidates,spec.stochastic)
    return result


def view_stream(p,dataset,repeat,fold,seed,scheme,view):
    if scheme not in ('same_projection','different_random','mixed'):raise ValueError('Unknown view scheme')
    strategy=('random','calibrated','target_aware')[view%3] if scheme=='mixed' else 'random'
    projection_view=0 if scheme=='same_projection' else view
    return {'network_seed':derive_seed(seed,'e05_network',dataset,repeat,fold,view),
            'encoder_seed':derive_seed(p['encoder_seed'],'e05_encoder',dataset,repeat,fold,seed,strategy,projection_view),
            'strategy':strategy,'projection_view':projection_view}


def majority(predictions):
    predictions=np.asarray(predictions)
    classes=np.unique(predictions)
    return classes[np.argmax(np.stack([(predictions==c).sum(axis=0) for c in classes]),axis=0)]


def error_diagnostics(predictions,truth):
    predictions=np.asarray(predictions);errors=predictions!=np.asarray(truth);rows=[]
    for a,b in combinations(range(len(predictions)),2):
        constant=np.std(errors[a])==0 or np.std(errors[b])==0
        rows.append({'view_a':a,'view_b':b,'disagreement':float(np.mean(predictions[a]!=predictions[b])),
                     'double_fault':float(np.mean(errors[a]&errors[b])),
                     'error_correlation':None if constant else float(np.corrcoef(errors[a],errors[b])[0,1]),
                     'correlation_reason':'constant error vector' if constant else None})
    return rows


def layer_diagnostics(model,orders):
    return diagnostics_from_layers([model.network_.graph.vertex_list[f'revision_ly{depth}'].index_matrix for depth in range(len(model.widths))],orders)


def diagnostics_from_layers(layers,orders):
    positions=inverse_positions(orders);rows=[]
    for depth,prototypes in enumerate(layers):
        count,vocab=prototypes.shape;dist=pdist(prototypes,metric='cityblock');diameter=(vocab*vocab)//2
        responses=cdist(positions,prototypes,metric='cityblock')
        unique=np.array([len(np.unique(row)) for row in responses])
        rows.append({'depth':depth+1,'prototype_count':count,'unique_count':len(np.unique(prototypes,axis=0)),
                     'duplicate_fraction':float(1-len(np.unique(prototypes,axis=0))/count),
                     'zero_distance_pair_fraction':float(np.mean(dist==0)) if len(dist) else 0.,
                     'normalized_distance_mean':float(np.mean(dist)/diameter) if len(dist) and diameter else 0.,
                     'normalized_distance_min':float(np.min(dist)/diameter) if len(dist) and diameter else 0.,
                     'normalized_distance_max':float(np.max(dist)/diameter) if len(dist) and diameter else 0.,
                     'query_any_response_tie_fraction':float(np.mean(unique<count)),
                     'query_repeated_response_fraction':float(np.mean(1-unique/count))})
        positions=inverse_positions(score_order(responses))
    return rows


def _content_state(value,active=None):
    """Structural state independent of pickle memo aliases and empty-array strides."""
    active=set() if active is None else active
    if isinstance(value,np.ndarray):return {'array':array_hash(value)}
    if isinstance(value,np.generic):return _content_state(value.item(),active)
    if value is None or isinstance(value,(str,int,bool)):return value
    if isinstance(value,float):return value if np.isfinite(value) else str(value)
    if isinstance(value,np.dtype):return str(value)
    if isinstance(value,type):return value.__module__+'.'+value.__qualname__
    if id(value) in active:return {'cycle':type(value).__name__}
    active.add(id(value))
    try:
        if isinstance(value,dict):return {str(k):_content_state(v,active) for k,v in sorted(value.items(),key=lambda item:str(item[0]))}
        if isinstance(value,(list,tuple)):return [_content_state(v,active) for v in value]
        if hasattr(value,'__dict__'):return {'type':type(value).__module__+'.'+type(value).__name__,'state':_content_state(vars(value),active)}
        if hasattr(value,'__getstate__'):return {'type':type(value).__module__+'.'+type(value).__name__,'state':_content_state(value.__getstate__(),active)}
        raise TypeError('Unsupported fitted state '+type(value).__name__)
    finally:active.remove(id(value))


def fitted_hash(model):
    if isinstance(model,ViewEnsemble):return config_id([(fitted_hash(enc),fitted_hash(net)) for enc,net in model.models])
    if isinstance(model,EncodedReadout):return config_id({'encoder':fitted_hash(model.encoder_),'readout':fitted_hash(model.readout_)})
    if isinstance(model,ArrowFlowEstimator):
        return config_id({'network':model.state_hash(),'encoder':fitted_hash(model.encoder_) if hasattr(model,'encoder_') else None})
    if hasattr(model,'steps'):return config_id({'steps':[(name,fitted_hash(step)) for name,step in model.steps]})
    # Prediction timing is instrumentation, not fitted model state.
    clone=copy.copy(model);clone.__dict__={k:v for k,v in model.__dict__.items() if not k.endswith('seconds_')}
    # sklearn kNN's derived KDTree stores changing query call counters; its
    # training coordinates/labels and neighbor/search settings remain hashed.
    if type(model).__name__=='KNeighborsClassifier':clone.__dict__.pop('_tree',None)
    return hashlib.sha256(canonical_json(_content_state(clone)).encode()).hexdigest()


def fixed_network(p,dim,seed,head=True):
    return ArrowFlowEstimator(embed_dim=dim,widths=p['architectures'][p['primary_architecture_index']],
        iterations=p['iterations'],learning_rate=p['learning_rate'],batch_size=p['batch_size'],
        ratio_data_backprop=p['ratio_data_backprop'],motion_normalization_mult=p['motion_normalization_mult'],
        p_correct=p['p_correct'],last_layer_update=head,seed=seed)


def subset_state_hash(model,head):
    arrays=model.state_snapshot();last=len(model.widths)
    keys=[k for k in arrays if k.startswith(f'layer_{last}_')==head and k.startswith('layer_')]
    return config_id({k:array_hash(arrays[k]) for k in keys})


def fit_views(X_train,y_train,X_query,p,*,dataset_id,repeat,fold,writer=None,retain_models=False):
    writer=writer or ArtifactWriter();encoders={};fits=[];outputs={};models={}
    for seed in p['fit_seeds']:
        for scheme in p['e05_schemes']:
            for view in range(p['e05_views']):
                stream=view_stream(p,dataset_id,repeat,fold,seed,scheme,view)
                key=(stream['encoder_seed'],stream['strategy'])
                try:
                    if key not in encoders:
                        start=time.perf_counter()
                        enc=OrdinalEncoder(stream['strategy'],p['embed_dim'],p['degree'],.3,stream['encoder_seed']).fit(X_train,y_train)
                        train=enc.transform(X_train);encoder_fit_seconds=time.perf_counter()-start
                        start=time.perf_counter();query=enc.transform(X_query);query_encoding_seconds=time.perf_counter()-start
                        train.setflags(write=False);query.setflags(write=False)
                        encoders[key]=(enc,train,query,encoder_fit_seconds,query_encoding_seconds)
                    enc,train,query,encoder_fit_seconds,query_encoding_seconds=encoders[key]
                    encoding_seconds=encoder_fit_seconds+query_encoding_seconds
                    initial=fixed_network(p,train.shape[1],stream['network_seed']).initialize_orders(train,y_train)
                    initial_hash=initial.state_hash()
                    for head in p['head_conditions']:
                        fit_id=f'{scheme}__s{seed}__h{int(head)}__v{view}'
                        event=dict(fit_id=fit_id,model_seed=seed,scheme=scheme,head_update=head,view=view,**stream)
                        try:
                            model=copy.deepcopy(initial)
                            model.last_layer_update=head;model.config_.last_layer_update=head;model.network_.last_layer_update=head
                            if model.state_hash()!=initial_hash:raise ValueError('Head pair initial states differ')
                            before_head=subset_state_hash(model,True);before_hidden=subset_state_hash(model,False)
                            writer.save(f'views/{fit_id}_initial.npz',initial.state_snapshot())
                            model.train_initialized(train,y_train);state=fitted_hash(model)
                            start=time.perf_counter();pred=model.predict_orders(query);inference=time.perf_counter()-start
                            if fitted_hash(model)!=state:raise ValueError('Prediction mutated fitted network')
                            after_head=subset_state_hash(model,True)
                            if not head and before_head!=after_head:raise ValueError('Frozen output head changed')
                            writer.save(f'views/{fit_id}_trained.npz',model.state_snapshot())
                            writer.save(f'views/{fit_id}_encoding.npz',{'train':train,'query':query,'serialized_encoder':np.frombuffer(pickle.dumps(enc,protocol=5),dtype=np.uint8)})
                            event.update(status='ok',initial_state_hash=initial_hash,final_state_hash=state,
                                encoder_hash=fitted_hash(enc),train_encoding_hash=array_hash(train),query_encoding_hash=array_hash(query),
                                head_before_hash=before_head,head_after_hash=after_head,hidden_before_hash=before_hidden,
                                hidden_after_hash=subset_state_hash(model,False),fit_seconds=model.training_seconds_,
                                encoding_seconds=encoding_seconds,encoder_fit_seconds=encoder_fit_seconds,query_encoding_seconds=query_encoding_seconds,inference_seconds=inference,
                                prediction_hash=array_hash(pred),layers=layer_diagnostics(model,query),
                                peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                            outputs[(scheme,seed,head,view)]=pred
                            if retain_models:models[(scheme,seed,head,view)]=(enc,model)
                        except Exception as exc:event.update(status='failed',exception=f'{type(exc).__name__}: {exc}')
                        fits.append(event)
                except Exception as exc:
                    for head in p['head_conditions']:
                        fits.append(dict(fit_id=f'{scheme}__s{seed}__h{int(head)}__v{view}',model_seed=seed,scheme=scheme,
                            head_update=head,view=view,**stream,status='failed',exception=f'{type(exc).__name__}: {exc}'))
    return {'fits':fits,'outputs':outputs,'models':models,'encoder_count':len(encoders)}


def workload_counts(p):
    outer=p['outer_folds']*p['outer_repeats'];seeds=len(p['fit_seeds']);inner=p['inner_folds']
    e06=sum(len(s.candidates)*inner+(min(3,len(s.candidates))*(seeds-1)*inner if s.stochastic else 0)
            +(seeds if s.stochastic else 1) for s in e06_registry(p).values())
    features={'iris':4,'wine':13,'digits':64}
    degrees=sum(sum(r['eligible'] for r in degree_schedule(features[name],p['degrees'],p['degree_column_cap'])) for name in p['panels']['e07'])
    return {'e05_network_fits':len(p['panels']['e05'])*outer*seeds*len(p['e05_schemes'])*len(p['head_conditions'])*p['e05_views'],
            'e06_conventional_fits':len(p['panels']['e06'])*outer*e06,'e06_af_fits':len(p['panels']['e06'])*outer*2*seeds,
            'e07_af_fits':degrees*outer*seeds,'e07_outer_encoders':degrees*outer,'e07_inner_encoders':degrees*outer*inner}


class EncodedReadout:
    def __init__(self,encoder,readout):self.encoder_=encoder;self.readout_=readout
    def predict(self,X):
        start=time.perf_counter();orders=self.encoder_.transform(X)
        self.last_encoding_seconds_=time.perf_counter()-start
        prediction=self.readout_.predict(orders)
        self.last_encoding_seconds_+=getattr(self.readout_,'last_encoding_seconds_',0.)
        return prediction


def make_bank(train,query,p,dataset,repeat,fold):
    return CorruptionBank.create(train,query,dataset_id=dataset,outer_repeat=repeat,outer_fold=fold,
        base_seeds=p['corruption_seeds'],gaussian_levels=p['gaussian_levels'],
        quantization_steps=p['quantization_steps'],masking_probabilities=p['masking_probabilities'])


def encoder_seed(p,dataset,split,inner=-1):
    return derive_seed(p['encoder_seed'],dataset,split['outer_repeat'],split['outer_fold'],inner,2,'random',0)


def model_seed(p,dataset,split,seed):
    return derive_seed(seed,'network',dataset,split['outer_repeat'],split['outer_fold'],-1,p['architectures'][p['primary_architecture_index']])


def save_encoder(writer,name,encoder,train,query,train_ids,query_ids):
    arrays={'train':train,'query':query,'train_ids':np.asarray(train_ids),'query_ids':np.asarray(query_ids),
            'serialized_encoder':np.frombuffer(pickle.dumps(encoder,protocol=5),dtype=np.uint8)}
    if hasattr(encoder,'poly_') and encoder.poly_ is not None:
        arrays.update(powers=encoder.poly_.powers_,projection=encoder.projection_,mean=encoder.scaler_.mean_,scale=encoder.scaler_.scale_)
    writer.save(name,arrays)


def fit_corruption_models(X,y,split,p,*,dataset_id,writer=None):
    """Clean inner selection and final fitting only; query labels unused."""
    validate_split(split,len(y));writer=writer or ArtifactWriter();models={};fits=[];selections={};failures=[];partitions=[]
    train=split['train'];query=split['test'];registry=e06_registry(p)
    for name,spec in registry.items():
        try:
            selected=select_model(X,y,split,spec,p['fit_seeds']);selections[name]=selected
            for seed in p['fit_seeds'] if spec.stochastic else p['fit_seeds'][:1]:
                start=time.perf_counter();model=spec.factory(selected['config'],seed).fit(X[train],y[train])
                event=dict(fit_id=f'{name}__s{seed}',model_id=name,model_seed=seed,config=selected['config'],config_id=selected['config_id'],
                    fit_seconds=time.perf_counter()-start,encoding_seconds=model.encoding_seconds_,classifier_fit_seconds=model.classifier_fit_seconds_,
                    fit_warnings=model.fit_warnings_,state_hash=fitted_hash(model),status='ok')
                writer.save(f'models/{event["fit_id"]}.npz',{'serialized_model':np.frombuffer(pickle.dumps(model,protocol=5),dtype=np.uint8)})
                fits.append(event);models[(name,seed)]=model
        except Exception as exc:
            if hasattr(exc,'fits'):selections[name]={'status':'failed','fits':exc.fits}
            failures.append({'model_id':name,'exception':f'{type(exc).__name__}: {exc}'})
    for name in ('row_af','projected_af'):
        try:
            start=time.perf_counter()
            enc=(RowRankingEncoder() if name=='row_af' else OrdinalEncoder('random',p['embed_dim'],2,.3,encoder_seed(p,dataset_id,split))).fit(X[train])
            orders=enc.transform(X[train]);query_orders=enc.transform(X[query]);encoding_seconds=time.perf_counter()-start
            partition={'partition_id':name,'train_ids':train,'query_ids':query,'raw_train_hash':array_hash(X[train]),
                'raw_query_hash':array_hash(X[query]),'training_labels_hash':array_hash(y[train]),'encoder_hash':fitted_hash(enc),
                'train_encoding_hash':array_hash(orders),'query_encoding_hash':array_hash(query_orders),'encoding_seconds':encoding_seconds}
            save_encoder(writer,f'encoders/{name}.npz',enc,orders,query_orders,train,query);partitions.append(partition)
            for seed in p['fit_seeds']:
                net_seed=model_seed(p,dataset_id,split,seed)
                model=fixed_network(p,orders.shape[1],net_seed).fit_orders(orders,y[train]);model.encoder_=enc
                event=dict(fit_id=f'{name}__s{seed}',model_id=name,model_seed=seed,network_seed=net_seed,
                    config={'widths':p['architectures'][p['primary_architecture_index']],'iterations':p['iterations']},
                    fit_seconds=model.training_seconds_,encoding_seconds=encoding_seconds,state_hash=fitted_hash(model),status='ok')
                event['config_id']=config_id(event['config'])
                writer.save(f'models/{event["fit_id"]}.npz',model.state_snapshot());fits.append(event);models[(name,seed)]=model
        except Exception as exc:failures.append({'model_id':name,'exception':f'{type(exc).__name__}: {exc}'})
    return {'models':models,'fits':fits,'selections':selections,'partitions':partitions,'failures':failures}


def predict_cases(models,cases):
    result={}
    for key,model in models.items():
        state=fitted_hash(model)
        for case in cases:
            if array_hash(case.raw)!=case.raw_hash:raise ValueError('Shared raw corruption changed')
            result[(*key,case.case_id)]=model.predict(case.raw)
            if fitted_hash(model)!=state:raise ValueError('Frozen model or encoder changed under corruption')
    return result


def fit_degrees(X,y,split,p,*,dataset_id,writer=None):
    validate_split(split,len(y));writer=writer or ArtifactWriter();models={};fits=[];selections={};partitions=[];failures=[];encoders=[]
    schedule=degree_schedule(X.shape[1],p['degrees'],p['degree_column_cap']);inner_count=0
    for record in schedule:
        degree=record['degree']
        if not record['eligible']:continue
        try:
            scores=[];selections[f'd{degree}_footrule']={'status':'running','fits':scores}
            for inner,indices in enumerate(split['inner']):
                train=indices['train'];query=indices['validation'];start=time.perf_counter()
                enc=NestedDegreeEncoder(p['embed_dim'],degree,encoder_seed(p,dataset_id,split,inner),p['degree_column_cap']).fit(X[train])
                orders=enc.transform(X[train]);query_orders=enc.transform(X[query]);inner_count+=1
                partition=dict(partition_id=f'd{degree}_inner{inner}',degree=degree,inner_fold=inner,train_ids=train,query_ids=query,
                    encoder_seed=enc.seed,encoder_hash=fitted_hash(enc),raw_train_hash=array_hash(X[train]),raw_query_hash=array_hash(X[query]),
                    training_labels_hash=array_hash(y[train]),train_encoding_hash=array_hash(orders),query_encoding_hash=array_hash(query_orders),
                    encoding_seconds=time.perf_counter()-start)
                save_encoder(writer,f'encoders/{partition["partition_id"]}.npz',enc,orders,query_orders,train,query);partitions.append(partition)
                knn=StableFootruleKNN(max(p['probe_neighbors'])).fit(orders,y[train],sample_ids=train)
                distances,indices_=knn.kneighbors(query_orders)
                for cfg in probe_candidates(p):
                    knn.n_neighbors=cfg['n_neighbors'];knn.weights=cfg['weights'];pred=knn.predict_neighbors(distances,indices_)
                    scores.append(dict(config=cfg,config_id=config_id(cfg),inner_fold=inner,model_seed=p['fit_seeds'][0],state='input',
                        score=float(np.mean(pred==y[query])),status='ok',partition_id=partition['partition_id'],prediction_hash=array_hash(pred)))
            chosen=select_symmetric_probe(scores,probe_candidates(p),inner_folds=p['inner_folds'],seeds=p['fit_seeds'][:1],states=('input',))
            selections[f'd{degree}_footrule']=dict(chosen,fits=scores)
            train=split['train'];query=split['test'];start=time.perf_counter()
            enc=NestedDegreeEncoder(p['embed_dim'],degree,encoder_seed(p,dataset_id,split),p['degree_column_cap']).fit(X[train])
            orders=enc.transform(X[train]);query_orders=enc.transform(X[query]);encoding_seconds=time.perf_counter()-start
            if encoders:
                previous=encoders[-1];n=len(previous.poly_.powers_)
                if (not np.array_equal(previous.poly_.powers_,enc.poly_.powers_[:n]) or not np.array_equal(previous.projection_,enc.projection_[:n])
                    or not np.allclose(previous.scaler_.mean_,enc.scaler_.mean_[:n],rtol=0,atol=1e-12)
                    or not np.allclose(previous.scaler_.scale_,enc.scaler_.scale_[:n],rtol=0,atol=1e-12)):
                    raise ValueError('Degree common monomials/scaling/projection changed')
            encoders.append(enc)
            partition=dict(partition_id=f'd{degree}_outer',degree=degree,inner_fold=-1,train_ids=train,query_ids=query,
                encoder_seed=enc.seed,encoder_hash=fitted_hash(enc),raw_train_hash=array_hash(X[train]),raw_query_hash=array_hash(X[query]),
                training_labels_hash=array_hash(y[train]),train_encoding_hash=array_hash(orders),query_encoding_hash=array_hash(query_orders),encoding_seconds=encoding_seconds)
            save_encoder(writer,f'encoders/{partition["partition_id"]}.npz',enc,orders,query_orders,train,query);partitions.append(partition)
            start=time.perf_counter();readout=StableFootruleKNN(**chosen['config']).fit(orders,y[train],sample_ids=train)
            model=EncodedReadout(enc,readout);key=(f'd{degree}_footrule',p['fit_seeds'][0]);models[key]=model
            fits.append(dict(fit_id=f'{key[0]}__s{key[1]}',model_id=key[0],model_seed=key[1],config=chosen['config'],config_id=chosen['config_id'],
                fit_seconds=time.perf_counter()-start,encoding_seconds=encoding_seconds,state_hash=fitted_hash(model),status='ok'))
            writer.save(f'models/{key[0]}__s{key[1]}.npz',{'serialized_model':np.frombuffer(pickle.dumps(model,protocol=5),dtype=np.uint8)})
            for seed in p['fit_seeds']:
                net_seed=model_seed(p,dataset_id,split,seed);model=fixed_network(p,orders.shape[1],net_seed).fit_orders(orders,y[train]);model.encoder_=enc
                key=(f'd{degree}_af',seed);models[key]=model
                cfg={'degree':degree,'widths':p['architectures'][p['primary_architecture_index']],'iterations':p['iterations']}
                fits.append(dict(fit_id=f'{key[0]}__s{seed}',model_id=key[0],model_seed=seed,network_seed=net_seed,config=cfg,config_id=config_id(cfg),
                    fit_seconds=model.training_seconds_,encoding_seconds=encoding_seconds,state_hash=fitted_hash(model),status='ok'))
                writer.save(f'models/{key[0]}__s{seed}.npz',model.state_snapshot())
        except Exception as exc:
            selections.setdefault(f'd{degree}_footrule',{})['status']='failed'
            failures.append({'degree':degree,'exception':f'{type(exc).__name__}: {exc}'})
    return {'models':models,'fits':fits,'selections':selections,'partitions':partitions,'failures':failures,'degree_schedule':schedule,
            'encoders':encoders,'inner_encoder_count':inner_count,'outer_encoder_count':len(encoders)}


def unique_array_payload(model,content_hash=False):
    """Unique allocated ndarray bytes; excludes Python containers/string objects."""
    seen=set();buffers=set();total=0;hashes=[];stack=[model];retained=[]
    while stack:
        value=stack.pop()
        if id(value) in seen:continue
        seen.add(id(value));retained.append(value)
        if isinstance(value,np.ndarray):
            while isinstance(value.base,np.ndarray):value=value.base
            buffer=(value.__array_interface__['data'][0],value.nbytes)
            if buffer not in buffers:buffers.add(buffer);total+=value.nbytes;hashes.append(array_hash(value))
        elif isinstance(value,dict):stack.extend(value.values())
        elif isinstance(value,(list,tuple)):stack.extend(value)
        elif hasattr(value,'__dict__'):stack.append(vars(value))
        elif not isinstance(value,(str,int,float,bool,type(None))) and hasattr(value,'__getstate__'):stack.append(value.__getstate__())
    return config_id(sorted(hashes)) if content_hash else total


class ViewEnsemble:
    def __init__(self,models):
        self.models=models
        self.classifier_fit_seconds_=sum(model.training_seconds_ for enc,model in models)
    def predict(self,X):
        predictions=[];encoding=0.
        for enc,model in self.models:
            start=time.perf_counter();orders=enc.transform(X);encoding+=time.perf_counter()-start
            predictions.append(model.predict_orders(orders))
        self.last_encoding_seconds_=encoding
        return majority(predictions)


def measure_cost(model,X_query,p,*,fit_seconds,encoder_fit_seconds=None,training_samples=None):
    """One warm-up, five complete predictions; no query-label argument exists."""
    hash_function=fitted_hash;scope='fitted parameters excluding query counters and timing'
    try:before=hash_function(model)
    except Exception:
        hash_function=lambda model:unique_array_payload(model,content_hash=True)
        before=hash_function(model);scope='array-content fallback after unsupported serialization'
    for _ in range(p['cost_warmups']):model.predict(X_query)
    totals=[];encodings=[];hashes=[]
    for _ in range(p['cost_repetitions']):
        start=time.perf_counter();prediction=model.predict(X_query);totals.append(time.perf_counter()-start)
        encodings.append(float(getattr(model,'last_encoding_seconds_',0.)))
        hashes.append(array_hash(np.asarray(prediction)))
    if before!=hash_function(model) or len(set(hashes))!=1:raise ValueError('Cost repetitions changed frozen model or predictions')
    serialization_error=None
    try:serialized_bytes=len(pickle.dumps(model,protocol=5))
    except Exception as exc:serialized_bytes=None;serialization_error=f'{type(exc).__name__}: {exc}'
    classifier=model.readout_ if isinstance(model,EncodedReadout) else model.steps[-1][1] if hasattr(model,'steps') else model
    training_batch=(getattr(classifier,'batch_size',None) if isinstance(classifier,ArrowFlowEstimator) or type(classifier).__name__ in ('MLPClassifier','OrderedPositionHDC') else None)
    if isinstance(model,ViewEnsemble):training_batch=model.models[0][1].batch_size
    if type(classifier).__name__=='MLPClassifier' and training_batch=='auto' and training_samples is not None:training_batch=min(200,training_samples)
    if isinstance(training_batch,int) and training_samples is not None:training_batch=min(training_batch,training_samples)
    inference_batch=min(getattr(classifier,'batch_size',len(X_query)),len(X_query)) if type(classifier).__name__ in ('StableFootruleKNN','OrderedPositionHDC') else len(X_query)
    networks=([m for enc,m in model.models] if isinstance(model,ViewEnsemble) else [model])
    prototypes=sum(sum(layer.index_matrix.size for layer in n.network_.graph.vertex_list.values()) for n in networks if hasattr(n,'network_'))
    return {'fit_seconds':fit_seconds,'encoder_fit_seconds':encoder_fit_seconds if encoder_fit_seconds is not None else getattr(model,'encoding_seconds_',None),
        'classifier_fit_seconds':getattr(model,'classifier_fit_seconds_',getattr(model,'training_seconds_',None)),
        'warmups':p['cost_warmups'],'repetitions':p['cost_repetitions'],'prediction_seconds':totals,
        'query_encoding_seconds':encodings,'classifier_inference_seconds':(np.array(totals)-encodings).tolist(),
        'median_prediction_seconds':float(np.median(totals)),'median_query_encoding_seconds':float(np.median(encodings)),
        'median_classifier_inference_seconds':float(np.median(np.array(totals)-encodings)),
        'query_batch_size':len(X_query),'internal_inference_batch_size':inference_batch,
        'training_batch_size':training_batch,'training_samples':training_samples,'numeric_threads':1,
        'serialized_bytes':serialized_bytes,'serialization_protocol':5,'serialization_error':serialization_error,
        'unique_array_payload_bytes':unique_array_payload(model),'array_payload_scope':'unique ndarray allocations; excludes Python containers and strings',
        'prototype_entries':prototypes,'whole_process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'prediction_hash':hashes[0],'fitted_state_hash':before,'fitted_state_hash_scope':scope,'fit_warnings':getattr(model,'fit_warnings_',[])}
