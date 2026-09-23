"""Paired-study isolation and complete-evidence checks on synthetic fixtures."""
import copy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api(): return importlib.import_module('experiments.make_revision.matched')


def tiny_protocol():
    from experiments.make_revision.run_studies import STUDY_PROTOCOL
    p=json.loads(STUDY_PROTOCOL.read_text())
    p.update(outer_folds=3,outer_repeats=1,inner_folds=2,
             architectures=[[4],[4,3],[8]],native_architectures=[[4],[4,3],[8]],
             embed_dim=6,iterations=4,hdc_dimensions=[8,16])
    return p


def toy():
    rng=np.random.RandomState(33)
    return rng.randn(60,4),np.tile([0,1,2],20)


def test_initial_copy_is_identical_and_training_does_not_mutate_control():
    from experiments.make_revision.models import ArrowFlowEstimator
    X=np.array([np.roll(np.arange(6),i%6) for i in range(18)]);y=np.tile([0,1,2],6)
    initial=ArrowFlowEstimator(embed_dim=6,widths=(4,3),iterations=4,seed=19).initialize_orders(X,y)
    before=initial.state_snapshot(); digest=initial.state_hash()
    trained=copy.deepcopy(initial)
    assert trained.state_hash()==digest
    for key in before: assert before[key].tobytes()==trained.state_snapshot()[key].tobytes()
    trained.train_initialized(X,y)
    assert initial.state_hash()==digest and trained.initial_state_hash_==digest
    assert trained.network_.update_iter==4 and initial.network_.update_iter==0


@pytest.mark.parametrize('widths',[(12,),(12,6)])
def test_every_depth_matches_actual_native_forward_inputs(widths,monkeypatch):
    from experiments.make_revision.models import ArrowFlowEstimator
    import scipy.spatial.distance
    X=np.array([np.roll(np.arange(8),i%8) for i in range(24)]);y=np.tile([0,1,2],8)
    model=ArrowFlowEstimator(embed_dim=8,widths=widths,iterations=4,seed=19).fit_orders(X,y)
    actual=[]; original=scipy.spatial.distance.cdist
    def capture(a,b,*args,**kwargs):
        actual.append(a.copy());return original(a,b,*args,**kwargs)
    monkeypatch.setattr(scipy.spatial.distance,'cdist',capture)
    model.predict_orders(X)
    depths=model.transform_orders_by_depth(X)
    assert len(depths)==len(widths)
    for depth,representation in enumerate(depths): assert np.array_equal(representation,actual[depth+1])


def test_symmetric_probe_selection_requires_both_states_and_all_inner_seeds():
    m=api(); configs=[{'n_neighbors':1,'weights':'uniform'},{'n_neighbors':3,'weights':'uniform'}]
    from experiments.make_revision.evaluation import config_id
    rows=[]
    for fold in range(2):
        for seed in [11,22,33]:
            for state in ['trained','untrained']:
                for cfg in configs:
                    score=(.9 if state=='trained' else .1) if cfg['n_neighbors']==1 else .6
                    rows.append(dict(inner_fold=fold,model_seed=seed,state=state,config_id=config_id(cfg),config=cfg,score=score,status='ok'))
    selected=m.select_symmetric_probe(rows,configs,inner_folds=2,seeds=[11,22,33])
    assert selected['config']['n_neighbors']==3
    swapped=[dict(r,state='untrained' if r['state']=='trained' else 'trained') for r in rows]
    assert m.select_symmetric_probe(swapped,configs,inner_folds=2,seeds=[11,22,33])['config_id']==selected['config_id']
    with pytest.raises(ValueError,match='complete'):
        m.select_symmetric_probe(rows[:-1],configs,inner_folds=2,seeds=[11,22,33])


def test_partition_encoder_shared_once_and_network_fit_count_independent_of_probe_grid(monkeypatch):
    m=api();p=tiny_protocol();X,y=toy()
    from experiments.make_revision.models import OrdinalEncoder,ArrowFlowEstimator
    fits=[]; encoders=[]; original=ArrowFlowEstimator.train_initialized; enc_fit=OrdinalEncoder.fit
    def fit(self,orders,labels): fits.append((self.widths,self.seed,len(labels)));return original(self,orders,labels)
    def fit_encoder(self,X,y=None): encoders.append(X.copy());return enc_fit(self,X,y)
    monkeypatch.setattr(ArrowFlowEstimator,'train_initialized',fit)
    monkeypatch.setattr(OrdinalEncoder,'fit',fit_encoder)
    cache=m.build_partition(X[:40],y[:40],X[40:],np.arange(40),np.arange(40,60),
        dataset_id='synthetic',outer_repeat=0,outer_fold=0,inner_fold=0,protocol=p)
    assert len(encoders)==1 and len(fits)==9
    assert not cache['shared'].train.flags.writeable and not cache['shared'].test.flags.writeable
    scores=m.score_inner_partition(cache,y[:40],y[40:],p)
    assert len(fits)==9 and len(encoders)==1
    assert len([r for r in scores if r['kind']=='probe'])==4*2*3*10
    hashes={(r['source_train_hash'],r['source_query_hash']) for r in cache['events'] if r['stage'] in ['network_pair','hdc_bundle']}
    assert hashes=={(cache['shared'].train_hash,cache['shared'].test_hash)}
    assert all(r['initial_hash']==r['trained_start_hash']==r['initial_after_hash'] for r in cache['events'] if r['stage']=='network_pair')


def test_fixed_indices_outer_label_perturbation_cannot_change_selection_or_fitted_state():
    m=api();p=tiny_protocol();X,y=toy()
    from experiments.make_revision.evaluation import make_splits
    split=make_splits(y,3,1,2,31)[0]
    a=m.evaluate_study_fold(X,y,split,p,dataset_id='synthetic',dataset_hash='fixed_features',code_revision='test')
    changed=y.copy();changed[split['test']]=(changed[split['test']]+1)%3
    b=m.evaluate_study_fold(X,changed,split,p,dataset_id='synthetic',dataset_hash='fixed_features',code_revision='test')
    assert a['status']==b['status']=='ok'
    assert a['selected']==b['selected']
    assert [r['y_pred'] for r in a['predictions']]==[r['y_pred'] for r in b['predictions']]
    assert [r.get('final_state_hash') for r in a['models']]==[r.get('final_state_hash') for r in b['models']]
    assert sum(r['stage']=='network_pair' for r in a['events'])==27
    assert sum(r['stage']=='encoding' for r in a['events'])==3
    assert all(not set(part['train_ids'])&set(split['test']) for part in a['partitions'])


def test_native_partition_uses_original_orders_without_numeric_encoder(monkeypatch):
    m=api();p=tiny_protocol();rng=np.random.RandomState(3)
    X=np.array([rng.permutation(10) for _ in range(24)]);y=np.tile([0,1],12)
    from experiments.make_revision.models import OrdinalEncoder
    def forbidden(*args,**kwargs): raise AssertionError('native data must bypass numerical encoder')
    monkeypatch.setattr(OrdinalEncoder,'fit',forbidden)
    cache=m.build_partition(X[:16],y[:16],X[16:],np.arange(16),np.arange(16,24),
        dataset_id='native_synthetic',outer_repeat=0,outer_fold=0,inner_fold=0,protocol=p,native=True)
    assert np.array_equal(cache['shared'].train,X[:16]) and not cache['failures']


def test_hdc_bundle_cache_reuses_exact_arrays_for_same_rank_hash_and_seeds():
    m=api();from experiments.make_revision.comparisons import OrderedPositionHDC
    X=np.array([np.roll(np.arange(6),i) for i in range(6)])
    model=OrderedPositionHDC(dimension=8,seed=19).fit_encoder(X)
    cache=m.BundleCache();a=cache.get(model,X);b=cache.get(model,X.copy())
    assert a is b and not a.flags.writeable and cache.computations==1
    other=OrderedPositionHDC(dimension=8,seed=20).fit_encoder(X)
    cache.get(other,X);assert cache.computations==2


def test_failed_network_produces_full_terminal_expected_model_seed_matrix():
    m=api();p=tiny_protocol();p['architectures'][0]=[0];X,y=toy()
    from experiments.make_revision.evaluation import make_splits
    result=m.evaluate_study_fold(X,y,make_splits(y,3,1,2,31)[0],p,
                                dataset_id='synthetic',dataset_hash='test',code_revision='test')
    expected=m.model_seed_schedule(p)
    assert result['status']=='failed' and not result['predictions']
    assert len(result['models'])==sum(map(len,expected.values()))
    assert all(r['status']=='failed' for r in result['models'])
    assert any(r['status']=='failed' for r in result['events'])


def test_cached_neighbor_predictions_equal_direct_stable_knn_for_all_probe_choices():
    from experiments.make_revision.comparisons import StableFootruleKNN
    rng=np.random.RandomState(4); X=np.array([rng.permutation(6) for _ in range(30)]);y=np.tile([0,1,2],10)
    query=X[[3,9,12,18]];ids=rng.permutation(np.arange(30))
    model=StableFootruleKNN(n_neighbors=21).fit(X,y,sample_ids=ids)
    distances,indices=model.kneighbors(query)
    for k in [1,3,5,11,21]:
        for weights in ['uniform','distance']:
            model.n_neighbors=k;model.weights=weights
            direct=StableFootruleKNN(k,weights).fit(X,y,sample_ids=ids).predict(query)
            assert np.array_equal(model.predict_neighbors(distances,indices),direct)


def test_study_cli_blocks_unfrozen_confirmation(tmp_path):
    from experiments.make_revision.run_studies import main
    with pytest.raises(ValueError,match='frozen'):
        main(['run','--output',str(tmp_path)])


def test_synthetic_spawn_smoke_seals_artifacts_and_rejects_missing_evidence(tmp_path):
    from experiments.make_revision.run_studies import main,collect_study_results
    main(['smoke','--output',str(tmp_path),'--workers','2'])
    result=collect_study_results(tmp_path,allow_smoke=True)
    assert len(result['synthetic'])==3*38
    files=sorted((tmp_path/'results').glob('*.json'))
    assert len(files)==3
    record=json.loads(files[0].read_text())
    assert len(record['predictions'])==20*38
    artifact=tmp_path/'artifacts'/files[0].stem/record['artifacts'][0]['path']
    original=artifact.read_bytes();artifact.write_bytes(original+b'corruption')
    with pytest.raises(ValueError,match='artifact'): collect_study_results(tmp_path,allow_smoke=True)
    artifact.write_bytes(original)
    saved=files[0].read_text()
    record['artifacts']=[]
    files[0].write_text(json.dumps(record))
    with pytest.raises(ValueError,match='artifact'): collect_study_results(tmp_path,allow_smoke=True)
    files[0].write_text(saved)
    originals={path:path.read_text() for path in files}
    for path in files:
        incomplete=json.loads(originals[path])
        incomplete['models']=[r for r in incomplete['models'] if not (r['model_id']=='hdc' and r['model_seed']==39019)]
        path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError,match='seed'): collect_study_results(tmp_path,allow_smoke=True)
    for path,content in originals.items(): path.write_text(content)
    # Delete one whole paired fold; an available-file glob may not redefine m.
    files[0].unlink()
    with pytest.raises(ValueError,match='missing result'): collect_study_results(tmp_path,allow_smoke=True)


def test_shuffle_uses_only_inner_training_labels_and_original_inner_validation(tmp_path):
    from experiments.make_revision.run_studies import shuffle_prerequisite
    from experiments.make_revision.run_revision import load_dataset
    from experiments.make_revision.evaluation import make_splits
    p=tiny_protocol();X,y,_=load_dataset('iris'); split=make_splits(y,p['outer_folds'],p['outer_repeats'],p['inner_folds'],p['split_seed'])[0]
    report=shuffle_prerequisite(tmp_path,p)
    inner=split['inner'][0]
    assert report['train_ids']==inner['train'] and report['query_ids']==inner['validation']
    assert not (set(report['train_ids'])|set(report['query_ids']))&set(split['test'])
    expected=np.random.RandomState(109387).permutation(y[inner['train']])
    from experiments.make_revision.models import array_hash
    assert report['shuffled_labels_hash']==array_hash(expected)
    assert [r['y_true'] for r in report['predictions'] if r['model_id']=='shuffle_arrowflow']==y[inner['validation']].tolist()
    assert report['purpose']=='inner_label_shuffle_prerequisite_not_outer_evidence'


def test_runtime_probe_grid_does_not_compute_accuracy():
    m=api();p=tiny_protocol();X,y=toy()
    cache=m.build_partition(X[:40],y[:40],X[:10],np.arange(40),np.arange(10),
        dataset_id='synthetic',outer_repeat=0,outer_fold=0,inner_fold=-1,protocol=p)
    rows=m.score_inner_partition(cache,y[:40],None,p)
    assert rows and all('score' not in r for r in rows)


def test_worker_persists_failed_selection_and_refuses_collision(tmp_path):
    from experiments.make_revision.run_studies import prepare_study,study_worker
    from experiments.make_revision.evaluation import dataset_fingerprint
    X,y=toy();p=tiny_protocol();p['architectures'][0]=[0]
    names=['a','b','c','d'];labels=['0','1','2']
    manifest={'dataset_id':'synthetic','feature_names':names,'label_map':labels,
              'dataset_hash':dataset_fingerprint(X,y,names,labels)}
    prepare_study(tmp_path,['synthetic'],p,purpose='synthetic_smoke_only',data_loader=lambda name:(X,y,manifest))
    path=Path(study_worker((str(tmp_path),'synthetic',0,0)))
    result=json.loads(path.read_text())
    events=[json.loads(line) for line in (tmp_path/'logs'/'synthetic__r0f0.jsonl').read_text().splitlines()]
    assert result['status']=='failed' and events==result['events']
    assert any(e['stage']=='terminal_failure' for e in events)
    assert len(result['models'])==38
    with pytest.raises(FileExistsError): study_worker((str(tmp_path),'synthetic',0,0))


@pytest.fixture(scope='module')
def coherent_study(tmp_path_factory):
    from experiments.make_revision.run_studies import main
    root=tmp_path_factory.mktemp('coherent_matched')
    main(['smoke','--output',str(root),'--workers','2'])
    return root


@pytest.mark.parametrize('corruption',[
    'model_config_id','prediction_config_id','model_revision','prediction_revision',
    'inner_outputs','partitions','inner_outputs_and_log','inner_borda_and_log',
    'network_event_and_log','hdc_event_and_log','partition_ids_and_log',
    'partition_raw_hash_and_log','partition_encoded_hash_and_log',
    'model_config_and_log','model_source_hash_and_log','inner_config_id_and_log',
    'network_seed_and_log','hdc_seed_and_log','wrong_model_fit_event_and_log',
])
def test_collector_rejects_incoherent_provenance(coherent_study,corruption):
    from experiments.make_revision.run_studies import collect_study_results
    from experiments.make_revision.evaluation import canonical_json,config_id
    root=coherent_study;path=sorted((root/'results').glob('*.json'))[0]
    log=root/'logs'/(path.stem+'.jsonl')
    original=path.read_text();original_log=log.read_text();result=json.loads(original)
    if corruption=='model_config_id':result['models'][0]['config_id']='wrong'
    elif corruption=='prediction_config_id':result['predictions'][0]['config_id']='wrong'
    elif corruption=='model_revision':result['models'][0]['code_revision']='wrong'
    elif corruption=='prediction_revision':result['predictions'][0]['code_revision']='wrong'
    elif corruption in ('inner_outputs','inner_outputs_and_log','inner_borda_and_log'):
        kind='borda' if corruption=='inner_borda_and_log' else 'output'
        result['inner_scores']=[r for r in result['inner_scores'] if r['kind']!=kind]
        if corruption.endswith('and_log'):
            result['events']=[r for r in result['events'] if not(r['stage']=='inner_score' and r['kind']==kind)]
    elif corruption=='partitions':result['partitions']=[]
    elif corruption in ('network_event_and_log','hdc_event_and_log'):
        stage='network_pair' if corruption.startswith('network') else 'hdc_bundle'
        idx=next(i for i,r in enumerate(result['events']) if r['stage']==stage)
        result['events'].pop(idx)
    elif corruption in ('network_seed_and_log','hdc_seed_and_log'):
        stage='network_pair' if corruption.startswith('network') else 'hdc_bundle'
        row=next(r for r in result['events'] if r['stage']==stage)
        row['network_seed' if stage=='network_pair' else 'item_seed']=123
    elif corruption=='wrong_model_fit_event_and_log':
        row=result['models'][0]
        row['fit_event']=next(r for r in result['events'] if r['stage']=='network_pair' and r['partition_id']=='outer' and r['model_seed']==row['model_seed'] and r['architecture']!=row['fit_event']['architecture'])
        result['events']=[dict(row,stage='outer_model') if r['stage']=='outer_model' and (r['model_id'],r['model_seed'])==(row['model_id'],row['model_seed']) else r for r in result['events']]
    elif corruption.startswith('partition_'):
        meta=result['partitions'][0]
        if corruption=='partition_ids_and_log':meta['train_ids']=meta['train_ids'][::-1]
        elif corruption=='partition_raw_hash_and_log':meta['raw_train_hash']='wrong'
        else:meta['source_train_hash']='wrong'
        result['events']=[dict(meta,stage='encoding',status='ok') if r['stage']=='encoding' and r['partition_id']==meta['partition_id'] else r for r in result['events']]
    elif corruption.startswith('model_'):
        row=result['models'][0]
        if corruption=='model_config_and_log':
            row['config']={'n_neighbors':999,'weights':'uniform'};row['config_id']=config_id(row['config'])
            for pred in result['predictions']:
                if (pred['model_id'],pred['model_seed'])==(row['model_id'],row['model_seed']):pred['config_id']=row['config_id']
        else:row['source_train_hash']='wrong'
        result['events']=[dict(row,stage='outer_model') if r['stage']=='outer_model' and (r['model_id'],r['model_seed'])==(row['model_id'],row['model_seed']) else r for r in result['events']]
    elif corruption=='inner_config_id_and_log':
        row=next(r for r in result['inner_scores'] if r['kind']=='output');row['config_id']='wrong'
        result['events']=[dict(row,stage='inner_score') if r['stage']=='inner_score' and all(r.get(k)==row.get(k) for k in ('kind','inner_fold','architecture','model_seed')) else r for r in result['events']]
    try:
        path.write_text(json.dumps(result))
        if corruption.endswith('and_log'):log.write_text(''.join(canonical_json(r)+'\n' for r in result['events']))
        with pytest.raises(ValueError,match='Incomplete matched evidence'):
            collect_study_results(root,allow_smoke=True)
    finally:path.write_text(original);log.write_text(original_log)


def test_collector_retains_a_coherently_logged_nonselected_inner_failure(coherent_study):
    from experiments.make_revision.run_studies import collect_study_results
    from experiments.make_revision.evaluation import canonical_json
    root=coherent_study;path=sorted((root/'results').glob('*.json'))[0];log=root/'logs'/(path.stem+'.jsonl')
    original=path.read_text();original_log=log.read_text();result=json.loads(original)
    row=next(r for r in result['inner_scores'] if r['kind']=='input' and r['config_id']!=result['selected']['input_footrule']['config_id'])
    row.update(status='failed',score=None,exception='RuntimeError: deliberate retained candidate failure')
    result['events']=[dict(row,stage='inner_score') if r['stage']=='inner_score' and all(r.get(k)==row.get(k) for k in ('kind','config_id','inner_fold','model_seed')) else r for r in result['events']]
    try:
        path.write_text(json.dumps(result));log.write_text(''.join(canonical_json(r)+'\n' for r in result['events']))
        assert len(collect_study_results(root,allow_smoke=True)['synthetic'])==114
    finally:path.write_text(original);log.write_text(original_log)


# --- v3 matched study: per-dataset encoder settings and the validation checkpoint ---
V2_MATCHED='experiments/make_revision/protocols/2026-09-11/matched.json'
V3_MATCHED='experiments/make_revision/protocols/2026-09-12/matched_v3.json'


def test_matched_v3_uses_per_dataset_encoder_settings():
    import json
    from experiments.make_revision.matched import encoder_settings
    p = json.load(open('experiments/make_revision/protocols/2026-09-12/matched_v3.json'))
    assert encoder_settings(p, 'digits') == {'embed_dim': 64, 'degree': 1}
    assert encoder_settings(p, 'iris') == {'embed_dim': 16, 'degree': 3}
    assert p['architectures'][p['primary_architecture_index']] == [128]
    assert p['learning_rate'] == .1


def test_encoder_settings_fall_back_to_global_fields_without_table():
    m=api();old=json.loads(Path(V2_MATCHED).read_text());new=json.loads(Path(V3_MATCHED).read_text())
    from experiments.make_revision.datasets import SUSHI_ID
    assert 'encoder_by_dataset' not in old
    assert m.encoder_settings(old,'iris')=={'embed_dim':old['embed_dim'],'degree':old['degree']}=={'embed_dim':32,'degree':2}
    assert m.encoder_settings(tiny_protocol(),'synthetic')=={'embed_dim':6,'degree':2}
    assert SUSHI_ID not in new['encoder_by_dataset']
    assert m.encoder_settings(new,SUSHI_ID)=={'embed_dim':new['embed_dim'],'degree':new['degree']}
    settings=m.encoder_settings(new,'iris');settings['embed_dim']=999
    assert new['encoder_by_dataset']['iris']['embed_dim']==16


def test_matched_v3_protocol_changes_only_the_learning_configuration():
    import hashlib
    m=api();old=json.loads(Path(V2_MATCHED).read_text());new=json.loads(Path(V3_MATCHED).read_text())
    template_hash=hashlib.sha256(Path(V2_MATCHED).read_bytes()).hexdigest()
    # Freeze-time provenance is written by hand (bridge convention); every design field is pinned here.
    provenance={'frozen','frozen_at_utc','source_template_sha256','resource_decision','status'}
    changed={'architectures','primary_architecture_index','learning_rate','protocol_id'}
    added={'validation_ratio','encoder_by_dataset','wallclock_cap_hours'}
    def pin(p):
        assert set(p)-provenance==(set(old)-provenance)|added
        assert {k for k in set(old)-provenance if old[k]!=p[k]}==changed
        assert p['architectures']==[[128],[64,128],[64,32]] and p['primary_architecture_index']==0
        assert p['learning_rate']==.1 and p['iterations']==200 and p['validation_ratio']==.1
        assert p['protocol_id']=='arrowflow-v3-matched-1' and p['wallclock_cap_hours']==2
        assert p['datasets']==old['datasets'] and p['primary_datasets']==old['primary_datasets']
        assert p['native_architectures']==old['native_architectures'] and p['embed_dim']==32 and p['degree']==2
        assert set(p['encoder_by_dataset'])==set(p['primary_datasets'])
        assert p['encoder_by_dataset']=={'iris':{'embed_dim':16,'degree':3},'digits':{'embed_dim':64,'degree':1},
            **{name:{'embed_dim':32,'degree':2} for name in ('wine','breast_cancer','wine_quality','vehicle','segment')}}
        if p['frozen']:
            assert p['frozen_at_utc']>='2026-09-12' and p['source_template_sha256']==template_hash
        else:
            assert 'frozen_at_utc' not in p and 'source_template_sha256' not in p
    pin(new)
    pin(dict(new,frozen=True,frozen_at_utc='2026-09-12T00:00:00+00:00',source_template_sha256=template_hash,
             resource_decision=old['resource_decision']+'; pilot projection recorded at freeze'))
    with pytest.raises(AssertionError):
        pin(dict(new,frozen=True,frozen_at_utc='2026-09-12T00:00:00+00:00',source_template_sha256=old['source_template_sha256']))
    with pytest.raises(AssertionError): pin(dict(new,iterations=100))
    arch=m.architecture_id(new['architectures'][new['primary_architecture_index']])
    assert {f'{arch}_d1_trained',f'{arch}_d1_untrained',arch+'_output','input_footrule'}<=m.model_seed_schedule(new).keys()


def test_partition_encoder_and_seed_stream_follow_the_dataset_table():
    m=api();p=tiny_protocol();p.update(architectures=[[4]],fit_seeds=p['fit_seeds'][:1],hdc_dimensions=[8]);X,y=toy()
    from experiments.make_revision.comparisons import derive_seed
    table=dict(p,encoder_by_dataset={'synthetic':{'embed_dim':5,'degree':1}})
    def build(protocol,dataset):
        cache=m.build_partition(X[:40],y[:40],X[40:],np.arange(40),np.arange(40,60),
            dataset_id=dataset,outer_repeat=0,outer_fold=0,inner_fold=0,protocol=protocol)
        assert not cache['failures'];return cache
    cache=build(table,'synthetic');meta=cache['metadata']
    assert cache['shared'].train.shape[1]==5 and cache['shared'].test.shape[1]==5
    assert meta['encoder_parameters']=={'degree':1,'strategy':'random','embed_dim':5,'native':False}
    assert meta['encoder_seed']==derive_seed(p['encoder_seed'],'synthetic',0,0,0,1,'random',0)
    assert 'polynomial_powers' not in meta['encoder_array_hashes']
    # A dataset outside the table and the table-free protocol give identical partitions.
    tabled=build(table,'other')['metadata'];plain=build(p,'other')['metadata']
    assert tabled['encoder_parameters']==plain['encoder_parameters']=={'degree':2,'strategy':'random','embed_dim':6,'native':False}
    assert tabled['encoder_seed']==plain['encoder_seed']==derive_seed(p['encoder_seed'],'other',0,0,0,2,'random',0)
    assert tabled['source_train_hash']==plain['source_train_hash'] and tabled['encoder_array_hashes']==plain['encoder_array_hashes']


def test_native_partition_ignores_the_dataset_table():
    m=api();p=tiny_protocol();p.update(native_architectures=[[4]],fit_seeds=p['fit_seeds'][:1],hdc_dimensions=[8])
    rng=np.random.RandomState(3);X=np.array([rng.permutation(10) for _ in range(24)]);y=np.tile([0,1],12)
    def build(protocol):
        cache=m.build_partition(X[:16],y[:16],X[16:],np.arange(16),np.arange(16,24),
            dataset_id='native_synthetic',outer_repeat=0,outer_fold=0,inner_fold=0,protocol=protocol,native=True)
        assert not cache['failures'];return cache
    tabled=build(dict(p,encoder_by_dataset={'iris':{'embed_dim':16,'degree':3}}));plain=build(p)
    def stable(cache): return {k:v for k,v in cache['metadata'].items() if k!='encoding_seconds'}
    assert stable(tabled)==stable(plain) and tabled['metadata']['encoder_seed'] is None
    assert tabled['metadata']['encoder_parameters']=={'degree':2,'strategy':'random','embed_dim':10,'native':True}
    assert np.array_equal(tabled['shared'].train,X[:16])


def test_validation_ratio_threads_from_protocol_into_every_matched_network(monkeypatch):
    m=api();X,y=toy()
    from experiments.make_revision.models import ArrowFlowEstimator
    seen=[];original=ArrowFlowEstimator.train_initialized
    def train(self,orders,labels):
        result=original(self,orders,labels)
        seen.append((self.validation_ratio,self.validation_sample_count_,self.training_sample_count_));return result
    monkeypatch.setattr(ArrowFlowEstimator,'train_initialized',train)
    def build(protocol):
        cache=m.build_partition(X[:40],y[:40],X[40:],np.arange(40),np.arange(40,60),
            dataset_id='synthetic',outer_repeat=0,outer_fold=0,inner_fold=0,protocol=protocol)
        assert not cache['failures'];return [e for e in cache['events'] if e['stage']=='network_pair']
    p=tiny_protocol();p.update(architectures=[[4],[4,3]],hdc_dimensions=[8])
    assert 'validation_ratio' not in p
    events=build(p)
    assert len(seen)==6 and all(row==(0.,0,40) for row in seen)
    assert all((e['validation_sample_count'],e['network_training_sample_count'])==(0,40) for e in events)
    seen.clear();events=build(dict(p,validation_ratio=.25))
    assert len(seen)==6 and all(row==(.25,10,30) for row in seen)
    assert all((e['validation_sample_count'],e['network_training_sample_count'])==(10,30) for e in events)


def test_reconciliation_accepts_per_dataset_encoder_settings(tmp_path):
    m=api();p=tiny_protocol();p.update(architectures=[[4]],hdc_dimensions=[8]);X,y=toy()
    from experiments.make_revision.run_studies import reconcile_job_records
    from experiments.make_revision.evaluation import make_splits
    p['encoder_by_dataset']={'synthetic':{'embed_dim':5,'degree':1}}
    split=make_splits(y,3,1,2,31)[0]
    result=m.evaluate_study_fold(X,y,split,p,dataset_id='synthetic',dataset_hash='fixed',code_revision='test',artifact_dir=tmp_path)
    assert result['status']=='ok'
    assert all(part['encoder_parameters']['embed_dim']==5 and part['encoder_parameters']['degree']==1 for part in result['partitions'])
    job={'dataset_id':'synthetic','outer_repeat':split['outer_repeat'],'outer_fold':split['outer_fold']}
    reconcile_job_records(result,result['events'],p,job,split,X,y,tmp_path,'test')
    with pytest.raises(ValueError,match='encoder'):
        reconcile_job_records(result,result['events'],dict(p,encoder_by_dataset={}),job,split,X,y,tmp_path,'test')


def test_encoder_table_keys_must_name_protocol_datasets(tmp_path):
    m=api();from experiments.make_revision.run_studies import prepare_study,verify_prepared,write_json
    from experiments.make_revision.evaluation import dataset_fingerprint,config_id
    v3=json.loads(Path(V3_MATCHED).read_text());v2=json.loads(Path(V2_MATCHED).read_text())
    assert m.validate_encoder_table(v3)==v3['encoder_by_dataset'] and m.validate_encoder_table(v2)=={}
    typo=dict(v3,encoder_by_dataset={**v3['encoder_by_dataset'],'digit':v3['encoder_by_dataset']['digits']})
    with pytest.raises(ValueError,match="'digit'"):m.validate_encoder_table(typo)
    with pytest.raises(ValueError,match="'iris'"):m.validate_encoder_table(dict(v3,encoder_by_dataset={'iris':{'embed_dim':16}}))
    X,y=toy();features=[f'x{i}' for i in range(4)];labels=['0','1','2']
    manifest={'dataset_id':'synthetic','feature_names':features,'label_map':labels,'shape':[60,4],
              'class_counts':[20,20,20],'dataset_hash':dataset_fingerprint(X,y,features,labels)}
    loader=lambda name:(X,y,manifest)
    p=tiny_protocol();p.update(datasets=['synthetic'],encoder_by_dataset={'synthetc':{'embed_dim':5,'degree':1}})
    with pytest.raises(ValueError,match="'synthetc'"):prepare_study(tmp_path/'typo',['synthetic'],p,data_loader=loader)
    assert not (tmp_path/'typo').exists()          # refused before anything is sealed
    p['encoder_by_dataset']={'synthetic':{'embed_dim':5,'degree':1}}
    prepare_study(tmp_path/'ok',['synthetic'],p,purpose='synthetic_smoke_only',data_loader=loader)
    verify_prepared(tmp_path/'ok',allow_smoke=True)
    sealed=dict(p,encoder_by_dataset={'synthetc':{'embed_dim':5,'degree':1}})
    (tmp_path/'ok'/'protocol.json').write_text(json.dumps(sealed))
    manifest_path=tmp_path/'ok'/'run_manifest.json'
    manifest_path.write_text(json.dumps(dict(json.loads(manifest_path.read_text()),protocol_hash=config_id(sealed))))
    with pytest.raises(ValueError,match="'synthetc'"):verify_prepared(tmp_path/'ok',allow_smoke=True)
