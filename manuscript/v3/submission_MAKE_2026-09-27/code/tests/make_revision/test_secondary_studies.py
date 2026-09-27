"""Executable ablation identities, diagnostics and evidence schedules."""
import copy
import importlib
import json
import numpy as np
import pytest


def api():return importlib.import_module('experiments.make_revision.secondary_studies')


def protocol():
    from experiments.make_revision.run_secondary import PROTOCOL
    p=json.loads(PROTOCOL.read_text())
    p.update(outer_folds=3,outer_repeats=1,inner_folds=2,iterations=4,
             architectures=[[4],[4,3],[8]],embed_dim=6,e05_views=3,e05_prefixes=[1,3],
             candidate_budget=1,degrees=[1,2],corruption_seeds=[104729],
             gaussian_levels=[0,.1],quantization_steps=[0,.1],masking_probabilities=[0,.1],
             crossed_levels=[.1],probe_neighbors=[1,3],probe_weights=['uniform','distance'])
    return p


def toy():
    rng=np.random.RandomState(43)
    return rng.randn(60,4),np.tile([0,1,2],20)


def test_view_streams_match_first_view_and_exclude_scheme_head_from_network_seed():
    m=api();p=protocol()
    streams=[m.view_stream(p,'iris',0,0,8129,scheme,0) for scheme in p['e05_schemes']]
    assert streams[0]==streams[1]==streams[2]
    for view in range(3):
        values=[m.view_stream(p,'iris',0,0,8129,s,view) for s in p['e05_schemes']]
        assert len({v['network_seed'] for v in values})==1
    assert m.view_stream(p,'iris',0,0,8129,'same_projection',2)['encoder_seed']==streams[0]['encoder_seed']


def test_prefix_and_pair_diagnostics_match_explicit_oracles():
    m=api();pred=np.array([[0,0,1,1],[1,0,0,1],[1,1,0,1]]);truth=np.array([0,1,0,1])
    assert m.majority(pred[:2]).tolist()==[0,0,0,1]
    assert m.majority(pred).tolist()==[1,0,0,1]
    rows=m.error_diagnostics(pred,truth)
    a=rows[0];e0=pred[0]!=truth;e1=pred[1]!=truth
    assert a['disagreement']==np.mean(pred[0]!=pred[1])
    assert a['double_fault']==np.mean(e0&e1)
    assert a['error_correlation']==pytest.approx(np.corrcoef(e0,e1)[0,1])
    constant=m.error_diagnostics(np.stack([truth,truth]),truth)[0]
    assert constant['error_correlation'] is None and constant['correlation_reason']


def test_row_encoder_fits_only_training_statistics_and_native_pipeline_is_nested():
    m=api();X,y=toy();X[0,0]=np.nan
    encoder=m.RowRankingEncoder().fit(X[:40],y[:40]);means=encoder.imputer_.means_.copy()
    result=encoder.transform(np.full((2,4),1e9))
    assert np.array_equal(means,encoder.imputer_.means_)
    assert np.all(np.sort(result,axis=1)==np.arange(4))
    model=m.e06_registry(protocol())['row_svc_rbf'].factory({'C':1,'gamma':'scale'},8129).fit(X[:40],y[:40])
    assert np.allclose(model.named_steps['row_encoder'].imputer_.means_,np.nanmean(X[:40],axis=0))
    assert 'positions' in model.named_steps and 'scaler' in model.named_steps


def test_schedule_counts_match_prespecified_full_design():
    m=api();from experiments.make_revision.run_secondary import PROTOCOL
    p=json.loads(PROTOCOL.read_text());counts=m.workload_counts(p)
    assert counts['e05_network_fits']==5670
    assert counts['e06_conventional_fits']==12180 and counts['e06_af_fits']==180
    assert counts['e07_af_fits']==360 and counts['e07_outer_encoders']==120 and counts['e07_inner_encoders']==360


def test_head_pair_initial_identity_and_frozen_head_state(tmp_path):
    m=api();p=protocol();X,y=toy();p['fit_seeds']=p['fit_seeds'][:1];p['e05_schemes']=['same_projection'];p['e05_views']=1;p['e05_prefixes']=[1]
    result=m.fit_views(X[:40],y[:40],X[40:],p,dataset_id='synthetic',repeat=0,fold=0)
    events=result['fits'];assert len(events)==2 and all(r['status']=='ok' for r in events)
    assert events[0]['initial_state_hash']==events[1]['initial_state_hash']
    frozen=next(r for r in events if not r['head_update'])
    assert frozen['head_before_hash']==frozen['head_after_hash']
    assert frozen['hidden_before_hash']!=frozen['hidden_after_hash']
    assert events[0]['encoder_hash']==events[1]['encoder_hash']


def test_corruption_models_frozen_and_projected_af_matches_e03_seed():
    m=api();p=protocol();X,y=toy()
    from experiments.make_revision.evaluation import make_splits
    split=make_splits(y,3,1,2,31)[0]
    models=m.fit_corruption_models(X,y,split,p,dataset_id='synthetic')
    assert len(models['models'])==16 and not models['failures']
    bank=m.make_bank(X[split['train']],X[split['test']],p,'synthetic',0,0)
    before={key:m.fitted_hash(model) for key,model in models['models'].items()}
    predictions=m.predict_cases(models['models'],bank.cases)
    assert len(predictions)==16*len(bank.cases)
    assert before=={key:m.fitted_hash(model) for key,model in models['models'].items()}
    expected=m.derive_seed(8129,'network','synthetic',0,0,-1,p['architectures'][1])
    assert models['models'][('projected_af',8129)].seed==expected
    zero=[c for c in bank.cases if c.severity==0]
    for key in models['models']:
        clean=predictions[(*key,'clean__level0__seedNone')]
        assert all(np.array_equal(clean,predictions[(*key,c.case_id)]) for c in zero)


def test_degree_fits_no_inner_networks_and_label_perturbation_has_no_effect(monkeypatch):
    m=api();p=protocol();X,y=toy()
    from experiments.make_revision.evaluation import make_splits
    split=make_splits(y,3,1,2,31)[0];calls=[];original=m.ArrowFlowEstimator.train_initialized
    def fit(self,orders,labels):calls.append(len(labels));return original(self,orders,labels)
    monkeypatch.setattr(m.ArrowFlowEstimator,'train_initialized',fit)
    a=m.fit_degrees(X,y,split,p,dataset_id='synthetic')
    assert len(calls)==6 and set(calls)=={40}
    assert a['inner_encoder_count']==4 and a['outer_encoder_count']==2
    assert all(r['status']=='ok' for r in a['fits'])
    changed=y.copy();changed[split['test']]=(changed[split['test']]+1)%3
    b=m.fit_degrees(X,changed,split,p,dataset_id='synthetic')
    assert a['selections']==b['selections']
    assert {k:m.fitted_hash(v) for k,v in a['models'].items()}=={k:m.fitted_hash(v) for k,v in b['models'].items()}
    powers=[e.poly_.powers_ for e in a['encoders']];projections=[e.projection_ for e in a['encoders']]
    assert np.array_equal(powers[0],powers[1][:len(powers[0])])
    assert np.array_equal(projections[0],projections[1][:len(projections[0])])


def test_source_revision_is_independent_of_callers_working_directory(tmp_path,monkeypatch):
    from experiments.make_revision.run_revision import code_revision
    expected=code_revision();monkeypatch.chdir(tmp_path)
    assert code_revision()==expected


def test_secondary_cli_refuses_unfrozen_runs(tmp_path):
    from experiments.make_revision.run_secondary import main
    with pytest.raises(ValueError,match='frozen'):main(['run','--output',str(tmp_path)])


def test_spawned_secondary_smoke_has_complete_rows_and_rejects_omissions(tmp_path):
    from experiments.make_revision.run_secondary import main,collect_results
    main(['smoke','--output',str(tmp_path),'--workers','2'])
    validated=[];result=collect_results(tmp_path,allow_smoke=True,validated_jobs=validated)
    assert len(validated)==9 and all(r['status']=='ok' for r in validated)
    assert set(result)=={'e05','e06','e07'}
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['view_diagnostics']['prefix_summaries']
    assert summary['view_diagnostics']['pair_summaries']
    assert summary['view_diagnostics']['layer_summaries']
    path=sorted((tmp_path/'results').glob('e07*.json'))[0];original=path.read_text();data=json.loads(original)
    data['selections']={};path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incomplete'):collect_results(tmp_path,allow_smoke=True)
    path.write_text(original);data=json.loads(original);data['predictions'][0]['config_id']='wrong';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incomplete'):collect_results(tmp_path,allow_smoke=True)
    path.write_text(original);data=json.loads(original);data['rows']=data['rows'][:-1];path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incomplete'):collect_results(tmp_path,allow_smoke=True)
    path.write_text(original)
    # Keep the changed fit/state rows and log mutually consistent; the saved
    # model snapshot must still reject an invented fitted-state identity.
    data=json.loads(original);fit=next(f for f in data['fits'] if f['model_id'].endswith('_af'))
    fit['state_hash']='invented'
    for row in data['rows']:
        if (row['model_id'],row['model_seed'])==(fit['model_id'],fit['model_seed']):row['state_hash']='invented'
    for event in data['events']:
        if event['stage']=='fit' and event['record']['fit_id']==fit['fit_id']:event['record']=fit
        if event['stage']=='row':event['record']=next(r for r in data['rows'] if r['record_id']==event['record']['record_id'])
    log=tmp_path/'logs'/(path.stem+'.jsonl');old_log=log.read_text()
    path.write_text(json.dumps(data));log.write_text(''.join(json.dumps(e)+'\n' for e in data['events']))
    with pytest.raises(ValueError,match='Incomplete'):collect_results(tmp_path,allow_smoke=True)
    path.write_text(original);log.write_text(old_log)
    data=json.loads(original);part=data['partitions'][0];part['encoder_seed']+=1
    for event in data['events']:
        if event['stage']=='partition' and event['record']['partition_id']==part['partition_id']:event['record']=part
    path.write_text(json.dumps(data));log.write_text(''.join(json.dumps(e)+'\n' for e in data['events']))
    with pytest.raises(ValueError,match='Incomplete'):collect_results(tmp_path,allow_smoke=True)
    path.write_text(original);log.write_text(old_log);path.unlink()
    with pytest.raises(ValueError,match='missing'):collect_results(tmp_path,allow_smoke=True)


def test_cost_uses_one_warmup_and_five_label_free_repetitions():
    m=api();X,y=toy();p=protocol()
    from experiments.make_revision.comparisons import conventional_factory
    model=conventional_factory('svc_rbf',{'C':1,'gamma':'scale'},8129).fit(X[:40],y[:40])
    before=m.fitted_hash(model)
    result=m.measure_cost(model,X[40:],p,fit_seconds=.1)
    assert len(result['prediction_seconds'])==5 and result['warmups']==1
    assert 'accuracy' not in result and 'score' not in result
    assert result['serialized_bytes']>0 and result['serialization_protocol']==5
    assert all(total>=encoding for total,encoding in zip(result['prediction_seconds'],result['query_encoding_seconds']))
    assert result['classifier_inference_seconds']==pytest.approx(np.array(result['prediction_seconds'])-result['query_encoding_seconds'])
    assert before==m.fitted_hash(model)


def test_matched_interval_uses_declared_ratio_with_unequal_fold_sizes(tmp_path,monkeypatch):
    from experiments.make_revision import run_studies as runner
    from experiments.make_revision.matched import architecture_id
    from experiments.make_revision.evaluation import make_splits
    from pathlib import Path
    p=json.loads(runner.STUDY_PROTOCOL.read_text());p.update(outer_folds=3,outer_repeats=1,test_train_ratio=.25)
    X=np.random.RandomState(2).normal(size=(61,4));y=np.resize([0,1,2],61)
    splits=make_splits(y,3,1,2,31);assert np.mean([len(s['test'])/len(s['train']) for s in splits])!=.25
    (tmp_path/'protocol.json').write_text(json.dumps(p));(tmp_path/'run_manifest.json').write_text(json.dumps({'purpose':'test'}))
    monkeypatch.setattr(runner,'collect_study_results',lambda *a,**k:{'iris':[]})
    monkeypatch.setattr(runner,'summarize_outer',lambda *a,**k:{})
    monkeypatch.setattr(runner,'load_prepared',lambda *a:(X,y,{},splits))
    observed=[]
    def interval(*a,**k):observed.append(k['q']);return {'p_approximate':1.}
    monkeypatch.setattr(runner,'paired_corrected_interval',interval)
    runner.summarize_study(tmp_path)
    assert observed==[.25,.25]


def test_compute_lock_rejects_overlapping_benchmark_commands():
    from experiments.make_revision.run_revision import execution_lock
    with execution_lock():
        with pytest.raises(RuntimeError,match='already active'):
            with execution_lock():pass


def test_cost_serialization_failure_keeps_explicit_array_payload():
    m=api();X,y=toy()
    class Unserializable:
        def __init__(self):self.weights_=np.ones((4,2))
        def __reduce__(self):raise TypeError('fixture cannot serialize')
        def predict(self,X):return np.zeros(len(X),dtype=int)
    result=m.measure_cost(Unserializable(),X,protocol(),fit_seconds=0.)
    assert result['serialized_bytes'] is None and 'fixture cannot serialize' in result['serialization_error']
    assert result['unique_array_payload_bytes']==64
    assert result['fitted_state_hash_scope']=='array-content fallback after unsupported serialization'


def test_cost_ensemble_reuses_fits_for_prefixes_and_preserves_state():
    m=api();p=protocol();X,y=toy();p['fit_seeds']=p['fit_seeds'][:1];p['head_conditions']=[True]
    learned=m.fit_views(X[:40],y[:40],X[40:],p,dataset_id='synthetic',repeat=0,fold=0,retain_models=True)
    assert len(learned['fits'])==9
    assert all(f['encoding_seconds']==pytest.approx(f['encoder_fit_seconds']+f['query_encoding_seconds']) for f in learned['fits'])
    for scheme in p['e05_schemes']:
        models=[learned['models'][(scheme,8129,True,v)] for v in range(3)]
        for prefix in (1,3):
            report=m.measure_cost(m.ViewEnsemble(models[:prefix]),X[40:],p,fit_seconds=0.)
            assert report['repetitions']==5
            assert report['classifier_fit_seconds']==pytest.approx(sum(model.training_seconds_ for enc,model in models[:prefix]))
    assert len(learned['fits'])==9


def test_cost_refuses_missing_completed_source_studies(tmp_path):
    from experiments.make_revision.run_secondary import cost
    with pytest.raises(ValueError,match='completed'):
        cost(tmp_path,protocol(),None,None,None)


@pytest.mark.parametrize('kind',['selected','fixed_af','footrule','hdc','borda','ensemble'])
def test_cost_worker_executes_real_small_models_without_outer_query_rows(tmp_path,kind):
    from experiments.make_revision.run_secondary import prepare,cost_worker
    from experiments.make_revision.evaluation import dataset_fingerprint
    X,y=toy();p=protocol();p['panels']={f:['synthetic'] for f in p['families']};features=['a','b','c','d'];labels=['0','1','2']
    data={'dataset_id':'synthetic','feature_names':features,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,features,labels)}
    source=tmp_path/'source';prepare(source,p,purpose='synthetic_smoke_only',loader=lambda name:(X,y,data))
    configs={'selected':{'C':1,'gamma':'scale'},'fixed_af':{'widths':[4,3]},'footrule':{'n_neighbors':3},'hdc':{'dimension':8},'borda':{},'ensemble':{}}
    job=dict(job_id=kind,kind=kind,source_root=str(source),dataset_id='synthetic',config=configs[kind],
             registry='experiments.make_revision.comparisons:e02_registry',source_protocol=p,model_id='svc_rbf',scheme='mixed')
    path=cost_worker((str(tmp_path),job,p));report=json.loads(open(path).read())
    assert report['status']=='ok',report.get('exception')
    assert set(report['query_ids'])<=set(report['train_ids'])
    assert len(report['rows'])==(2 if kind=='ensemble' else 1)
    assert all('accuracy' not in r and len(r['prediction_seconds'])==5 for r in report['rows'])


def test_secondary_noise_summary_averages_draws_and_uses_declared_clean_difference(tmp_path,monkeypatch):
    from experiments.make_revision import run_secondary as runner
    p=protocol();p['test_train_ratio']=.25;(tmp_path/'protocol.json').write_text(json.dumps(p));rows=[]
    for fold in range(3):
        for seed in p['fit_seeds']:
            for condition,errors in [('clean',[.1]),('gaussian_isotropic',[.2,.4,.6])]:
                for draw,error in enumerate(errors):
                    rows.append(dict(dataset_id='synthetic',model_id='m',model_seed=seed,outer_repeat=0,outer_fold=fold,
                        corruption_family=condition,severity=0 if condition=='clean' else .1,corruption_seed=draw,
                        accuracy=1-error,error=error,balanced_accuracy=1-error,macro_f1=1-error))
    monkeypatch.setattr(runner,'collect_results',lambda *a,**k:{'e06':rows})
    report=runner.summary(tmp_path)['summaries']['e06']
    noise=next(r for r in report if r['corruption_family']=='gaussian_isotropic')
    assert noise['change_from_clean']['mean_difference']==pytest.approx(.3)
    assert noise['change_from_clean']['test_train_ratio']==.25


def test_prototype_diagnostic_oracle_for_duplicates_and_response_ties():
    m=api();positions=np.array([[0,1,2],[0,1,2],[2,1,0]])
    rows=m.diagnostics_from_layers([positions],np.array([[0,1,2],[2,1,0]]))
    assert rows[0]['duplicate_fraction']==pytest.approx(1/3)
    assert rows[0]['zero_distance_pair_fraction']==pytest.approx(1/3)
    assert rows[0]['normalized_distance_mean']==pytest.approx(2/3)
    assert rows[0]['query_any_response_tie_fraction']==1
    assert rows[0]['query_repeated_response_fraction']==pytest.approx(1/3)


def test_training_pilot_never_calls_accuracy_boundary(tmp_path,monkeypatch):
    from experiments.make_revision import run_secondary as runner
    from experiments.make_revision.evaluation import dataset_fingerprint
    X,y=toy();p=protocol();features=['a','b','c','d'];labels=['0','1','2']
    data={'feature_names':features,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,features,labels)}
    monkeypatch.setattr(runner,'load_dataset',lambda name:(X,y,dict(data,dataset_id=name)))
    def forbidden(*a,**k):raise AssertionError('runtime pilot may not compute accuracy')
    monkeypatch.setattr(runner,'metric_values',forbidden)
    report=runner.runtime_pilot(tmp_path,p)
    assert report['purpose']=='training_only_fixed_runtime_no_scores'
    assert all(r['status'] in ('ok','excluded') and 'accuracy' not in r for r in report['records'])


def test_failed_secondary_job_persists_complete_terminal_rows_and_logs(tmp_path):
    from experiments.make_revision.run_secondary import prepare,worker
    from experiments.make_revision.evaluation import dataset_fingerprint
    X,y=toy();p=protocol();p['families']=['e05'];p['panels']={'e05':['synthetic']};p['architectures'][1]=[0]
    features=['a','b','c','d'];labels=['0','1','2'];data={'dataset_id':'synthetic','feature_names':features,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,features,labels)}
    jobs=prepare(tmp_path,p,purpose='synthetic_smoke_only',loader=lambda name:(X,y,data))
    path=worker((str(tmp_path),jobs[0]));result=json.loads(open(path).read())
    events=[json.loads(line) for line in (tmp_path/'logs'/(jobs[0]['stem']+'.jsonl')).read_text().splitlines()]
    assert result['status']=='failed' and result['events']==events
    assert len(result['rows'])==len(jobs[0]['records'])
    assert all(r['status']=='failed' for r in result['rows'])
    assert any(e['stage']=='terminal_failure' for e in events)
    with pytest.raises(FileExistsError):worker((str(tmp_path),jobs[0]))


def test_cost_requires_its_own_frozen_prepared_protocol(tmp_path):
    from experiments.make_revision.run_secondary import cost
    with pytest.raises(ValueError,match='frozen'):
        cost(tmp_path,protocol(),tmp_path,tmp_path,tmp_path)


def test_array_payload_deduplicates_shared_memoryview_buffers():
    m=api();a=np.arange(12,dtype=np.float64);b=np.frombuffer(memoryview(a),dtype=a.dtype)
    assert m.unique_array_payload({'a':a,'b':b,'slice':a[::2]})==a.nbytes


def test_cost_reports_readout_training_and_inference_batches_through_encoder_wrapper():
    m=api();X,y=toy();p=protocol()
    from experiments.make_revision.comparisons import OrderedPositionHDC,StableFootruleKNN
    enc=m.RowRankingEncoder().fit(X[:40]);orders=enc.transform(X[:40])
    for readout,train_batch in [(OrderedPositionHDC(dimension=8,batch_size=16).fit(orders,y[:40]),16),
                               (StableFootruleKNN(batch_size=16).fit(orders,y[:40]),None)]:
        report=m.measure_cost(m.EncodedReadout(enc,readout),X[40:],p,fit_seconds=0.,training_samples=40)
        assert report['training_batch_size']==train_batch
        assert report['internal_inference_batch_size']==16


@pytest.mark.parametrize('corruption',['remove_panel','remove_fold'])
def test_protocol_panels_and_regenerated_folds_cannot_be_redefined_by_manifests(tmp_path,corruption):
    from experiments.make_revision.run_secondary import prepare,verify
    from experiments.make_revision.evaluation import dataset_fingerprint,config_id
    X,y=toy();p=protocol();p['panels']={family:['synthetic'] for family in p['families']}
    features=['a','b','c','d'];labels=['0','1','2'];data={'dataset_id':'synthetic','feature_names':features,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,features,labels)}
    prepare(tmp_path,p,purpose='synthetic_smoke_only',loader=lambda name:(X,y,data))
    verify(tmp_path,allow_smoke=True)
    if corruption=='remove_panel':
        path=tmp_path/'manifest.json';manifest=json.loads(path.read_text());manifest['datasets']=[];path.write_text(json.dumps(manifest))
        (tmp_path/'planned_jobs.json').write_text('[]')
    else:
        path=tmp_path/'synthetic'/'splits.json';splits=json.loads(path.read_text())[:-1];path.write_text(json.dumps(splits))
        path=tmp_path/'synthetic'/'manifest.json';manifest=json.loads(path.read_text());manifest['splits_hash']=config_id(splits);path.write_text(json.dumps(manifest))
        path=tmp_path/'planned_jobs.json';jobs=[j for j in json.loads(path.read_text()) if j['outer_fold']!=2];path.write_text(json.dumps(jobs))
    with pytest.raises(ValueError):verify(tmp_path,allow_smoke=True)


@pytest.mark.parametrize('corruption',['remove_panel','remove_fold'])
def test_matched_protocol_panel_and_splits_are_authoritative(tmp_path,corruption):
    from experiments.make_revision.run_studies import prepare_study,verify_prepared
    from experiments.make_revision.evaluation import dataset_fingerprint,config_id
    X,y=toy();p=protocol();p['datasets']=['synthetic']
    features=['a','b','c','d'];labels=['0','1','2'];data={'dataset_id':'synthetic','feature_names':features,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,features,labels)}
    prepare_study(tmp_path,['synthetic'],p,purpose='synthetic_smoke_only',data_loader=lambda name:(X,y,data))
    verify_prepared(tmp_path,allow_smoke=True)
    if corruption=='remove_panel':
        path=tmp_path/'run_manifest.json';manifest=json.loads(path.read_text());manifest['datasets']=[];path.write_text(json.dumps(manifest))
        (tmp_path/'planned_jobs.json').write_text('[]')
    else:
        path=tmp_path/'synthetic'/'splits.json';splits=json.loads(path.read_text())[:-1];path.write_text(json.dumps(splits))
        path=tmp_path/'synthetic'/'manifest.json';manifest=json.loads(path.read_text());manifest['splits_hash']=config_id(splits);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):verify_prepared(tmp_path,allow_smoke=True)


@pytest.fixture(scope='module')
def selection_evidence(tmp_path_factory):
    from experiments.make_revision.run_secondary import evaluate_job,record_schedule
    from experiments.make_revision.evaluation import make_splits
    X,y=toy();p=protocol();split=make_splits(y,3,1,2,p['split_seed'])[0];output={}
    for family in ['e05','e06']:
        root=tmp_path_factory.mktemp(family)
        job=dict(family=family,dataset_id='synthetic',outer_repeat=0,outer_fold=0,records=record_schedule(family,p,4))
        result=evaluate_job(X,y,split,p,job,dataset_hash='fixture',code_revision='fixture',artifact_root=root)
        assert result['status']=='ok'
        output[family]=(result,job,p,X,y,split,{'dataset_hash':'fixture'},root)
    return output


@pytest.mark.parametrize('corruption',['view_hash','negative_score','nonfinite_score','wrong_stage','wrong_repeat','wrong_fold','failure_reason','failure_score'])
def test_selection_and_view_evidence_cannot_contradict_sources(selection_evidence,corruption):
    from experiments.make_revision.run_secondary import validate_job
    family='e05' if corruption=='view_hash' else 'e06'
    original,job,p,X,y,split,data,root=selection_evidence[family];result=copy.deepcopy(original)
    validate_job(result,result['events'],job,p,X,y,split,data,'fixture',root)
    if family=='e05':
        fit=result['fits'][0];fit['prediction_hash']='invented'
        for event in result['events']:
            if event['stage']=='fit' and event['record']['fit_id']==fit['fit_id']:event['record']=fit
    else:
        name=next(iter(result['selections']));selection=result['selections'][name]
        for row in selection['fits']:
            if corruption in ('negative_score','nonfinite_score'):row['score']=-1 if corruption=='negative_score' else float('nan')
            elif corruption=='wrong_stage':row['stage']='outer'
            elif corruption=='wrong_repeat':row['outer_repeat']=99
            elif corruption=='wrong_fold':row['outer_fold']=99
            else:
                row['status']='failed';row['score']=None if corruption=='failure_reason' else .5;row['exception']='' if corruption=='failure_reason' else 'declared failure'
        if corruption in ('negative_score','nonfinite_score'):selection['inner_score']=selection['fits'][0]['score']
        for event in result['events']:
            if event['stage']=='selection' and event['record']['model_id']==name:event['record']['selection']=selection
    with pytest.raises(ValueError):validate_job(result,result['events'],job,p,X,y,split,data,'fixture',root)


def test_environment_seals_imported_core_config():
    import hashlib
    from pathlib import Path
    from experiments.make_revision.run_revision import environment_record
    path=Path(__file__).resolve().parents[2]/'arrowflow'/'config.py'
    env=environment_record('experiments.make_revision.run_secondary:environment')
    assert env['source_hashes']['arrowflow/config.py']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'experiments/make_revision/view_reporting.py' in env['source_hashes']


def test_secondary_selection_retains_legitimate_failed_screen_candidate(selection_evidence):
    from dataclasses import replace
    from experiments.make_revision.run_secondary import validate_selection
    from experiments.make_revision.evaluation import config_id
    result,job,p,X,y,split,data,root=selection_evidence['e06']
    registry=api().e06_registry(p);name=next(iter(result['selections']));selection=copy.deepcopy(result['selections'][name]);spec=registry[name]
    failed_config=dict(spec.candidates[0],unavailable_fixture_parameter=True)
    for row in list(selection['fits']):
        if row['model_seed']==p['fit_seeds'][0]:
            selection['fits'].append(dict(row,config=failed_config,config_id=config_id(failed_config),status='failed',score=None,exception='Explicit fixture fit failure'))
    validate_selection(selection,replace(spec,candidates=[*spec.candidates,failed_config]),p,split)


def test_cost_native_adapter_binds_iterations_and_seed_by_keyword(tmp_path,monkeypatch):
    # NativeArrowFlow(widths, iterations, learning_rate, validation_ratio, p_correct, seed): the third
    # positional slot is learning_rate, and the adapter's later __dict__.update would mask a misbinding.
    from experiments.make_revision import comparisons
    from experiments.make_revision.run_secondary import prepare,cost_worker
    from experiments.make_revision.evaluation import dataset_fingerprint
    constructed=[]
    class Spy(comparisons.NativeArrowFlow):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);constructed.append(dict(vars(self)))   # state before any update
    monkeypatch.setattr(comparisons,'NativeArrowFlow',Spy)
    sushi='sushi_childhood_east_west_v2'
    rng=np.random.RandomState(45);X=np.asarray([rng.permutation(10) for _ in range(60)],dtype=np.int64);y=np.tile([0,1,2],20)
    p=protocol();p['panels']={f:[sushi] for f in p['families']};items=[str(i) for i in range(10)];labels=['0','1','2']
    data={'dataset_id':sushi,'feature_names':items,'label_map':labels,'dataset_hash':dataset_fingerprint(X,y,items,labels)}
    source=tmp_path/'source';prepare(source,p,purpose='synthetic_smoke_only',loader=lambda name:(X,y,data))
    widths=[4,3]
    job=dict(job_id='native_fixed_af',kind='fixed_af',source_root=str(source),dataset_id=sushi,config={'widths':widths},
             registry='experiments.make_revision.comparisons:e02_registry',source_protocol=p,model_id='svc_rbf',scheme='mixed')
    report=json.loads(open(cost_worker((str(tmp_path),job,p))).read())
    assert report['status']=='ok',report.get('exception')
    [state]=constructed
    expected_seed=api().derive_seed(p['cost_seed'],'network',sushi,0,0,-1,widths)
    assert (state['widths'],state['iterations'],state['learning_rate'],state['seed'])==(widths,p['iterations'],.5,expected_seed)
    assert (state['embed_dim'],state['validation_ratio'],state['p_correct'])==(10,0.,.01)
