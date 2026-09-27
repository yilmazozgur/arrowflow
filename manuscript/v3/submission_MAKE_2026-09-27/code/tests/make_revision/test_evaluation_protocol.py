"""Executable leakage, selection, identity, and inference-uncertainty checks."""
import importlib
import json
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def api():
    return importlib.import_module('experiments.make_revision.evaluation')


def toy():
    rng = np.random.RandomState(17)
    return rng.randn(60, 4), np.tile(np.arange(3), 20)


def test_saved_nested_splits_are_disjoint_cover_each_repeat_and_stable():
    e = api(); X, y = toy()
    splits = e.make_splits(y, outer_folds=5, repeats=3, inner_folds=3, seed=71)
    assert splits == e.make_splits(y, outer_folds=5, repeats=3, inner_folds=3, seed=71)
    for repeat in range(3):
        rows = []
        for split in [s for s in splits if s['outer_repeat'] == repeat]:
            tr, te = set(split['train']), set(split['test'])
            assert not tr & te and tr | te == set(range(60))
            rows += split['test']
            for inner in split['inner']:
                it, iv = set(inner['train']), set(inner['validation'])
                assert not it & iv and it | iv == tr and not (it | iv) & te
        assert sorted(rows) == list(range(60))


class AuditScaler(StandardScaler):
    fits = []
    def fit(self, X, y=None, **kwargs):
        self.fits.append(tuple(X[:, 0].astype(int)))
        return super().fit(X, y, **kwargs)


def test_inner_pipeline_refits_and_outer_labels_cannot_select_configuration():
    e = api(); X, y = toy(); X[:, 0] = np.arange(len(X))
    split = e.make_splits(y, outer_folds=3, repeats=1, inner_folds=2, seed=31)[0]
    spec = e.ModelSpec('audit', lambda cfg, seed: make_pipeline(AuditScaler(), KNeighborsClassifier(n_neighbors=cfg['k'])), [{'k':1}, {'k':3}], False)
    AuditScaler.fits = []
    first = e.select_model(X, y, split, spec, [11, 22, 33])
    assert sorted(AuditScaler.fits) == sorted(tuple(i['train']) for i in split['inner'] for _ in spec.candidates)
    changed = y.copy(); changed[split['test']] = (changed[split['test']] + 1) % 3
    second = e.select_model(X, changed, split, spec, [11, 22, 33])
    assert first['config_id'] == second['config_id']
    assert [r['score'] for r in first['fits']] == [r['score'] for r in second['fits']]
    a=spec.factory(first['config'],11).fit(X[split['train']],y[split['train']])
    b=spec.factory(second['config'],11).fit(X[split['train']],changed[split['train']])
    assert np.array_equal(a[0].mean_,b[0].mean_)
    assert np.array_equal(a.predict(X),b.predict(X))


def test_stochastic_screen_seed_is_reused_and_deterministic_not_replicated():
    e=api(); X,y=toy(); split=e.make_splits(y,3,1,2,31)[0]
    factory=lambda cfg,seed: KNeighborsClassifier(n_neighbors=cfg['k'])
    stochastic=e.ModelSpec('stochastic',factory,[{'k':k} for k in (1,3,5,7)],True)
    chosen=e.select_model(X,y,split,stochastic,[11,22,33])
    assert len(chosen['fits']) == 4*2 + 3*2*2
    keys=[(r['config_id'],r['inner_fold'],r['model_seed']) for r in chosen['fits']]
    assert len(keys)==len(set(keys))
    deterministic=e.ModelSpec('deterministic',factory,[{'k':1}],False)
    result=e.evaluate_fold(X,y,split,deterministic,[11,22,33],dataset_id='toy',dataset_hash='hash',code_revision='test')
    assert len(result['predictions']) == len(split['test'])
    assert len(result['models']) == 1
    required={'dataset_id','dataset_hash','sample_id','outer_repeat','outer_fold','model_id','model_seed','view_id','condition','perturbation_seed','y_true','y_pred','config_id','code_revision'}
    assert required <= result['predictions'][0].keys()


def test_dataset_hash_covers_labels_values_metadata_and_order():
    e=api(); X,y=toy()
    h=e.dataset_fingerprint(X,y,['a','b','c','d'],['A','B','C'])
    for args in [(X[::-1],y[::-1],['a','b','c','d'],['A','B','C']), (X,y+1,['a','b','c','d'],['A','B','C']), (X,y,['b','a','c','d'],['A','B','C']), (X,y,['a','b','c','d'],['X','B','C'])]:
        assert h != e.dataset_fingerprint(*args)


@pytest.mark.parametrize('strategy',['random','target_aware','calibrated'])
def test_frozen_encoder_matches_legacy_finite_encoding(strategy):
    from experiments.make_revision.models import OrdinalEncoder
    from experiments.exp_knn_vs_arrowflow import encode_view
    X,y=toy(); enc=OrdinalEncoder(strategy=strategy,embed_dim=8,seed=13)
    enc.fit(X[:45],y[:45]); a,b=encode_view(X[:45],y[:45],X[45:],strategy,8,.3,13)
    assert np.array_equal(enc.transform(X[:45]),a)
    assert np.array_equal(enc.transform(X[45:]),b)
    assert np.array_equal(enc.transform(X[45:]),enc.transform(X[45:]))


def test_imputation_keeps_empty_columns_ecdf_uses_only_training_distribution():
    from experiments.make_revision.models import NumericImputer, FittedECDF
    X=np.array([[1,np.nan],[3,np.nan],[3,np.nan]])
    imp=NumericImputer().fit(X)
    assert np.array_equal(imp.transform([[np.nan,7]]),[[7/3,7]])
    with pytest.raises(ValueError): imp.transform([[np.inf,1]])
    ecdf=FittedECDF().fit([[1],[3],[3]])
    assert np.allclose(ecdf.transform([[0],[1],[2],[3],[4]]).ravel(),[0,1/6,1/3,2/3,1])


def test_shared_bundle_is_readonly_and_checks_exact_arrays():
    from experiments.make_revision.models import OrdinalEncoder, SharedEncoding
    X,y=toy(); enc=OrdinalEncoder(embed_dim=8,seed=1).fit(X[:45],y[:45])
    bundle=SharedEncoding.create(enc,X[:45],X[45:])
    bundle.assert_identical(bundle.train.copy(),bundle.test.copy())
    altered=bundle.test.copy(); altered[0]=altered[0][::-1]
    with pytest.raises(ValueError): bundle.assert_identical(bundle.train,altered)
    with pytest.raises(ValueError): bundle.train[0,0]=9


def test_corrected_interval_averages_seeds_before_variance_and_holm():
    e=api()
    rows=[{'outer_repeat':0,'outer_fold':fold,'model_id':model,'model_seed':seed,'accuracy':value+noise} for fold,difference in enumerate([.1,.2,.3]) for seed,noise in [(1,-.02),(2,.02)] for model,value in [('a',difference),('b',0)]]
    result=e.paired_corrected_interval(rows,'a','b',q=.25,expected_folds=[(0,f) for f in range(3)],expected_seeds={'a':[1,2],'b':[1,2]})
    assert result['n_folds']==3 and result['mean_difference']==pytest.approx(.2)
    assert result['standard_error']==pytest.approx(np.sqrt((1/3+.25)*.01))
    assert e.holm_adjust([.01,.04,.03]) == pytest.approx([.03,.06,.06])


def test_seeded_arrowflow_predict_and_hidden_representations_reproducible():
    from experiments.make_revision.models import ArrowFlowEstimator
    X,y=toy(); cfg=dict(embed_dim=8,widths=(6,),iterations=2,seed=19)
    a=ArrowFlowEstimator(**cfg).fit(X[:45],y[:45]); b=ArrowFlowEstimator(**cfg).fit(X[:45],y[:45])
    assert np.array_equal(a.predict(X[45:]),b.predict(X[45:]))
    assert np.array_equal(a.transform(X[45:]),b.transform(X[45:]))
    assert a.transform(X[45:]).shape==(15,6)
    assert a.config_.val_data_ratio==0 and a.config_.device=='cpu'


def test_cli_prepare_seals_dataset_splits_candidates_and_smoke_logs(tmp_path):
    from experiments.make_revision.run_revision import main
    main(['prepare','--dataset','iris','--output',str(tmp_path)])
    manifest=json.loads((tmp_path/'iris'/'manifest.json').read_text())
    assert manifest['shape']==[150,4] and manifest['class_counts']==[50,50,50]
    splits=json.loads((tmp_path/'iris'/'splits.json').read_text())
    assert len(splits)==15 and all(len(s['inner'])==3 for s in splits)
    candidates=json.loads((tmp_path/'candidates.json').read_text())
    assert len(candidates['arrowflow']['candidates'])==24
    main(['smoke','--dataset','iris','--output',str(tmp_path/'smoke')])
    output=json.loads((tmp_path/'smoke'/'smoke.json').read_text())
    assert output['purpose']=='smoke_only_not_paper_evidence'
    assert len(output['result']['models'])==3
    assert all(r['status']=='ok' for r in output['result']['models'])
    assert len(output['result']['predictions'])==150


def test_confirmatory_run_refuses_unfrozen_protocol(tmp_path):
    from experiments.make_revision.run_revision import main
    with pytest.raises(ValueError,match='frozen'):
        main(['run','--dataset','iris','--output',str(tmp_path)])


def test_failure_logs_and_canonical_score_ties():
    e=api(); X,y=toy(); split=e.make_splits(y,3,1,2,31)[0]
    def factory(cfg,seed):
        from sklearn.dummy import DummyClassifier
        if cfg['bad']: raise ValueError('deliberate failed configuration')
        return DummyClassifier(strategy='most_frequent')
    configs=[{'bad':True,'id':0},{'bad':False,'id':2},{'bad':False,'id':1}]
    logs=[]
    selection=e.select_model(X,y,split,e.ModelSpec('failure',factory,configs,False),sink=logs.append)
    assert sum(r['status']=='failed' for r in logs)==2
    assert selection['config_id']==min(e.config_id(c) for c in configs if not c['bad'])


def test_inner_split_tampering_is_rejected():
    e=api(); X,y=toy(); split=e.make_splits(y,3,1,2,31)[0]
    split['inner'][0]['train'][0]=split['test'][0]
    with pytest.raises(ValueError,match='leaks'):
        e.validate_split(split,len(y))


def test_frozen_supervised_encoder_state_does_not_depend_on_query_or_labels():
    from experiments.make_revision.models import OrdinalEncoder
    X,y=toy(); enc=OrdinalEncoder(strategy='target_aware',degree=2,embed_dim=8,seed=13).fit(X[:45],y[:45])
    before=enc.transform(X[45:]); means=enc.scaler_.mean_.copy(); coef=enc.lda_.coef_.copy()
    enc.transform(np.full_like(X[45:],1000))
    assert np.array_equal(before,enc.transform(X[45:]))
    assert np.array_equal(means,enc.scaler_.mean_) and np.array_equal(coef,enc.lda_.coef_)


def test_summary_keeps_seed_spread_separate_from_fold_spread():
    e=api()
    rows=[{'model_id':'a','outer_repeat':0,'outer_fold':fold,'model_seed':seed,'accuracy':score+noise}
          for fold,score in enumerate([.6,.8]) for seed,noise in [(1,-.1),(2,.1)]]
    result=e.summarize_outer(rows,'a',expected_folds=[(0,0),(0,1)],expected_seeds=[1,2])
    assert result['mean']==pytest.approx(.7)
    assert result['outer_fold_sd']==pytest.approx(np.sqrt(.02))
    assert result['mean_within_fold_seed_sd']==pytest.approx(np.sqrt(.02))
    assert result['n_folds']==2
    with pytest.raises(ValueError,match='seed'):
        e.summarize_outer(rows[:-1],'a',expected_folds=[(0,0),(0,1)],expected_seeds=[1,2])


def test_shared_bundle_saved_arrays_keep_identity(tmp_path):
    from experiments.make_revision.models import OrdinalEncoder, SharedEncoding
    X,y=toy(); bundle=SharedEncoding.create(OrdinalEncoder(embed_dim=8).fit(X[:45],y[:45]),X[:45],X[45:])
    path=tmp_path/'view.npz'; bundle.save(path)
    with np.load(path,allow_pickle=False) as data:
        bundle.assert_identical(data['train'],data['test'])
        assert data['train_hash'].item()==bundle.train_hash


def test_runtime_pilot_never_passes_outer_test_rows(tmp_path):
    from experiments.make_revision.run_revision import runtime_pilot, PROTOCOL
    protocol=json.loads(PROTOCOL.read_text()); protocol['candidate_budget']=1
    report=runtime_pilot(tmp_path,['iris'],protocol,'experiments.make_revision.run_revision:default_registry')
    split=json.loads((tmp_path/'iris'/'splits.json').read_text())[0]
    assert all(set(r['fit_rows'])==set(split['train']) for r in report['rows'])
    assert all(not set(r['fit_rows']) & set(split['test']) for r in report['rows'])
    assert all('accuracy' not in r and 'score' not in r for r in report['rows'])
    assert report['workload_estimates']['arrowflow']['fits_per_outer']==12


def integration_registry(protocol):
    from experiments.make_revision.run_revision import dummy_factory, arrowflow_factory
    e=api()
    return {'dummy':e.ModelSpec('dummy',dummy_factory,[{}],False),
            'arrowflow':e.ModelSpec('arrowflow',arrowflow_factory,[{'widths':(4,),'embed_dim':4,'iterations':4}],True)}


def prepare_synthetic_fixture(tmp_path, registry):
    from experiments.make_revision import run_revision as runner
    e=api(); X,y=toy()
    protocol=json.loads(runner.PROTOCOL.read_text())
    protocol.update(frozen=True,outer_folds=3,outer_repeats=1,inner_folds=2,datasets=['iris'])
    protocol_path=tmp_path/'test_protocol.json';runner.write_json(protocol_path,protocol)
    output=tmp_path/'synthetic_integration_only'
    names=['a','b','c','d']; labels=['0','1','2']
    manifest={'purpose':'synthetic_test_fixture_not_iris','feature_names':names,'label_map':labels,
              'dataset_hash':e.dataset_fingerprint(X,y,names,labels)}
    # Supply a synthetic data source while exercising actual prepare serialization.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner,'load_dataset',lambda name:(X,y,manifest.copy()))
        runner.prepare(output,['iris'],protocol,registry)
    return output,protocol,protocol_path


def test_spawned_run_records_complete_synthetic_folds_and_rejects_overwrite(tmp_path):
    from experiments.make_revision.run_revision import main
    registry='tests.make_revision.test_evaluation_protocol:integration_registry'
    output,protocol,protocol_path=prepare_synthetic_fixture(tmp_path,registry)
    args=['run','--dataset','iris','--output',str(output),'--protocol',str(protocol_path),'--registry',registry,'--workers','2']
    main(args)
    files=list((output/'results').glob('*.json'))
    assert len(files)==6
    results=[json.loads(p.read_text()) for p in files]
    assert all(r['status']=='ok' for result in results for r in result['models'])
    assert sum(len(r['predictions']) for r in results)==len(toy()[1])*4
    with pytest.raises(FileExistsError): main(args)
    from experiments.make_revision.run_revision import collect_confirmatory_results
    rows=collect_confirmatory_results(output,['iris'],protocol,integration_registry(protocol))
    assert len(rows['iris'])==12
    log=files[0].with_suffix('.fits.jsonl'); saved_log=log.read_text(); log.unlink()
    with pytest.raises(ValueError,match='fit log'):
        collect_confirmatory_results(output,['iris'],protocol,integration_registry(protocol))
    log.write_text(saved_log); files[0].unlink()
    with pytest.raises(ValueError,match='result'):
        collect_confirmatory_results(output,['iris'],protocol,integration_registry(protocol))


def test_shared_orders_reject_fractional_tokens_before_integer_conversion():
    from experiments.make_revision.models import ArrowFlowEstimator
    model=ArrowFlowEstimator(embed_dim=3,widths=(3,),iterations=0)
    with pytest.raises(ValueError,match='permutation'):
        model.fit_orders(np.array([[0.1,1.1,2.1],[2,1,0]]),[0,1])
    with pytest.raises(ValueError,match='length|samples'):
        model.fit_orders(np.array([[0,1,2],[2,1,0]]),[0])


def scheduled_rows():
    return [{'model_id':model,'outer_repeat':0,'outer_fold':fold,'model_seed':seed,
             'accuracy':(.7 if model=='a' else .6)+.01*fold,'status':'ok'}
            for model in ['a','b'] for fold in range(5) for seed in [8129,19391,39019]]


@pytest.mark.parametrize('omission',['whole_paired_fold','uniform_seed','wrong_deterministic_seed'])
def test_inference_requires_complete_declared_fold_and_seed_schedule(omission):
    e=api(); rows=scheduled_rows(); folds=[(0,f) for f in range(5)]
    seeds={'a':[8129,19391,39019],'b':[8129,19391,39019]}
    if omission=='whole_paired_fold': rows=[r for r in rows if r['outer_fold']!=4]
    elif omission=='uniform_seed': rows=[r for r in rows if r['model_seed']!=39019]
    else:
        rows=[r for r in rows if r['model_id']=='a' or r['model_seed']==19391]
        seeds['b']=[8129]
    with pytest.raises(ValueError,match='schedule'):
        e.paired_corrected_interval(rows,'a','b',expected_folds=folds,expected_seeds=seeds)


def test_confirmatory_arithmetic_has_no_inferred_schedule_fallback():
    e=api(); rows=scheduled_rows()
    with pytest.raises(TypeError): e.summarize_outer(rows,'a')
    with pytest.raises(TypeError): e.paired_corrected_interval(rows,'a','b')
    summary=e.summarize_outer(rows,'a',expected_folds=[(0,f) for f in range(5)],expected_seeds=[8129,19391,39019])
    assert summary['n_folds']==5 and summary['seeds_per_fold']==3


def failing_factory(config,seed):
    raise ValueError('synthetic selection failure')


@pytest.mark.parametrize('fail_stage',['screen','rerank'])
def test_all_failed_selection_returns_terminal_rows_and_full_fit_history(fail_stage):
    e=api(); X,y=toy(); split=e.make_splits(y,3,1,2,31)[0]; logs=[]
    def factory(config,seed):
        if fail_stage=='rerank' and seed==8129:
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy='most_frequent')
        return failing_factory(config,seed)
    spec=e.ModelSpec('failed',factory,[{}],True)
    result=e.evaluate_fold(X,y,split,spec,dataset_id='synthetic',dataset_hash='test',code_revision='test',sink=logs.append)
    assert result['status']=='failed_selection' and not result['predictions']
    expected_fits=2 if fail_stage=='screen' else 6
    assert len(result['selection']['fits'])==expected_fits
    assert {r['model_seed'] for r in result['models']}=={8129,19391,39019}
    assert all(r['status']=='failed_selection' for r in result['models'])
    assert len(logs)==expected_fits+3


def test_declared_registry_dependency_modules_are_source_sealed():
    from experiments.make_revision.run_revision import environment_record
    environment=environment_record('tests.make_revision.test_evaluation_protocol:integration_registry')
    assert 'experiments/exp_knn_vs_arrowflow.py' in environment['source_hashes']


# Extra estimator dependencies must be declared explicitly by Task5 registries.
SOURCE_MODULES=['experiments.exp_knn_vs_arrowflow']


def failing_registry(protocol):
    return {'failed':api().ModelSpec('failed',failing_factory,[{}],True)}


def test_failed_selection_is_persisted_for_every_planned_spawned_job(tmp_path):
    from experiments.make_revision.run_revision import main
    registry='tests.make_revision.test_evaluation_protocol:failing_registry'
    output,protocol,protocol_path=prepare_synthetic_fixture(tmp_path,registry)
    with pytest.raises(ValueError,match='failed_selection'):
        main(['run','--dataset','iris','--output',str(output),'--protocol',str(protocol_path),
              '--registry',registry,'--workers','2'])
    planned=json.loads((output/'planned_jobs.json').read_text())
    assert len(planned)==3
    for job in planned:
        result=json.loads((output/job['result_file']).read_text())
        assert result['status']=='failed_selection'
        assert len(result['models'])==3 and len(result['selection']['fits'])==2
        events=[json.loads(line) for line in (output/job['log_file']).read_text().splitlines()]
        assert len(events)==5


def test_expected_schedule_includes_all_folds_and_one_deterministic_seed():
    from experiments.make_revision.run_revision import PROTOCOL
    e=api(); protocol=json.loads(PROTOCOL.read_text())
    schedule=e.expected_schedule(protocol,integration_registry(protocol))
    assert len(schedule['expected_folds'])==15
    assert schedule['expected_seeds']=={'dummy':[8129],'arrowflow':[8129,19391,39019]}
