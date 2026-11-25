"""Comparator semantics and source identity, without outer benchmark scoring."""
import importlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from sklearn.base import clone


def api():
    return importlib.import_module('experiments.make_revision.comparisons')


def protocol():
    return json.loads(Path('experiments/make_revision/protocol.json').read_text())


def orders_fixture():
    return np.array([[0,1,2,3],[0,2,1,3],[3,2,1,0],[3,1,2,0]]),np.array([0,0,1,1])


def test_e02_registry_preserves_arrowflow_candidates_and_fold_local_steps():
    from experiments.make_revision.run_revision import default_registry
    c=api(); p=protocol(); registry=c.e02_registry(p)
    assert set(registry)=={'arrowflow','dummy','svc_rbf','random_forest','mlp','numeric_knn','gradient_boosting'}
    assert registry['arrowflow'].candidates==default_registry(p)['arrowflow'].candidates
    expected_counts={'svc_rbf':16,'random_forest':24,'mlp':24,'numeric_knn':20,'gradient_boosting':24}
    for name,count in expected_counts.items():
        spec=registry[name]; assert len(spec.candidates)==count
        estimator=spec.factory(spec.candidates[0],8129)
        assert estimator.steps[0][0]=='imputer'
        assert ('scaler' in estimator.named_steps)==(name in {'svc_rbf','mlp','numeric_knn'})
        assert clone(estimator).get_params()['steps'] is not estimator.steps
        final=estimator.steps[-1][1]
        if name in {'random_forest','mlp','gradient_boosting'}:
            assert spec.stochastic and final.random_state==8129
        else: assert not spec.stochastic
    assert registry['mlp'].factory(registry['mlp'].candidates[0],1).steps[-1][1].early_stopping is False


def test_real_pipeline_refits_imputation_scaling_and_records_convergence_warning():
    c=api(); X=np.array([[0,np.nan],[1,4],[2,6],[3,8],[1000,1000]],float); y=[0,1,0,1]
    estimator=c.conventional_factory('mlp',{'hidden_layer_sizes':(4,), 'max_iter':1},19)
    estimator.fit(X[:4],y)
    assert np.array_equal(estimator.named_steps['imputer'].means_,[1.5,6])
    assert np.allclose(estimator.named_steps['scaler'].mean_,[1.5,6])
    estimator.predict(X[4:])
    assert estimator.fit_warnings_ and any(w['category']=='ConvergenceWarning' for w in estimator.fit_warnings_)
    assert estimator.encoding_seconds_>=0 and estimator.classifier_fit_seconds_>=0


@pytest.mark.parametrize('weights',['uniform','distance'])
def test_footrule_full_tie_cutoff_uses_original_source_ids(weights):
    c=api(); X=np.tile([0,1,2,3],(3,1)); labels=np.array([1,0,1]); ids=np.array([30,10,20])
    estimator=c.StableFootruleKNN(n_neighbors=1,weights=weights).fit(X,labels,sample_ids=ids)
    assert estimator.predict(X[:1]).tolist()==[0]
    assert estimator.kneighbors(X[:1])[1].tolist()==[[1]]
    shuffled=np.array([2,0,1]); other=c.StableFootruleKNN(n_neighbors=1,weights=weights).fit(X[shuffled],labels[shuffled],sample_ids=ids[shuffled])
    assert np.array_equal(other.predict(X[:1]),estimator.predict(X[:1]))


def test_footrule_distance_weight_zero_rule_and_lowest_class_vote_tie():
    c=api(); X=np.array([[0,1,2],[0,1,2],[0,2,1]]); y=[3,5,5]
    weighted=c.StableFootruleKNN(n_neighbors=3,weights='distance').fit(X,y)
    assert weighted.predict([[0,1,2]]).tolist()==[3]
    uniform=c.StableFootruleKNN(n_neighbors=9).fit(X,y)
    assert uniform.predict([[0,1,2]]).tolist()==[5]
    with pytest.raises(ValueError,match='sample'):
        c.StableFootruleKNN().fit(X,y,sample_ids=[1,1,2])


def test_footrule_correct_metric_and_inverse_transformer():
    c=api(); X=np.array([[0,1,3,2],[0,2,3,1]])
    model=c.StableFootruleKNN(n_neighbors=2).fit(X,[0,1])
    assert model.kneighbors(X[:1])[0].tolist()==[[0,4]]
    transformer=c.InversePositionTransformer().fit(X)
    positions=transformer.transform(X)
    assert positions.tolist()==[[0,1,3,2],[0,3,1,2]]
    probe=c.StableFootruleKNN(n_neighbors=2,input_kind='positions').fit(positions,[0,1])
    assert np.array_equal(probe.kneighbors(positions[:1])[0],model.kneighbors(X[:1])[0])
    with pytest.raises(ValueError): transformer.transform([[0,1,2]])


def test_borda_prototypes_sort_mean_positions_with_item_and_class_ties():
    c=api(); X=np.array([[0,1,2],[1,0,2],[0,1,2],[1,0,2]])
    model=c.BordaClassifier().fit(X,[2,2,1,1])
    assert model.prototype_orders_.tolist()==[[0,1,2],[0,1,2]]
    assert model.predict([[2,1,0]]).tolist()==[1]


@pytest.mark.parametrize('dimension',[1024,10000])
def test_hdc_ordered_positions_have_exact_half_dimension_endpoints(dimension):
    c=api(); X,y=orders_fixture(); model=c.OrderedPositionHDC(dimension=dimension,seed=19).fit_encoder(X)
    positions=model.position_codes_
    assert set(np.unique(positions))=={-1,1}
    assert model.item_keys_.dtype==np.int8 and positions.dtype==np.int8
    assert np.sum(positions[0]!=positions[-1])==dimension//2
    for p in range(4):
        assert np.sum(positions[0]!=positions[p])==p*dimension//6
    repeated=c.OrderedPositionHDC(dimension=dimension,seed=19).fit_encoder(X)
    assert np.array_equal(model.item_keys_,repeated.item_keys_)
    assert np.array_equal(model.position_codes_,repeated.position_codes_)
    assert model.item_seed_ != model.position_seed_


def test_hdc_bundling_matches_explicit_binding_and_batch_invariance():
    c=api(); X,y=orders_fixture(); model=c.OrderedPositionHDC(dimension=8,batch_size=1,seed=19).fit(X,y)
    expected=[]
    for order in X:
        z=sum(model.item_keys_[item].astype(np.int32)*model.position_codes_[position] for position,item in enumerate(order))
        expected.append(z/np.linalg.norm(z) if np.linalg.norm(z) else z)
    expected=np.asarray(expected)
    assert np.allclose(model.transform(X),expected)
    class_means=np.array([expected[y==label].mean(axis=0) for label in [0,1]])
    norms=np.linalg.norm(class_means,axis=1,keepdims=True)
    class_means=np.divide(class_means,norms,out=np.zeros_like(class_means),where=norms!=0)
    assert np.allclose(model.class_prototypes_,class_means)
    large=c.OrderedPositionHDC(dimension=8,batch_size=100,seed=19).fit(X,y)
    assert np.allclose(model.class_prototypes_,large.class_prototypes_)
    assert np.array_equal(model.predict(X),large.predict(X))
    assert model.max_bundle_batch_rows_==1


def test_hdc_zero_vectors_remain_zero_and_cosine_ties_choose_lowest_class():
    c=api(); X,y=orders_fixture(); model=c.OrderedPositionHDC(dimension=4,seed=19).fit_encoder(X)
    model.fit_bundles(np.zeros((4,4)),[5,5,2,2])
    assert not model.class_prototypes_.any()
    assert model.predict_bundles(np.zeros((2,4))).tolist()==[2,2]
    assert np.isfinite(model.class_prototypes_).all()


def test_native_registry_receives_inverse_positions_and_arrowflow_bypasses_encoder():
    c=api(); registry=c.native_registry(protocol())
    X=np.array([np.arange(10),np.arange(9,-1,-1),np.roll(np.arange(10),1),np.roll(np.arange(10),2)])
    y=np.array([0,1,0,1])
    conventional=registry['native_numeric_knn'].factory({'n_neighbors':1,'weights':'uniform','p':1},19).fit(X,y)
    assert np.array_equal(conventional.named_steps['positions'].transform(X),np.argsort(X,axis=1))
    model=c.NativeArrowFlow(widths=(4,),iterations=4,seed=19).fit(X,y)
    assert not hasattr(model,'encoder_') and model.embed_dim==10
    assert model.predict(X).shape==(4,) and model.transform(X).shape==(4,4)
    assert {'native_footrule_knn','native_borda','native_hdc','native_arrowflow_64','native_arrowflow_64_32','native_arrowflow_230'} <= registry.keys()
    assert [c['dimension'] for c in registry['native_hdc'].candidates]==[1024,10000]


def test_sushi_parser_checks_header_rows_unique_ids_and_label_column():
    from experiments.make_revision.datasets import parse_sushi_files
    orders='10 1\n0 10 0 1 2 3 4 5 6 7 8 9\n0 10 9 8 7 6 5 4 3 2 1 0\n'
    users='101 0 0 0 40 0 0 0 0 0 0\n202 0 0 0 0 0 1 0 0 0 0\n'
    X,y,ids=parse_sushi_files(orders,users,expected_rows=2)
    assert y.tolist()==[0,1] and ids.tolist()==[101,202]
    assert X[0].tolist()==list(range(10))
    for bad_orders,bad_users in [(orders.replace('8 9','8 8'),users),(orders,users.replace('202','101')),(orders,users.splitlines()[0]),(orders.replace('10 1','10 0',1),users)]:
        with pytest.raises(ValueError): parse_sushi_files(bad_orders,bad_users,expected_rows=2)


def test_sushi_loader_requires_explicit_archive_and_rejects_bad_hash(tmp_path,monkeypatch):
    from experiments.make_revision.datasets import load_sushi
    monkeypatch.delenv('ARROWFLOW_SUSHI_ARCHIVE',raising=False)
    with pytest.raises(FileNotFoundError,match='ARROWFLOW_SUSHI_ARCHIVE'): load_sushi()
    path=tmp_path/'wrong.zip';path.write_bytes(b'not the audited archive')
    with pytest.raises(ValueError,match='hash'): load_sushi(path)


@pytest.mark.skipif(not os.environ.get('ARROWFLOW_SUSHI_ARCHIVE'),reason='Audited provider archive supplied only through environment')
def test_real_sushi_identity_and_corrected_labels():
    from experiments.make_revision.datasets import load_sushi,SUSHI_ID
    X,y,manifest=load_sushi()
    assert SUSHI_ID=='sushi_childhood_east_west_v2'
    assert X.shape==(5000,10) and np.bincount(y).tolist()==[3258,1742]
    assert np.all(np.sort(X,axis=1)==np.arange(10))
    assert len(set(manifest['provider_user_ids']))==5000
    assert manifest['label_column_one_based']==7 and manifest['legacy_label_disagreements']==105


def test_nested_fit_record_keeps_pipeline_warning_and_time_decomposition():
    from experiments.make_revision.evaluation import ModelSpec,_fit_predict
    c=api(); X=np.arange(40,dtype=float).reshape(10,4); y=np.tile([0,1],5)
    spec=ModelSpec('mlp',lambda config,seed:c.conventional_factory('mlp',config,seed),[{'max_iter':1,'hidden_layer_sizes':(4,)}],True)
    _,record=_fit_predict(spec,spec.candidates[0],19,X,y,X)
    assert any(w['category']=='ConvergenceWarning' for w in record['fit_warnings'])
    assert record['classifier_fit_seconds']>=0 and record['encoding_seconds']>=0


def test_registry_source_seals_include_comparators_and_native_loader():
    from experiments.make_revision.run_revision import environment_record
    record=environment_record('experiments.make_revision.comparisons:e02_registry')
    assert {'experiments/make_revision/comparisons.py','experiments/make_revision/datasets.py'}<=record['source_hashes'].keys()


@pytest.mark.skipif(not os.environ.get('ARROWFLOW_SUSHI_ARCHIVE'),reason='Audited provider archive supplied through environment')
def test_native_cli_prepares_new_target_and_nested_source_row_splits(tmp_path):
    from experiments.make_revision.run_revision import main,load_prepared
    from experiments.make_revision.datasets import SUSHI_ID
    main(['prepare','--dataset',SUSHI_ID,'--output',str(tmp_path),'--sushi-archive',os.environ['ARROWFLOW_SUSHI_ARCHIVE'],
          '--registry','experiments.make_revision.comparisons:native_registry'])
    X,y,manifest,splits=load_prepared(tmp_path,SUSHI_ID)
    assert manifest['dataset_id']==SUSHI_ID and len(splits)==15
    assert all(len(s['inner'])==3 for s in splits)
    assert sorted(splits[0]['train']+splits[0]['test'])==list(range(5000))


@pytest.mark.parametrize('family',['svc_rbf','random_forest','mlp','numeric_knn','gradient_boosting'])
def test_conventional_registry_runs_on_small_synthetic_classification(family):
    c=api(); rng=np.random.RandomState(11); y=np.repeat([0,1,2],20)
    X=rng.normal(size=(60,4))*.1+y[:,None]*3
    spec=c.e02_registry(protocol())[family]
    estimator=spec.factory(spec.candidates[0],19).fit(X,y)
    assert np.array_equal(estimator.predict(X),y)


def test_footrule_nonzero_inverse_distance_can_reverse_uniform_vote():
    c=api(); X=np.array([[0,1,3,2],[3,2,1,0],[3,1,2,0]]); y=[0,1,1]; query=[[0,1,2,3]]
    assert c.StableFootruleKNN(3,weights='uniform').fit(X,y).predict(query).tolist()==[1]
    assert c.StableFootruleKNN(3,weights='distance').fit(X,y).predict(query).tolist()==[0]


def test_seed_derivation_uses_declared_big_endian_sha256_prefix():
    import hashlib
    from experiments.make_revision.evaluation import canonical_json
    parts=[51047,'iris',0,0,-1,2,'random',0]
    assert api().derive_seed(parts[0],*parts[1:])==int.from_bytes(hashlib.sha256(canonical_json(parts).encode()).digest()[:4],'big')


def test_svc_pre_score_convergence_budget_is_one_million_updates():
    assert api().conventional_factory('svc_rbf',{'C':1,'gamma':'scale'},19).named_steps['classifier'].max_iter==1000000


def test_native_cli_smoke_uses_native_adapter_on_synthetic_rankings(tmp_path,monkeypatch):
    from experiments.make_revision import run_revision as runner
    from experiments.make_revision.datasets import SUSHI_ID
    rng=np.random.RandomState(9); X=np.array([rng.permutation(10) for _ in range(60)]); y=np.tile([0,1],30)
    monkeypatch.setattr(runner,'load_dataset',lambda name:(X,y,{'dataset_hash':'synthetic_fixture'}))
    runner.main(['smoke','--dataset',SUSHI_ID,'--output',str(tmp_path)])
    result=json.loads((tmp_path/'smoke.json').read_text())['result']
    assert all(r['model_id']=='smoke_native_arrowflow' and r['status']=='ok' for r in result['models'])
    assert all(r['encoding_seconds']==0 for r in result['models'])


def test_native_v3_registry_gives_arrowflow_a_tuned_grid():
    p=json.loads(Path('experiments/make_revision/protocols/2026-09-12/native_v3.json').read_text())
    reg=api().native_registry(p)
    assert 'native_arrowflow' in reg and len(reg['native_arrowflow'].candidates)==24
    assert reg['native_arrowflow'].stochastic


def native_protocol_2026_09_11():
    return json.loads(Path('experiments/make_revision/protocols/2026-09-11/native.json').read_text())


def native_protocol_v3():
    return json.loads(Path('experiments/make_revision/protocols/2026-09-12/native_v3.json').read_text())


def test_native_v3_grid_is_sampled_from_the_documented_36_combinations():
    from sklearn.model_selection import ParameterGrid
    from experiments.make_revision.evaluation import candidate_grid,config_id
    from experiments.make_revision.run_revision import get_registry
    c=api(); p=native_protocol_v3(); registry=c.native_registry(p)
    assert set(registry)=={'native_svc_rbf','native_random_forest','native_mlp','native_numeric_knn','native_gradient_boosting',
                           'native_dummy','native_footrule_knn','native_borda','native_hdc','native_arrowflow'}
    assert set(get_registry('experiments.make_revision.comparisons:native_registry',p))==set(registry)
    spec=registry['native_arrowflow']
    assert p['native_arrowflow_v3']['candidate_grid']==c.NATIVE_ARROWFLOW_GRID_V3
    assert len(list(ParameterGrid(c.NATIVE_ARROWFLOW_GRID_V3)))==36
    assert spec.candidates==candidate_grid(c.NATIVE_ARROWFLOW_GRID_V3,p['candidate_budget'],p['candidate_seed'])
    assert len({config_id(x) for x in spec.candidates})==24
    assert all(set(x)=={'widths','learning_rate','validation_ratio','p_correct'} for x in spec.candidates)
    for key,levels in c.NATIVE_ARROWFLOW_GRID_V3.items():  # every level of every axis survives the sampling
        assert {json.dumps(x[key]) for x in spec.candidates}=={json.dumps(level) for level in levels}


def test_native_v3_factory_forwards_every_tuned_parameter_and_the_seed():
    c=api(); spec=c.native_registry(native_protocol_v3())['native_arrowflow']
    config={'widths':[64,32],'learning_rate':.05,'validation_ratio':.1,'p_correct':.1}
    model=spec.factory(config,19)
    assert isinstance(model,c.NativeArrowFlow) and model.embed_dim==10
    params=model.get_params()
    assert params=={'widths':[64,32],'iterations':200,'learning_rate':.05,'validation_ratio':.1,'p_correct':.1,'seed':19}
    assert clone(model).get_params()==params
    rng=np.random.RandomState(9); X=np.array([rng.permutation(10) for _ in range(60)]); y=np.tile([0,1],30)
    fitted=c.NativeArrowFlow(widths=(4,),iterations=2,learning_rate=.1,validation_ratio=.1,p_correct=.1,seed=19).fit(X,y)
    assert (fitted.validation_sample_count_,fitted.training_sample_count_)==(6,54)
    assert fitted.config_.val_data_ratio==.1 and fitted.config_.change_probability_when_decision_correct==.1
    assert fitted.config_.learning_rate==.1 and fitted.predict(X).shape==(60,)


def test_native_registry_without_grid_key_reproduces_the_2026_09_11_fixed_families():
    from experiments.make_revision.models import ArrowFlowEstimator
    c=api(); registry=c.native_registry(native_protocol_2026_09_11())
    assert 'native_arrowflow' not in registry
    for name,widths in {'native_arrowflow_64':[64],'native_arrowflow_64_32':[64,32],'native_arrowflow_230':[230]}.items():
        spec=registry[name]
        assert spec.stochastic and spec.candidates==[{'widths':widths,'iterations':200}]
        model=spec.factory(spec.candidates[0],8129)
        reference=ArrowFlowEstimator(embed_dim=10,widths=widths,iterations=200,learning_rate=.5,seed=8129)
        assert vars(model)==vars(reference)
    with pytest.raises(ValueError,match='native_arrowflow_grid'):
        c.native_registry({**native_protocol_2026_09_11(),'native_arrowflow_grid':'v2'})


def test_native_v3_protocol_keeps_the_2026_09_11_design_and_names_its_template():
    import hashlib
    v1=native_protocol_2026_09_11(); v3=native_protocol_v3()
    assert v3['native_arrowflow_grid']=='v3' and v3['protocol_id']=='arrowflow-v3-native-1' and v3['wallclock_cap_hours']==1
    for key in ['datasets','fit_seeds','candidate_budget','candidate_seed','split_seed','outer_folds','outer_repeats','inner_folds',
                'test_train_ratio','selection_metric','stochastic_finalists','report_metrics','production_family',
                'candidate_tie_rule','confidence','augmentation','paired_interval','failure_policy','aggregation_integrity']:
        assert v3[key]==v1[key],key
    template=Path('experiments/make_revision/protocols/2026-09-11/native.json').read_bytes()
    assert v3['source_template_sha256']==hashlib.sha256(template).hexdigest()
    assert v3['internal_validation_ratio']==api().NATIVE_ARROWFLOW_GRID_V3['validation_ratio']
