"""Oracle for nested diversity summaries and explicitly missing correlations."""
import copy
import importlib
import numpy as np
import pytest


def api():
    name = 'experiments.make_revision.view_reporting'
    assert importlib.util.find_spec(name) is not None, 'Missing verified view-diagnostic reporting'
    return importlib.import_module(name).summarize_view_diagnostics


def fixture():
    p = dict(panels={'e05': ['toy']}, outer_repeats=1, outer_folds=3,
             fit_seeds=[11,22,33], e05_schemes=['same'], head_conditions=[True],
             e05_views=3, e05_prefixes=[1,3], architectures=[[4,3]], primary_architecture_index=0)
    results = []
    for fold, base in enumerate([.2,.4,.6]):
        diagnostics, fits = [], []
        for seed_index, (seed, noise) in enumerate(zip(p['fit_seeds'], [-.03,0,.03])):
            error = base + noise
            pairs = []
            for index, (a,b) in enumerate([(0,1),(0,2),(1,2)]):
                corr = None if seed == 11 or seed == 22 and index > 0 else .2 if seed == 22 else .8
                pairs.append(dict(view_a=a,view_b=b,disagreement=.1*(index+1)+.01*(fold+seed_index),
                    double_fault=.1,error_correlation=corr,correlation_reason='constant error vector' if corr is None else None))
            diagnostics.append(dict(scheme='same',head_update=True,model_seed=seed,pairs=pairs,prefixes=[
                dict(prefix=1,first_view_error=error+.1,mean_single_view_error=error+.1,ensemble_error=error+.1,change_from_first=0.,change_from_mean_single=0.),
                dict(prefix=3,first_view_error=error+.1,mean_single_view_error=error+.05,ensemble_error=error,change_from_first=-.1,change_from_mean_single=-.05)]))
            for view in range(3):
                fits.append(dict(scheme='same',head_update=True,model_seed=seed,view=view,status='ok',layers=[
                    dict(depth=depth,prototype_count=count,unique_count=count-1,duplicate_fraction=.1*(view+1),
                         zero_distance_pair_fraction=.05,normalized_distance_mean=.5,normalized_distance_min=.1,
                         normalized_distance_max=.8,query_any_response_tie_fraction=.7,query_repeated_response_fraction=.2)
                    for depth,count in enumerate([4,3],1)]))
        results.append(dict(status='ok',provenance=dict(family='e05',dataset_id='toy',outer_repeat=0,outer_fold=fold),
                            diagnostics=diagnostics,fits=fits))
    return results,p


def test_prefix_changes_average_fitting_seeds_inside_each_fold():
    results,p=fixture(); out=api()(results,p)
    row=next(r for r in out['prefix_summaries'] if r['prefix']==3)
    assert row['metrics']['ensemble_error']['mean']==pytest.approx(.4)
    assert row['metrics']['ensemble_error']['outer_fold_sd']==pytest.approx(.2)
    assert row['metrics']['ensemble_error']['mean_within_fold_seed_sd']==pytest.approx(.03)
    assert row['metrics']['change_from_first']['mean']==pytest.approx(-.1)
    assert len(out['prefix_seed_fold_records'])==18


def test_correlation_uses_replicate_means_and_preserves_missing_denominators():
    results,p=fixture(); out=api()(results,p)
    row=next(r for r in out['pair_summaries'] if r['prefix']==3)
    assert row['metrics']['disagreement']['mean']==pytest.approx(.22)
    assert row['metrics']['error_correlation']['mean']==pytest.approx(.5)
    assert row['defined_correlation_pairs']==12 and row['total_pairs']==27
    assert row['correlation_missing_reasons']=={'constant error vector':15}
    assert row['metrics']['error_correlation']['contributing_seed_cells']==6
    assert row['metrics']['error_correlation']['expected_seed_cells']==9
    single=next(r for r in out['pair_summaries'] if r['prefix']==1)
    assert single['total_pairs']==0 and single['metrics']['error_correlation']['mean'] is None
    assert single['metrics']['error_correlation']['n_contributing_folds']==0
    assert all(r['no_pair_reason']=='one-view prefix has no pair' for r in out['pair_seed_fold_records'] if r['prefix']==1)


def test_layer_diagnostics_average_views_before_seeds_and_folds():
    results,p=fixture(); out=api()(results,p)
    row=next(r for r in out['layer_summaries'] if r['prefix']==3 and r['depth']==2)
    assert row['prototype_count']==3
    assert row['metrics']['duplicate_fraction']['mean']==pytest.approx(.2)
    assert row['metrics']['duplicate_fraction']['n_contributing_folds']==3
    assert len(out['layer_seed_fold_records'])==36


@pytest.mark.parametrize('change',['empty','fold','fit_seed','view','prefix','pair','layer'])
def test_diagnostic_aggregation_rejects_an_incomplete_declared_schedule(change):
    results,p=fixture()
    if change=='empty': results=[]
    elif change=='fold': results.pop()
    elif change=='fit_seed': results[0]['diagnostics'].pop()
    elif change=='view': results[0]['fits'].pop()
    elif change=='prefix': results[0]['diagnostics'][0]['prefixes'].pop()
    elif change=='pair': results[0]['diagnostics'][0]['pairs'].pop()
    elif change=='layer': results[0]['fits'][0]['layers'].pop()
    with pytest.raises(ValueError): api()(results,p)
