"""The prespecified analysis of the permuted-label control (design: the header of label_controls.py and the frozen protocol
protocols/2026-11-25/permuted_labels.json). It is run through label_controls:

    python -m experiments.make_revision.label_controls analyse --output DIR [--tables DIR]

Order of work, so that no score is read before the run is complete and verified:
1. the completeness gate parses file names only: design.json must be the frozen design, code_revision.txt a committed
   revision, and results/ must hold one record for each of the 255 planned jobs and nothing else;
2. the registered inputs are re-verified: the pinned files and the digest of every per-job file of the motion run,
   motion_controls.summary (which re-derives every sealed selection from its reference run and re-checks every job) must
   equal the published summary, every reference production run is re-verified with compare_runs.load_run and verify_run
   (compare_motion.verify_references), and the diagnostics checkpoints are pinned;
3. every new record is validated against the plan, the registered predictions (AF's refit and the readout of the initial
   filters must equal the registered views7 and frozen predictions row by row) and its own checks;
4. only then is any accuracy computed: every number is recomputed from per-row predictions against the registered labels;
   AF's and FR's are also checked against the registered model rows, and G against the registered motion analysis.
Outputs: DIR/analysis.json, DIR/ANALYSIS.md (the reading the rule produces) and the LaTeX tables tab_s_label_*.tex in
DIR/tables (or --tables), in the style of the manuscript's tab_s_motion_family.tex and tab_s_motion_scramble.tex.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy import stats
from . import compare_motion as cm
from . import label_controls as lc
from . import motion_controls as mc
from . import training_diagnostics as td
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .run_revision import load_prepared

ARMS = ('AF', 'FR', 'PL[match]', 'PL[own]', 'PL[final]')
PL_ARMS = {'match': 'PL[match]', 'own': 'PL[own]', 'final': 'PL[final]'}
FAMILIES = {'F1': ('AF', 'PL[match]', 'primary'), 'F2': ('PL[match]', 'FR', 'primary, separate family'),
            'F3': ('AF', 'PL[own]', 'secondary'), 'F4': ('PL[own]', 'FR', 'secondary'),
            'G': ('AF', 'FR', 'the size of the gain')}
READING_FAMILIES = {'primary': ('F1', 'F2', 'match'), 'secondary': ('F3', 'F4', 'own')}
SUBSETS = {'all': None, 'further': 'further', 'benchmark': 'benchmark'}
OUTPUT_TABLES = ('tab_s_label_family', 'tab_s_label_secondary', 'tab_s_label_summary', 'tab_s_label_readouts',
                 'tab_s_label_mechanism', 'tab_s_label_checkpoints', 'tab_s_label_purity', 'tab_s_label_views',
                 'tab_s_label_recall')
TOLERANCE = 1e-12


class AnalysisRefused(SystemExit):
    """The analysis refuses: the run is incomplete or unverified; nothing is written."""


# ----------------------------------------------------------------------------- statistics

def signed_rank(differences):
    """Exact two-sided Wilcoxon signed-rank test over the given differences: zeros dropped, tied absolute differences at
    their average rank, p the share of the 2^n sign assignments whose positive rank sum lies at least as far from its
    centre as the observed one. Equals scipy's exact test when there is no zero and no tie (checked)."""
    d = np.asarray(differences, dtype=float)
    nonzero = d[d != 0]
    n = len(nonzero)
    if n == 0:
        return {'n': 0, 'zeros': int(len(d)), 'ties': False, 'w_plus': 0., 'w_minus': 0., 'p': 1.}
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus, w_minus = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    doubled = [int(round(2 * r)) for r in ranks]
    total = sum(doubled)
    ways = [0] * (total + 1)
    ways[0] = 1
    for r in doubled:
        for s in range(total - r, -1, -1):
            if ways[s]:
                ways[s + r] += ways[s]
    observed = abs(2 * int(round(2 * w_plus)) - total)
    p = sum(count for s, count in enumerate(ways) if abs(2 * s - total) >= observed) / 2 ** n
    ties = len(np.unique(np.abs(nonzero))) != n
    if not ties and n == len(d):
        reference = float(stats.wilcoxon(d, alternative='two-sided', method='exact').pvalue)
        if abs(reference - p) > 1e-12:
            raise AssertionError(f'signed-rank test: the exact count {p} differs from scipy\'s {reference}')
    return {'n': n, 'zeros': int(len(d) - n), 'ties': bool(ties), 'w_plus': w_plus, 'w_minus': w_minus,
            'p': float(min(1., p))}


def t_interval(values, confidence=.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    half = float(stats.t.ppf(.5 + confidence / 2, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def direction(test):
    if test['p'] < lc.ALPHA and test['w_plus'] != test['w_minus']:
        return 'positive' if test['w_plus'] > test['w_minus'] else 'negative'
    return 'n.s.'


def pattern(d, s):
    """The reading rule of the design, in its order: (f), then (d), (c), (b), (a), and (e) for the rest."""
    if d == 'negative':
        return 'f'
    if d == 'positive':
        return {'negative': 'd', 'n.s.': 'c', 'positive': 'b'}[s]
    return 'a' if s == 'positive' else 'e'


def percentile(values, q):
    """NumPy's default (linear) percentile, extended to infinite values: an end on or next to an infinite order statistic
    is infinite."""
    v = np.sort(np.asarray(values, dtype=float))
    at = (len(v) - 1) * q / 100
    lo, hi = int(np.floor(at)), int(np.ceil(at))
    if not np.isfinite(v[lo]):
        return float(v[lo])
    if not np.isfinite(v[hi]):
        return float(v[hi])
    return float(v[lo] + (v[hi] - v[lo]) * (at - lo))


def ratio_of_sums(numerator, denominator):
    """sum(numerator)/sum(denominator) over the 17 datasets with the design's bootstrap percentile interval."""
    num, den = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    idx = np.random.default_rng(lc.BOOTSTRAP_SEED).integers(0, len(num), (lc.BOOTSTRAP_RESAMPLES, len(num)))
    top, bottom = num[idx].sum(1), den[idx].sum(1)
    defined = bottom > 0
    ratio = np.divide(top, bottom, out=np.zeros_like(top), where=defined)
    tail = (100 - lc.CONFIDENCE) / 2
    lo = percentile(np.where(defined, ratio, -np.inf), tail)
    hi = percentile(np.where(defined, ratio, np.inf), 100 - tail)
    return {'value': float(num.sum() / den.sum()) if den.sum() > 0 else None, 'numerator_sum': float(num.sum()),
            'denominator_sum': float(den.sum()), 'ci_low': lo if np.isfinite(lo) else None,
            'ci_high': hi if np.isfinite(hi) else None, 'undefined_resamples': int((~defined).sum())}


# ----------------------------------------------------------------------------- the gate and the inputs

def completeness_gate(out, allow_uncommitted):
    """Refuse unless the run is complete; only file names are parsed, so no record holding a score is read before."""
    problems = []
    if not (out/'design.json').is_file() or (out/'design.json').read_text() != lc.protocol_text():
        problems.append('design.json is missing or is not the frozen design')
    revision = (out/'code_revision.txt').read_text().strip() if (out/'code_revision.txt').is_file() else None
    if revision is None:
        problems.append('code_revision.txt is missing')
    elif revision.endswith('+uncommitted') and not allow_uncommitted:
        problems.append(f'the records come from uncommitted code ({revision})')
    expected = {f'{stem}.json' for stem in lc.job_stems()}
    present = {p.name for p in (out/'results').iterdir()} if (out/'results').is_dir() else set()
    if expected - present:
        problems.append(f'{len(expected - present)} of {len(expected)} records missing (first: {sorted(expected - present)[0]})')
    if present - expected:
        problems.append(f'unexpected files in results/: {sorted(present - expected)[:3]}')
    if problems:
        raise AnalysisRefused(f'analyse refused, nothing written: {out}: ' + '; '.join(problems))
    return revision


def verify_registered():
    """The motion run and its references re-verified before any score is read."""
    start = time.perf_counter()
    lc.verify_protocol()
    lc.verify_pins()
    protocol = json.loads((lc.MOTION_RUN/'protocol.json').read_text())
    manifest = json.loads((lc.MOTION_RUN/'manifest.json').read_text())
    report, _ = mc.summary(lc.MOTION_RUN)
    published = json.loads((lc.MOTION_RUN/mc.SUMMARY_JSON).read_text())
    if canonical_json(report) != canonical_json(published):
        raise AnalysisRefused(f'the re-verified motion summary differs from the published {mc.SUMMARY_JSON}')
    if any(v['matching_fold_seeds'] != v['total_fold_seeds'] for v in
           (e['views7_reproduces_reference'] for e in report['summaries'].values())):
        raise AnalysisRefused('the registered views7 arm does not reproduce its reference everywhere')
    references = cm.verify_references(protocol, manifest, {}, smoke=False)
    analysis = json.loads((lc.MOTION_ANALYSIS/'motion_analysis.json').read_text())
    return {'motion_run': str(lc.MOTION_RUN), 'motion_protocol_id': protocol['protocol_id'],
            'motion_code_revision': report['code_revision'], 'motion_summary_reverified': True,
            'references': references, 'pinned': lc.PINNED, 'motion_files_digest': lc.MOTION_FILES_DIGEST,
            'diagnostics_run': str(lc.DIAGNOSTICS_RUN), 'seconds': time.perf_counter() - start,
            'registered_frozen_rows': {r['dataset']: r for r in analysis['primary_family'] if r['arm'] == 'frozen'}}


def validate_record(record, job, registered, y, test, revision):
    """One new record against the plan, the registered predictions and its own checks."""
    def require(condition, message):
        if not condition:
            raise AnalysisRefused(f"record {job['stem']}: {message}")
    require(record['stem'] == job['stem'] and record['dataset_id'] == job['dataset_id']
            and (record['outer_repeat'], record['outer_fold']) == (job['outer_repeat'], job['outer_fold'])
            and record['config_id'] == job['config_id'] and record['hidden_layers'] == job['hidden_layers'],
            'it describes another job')
    require(record['labels'] == 'permuted', 'it was not fitted with the permuted labels')
    require(record['n_test'] == len(test) and record['test_ids_sha256'] == lc.array_hash(np.asarray(test)),
            'its test rows differ')
    require(record['environment']['code_revision'] == revision.replace('+uncommitted', ''),
            'it was written by another code revision')
    require(sorted(record['checks']) == sorted(lc.JOB_CHECKS), 'its check schedule differs')
    first = bool(job['check_uninstrumented'])
    for name, entry in record['checks'].items():
        expected = first if name in ('frozen_permuted', 'uninstrumented_fit') else True
        require(entry['performed'] is expected and entry['passed'] is (True if expected else None), f'check {name}')
    require([f['seed'] for f in record['fits']] == list(lc.FIT_SEEDS), 'its fitting seeds differ')
    classes = set(np.unique(y).tolist())
    for fit in record['fits']:
        seed = fit['seed']
        require(fit['af']['predictions'] == registered['predictions'][('views7', seed)].tolist(),
                f'seed {seed}: AF\'s refit differs from the registered views7 predictions')
        require(fit['fr0']['predictions'] == registered['predictions'][('frozen', seed)].tolist(),
                f'seed {seed}: the readout of the initial filters differs from the registered frozen predictions')
        if first:
            require(fit['fr_perm']['predictions'] == registered['predictions'][('frozen', seed)].tolist(),
                    f'seed {seed}: the frozen refit with permuted labels differs from FR')
        require(fit['pl']['permutation']['class_counts_equal'] and fit['pl']['permutation']['rng_unchanged'],
                f'seed {seed}: the permutation record fails')
        for readout in lc.READOUTS:
            block = fit['pl']['readouts'][readout]
            require(len(block['predictions']) == len(test) and set(block['predictions']) <= classes
                    and len(block['view_predictions']) == 7 and all(len(v) == len(test) for v in block['view_predictions']),
                    f'seed {seed}: the {readout} predictions have another shape or unknown labels')
            require(block['predictions'] == np.asarray(mc._majority(np.asarray(block['view_predictions']))).tolist(),
                    f'seed {seed}: the {readout} predictions are not the majority of its views')
        require(fit['pl']['readouts']['match']['updates'] == fit['af']['checkpoint'],
                f'seed {seed}: the matched stop is not ArrowFlow\'s checkpoint')


# ----------------------------------------------------------------------------- per-row predictions to accuracies

def accuracy(pred, truth):
    return float(np.mean(np.asarray(pred) == np.asarray(truth)))


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def arm_mean(rows, name, arm):
    """Mean outer-fold accuracy (%) of one arm, fitting seeds averaged within each fold."""
    by_fold = defaultdict(list)
    for r in rows[name]:
        if r['model_id'] == arm:
            by_fold[(r['outer_repeat'], r['outer_fold'])].append(r['accuracy'])
    return 100 * float(np.mean([np.mean(v) for _, v in sorted(by_fold.items())]))


def collect(out, revision):
    """Every record validated, then the per-fold accuracies and descriptive values recomputed from predictions."""
    plan = lc.motion_plan()
    data = {}
    for name in plan['datasets']:
        X, y, manifest, splits = load_prepared(lc.MOTION_RUN, name)
        data[name] = {'y': y, 'splits': {(s['outer_repeat'], s['outer_fold']): s for s in splits},
                      'labels': list(manifest['label_map']), 'classes': np.unique(y)}
    rows = defaultdict(list)           # dataset -> pseudo model rows for paired_corrected_interval
    per_fold = defaultdict(lambda: defaultdict(dict))
    desc = defaultdict(lambda: defaultdict(list))
    for stem in lc.job_stems():
        job = plan['jobs'][stem]
        name, key = job['dataset_id'], (job['outer_repeat'], job['outer_fold'])
        y, split = data[name]['y'], data[name]['splits'][key]
        test = [int(i) for i in split['test']]
        truth, classes = y[test], data[name]['classes']
        registered = lc.registered_job(stem, y, test)
        record = json.loads((out/'results'/f'{stem}.json').read_text())
        validate_record(record, job, registered, y, test, revision)
        L = int(job['hidden_layers'])
        fold_values = defaultdict(list)
        for fit in record['fits']:
            seed = fit['seed']
            preds = {'AF': registered['predictions'][('views7', seed)], 'FR': registered['predictions'][('frozen', seed)],
                     **{PL_ARMS[r]: np.asarray(fit['pl']['readouts'][r]['predictions']) for r in lc.READOUTS}}
            for arm, pred in preds.items():
                value = accuracy(pred, truth)
                if arm in ('AF', 'FR'):
                    stored = registered['models'][('views7' if arm == 'AF' else 'frozen', seed)]['accuracy']
                    if abs(stored - value) > TOLERANCE:
                        raise AnalysisRefused(f'{stem} seed {seed}: the registered {arm} accuracy differs from its '
                                              'predictions')
                rows[name].append({'dataset_id': name, 'model_id': arm, 'outer_repeat': key[0], 'outer_fold': key[1],
                                   'model_seed': seed, 'accuracy': value, 'status': 'ok'})
                fold_values[('recall', arm)].append(mc.per_class_recall(truth, pred, classes))
            # per-view accuracies by encoder type
            views = {'AF': fit['af']['view_predictions'], 'FR': fit['fr0']['view_predictions'],
                     **{PL_ARMS[r]: fit['pl']['readouts'][r]['view_predictions'] for r in lc.READOUTS}}
            for arm, by_view in views.items():
                accs = [accuracy(p, truth) for p in by_view]
                fold_values[('views_lda', arm)].append(float(np.mean([accs[v] for v in lc.LDA_VIEWS])))
                fold_values[('views_other', arm)].append(float(np.mean([a for v, a in enumerate(accs)
                                                                         if v not in lc.LDA_VIEWS])))
            # purity: PL from the records, AF and FR registered
            for arm, model in (('AF', 'views7'), ('FR', 'frozen')):
                row = registered['models'][(model, seed)]
                fold_values[('purity_test', arm)].append(row['neighborhood_purity_test'])
                fold_values[('purity_train', arm)].append(row['neighborhood_purity_train'])
            for r in lc.READOUTS:
                purity = fit['pl']['readouts'][r]['purity']
                fold_values[('purity_test', PL_ARMS[r])].append(purity['test'])
                fold_values[('purity_train', PL_ARMS[r])].append(purity['train'])
            # displacement: last hidden layer and output layer, per readout
            for label, entries in (('AF[checkpoint]', fit['af']['displacement']['checkpoint']),
                                   ('AF[final]', fit['af']['displacement']['final']),
                                   *((PL_ARMS[r], fit['pl']['readouts'][r]['displacement']) for r in lc.READOUTS)):
                by_layer = {e['layer']: e for e in entries}
                fold_values[('displacement_last', label)].append(by_layer[L - 1]['mean_normalized_footrule'])
                fold_values[('displacement_output', label)].append(by_layer[L]['mean_normalized_footrule'])
                fold_values[('changed_last', label)].append(by_layer[L - 1]['mean_changed_share'])
                if L == 2:
                    fold_values[('displacement_first', label)].append(by_layer[0]['mean_normalized_footrule'])
            # checkpoints and votes
            desc[name]['checkpoint_AF'].extend(fit['af']['checkpoint'])
            desc[name]['checkpoint_PL'].extend(fit['pl']['checkpoint'])
            for af_votes, pl_votes in zip(fit['af']['votes'], fit['pl']['votes']):
                for layer in af_votes['final']:
                    position = 'last' if int(layer) == L - 1 else 'first'
                    for label, counts in (('AF[run]', af_votes['final'][layer]), ('AF[to match]', af_votes['checkpoint'][layer]),
                                          ('PL[run]', pl_votes['final'][layer]), ('PL[to match]', pl_votes['match'][layer]),
                                          ('PL[to own]', pl_votes['own'][layer])):
                        desc[name][f'votes|{position}|{label}'].append(counts)
            permutation = fit['pl']['permutation']
            for field in ('agreement', 'expected_agreement', 'cramers_v'):
                desc[name][f'permutation_{field}'].append(permutation[field])
        for (kind, arm), values in fold_values.items():
            if kind == 'recall':
                labels = sorted(values[0])
                per_fold[name][(kind, arm)][key] = {c: float(np.mean([v[c] for v in values])) for c in labels}
            else:
                per_fold[name][(kind, arm)][key] = mean_or_none(values)
    return data, rows, per_fold, desc


# ----------------------------------------------------------------------------- families and the reading

def family(rows, names, first, second):
    out = []
    for name in names:
        expected = sorted({(r['outer_repeat'], r['outer_fold']) for r in rows[name]})
        interval = paired_corrected_interval(rows[name], first, second, metric='accuracy', q=lc.TEST_RATIO,
                                             confidence=lc.CONFIDENCE / 100, expected_folds=expected,
                                             expected_seeds={first: list(lc.FIT_SEEDS), second: list(lc.FIT_SEEDS)})
        out.append({'dataset': name, 'mean_difference': 100 * interval['mean_difference'],
                    'ci_low': 100 * interval['ci_low'], 'ci_high': 100 * interval['ci_high'],
                    'standard_error': 100 * interval['standard_error'], 'p': interval['p_approximate'],
                    'n_folds': interval['n_folds'], 'df': interval['df'], 'raw': interval})
    for row, adjusted in zip(out, holm_adjust([r['p'] for r in out])):
        row['p_holm'] = float(adjusted)
    return {'first': first, 'second': second, 'rows': out,
            'higher': int(sum(r['mean_difference'] > 0 for r in out)),
            'lower': int(sum(r['mean_difference'] < 0 for r in out)),
            'significantly_higher': int(sum(r['mean_difference'] > 0 and r['p_holm'] < lc.ALPHA for r in out)),
            'significantly_lower': int(sum(r['mean_difference'] < 0 and r['p_holm'] < lc.ALPHA for r in out))}


def across(fam, names):
    keyed = {r['dataset']: r for r in fam['rows']}
    values = [keyed[d]['mean_difference'] for d in names]
    mean, lo, hi = t_interval(values)
    test = signed_rank(values)
    return {'datasets': list(names), 'n': len(names), 'mean': mean, 'ci_low': lo, 'ci_high': hi, 'signed_rank': test,
            'direction': direction(test),
            'higher': int(sum(v > 0 for v in values)), 'lower': int(sum(v < 0 for v in values)),
            'significantly_higher': int(sum(keyed[d]['mean_difference'] > 0 and keyed[d]['p_holm'] < lc.ALPHA for d in names)),
            'significantly_lower': int(sum(keyed[d]['mean_difference'] < 0 and keyed[d]['p_holm'] < lc.ALPHA for d in names))}


def subsets(datasets):
    return {'all': list(datasets), 'further': [d for d in datasets if d not in lc.BENCHMARK],
            'benchmark': [d for d in datasets if d in lc.BENCHMARK]}


def verify_gain(fam, registered_rows):
    """G recomputed here must be the registered motion analysis's frozen-arm family, row by row."""
    for row in fam['rows']:
        held = registered_rows[row['dataset']]
        for mine, theirs in (('mean_difference', 'mean_difference'), ('ci_low', 'ci_low'), ('ci_high', 'ci_high')):
            if abs(row['raw'][mine] - held[theirs]) > 1e-12:
                raise AnalysisRefused(f"G on {row['dataset']}: {mine} differs from the registered motion analysis")
        for mine, theirs in (('p', 'p_approximate'), ('p_holm', 'holm_p_approximate')):
            if abs(row[mine] - held[theirs]) > 1e-12:
                raise AnalysisRefused(f"G on {row['dataset']}: {mine} differs from the registered motion analysis")
    return True


def share_text(x):
    return 'unbounded' if x is None else f'{100 * x:.0f} %'


def interval_text(lo, hi):
    return f"95 % interval {'unbounded' if lo is None else format(100 * lo, '.0f')} to " \
           f"{'unbounded' if hi is None else format(100 * hi, '.0f') + ' %'}"


def reading(families, summary, shares, displacement):
    """The design's reading rule with its wordings filled in, for the primary and the secondary readout."""
    gain = summary['G']['all']
    out = {}
    for which, (fd, fs, readout) in READING_FAMILIES.items():
        d, s = summary[fd]['all'], summary[fs]['all']
        letter = pattern(d['direction'], s['direction'])
        x_share = shares[readout]['label_free']
        values = {'k': families[fd]['higher'], 'm': families[fd]['significantly_higher'], 'abs_s': f"{abs(s['mean']):.2f}",
                  'x': f"{displacement[PL_ARMS[readout]]:.3f}", 'y': f"{displacement['AF[checkpoint]']:.3f}",
                  's': f"{s['mean']:+.2f}", 'l': f"{s['ci_low']:+.2f}", 'u': f"{s['ci_high']:+.2f}",
                  's_points': f"{s['mean']:.2f}", 'x_share': share_text(x_share['value']),
                  'x_interval': interval_text(x_share['ci_low'], x_share['ci_high']), 'd_points': f"{d['mean']:.2f}",
                  'k_s': families[fs]['higher'], 'd': f"{d['mean']:+.2f}", 'p_d': f"{d['signed_rank']['p']:.3g}"}
        sentences = [lc.WORDINGS[letter].format(**values)]
        if letter == 'd':
            sentences.append(lc.D_DAMAGE)
        sentences.append(lc.G_SENTENCE.format(g=f"{gain['mean']:+.2f}", k_g=families['G']['higher'],
                                              m_g=families['G']['significantly_higher']))
        if d['direction'] != 'positive':
            sentences.append(lc.WITHDRAW)
        out[which] = {'families': [fd, fs], 'readout': readout, 'D': d['direction'], 'S': s['direction'],
                      'pattern': letter, 'unlisted_cell': d['direction'] == 'n.s.' and s['direction'] == 'negative',
                      'label_share_reported': letter == 'b', 'sentences': sentences}
    out['same_pattern'] = out['primary']['pattern'] == out['secondary']['pattern']
    out['stopping_rule'] = None if out['same_pattern'] else lc.STOPPING.format(
        secondary=out['secondary']['pattern'], primary=out['primary']['pattern'])
    return out


# ----------------------------------------------------------------------------- descriptive summaries

def fold_mean(per_fold, name, kind, arm):
    values = [v for v in per_fold[name][(kind, arm)].values() if v is not None]
    return float(np.mean(values)) if values else None


def descriptive(datasets, data, per_fold, desc, means):
    out = {'votes': {}, 'displacement': {}, 'checkpoints': {}, 'purity': {}, 'recall': {}, 'final': {}, 'permutation': {},
           'views': {}}
    labels = ('AF[checkpoint]', 'AF[final]', *PL_ARMS.values())
    for name in datasets:
        votes = {}
        for position in ('last', 'first'):
            entry = {}
            for label in ('AF[run]', 'AF[to match]', 'PL[run]', 'PL[to match]', 'PL[to own]'):
                counts = desc[name].get(f'votes|{position}|{label}')
                if counts:
                    slots, nonzero = map(int, np.sum(np.asarray(counts, dtype=np.int64), axis=0))
                    entry[label] = {'slots': slots, 'nonzero': nonzero, 'share': nonzero / slots if slots else None}
            if entry:
                entry['PL/AF nonzero, run'] = (entry['PL[run]']['nonzero'] / entry['AF[run]']['nonzero']
                                               if entry['AF[run]']['nonzero'] else None)
                entry['PL/AF nonzero, to match'] = (entry['PL[to match]']['nonzero'] / entry['AF[to match]']['nonzero']
                                                    if entry['AF[to match]']['nonzero'] else None)
                votes[position] = entry
        out['votes'][name] = votes
        out['displacement'][name] = {kind: {label: fold_mean(per_fold, name, kind, label) for label in labels}
                                     for kind in ('displacement_last', 'displacement_output', 'changed_last',
                                                  'displacement_first')}
        out['checkpoints'][name] = {}
        for arm in ('AF', 'PL'):
            values = np.asarray(desc[name][f'checkpoint_{arm}'], dtype=float)
            out['checkpoints'][name][arm] = {'n': int(len(values)), 'median': float(np.median(values)),
                                             'q25': float(np.percentile(values, 25)), 'q75': float(np.percentile(values, 75)),
                                             'share_at_0': float(np.mean(values == 0))}
        out['purity'][name] = {f'{where}|{arm}': fold_mean(per_fold, name, f'purity_{where}', arm)
                               for where in ('test', 'train') for arm in ARMS}
        out['recall'][name] = {}
        for arm in ARMS:
            by_fold = list(per_fold[name][('recall', arm)].values())
            out['recall'][name][arm] = {data[name]['labels'][int(c)]: float(np.mean([f[c] for f in by_fold]))
                                        for c in sorted(by_fold[0], key=int)}
        out['final'][name] = {'PL[final]': means[(name, 'PL[final]')],
                              'PL[final] - FR': means[(name, 'PL[final]')] - means[(name, 'FR')],
                              'AF - PL[final]': means[(name, 'AF')] - means[(name, 'PL[final]')]}
        out['permutation'][name] = {field: float(np.mean(desc[name][f'permutation_{field}']))
                                    for field in ('agreement', 'expected_agreement', 'cramers_v')}
        group = {}
        for kind in ('views_lda', 'views_other'):
            acc = {arm: 100 * fold_mean(per_fold, name, kind, arm) for arm in ARMS}
            group[kind] = {'accuracy': acc,
                           **{f'D[{r}]': acc['AF'] - acc[PL_ARMS[r]] for r in ('match', 'own')},
                           **{f'S[{r}]': acc[PL_ARMS[r]] - acc['FR'] for r in ('match', 'own')}}
        out['views'][name] = group
    mean_over = lambda f: float(np.mean([f(d) for d in datasets]))
    out['views_mean'] = {kind: {key: mean_over(lambda d: out['views'][d][kind][key])
                                for key in ('D[match]', 'S[match]', 'D[own]', 'S[own]')}
                         for kind in ('views_lda', 'views_other')}
    out['displacement_mean_last'] = {label: mean_over(lambda d: out['displacement'][d]['displacement_last'][label])
                                     for label in labels}
    return out


# ----------------------------------------------------------------------------- the analysis

def analyse(out, *, tables=None, allow_uncommitted=False):
    out = lc.guard_output(out)
    revision = completeness_gate(out, allow_uncommitted)
    tables = out/'tables' if tables is None else Path(tables).resolve()
    for protected in lc.PROTECTED:
        if tables == protected.resolve() or protected.resolve() in tables.parents:
            raise AnalysisRefused(f'refusing to write tables into the registered run {protected}')
    registered = verify_registered()
    datasets = list(lc.motion_plan()['datasets'])
    data, rows, per_fold, desc = collect(out, revision)
    means = {(name, arm): arm_mean(rows, name, arm) for name in datasets for arm in ARMS}
    families = {key: {'role': role, **family(rows, datasets, a, b)} for key, (a, b, role) in FAMILIES.items()}
    verify_gain(families['G'], registered['registered_frozen_rows'])
    for name in datasets:                 # G = D + S on every dataset, for both readouts
        keyed = {k: {r['dataset']: r['mean_difference'] for r in families[k]['rows']} for k in families}
        for fd, fs in (('F1', 'F2'), ('F3', 'F4')):
            if abs(keyed['G'][name] - keyed[fd][name] - keyed[fs][name]) > 1e-9:
                raise AnalysisRefused(f'{name}: G differs from D + S ({fd}, {fs})')
    groups = subsets(datasets)
    summary = {key: {label: across(families[key], names) for label, names in groups.items()} for key in families}
    gaps = {key: [{r['dataset']: r['mean_difference'] for r in families[key]['rows']}[d] for d in datasets] for key in families}
    shares = {readout: {'label': ratio_of_sums(gaps[fd], gaps['G']), 'label_free': ratio_of_sums(gaps[fs], gaps['G'])}
              for fd, fs, readout in READING_FAMILIES.values()}
    described = descriptive(datasets, data, per_fold, desc, means)
    result_reading = reading(families, summary, shares, described['displacement_mean_last'])
    named = [{'family': key, **{k: v for k, v in row.items() if k != 'raw'}} for key in ('F1', 'F2')
             for row in families[key]['rows'] if row['dataset'] in lc.NAMED_SUBSET]
    result = {'design': f'label_controls.py header and {lc.PROTOCOL.name} (sha256 {lc.PROTOCOL_SHA256})',
              'output': str(out), 'code_revision': revision, 'registered': {k: v for k, v in registered.items()
                                                                           if k != 'registered_frozen_rows'},
              'arms': list(ARMS), 'means': {f'{d}|{a}': means[(d, a)] for d in datasets for a in ARMS},
              'families': {k: {**{kk: vv for kk, vv in f.items() if kk != 'rows'},
                               'rows': [{kk: vv for kk, vv in r.items() if kk != 'raw'} for r in f['rows']]}
                           for k, f in families.items()},
              'named_subset': named, 'across_datasets': summary, 'shares': shares, 'reading': result_reading,
              'descriptive': described, 'interpretation': lc.INTERPRETATION}
    result = td.native(result)
    (out/'analysis.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    text = markdown(result, datasets)
    (out/'ANALYSIS.md').write_text(text)
    write_tables(tables, result, datasets)
    print(text)
    return result


# ----------------------------------------------------------------------------- ANALYSIS.md

def md_table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |'
                                                                           for r in rows]


def fmt(v, d=2, signed=True):
    if v is None:
        return '-'
    return f'{v:+.{d}f}' if signed else f'{v:.{d}f}'


def markdown(result, datasets):
    fam, summary, rd, desc = result['families'], result['across_datasets'], result['reading'], result['descriptive']
    lines = ['# Permuted-label control: analysis', '',
             f"Run `{result['output']}` (code revision {result['code_revision']}). Design: {result['design']}. "
             f"Registered inputs re-verified in {result['registered']['seconds']:.0f} s (motion run "
             f"{result['registered']['motion_protocol_id']} at {result['registered']['motion_code_revision']}).", '',
             '## Reading (the pre-registered rule)', '']
    for which in ('primary', 'secondary'):
        r = rd[which]
        lines.append(f"- {which.capitalize()} ({' and '.join(r['families'])}, PL[{r['readout']}]): D {r['D']}, S {r['S']} "
                     f"-> pattern **({r['pattern']})**" + (' (a cell the specification\'s table does not list; read as (e))'
                                                           if r['unlisted_cell'] else '') + '.')
    lines += ['', 'Wording (primary):'] + [f'\n> {s}' for s in rd['primary']['sentences']]
    if not rd['same_pattern']:
        lines += ['', 'Wording (secondary):'] + [f'\n> {s}' for s in rd['secondary']['sentences']]
        lines += ['', f"> {rd['stopping_rule']}"]
    for readout in ('match', 'own'):
        if rd['primary' if readout == 'match' else 'secondary']['label_share_reported']:
            s = result['shares'][readout]
            lines += ['', f"Label share sum(D)/sum(G), PL[{readout}]: {share_text(s['label']['value'])} "
                          f"({interval_text(s['label']['ci_low'], s['label']['ci_high'])}); X = sum(S)/sum(G): "
                          f"{share_text(s['label_free']['value'])} "
                          f"({interval_text(s['label_free']['ci_low'], s['label_free']['ci_high'])})."]
    lines += ['', '## Across datasets (points; signed-rank p exact, two-sided)', '']
    header = ['contrast', 'datasets', 'mean', '95 % t interval', 'signed-rank p', 'direction', 'higher', 'lower',
              'Holm higher', 'Holm lower']
    body = []
    for key in ('G', 'F1', 'F2', 'F3', 'F4'):
        for label in SUBSETS:
            s = summary[key][label]
            body.append([f"{key} ({fam[key]['first']} - {fam[key]['second']})", f"{label} ({s['n']})", fmt(s['mean']),
                         f"[{fmt(s['ci_low'])}, {fmt(s['ci_high'])}]", f"{s['signed_rank']['p']:.4g}", s['direction'],
                         s['higher'], s['lower'], s['significantly_higher'], s['significantly_lower']])
    lines += md_table(header, body)
    lines += ['', '## Mean outer-fold accuracy (%), fitting seeds averaged', '']
    lines += md_table(['dataset'] + list(result['arms']),
                      [[d] + [f"{result['means'][f'{d}|{a}']:.2f}" for a in result['arms']] for d in datasets])
    for key, f in fam.items():
        lines += ['', f"## {key} ({f['role']}): {f['first']} - {f['second']} (points; Holm over 17)", '',
                  f"Higher on {f['higher']}/17, lower on {f['lower']}; significantly higher on {f['significantly_higher']}, "
                  f"significantly lower on {f['significantly_lower']}.", '']
        lines += md_table(['dataset', 'difference', '95 % interval', 'p', 'Holm p'],
                          [[r['dataset'], fmt(r['mean_difference']), f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]",
                            f"{r['p']:.4f}", f"{r['p_holm']:.4f}"] for r in f['rows']])
    lines += ['', '## Named subset (F1 and F2 rows, their own Holm p)', '']
    lines += md_table(['family', 'dataset', 'difference', 'Holm p'],
                      [[r['family'], r['dataset'], fmt(r['mean_difference']), f"{r['p_holm']:.4f}"]
                       for r in result['named_subset']])
    lines += ['', '## Descriptive (declared in advance; no test)', '', '### Vote volume (share of selected slots with a '
              'nonzero vote, last hidden layer)', '']
    lines += md_table(['dataset', 'AF run', 'PL run', 'AF to match', 'PL to match', 'PL to own', 'PL/AF run',
                       'PL/AF to match'],
                      [[d] + [fmt(desc['votes'][d]['last'][k]['share'], 3, False) for k in
                              ('AF[run]', 'PL[run]', 'AF[to match]', 'PL[to match]', 'PL[to own]')]
                       + [fmt(desc['votes'][d]['last'][k], 2, False) for k in ('PL/AF nonzero, run', 'PL/AF nonzero, to match')]
                       for d in datasets])
    lines += ['', '### Displacement of the last hidden layer (mean normalized footrule from the initial filters)', '']
    labels = ('AF[checkpoint]', 'AF[final]', 'PL[match]', 'PL[own]', 'PL[final]')
    lines += md_table(['dataset', *labels],
                      [[d] + [fmt(desc['displacement'][d]['displacement_last'][k], 4, False) for k in labels]
                       for d in datasets])
    lines += ['', '### Checkpoints (update; over folds, seeds and views)', '']
    lines += md_table(['dataset', 'AF median [IQR]', 'AF at 0', 'PL own median [IQR]', 'PL own at 0'],
                      [[d] + [x for arm in ('AF', 'PL') for x in (
                          f"{desc['checkpoints'][d][arm]['median']:.0f} [{desc['checkpoints'][d][arm]['q25']:.0f}, "
                          f"{desc['checkpoints'][d][arm]['q75']:.0f}]", f"{100 * desc['checkpoints'][d][arm]['share_at_0']:.1f} %")]
                       for d in datasets])
    lines += ['', '### Neighbourhood purity of the last hidden ranking (outer test rows) and the permutations', '']
    lines += md_table(['dataset', *ARMS, 'kept label', 'expected', "Cramer's V"],
                      [[d] + [fmt(desc['purity'][d][f'test|{a}'], 4, False) for a in ARMS]
                       + [fmt(desc['permutation'][d][k], 3, False) for k in ('agreement', 'expected_agreement', 'cramers_v')]
                       for d in datasets])
    lines += ['', '### PL[final]', '']
    lines += md_table(['dataset', 'PL[final] (%)', 'PL[final] - FR', 'AF - PL[final]'],
                      [[d, fmt(desc['final'][d]['PL[final]'], 2, False), fmt(desc['final'][d]['PL[final] - FR']),
                        fmt(desc['final'][d]['AF - PL[final]'])] for d in datasets])
    lines += ['', '### Per-view split by encoder type (points)', '']
    split = [(key, kind) for key in ('D[match]', 'S[match]', 'D[own]', 'S[own]') for kind in ('views_lda', 'views_other')]
    lines += md_table(['dataset'] + [f"{key} {'LDA' if kind == 'views_lda' else 'other'}" for key, kind in split],
                      [[d] + [fmt(desc['views'][d][kind][key]) for key, kind in split] for d in datasets]
                      + [['mean'] + [fmt(desc['views_mean'][kind][key]) for key, kind in split]])
    lines += ['', '### Per-class recall (%)', '']
    for d in datasets:
        classes = list(desc['recall'][d]['AF'])
        lines += ['', f'{d}:', ''] + md_table(['arm', *classes], [[a] + [f"{100 * desc['recall'][d][a][c]:.1f}"
                                                                         for c in classes] for a in ARMS])
    lines += ['', f"Interpretation fixed in advance: {result['interpretation']}"]
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------- LaTeX

TEX_ORDER = ('iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits', 'balance_scale',
             'mfeat_zernike', 'ionosphere', 'vertebra_column', 'diabetes', 'banknote_authentication', 'qsar_biodeg',
             'steel_plates_fault', 'climate_model_simulation_crashes', 'hcv_egyptian_patients')
TEX_NAME = {'iris': 'Iris', 'wine': 'Wine', 'breast_cancer': 'Breast cancer', 'wine_quality': 'Wine quality',
            'vehicle': 'Vehicle', 'segment': 'Segment', 'digits': 'Digits',
            **{d: d.replace('_', '-') for d in TEX_ORDER[7:]}, 'hcv_egyptian_patients': 'HCV'}
WORDS = ('none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
         'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen')
INTERVAL_NOTE = ('95\\% corrected resampled-$t$ intervals (test-to-training ratio 0.25, 14 degrees of freedom); positive '
                 'favors the first-named model, and each contrast has its own Holm adjustment across the seventeen datasets')
ARM_DEFINITION = ('The permuted-label arm trains the same seven-view networks as ArrowFlow, at the same per-fold selections, '
                  'folds, fitting seeds, encoders and initial filters, with the class labels of the outer training '
                  'partition permuted once per outer fold and fitting seed and shared by the seven views. The gate, the '
                  'class-filter votes and so the motions read the permuted labels; the encoder, the nearest-neighbor readout '
                  'and the scoring keep the true labels.')


def signed(v, d=2):
    s = f'{abs(v):.{d}f}'
    if s == '0.' + '0' * d:
        return f'$-${s}' if v < -1e-9 else f'+{s}'
    return f'$-${s}' if v < 0 else f'+{s}'


def tex_interval(lo, hi, d=2):
    return f'[{signed(lo, d)}, {signed(hi, d)}]'


def tex_p(p):
    return '$<$0.001' if p < .001 else f'{p:.3f}'


def two_lines(first, second):
    guard = '{}' if str(second).startswith('[') else ''
    return f'\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{first}\\\\{guard}{second}\\end{{tabular}}'


def counted(n, total=17):
    return 'all seventeen datasets' if n == total else f'{WORDS[n]} of the seventeen datasets'


def counts_text(f):
    return (f"{counted(f['higher'])}, significantly on {WORDS[f['significantly_higher']]}, and significantly behind on "
            f"{WORDS[f['significantly_lower']]}")


def write_table(path, caption, label, header, rows, colspec, *, wide=True, subheads=None):
    lines = ['% Rendered by experiments/make_revision/compare_labels.py (analyse) from run files; do not edit by hand.',
             '\\begin{table}[!htbp]', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\centering', '\\scriptsize',
             '\\setlength{\\tabcolsep}{3pt}']
    if wide:
        lines.append('\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%')
    lines += [f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for i, row in enumerate(rows):
        if subheads and i in subheads:
            if i:
                lines.append('\\midrule')
            lines.append(f'\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{subheads[i]}}}}} \\\\')
        lines.append(' & '.join(str(c) for c in row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}'] + (['}'] if wide else []) + ['\\end{table}']
    path.write_text('\n'.join(lines) + '\n')


def write_tables(tables, result, datasets):
    tables.mkdir(parents=True, exist_ok=True)
    fam, summary, desc, rd = result['families'], result['across_datasets'], result['descriptive'], result['reading']
    order = [d for d in TEX_ORDER if d in datasets]
    cell = lambda key, d: next(r for r in fam[key]['rows'] if r['dataset'] == d)

    def contrast_rows(keys):
        return [[TEX_NAME[d]] + [c for k in keys for c in (signed(cell(k, d)['mean_difference']),
                                                          tex_interval(cell(k, d)['ci_low'], cell(k, d)['ci_high']),
                                                          tex_p(cell(k, d)['p_holm']))] for d in order]
    head = lambda tag: [two_lines(tag, 'Diff. (pp)'), two_lines(tag, '95\\% interval'), two_lines(tag, 'Holm $p$')]
    for key_pair, name, label, readout_text in (
            (('F1', 'F2'), 'tab_s_label_family', 'tab:s-label-family',
             'It is read at the update where ArrowFlow\'s checkpoint fell for the same fold, fitting seed and view, so that '
             'both train for the same number of updates (the primary readout).'),
            (('F3', 'F4'), 'tab_s_label_secondary', 'tab:s-label-secondary',
             'It is read at its own checkpoint, the network that training restores, chosen by the validation error under the '
             'permuted labels (the secondary readout).')):
        d_key, s_key = key_pair
        caption = (f"\\textbf{{Permuted-label control: {'the primary families' if d_key == 'F1' else 'its own checkpoint'}.}} "
                   'ArrowFlow minus the permuted-label arm ($D$) and the permuted-label arm minus the frozen arm ($S$), '
                   f'accuracy, in percentage points. {ARM_DEFINITION} {readout_text} Fitting seeds are averaged within each '
                   f'outer fold; intervals are {INTERVAL_NOTE}, as the design fixed before the arm was scored. ArrowFlow is '
                   f'ahead of the permuted-label arm on {counts_text(fam[d_key])}. The permuted-label arm is ahead of frozen '
                   f'filters on {counts_text(fam[s_key])}. HCV: hepatitis C virus.')
        write_table(tables/f'{name}.tex', caption, label, ['Dataset', *head('$D$'), *head('$S$')],
                    contrast_rows(key_pair), 'lrrrrrr')
    # across datasets
    names = {'G': 'ArrowFlow $-$ frozen ($G$)', 'F1': '$D$, matched stop', 'F2': '$S$, matched stop',
             'F3': '$D$, own checkpoint', 'F4': '$S$, own checkpoint'}
    blocks = {'all': 'All seventeen datasets', 'further': 'The ten further datasets (descriptive)',
              'benchmark': 'The seven benchmark datasets (descriptive)'}
    rows, subheads = [], {}
    for label, title in blocks.items():
        subheads[len(rows)] = title
        for key in ('G', 'F1', 'F2', 'F3', 'F4'):
            s = summary[key][label]
            rows.append([names[key], signed(s['mean']), tex_interval(s['ci_low'], s['ci_high']), tex_p(s['signed_rank']['p']),
                         s['higher'], s['lower'], s['significantly_higher'], s['significantly_lower']])
    primary, secondary = rd['primary'], rd['secondary']
    share = ''
    if primary['label_share_reported']:
        s = result['shares']['match']['label']
        share = (f" In this pattern the labels' share of the gain, $\\sum D/\\sum G$, is {share_text(s['value']).replace('%', '\\%')} "
                 f"({interval_text(s['ci_low'], s['ci_high']).replace('%', '\\%')}, 10,000 bootstrap resamples of the "
                 'datasets).')
    write_table(tables/'tab_s_label_summary.tex',
                '\\textbf{Permuted-label control: across datasets.} The mean over datasets of the per-dataset mean '
                'differences, in percentage points, with its 95\\% $t$ interval, the exact two-sided Wilcoxon signed-rank '
                '$p$ over the datasets, the numbers of datasets on which the difference is positive or negative, and the '
                'numbers on which it is significantly so after the Holm adjustment of the seventeen-dataset family. '
                f'Under the reading rule fixed before the arm was scored, the matched stop gives pattern ({primary["pattern"]}) '
                f'and the own checkpoint pattern ({secondary["pattern"]}).{share} The ten further datasets and the seven '
                'benchmark datasets are descriptive.',
                'tab:s-label-summary', ['Contrast', two_lines('Mean', '(pp)'), '95\\% interval', two_lines('Signed-rank', '$p$'),
                                        'Higher', 'Lower', two_lines('Holm:', 'higher'), two_lines('Holm:', 'lower')],
                rows, 'lrrrrrrr', subheads=subheads)
    # each readout against frozen filters (no interval), in the style of the scramble table
    means = result['means']
    rows = [[TEX_NAME[d]] + [signed(means[f'{d}|{arm}'] - means[f'{d}|FR']) for arm in ('PL[match]', 'PL[own]', 'PL[final]')]
            + [signed(means[f'{d}|AF'] - means[f'{d}|FR'])] for d in order]
    write_table(tables/'tab_s_label_readouts.tex',
                '\\textbf{Permuted-label control: each readout against frozen filters.} The mean accuracy of the '
                'permuted-label arm, read at the matched stop, at its own checkpoint and after the last update, minus that '
                'of the frozen arm, in percentage points, beside ArrowFlow minus the frozen arm. Negative means that '
                'training with permuted labels left the network less accurate than frozen filters. The values are '
                'differences of mean accuracies, so they carry no interval; the first two are tested in '
                'Tables~\\ref{tab:s-label-family} and~\\ref{tab:s-label-secondary}, the third is descriptive. HCV: '
                'hepatitis C virus.', 'tab:s-label-readouts',
                ['Dataset', two_lines('Matched stop', '$-$ frozen (pp)'), two_lines('Own checkpoint', '$-$ frozen (pp)'),
                 two_lines('Update 200', '$-$ frozen (pp)'), two_lines('ArrowFlow', '$-$ frozen (pp)')], rows, 'lrrrr', wide=False)
    # vote volume and displacement
    pct = lambda v: '--' if v is None else f'{100 * v:.1f}'
    rows = [[TEX_NAME[d]] + [pct(desc['votes'][d]['last'][k]['share']) for k in ('AF[run]', 'PL[run]', 'AF[to match]',
                                                                                   'PL[to match]')]
            + [f"{desc['displacement'][d]['displacement_last'][k]:.3f}" for k in ('AF[checkpoint]', 'PL[match]', 'PL[own]',
                                                                                  'PL[final]')] for d in order]
    write_table(tables/'tab_s_label_mechanism.tex',
                '\\textbf{Permuted-label control: votes and movement.} Left: the share of the selected slots of the last '
                'hidden layer that carry a nonzero vote, in percent, over the whole run of 200 updates and up to the '
                'matched stop, summed over folds, fitting seeds and views, as in Table~\\ref{tab:s-motion-statistics}. '
                'Right: the mean normalized footrule of the last hidden layer\'s filters from their initial orders, at '
                'ArrowFlow\'s checkpoint and at the three readouts of the permuted-label arm. Descriptive: no test. HCV: '
                'hepatitis C virus.', 'tab:s-label-mechanism',
                ['Dataset', two_lines('ArrowFlow:', 'nonzero, run'), two_lines('Permuted:', 'nonzero, run'),
                 two_lines('ArrowFlow:', 'to stop'), two_lines('Permuted:', 'to stop'), two_lines('ArrowFlow:', 'moved'),
                 two_lines('Matched:', 'moved'), two_lines('Own ckpt.:', 'moved'), two_lines('Update 200:', 'moved')],
                rows, 'lrrrrrrrr')
    # checkpoints
    ck = lambda d, arm: (f"{desc['checkpoints'][d][arm]['median']:.0f} [{desc['checkpoints'][d][arm]['q25']:.0f}, "
                         f"{desc['checkpoints'][d][arm]['q75']:.0f}]")
    rows = [[TEX_NAME[d], ck(d, 'AF'), pct(desc['checkpoints'][d]['AF']['share_at_0']), ck(d, 'PL'),
             pct(desc['checkpoints'][d]['PL']['share_at_0'])] for d in order]
    write_table(tables/'tab_s_label_checkpoints.tex',
                '\\textbf{Permuted-label control: where the checkpoints fall.} The update of ArrowFlow\'s checkpoint, which '
                'is the matched stop, and of the permuted-label arm\'s own checkpoint: median and interquartile range over '
                'the outer folds, fitting seeds and views, and the share of views whose checkpoint is the initial state '
                '(update 0), in percent. HCV: hepatitis C virus.', 'tab:s-label-checkpoints',
                ['Dataset', two_lines('ArrowFlow:', 'median [IQR]'), two_lines('ArrowFlow:', 'at 0 (\\%)'),
                 two_lines('Permuted:', 'median [IQR]'), two_lines('Permuted:', 'at 0 (\\%)')], rows, 'lrrrr', wide=False)
    # purity and the permutations
    rows = [[TEX_NAME[d]] + [pct(desc['purity'][d][f'test|{a}']) for a in ARMS]
            + [pct(desc['permutation'][d]['agreement']), pct(desc['permutation'][d]['expected_agreement']),
               f"{desc['permutation'][d]['cramers_v']:.3f}"] for d in order]
    write_table(tables/'tab_s_label_purity.tex',
                '\\textbf{Permuted-label control: neighborhood purity and the permutations.} Same-class purity of the last '
                'hidden ranking on the outer test rows, in percent, as in Table~\\ref{tab:s-motion-purity}: ArrowFlow and '
                'the frozen arm as registered, the permuted-label arm at its three readouts. Right: the share of training '
                'rows whose permuted label is their true one, its expectation (the sum of the squared class shares), and '
                'Cram\\\'er\'s $V$ between the permuted and the true labels, averaged over the 45 permutations of each '
                'dataset. Descriptive: no test. HCV: hepatitis C virus.', 'tab:s-label-purity',
                ['Dataset', 'ArrowFlow', 'Frozen', two_lines('Permuted:', 'matched'), two_lines('Permuted:', 'own'),
                 two_lines('Permuted:', 'update 200'), two_lines('Label', 'kept (\\%)'), two_lines('Expected', '(\\%)'),
                 "Cram\\'er's $V$"], rows, 'lrrrrrrrr')
    # the per-view split by encoder type
    rows = [[TEX_NAME[d]] + [signed(desc['views'][d][kind][key]) for key in ('D[match]', 'S[match]')
                             for kind in ('views_lda', 'views_other')] for d in order]
    rows.append(['Mean over the seventeen'] + [signed(desc['views_mean'][kind][key]) for key in ('D[match]', 'S[match]')
                                               for kind in ('views_lda', 'views_other')])
    write_table(tables/'tab_s_label_views.tex',
                '\\textbf{Permuted-label control: views with and without linear-discriminant coordinates.} $D$ and $S$ at '
                'the matched stop, in percentage points, computed from the mean per-view nearest-neighbor accuracy over '
                'the three views whose encoder holds linear-discriminant coordinates (views 1, 4 and 7) and over the four '
                'others, whose encoders do not read the labels. The frozen arm\'s per-view predictions are those of the '
                'readout of the initial filters, which reproduces its registered seven-view predictions. Descriptive: no '
                'test. HCV: hepatitis C virus.', 'tab:s-label-views',
                ['Dataset', two_lines('$D$,', 'LDA views'), two_lines('$D$,', 'other views'), two_lines('$S$,', 'LDA views'),
                 two_lines('$S$,', 'other views')], rows, 'lrrrr', wide=False, subheads={len(order): 'All datasets'})
    # per-class recall on balance-scale
    if 'balance_scale' in datasets:
        recall = desc['recall']['balance_scale']
        classes = list(recall['AF'])
        arm_names = {'AF': 'ArrowFlow', 'FR': 'Frozen', 'PL[match]': 'Permuted, matched stop', 'PL[own]': 'Permuted, own checkpoint',
                     'PL[final]': 'Permuted, update 200'}
        rows = [[arm_names[a]] + [f'{100 * recall[a][c]:.1f}' for c in classes] for a in ARMS]
        write_table(tables/'tab_s_label_recall.tex',
                    '\\textbf{Permuted-label control: per-class recall on balance-scale.} Recall of each class, in percent, '
                    'averaged over the fitting seeds and then the outer folds; B is the level-balance class. Descriptive: no '
                    'test.', 'tab:s-label-recall', ['Arm', *[f'Class {c}' for c in classes]], rows, 'l' + 'r' * len(classes),
                    wide=False)
