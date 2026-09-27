#!/usr/bin/env python3
"""Render every table of the ArrowFlow v3 manuscript and supplement from run files.

Nothing numeric is typed by hand: every cell, and every count or setting quoted in
a caption, is read from a verified harness summary, a run manifest or a frozen
protocol.  The sections only ``\\input`` the files written to ``tables/``.  At
render time the script parses every main-text table and its complete supplement
table(s) back into cells keyed by (row, column) and asserts that each main-text
cell equals the supplement cell of the same row and column, sign and format
included; a mutation test on in-memory copies (a flipped sign, two swapped
cells) proves on every run that the check detects drift.

Sources (defaults resolve the v3 run directories beside the repository):

    --runs DIR          holds 2026-09-12-{bridge,bridge-knn,knn-vs-full,matched,ablation,contrasts,devlab},
                        2026-09-13-knn-{training,ablation,projected}, 2026-09-13-referee-analyses (comparator
                        intervals, ranks and duplicate rows), 2026-09-14-newdata-{batch1,batch2,analysis,ablation},
                        the outputs of holistic benchmark, training and components in 2026-09-14-holistic, and the
                        production logs of the later families, whose scripts are tracked copies in docs/superpowers
    --out DIR           where the .tex tables are written (default tables/)

Also rendered: the learning-controls figure through ``render_figures.figure7``
into ``figures/data/fig7_learning_controls.pdf``, and its wrapper
``figures/fig7_learning_controls.tex``, whose caption counts are computed from
the contrasts file.

    python render_tables.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, OrderedDict, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUNS = REPO.parent / '.superpowers' / 'sdd' / '2026-09-12-arrowflow-story-restoration-plan' / 'runs'
PROTOCOLS_V3 = REPO / 'experiments' / 'make_revision' / 'protocols' / '2026-09-12'
PROTOCOLS_G5 = REPO / 'experiments' / 'make_revision' / 'protocols' / '2026-09-14'
# tracked copies of the selection document and the production scripts; the run logs stay in the runs directory
SELECTION_DOCUMENT = REPO / 'docs' / 'superpowers' / 'plans' / '2026-09-13-task-23-dataset-selection.md'
PRODUCTION_SCRIPTS = REPO / 'docs' / 'superpowers' / 'production-scripts'
TABLES = HERE / 'tables'
FIG_DATA = HERE / 'figures' / 'data'
HASH_DIGITS = 12

DATASETS = ['iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits']
LABEL = {'iris': 'Iris', 'wine': 'Wine', 'breast_cancer': 'Breast cancer', 'wine_quality': 'Wine quality',
         'vehicle': 'Vehicle', 'segment': 'Segment', 'digits': 'Digits'}

BRIDGE_MODELS = ['arrowflow_full', 'svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting', 'dummy']
COMPARATORS = BRIDGE_MODELS[1:-1]
KNN = 'arrowflow_full_knn'   # ArrowFlow (nearest-neighbor readout), fitted in its own run on the benchmark's outer partitions
ARROWFLOW_MODELS = [KNN, 'arrowflow_full']   # ArrowFlow, then its prototype-readout ablation
MAIN_MODELS = ARROWFLOW_MODELS + BRIDGE_MODELS[1:]
MODEL = {'arrowflow_full': 'Prototype readout$^\\dagger$', KNN: 'ArrowFlow', 'svc_rbf': 'SVC (RBF)',
         'random_forest': 'Random forest', 'mlp': 'MLP', 'numeric_knn': 'Numeric kNN',
         'gradient_boosting': 'Gradient boosting', 'dummy': 'Majority class'}

UNTRAINED, INPUT_KNN = 'arrowflow_knn_untrained', 'input_footrule_knn'   # the two training controls
CONTROLS = [UNTRAINED, INPUT_KNN]
CONTROL = {UNTRAINED: 'Untrained ArrowFlow', INPUT_KNN: 'Tuned input footrule kNN'}

ABLATION_ORDER = ['views1', 'views3', 'views7', 'borda_views7', 'no_checkpoint', 'no_augment', 'multiview_footrule_knn']
ABLATION_S4 = ['single_view_no_checkpoint_no_augment'] + ABLATION_ORDER
VARIANT = {'views1': 'One view', 'views3': 'Three views', 'views7': 'Seven views, majority vote (reference)',
           'borda_views7': 'Seven views, Borda count', 'no_checkpoint': 'Without validation checkpoint',
           'no_augment': 'Without augmentation',
           'single_view_no_checkpoint_no_augment': 'One view, no checkpoint, no augmentation',
           'multiview_footrule_knn': 'Seven-view footrule kNN control'}

WORDS = {0: 'no', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
         10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
         17: 'seventeen'}
FACTS = {}


# ----------------------------------------------------------------------------- formatting
def words(n):
    return WORDS.get(int(n), f'{int(n):,}')


def num_word(n):
    """A count after 'on' or 'in': 'none' rather than 'no' for zero."""
    return 'none' if int(n) == 0 else words(n)


def of_total(k, total, noun, verb_one, verb_many, capital=True):
    """'None of the six variants meets', 'One of the 14 contrasts remains', 'Two of the three variants meet'."""
    head = 'none' if int(k) == 0 else words(k)
    text = f'{head} of the {total} {noun} {verb_one if int(k) <= 1 else verb_many}'
    return text[0].upper() + text[1:] if capital else text


def listing(values, conjunction='or'):
    values = [f'{v:g}' if isinstance(v, float) else str(v) for v in values]
    return values[0] if len(values) == 1 else ', '.join(values[:-1]) + f' {conjunction} ' + values[-1]


def pct(x, d=1):
    return f'{100 * float(x):.{d}f}'


def pm(mean, sd, d=1):
    return f'{pct(mean, d)} ({pct(sd, d)})' if sd is not None and not pd.isna(sd) else pct(mean, d)


def signed(x, d=1):
    """Signed percentage points with a TeX minus."""
    v = 100 * float(x)
    s = f'{abs(v):.{d}f}'
    if s == '0.' + '0' * d:
        # keep the sign of a small negative value so that an interval bound below zero never reads as +0.0
        return f'$-${s}' if v < -1e-9 else f'+{s}'
    return f'$-${s}' if v < 0 else f'+{s}'


def interval(lo, hi, d=1):
    return f'[{signed(lo, d)}, {signed(hi, d)}]'


def pval(p):
    p = float(p)
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def tex(s):
    return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def widths_label(w):
    w = ast.literal_eval(w) if isinstance(w, str) else list(w)
    return '$[' + ','.join(str(int(v)) for v in w) + ']$'


def read_json(path):
    return json.loads(Path(path).read_text())


def lookup(rows, model, metric):
    for r in rows:
        if r.get('model_id') == model and r.get('metric') == metric:
            return r
    raise KeyError(f'{model}/{metric} not in summary rows')


def interval_note(subset=False):
    """The interval rule a caption states. A contrast over all outer folds uses the protocol's degrees of freedom; a row over
    a subset of n outer folds uses n - 1 degrees of freedom and the variance factor 1/n + ratio (referee panel of 2026-09-23,
    item G: interval_note once stated 14 degrees of freedom for such rows too). Every table that states the subset rule has
    its rows checked against it by check_subset_intervals, and no caption of such a table may state the full-fold count."""
    if not subset:
        return (f"{FACTS['confidence']}\\% corrected resampled-$t$ interval (test-to-training ratio {FACTS['ratio']:g}, "
                f"{FACTS['df']} degrees of freedom)")
    return (f"{FACTS['confidence']}\\% corrected resampled-$t$ intervals over the $n$ folds of a row, with the variance of the "
            f"fold differences multiplied by $1/n+{FACTS['ratio']:g}$ and $n-1$ degrees of freedom")


SUBSET_DF = 'and $n-1$ degrees of freedom'


def check_subset_intervals(frame, where, p_column=None):
    """Rows over a subset of n outer folds (n recorded per row): df must be n - 1, the interval must be the mean plus or
    minus the t quantile at n - 1 degrees of freedom times the recorded standard error, and a recorded p value must be the
    two-sided t tail at n - 1 degrees of freedom. A row with fewer than two folds carries nothing, and a row whose standard
    error is zero (all fold differences equal) carries no interval and no test. Returns the rows of the second kind."""
    from scipy import stats
    degenerate = []
    for _, r in frame.iterrows():
        n = int(r['n_folds'])
        if n < 2 or blank(r['mean_difference']):
            continue
        if int(r['df']) != n - 1:
            raise AssertionError(f'{where}: a row over {n} folds has {int(r["df"])} degrees of freedom, not n - 1')
        se, mean = float(r['standard_error']), float(r['mean_difference'])
        if se == 0.0:
            degenerate.append(r)
            continue
        q = stats.t.ppf(0.5 + FACTS['confidence'] / 200, n - 1)
        if abs(float(r['ci_low']) - (mean - q * se)) > 1e-9 or abs(float(r['ci_high']) - (mean + q * se)) > 1e-9:
            raise AssertionError(f'{where}: an interval over {n} folds is not the t interval at n - 1 degrees of freedom')
        if p_column and abs(float(r[p_column]) - 2 * stats.t.sf(abs(mean) / se, n - 1)) > 1e-9:
            raise AssertionError(f'{where}: a p value over {n} folds is not the t tail at n - 1 degrees of freedom')
    return degenerate


def signed_rank_less(differences):
    """Exact one-sided Wilcoxon signed-rank test that the differences lie below zero. W+ is the sum of the ranks of the
    absolute differences over the positive ones, and p is the share of the 2^n sign assignments with W+ at most the
    observed value. The exact null needs no zero and no tied absolute difference; SciPy's exact test must agree."""
    from scipy import stats
    d = [float(x) for x in differences]
    a = [abs(x) for x in d]
    if any(x == 0.0 for x in a) or len(set(a)) != len(a):
        raise AssertionError('signed-rank test: a zero or tied absolute difference, for which the exact null is not this one')
    rank = {i: r + 1 for r, i in enumerate(sorted(range(len(d)), key=lambda i: a[i]))}
    w = sum(rank[i] for i in range(len(d)) if d[i] > 0)
    n, ways = len(d), [1] + [0] * (len(d) * (len(d) + 1) // 2)
    for k in range(1, n + 1):
        for s in range(len(ways) - 1, k - 1, -1):
            ways[s] += ways[s - k]
    p = sum(ways[:w + 1]) / 2 ** n
    reference = stats.wilcoxon(d, alternative='less', method='exact')
    if float(reference.statistic) != w or abs(float(reference.pvalue) - p) > 1e-12:
        raise AssertionError('signed-rank test: the exact count differs from SciPy')
    return dict(w=w, p=p, n=n)


def matched_encoder_sentence(matched):
    """Degree and encoded vocabulary size of the matched study's encoder on every tabular dataset, from the frozen protocol
    as the run recorded it; the study code must encode each dataset with exactly its protocol entry."""
    p = matched.protocol
    frozen = read_json(REPO / 'experiments' / 'make_revision' / 'protocols' / '2026-09-12' / 'matched_v3.json')
    table = p.get('encoder_by_dataset') or {}
    if p != frozen or sorted(table) != sorted(DATASETS) or any(set(v) != {'degree', 'embed_dim'} for v in table.values()):
        raise AssertionError('matched protocol: the run protocol differs from matched_v3.json, or its encoder table does not '
                             'cover exactly the tabular datasets')
    src = (REPO / 'experiments' / 'make_revision' / 'matched.py').read_text()
    if "if table and dataset_id in table:\n        return dict(table[dataset_id])" not in src or \
            'settings=encoder_settings(p,dataset_id)' not in src or \
            "OrdinalEncoder(strategy='random',degree=settings['degree'],embed_dim=settings['embed_dim']," not in src:
        raise AssertionError('matched.py: the study no longer encodes each dataset with the degree and size of its protocol entry')
    groups = OrderedDict()
    for d in DATASETS:
        groups.setdefault((int(table[d]['degree']), int(table[d]['embed_dim'])), []).append(d)
    common = max(groups, key=lambda k: len(groups[k]))
    if len(groups[common]) < 2 or sum(len(v) == len(groups[common]) for v in groups.values()) != 1:
        raise AssertionError('matched protocol: no single most common encoder setting for "the other datasets"')
    parts = [f'$p={deg}$ with $e={e}$ on {listing([LABEL[d] for d in ds], "and")}' for (deg, e), ds in groups.items()
             if (deg, e) != common]
    parts.append(f'$p={common[0]}$ with $e={common[1]}$ on the other {words(len(groups[common]))} datasets')
    return f'Its encoder has degree {listing(parts, "and")}.'


# ----------------------------------------------------------------------------- table writer
WRITTEN = OrderedDict()


COLUMN = re.compile(r'[lrc]|p\{[^{}]*\}')


def two_lines(first, second):
    """A header cell set on two right-aligned lines, so that its column need not be shrunk.

    A second line that opens with '[' is protected with an empty group, since LaTeX would otherwise read the bracket
    as the optional vertical-space argument of the line break."""
    guard = '{}' if str(second).startswith('[') else ''
    return f'\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{first}\\\\{guard}{second}\\end{{tabular}}'


def write_table(name, caption, label, header, rows, colspec, *, size='\\footnotesize', placement='!htbp',
                block_rules=(), out_dir=None, wide=False, subheads=None, row_comments=None, preamble=(), stack=None,
                rotate=False, stretch=None):
    """Write a booktabs table.

    ``block_rules`` are row indices before which a \\midrule is drawn; ``subheads`` maps a row index to an
    italic sub-header row drawn before it (after a \\midrule unless it opens the body), or holds one such map per
    tabular of a stacked table, all with the same row indices; ``row_comments`` maps
    a row index to a TeX comment line written after that row; ``preamble`` holds comment lines written
    above the table environment. ``wide`` shrinks a tabular that is wider than the line to the line width and
    never enlarges one. ``stack`` sets the columns after the first ``stack`` data columns as a second tabular
    below the first, both led by the first column, so that neither needs shrinking. ``rotate`` sets the float
    on a page of its own, turned by 90 degrees, with a line as long as the text height. ``stretch`` scales the row
    height (\\arraystretch) of a long table whose float would otherwise exceed the text height.
    """
    row_comments = row_comments or {}
    parts, columns = [None], None
    if stack is not None:
        columns = COLUMN.findall(colspec)
        if ''.join(columns) != colspec or len(columns) != len(header) or any(len(row) != len(header) for row in rows):
            raise AssertionError(f'{name}: a stacked table needs one column letter and one cell per header column')
        parts = [[0, *range(1, stack + 1)], [0, *range(stack + 1, len(header))]]
    heads = list(subheads) if isinstance(subheads, (list, tuple)) else [subheads or {}] * len(parts)
    if len(heads) != len(parts) or any(sorted(h) != sorted(heads[0]) for h in heads):
        raise AssertionError(f'{name}: a stacked table needs one sub-header map per tabular, all with the same row indices')
    lines = ['% Rendered by render_tables.py from run files; do not edit by hand.', *preamble,
             f'\\begin{{table}}[{"p" if rotate else placement}]']
    if rotate:
        lines += ['\\centering', '\\rotatebox{90}{\\begin{minipage}{\\textheight}']
    lines += [f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\centering', size, '\\setlength{\\tabcolsep}{3pt}']
    if stretch is not None:
        lines.append(f'\\renewcommand{{\\arraystretch}}{{{stretch:g}}}')
    for k, part in enumerate(parts):
        pick = (lambda cells: list(cells)) if part is None else (lambda cells, part=part: [cells[j] for j in part])
        if k:
            lines.append('\\par\\medskip')
        if wide:
            lines.append('\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%')
        spec = colspec if part is None else ''.join(pick(columns))
        lines += [f'\\begin{{tabular}}{{{spec}}}', '\\toprule', ' & '.join(pick(header)) + ' \\\\', '\\midrule']
        for i, row in enumerate(rows):
            if (i in block_rules or i in heads[k]) and i > 0:
                lines.append('\\midrule')
            if i in heads[k]:
                lines.append(f'\\multicolumn{{{len(pick(header))}}}{{l}}{{\\emph{{{heads[k][i]}}}}} \\\\')
            lines.append(' & '.join(str(c) for c in pick(row)) + ' \\\\')
            if i in row_comments:
                lines.append(row_comments[i])
        lines += ['\\bottomrule', '\\end{tabular}']
        if wide:
            lines.append('}')
    if rotate:
        lines.append('\\end{minipage}}')
    lines.append('\\end{table}')
    text = '\n'.join(lines) + '\n'
    path = (out_dir or TABLES) / f'{name}.tex'
    path.write_text(text)
    WRITTEN[name] = text
    print(f'wrote {path}  ({len(rows)} rows)')
    return text


# ----------------------------------------------------------------------------- main-to-supplement check
# Every main-text table is parsed back into cells keyed by (row, column) and compared, as formatted and
# signed strings, with the cells of the same rows and columns in its complete supplement table(s).
CELL_SEPARATOR = re.compile(r'(?<!\\)&')


def split_cells(line):
    """Cells of one tabular row without its row terminator; an escaped \\& stays inside its cell."""
    return [c.strip() for c in CELL_SEPARATOR.split(line.strip().removesuffix('\\\\'))]


def table_rows(text):
    """Header cells and body rows of a rendered table.

    A body row is a list of stripped cells; a \\midrule between body rows becomes ('BLOCK', None) and an
    italic sub-header row ('BLOCK', its text).  Comment lines are skipped; empty cells stay ''. A table set as
    stacked tabulars (write_table's ``stack``) is read as one table: every further tabular must repeat the first
    column and the positions of the block rules and sub-headers of the first, whose sub-header texts are kept, and its
    other columns are appended in order.
    """
    parts = [tabular_rows(chunk.split('\\bottomrule', 1)[0]) for chunk in text.split('\\toprule')[1:]]
    header, rows = parts[0]
    for more_header, more_rows in parts[1:]:
        if more_header[0] != header[0] or len(more_rows) != len(rows) or any(
                (isinstance(a, tuple) != isinstance(b, tuple) or (a[1] is None) != (b[1] is None))
                if isinstance(a, tuple) or isinstance(b, tuple) else a[0] != b[0]
                for a, b in zip(rows, more_rows)):
            raise AssertionError('stacked tabulars differ in their first column or their block structure')
        header = header + more_header[1:]
        rows = [a if isinstance(a, tuple) else a + b[1:] for a, b in zip(rows, more_rows)]
    return header, rows


def tabular_rows(body):
    """Header cells and body rows of one tabular, given its text between \\toprule and \\bottomrule."""
    head, rest = body.split('\\midrule', 1)
    header = split_cells(head)
    rows = []
    for line in rest.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line == '\\midrule':
            rows.append(('BLOCK', None))
        elif line.startswith('\\multicolumn') and not CELL_SEPARATOR.search(line):
            m = re.search(r'\\emph\{(.*)\}\}', line)
            rows.append(('BLOCK', m.group(1) if m else line))
        elif line.endswith('\\\\'):
            rows.append(split_cells(line))
        else:
            raise AssertionError(f'unparsed table line: {line}')
    return header, rows


def labelled(rows):
    """(sub-header, row) for every body row; an empty first cell is filled from the row above."""
    out, sub, last = [], None, ''
    for r in rows:
        if isinstance(r, tuple):
            sub = r[1] if r[1] is not None else sub
            continue
        r = list(r)
        r[0] = r[0] or last
        last = r[0]
        out.append((sub, r))
    return out


def unbold(cell):
    m = re.fullmatch(r'\\textbf\{(.*)\}', cell)
    return m.group(1) if m else cell


def lead(cell):
    """The value of a cell that appends an interval or a seed SD in brackets: '+0.1 [...]' -> '+0.1'."""
    return cell.split(' [', 1)[0]


def cells_contrasts(t):
    header, rows = table_rows(t['tab_contrasts'])
    main = {(r[0], sub, h): c for sub, r in labelled(rows) for h, c in zip(header[1:], r[1:])}
    hs, rs = table_rows(t['tab_s_contrasts_complete'])
    column = {'Difference (pp)': 'Diff. (pp)'}
    supp = {(r[1], r[2], h): r[hs.index(column.get(h, h))] for _, r in labelled(rs) for h in header[1:]}
    return main, supp


def cells_knn(t):
    header, rows = table_rows(t['tab_knn'])
    main = {(r[0], h): c for _, r in labelled(rows) for h, c in zip(header[1:], r[1:])}
    hs, rs = table_rows(t['tab_s_knn_complete'])
    column = {'Difference (pp)': 'Diff. (pp)'}
    supp = {(r[0], h): r[hs.index(column.get(h, h))] for _, r in labelled(rs) for h in header[1:]}
    return main, supp


def cells_components(t):
    header, rows = table_rows(t['tab_components'])
    main = {(sub, r[0], h): c for sub, r in labelled(rows) for h, c in zip(header[1:], r[1:])}
    blocks = list(dict.fromkeys(sub for sub, _ in labelled(rows)))   # error block, change block
    supp = {}
    for sub, name in zip(blocks, ('tab_s_ablation_complete', 'tab_s_ablation_changes')):
        hs, rs = table_rows(t[name])
        supp.update({(sub, r[0], h): lead(c) for _, r in labelled(rs) for h, c in zip(hs[1:], r[1:])})
    return main, supp


def compare_cells(name, supplements, main, supp):
    missing = [k for k in main if k not in supp]
    differ = [(k, main[k], supp[k]) for k in main if k in supp and main[k] != supp[k]]
    if not main or missing or differ:
        raise AssertionError(f'{name} differs from {" + ".join(supplements)}: {len(missing)} cells without a '
                             f'supplement counterpart {missing[:3]}; {len(differ)} differing cells {differ[:3]}')
    return len(main)


def drift_checks():
    """(summary table, its complete supplement tables, cell function) for every table that has a complete record."""
    checks = [('tab_contrasts', ['tab_s_contrasts_complete'], cells_contrasts),
              ('tab_knn', ['tab_s_knn_complete'], cells_knn),
              ('tab_components', ['tab_s_ablation_complete', 'tab_s_ablation_changes'], cells_components)]
    if 'tab_main_benchmark' in WRITTEN:
        checks.append(('tab_main_benchmark', ['tab_s_errors'], cells_main_benchmark))
    if 'tab_training' in WRITTEN:
        checks.append(('tab_training', ['tab_s_ladder', 'tab_s_training_complete'], cells_training))
    if 'tab_ablation' in WRITTEN:
        checks.append(('tab_ablation', ['tab_s_components_changes'], cells_ablation))
    if 'tab_motion' in WRITTEN:
        checks.append(('tab_motion', ['tab_s_motion_family'], cells_motion))
    return checks


def run_drift_checks(texts, checks, quiet=False):
    for name, supplements, cells in checks:
        n = compare_cells(name, supplements, *cells(texts))
        if not quiet:
            print(f'check {name} = {" + ".join(supplements)}: {n} keyed cells equal')


def mutate_cell(text, row_start, change, block=None, occurrence=1):
    """Copy of a rendered table with the ``occurrence``-th row starting with ``row_start`` (under sub-header ``block``)
    rewritten by ``change(cells, header)``; a stacked table repeats every row once per tabular."""
    header, _ = table_rows(text)
    lines, current, seen = text.splitlines(), None, 0
    for i, line in enumerate(lines):
        if line.startswith('\\multicolumn'):
            m = re.search(r'\\emph\{(.*)\}\}', line)
            current = m.group(1) if m else line
        elif line.startswith(row_start + ' & ') and (block is None or current == block):
            seen += 1
            if seen == occurrence:
                lines[i] = ' & '.join(change(split_cells(line), header)) + ' \\\\'
                break
    else:
        raise AssertionError(f'mutation test: no row starting with {row_start!r} under {block!r}')
    mutated = '\n'.join(lines) + '\n'
    if mutated == text:
        raise AssertionError('mutation test: the mutation changed nothing')
    return mutated


def flip_difference(cells, header):
    j = header.index('Difference (pp)')
    cells[j] = '+' + cells[j][3:] if cells[j].startswith('$-$') else '$-$' + cells[j].lstrip('+')
    return cells


def mutation_self_test(checks):
    """Apply the two drifts a reviewer used against the old presence-only check to in-memory copies of the
    rendered tables; the keyed check must fail on both."""
    by_name = {c[0]: c for c in checks}
    cases = [('tab_contrasts', f'{LABEL["wine_quality"]} prototype readout minus seven-view footrule kNN control, sign flipped',
              lambda text: mutate_cell(text, LABEL['wine_quality'], flip_difference, CONTRAST_LABEL['output_vs_knn'])),
             ('tab_knn', f'{LABEL["wine_quality"]} ArrowFlow minus prototype readout, sign flipped',
              lambda text: mutate_cell(text, LABEL['wine_quality'], flip_difference)),
             ('tab_main_benchmark', f'{LABEL["iris"]} ArrowFlow and {MODEL["svc_rbf"]} cells swapped',
              lambda text: mutate_cell(text, LABEL['iris'], swap_columns(MODEL[KNN], MODEL['svc_rbf']))),
             ('tab_training', f'{LABEL["vehicle"]} ArrowFlow minus the untrained ArrowFlow, sign flipped',
              lambda text: mutate_cell(text, LABEL['vehicle'], flip_column(1), occurrence=2)),
             ('tab_ablation', f'{LABEL["vehicle"]} ArrowFlow minus one view, sign flipped',
              lambda text: mutate_cell(text, LABEL['vehicle'], flip_column(1))),
             ('tab_motion', 'balance-scale ArrowFlow minus the frozen arm, sign flipped',
              lambda text: mutate_cell(text, NEW_LABEL['balance_scale'],
                                       flip_motion(two_lines(ARM_SHORT['frozen'], 'Diff. (pp)'))))]
    for name, description, mutate in cases:
        texts = dict(WRITTEN)
        texts[name] = mutate(texts[name])
        try:
            run_drift_checks(texts, [by_name[name]], quiet=True)
        except AssertionError as exc:
            print(f'mutation test ({description}): detected; {exc}')
        else:
            raise SystemExit(f'mutation test ({description}): the drift check did not fail')


# ----------------------------------------------------------------------------- loaders
class Family:
    def __init__(self, name, directory, summary_file='summary.json'):
        self.name, self.dir = name, Path(directory)
        self.summary = read_json(self.dir / summary_file) if (self.dir / summary_file).is_file() else None
        self.protocol = read_json(self.dir / 'protocol.json') if (self.dir / 'protocol.json').is_file() else {}
        self.environment = read_json(self.dir / 'environment.json') if (self.dir / 'environment.json').is_file() else {}
        planned = self.dir / 'planned_jobs.json'
        self.planned = len(read_json(planned)) if planned.is_file() else None
        results = self.dir / 'results'
        self.completed = len([f for f in results.glob('*.json') if '__r' in f.name and 'fit' not in f.name]) if results.is_dir() else 0


def load_facts(bridge, matched):
    b = bridge.protocol
    FACTS.update(outer_folds=b['outer_folds'], outer_repeats=b['outer_repeats'],
                 n_outer=b['outer_folds'] * b['outer_repeats'], inner_folds=b['inner_folds'],
                 budget=b['candidate_budget'], views=b['full_method']['n_views'], n_seeds=len(b['fit_seeds']),
                 ratio=b['test_train_ratio'], df=b['outer_folds'] * b['outer_repeats'] - 1,
                 confidence=round(100 * b['confidence']), family=b['primary_family_size'],
                 grid_size=math.prod(len(v) for v in b['full_method']['candidate_grid'].values()),
                 matched=matched.protocol, bridge=b, max_workers=protocol_workers(b))


def protocol_workers(protocol):
    """Worker processes of the benchmark run, as its protocol's resource decision records them."""
    m = re.search(r'(\d+) workers', str(protocol.get('resource_decision', '')))
    if not m:
        raise AssertionError('benchmark protocol: no worker count in its resource decision')
    return int(m.group(1))


# ----------------------------------------------------------------------------- main benchmark (bridge)
def main_table_rows(main_table, bridge, knn):
    """Error rows of main_table.json (compare_runs knn), checked field by field against the run summaries that the
    complete table reads: the first benchmark run for the prototype readout, the ArrowFlow run for ArrowFlow and the comparators."""
    if main_table.get('metric') != 'error' or main_table.get('comparators_reproduced') is not True:
        raise AssertionError('main_table.json: expected error rows with comparators_reproduced true')
    rows = {}
    for d in DATASETS:
        by_model = {r['model_id']: r for r in main_table['rows'][d]}
        if sorted(by_model) != sorted(MAIN_MODELS):
            raise AssertionError(f'main_table.json/{d}: models {sorted(by_model)}, expected {sorted(MAIN_MODELS)}')
        for m, r in by_model.items():
            run = 'bridge' if m == 'arrowflow_full' else 'knn'
            if r.get('source_run') != run:
                raise AssertionError(f'main_table.json/{d}/{m}: source run {r.get("source_run")}, expected {run}')
            s = lookup((knn if m == KNN else bridge).summary['summaries'][d], m, 'error')
            for field, key in (('mean_error', 'mean'), ('outer_fold_sd', 'outer_fold_sd'),
                               ('mean_within_fold_seed_sd', 'mean_within_fold_seed_sd'), ('n_folds', 'n_folds'),
                               ('seeds_per_fold', 'seeds_per_fold')):
                a, b = r.get(field), s.get(key)
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                    raise AssertionError(f'main_table.json/{d}/{m}: {field} {a} differs from the summary value {b}')
        rows[d] = by_model
    return rows


def render_bridge(bridge, knn, main_table):
    """Main benchmark table from main_table.json, one row per model with the prototype-readout ablation in its own
    block, and the complete versions from the run summaries."""
    T = main_table_rows(main_table, bridge, knn)
    err = lambda m, d: T[d][m]['mean_error']
    best = {d: min(COMPARATORS, key=lambda m: err(m, d)) for d in DATASETS}
    headline = [KNN] + COMPARATORS
    lowest = {d: min(err(m, d) for m in headline) for d in DATASETS}
    gap = {(m, d): err(m, d) - err(best[d], d) for m in ARROWFLOW_MODELS for d in DATASETS}
    order = [KNN] + BRIDGE_MODELS[1:] + ['arrowflow_full']
    rows = []
    for m in order:
        cells = [MODEL[m]]
        for d in DATASETS:
            cell = pm(err(m, d), T[d][m]['outer_fold_sd'])
            cells.append(f'\\textbf{{{cell}}}' if m in headline and err(m, d) == lowest[d] else cell)
        rows.append(cells)
    subheads = {0: 'Error, \\% (outer-fold SD)', len(order) - 1: 'Ablation: error, \\% (outer-fold SD)',
                len(order): 'Gap to the best classical model (pp)'}
    rows += [[MODEL[m]] + [signed(gap[(m, d)]) for d in DATASETS] for m in ARROWFLOW_MODELS]
    stats = {m: dict(lowest=sum(err(m, d) <= min(err(c, d) for c in COMPARATORS) for d in DATASETS),
                     within3=sum(gap[(m, d)] <= 0.03 for d in DATASETS)) for m in ARROWFLOW_MODELS}
    n = len(DATASETS)
    alt, proto = stats[KNN], stats['arrowflow_full']
    if 'does not count against candidate_budget' not in knn.protocol['knn_readout']['candidates']:
        raise AssertionError('bridge_knn protocol: the readout grid is no longer outside the candidate budget, but the '
                             'caption says so')
    # the combined analysis writes the main table; this run keeps the complete metrics of the prototype readout (S4)
    prows, seed_sd = [], {}
    for d in DATASETS:
        S = bridge.summary['summaries'][d]
        e, b, f = (lookup(S, 'arrowflow_full', key) for key in ('error', 'balanced_accuracy', 'macro_f1'))
        if not e['mean_within_fold_seed_sd'] < e['outer_fold_sd']:
            raise AssertionError(f'{d}: the text says the prototype readout has a seed SD below its outer-fold SD on every dataset')
        seed_sd[d] = e['mean_within_fold_seed_sd']
        prows.append([LABEL[d], pm(e['mean'], e['outer_fold_sd']), pm(b['mean'], b['outer_fold_sd']), pm(f['mean'], f['outer_fold_sd']),
                      pct(e['mean_within_fold_seed_sd']), f"{e['n_folds']} $\\times$ {e['seeds_per_fold']}"])
    caption = (f'\\textbf{{Complete metrics of the prototype readout.}} Mean outer-fold error, balanced accuracy and macro-F1 in percent, '
               f'with the outer-fold SD in parentheses, of the prototype readout in the first benchmark run on the {words(n)} benchmark '
               f'datasets. Macro-F1 is the unweighted mean of the per-class F1 scores, a class never predicted scoring 0. The seed SD is the '
               f'mean over folds of the SD of the error across the {words(FACTS["n_seeds"])} fitting seeds, and the last column gives outer '
               f'folds $\\times$ fitting seeds. Protocol \\texttt{{{tex(bridge.protocol["protocol_id"])}}}.')
    write_table('tab_s_prototype_metrics', caption, 'tab:s-prototype-metrics',
                ['Dataset', 'Error', 'Bal. acc.', 'Macro-F1', 'Seed SD', 'Folds $\\times$ seeds'], prows, 'lrrrrr', size='\\scriptsize')
    low, high = min(DATASETS, key=seed_sd.get), max(DATASETS, key=seed_sd.get)
    return dict(best=best, gap=gap, stats=stats, seed_low=(low, seed_sd[low]), seed_high=(high, seed_sd[high]))


def render_selected_configurations(ablation, selections):
    S = ablation.summary['summaries']
    inner = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s['inner_score'] for s in selections}
    rows, blocks, stats = [], [], {}
    for d in DATASETS:
        blocks.append(len(rows))
        groups = defaultdict(list)
        for rc in S[d]['resolved_configurations']:
            key = (widths_label(rc['widths']), rc['learning_rate'], int(rc['embed_dim']), int(rc['degree']), bool(rc['augment']))
            groups[key].append(inner[(d, rc['outer_repeat'], rc['outer_fold'])])
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        stats[d] = dict(distinct=len(groups), modal=len(ordered[0][1]))
        for i, ((w, lr, e, p, aug), scores) in enumerate(ordered):
            rows.append([LABEL[d] if i == 0 else '', w, f'{lr:g}', str(e), str(p), 'on' if aug else 'off',
                         f'{len(scores)} of {FACTS["n_outer"]}', pct(sum(scores) / len(scores))])
    caption = (f'\\textbf{{Configurations of the prototype readout chosen on the inner folds.}} For every dataset, the '
               f'configurations selected across the {FACTS["n_outer"]} outer folds of the nested benchmark from its '
               f'{FACTS["grid_size"]} candidates. Each row gives the hidden widths, initial learning rate $\\eta$, encoded vocabulary size $e$ '
               f'and polynomial degree $p$ as resolved from the training partition, and the augmentation switch (resolved by '
               f'the adaptive rule from the training partition). The remaining columns give the number of outer folds that chose the configuration '
               f'and the mean inner-fold selection accuracy of those folds in percent. No outer-fold score enters the '
               f'choice.')
    write_table('tab_s_selected_configs', caption, 'tab:s-selected',
                ['Dataset', 'Widths', '$\\eta$', '$e$', '$p$', 'Augment', 'Folds', 'Inner acc.'],
                rows, 'llrrrlrr', block_rules=blocks)
    return stats


# ----------------------------------------------------------------------------- learning controls
CONTRAST_LABEL = {'output_vs_knn': 'Prototype readout $-$ seven-view footrule kNN control',
                  'trained_vs_initial': 'Trained $-$ initial probe'}


SOURCE_LABEL = {'ablation': 'component ablation', 'matched_v3': 'matched study'}


def render_contrasts(contrasts_csv):
    c = pd.read_csv(contrasts_csv)
    rows, crows, blocks, subheads = [], [], [], {}
    for kind in ('output_vs_knn', 'trained_vs_initial'):
        blocks.append(len(crows))
        subheads[len(rows)] = CONTRAST_LABEL[kind]
        for _, r in c[c['kind'] == kind].iterrows():
            rows.append([LABEL[r['dataset_id']], signed(r['mean_difference']), interval(r['ci_low'], r['ci_high']),
                         pval(r['p_approximate']), pval(r['holm_p_approximate'])])
            crows.append([str(int(r['family_index'])), LABEL[r['dataset_id']], CONTRAST_LABEL[kind], SOURCE_LABEL[r['source']],
                          signed(r['mean_difference']), pct(r['standard_error']), interval(r['ci_low'], r['ci_high']),
                          pval(r['p_approximate']), pval(r['holm_p_approximate'])])
    n = len(c)
    holm_sig = int((c['holm_p_approximate'] < 0.05).sum())
    excl0 = int(((c['ci_low'] > 0) | (c['ci_high'] < 0)).sum())
    ok = c[c['kind'] == 'output_vs_knn']
    tr = c[c['kind'] == 'trained_vs_initial']
    above, tr_above = int((ok['mean_difference'] >= 0).sum()), int((tr['mean_difference'] > 0).sum())
    probe = FACTS['matched']['architectures'][FACTS['matched']['primary_architecture_index']]
    caption = (f'\\textbf{{The {n} prototype-readout contrasts, fixed in the benchmark protocol before its outer folds were scored.}} Paired accuracy differences in percentage points '
               f'(first-named minus second-named; positive favors the first) on the {words(len(ok))} benchmark datasets. '
               f'Prototype readout $-$ seven-view footrule kNN control: the prototype readout with $K={FACTS["views"]}$ '
               f'views against the seven-view footrule kNN control, footrule kNN on each of the same encoded views, tuned '
               f'on inner folds and combined by majority vote (component ablation). Trained $-$ initial probe: a footrule '
               f'kNN probe on the hidden ranking of the trained single-view {widths_label(probe)} network against the same '
               f'probe on its initial filters (matched study). Each difference carries the {interval_note()}; intervals '
               f'are not simultaneous, $p$ values are approximate, and the Holm column adjusts them over the family of '
               f'{n}. The prototype readout is at or above the kNN control on {num_word(above)} datasets and the trained probe '
               f'above the initial probe on {num_word(tr_above)}; '
               f'{of_total(excl0, n, "intervals", "excludes", "exclude", capital=False)} zero, and '
               f'{of_total(holm_sig, n, "contrasts", "remains", "remain", capital=False)} significant after Holm '
               f'adjustment.')
    write_table('tab_contrasts', caption, 'tab:contrasts',
                ['Dataset', 'Difference (pp)', f'{FACTS["confidence"]}\\% interval', '$p$', 'Holm $p$'],
                rows, 'lrrrr', subheads=subheads)
    caption = (f'\\textbf{{Complete record of the {n} contrasts fixed in the benchmark protocol before its outer folds were scored.}} Family index, dataset, contrast, source '
               f'family, mean paired accuracy difference in percentage points, its standard error (SE) in points, the '
               f'{interval_note()}, the approximate $p$ value and the Holm-adjusted $p$ value over the family of {n}. '
               f'Every member is computed, and one Holm adjustment covers the family.')
    write_table('tab_s_contrasts_complete', caption, 'tab:s-contrasts',
                ['\\#', 'Dataset', 'Contrast', 'Source', 'Diff. (pp)', 'SE', f'{FACTS["confidence"]}\\% interval', '$p$',
                 'Holm $p$'], crows, 'rlllrrrrr', block_rules=blocks, size='\\scriptsize', wide=True)
    return c


def fold_accuracy(fam, model, d):
    """Seed-averaged accuracy of every outer fold, keyed by (outer repeat, outer fold), in exact rational arithmetic from
    the model rows."""
    from fractions import Fraction
    per = defaultdict(list)
    for r in fam.summary['model_rows'][d]:
        if r['model_id'] == model:
            n = len(r['test_rows'])
            acc = Fraction(round(r['accuracy'] * n), n)
            if abs(float(acc) - r['accuracy']) > 1e-12:
                raise AssertionError(f'{fam.name}/{d}/{model}: accuracy {r["accuracy"]} is not a count over {n} rows')
            per[(r['outer_repeat'], r['outer_fold'])].append(acc)
    if len(per) != FACTS['n_outer']:
        raise AssertionError(f'{fam.name}/{d}/{model}: {len(per)} outer folds, expected {FACTS["n_outer"]}')
    return {k: sum(v) / len(v) for k, v in per.items()}


def exact_accuracy(fam, model, d):
    """Mean over outer folds of the seed-averaged accuracy, in exact rational arithmetic from the model rows."""
    folds = fold_accuracy(fam, model, d)
    return sum(folds.values()) / len(folds)


def shared_configurations(bridge, knn):
    """Outer folds and outer fits in which ArrowFlow chose the prototype readout's configuration, from the config_id of
    every outer model record in both runs' result files; each method chooses one configuration per outer fold."""
    ids = {}
    for fam, model in ((knn, KNN), (bridge, 'arrowflow_full')):
        for d in DATASETS:
            for path in sorted((fam.dir / 'results').glob(f'{d}__{model}__r*f*.json')):
                for m in read_json(path)['models']:
                    if m.get('stage', 'outer') == 'outer':
                        ids[(fam.name, d, m['outer_repeat'], m['outer_fold'], m['model_seed'])] = m['config_id']
    fits = [key[1:] for key in ids if key[0] == knn.name]
    if len(fits) != len(DATASETS) * FACTS['n_outer'] * FACTS['n_seeds'] or \
            any((bridge.name, *k) not in ids for k in fits):
        raise AssertionError('shared configurations: the two runs do not cover the same outer fits')
    same = {k: ids[(knn.name, *k)] == ids[(bridge.name, *k)] for k in fits}
    folds = defaultdict(set)
    for (d, r, f, _), v in same.items():
        folds[(d, r, f)].add(v)
    if any(len(v) != 1 for v in folds.values()):
        raise AssertionError('shared configurations: the fitting seeds of one outer fold disagree')
    return dict(folds=sum(v == {True} for v in folds.values()), n_folds=len(folds), fits=sum(same.values()),
                n_fits=len(same))


def render_knn_contrasts(compare_dir, bridge, knn, main_table):
    """Table of the seven contrasts of ArrowFlow with the prototype readout (S4), and its complete record (S3)."""
    c = pd.read_csv(compare_dir / 'knn_vs_full_contrasts.csv')
    s = read_json(compare_dir / 'knn_vs_full_summary.json')
    fam, rep = s['family'], s['reproduction']
    if list(c['dataset']) != DATASETS or set(c['model_a']) != {KNN} or set(c['model_b']) != {'arrowflow_full'}:
        raise AssertionError('knn_vs_full_contrasts.csv: expected ArrowFlow minus the prototype readout on the seven datasets')
    if (fam['size'], fam['multiplicity'], fam['n_folds'], fam['df'], fam['test_train_ratio'], round(100 * fam['confidence'])) != \
            (len(DATASETS), 'holm', FACTS['n_outer'], FACTS['df'], FACTS['ratio'], FACTS['confidence']) or \
            knn.protocol['primary_family_size'] != len(DATASETS):
        raise AssertionError('knn_vs_full_summary.json: family settings differ from the captions')
    by_dataset = {r['dataset']: r for r in s['contrasts']}
    for _, r in c.iterrows():
        j = by_dataset[r['dataset']]
        for key in ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate'):
            if abs(float(r[key]) - float(j[key])) > 1e-15:
                raise AssertionError(f'{r["dataset"]}: {key} differs between the contrasts CSV and the summary')
    if not (s['comparators_reproduced'] is True and rep['comparators_reproduced'] is True and rep['cells_mismatched'] == 0
            and not rep['mismatches'] and not rep['definition_mismatches']):
        raise AssertionError('knn_vs_full_summary.json: a comparator did not reproduce the benchmark run')
    T = main_table_rows(main_table, bridge, knn)
    exact = {d: exact_accuracy(knn, KNN, d) - exact_accuracy(bridge, 'arrowflow_full', d) for d in DATASETS}
    rows, crows = [], []
    for _, r in c.iterrows():
        d = r['dataset']
        if abs(float(exact[d]) - float(r['mean_difference'])) > 1e-12:
            raise AssertionError(f'{d}: the exact accuracy difference {float(exact[d])} differs from the CSV')
        rows.append([LABEL[d], signed(r['mean_difference']), interval(r['ci_low'], r['ci_high']),
                     pval(r['p_approximate']), pval(r['holm_p_approximate'])])
        crows.append([LABEL[d], pm(T[d][KNN]['mean_error'], T[d][KNN]['outer_fold_sd']),
                      pm(T[d]['arrowflow_full']['mean_error'], T[d]['arrowflow_full']['outer_fold_sd']),
                      signed(r['mean_difference']), pct(r['standard_error']), interval(r['ci_low'], r['ci_high']),
                      pval(r['p_approximate']), pval(r['holm_p_approximate'])])
    n = len(DATASETS)
    not_lower = sum(v >= 0 for v in exact.values())
    significant = [d for d in DATASETS if float(c.set_index('dataset').loc[d, 'holm_p_approximate']) < 0.05]
    if any(exact[d] <= 0 for d in significant):
        raise AssertionError('a significant contrast of ArrowFlow with the prototype readout does not favor ArrowFlow')
    lead_all = 'on all seven datasets' if not_lower == n else f'on {num_word(not_lower)} of the {words(n)} datasets'
    cells = rep['cells_compared']
    shared = shared_configurations(bridge, knn)
    shared_note = (f'ArrowFlow chose the prototype readout\'s configuration in {shared["folds"]} of the '
                   f'{shared["n_folds"]} outer folds ({shared["fits"]} of {shared["n_fits"]} outer fits)')
    caption = (f'\\textbf{{ArrowFlow against the prototype readout on the benchmark\'s outer partitions.}} Paired accuracy '
               f'differences in percentage points, ArrowFlow minus the prototype readout$^\\dagger$ (positive favors '
               f'ArrowFlow), on the {words(n)} benchmark datasets. ArrowFlow ran in its own run and is paired fold by fold with '
               f'the prototype readout\'s run. Each method chose its configuration by the inner-fold accuracy of its own '
               f'readout, and {shared_note}. Each difference carries the {interval_note()}; $p$ values are approximate, and '
               f'the Holm column adjusts them over these {words(n)} contrasts, a family separate from the {FACTS["family"]} '
               f'prototype-readout contrasts of Table~\\ref{{tab:s-contrasts}}. ArrowFlow has equal or higher mean accuracy '
               f'{lead_all}, and {of_total(len(significant), words(n), "differences", "remains", "remain", capital=False)} '
               f'significant after Holm adjustment. '
               f'{"None" if rep["cells_mismatched"] == 0 else rep["cells_mismatched"]} '
               f'of the {cells:,} comparator cells differ between the two runs. A cell is one outer fit of one of the five '
               f'classical models or the majority class, compared by its chosen configuration and every test prediction. '
               f'$^\\dagger$: from the prototype readout\'s own run on the same partitions.')
    write_table('tab_knn', caption, 'tab:knn',
                ['Dataset', 'Difference (pp)', f'{FACTS["confidence"]}\\% interval', '$p$', 'Holm $p$'], rows, 'lrrrr')
    caption = (f'\\textbf{{Complete record of the {words(n)} contrasts of ArrowFlow with the prototype readout.}} For '
               f'every dataset, the table gives the mean outer-fold error in percent (outer-fold SD) of ArrowFlow in its '
               f'own run and of the prototype readout$^\\dagger$ in the first benchmark run, on the same outer partitions. '
               f'It then gives the mean paired accuracy difference in percentage points (ArrowFlow minus the prototype '
               f'readout) and its standard error (SE) in points. The last columns hold the {interval_note()}, the '
               f'approximate $p$ value and the Holm-adjusted $p$ value over the {words(n)} datasets. Each method chose its '
               f'configuration by the inner-fold accuracy of its own readout: {shared_note}, where the two read the same '
               f'networks. '
               f'$^\\dagger$: from the first benchmark run. Protocol \\texttt{{{tex(knn.protocol["protocol_id"])}}}; '
               f'computed by \\texttt{{compare\\_runs '
               f'knn}} from the model rows of both runs.')
    write_table('tab_s_knn_complete', caption, 'tab:s-knn',
                ['Dataset', MODEL[KNN], MODEL['arrowflow_full'], 'Diff. (pp)', 'SE', f'{FACTS["confidence"]}\\% interval',
                 '$p$', 'Holm $p$'], crows, 'lrrrrrrr', size='\\scriptsize', wide=True)
    return dict(exact=exact, contrasts=c.set_index('dataset'), significant=significant, not_lower=not_lower,
                cells=cells, mismatched=rep['cells_mismatched'], shared=shared)


# ----------------------------------------------------------------------------- referee analyses (S3; descriptive)
REFEREE_STATUS = 'descriptive; not a registered family; no multiplicity adjustment'
PLAIN = {KNN: 'ArrowFlow', 'svc_rbf': 'the SVC', 'random_forest': 'random forest', 'mlp': 'the MLP',
         'numeric_knn': 'numeric kNN', 'gradient_boosting': 'gradient boosting'}
THE = dict(PLAIN, random_forest='the random forest')


def sealed(record, directory, names):
    """Every CSV that an analysis record seals must match its recorded sha256 and row count."""
    for name in names:
        seal, path = record['outputs'][name], Path(directory) / name
        if sha256_file(path) != seal['sha256'] or len(pd.read_csv(path)) != seal['rows']:
            raise AssertionError(f'{path}: differs from the seal in its analysis record')


def referee_source(block, fam):
    """The run that an analysis read must be the run on disk: protocol, code revision and summary hash."""
    if (block['protocol_id'], block['code_revision'], block['summary_sha256']) != \
            (fam.protocol['protocol_id'], fam.environment['code_revision'], sha256_file(fam.dir / 'summary.json')):
        raise AssertionError(f'referee analyses: the {fam.name} source is not the run on disk')


def render_referee(ref_dir, bridge, knn, main_table, compare_dir):
    """Three descriptive analyses of the frozen benchmark runs (S3): ArrowFlow minus every comparator with paired intervals,
    the readout difference and every model's accuracy on the test rows without an exact training copy, and Friedman ranks
    with the Nemenyi critical difference. Every value is read from the sealed outputs, whose source runs must be the runs on
    disk, and is checked against the benchmark tables it describes; the rank statistics are recomputed. The restricted-row
    accuracy table and every sentence built on it left the supplement by the author's decision of 2026-09-23 (the duplicate
    material shrinks to one paragraph and the audit table), so the duplicate analysis is only verified here, as the record
    that Section S5 describes."""
    from scipy import stats
    T = main_table_rows(main_table, bridge, knn)
    err = lambda m, d: T[d][m]['mean_error']
    tuned, n, conf = [KNN] + COMPARATORS, len(DATASETS), FACTS['confidence']
    revisions = set()

    # ArrowFlow minus every comparator: 42 unadjusted paired contrasts
    cdir = ref_dir / 'comparators'
    cj = read_json(cdir / 'comparator_contrasts.json')
    sealed(cj, cdir, ['comparator_contrasts.csv'])
    referee_source(cj['provenance']['knn'], knn)
    revisions.add(cj['provenance']['code_revision'])
    comparators = ['dummy'] + COMPARATORS
    if (cj['status'], cj['model_a'], cj['comparators'], cj['datasets'], cj['n_folds'], cj['df'], cj['test_train_ratio'],
            round(100 * cj['confidence'])) != (REFEREE_STATUS, KNN, comparators, DATASETS, FACTS['n_outer'], FACTS['df'],
                                               FACTS['ratio'], conf):
        raise AssertionError('comparator_contrasts.json: status, models or design differ from the captions')
    cc = pd.read_csv(cdir / 'comparator_contrasts.csv')
    best = {d: min(comparators, key=lambda m: err(m, d)) for d in DATASETS}
    if len(cc) != n * len(comparators) or set(cc['status']) != {REFEREE_STATUS}:
        raise AssertionError('comparator_contrasts.csv: unexpected rows or status')
    C = {}
    for _, r in cc.iterrows():
        d, m = r['dataset'], r['model_b']
        if r['model_a'] != KNN or abs(r['mean_error_a'] - err(KNN, d)) > 1e-12 or abs(r['mean_error_b'] - err(m, d)) > 1e-12 \
                or bool(r['best_comparator']) != (m == best[d]) or cj['best_comparator'][d] != [best[d]] \
                or (int(r['n_folds']), int(r['df'])) != (FACTS['n_outer'], FACTS['df']):
            raise AssertionError(f'comparator_contrasts.csv/{d}/{m}: errors, best flag or folds differ from Table 2')
        C[(d, m)] = r
    gap = {d: C[(d, best[d])] for d in DATASETS}
    if any(gap[d]['mean_difference'] >= 0 or gap[d]['ci_low'] > 0 for d in DATASETS):
        raise AssertionError('the text says ArrowFlow is less accurate than the best comparator on every dataset')
    clear = [d for d in DATASETS if gap[d]['ci_high'] < 0]
    near = min(DATASETS, key=lambda d: abs(gap[d]['mean_difference']))
    far = max(DATASETS, key=lambda d: abs(gap[d]['mean_difference']))

    # exact duplicate rows: the sealed record of the readout difference and of every model's accuracy without the test rows that
    # have a training copy, verified but no longer printed (author's decision of 2026-09-23)
    ddir = ref_dir / 'duplicates'
    dj = read_json(ddir / 'duplicate_sensitivity.json')
    sealed(dj, ddir, ['duplicate_accuracy.csv', 'duplicate_folds.csv', 'duplicate_groups.csv', 'duplicate_readout.csv'])
    referee_source(dj['provenance']['knn'], knn)
    referee_source(dj['provenance']['bridge'], bridge)
    revisions.add(dj['provenance']['code_revision'])
    every, kept = 'all_test_rows', 'test_rows_without_training_duplicate'
    if (dj['status'], dj['row_sets'], dj['test_train_ratio'], round(100 * dj['confidence'])) != \
            (REFEREE_STATUS, [every, kept], FACTS['ratio'], conf) or dj['reproduction']['all_rows_reproduced_exactly'] is not True:
        raise AssertionError('duplicate_sensitivity.json: status, row sets, design or reproduction differ from the captions')
    G = pd.read_csv(ddir / 'duplicate_groups.csv').set_index('dataset')
    RD = pd.read_csv(ddir / 'duplicate_readout.csv').set_index(['dataset', 'row_set'])
    A = pd.read_csv(ddir / 'duplicate_accuracy.csv').set_index(['dataset', 'model_id'])
    K = pd.read_csv(compare_dir / 'knn_vs_full_contrasts.csv').set_index('dataset')
    if list(G.index) != DATASETS or (G['label_conflicting_groups'] != 0).any() or (G['folds_without_remaining_rows'] != 0).any() \
            or {REFEREE_STATUS} != set(G['status']) | set(RD['status']) | set(A['status']):
        raise AssertionError('duplicate outputs: datasets, label conflicts, dropped folds or status differ from the text')
    rd = lambda d, s, x='mean_difference': float(RD.loc[(d, s)][x])
    for d in DATASETS:
        for s in (every, kept):
            r = RD.loc[(d, s)]
            if (r['model_a'], r['run_a'], r['model_b'], r['run_b'], int(r['n_folds']), int(r['folds_without_remaining_rows'])) != \
                    (KNN, 'knn', 'arrowflow_full', 'bridge', FACTS['n_outer'], 0):
                raise AssertionError(f'duplicate_readout.csv/{d}/{s}: pairing or folds differ from the text')
        if any(abs(rd(d, every, x) - float(K.loc[d, x])) > 1e-12 for x in ('mean_difference', 'ci_low', 'ci_high')):
            raise AssertionError(f'duplicate_readout.csv/{d}: the all-rows difference is not the contrast of Table 5')
        if int(G.loc[d, 'duplicate_rows']) == 0 and any(rd(d, every, x) != rd(d, kept, x) for x in ('mean_difference', 'ci_low', 'ci_high')):
            raise AssertionError(f'duplicate_readout.csv/{d}: without duplicate rows both row sets must give the same difference')
    models = tuned + ['dummy', 'arrowflow_full']
    for d in DATASETS:
        for m in models:
            r = A.loc[(d, m)]
            expected = 1 - (T[d]['arrowflow_full']['mean_error'] if m == 'arrowflow_full' else err(m, d))
            if r['source_run'] != ('bridge' if m == 'arrowflow_full' else 'knn') or str(r['reproduced_exactly']) != 'True' \
                    or abs(r['accuracy_all_test_rows'] - expected) > 1e-12:
                raise AssertionError(f'duplicate_accuracy.csv/{d}/{m}: the all-rows accuracy differs from the benchmark tables')
    share = lambda d: float(G.loc[d, 'share_with_training_duplicate'])
    with_dups = sorted((d for d in DATASETS if int(G.loc[d, 'duplicate_rows']) > 0), key=lambda d: -share(d))
    if len(with_dups) < 2:
        raise AssertionError('the duplicate paragraph of S3.2 names two benchmark datasets with duplicate rows')
    wq, seg = with_dups[:2]

    # Friedman ranks and the Nemenyi critical difference over the six tuned models
    rdir = ref_dir / 'ranks'
    fj = read_json(rdir / 'friedman_nemenyi.json')
    sealed(fj, rdir, ['mean_ranks.csv', 'rank_matrix.csv'])
    referee_source(fj['provenance']['knn'], knn)
    revisions.add(fj['provenance']['code_revision'])
    if fj['models'] != tuned or fj['excluded_models'] != ['dummy'] or fj['metric'] != 'error' or fj['status'] != 'descriptive':
        raise AssertionError('friedman_nemenyi.json: models, metric or status differ from the caption')
    matrix = np.array([[err(m, d) for m in tuned] for d in DATASETS])
    k = len(tuned)
    ranks = np.vstack([stats.rankdata(row) for row in matrix])
    if any(len(set(row)) < k for row in matrix.tolist()) or any(len({pct(x) for x in row}) < k for row in matrix):
        raise AssertionError('two errors tie, but the caption says no errors tie at full or table precision')
    mean_rank = dict(zip(tuned, ranks.mean(axis=0)))
    chi2, p = stats.friedmanchisquare(*matrix.T)
    F = (n - 1) * chi2 / (n * (k - 1) - chi2)
    pF = stats.f.sf(F, k - 1, (k - 1) * (n - 1))
    alpha = fj['nemenyi']['alpha']
    cd = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n))
    lowest, highest = min(tuned, key=mean_rank.get), max(tuned, key=mean_rank.get)
    spread = mean_rank[highest] - mean_rank[lowest]
    recorded = pd.read_csv(rdir / 'mean_ranks.csv').set_index('model_id')['mean_rank']
    for i, d in enumerate(DATASETS):
        for j, m in enumerate(tuned):
            if abs(fj['matrix'][d][m] - matrix[i, j]) > 1e-12 or fj['ranks'][d][m] != ranks[i, j]:
                raise AssertionError(f'friedman_nemenyi.json/{d}/{m}: error or rank differs from Table 2')
    fr, nm = fj['friedman'], fj['nemenyi']
    if any(abs(fj['mean_ranks'][m] - mean_rank[m]) > 1e-12 or abs(recorded[m] - mean_rank[m]) > 1e-12 for m in tuned) or \
            max(abs(fr['chi2'] - chi2), abs(fr['p'] - p), abs(fr['iman_davenport_f'] - F), abs(fr['iman_davenport_p'] - pF),
                abs(nm['critical_difference'] - cd), abs(nm['largest_mean_rank_difference']['mean_rank_difference'] - spread)) > 1e-9 \
            or fr['df'] != k - 1 or nm['pairs_exceeding_critical_difference'] or spread >= cd:
        raise AssertionError('friedman_nemenyi.json: ranks or statistics differ from a recomputation, or a pair is separated')

    behind = sorted((m for m in tuned if mean_rank[m] < mean_rank[KNN]), key=mean_rank.get)
    equal = [m for m in tuned if m != KNN and mean_rank[m] == mean_rank[KNN]]
    if len(behind) != 2 or len(equal) != 1 or len(revisions) != 1:
        raise AssertionError('the text names two models ahead of ArrowFlow in mean rank and one equal, from one code revision')
    return dict(
        k=k, n_contrasts=len(cc), clear=clear, near=pct(abs(gap[near]['mean_difference'])), near_d=near,
        far=pct(abs(gap[far]['mean_difference'])), far_d=far, mean_rank=mean_rank, chi2=chi2, p=p, cd=cd,
        rank_sentence=(f"ArrowFlow's mean rank is ${mean_rank[KNN]:.2f}$, behind {THE[behind[0]]} (${mean_rank[behind[0]]:.2f}$) "
                       f"and {THE[behind[1]]} (${mean_rank[behind[1]]:.2f}$) and equal to that of {THE[equal[0]]}."),
        wq=wq, seg=seg, revision=revisions.pop()[:9], duplicate_rows={d: int(G.loc[d, 'duplicate_rows']) for d in DATASETS})


def render_grids(knn, bridge, training, projected, runs):
    """Candidate grids of every model (S2): the classical grids from the harness source, checked against the candidates that
    both benchmark runs evaluated, and ArrowFlow's grid and readout grid from its protocol."""
    tree = ast.parse((REPO / 'experiments' / 'make_revision' / 'comparisons.py').read_text())
    found = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(getattr(t, 'id', None) == 'CONVENTIONAL_GRIDS' for t in node.targets)]
    if len(found) != 1 or sorted(found[0]) != sorted(COMPARATORS):
        raise AssertionError('comparisons.py: no single grid for every classical model')
    grids, budget, seed = found[0], knn.protocol['candidate_budget'], knn.protocol['candidate_seed']
    ck, cb = read_json(knn.dir / 'candidates.json'), read_json(bridge.dir / 'candidates.json')
    as_json = lambda v: list(v) if isinstance(v, tuple) else v

    def value(v, key=None):
        if v is None:
            return 'None'
        if isinstance(v, tuple):
            return widths_label(v)
        if key == 'max_features' and isinstance(v, float) and v == 1.0:
            return '1.0 (all features)'   # scikit-learn reads the integer 1 as one feature and the float 1.0 as all of them
        return f'{v:g}' if isinstance(v, float) else tex(v)
    fm, ro = knn.protocol['full_method'], knn.protocol['knn_readout']['grid']
    af = ck[KNN]['candidates']
    if len(af) != FACTS['grid_size'] or len(set(ck[KNN]['config_ids'])) != len(af) or FACTS['grid_size'] > budget:
        raise AssertionError('ArrowFlow run: its candidates are not the whole protocol grid')
    symbol = {'widths': 'widths', 'learning_rate': '$\\eta$', 'embed_scale': '$s_e$', 'degree_offset': '$\\delta_p$'}
    shown = lambda key, v: widths_label(v) if key == 'widths' else f'${v:g}$'
    af_text = ('; '.join(f'{symbol[key]}: ' + ', '.join(shown(key, v) for v in fm['candidate_grid'][key]) for key in symbol)
               + '; readout, inside every fit: ' + ', '.join(str(v) for v in ro['n_neighbors']) + ' neighbors with '
               + ' or '.join(ro['weights']) + ' weights')
    rows, sampled = [[MODEL[KNN], af_text, str(FACTS['grid_size']), str(len(af))]], []
    for m in COMPARATORS:
        g = grids[m]
        size = math.prod(len(v) for v in g.values())
        cand = ck[m]['candidates']
        if cand != cb[m]['candidates'] or len(set(ck[m]['config_ids'])) != len(cand) or len(cand) != min(size, budget) or \
                any(sorted(c) != sorted(g) or any(c[key] not in [as_json(v) for v in g[key]] for key in g) for c in cand):
            raise AssertionError(f'{m}: the candidates of the two runs are not drawn from the grid of comparisons.py')
        if size > budget:
            sampled.append(f'{MODEL[m][0].lower() + MODEL[m][1:]} uses {len(cand)} of its {size} combinations')
        rows.append([MODEL[m], '; '.join(f'\\texttt{{{tex(key).replace("\\_", "\\_\\allowbreak ")}}}: ' + ', '.join(value(v, key) for v in vals)
                                         for key, vals in g.items()), str(size), str(len(cand))])
    rows.append([MODEL['dummy'], '--', '1', str(len(ck['dummy']['candidates']))])

    # the controls and baselines of Sections S3.3 to S3.7, from their frozen protocols and the candidates their runs evaluated
    # (referee panel of 2026-09-23, item R04-5: the table once claimed every grid but listed only the models of Table 3)
    first_control = len(rows)
    encoder_keys = {'widths': 'widths', 'embed_scale': '$s_e$', 'degree_offset': '$\\delta_p$'}
    tp, pp = training.protocol['training_controls']['models'], projected.protocol['projected_control']
    baseline_protocol = read_json(PROTOCOLS_G5 / 'neighbour_baselines.json')
    if sha256_file(runs / BASELINES / 'run' / 'protocol.json') != sha256_file(PROTOCOLS_G5 / 'neighbour_baselines.json'):
        raise AssertionError('neighbor baselines: the run protocol differs from the frozen protocol file')
    cands = {**read_json(training.dir / 'candidates.json'), **read_json(projected.dir / 'candidates.json'),
             **read_json(runs / BASELINES / 'run' / 'candidates.json')}
    readout_text = ', '.join(str(v) for v in ro['n_neighbors']) + ' neighbors with ' + ' or '.join(ro['weights']) + ' weights'
    numeric = grids['numeric_knn']

    def encoder_grid(keys):
        return {k: fm['candidate_grid'][k] for k in encoder_keys if k in keys}

    def checked(model, grid, label):
        c = cands[model]['candidates']
        size = math.prod(len(v) for v in grid.values())
        if len(c) != min(size, budget) or any(any(x[k] not in [as_json(v) for v in grid[k]] for k in grid) for x in c):
            raise AssertionError(f'{model}: its evaluated candidates are not drawn from the grid its protocol declares')
        if size > budget:
            sampled.append(f'{label} uses {len(c)} of its {size} combinations')
        return str(size), str(len(c))

    def key_name(k):
        return {'embed_scale': '$s_e$', 'degree_offset': '$\\delta_p$', 'component_scale': 'component share'}.get(
            k, '\\texttt{' + tex(k).replace('\\_', '\\_\\allowbreak ') + '}')

    if baseline_protocol['candidate_seed'] != seed or baseline_protocol['candidate_budget'] != budget:
        raise AssertionError('neighbor baselines: another candidate seed or budget than the benchmark\'s')

    for model, label, keys, readout in ((UNTRAINED, CONTROL[UNTRAINED], tp[UNTRAINED]['candidate_keys'], readout_text),
                                        (INPUT_KNN, CONTROL[INPUT_KNN], tp[INPUT_KNN]['candidate_keys'], readout_text),
                                        (PROJECTED, 'Numeric kNN, projected scores', pp['model']['candidate_keys'],
                                         '; '.join(f'\\texttt{{{tex(k)}}}: ' + ', '.join(value(v) for v in numeric[k]) for k in numeric))):
        g = encoder_grid(keys)
        if pp['readout']['grid'] != {k: list(v) for k, v in numeric.items()} and model == PROJECTED:
            raise AssertionError('projected control: its readout grid is not the numeric kNN grid')
        rows.append([label, '; '.join(f'{encoder_keys[k]}: ' + ', '.join(shown(k, v) for v in vals) for k, vals in g.items())
                     + '; readout, inside every fit: ' + readout, *checked(model, g, label)])
    for model in BASELINE_MODELS:
        g = baseline_protocol['models'][model]['grid']
        shown_b = lambda k, v: (f'${v:g}$' if k in ('embed_scale', 'degree_offset') else value(v))
        rows.append([BASELINE[model], '; '.join(key_name(k) + ': ' + ', '.join(shown_b(k, v) for v in vals) for k, vals in g.items()),
                     *checked(model, g, BASELINE[model])])
    caption = (f'\\textbf{{Candidate grids.}} For every model of the benchmark, the hyperparameters and values of its grid '
               f'(scikit-learn names for the classical models), the grid size and the number of distinct candidates that its '
               f'inner folds choose among. The lower block gives the same for every control and baseline of '
               f'Section~\\ref{{supp:results}}, and every model chooses on the same inner folds under the same budget. In the lower '
               f'block, nearest-neighbor classifiers follow a linear discriminant analysis (LDA), a principal component analysis '
               f'(PCA) or a neighborhood components analysis (NCA) and keep a share of the largest number of its components. The '
               f'Kendall support vector classifier (SVC) combines seven views by majority vote. A grid larger than the budget of '
               f'{budget} candidates is sampled once with the candidate seed {seed}: '
               f'{listing(sampled, "and") if sampled else "no grid is"}. Smaller grids are used whole, and both benchmark runs used '
               f'the same candidates. ArrowFlow\'s readout settings, {len(ro["n_neighbors"])} neighbor counts with '
               f'{words(len(ro["weights"]))} weightings, are chosen inside every fit and do not count against the budget, so '
               f'ArrowFlow tunes more settings than its candidate count shows. The untrained ArrowFlow, tuned input footrule kNN and '
               f'numeric kNN on the projected scores choose among ArrowFlow\'s encoder settings and tune their readout inside every '
               f'fit in the same way. RBF: radial basis function; MLP: multilayer perceptron; kNN: nearest-neighbor classifier.')
    write_table('tab_s_grids', caption, 'tab:s-grids', ['Model', 'Hyperparameters and values', 'Grid', 'Candidates'], rows,
                'lp{0.6\\textwidth}rr', size='\\scriptsize', subheads={first_control: 'Controls and baselines'})


# ----------------------------------------------------------------------------- selection records of a nested run
RECORDS = {}
HASHES = {}


def sha256_file(path):
    path = Path(path)
    if path not in HASHES:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def selection_records(fam, model, datasets=None):
    """The selection of every outer fold of one model in a nested run, read once from its result files and keyed by
    (dataset, outer repeat, outer fold): the chosen configuration, its inner score and finalists, the score and
    validation size of every inner fit, and the training size and recorded encoder settings of every outer fit. The
    datasets default to the seven benchmark datasets."""
    datasets = DATASETS if datasets is None else list(datasets)
    key = (str(fam.dir), model, tuple(datasets))
    if key in RECORDS:
        return RECORDS[key]
    out = {}
    for d in datasets:
        for path in sorted((fam.dir / 'results').glob(f'{d}__{model}__r*f*.json')):
            rec = read_json(path)
            sel = rec['selection']
            outer = [m for m in rec['models'] if m.get('stage', 'outer') == 'outer']
            folds = {(m['outer_repeat'], m['outer_fold']) for m in outer}
            if rec.get('status') != 'ok' or len(folds) != 1 or any(m.get('status') != 'ok' for m in outer):
                raise AssertionError(f'{fam.name}/{path.name}: not one completed outer fold')
            out[(d, *folds.pop())] = dict(
                config=sel['config'], config_id=sel['config_id'], inner_score=sel['inner_score'],
                finalist_ids=list(sel['finalist_ids']),
                fits=[dict(config_id=f['config_id'], model_seed=f['model_seed'], status=f['status'], score=f['score'],
                           n_validation=len(f['validation_rows'])) for f in sel['fits']],
                outer=[dict(config_id=m['config_id'], model_seed=m['model_seed'], n_train=len(m['fit_rows']),
                            settings=m.get('preprocessing_settings') or {}) for m in outer])
    if len(out) != len(datasets) * FACTS['n_outer']:
        raise AssertionError(f'{fam.name}/{model}: {len(out)} outer folds in the result files, expected '
                             f'{len(datasets) * FACTS["n_outer"]}')
    RECORDS[key] = out
    return out


def selection_round_off(records, fit_seeds):
    """Outer folds on which ranking the inner-fold mean accuracies as computed in double precision, as the harness does,
    changed the set of rescored finalists against exact rational arithmetic under the same tie rule (lowest configuration
    ID), and folds on which exact arithmetic would rank the recorded finalists differently. The double-precision ranking
    must first reproduce every recorded selection."""
    from fractions import Fraction

    def rank(configs, fits, exact):
        keyed = []
        for c in configs:
            scores = [f for f in fits if f['config_id'] == c]
            if exact:
                values = [Fraction(round(f['score'] * f['n_validation']), f['n_validation']) for f in scores]
                if any(abs(float(v) - f['score']) > 1e-12 for v, f in zip(values, scores)):
                    raise AssertionError('an inner score is not a count over its validation rows')
                keyed.append((-(sum(values) / len(values)), c))
            else:
                keyed.append((-float(np.mean([f['score'] for f in scores])), c))
        return [c for _, c in sorted(keyed)]

    finalists = choice = 0
    for key, rec in records.items():
        fits = rec['fits']
        if any(f['status'] != 'ok' for f in fits):
            raise AssertionError(f'{key}: a failed inner fit, which the round-off count does not handle')
        candidates = list(dict.fromkeys(f['config_id'] for f in fits))
        screen = [f for f in fits if f['model_seed'] == fit_seeds[0]]
        k = len(rec['finalist_ids'])
        computed = rank(candidates, screen, False)[:k]
        if computed != rec['finalist_ids'] or rank(computed, fits, False)[0] != rec['config_id']:
            raise AssertionError(f'{key}: the double-precision ranking does not reproduce the recorded selection')
        exact_set = set(rank(candidates, screen, True)[:k])
        if exact_set != set(computed):
            if len(exact_set ^ set(computed)) != 2:
                raise AssertionError(f'{key}: round-off replaced more than one rescored candidate, but Section S2 says one')
            finalists += 1
        choice += rank(computed, fits, True)[0] != rec['config_id']
    return dict(finalists=finalists, choice=choice, n_folds=len(records), n_finalists=k)


def configuration_rows(method, datasets, records, n_features_of, name):
    """Rows of a table of the configurations ArrowFlow chose on the inner folds, from the selection records of one run,
    resolved as the harness resolves them from the training partition; the resolved vocabulary size and degree must equal
    the encoder settings recorded with every outer fit. Also counts the folds with forced ties in the first hidden layer
    and the folds whose resolved degree is one, at which the encoding ignores rescaling about the training mean."""
    grid, defaults = method['candidate_grid'], method['adaptive_encoding_defaults']
    cap = int(re.search(r'comb\(n_features\+degree,degree\)>(\d+)', method['resolution']).group(1))
    minimum_rows = int(re.search(r'n_train>=(\d+)', method['resolution']).group(1))
    rows, blocks, distinct, stats = [], [], [], Counter()
    ties, tie_layers, degree_one = Counter(), set(), Counter()
    for d in datasets:
        blocks.append(len(rows))
        n_features = int(n_features_of(d))
        base = defaults['n_features<=10' if n_features <= 10 else 'n_features<=30' if n_features <= 30 else 'n_features>30']
        groups = defaultdict(list)
        for (dd, r, f), rec in sorted(records.items()):
            if dd != d:
                continue
            c = rec['config']
            n_train = {o['n_train'] for o in rec['outer']}
            if len(n_train) != 1 or any(o['config_id'] != rec['config_id'] for o in rec['outer']):
                raise AssertionError(f'{d} r{r}f{f}: the outer fits disagree on the training partition or configuration')
            e = int(min(128, max(8, round(base['embed_dim'] * c['embed_scale']))))
            p = max(1, base['degree'] + c['degree_offset'])
            while p > 1 and math.comb(n_features + p, p) > cap:
                p -= 1
            augment = bool(base['augment'] and n_train.pop() >= minimum_rows)
            if {(o['settings'].get('embed_dim'), o['settings'].get('degree')) for o in rec['outer']} != {(e, p)}:
                raise AssertionError(f'{d} r{r}f{f}: the resolved vocabulary size or degree differs from the recorded encoder')
            groups[(widths_label(c['widths']), c['learning_rate'], e, p, augment)].append(rec['inner_score'])
            degree_one[d] += p == 1
            # a layer of N filters over V items has at most V^2/4+1 distinct footrule responses, so N beyond that forces ties
            layers = [e] + [int(w) for w in c['widths']]
            forced = [(v, nf) for v, nf in zip(layers, layers[1:]) if nf > v * v // 4 + 1]
            if forced:
                if forced != [(e, int(c['widths'][0]))]:
                    raise AssertionError(f'{d} r{r}f{f}: the text says the forced ties lie in the first hidden layer only')
                ties[d] += 1
                tie_layers.update(forced)
            stats['folds'] += 1
            stats['single'] += list(c['widths']) == list(grid['widths'][0])
            stats['rate'] += c['learning_rate'] == max(grid['learning_rate'])
            stats['scale'] += c['embed_scale'] == max(grid['embed_scale'])
            stats['offset'] += c['degree_offset'] == min(grid['degree_offset'])
        distinct.append(len(groups))
        for i, ((w, lr, e, p, aug), scores) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
            rows.append([name(d) if i == 0 else '', w, f'{lr:g}', str(e), str(p), 'on' if aug else 'off',
                         f'{len(scores)} of {FACTS["n_outer"]}', pct(sum(scores) / len(scores))])
    return dict(rows=rows, blocks=blocks, distinct=distinct, stats=stats, ties=ties, tie_layers=tie_layers,
                degree_one=degree_one)


SELECTED_HEAD = ['Dataset', 'Widths', '$\\eta$', '$e$', '$p$', 'Augment', 'Folds', 'Inner acc.']


def selected_caption(which, source):
    return (f'\\textbf{{Configurations of ArrowFlow chosen on the inner folds, {which}.}} For every {which[:-1]}, the '
            f'configurations selected across the {FACTS["n_outer"]} outer folds of the nested evaluation from its '
            f'{FACTS["grid_size"]} candidates, read from the selection records of {source}. Each row gives the hidden widths, '
            f'initial learning rate $\\eta$, encoded vocabulary size $e$ and polynomial degree $p$ as resolved from the training '
            f'partition, and the augmentation switch (resolved by the adaptive rule from the training partition). The '
            f'remaining columns give the number of outer folds that chose the configuration and the mean inner-fold '
            f'selection accuracy of those folds in percent, measured with ArrowFlow\'s own readout. No outer-fold score '
            f'enters the choice.')


def render_knn_selected_configurations(knn, records):
    """Configurations ArrowFlow chose on the inner folds of the benchmark datasets (S3), from the selection records of its run."""
    method = knn.protocol['full_method']
    t = configuration_rows(method, DATASETS, records, lambda d: read_json(knn.dir / d / 'manifest.json')['shape'][1],
                           lambda d: LABEL[d])
    write_table('tab_s_knn_selected_configs', selected_caption('benchmark datasets', 'the ArrowFlow run'), 'tab:s-knn-selected',
                SELECTED_HEAD, t['rows'], 'llrrrlrr', block_rules=t['blocks'], size='\\scriptsize')
    if len(t['tie_layers']) != 1:
        raise AssertionError('the text names one layer size whose responses tie at every input')
    (tie_v, tie_n), = t['tie_layers']
    grid = method['candidate_grid']
    return dict(t['stats'], single_label=widths_label(grid['widths'][0]), rate_value=max(grid['learning_rate']),
                distinct_min=min(t['distinct']), distinct_max=max(t['distinct']), tie_folds=sum(t['ties'].values()),
                tie_datasets=[d for d in DATASETS if t['ties'][d]], tie_V=tie_v, tie_N=tie_n, tie_min=tie_n - tie_v * tie_v // 4,
                degree_one=t['degree_one'])


def render_further_selected_configurations(knn, newdata):
    """The same table for the ten further datasets (referee panel of 2026-09-23, item R07-7), from the selection records of
    the two batch runs, whose design must be ArrowFlow's; also the round-off count of their selections (item A16)."""
    B = newdata['B']
    runs_of = {d: B[b] for b in B for d in B[b].protocol['datasets']}
    if any(B[b].protocol['full_method'] != knn.protocol['full_method'] for b in B) or sorted(runs_of) != sorted(NEWDATA):
        raise AssertionError('batch protocols: ArrowFlow\'s design differs from that of the benchmark run')
    records = {}
    for d in NEWDATA:
        records.update(selection_records(runs_of[d], KNN, [d]))
    t = configuration_rows(knn.protocol['full_method'], NEWDATA, records, lambda d: newdata['pin'][d]['shape'][1],
                           lambda d: NAME[d])
    write_table('tab_s_knn_selected_further',
                '\\textbf{Configurations of ArrowFlow chosen on the inner folds, further datasets.} The same record as '
                'Table~\\ref{tab:s-knn-selected}, read from the selection records of the two batch runs; no outer-fold score enters '
                'the choice. HCV: hepatitis C virus.',
                'tab:s-knn-selected-further', SELECTED_HEAD, t['rows'], 'llrrrlrr', block_rules=t['blocks'], size='\\scriptsize',
                stretch=0.85)
    fit_seeds = {tuple(B[b].protocol['fit_seeds']) for b in B}
    if len(fit_seeds) != 1:
        raise AssertionError('batch protocols: the fitting seeds differ between the batches')
    return dict(degree_one=t['degree_one'], round_off=selection_round_off(records, list(fit_seeds.pop())), folds=len(records))


# ----------------------------------------------------------------------------- training controls (Section 7.3, S3)
def holm(p_values):
    """Step-down Holm adjustment over one family, monotone and capped at one."""
    order = sorted(range(len(p_values)), key=lambda i: (p_values[i], i))
    adjusted, running, m = [0.0] * len(p_values), 0.0, len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def corrected_t(values, confidence):
    """Mean, standard error, interval and two-sided p value of the corrected resampled t over fold differences."""
    from scipy import stats
    n = len(values)
    mean = sum(values) / n
    se = math.sqrt((1 / n + FACTS['ratio']) * statistics.variance(values))
    q = stats.t.ppf(0.5 + confidence / 200, n - 1)
    p = 2 * stats.t.sf(abs(mean) / se, n - 1) if se > 0 else float('nan')
    return mean, se, mean - q * se, mean + q * se, p


def render_training(training, knn, knn_records):
    """The fourteen training contrasts of ArrowFlow with the untrained ArrowFlow and tuned input footrule kNN, their
    errors and the descriptive depth split (Table 3, Tables S6-S7 by source order), from compare_runs training. Every
    difference, interval and p value is recomputed from the model rows of both runs, the Holm column over the family, the
    errors from both summaries and the depth groups from the selection records of both runs."""
    cdir = training.dir / 'compare_training'
    c = pd.read_csv(cdir / 'training_contrasts.csv')
    cj = read_json(cdir / 'training_contrasts.json')
    et = read_json(cdir / 'training_error_table.json')
    split = read_json(cdir / 'training_depth_split.json')
    tp, n, n_outer, conf = training.protocol, len(DATASETS), FACTS['n_outer'], FACTS['confidence']
    tc, size = tp['training_controls'], tp['primary_family_size']
    frozen_file = PROTOCOLS_V3 / 'knn_training.json'
    if sha256_file(frozen_file) != sha256_file(training.dir / 'protocol.json'):
        raise AssertionError('knn_training: the protocol of the run differs from the frozen protocol file')
    if not (tp.get('frozen') is True and tp['frozen_at_utc'] > knn.protocol['frozen_at_utc']
            and f'training-pairing against {knn.protocol["protocol_id"]} passed' in tp['resource_decision']):
        raise AssertionError('knn_training: the text says the family was registered after the benchmark results')
    if size != 2 * n or tp['datasets'] != DATASETS or tp['primary_contrasts'] != [f'{KNN}_vs_{m}' for m in CONTROLS] \
            or tc['reference']['protocol_id'] != knn.protocol['protocol_id'] or tc['reference']['model_id'] != KNN:
        raise AssertionError('knn_training: family size, datasets, contrasts or reference differ from the text')
    for key in ('split_seed', 'outer_folds', 'outer_repeats', 'inner_folds', 'fit_seeds', 'candidate_budget',
                'candidate_tie_rule', 'selection_metric', 'test_train_ratio', 'confidence'):
        if tp[key] != knn.protocol[key]:
            raise AssertionError(f'knn_training: {key} differs from the ArrowFlow run, but the text says the design is shared')
    grid, candidates = knn.protocol['full_method']['candidate_grid'], read_json(training.dir / 'candidates.json')
    dropped, counts = {}, {}
    for m in CONTROLS:
        keys = set(tc['models'][m]['candidate_keys'])
        dropped[m] = sorted(set(grid) - keys)
        counts[m] = math.prod(len(grid[k]) for k in grid if k in keys)
        if len(candidates[m]['candidates']) != counts[m] or tc['models'][m]['candidates'] != counts[m]:
            raise AssertionError(f'knn_training/{m}: the candidates are not the projection of ArrowFlow\'s grid')
    if dropped[UNTRAINED] != ['learning_rate'] or dropped[INPUT_KNN] != ['learning_rate', 'widths']:
        raise AssertionError('knn_training: the text says the untrained ArrowFlow drops only the learning rate and tuned '
                             'input footrule kNN also the widths')
    for name, block in (('training_contrasts.json', cj['provenance']), ('training_error_table.json', et['sources']),
                        ('training_depth_split.json', split['sources'])):
        for run, fam in (('knn', knn), ('training', training)):
            s = block[run]
            if (s['protocol_id'], s['code_revision'], s['summary_sha256']) != \
                    (fam.protocol['protocol_id'], fam.environment['code_revision'], sha256_file(fam.dir / 'summary.json')):
                raise AssertionError(f'{name}: its {run} source is not the run on disk')
    if cj['provenance']['training']['protocol_sha256'] != sha256_file(frozen_file) or \
            cj['outputs']['training_contrasts.csv']['sha256'] != sha256_file(cdir / 'training_contrasts.csv'):
        raise AssertionError('training_contrasts.json: protocol or CSV hash differs from the files')

    # the family: fourteen members in the frozen order, one Holm adjustment, every statistic from the model rows
    if list(zip(c['dataset'], c['model_b'])) != [(d, m) for d in DATASETS for m in CONTROLS] or \
            list(c['family_index']) != list(range(1, size + 1)) or set(c['model_a']) != {KNN} or \
            set(c['n_folds']) != {n_outer} or set(c['df']) != {FACTS['df']}:
        raise AssertionError('training_contrasts.csv: members, order or folds differ from the frozen family')
    fam = cj['family']
    if (fam['size'], fam['multiplicity'], fam['metric'], fam['n_folds'], fam['df'], fam['test_train_ratio'],
            round(100 * fam['confidence'])) != (size, 'holm', 'accuracy', n_outer, FACTS['df'], FACTS['ratio'], conf):
        raise AssertionError('training_contrasts.json: family settings differ from the captions')
    by_member = {(r['dataset'], r['model_b']): r for r in cj['contrasts']}
    for _, r in c.iterrows():
        j = by_member[(r['dataset'], r['model_b'])]
        if any(abs(float(r[k]) - float(j[k])) > 1e-15 for k in ('mean_difference', 'standard_error', 'ci_low', 'ci_high',
                                                               'p_approximate', 'holm_p_approximate')):
            raise AssertionError(f'{r["dataset"]}/{r["model_b"]}: the contrasts CSV and JSON differ')
    if any(abs(a - b) > 1e-12 for a, b in zip(holm(list(c['p_approximate'])), c['holm_p_approximate'])):
        raise AssertionError('training_contrasts.csv: the Holm column is not one adjustment over the family')
    C = {}
    for _, r in c.iterrows():
        d, m = r['dataset'], r['model_b']
        a, b = fold_accuracy(knn, KNN, d), fold_accuracy(training, m, d)
        if set(a) != set(b):
            raise AssertionError(f'{d}/{m}: the two runs do not cover the same outer folds')
        mean, se, lo, hi, p = corrected_t([float(a[k] - b[k]) for k in sorted(a)], conf)
        exact = sum(a.values()) / len(a) - sum(b.values()) / len(b)
        if abs(float(exact) - r['mean_difference']) > 1e-12 or max(abs(se - r['standard_error']), abs(lo - r['ci_low']),
                                                                  abs(hi - r['ci_high']), abs(p - r['p_approximate'])) > 1e-9:
            raise AssertionError(f'training_contrasts.csv/{d}/{m}: differs from the fold differences of the model rows')
        C[(d, m)] = r
    E = {}
    for d in DATASETS:
        by = {r['model_id']: r for r in et['rows'][d]}
        if sorted(by) != sorted([KNN] + CONTROLS):
            raise AssertionError(f'training_error_table.json/{d}: models {sorted(by)}')
        for m, r in by.items():
            fam_m, run = (knn, 'knn') if m == KNN else (training, 'training')
            s = lookup(fam_m.summary['summaries'][d], m, 'error')
            if r['source_run'] != run or (r['n_folds'], r['seeds_per_fold']) != (s['n_folds'], s['seeds_per_fold']) or \
                    any(abs(r[f] - s[k]) > 1e-12 for f, k in (('mean_error', 'mean'), ('outer_fold_sd', 'outer_fold_sd'),
                                                             ('mean_within_fold_seed_sd', 'mean_within_fold_seed_sd'))):
                raise AssertionError(f'training_error_table.json/{d}/{m}: differs from the summary of the {run} run')
        E[d] = by

    # the depth split: groups by the widths ArrowFlow chose, fold differences from the model rows, same-width counts from
    # the selection records of the untrained ArrowFlow; descriptive
    depths = [tuple(w) for w in tc['depth_split']['depths']]
    if [tuple(w) for w in split['depths']] != depths or f'{KNN} (knn run) minus {UNTRAINED}' not in split['difference']:
        raise AssertionError('training_depth_split.json: depths or difference differ from the frozen protocol')
    chosen = {k: tuple(v['config']['widths']) for k, v in knn_records.items()}
    untrained = selection_records(training, UNTRAINED)
    G = {}
    for d in DATASETS:
        a, b = fold_accuracy(knn, KNN, d), fold_accuracy(training, UNTRAINED, d)
        seen = []
        if [tuple(g['widths']) for g in split['by_dataset'][d]] != depths:
            raise AssertionError(f'training_depth_split.json/{d}: groups differ from the declared depths')
        for g in split['by_dataset'][d]:
            w, folds, diffs = tuple(g['widths']), [tuple(x) for x in g['folds']], g['fold_differences']
            if len(folds) != g['n_folds'] or len(diffs) != len(folds) or any(chosen[(d, *k)] != w for k in folds) or \
                    any(abs(float(a[k] - b[k]) - x) > 1e-12 for k, x in zip(folds, diffs)):
                raise AssertionError(f'training_depth_split.json/{d}/{w}: folds or differences differ from the runs')
            if sum(tuple(untrained[(d, *k)]['config']['widths']) == w for k in folds) != g['untrained_selected_the_same_widths']:
                raise AssertionError(f'training_depth_split.json/{d}/{w}: the same-width count differs from the records')
            if folds:
                if abs(sum(diffs) / len(diffs) - g['mean_difference']) > 1e-12 or min(diffs) != g['min'] or max(diffs) != g['max']:
                    raise AssertionError(f'training_depth_split.json/{d}/{w}: summary statistics differ from its folds')
                if len(folds) > 1:
                    _, _, lo, hi, _ = corrected_t(diffs, conf)
                    iv = g['interval']
                    if abs(statistics.stdev(diffs) - g['sd']) > 1e-12 or abs(lo - iv['ci_low']) > 1e-9 or abs(hi - iv['ci_high']) > 1e-9:
                        raise AssertionError(f'training_depth_split.json/{d}/{w}: SD or interval differs from its folds')
                elif g['interval'] is not None:
                    raise AssertionError(f'training_depth_split.json/{d}/{w}: an interval over one fold')
            seen += folds
            G[(d, w)] = g
        if sorted(seen) != sorted(k[1:] for k in chosen if k[0] == d):
            raise AssertionError(f'training_depth_split.json/{d}: the groups do not partition the outer folds')
    pooled = {tuple(p['widths']): p for p in split['pooled']}
    for w in depths:
        diffs = [x for d in DATASETS for x in G[(d, w)]['fold_differences']]
        p = pooled[w]
        if p['n_dataset_folds'] != len(diffs) or sorted(p['datasets']) != sorted(d for d in DATASETS if G[(d, w)]['n_folds']) \
                or abs(sum(diffs) / len(diffs) - p['mean_difference']) > 1e-12 or abs(statistics.stdev(diffs) - p['sd']) > 1e-12:
            raise AssertionError(f'training_depth_split.json: the pooled {w} row differs from its groups')

    higher = {m: [d for d in DATASETS if C[(d, m)]['mean_difference'] > 0] for m in CONTROLS}
    significant = [k for k in C if C[k]['holm_p_approximate'] < 0.05]
    excludes = [k for k in C if C[k]['ci_low'] > 0 or C[k]['ci_high'] < 0]
    same = {w: sum(G[(d, w)]['untrained_selected_the_same_widths'] for d in DATASETS) for w in depths}
    return dict(C=C, E=E, G=G, pooled=pooled, depths=depths, size=size, higher=higher, significant=significant,
                excludes=excludes, counts=counts, dropped=dropped, same=same, protocol=tp)


def flip_column(j):
    def change(cells, header):
        cells[j] = '+' + cells[j][3:] if cells[j].startswith('$-$') else '$-$' + cells[j].lstrip('+')
        return cells
    return change


def production_log(runs):
    """Stage times of the production log of the training controls and the component ablation, and the full command lines
    and pinned commit of its script, with the script's variables replaced by the placeholders that Section S5 prints."""
    text = (runs / 'task20-run.log').read_text()
    start = re.search(r'^task20 start (\S+) HEAD (\w+)$', text, re.M)
    end = re.search(r'^task20 end (\S+) OK$', text, re.M)
    status = re.search(r'^task20 exit=(\d+)$', text, re.M)
    script = (PRODUCTION_SCRIPTS / 'task20-prod.sh').read_text()
    pinned = re.search(r'test "\$\(git rev-parse HEAD\)" = (\w+)', script)
    variables = dict(re.findall(r'^([A-Z]+)=([^$"\s]\S*)$', script, re.M))
    if not (start and pinned) or any(name not in variables for name in ('TP', 'AP', 'REG')):
        raise AssertionError('task20 log or script: no start line, pinned commit or protocol and registry variables')
    shown = lambda path: 'P/' + path.split('experiments/make_revision/protocols/', 1)[1]
    replace = {'$DATASETS': ' '.join(read_json(REPO / variables['TP'])['datasets']), '$TP': shown(variables['TP']),
               '$AP': shown(variables['AP']), '$REG': variables['REG'], '"$OUTA"': '<run>', '"$OUTB"': '<run>',
               '"$K"': '<ArrowFlow run>', '"$OUTA/compare_training"': '<run>/compare_training'}
    commands = defaultdict(list)
    for family, module, args in re.findall(r'^echo "== (20[AB]) [^"]*"; "\$PY" -m (experiments\.make_revision\.\w+) (.*)$',
                                           script, re.M):
        tokens = tuple(token for word in args.split() for token in replace.get(word, word).split())
        if any('$' in token for token in tokens):
            raise AssertionError(f'task20 script: an unresolved variable in {module} {args}')
        commands[family].append((module, tokens))
    return dict(start=start.group(1), head=start.group(2), pinned=pinned.group(1),
                stages=dict(re.findall(r'^== (20[AB] [\w-]+) (\d\d:\d\d:\d\d)Z$', text, re.M)),
                end=end.group(1) if end else None, exit=int(status.group(1)) if status else None, commands=commands,
                reproduced={d: (int(a), int(b)) for d, a, b in
                            re.findall(r'^(\w+): views7 reproduced (\d+)/(\d+) fold-seeds$', text, re.M)})


def verbatim_commands(text, heading):
    """(module, argument tokens) of every command under one '# heading' line of the verbatim block in S5, in order."""
    block = text.split('\\begin{verbatim}', 1)[1].split('\\end{verbatim}', 1)[0]
    sections = re.split(r'^# ', block, flags=re.M)
    body = [s for s in sections if s.startswith(heading)]
    if len(body) != 1:
        raise AssertionError(f'S5 commands: {len(body)} sections headed {heading!r}')
    out = []
    for command in re.sub(r'\\\n\s*', ' ', body[0]).splitlines()[1:]:
        m = re.match(r'python -m (experiments\.make_revision\.\w+) (.*)$', command.strip())
        if m:
            out.append((m.group(1), tuple(m.group(2).split())))
    return out


# ----------------------------------------------------------------------------- component ablation of ArrowFlow (Section 7.4, S4)
KNN_VARIANTS = ['views1', 'views3', 'no_checkpoint', 'no_augment', 'prototype_readout', 'untrained', 'input_knn']


def verify_ablation(name, s, p, table, datasets, reference_error, records, reproduced=None):
    """Checks shared by the component ablations of ArrowFlow on the benchmark and on the further datasets. On every dataset the
    seven-view refit reproduces ArrowFlow on every fold and seed, the resolved configurations are ArrowFlow's selections, a variant
    coincides with the seven-view fit only without augmentation and only where the adaptive rule switches it off on every fold,
    the CSV summary equals the JSON summary, the seven-view errors equal ArrowFlow's rows of its reference run, every change
    equals the corrected resampled t over the fold differences of the model rows, and the depth groups partition the outer folds
    by the selected widths. Returns the coinciding variants, the fold accuracies, the depth groups and the depths."""
    conf, pairs, n_outer = FACTS['confidence'], FACTS['n_outer'] * FACTS['n_seeds'], FACTS['n_outer']
    S = s['summaries']
    if len(table) != len(datasets) * len(p['variants']) * len(p['report_metrics']):
        raise AssertionError(f'{name}: unexpected number of rows in the CSV summary')
    identical, fold = {}, {}
    for d in datasets:
        rep = S[d]['views7_reproduces_reference']
        if (rep['matching_fold_seeds'], rep['total_fold_seeds']) != (pairs, pairs) or \
                (reproduced is not None and reproduced.get(d) != (pairs, pairs)):
            raise AssertionError(f'{name}/{d}: the seven-view refit did not reproduce ArrowFlow on every fold and seed')
        rcs = S[d]['resolved_configurations']
        if len(rcs) != n_outer or any(rc['config_id'] != records[(d, rc['outer_repeat'], rc['outer_fold'])]['config_id']
                                      for rc in rcs):
            raise AssertionError(f'{name}/{d}: the configurations are not ArrowFlow\'s selections')
        identical[d] = {v for rc in rcs for v, src in rc['fit_sources'].items() if src == 'identical_to_views7'}
        if identical[d] - {'no_augment'} or (identical[d] and any(rc['fit_sources']['no_augment'] != 'identical_to_views7'
                                                                  or rc['augment'] for rc in rcs)):
            raise AssertionError(f'{name}/{d}: a variant coincides with the seven-view fit on some folds only')
        for v in p['variants']:
            for metric in p['report_metrics']:
                j = S[d]['variants'][v]['metrics'][metric]
                row = table[(table['dataset_id'] == d) & (table['variant_id'] == v) & (table['metric'] == metric)]
                if len(row) != 1 or any(abs(float(row.iloc[0][k]) - j[k]) > 1e-12 for k in ('mean', 'outer_fold_sd', 'mean_within_fold_seed_sd')):
                    raise AssertionError(f'{name}/{d}/{v}/{metric}: the CSV and JSON summaries differ')
        a, b = S[d]['variants']['views7']['metrics']['error'], reference_error(d)
        if any(abs(a[k] - b[k]) > 1e-12 for k in ('mean', 'outer_fold_sd', 'mean_within_fold_seed_sd')):
            raise AssertionError(f'{name}/{d}: the seven-view errors differ from ArrowFlow\'s rows of its reference run')
        per = defaultdict(lambda: defaultdict(list))
        for r in s['model_rows'][d]:
            per[r['model_id']][(r['outer_repeat'], r['outer_fold'])].append(r['accuracy'])
        if sorted(per) != sorted(p['variants']) or any(len(per[v]) != n_outer or any(len(x) != FACTS['n_seeds'] for x in per[v].values())
                                                       for v in per):
            raise AssertionError(f'{name}/{d}: the model rows do not cover every variant, fold and seed')
        fold[d] = {v: {k: sum(x) / len(x) for k, x in per[v].items()} for v in per}
        for v in KNN_VARIANTS:
            diffs = [fold[d][v][k] - fold[d]['views7'][k] for k in sorted(fold[d]['views7'])]
            ch = S[d]['variants'][v]['change_from_views7']['accuracy']
            if v in identical[d]:
                if any(x != 0 for x in diffs) or ch['mean_difference'] != 0:
                    raise AssertionError(f'{name}/{d}/{v}: an identical variant differs from the seven-view fit')
                continue
            mean, _, lo, hi, _ = corrected_t(diffs, conf)
            if max(abs(mean - ch['mean_difference']), abs(lo - ch['ci_low']), abs(hi - ch['ci_high'])) > 1e-9:
                raise AssertionError(f'{name}/{d}/{v}: the change differs from the fold differences of the model rows')
    depths = [tuple(w) for w in p['depth_split']['depths']]
    split = s['depth_split']
    if [tuple(w) for w in split['depths']] != depths or not split['difference'].startswith('untrained minus views7'):
        raise AssertionError(f'{name}: the depth split of the summary differs from the protocol')
    G = {}
    for d in datasets:
        seen = []
        for g in split['by_dataset'][d]:
            w, folds, diffs = tuple(g['widths']), [tuple(x) for x in g['folds']], g['fold_differences']
            if any(tuple(records[(d, *k)]['config']['widths']) != w for k in folds) or len(diffs) != len(folds) != g['n_folds'] or \
                    any(abs(fold[d]['untrained'][k] - fold[d]['views7'][k] - x) > 1e-12 for k, x in zip(folds, diffs)):
                raise AssertionError(f'{name}/{d}/{w}: the depth group differs from the model rows or the selections')
            if folds:
                if abs(sum(diffs) / len(diffs) - g['mean_difference']) > 1e-12:
                    raise AssertionError(f'{name}/{d}/{w}: the group mean differs from its folds')
                if len(folds) > 1:
                    _, _, lo, hi, _ = corrected_t(diffs, conf)
                    if abs(lo - g['interval']['ci_low']) > 1e-9 or abs(hi - g['interval']['ci_high']) > 1e-9:
                        raise AssertionError(f'{name}/{d}/{w}: the group interval differs from its folds')
            seen += folds
            G[(d, w)] = g
        if sorted(seen) != sorted(fold[d]['views7']):
            raise AssertionError(f'{name}/{d}: the depth groups do not partition the outer folds')
    return identical, fold, G, depths


def render_knn_ablation(kab, knn, knn_records, production):
    """Component ablation of ArrowFlow at its reconstructed per-fold selections on the benchmark datasets, verified from its summary
    against the ArrowFlow run and the production log (verify_ablation). render_components sets it beside the ablation of the further
    datasets in Table 4 and Section S4. Changes are ArrowFlow minus the variant, so the untrained rows are trained minus untrained;
    all of it is descriptive."""
    s, p = kab.summary, kab.protocol
    frozen = PROTOCOLS_V3 / 'knn_ablation.json'
    if sha256_file(frozen) != sha256_file(kab.dir / 'protocol.json') or p.get('frozen') is not True:
        raise AssertionError('knn_ablation: the protocol of the run is not the frozen protocol file')
    if s['protocol_id'] != p['protocol_id'] or s['code_revision'] != kab.environment['code_revision'] or \
            s['inferential_significance_claims'] is not False:
        raise AssertionError('knn_ablation_summary.json: protocol, revision or descriptive status differs')
    ref = s['reference_source']
    if (ref['protocol_id'], ref['code_revision'], ref['summary_sha256'], ref['model_id']) != \
            (knn.protocol['protocol_id'], knn.environment['code_revision'], sha256_file(knn.dir / 'summary.json'), KNN):
        raise AssertionError('knn_ablation_summary.json: the reference is not the ArrowFlow run on disk')
    if p['variants'] != ['views7'] + KNN_VARIANTS or p['datasets'] != DATASETS or p['fit_seeds'] != knn.protocol['fit_seeds']:
        raise AssertionError('knn_ablation protocol: variants, datasets or seeds differ from the tables')
    if production['exit'] != 0 or not production['end']:
        raise AssertionError('task20 log: the production script did not end with exit status 0')
    identical, fold, G, depths = verify_ablation(kab.name, s, p, pd.read_csv(kab.dir / 'knn_ablation_summary.csv'), DATASETS,
                                                 lambda d: lookup(knn.summary['summaries'][d], KNN, 'error'), knn_records,
                                                 production['reproduced'])
    gain = {(d, v): None if v in identical[d] else -s['summaries'][d]['variants'][v]['change_from_views7']['accuracy']['mean_difference']
            for d in DATASETS for v in KNN_VARIANTS}
    return dict(identical=identical, gain=gain, G=G, depths=depths, pairs=FACTS['n_outer'] * FACTS['n_seeds'], protocol=p, fold=fold)


def render_matched(matched, contrasts_csv):
    S = matched.summary['summaries']
    p = matched.protocol
    if p.get('primary_datasets') != DATASETS:
        raise AssertionError('matched protocol: the datasets it reports are not the seven benchmark datasets')
    arch_ids = [('af_h128', '$[128]$', 1), ('af_h64_128', '$[64,128]$', 2), ('af_h64_32', '$[64,32]$', 2)]
    rows, blocks = [], []
    for d in DATASETS:
        blocks.append(len(rows))
        for i, (aid, lab, depth) in enumerate(arch_ids):
            out = lookup(S[d], f'{aid}_output', 'error')
            t1, u1 = lookup(S[d], f'{aid}_d1_trained', 'error'), lookup(S[d], f'{aid}_d1_untrained', 'error')
            tl, ul = lookup(S[d], f'{aid}_d{depth}_trained', 'error'), lookup(S[d], f'{aid}_d{depth}_untrained', 'error')
            rows.append([LABEL[d] if i == 0 else '', lab, pm(out['mean'], out['outer_fold_sd']),
                         pm(t1['mean'], t1['outer_fold_sd']), pm(u1['mean'], u1['outer_fold_sd']),
                         pm(tl['mean'], tl['outer_fold_sd']) if depth > 1 else '(= layer 1)',
                         pm(ul['mean'], ul['outer_fold_sd']) if depth > 1 else '(= layer 1)'])
    caption = (f'\\textbf{{Matched learning controls at every architecture (single view).}} Mean outer-fold error in '
               f'percent (outer-fold SD) on the {words(len(DATASETS))} benchmark datasets for the network\'s own prototype readout '
               f'and for footrule kNN probes, tuned symmetrically on inner folds. The probes are applied to the hidden ranking of layer 1 '
               f'and of the last hidden layer, of the trained network and of the same network before training (initial '
               f'filters). One view per dataset, with learning '
               f'rate {p.get("learning_rate")}, no augmentation and '
               f'{p.get("iterations")} iterations of batch size {p.get("batch_size")}. {matched_encoder_sentence(matched)} The validation fraction is {p.get("validation_ratio")} (the network trains on the remaining rows of each partition; probes and reference classifiers use every row). Protocol '
               f'\\texttt{{{tex(p.get("protocol_id"))}}}.')
    write_table('tab_s_matched_complete', caption, 'tab:s-matched',
                ['Dataset', 'Widths', 'Prototype readout', 'Probe, trained L1', 'Probe, initial L1',
                 'Probe, trained last', 'Probe, initial last'], rows, 'llrrrrr', block_rules=blocks, size='\\scriptsize', wide=True)
    rows = []
    for d in DATASETS:
        cells = [LABEL[d]]
        for m in ('input_footrule', 'borda', 'hdc'):
            r = lookup(S[d], m, 'error')
            cells.append(pm(r['mean'], r['outer_fold_sd']))
        rows.append(cells)
    lowest = [d for d in DATASETS if lookup(S[d], 'input_footrule', 'error')['mean']
              < min(lookup(S[d], m, 'error')['mean'] for m in ('borda', 'hdc'))]
    lowest_sentence = ('Footrule kNN has the lowest error of the three on every dataset.' if len(lowest) == len(DATASETS)
                       else f'Footrule kNN has the lowest error of the three on {num_word(len(lowest))} of the '
                            f'{words(len(DATASETS))} datasets.')
    caption = (f'\\textbf{{Reference classifiers on the same single encoded view.}} Mean outer-fold error in percent '
               f'(outer-fold SD) of the reference classifiers, each receiving the same encoded view as the networks of '
               f'Table~\\ref{{tab:s-matched}}. The first is fixed-configuration input footrule kNN: footrule kNN on the '
               f'input rankings of this encoded view, with neighbors and weights tuned on inner folds. The others are '
               f'the mean-position prototype classifier (one Borda prototype per class, nearest by footrule) and an '
               f'ordered-position hyperdimensional computing (HDC) control. The HDC control has dimension '
               f'{listing([f"{int(v):,}" for v in p.get("hdc_dimensions", [])])}, chosen on inner folds. '
               f'{lowest_sentence}')
    write_table('tab_s_matched_controls', caption, 'tab:s-matched-controls',
                ['Dataset', 'Fixed-configuration input footrule kNN', 'Borda prototype', 'HDC'], rows, 'lrrr')
    two = [aid for aid, _, depth in arch_ids if depth == 2]
    trained = lambda d, aid, layer: lookup(S[d], f'{aid}_d{layer}_trained', 'error')['mean']
    worse = [d for d in DATASETS if all(trained(d, a, 2) > trained(d, a, 1) for a in two)]
    better = [d for d in DATASETS if all(trained(d, a, 2) < trained(d, a, 1) for a in two)]
    if len(worse) + len(better) != len(DATASETS) or not worse or not better:
        raise AssertionError('the text says the probe on the trained second layer has higher error than on the first at '
                             'both two-layer architectures or lower error at both')
    family, declared = matched.summary.get('primary_contrasts') or [], p.get('primary_tabular_comparisons')
    if matched.summary.get('complete_primary_family') is not True or matched.summary.get('holm_applied') is not True or \
            len(family) != declared or declared != 2 * len(DATASETS):
        raise AssertionError('matched summary: its declared contrast family is incomplete or not Holm-adjusted')
    if any(abs(a - r['holm_p_approximate']) > 1e-12 for a, r in zip(holm([r['p_approximate'] for r in family]), family)):
        raise AssertionError('matched summary: the Holm column is not one adjustment over the declared family')
    probes = pd.read_csv(contrasts_csv)
    probes = probes[probes['kind'] == 'trained_vs_initial'].set_index('dataset_id')
    kinds = [(('af_h128_d1_trained', 'af_h128_d1_untrained'), 'Trained $-$ initial probe'),
             (('af_h128_output', 'input_footrule'), 'Prototype readout $-$ fixed-configuration input footrule kNN')]
    frows, fblocks, significant = [], [], []
    for i, d in enumerate(DATASETS):
        fblocks.append(len(frows))
        for j, ((a, b), label) in enumerate(kinds):
            r = family[2 * i + j]
            if (r['dataset_id'], r['model_a'], r['model_b'], r['n_folds'], r['df']) != (d, a, b, FACTS['n_outer'], FACTS['df']):
                raise AssertionError(f'matched summary: family member {2 * i + j} is not {d} {a} minus {b}')
            if j == 0 and any(abs(r[x] - float(probes.loc[d, x])) > 1e-12 for x in ('mean_difference', 'ci_low', 'ci_high')):
                raise AssertionError(f'matched summary/{d}: the probe contrast differs from the benchmark family')
            if r['holm_p_approximate'] < 0.05:
                significant.append((j, r['mean_difference']))
            frows.append([LABEL[d] if j == 0 else '', label, signed(r['mean_difference']), interval(r['ci_low'], r['ci_high']),
                          pval(r['p_approximate']), pval(r['holm_p_approximate'])])
    if not significant or any(j != 1 or md >= 0 for j, md in significant):
        raise AssertionError('the caption says every significant member is a prototype readout below input footrule kNN')
    arch = widths_label(p['architectures'][p['primary_architecture_index']])
    caption = (f'\\textbf{{Contrast family of the matched protocol.}} The matched protocol declared these {words(len(family))} '
               f'contrasts, two per dataset, with one Holm adjustment over them. The first is the footrule kNN probe on the '
               f'trained hidden ranking of the single-view {arch} network minus the same probe on its initial filters; the '
               f'second is that network\'s prototype readout minus fixed-configuration input footrule kNN. Differences are in '
               f'percentage points with the {interval_note()}; $p$ values are approximate. The probe contrasts equal those of '
               f'Table~\\ref{{tab:s-contrasts}}, whose family adjusts them together with the seven-view control instead. This '
               f'family is reported for completeness, and no claim of this paper rests on it. '
               f'{of_total(len(significant), words(len(family)), "contrasts", "remains", "remain")} significant after Holm '
               f'adjustment, each a prototype readout below fixed-configuration input footrule kNN.')
    write_table('tab_s_matched_family', caption, 'tab:s-matched-family',
                ['Dataset', 'Contrast', 'Difference (pp)', f'{FACTS["confidence"]}\\% interval', '$p$', 'Holm $p$'], frows,
                'llrrrr', block_rules=fblocks, size='\\scriptsize', wide=True)
    return dict(worse=worse, better=better, family_size=len(family), datasets=list(p['primary_datasets']))


# ----------------------------------------------------------------------------- components (ablation)
def render_ablation(ablation):
    S = ablation.summary['summaries']
    identical = {d: {v for rc in S[d]['resolved_configurations'] for v, src in rc['fit_sources'].items()
                     if src == 'identical_to_views7'} for d in DATASETS}
    metric = lambda d, v, m='error': S[d]['variants'][v]['metrics'][m]
    change = lambda d, v: S[d]['variants'][v]['change_from_views7']['accuracy']
    NA = 'n/a'

    def change_rows(order):
        rows = []
        for v in order:
            if v == 'views7':
                continue
            cells = [VARIANT[v]]
            for d in DATASETS:
                if v in identical[d]:
                    cells.append(NA)
                else:
                    c = change(d, v)
                    cells.append(f"{signed(c['mean_difference'])} {interval(c['ci_low'], c['ci_high'])}")
            rows.append(cells)
        return rows

    v1 = {d: change(d, 'views1')['mean_difference'] for d in DATASETS}
    views_help = [d for d in DATASETS if v1[d] < 0]
    borda_max = max(abs(change(d, 'borda_views7')['mean_difference']) for d in DATASETS)
    ck = {d: change(d, 'no_checkpoint')['mean_difference'] for d in DATASETS}
    ck_costs = [d for d in DATASETS if ck[d] > 0]
    trails = [d for d in DATASETS if metric(d, 'views7', 'accuracy')['mean'] < metric(d, 'multiview_footrule_knn', 'accuracy')['mean']]
    heads = ['Variant'] + [LABEL[d] for d in DATASETS]
    rows_a = []
    for v in ABLATION_ORDER:
        rows_a.append([VARIANT[v]] + [pm(metric(d, v)['mean'], metric(d, v)['outer_fold_sd']) if v not in identical[d]
                                      else NA for d in DATASETS])
    rows_b = [[VARIANT[v]] + [NA if v in identical[d] else signed(change(d, v)['mean_difference']) for d in DATASETS]
              for v in ABLATION_ORDER if v != 'views7']
    caption = (f'\\textbf{{Component ablation with the prototype readout.}} Upper block: mean outer-fold '
               f'error in percent (outer-fold SD) of each variant on the {words(len(DATASETS))} benchmark datasets. Each variant is evaluated '
               f'on the same {FACTS["n_outer"]} outer folds and {words(FACTS["n_seeds"])} fitting seeds with the '
               f'configuration the prototype readout chose on the inner folds of that fold. Read from the seven-view fit, one and three views are the first '
               f'one and three of the {words(FACTS["views"])} fitted views under majority vote, and the Borda row '
               f're-aggregates the same class rankings. The checkpoint and augmentation rows refit without the validation '
               f'checkpoint or without augmentation. The seven-view footrule kNN control is footrule kNN on the same encoded '
               f'views (neighbors and weights tuned on inner folds), combined by majority vote. Lower block: accuracy change '
               f'from the prototype readout with seven views '
               f'in percentage points, positive when the variant is more accurate; descriptive, with the corrected '
               f'resampled-$t$ intervals in Table~\\ref{{tab:s-ablation-changes}}. n/a: the adaptive rule '
               f'already switches augmentation off on that dataset, so the variant coincides with the prototype readout. Seven '
               f'views lower the error against one view on {num_word(len(views_help))} datasets; Borda and majority '
               f'aggregation differ by at most {100 * borda_max:.1f} points; removing the checkpoint raises accuracy on '
               f'{num_word(len(ck_costs))} datasets; the prototype readout trails the kNN control on {num_word(len(trails))}.')
    lines = ['% Rendered by render_tables.py from run files; do not edit by hand.', '\\begin{table}[!htbp]',
             f'\\caption{{{caption}}}', '\\label{tab:components}', '\\centering', '\\scriptsize',
             '\\setlength{\\tabcolsep}{3pt}',
             '\\begin{tabular}{l' + 'r' * len(DATASETS) + '}', '\\toprule',
             ' & '.join(heads) + ' \\\\', '\\midrule',
             f'\\multicolumn{{{len(heads)}}}{{l}}{{\\emph{{Error, \\% (outer-fold SD)}}}} \\\\']
    lines += [' & '.join(r) + ' \\\\' for r in rows_a]
    lines += ['\\midrule', f'\\multicolumn{{{len(heads)}}}{{l}}{{\\emph{{Accuracy change from the prototype readout, pp}}}} \\\\']
    lines += [' & '.join(r) + ' \\\\' for r in rows_b]
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    text = '\n'.join(lines) + '\n'
    (TABLES / 'tab_components.tex').write_text(text)
    WRITTEN['tab_components'] = text
    print(f'wrote {TABLES / "tab_components.tex"}  ({len(rows_a) + len(rows_b)} rows)')
    rows = []
    for v in ABLATION_S4:
        cells = [VARIANT[v]]
        for d in DATASETS:
            if v in identical[d]:
                cells.append(NA)
            else:
                m = metric(d, v)
                cells.append(f"{pm(m['mean'], m['outer_fold_sd'])} [{pct(m['mean_within_fold_seed_sd'])}]")
        rows.append(cells)
    caption = (f'\\textbf{{Complete component ablation.}} Mean outer-fold error in percent (outer-fold SD) [mean '
               f'within-fold SD across the {words(FACTS["n_seeds"])} fitting seeds] of every variant of the ablation '
               f'protocol \\texttt{{{tex(ablation.protocol["protocol_id"])}}}, including the single-view variant without '
               f'checkpoint and augmentation. Variants inherit the configuration the prototype readout chose on the inner folds '
               f'of each outer fold; n/a marks variants identical to the prototype readout because the adaptive rule already '
               f'resolves augmentation to off.')
    write_table('tab_s_ablation_complete', caption, 'tab:s-ablation', heads, rows, 'l' + 'r' * len(DATASETS),
                size='\\scriptsize', wide=True, stack=4)
    caption = (f'\\textbf{{Accuracy change of every variant from the prototype readout.}} Percentage points with the '
               f'{interval_note()}, fitting seeds averaged within fold; descriptive, without $p$ values. The '
               f'prototype-readout contrasts of Table~\\ref{{tab:contrasts}}, fixed in the benchmark protocol before its outer folds were scored, carry the only tests of this study. n/a as in Table~\\ref{{tab:s-ablation}}.')
    write_table('tab_s_ablation_changes', caption, 'tab:s-ablation-changes', heads, change_rows(ABLATION_S4),
                'l' + 'r' * len(DATASETS), size='\\scriptsize', wide=True, stack=4)
    rows = []
    for d in DATASETS:
        rep = S[d]['views7_reproduces_bridge']
        sel = Counter((s['config']['n_neighbors'], s['config']['weights']) for s in S[d]['knn_selection'])
        rows.append([LABEL[d], f"{rep['matching_fold_seeds']} of {rep['total_fold_seeds']}",
                     ', '.join(f'{k} {w}: {n}' for (k, w), n in sel.most_common())])
    caption = (f'\\textbf{{Ablation diagnostics.}} Number of fold--seed pairs ({FACTS["n_outer"]} folds $\\times$ '
               f'{FACTS["n_seeds"]} seeds) on which the refit of the prototype readout reproduced the first benchmark\'s outer '
               f'predictions exactly. The last column gives the neighbor count and weighting that the seven-view footrule kNN control chose on '
               f'the inner folds, with the number of outer folds choosing each.')
    write_table('tab_s_ablation_knn', caption, 'tab:s-ablation-knn',
                ['Dataset', 'Refit reproduces benchmark', 'Seven-view footrule kNN control: neighbors, weights (folds)'],
                rows, 'llp{0.45\\textwidth}')
    return dict(views_help=views_help, borda_max=borda_max, ck_costs=ck_costs, trails=trails)


def ablation_levels_for_figure7(ablation):
    """The two ablation models of Figure 7 in the display-summary shape ``figure7`` reads."""
    S = ablation.summary['summaries']
    out = {'summaries': {}}
    for d in DATASETS:
        rows = []
        for v in ('views7', 'multiview_footrule_knn'):
            m = S[d]['variants'][v]['metrics']['error']
            rows.append({'model_id': v, 'metric': 'error', 'mean': m['mean'], 'outer_fold_sd': m['outer_fold_sd']})
        out['summaries'][d] = rows
    return out


# ----------------------------------------------------------------------------- datasets, protocols, environment (S5)
def copy_label(m):
    source = str(m.get('source', ''))
    if source.startswith('sklearn'):
        return 'scikit-learn, \\texttt{' + tex(source.split()[-1]) + '}'
    if source.startswith('OpenML'):
        return 'OpenML, data ID ' + source.split('=')[-1]
    raise AssertionError(f'dataset source {source!r}: expected a scikit-learn loader or an OpenML copy')


def render_datasets(bridge, newdata):
    """Table S-datasets: the benchmark datasets from their prepared manifests and the further datasets from the pins of the batch
    protocols, which render_newdata has checked against the prepared manifests."""
    rows = []
    dims = lambda m: (f"{int(m['shape'][0]):,}", f"{int(m['shape'][1]):,}", str(len(m['class_counts'])))
    for d in DATASETS:
        m = read_json(bridge.dir / d / 'manifest.json')
        rows.append([LABEL[d], *dims(m), copy_label(m), f"\\texttt{{{m['dataset_hash'][:HASH_DIGITS]}}}"])
    for d in NEWDATA:
        e = newdata['pin'][d]
        rows.append([NEW_LABEL[d], *dims(e), f"OpenML, data ID {e['data_id']}, version {e['version']}",
                     f"\\texttt{{{e['dataset_hash'][:HASH_DIGITS]}}}"])
    caption = (f'\\textbf{{Datasets.}} Rows, features, classes, the evaluated copy (a scikit-learn loader, or an OpenML data '
               f'identifier, with the pinned version for the further datasets) and the first {words(HASH_DIGITS)} hexadecimal digits '
               f'of the content hash recorded by the harness at preparation. The batch protocols also pin the checksums of the OpenML '
               f'files of the further datasets (Table~\\ref{{tab:s-newdata-evidence}}). HCV: hepatitis C virus.')
    write_table('tab_s_datasets', caption, 'tab:s-datasets', ['Dataset', 'Rows', 'Features', 'Classes', 'Copy', 'Hash'],
                rows, 'lrrrll', size='\\scriptsize', subheads={0: GROUP_BLOCK['benchmark'], len(DATASETS): GROUP_BLOCK['further']})
    return rows


def render_protocols(families, devlab_dir=None, extra_rows=()):
    """Table S-protocols. The Results column of every row counts the result files of the datasets reported in this paper, out of those
    planned for them. A family may name those datasets: the matched run also holds result files of a dataset that this paper does not
    report, so its counts are restricted to the benchmark datasets, and every other family must plan jobs on reported datasets only."""
    rows = []
    for fam, label, pfile, *reported in families:
        p = fam.protocol or {}
        state = 'complete' if fam.summary else ('running' if fam.completed else 'pending')
        planned, completed = fam.planned, fam.completed
        if reported:
            jobs = read_json(fam.dir / 'planned_jobs.json')
            planned = sum(job['dataset_id'] in DATASETS for job in jobs)
            completed = sum(1 for d in DATASETS for f in (fam.dir / 'results').glob(f'{d}__r*f*.json') if 'fit' not in f.name)
            if reported != [DATASETS] or planned != len(DATASETS) * FACTS['n_outer'] or completed != planned:
                raise AssertionError(f'{fam.name}: the result files of the benchmark datasets are not complete')
        elif fam.planned is not None and any(job['dataset_id'] not in COMBINED for job in read_json(fam.dir / 'planned_jobs.json')):
            raise AssertionError(f'{fam.name}: jobs planned on a dataset that this paper does not report, but the Results column counts '
                                 f'the result files of the reported datasets')
        rows.append([label, f"\\texttt{{{tex(p.get('protocol_id', '--'))}}}", tex(pfile),
                     tex(str(p.get('frozen_at_utc', '--'))[:16].replace('T', ' ')),
                     f"\\texttt{{{fam.environment.get('code_revision', '--')[:HASH_DIGITS]}}}",
                     str(p.get('wallclock_cap_hours', '--')), f"{completed} / {planned}" if planned else '--', state])
    if devlab_dir and (Path(devlab_dir) / 'readouts_summary.json').is_file():
        for fam, label in (('readouts', 'laboratory: readouts (inner folds)'), ('permlvq', 'laboratory: permutation LVQ (inner folds)')):
            d = read_json(Path(devlab_dir) / f'{fam}_summary.json')
            p = read_json(Path(devlab_dir) / 'protocol.json')
            if not set(d['mean_inner_accuracy']) <= set(DATASETS):
                raise AssertionError(f'laboratory {fam}: units on a dataset that this paper does not report')
            cap = p['families'][fam].get('wallclock_cap_hours', p.get('wallclock_cap_hours', '--'))
            rows.append([label, f"\\texttt{{{tex(d['protocol_id'])}}}", '2026-09-12/devlab.json', 'not frozen (no outer score)',
                         f"\\texttt{{{d['code_revision'][:HASH_DIGITS]}}}", str(cap),
                         f"{d['units']['completed']} / {d['units']['planned']}", 'complete'])
    rows += list(extra_rows)
    caption = ('\\textbf{Protocols and code revisions.} For every evidence family: the protocol identifier, its file under '
               '\\texttt{experiments/make\\_revision/protocols/}, the freeze time in Coordinated Universal Time (UTC), and the git revision recorded in the '
               'run\'s \\texttt{environment.json} or laboratory summary. The two studies of the learned encoder fixed their designs in '
               'the header of their module under \\texttt{experiments/}; their freeze time is that of the commit that fixed the design, '
               'and each ran at that commit. The remaining columns give the wall-clock cap in hours, the result '
               'files (laboratory: inner-fold units) completed for the datasets reported here, out of those planned for them, and the state '
               'at rendering. A run whose training-only pilot projected more than its cap was not started. The laboratory protocol is a '
               'development protocol that scores no outer test fold, and the mechanism analysis fits no model, so neither '
               'has a cap or a job count.')
    write_table('tab_s_protocols', caption, 'tab:s-protocols',
                ['Family', 'Protocol', 'File', 'Frozen', 'Revision', 'Cap (h)', 'Results', 'State'], rows, 'llllllrl',
                size='\\scriptsize', wide=True, rotate=True)


def controlled_protocol_rows(runs):
    """Table S-protocols rows for the eight controlled experiments of Sections 7.5 and 7.6 and the duplicate-free rerun, whose
    protocol stays in the record although its tables left the supplement on 2026-09-23. Each fitted family is read from
    its own run directory: the frozen protocol it was launched with, the code revision of its environment, and the number
    of completed per-job artifacts out of the planned jobs. The mechanism analysis fits nothing and has no protocol. The
    two families of the final round, depth with the corrected relay and the representation test, have protocols of
    2026-09-23."""
    families = [
        ('matched motion-signal controls', 'motion_controls.json', runs / MOTION / 'run', 'logs', '*.jsonl'),
        ('training diagnostics', 'training_diagnostics.json', runs / '2026-09-14-training-diagnostics', 'artifacts', '*.npz'),
        ('fixed-configuration depth', 'depth.json', runs / DEPTH_AGGREGATION / 'depth' / 'run', 'artifacts', '*.npz'),
        ('depth with the corrected relay', 'signed_relay_depth.json', runs / SIGNED_RELAY / 'run', 'artifacts', '*.npz'),
        ('aggregation inside the update', 'aggregation.json', runs / DEPTH_AGGREGATION / 'aggregation' / 'run', 'artifacts', '*.npz'),
        ('duplicate-free rerun', 'dedup.json', runs / DEDUP / 'run', 'results', '*__r*f*.json'),
        ('neighbor baselines', 'neighbour_baselines.json', runs / BASELINES / 'run', 'results', '*__r*f*.json'),
        ('representation test', 'representation_test.json', runs / REPRESENTATION / 'run', 'results', '*__r*f*.json'),
    ]
    rows = []
    for label, pfile, directory, subdir, pattern in families:
        folder = PROTOCOLS_23 if (PROTOCOLS_23 / pfile).is_file() else PROTOCOLS_G5
        frozen = folder / pfile
        protocol, environment = read_json(directory / 'protocol.json'), read_json(directory / 'environment.json')
        if sha256_file(frozen) != sha256_file(directory / 'protocol.json') or protocol.get('frozen') is not True:
            raise AssertionError(f'{label}: the run did not execute the frozen protocol {pfile}')
        planned = len(read_json(directory / 'planned_jobs.json'))
        completed = len([f for f in (directory / subdir).glob(pattern) if 'fits' not in f.name])
        if completed != planned:
            raise AssertionError(f'{label}: {completed} completed job records against {planned} planned')
        rows.append([label, f"\\texttt{{{tex(protocol['protocol_id'])}}}", tex(f'{folder.name}/{pfile}'),
                     tex(str(protocol['frozen_at_utc'])[:16].replace('T', ' ')),
                     f"\\texttt{{{environment['code_revision'][:HASH_DIGITS]}}}",
                     str(protocol.get('wallclock_cap_hours', '--')), f'{completed} / {planned}', 'complete'])
    mechanism = read_json(runs / MECHANISM / 'mechanism_analysis.json')
    if 'no model fitted' not in str(mechanism.get('status')):
        raise AssertionError('the mechanism analysis record no longer says that no model was fitted')
    rows.append(['mechanism analysis (no model is fitted)', '--', '--', '--',
                 f"\\texttt{{{mechanism['provenance']['code_revision'][:HASH_DIGITS]}}}", '--', '--', 'complete'])
    return rows


ENVIRONMENT_KEYS = ('python', 'numpy', 'scipy', 'sklearn', 'torch', 'platform', 'numeric_threads')


def render_environment(env, others, runs):
    """Software versions of the nested benchmark. `others` holds (family, environment.json contents) for every other
    v3 family on disk; the caption's 'every other v3 family records the same versions' is asserted against them. The
    referee-panel revision of 2026-09-25 (item H9) adds the two studies of the learned encoder, which write no environment.json
    and ran outside the harness lock, and the processor: the harness builds every ArrowFlow network with device='cpu', its
    ranking layers use a graphics processor only for another device, and the learned-encoder runs hide it, although the
    recorded PyTorch build supports one."""
    if not env:
        raise AssertionError('tab_s_environment: the nested benchmark has no environment.json')
    for family, other in others:
        differ = [k for k in ENVIRONMENT_KEYS if other.get(k) != env.get(k)] if other else ['environment.json']
        if differ:
            raise AssertionError(f'tab_s_environment: {family} differs from the nested benchmark in {differ}, but the '
                                 f'caption says every other v3 family records the same versions')
    rows = [['Python', tex(env.get('python', '--').split(' (')[0])], ['NumPy', tex(env.get('numpy', '--'))],
            ['SciPy', tex(env.get('scipy', '--'))], ['scikit-learn', tex(env.get('sklearn', '--'))],
            ['PyTorch', tex(env.get('torch', '--'))], ['Platform', tex(env.get('platform', '--'))],
            ['Numerical threads per worker', str(env.get('numeric_threads', '--'))]]
    if any((runs / run / 'environment.json').exists() for run in (LEARNED_NESTED, LEARNED_SWAP)):
        raise AssertionError('tab_s_environment: the caption says the two studies of the learned encoder write no environment.json')
    models_src = (REPO / 'experiments' / 'make_revision' / 'models.py').read_text()
    harness = ''.join(f.read_text() for f in sorted((REPO / 'experiments' / 'make_revision').glob('*.py')))
    core = (REPO / 'arrowflow' / 'arrowflow.py').read_text()
    if harness.count('ArrowFlowConfig(') != 1 or "device='cpu'" not in models_src.split('ArrowFlowConfig(', 1)[1][:400] \
            or "if self.device != 'cpu' and torch.cuda.is_available():" not in core \
            or "os.environ['CUDA_VISIBLE_DEVICES'] = ''" not in (REPO / LEARNED_MODULES['conversion']).read_text() \
            or not re.search(r'\+cu\d+$', env.get('torch', '')):
        raise AssertionError('tab_s_environment: the caption says no run used a graphics processor that the PyTorch build supports')
    pools = learned_worker_pools(runs)
    caption = (f'\\textbf{{Software environment.}} Versions recorded in \\texttt{{environment.json}} of the nested '
               'benchmark. Every other harness family records the same versions, and the two studies of the learned encoder write '
               'no \\texttt{environment.json} (Section~\\ref{supp:protocols}). Every worker runs single-threaded, and no run used a '
               'graphics processor, although this PyTorch build supports one. At most '
               f'{FACTS["matched"].get("max_workers", "--")} workers run at once under the harness execution lock. The two studies of '
               f"the learned encoder ran outside it, with {pools['nested']} workers, {pools['swap'] + pools['second']} in the encoder "
               'swap.')
    write_table('tab_s_environment', caption, 'tab:s-environment', ['Component', 'Value'], rows, 'll')


def summed_fit_times(fam):
    """Summed fit seconds per model of one nested run: outer fits from the summary's model rows, inner selection fits
    from the selection records of the run's result files."""
    outer, inner = defaultdict(float), defaultdict(float)
    for d in DATASETS:
        for r in fam.summary['model_rows'][d]:
            outer[r['model_id']] += r.get('fit_seconds') or 0
    for path in sorted((fam.dir / 'results').glob('*.json')):
        for fit in (read_json(path).get('selection') or {}).get('fits', []):
            inner[fit['model_id']] += fit.get('fit_seconds') or 0
    return outer, inner


def render_compute(runs):
    """Summed wall-clock fit time of every nested benchmark run by model (S5), as reproducibility information only.
    `runs` holds (family, block title, model ids) in display order."""
    def hms(seconds):
        s = int(round(seconds))
        return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'
    rows, subheads, totals = [], {}, {}
    for fam, title, models in runs:
        if protocol_workers(fam.protocol) != FACTS['max_workers']:
            raise AssertionError(f'{fam.name}: its protocol records a worker limit other than the one the caption states')
        outer, inner = summed_fit_times(fam)
        missing = [m for m in models if m not in outer]
        if missing:
            raise AssertionError(f'{fam.name}: no outer fit times for {missing}')
        subheads[len(rows)] = f'{title}, protocol \\texttt{{{tex(fam.protocol["protocol_id"])}}}'
        rows += [[MODEL[m], hms(outer[m]), hms(inner[m])] for m in models]
        totals[fam.name] = {m: outer[m] + inner[m] for m in models}
    caption = (f'\\textbf{{Summed wall-clock fit time of the nested benchmark runs by model.}} Sum of the fit times in '
               f'h:mm:ss over the {words(len(DATASETS))} datasets and {FACTS["n_outer"]} outer folds of each run, given as '
               f'a guide to the compute that a reproduction needs. Outer fits are the fits of the chosen configuration with '
               f'every fitting seed; inner selection fits are those of every candidate, inner fold and finalist seed. Each '
               f'fit was measured once with a monotonic clock inside a single-thread worker process while up to '
               f'{FACTS["max_workers"]} workers ran at once. $^\\dagger$: from the first benchmark run.')
    write_table('tab_s_compute', caption, 'tab:s-compute', ['Model', 'Outer fits', 'Inner selection fits'], rows, 'lrr',
                subheads=subheads)
    return totals


# ----------------------------------------------------------------------------- inner-fold laboratory (S4)
LAB_LABEL = {
    'output_rule': 'Prototype readout (reference)',
    'knn_hidden': 'Footrule kNN on the hidden ranking',
    'kproto_borda': 'Several Borda prototypes per class, nearest by footrule',
    'borda1': 'One Borda prototype per class',
    'lvq1_borda_nearest': 'Permutation LVQ, Borda update, nearest prototype',
    'lvq1_borda_plurality': 'Permutation LVQ, Borda update, plurality of the nearest',
    'lvq1_borda_borda': 'Permutation LVQ, Borda update, Borda readout',
    'lvq2_borda_plurality': 'Permutation LVQ, two layers, Borda update, plurality',
    'lvq1_median_plurality': 'Permutation LVQ, footrule-median update, unit prior, plurality',
    'lvq1_median_priorlr_plurality': 'Permutation LVQ, footrule-median update, prior $\\eta$, plurality',
}
READOUT_ORDER = ['output_rule', 'knn_hidden', 'kproto_borda', 'borda1']
LVQ_ORDER = ['lvq1_borda_nearest', 'lvq1_borda_plurality', 'lvq1_borda_borda', 'lvq2_borda_plurality',
             'lvq1_median_plurality', 'lvq1_median_priorlr_plurality']


def render_laboratory(devlab_dir):
    """Inner-fold laboratory: mean inner accuracy per variant, the paired change against the reference and its SD."""
    out = {}
    proto = read_json(Path(devlab_dir) / 'protocol.json')
    for fam, order, name, stage in (('readouts', READOUT_ORDER, 'tab_s_lab_readouts', 'readout'),
                                    ('permlvq', ['output_rule'] + LVQ_ORDER, 'tab_s_lab_permlvq', 'update')):
        d = read_json(Path(devlab_dir) / f'{fam}_summary.json')
        acc, chg, ver = d['mean_inner_accuracy'], d['paired_change_vs_reference'], d['verdicts']
        units = d['units']
        n_outer = len(next(iter(d['resolved_configurations'].values())))
        per_dataset = units['completed'] // len(DATASETS)
        variants = [v for v in order if v != 'output_rule']
        rows, subheads = [], {0: 'Mean inner-validation accuracy, \\%'}
        for v in order:
            rows.append([LAB_LABEL[v]] + [pct(acc[ds][v]) for ds in DATASETS] + ['--', '--', '--'])
        subheads[len(rows)] = 'Mean paired change from the reference, pp'
        for v in variants:
            vv = ver[v]
            rows.append([LAB_LABEL[v]] + [signed(chg[ds][v]['mean_change_pp'] / 100) for ds in DATASETS]
                        + [signed(vv['mean_change_pp'] / 100), f"{vv['improved']} / {vv['datasets']}", tex(vv['verdict'])])
        subheads[len(rows)] = f'SD of the paired change over the {per_dataset} units of a dataset, pp'
        for v in variants:
            if any(chg[ds][v]['n_units'] != per_dataset for ds in DATASETS):
                raise AssertionError(f'{fam}/{v}: the paired change does not cover {per_dataset} units on every dataset')
            rows.append([LAB_LABEL[v]] + [f"{chg[ds][v]['sd_change_pp']:.1f}" for ds in DATASETS] + ['--', '--', '--'])
        heads = ['Variant'] + [LABEL[ds] for ds in DATASETS] + ['Mean', 'Improved', 'Verdict']
        g = proto['families'][fam]
        if fam == 'readouts':
            what = (f'The readouts act on the trained hidden ranking of the fitted first view (target-aware encoding). '
                    f'They are the network\'s own prototype readout (reference), footrule kNN with '
                    f'{listing(g["knn_grid"]["n_neighbors"])} neighbors and uniform or distance weights, and the prototype '
                    f'readouts. The prototype readouts use {listing(g["kproto_grid"]["k"])} Borda prototypes per class, found '
                    f'by clustering each class\'s hidden rankings and read out by the nearest prototype, or one Borda '
                    f'prototype per class')
        else:
            what = (f'Each variant is a permutation learning vector quantization (LVQ) layer that replaces the ranking '
                    f'layers on the same first-view '
                    f'encodings, seeds and budget, with {listing(g["prototype_grid"]["prototypes_per_class"])} labeled '
                    f'permutation prototypes per class. An accepted sample votes for the nearest prototype of its class '
                    f'toward the sample and for the nearest prototype of another class toward the reversed sample. The '
                    f'votes are aggregated by the Borda count or by the exact footrule median with a unit prior or a prior '
                    f'of weight $\\eta$. The prediction is the class of the nearest prototype or a plurality or Borda count '
                    f'over the {listing(g["readout_k_grid"])} nearest. The reference is the prototype readout of the fitted first '
                    f'view')
        rule = d['verdict_rule']
        adopted = sum(1 for v in variants if ver[v]['verdict'] == 'adopt')
        caption = (f'\\textbf{{Inner-fold laboratory, {stage} stage.}} First block: mean inner-validation accuracy in '
                   f'percent over {per_dataset} units per dataset ({words(n_outer)} outer folds of the first repeat '
                   f'$\\times$ {words(proto["inner_folds"])} inner folds $\\times$ {words(len(proto["fit_seeds"]))} fitting '
                   f'seeds). {what}. Every unit fits the configuration the prototype readout chose on that outer fold on the '
                   f'inner training partition; each variant\'s own settings are chosen one level deeper, on splits of the '
                   f'inner training partition. Second block: mean paired change from the reference in percentage points per '
                   f'dataset, its mean over datasets, the number of datasets improved, and the verdict of the adoption rule. '
                   f'The rule adopts a variant when at least {rule["minimum_improved"]} of {len(DATASETS)} datasets improve and none worsens by '
                   f'more than {rule["tolerance_pp"]:g} point. Third block: the SD of the paired change over the '
                   f'{per_dataset} units of each dataset, in percentage points; it describes dispersion across units and is '
                   f'not a confidence interval. {of_total(adopted, words(len(variants)), "variants", "meets", "meet")} the '
                   f'rule. No outer test fold is scored; the laboratory is exploratory. Protocol '
                   f'\\texttt{{{tex(d["protocol_id"])}}}, code \\texttt{{{d["code_revision"][:HASH_DIGITS]}}}.')
        write_table(name, caption, f'tab:s-lab-{fam}', heads, rows, 'l' + 'r' * (len(DATASETS) + 3),
                    subheads=subheads, size='\\scriptsize', wide=True)
        out[fam] = ver
    return out


# ----------------------------------------------------------------------------- guarded numbers and wording in the text
FORBIDDEN = ('never worse', 'improves everywhere', 'confirmation', 'confirmatory test',
             # no evaluation is called independent (ruling "independence disclosure, revised")
             'independent test', 'independent data', 'independent evaluation', 'independently confirmed',
             # names retired by the restructure around the nearest-neighbor readout
             'arrowflow-knn', 'full method', 'output rule', 'output-rule', 're-evaluat',
             # efficiency material removed by the author's decision
             'energy', 'hardware', 'operation count', 'integer operations', 'fixed size', 'computational cost',
             'prediction time', 'query time',
             # fix review I-1: accuracy reports on random-weight models stay attributed to that literature, never carried over
             'most of the accuracy', "most of arrowflow's accuracy",
             # studies removed by the author's decision of 2026-09-14: robustness, degree, gene expression, native rankings
             'robust', 'corrupt', 'sushi', 'tcga', 'invarian', 'monotone', 'gene expression', 'gene-expression',
             'degree study', 'noise sensitiv')


def prose_claims(bench, knn_facts, knn, ablation_facts, train, selected, round_off, production, training, kab, knn_ablation,
                 referee, matched_facts):
    """(file, phrase) for every guarded number and count printed in the prose, rebuilt from the run files."""
    n, c, sig, sh = len(DATASETS), knn_facts['contrasts'], knn_facts['significant'], knn_facts['shared']
    alt, proto = bench['stats'][KNN], bench['stats']['arrowflow_full']
    if knn_facts['not_lower'] != n:
        raise AssertionError('ArrowFlow has lower mean accuracy than its prototype readout on some dataset, but the text '
                             'says equal or higher on every dataset')
    if knn_facts['mismatched'] != 0:
        raise AssertionError('a comparator did not reproduce the first benchmark run, but the text says all of them did')
    if knn.protocol['primary_family_size'] != n or \
            math.prod(len(v) for v in knn.protocol['full_method']['candidate_grid'].values()) != FACTS['grid_size']:
        raise AssertionError('bridge_knn protocol: family size or candidate grid differs from the text')
    if alt['lowest'] != 0 or proto['lowest'] != 0:
        raise AssertionError('an ArrowFlow variant has the lowest error on some dataset, but the text says none')
    if not 2 * sh['folds'] < sh['n_folds']:
        raise AssertionError('ArrowFlow shares the configuration of the prototype readout on most folds, but the text '
                             'says most paired differences compare different networks')
    behind = [d for d in DATASETS if bench['gap'][(KNN, d)] > 0.03]
    if len(behind) != 1:
        raise AssertionError('Section 7.2 names exactly one dataset on which ArrowFlow trails by more than three points')
    ex, short = behind[0], {'svc_rbf': 'SVC', 'random_forest': 'random forest', 'mlp': 'MLP',
                            'numeric_knn': 'numeric kNN', 'gradient_boosting': 'gradient boosting'}
    parts = [f'${signed(bench["gap"][(KNN, d)])}${" pp" if i == 0 else ""} on {LABEL[d]}'
             for i, d in enumerate(d for d in DATASETS if d != ex)]
    gap_list = ', '.join(parts[:-1]) + ' and ' + parts[-1]
    proto_behind = sorted((d for d in DATASETS if bench['gap'][('arrowflow_full', d)] > 0.03),
                          key=lambda d: bench['gap'][('arrowflow_full', d)])
    effects = [f'{LABEL[d]} (${signed(c.loc[d, "mean_difference"])}${" pp" if i == 0 else ""})' for i, d in enumerate(sig)]
    refit = sorted({r['model_id'] for d in DATASETS for r in knn.summary['model_rows'][d]} - {KNN})
    if 'dummy' not in refit or sorted(set(refit) - {'dummy'}) != sorted(COMPARATORS):
        raise AssertionError('the ArrowFlow run did not refit exactly the classical models and the majority class')
    pilots = len(re.findall(r'pilot \d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z', knn.protocol['resource_decision']))
    frozen = knn.protocol['frozen_at_utc']
    revision = knn.environment['code_revision'][:9]
    headline = (f'ArrowFlow comes within three percentage points of the best tuned classical model on '
                f'{words(alt["within3"])} of {words(n)} tabular datasets and is best on none')
    readout = knn.protocol['knn_readout']
    if readout['grid']['weights'] != ['uniform', 'distance']:
        raise AssertionError('bridge_knn protocol: the readout weightings differ from those Section S2 states')
    split_count = re.search(r'StratifiedKFold\((\d+)', readout['selection'])
    if not split_count:
        raise AssertionError('bridge_knn protocol: no stratified split count in the readout selection')
    shared = (f'ArrowFlow chose the prototype readout\'s configuration in {sh["folds"]} of the {sh["n_folds"]} outer '
              f'folds ({sh["fits"]} of {sh["n_fits"]} outer fits')

    # training controls (ruling F-T): every number and count of the training result is rebuilt from the run files
    C, T, size = train['C'], train, train['size']
    if size != 2 * n:
        raise AssertionError('the text says the training family has two contrasts per dataset')
    effect = {d: C[(d, UNTRAINED)]['mean_difference'] for d in DATASETS}
    if len(T['higher'][UNTRAINED]) != n:
        raise AssertionError('the text says ArrowFlow is more accurate than the untrained ArrowFlow on every dataset')
    low_effect, high_effect = pct(min(effect.values())), pct(max(effect.values()))
    equal = [d for d in DATASETS if d not in T['higher'][INPUT_KNN]]
    if len(equal) != 1 or pct(abs(C[(equal[0], INPUT_KNN)]['mean_difference'])) != pct(0) or \
            pct(T['E'][equal[0]][KNN]['mean_error']) != pct(T['E'][equal[0]][INPUT_KNN]['mean_error']):
        raise AssertionError('the text says ArrowFlow and tuned input footrule kNN have the same mean error at table '
                             'precision on the one dataset where ArrowFlow is not more accurate')
    above = len(T['higher'][INPUT_KNN])
    if len(T['significant']) != 1 or T['significant'][0][1] != INPUT_KNN or C[T['significant'][0]]['mean_difference'] <= 0:
        raise AssertionError('the text says one training contrast is significant, a gain over tuned input footrule kNN')
    sig_d, sig_r = T['significant'][0][0], C[T['significant'][0]]
    clear = [d for d in DATASETS if any(k[0] == d for k in T['excludes'])]
    if len(clear) != 2 or set(T['excludes']) != {(d, m) for d in clear for m in CONTROLS} or \
            any(C[k]['ci_low'] <= 0 for k in T['excludes']):
        raise AssertionError('the text says the unadjusted intervals exclude zero in favor of ArrowFlow on two datasets '
                             'for both controls and nowhere else')
    others = [d for d in DATASETS if d not in clear]
    within = [d for d in DATASETS if abs(effect[d]) <= 0.01]
    if within != others or any(abs(float(pct(effect[d]))) > 1 for d in within) or \
            any(abs(float(pct(effect[d]))) <= 1 for d in clear):
        raise AssertionError('the text says the untrained ArrowFlow is within one point of ArrowFlow exactly on the other '
                             'datasets, also at table precision')
    favor = sum(C[k]['mean_difference'] > 0 for k in C)
    if favor != size - 1 or C[(equal[0], INPUT_KNN)]['mean_difference'] > 0:
        raise AssertionError('Section S3 says every training point estimate but one favors ArrowFlow')
    bounds = lambda k: f'$[{100 * C[k]["ci_low"]:+.1f},{100 * C[k]["ci_high"]:+.1f}]$'
    single, two = T['depths']
    groups = [(d, T['G'][(d, two)]) for d in DATASETS if T['G'][(d, two)]['n_folds']]
    if any(g['mean_difference'] <= 0 for _, g in groups):
        raise AssertionError('the text says the training effect at two layers is positive on every dataset where ArrowFlow '
                             'chose them')
    low = min(groups, key=lambda x: x[1]['mean_difference'])
    high = max(groups, key=lambda x: x[1]['mean_difference'])
    fewest, most = min(g['n_folds'] for _, g in groups), max(g['n_folds'] for _, g in groups)
    never = [LABEL[d] for d in DATASETS if not T['G'][(d, two)]['n_folds']]
    p1, p2 = T['pooled'][single], T['pooled'][two]
    tp = T['protocol']
    tfrozen, trev = tp['frozen_at_utc'], training.environment['code_revision'][:9]
    pilots_t = len(re.findall(r'pilot \d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z', tp['resource_decision']))
    if 'synthetic smoke' not in tp['resource_decision'] or pilots_t < 1:
        raise AssertionError('knn_training protocol: its resource decision records no smoke stage or pilot')
    P, stages = production, production['stages']
    if P['pinned'] != training.environment['code_revision'] or not P['pinned'].startswith(P['head']):
        raise AssertionError('the production script was not pinned to the revision that the training run recorded')
    S5_text = (HERE / 'supplement_sections' / 'S5_reproducibility.tex').read_text()
    if verbatim_commands(S5_text, 'Training controls') != P['commands']['20A']:
        raise AssertionError('Section S5: the commands of the training controls differ from the production script')
    block_20a = S5_text.split('# Training controls', 1)[1].split('\n# ', 1)[0]
    for old, new in (('--workers 16', '--workers 12'), ('segment digits', 'segment'),
                     ('P/2026-09-12/knn_training.json', 'P/2026-09-12/knn_ablation.json')):
        if old not in block_20a or verbatim_commands(S5_text.replace(block_20a, block_20a.replace(old, new, 1), 1),
                                                     'Training controls') == P['commands']['20A']:
            raise SystemExit(f'command guard mutation test ({old} -> {new}): not detected')
    print('command guard mutation test: 3 value-only mutations detected')
    if round_off['choice'] != 0:
        raise AssertionError('Section S2 says exact arithmetic picks the recorded configuration among the rescored '
                             'candidates on every fold')

    # component ablation of ArrowFlow on the benchmark datasets: its protocol, stages and commands (Section S5); its results are
    # guarded together with those of the further datasets in components_claims
    pa, ea = kab['protocol'], knn_ablation.environment
    if pa['frozen_at_utc'] != tp['frozen_at_utc'] or ea['code_revision'] != training.environment['code_revision']:
        raise AssertionError('Section S5 says both protocols were frozen at the same time and ran at the same revision')
    pilots_b = len(re.findall(r'pilot \d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z', pa['resource_decision']))
    if 'synthetic smoke' not in pa['resource_decision'] or pilots_b < 1:
        raise AssertionError('knn_ablation protocol: its resource decision records no smoke stage or pilot')
    if verbatim_commands(S5_text, 'Component ablation of ArrowFlow') != P['commands']['20B']:
        raise AssertionError('Section S5: the commands of the component ablation differ from the production script')

    R, M, folds_total = referee, matched_facts, n * FACTS['n_outer']

    E, S2, S3, S5 = ('sections/07_experiments.tex', 'supplement_sections/S2_protocol.tex',
                     'supplement_sections/S3_results.tex', 'supplement_sections/S5_reproducibility.tex')
    D, K, S4 = 'sections/08_discussion.tex', 'sections/09_conclusion.tex', 'supplement_sections/S4_ablations.tex'
    m_rd = re.search(r'grid reduced from (\d+) to (\d+) candidates by dropping embed_scale ([\d.]+) after the training-only '
                     r'pilot projected ([\d.]+)-([\d.]+) h at (\d+) workers', FACTS['bridge']['resource_decision'])
    if not m_rd:
        raise AssertionError('bridge protocol: its resource decision no longer records the pilot projection Section S2 quotes')
    extra_claims = [
        (S2, f'A pilot run timed only the training work of the first benchmark. It projected {m_rd.group(4)} to {m_rd.group(5)} h at '
             f'{m_rd.group(6)} workers, so the grid dropped the vocabulary scale {m_rd.group(3)} and kept {m_rd.group(2)} of its '
             f'{m_rd.group(1)} candidates.'),
        (S2, f'The matched protocol\'s own family of {words(M["family_size"])} contrasts is reported for completeness with '
             f'the matched study'),
        (S3, f'declares its own family of {M["family_size"]} contrasts, two per dataset, with its own Holm adjustment'),
    ]
    return extra_claims + [
        ('supplement_sections/S2_protocol.tex', f'$\\{{{",".join(str(k) for k in readout["grid"]["n_neighbors"])}\\}}$ '
                                                f'neighbors with uniform or inverse-distance weights'),
        ('supplement_sections/S2_protocol.tex', f'over {words(int(split_count.group(1)))} stratified splits of the stored rankings'),
        # the protocol details of Section 7.1, moved to S2.5 by the simplification pass; the main text's readout sentence and its
        # count of candidates are guarded in simplify_claims
        (S2, f'ArrowFlow\'s grid has {FACTS["grid_size"]} candidates.'),
        (S2, f'The family of {words(size)} training contrasts on the benchmark datasets was registered after the outer results of '
             f'ArrowFlow and of the prototype readout on these datasets were known. The {words(len(NEW_FAMILIES))} families of '
             f'{words(len(NEWDATA))} on the further datasets and their stratum test were registered before those runs. The '
             f'prototype-readout protocol fixed its family of {words(FACTS["family"])} prototype-readout contrasts before its outer folds '
             f'were scored.'),
        (S4, 'It became ArrowFlow\'s readout after the prototype readout\'s benchmark results were known.'),
        (S4, f'ArrowFlow ran with the same nested design, splits and {FACTS["grid_size"]} candidates as the prototype readout, '
             f'under protocol \\texttt{{{knn.protocol["protocol_id"]}}}. Its reported numbers come from the outer partitions '
             f'whose results informed the choice of its readout'),
        (S4, shared + '), so most paired differences also compare different networks.'),
        (S4, 'In Table~\\ref{tab:knn}, ArrowFlow has equal or higher mean accuracy than the prototype readout on every dataset.'),
        (S4, f'ArrowFlow is significantly higher on {listing(effects, "and")}, with Holm-adjusted $p<0.05$ over the '
             f'{words(n)} datasets, and not significantly different on the other {words(n - len(sig))}.'),
        (S4, f'the complete record of the {words(n)} contrasts of ArrowFlow with the prototype readout'),
        (S4, f'also refitted the {words(len(refit) - 1)} classical models and the majority class'),
        (S4, f'In each of the {knn_facts["cells"]:,} comparator cells (Table~\\ref{{tab:knn}})'),
        (S3, f'ArrowFlow chose {selected["single_label"]} on {selected["single"]} of {selected["folds"]} folds, the learning '
             f'rate ${selected["rate_value"]:g}$ on {selected["rate"]}, the doubled vocabulary size on {selected["scale"]} and '
             f'the lowered degree on {selected["offset"]}, with between {words(selected["distinct_min"])} and '
             f'{words(selected["distinct_max"])} distinct configurations per dataset.'),
        (S3, f'On {selected["tie_folds"]} of these folds, all on {listing([LABEL[d] for d in selected["tie_datasets"]], "and")}, '
             f'the first hidden layer has {selected["tie_N"]} filters over $e={selected["tie_V"]}$ items, so at least '
             f'{selected["tie_min"]} of its filters tie with another filter at every input'),
        (S3, f'The probe on the trained last hidden layer has higher error than the probe on the trained first layer on '
             f'{listing([LABEL[d] for d in M["worse"]], "and")} at both two-layer architectures, and lower error on '
             f'{listing([LABEL[d] for d in M["better"]], "and")}.'),
        (S3, f'The untrained ArrowFlow has {T["counts"][UNTRAINED]} candidates, the {FACTS["grid_size"]} of ArrowFlow without '
             f'the learning rate, which has no effect without training. Tuned input footrule kNN has '
             f'{T["counts"][INPUT_KNN]}, without the learning rate and the widths.'),
        (S3, f'The prototype-readout protocol fixed {words(FACTS["family"])} contrasts before its outer folds were scored.'),
        (S5, f'was frozen on {frozen[:10]} at {frozen[11:16]} UTC (Table~\\ref{{tab:s-protocols}})'),
        (S5, f'\\texttt{{{revision}}} for the ArrowFlow run'),
        (S5, f'checked out at commit \\texttt{{{revision}}}'),
        (S5, f'its run executed at code revision \\texttt{{{revision}}}'),
        (S5, f'ran its smoke stage and then {words(pilots)} training-only pilots before that protocol was frozen'),
        (S5, f'The training controls ran under their own frozen protocol, \\texttt{{{tp["protocol_id"]}}} (file '
             f'\\texttt{{2026-09-12/knn\\_training.json}}, whose \\texttt{{sha256}} digest begins with '
             f'\\texttt{{{sha256_file(PROTOCOLS_V3 / "knn_training.json")[:HASH_DIGITS]}}}), at code revision '
             f'\\texttt{{{trev}}}.'),
        (S5, f'The training controls ran their smoke stage and {words(pilots_t)} training-only '
             f'pilot{"s" if pilots_t > 1 else ""} before their protocol was frozen.'),
        (S5, f'from a git worktree detached at commit \\texttt{{{P["head"]}}} on {P["start"][:10]}, and its log records the stage '
             f'start times: \\texttt{{prepare}} at {stages["20A prepare"]} Coordinated Universal Time (UTC), the pairing check at {stages["20A pairing"]}, \\texttt{{run}} at '
             f'{stages["20A run"]}, reporting at {stages["20A reporting"]} and \\texttt{{compare\\_runs training}} at '
             f'{stages["20A compare"]}.'),
        (S5, f'The protocol of the training controls was frozen on {tfrozen[:10]} at {tfrozen[11:16]} UTC'),
        (S5, f'and its run executed at code revision \\texttt{{{trev}}}.'),
        (S5, f'ran at revision \\texttt{{{R["revision"]}}} on the saved predictions of the ArrowFlow run and the first benchmark run'),
        (S5, f'\\texttt{{{pa["protocol_id"]}}} (file \\texttt{{2026-09-12/knn\\_ablation.json}}, whose \\texttt{{sha256}} '
             f'digest begins with \\texttt{{{sha256_file(PROTOCOLS_V3 / "knn_ablation.json")[:HASH_DIGITS]}}}), at the same '
             f'revision.'),
        (S5, f'The component ablation of ArrowFlow also ran its smoke stage and {words(pilots_b)} training-only '
             f'pilot{"s" if pilots_b > 1 else ""} before its protocol was frozen. It followed in the same script, which logged '
             f'\\texttt{{prepare}} at {stages["20B prepare"]}, '
             f'\\texttt{{ablation}} at {stages["20B ablation"]} and \\texttt{{summary}} at {stages["20B summary"]} UTC. '
             f'The script ended at {P["end"][11:19]} UTC with exit status {P["exit"]}.'),
        (S5, 'The protocol of the component ablation of ArrowFlow was frozen at the same time, and its run executed at the '
             'same revision.'),
    ]


# ----------------------------------------------------------------------------- guarded prose of results fill 2
OLDER_WORK = ('arrowflow', 'manuscript/NeuralComputation', 'manuscript/MAKE')
TEXT_SUFFIXES = {'.py', '.tex', '.md', '.csv', '.json', '.txt', '.bib', '.yaml', '.yml', '.sh', '.cfg', '.ipynb', '.log'}


def older_work_mentions(pins):
    """Text files of the method's earlier benchmark code and manuscripts that name one of the additional datasets, by its
    OpenML name (any separator) or a distinctive part of it."""
    names = [p['openml_name'] for p in pins]
    distinctive = ['zernike', 'ionosphere', 'vertebra', 'banknote', 'biodeg', 'hepatitis', 'diabetes']
    if any(not any(token in name for name in names) for token in distinctive):
        raise AssertionError('a search token is not part of a pinned dataset name')
    full = ['[-_ ]'.join(re.escape(part) for part in re.split(r'[-_]', name)) for name in names]
    pattern = re.compile('|'.join(full + distinctive + [r'\bhcv\b']), re.I)
    hits = []
    for root in OLDER_WORK:
        base = REPO / root
        files = [p for p in sorted(base.rglob('*')) if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES] if base.is_dir() else []
        if not files:
            raise AssertionError(f'older-work scan: {base} is missing or holds no text files; the scan covers '
                                 f'{", ".join(OLDER_WORK)} under {REPO}')
        hits += [str(p.relative_to(REPO)) for p in files if pattern.search(p.read_text(errors='ignore'))]
    return hits


def results_fill2_claims(train, knn_records, sorting, newdata, ablation_facts, runs, projected, training, knn):
    """(file, phrase) for every guarded number, count and fact of the sorting control and the additional datasets, rebuilt
    from the run files, the production logs and scripts, the frozen protocols and the selection document."""
    E, S2, S3, S4, S5 = ('sections/07_experiments.tex', 'supplement_sections/S2_protocol.tex',
                         'supplement_sections/S3_results.tex', 'supplement_sections/S4_ablations.tex',
                         'supplement_sections/S5_reproducibility.tex')
    D, K, I = 'sections/08_discussion.tex', 'sections/09_conclusion.tex', 'sections/01_introduction.tex'
    n, k, n_outer = len(DATASETS), len(NEWDATA), FACTS['n_outer']
    C, size, P, ND = train['C'], train['size'], sorting, newdata
    EN, F, mt, sp, rules = ND['E'], ND['F'], ND['mt'], ND['sp'], ND['rules']
    marked, others, lab = ND['marked'], ND['others'], (lambda d: NEW_LABEL[d])
    signed1, signed2 = (lambda x: f'{100 * float(x):+.1f}'), (lambda x: f'{100 * float(x):+.2f}')

    # the additional datasets: the same two significant gains in both families, and the p <= 0.05 branch of the stratum test
    both = [d for d in ND['significant']['primary'] if d in ND['significant']['secondary']]
    if not (both == ND['significant']['primary'] == ND['significant']['secondary']) or len(both) != 2 or \
            any(F[(f, d)]['mean_difference'] <= 0 for d in both for f in ('primary', 'secondary')):
        raise AssertionError('the text says the same two datasets have significant gains over both controls')
    a, b = both
    if not (mt['mean_h'] > mt['mean_c'] and mt['p_one_sided'] <= mt['alpha']):
        raise AssertionError('the text states the stratum result of the p <= 0.05 branch')
    p_str = pval(mt['p_one_sided'])
    if older_work_mentions(list(ND['pin'].values())):
        raise AssertionError(f'the text says the additional datasets were never used in the earlier development of the method, '
                             f'but the scan of {", ".join(OLDER_WORK)} names one')
    higher = ND['higher']['primary']
    if [d for d in NEWDATA if d not in higher] != [HCV] or any(f'{abs(100 * F[("primary", d)]["mean_difference"]):.2f}' == '0.00' for d in NEWDATA):
        raise AssertionError('the text says ArrowFlow is less accurate than the untrained ArrowFlow only on HCV, at two decimals')
    sec = {d: F[('secondary', d)] for d in NEWDATA}
    rest = [d for d in NEWDATA if d not in both]
    near = min(rest, key=lambda d: sec[d]['holm_p_approximate'])
    if sec[near]['mean_difference'] <= 0 or any(abs(sec[d]['mean_difference']) >= 0.01 for d in rest if d != near):
        raise AssertionError('the text names one further gain over tuned input footrule kNN and puts every other difference within one point')
    eff = ND['effect']
    rows = {d: ND['pin'][d]['shape'][0] for d in NEWDATA}
    width = {d: F[('primary', d)]['ci_high'] - F[('primary', d)]['ci_low'] for d in NEWDATA}
    small_marked = [d for d in marked if d not in both]
    if sorted(small_marked) != sorted(sorted(NEWDATA, key=rows.get)[:2]) or \
            sorted(small_marked) != sorted(sorted(NEWDATA, key=width.get, reverse=True)[:2]) or \
            not set(both) < set(marked) or eff[a] + eff[b] <= sum(eff[d] for d in small_marked) or \
            (eff[a] + eff[b]) / len(marked) <= (mt['mean_h'] - mt['mean_c']) / 2:
        raise AssertionError('the text says the stratum difference comes mainly from the two significant datasets, and that the '
                             'other two marked datasets are the two smallest, with the widest intervals')
    notes = ND['notes']['ceiling_or_floor']
    majority = {d: max(ND['pin'][d]['class_counts']) / sum(ND['pin'][d]['class_counts']) for d in NEWDATA}
    ceiling = [d for d in NEWDATA if 'near ceiling' in notes.get(d, '')]
    floor = [d for d in NEWDATA if f'majority class {pct(majority[d])}%' in notes.get(d, '')]
    if len(ceiling) != 1 or len(floor) != 1:
        raise AssertionError('the text names one dataset near ceiling and one with a dominant majority class')
    if ND['lowest'][HCV] != 'dummy' or ND['pin'][HCV]['external_gap']['sources']:
        raise AssertionError('the text says HCV has no published results and no model beats its majority class')
    best = ND['best']
    gap_best = {d: EN[(d, KNN)]['mean'] - EN[(d, best[d])]['mean'] for d in NEWDATA}
    if any(g <= 0 for g in gap_best.values()) or \
            any(signed2(ND['CMP'][(d, best[d])]['mean_difference']) != f'{-100 * gap_best[d]:+.2f}' for d in NEWDATA):
        raise AssertionError('the text says ArrowFlow has higher mean error than the best tuned model on every additional dataset, '
                             'as the comparator tables print it')
    sortgap = {d: EN[(d, INPUT_KNN)]['mean'] - EN[(d, PROJECTED)]['mean'] for d in NEWDATA}
    worst, reverse = max(NEWDATA, key=sortgap.get), min(NEWDATA, key=sortgap.get)
    worse, better = [d for d in NEWDATA if sortgap[d] > 0], [d for d in NEWDATA if sortgap[d] < 0]
    if len(worse) + len(better) != k or sortgap[reverse] >= 0 or \
            EN[(worst, INPUT_KNN)]['mean'] - EN[(worst, KNN)]['mean'] <= sortgap[worst] / 2 or \
            any(pct(EN[(d, INPUT_KNN)]['mean']) == pct(EN[(d, PROJECTED)]['mean']) for d in NEWDATA):
        raise AssertionError('the text says sorting cost most on one dataset, where training recovered most of it, and helped on others')

    # the sorting control
    sort = P['sorting']
    if any(pct(abs(sort[d])) == pct(0) for d in DATASETS) or any(P['C'][(d, SORTING)]['holm_p_approximate'] < 0.05 for d in DATASETS):
        raise AssertionError('a sorting effect prints as zero or is significant, but the text counts directions and finds none significant')
    gain_sig = [d for d in DATASETS if P['C'][(d, PROJECTED_GAIN)]['holm_p_approximate'] < 0.05]
    fam_sig = [x for x in P['C'] if P['C'][x]['holm_p_approximate'] < 0.05]
    if len(gain_sig) != 1 or fam_sig != [(gain_sig[0], PROJECTED_GAIN)] or P['C'][fam_sig[0]]['mean_difference'] <= 0:
        raise AssertionError('the text says one contrast of the sorting family is significant, a gain of ArrowFlow')
    v = gain_sig[0]
    gains = {d: float(P['C'][(d, PROJECTED_GAIN)]['mean_difference']) for d in DATASETS}
    raw = {d: float(P['RAW'][d]['mean_difference']) for d in DATASETS}
    raw_top = max(DATASETS, key=lambda d: abs(raw[d]))
    ladder_v = [pct(P['L'][(v, m)]['mean_error']) for m, _ in LADDER]
    if raw_top != v or raw[v] >= 0 or [float(x) for x in ladder_v] != sorted((float(x) for x in ladder_v), reverse=True):
        raise AssertionError('the text says the raw-feature difference is largest and negative on the dataset of the significant gain, '
                             'where the ladder falls at every rung')
    shown_a = {'"$OUT"': '<run>', '"$K"': '<ArrowFlow run>', '"$TR"': '<training controls run>',
               '"$OUT/compare_projected"': '<run>/compare_projected'}
    log_a = production_script(runs, 'task23a-run.log', 'task23a-prod.sh', 'task23a',
                              lambda w, var: {'$DATASETS': ' '.join(read_json(REPO / var['PP'])['datasets']),
                                              '$PP': protocol_path(var['PP']), '$REG': var['REG'], **shown_a}.get(w, w))
    log_b, (p1, p2) = ND['log'], ND['protocols']
    pf, nf = P['protocol']['frozen_at_utc'], p1['frozen_at_utc']
    if not pf[:19] < log_a['start'][:19] or log_a['pinned'] != projected.environment['code_revision']:
        raise AssertionError('the sorting control was not frozen before its run, or its script pins another revision')
    for protocol in (P['protocol'], p1):
        if not (re.search(r'synthetic [\w-]*\s*smoke', protocol['resource_decision']) and 'training-only pilot' in protocol['resource_decision']):
            raise AssertionError(f'{protocol["protocol_id"]}: its resource decision records no synthetic smoke stage or training-only pilot')
    if 'real-data fit check' not in p1['resource_decision'] or \
            not ('before any ArrowFlow or comparator fit on these datasets' in p1['selection_ruling'] and 'all ten eligible datasets' in p1['selection_ruling']):
        raise AssertionError('batch protocols: the fit check or the selection ruling differs from Section S5')
    scripts = {tag: (PRODUCTION_SCRIPTS / f'{tag}-prod.sh').read_text() for tag in ('task23a', 'task23b')}
    if any(f'.worktrees/arrowflow-v3-{tag}-run' not in text for tag, text in scripts.items()):
        raise AssertionError('a production script did not run from its own git worktree')
    st_a, st_b = log_a['stages'], log_b['stages']
    if pf[:10] != nf[:10] or log_b['start'][:10] == log_b['end'][:10] or not st_b['B2 reporting'] < st_b['B2 run'] or \
            p1['analysis']['requires_both_batches_complete'] is not True:
        raise AssertionError('Section S5 dates the freezes on one day and batch 2 across midnight')
    S5_text = (HERE / 'supplement_sections' / 'S5_reproducibility.tex').read_text()
    if verbatim_commands(S5_text, 'Sorting control') != log_a['commands'] or verbatim_commands(S5_text, 'Further datasets') != log_b['commands']:
        raise AssertionError('Section S5: the commands of the sorting control or the additional datasets differ from their scripts')
    rev_p, rev_n = projected.environment['code_revision'][:9], ND['B'][1].environment['code_revision'][:9]

    # the benchmark side of the counts across both studies
    bench_higher = [d for d in DATASETS if C[(d, UNTRAINED)]['mean_difference'] > 0 and pct(abs(C[(d, UNTRAINED)]['mean_difference'])) != pct(0)]
    bench_sig = [d for d in DATASETS if any(C[(d, m)]['holm_p_approximate'] < 0.05 and C[(d, m)]['mean_difference'] > 0 for m in CONTROLS)]
    new_sig = [d for d in NEWDATA if any(F[(f, d)]['holm_p_approximate'] < 0.05 and F[(f, d)]['mean_difference'] > 0 for f in ('primary', 'secondary'))]
    if len(bench_higher) != n:
        raise AssertionError('the count across both studies assumes a positive printed training effect on every benchmark dataset')
    if ND['two'][a] or ND['two'][b]:
        raise AssertionError('the text says two hidden layers were never chosen on the datasets with significant gains')
    part = {LABEL[d]: max(o['n_train'] for key, rec in knn_records.items() if key[0] == d for o in rec['outer']) for d in DATASETS}
    for d in NEWDATA:
        fam = next(f for f in ND['B'].values() if d in f.protocol['datasets'])
        part[lab(d)] = max(o['n_train'] for rec in selection_records(fam, KNN, [d]).values() for o in rec['outer'])
    largest = max(part, key=part.get)
    depth_bench = train['pooled'][train['depths'][1]]['n_dataset_folds']
    two_total = sum(ND['two'].values())

    # the Discussion's aggregation numbers (Ruling (final review), FINAL-M1): inside the laboratory's permutation LVQ layer, with the
    # plurality readout, the mean inner accuracy of the Borda update minus that of each footrule-median update over the benchmark datasets;
    # across views, the largest difference between the Borda count and the majority vote in the prototype-readout ablation
    lab = read_json(runs / '2026-09-12-devlab' / 'permlvq_summary.json')
    acc, verdict = lab['mean_inner_accuracy'], lab['verdicts']
    median_gap = {v: 100 * statistics.mean(acc[d]['lvq1_borda_plurality'] - acc[d][v] for d in DATASETS)
                  for v in ('lvq1_median_plurality', 'lvq1_median_priorlr_plurality')}
    if sorted(acc) != sorted(DATASETS) or min(median_gap.values()) <= 0 or \
            any(abs(g - (verdict['lvq1_borda_plurality']['mean_change_pp'] - verdict[v]['mean_change_pp'])) > 1e-6 for v, g in median_gap.items()):
        raise AssertionError('laboratory: the Borda update does not beat both footrule-median updates on average, or the mean accuracies '
                             'and the paired changes disagree')

    return [
        # the sorting control left Section 7.3 for S3.4 in the simplification pass; its range, its family and its shared encoder
        # settings stay guarded in S3 below, and the family sizes in S2
        (S3, f'The sorting contrast favors the unsorted scores on {words(sum(x < 0 for x in sort.values()))} datasets and is '
             f'significant on none after Holm adjustment over its family of {words(P["size"])}, so on these {words(n)} datasets no '
             f'sorting effect was detected.'),
        (S2, f'its $p$ value is the share of the {mt["splits"]} splits of the {words(k)} datasets into {words(len(marked))} and '
             f'{words(len(others))} whose difference of mean training effects is at least the observed one.'),
        (S3, f'Table~\\ref{{tab:s-projected}} is the complete record of the sorting control of Section~\\ref{{sec:training-controls}}, a '
             f'family of {words(P["size"])} contrasts with one Holm adjustment'),
        (S3, f'Numeric kNN on the projected scores reads the scores of its {words(FACTS["views"])} encoders before they are sorted, '
             f'exactly the array that each encoder sorts, with its column scales.'),
        (S3, f'It uses the encoder family and the {words(P["candidates"])} candidates of tuned input footrule kNN but chooses among them '
             f'on its own inner folds, and the two chose the same encoder settings on {sum(P["same"].values())} of the {n * n_outer} '
             f'outer folds.'),
        (S3, f'tuned input footrule kNN minus numeric kNN on the projected scores lies between ${signed1(min(sort.values()))}$ '
             f'({LABEL[min(sort, key=sort.get)]}) and ${signed1(max(sort.values()))}$ pp ({LABEL[max(sort, key=sort.get)]}), and '
             f'ArrowFlow minus it between ${signed1(min(gains.values()))}$ ({LABEL[min(gains, key=gains.get)]}) and '
             f'${signed1(max(gains.values()))}$ pp ({LABEL[max(gains, key=gains.get)]}). Only the {LABEL[v]} contrast of ArrowFlow is '
             f'significant after Holm adjustment.'),
        (S3, f'It is ${pct(abs(raw[v]))}$ pp less accurate than numeric kNN on the projected scores on {LABEL[v]} and differs from it by '
             f'at most ${pct(max(abs(raw[d]) for d in DATASETS if d != v))}$ pp on the other datasets.'),
        (S3, f'the error on {LABEL[v]} falls from ${ladder_v[0]}\\%$ on the raw features to ${ladder_v[1]}\\%$ on the projected scores, '
             f'${ladder_v[2]}\\%$ after sorting, ${ladder_v[3]}\\%$ with the untrained ranking layers and ${ladder_v[4]}\\%$ after training.'),
        (S3, f'Their strata and analysis were frozen in {words(len(ND["B"]))} batch protocols before any outer fold was scored, and every '
             f'eligible dataset was run and is reported.'),
        (S3, f'A dataset was eligible with {rules["rows"][0]:,} to {rules["rows"][1]:,} rows, {rules["features"][0]} to '
             f'{rules["features"][1]} features, {rules["classes"][0]} to {rules["classes"][1]} classes, a smallest class of at least '
             f'{rules["minority"]} rows, missing values in at most {rules["missing"]}\\% of cells and exact duplicate rows in at most '
             f'{rules["duplicates"]}\\% of rows.'),
        (S3, f'Of {rules["funnel"][0]:,} active OpenML datasets in the size and class range, {rules["funnel"][1]} passed the metadata '
             f'filter, {rules["funnel"][2]} were in the frame and {rules["funnel"][3]} passed every rule.'),
        (S3, f'The counted sources are OpenML evaluations with at least {words(rules["runs"])} runs of a nearest-neighbor classifier (kNN) and the accuracy table of '
             f'\\citet{{fernandezdelgado2014hundreds}}.'),
        (S3, f'A dataset was marked when a counted source put tuned kNN at least {rules["gap"]:.1f} pp behind the best tuned random forest'),
        (S3, f'A $p$ value of at most {mt["alpha"]:g} requires a place among the {words(int(round(mt["alpha"] * mt["splits"])))} largest, '
             f'so the test has low power.'),
        (S4, f'seven views lower the error against one view on {words(len(ablation_facts["views_help"]))} datasets'),
        (S4, f'the Borda count over the seven class rankings and the majority vote differ by at most '
             f'${100 * ablation_facts["borda_max"]:.1f}$ pp'),
        (S5, f'\\texttt{{{rev_p}}} for the sorting control and \\texttt{{{rev_n}}} for both batches of the further datasets'),
        (S5, f'The sorting control ran under \\texttt{{{P["protocol"]["protocol_id"]}}} (file \\texttt{{2026-09-12/knn\\_projected.json}}, '
             f'whose digest begins with \\texttt{{{sha256_file(PROTOCOLS_V3 / "knn_projected.json")[:HASH_DIGITS]}}}) at code revision '
             f'\\texttt{{{rev_p}}}.'),
        (S5, f'The further datasets ran under \\texttt{{{p1["protocol_id"]}}} and \\texttt{{{p2["protocol_id"]}}} (files '
             f'\\texttt{{2026-09-12/newdata\\_batch1.json}} and \\texttt{{newdata\\_batch2.json}}, whose digests begin with '
             f'\\texttt{{{sha256_file(PROTOCOLS_V3 / "newdata_batch1.json")[:HASH_DIGITS]}}} and '
             f'\\texttt{{{sha256_file(PROTOCOLS_V3 / "newdata_batch2.json")[:HASH_DIGITS]}}}) at code revision \\texttt{{{rev_n}}}.'),
        (S5, 'The two batch protocols differ only in their identifiers and dataset lists'),
        (S5, 'The sorting control and the further datasets each ran a synthetic smoke stage and a training-only pilot before their '
             'protocols were frozen. Each production script ran from its own git worktree checked out at the frozen commit.'),
        (S5, f'The script of the sorting control, at commit \\texttt{{{log_a["head"]}}} on {log_a["start"][:10]}, logged '
             f'\\texttt{{prepare}} at {st_a["prepare"]} UTC, the pairing check at {st_a["pairing"]}, \\texttt{{run}} at {st_a["run"]}, '
             f'reporting at {st_a["reporting"]} and \\texttt{{compare\\_runs projected}} at {st_a["compare"]}, and it ended at '
             f'{log_a["end"][11:19]} UTC with exit status {log_a["exit"]}.'),
        (S5, f'The script of the further datasets, at commit \\texttt{{{log_b["head"]}}}, logged \\texttt{{prepare}} of both batches at '
             f'{st_b["prepare"]} UTC on {log_b["start"][:10]} and the pairing check at {st_b["pairing"]}.'),
        (S5, f'Batch 1 ran from {st_b["B1 run"]} to its reporting at {st_b["B1 reporting"]}, and batch 2 from {st_b["B2 run"]} to its '
             f'reporting at {st_b["B2 reporting"]} UTC on {log_b["end"][:10]}. Then the analysis, \\texttt{{compare\\_newdata analyse}}, '
             f'started at {st_b["analyse"]}, and the script ended at {log_b["end"][11:19]} UTC with exit status {log_b["exit"]}.'),
        (S5, 'That analysis refuses to run unless both batches are complete.'),
        (S5, f'The protocol of the sorting control was frozen on {pf[:10]} at {pf[11:16]} UTC, and its run executed at code revision '
             f'\\texttt{{{rev_p}}}.'),
        (S5, f'The {words(k)} further datasets were selected by prespecified rules, every eligible dataset included, before any model '
             f'was fitted on them.'),
        (S5, f'Their two batch protocols also fix the analysis. They were frozen at {nf[11:16]} UTC, after a fit check and a pilot that '
             f'predicted training rows only. Their runs executed at code revision \\texttt{{{rev_n}}}.'),
        (S5, f'The {words(k)} further datasets are OpenML copies pinned by identifier, version and content hashes '
             f'(Table~\\ref{{tab:s-newdata-evidence}}).'),
        (D, f'On the {words(n)} benchmark datasets, sorting the projected scores made no detectable difference to a nearest-neighbor '
            f'readout, so there the ranking itself cannot be credited with any gain in accuracy. On the {words(k)} further datasets, '
            f'where the comparison is descriptive, sorting raised the error on {words(len(worse))}, by up to {pct(sortgap[worst])} points '
            f'(Section~S3.4).'),
        (D, f'{words(n + k).capitalize()} small tabular datasets are a narrow basis. The largest training partition has '
            f'{part[largest]:,} rows ({largest}).'),
        # the laboratory's footrule-median gap left the Discussion; S4.3 states it
        (S4, f'the Borda count beat the footrule median by {min(median_gap.values()):.0f} to {max(median_gap.values()):.0f} points on '
             f'average'),
        (D, f'Across views, with the prototype readout, Borda and majority votes differ by at most '
            f'${100 * ablation_facts["borda_max"]:.1f}$ percentage points (Section~S4).'),
    ]


# ----------------------------------------------------------------------------- guarded prose of restructure 2
def restructure2_claims(bridge, newdata, combined, train, sorting, referee, bench, matched_facts):
    """(file, phrase) for the guarded numbers and facts of the combined presentation of the seventeen datasets, rebuilt from the
    dataset manifests, the pins and the selection document of the batch protocols, the older-work scan, the verified runs and
    the sealed outputs of the combined analysis."""
    E, I, D, K = ('sections/07_experiments.tex', 'sections/01_introduction.tex', 'sections/08_discussion.tex',
                  'sections/09_conclusion.tex')
    S1, S2, S3, S4, S5 = ('supplement_sections/S1_proofs.tex', 'supplement_sections/S2_protocol.tex',
                          'supplement_sections/S3_results.tex', 'supplement_sections/S4_ablations.tex',
                          'supplement_sections/S5_reproducibility.tex')
    ND, C, n, k, n_outer = newdata, combined, len(DATASETS), len(NEWDATA), FACTS['n_outer']
    k17, folds17 = n + k, (n + k) * n_outer
    signed2 = lambda x: f'{100 * float(x):+.2f}'
    name = lambda d: NAME[d]

    # Section 7.1: the seventeen datasets
    manifests = {d: read_json(bridge.dir / d / 'manifest.json') for d in DATASETS}
    shape = {**{d: (int(m['shape'][0]), int(m['shape'][1]), len(m['class_counts'])) for d, m in manifests.items()},
             **{d: (int(ND['pin'][d]['shape'][0]), int(ND['pin'][d]['shape'][1]), len(ND['pin'][d]['class_counts'])) for d in NEWDATA}}
    rows, features, classes = ([s[i] for s in shape.values()] for i in range(3))
    if older_work_mentions(list(ND['pin'].values())):
        raise AssertionError(f"the text says none of the further datasets was used in the method's earlier development, but "
                             f"the scan of {', '.join(OLDER_WORK)} names one")
    if NEWDATA[-1] != HCV:
        raise AssertionError('Section 7.1 names HCV last, where it is expanded')
    marked, others = ND['marked'], ND['others']

    # the main benchmark
    gap = C['gap']
    dmin, dmax = min(COMBINED, key=gap.get), max(COMBINED, key=gap.get)
    flag_of = {flag: [d for d in COMBINED if flag in C['flags'][d]] for flag in FLAG_MARK}
    if any(len(v) != 1 or v[0] not in C['within'] for v in flag_of.values()):
        raise AssertionError('Section 7.2 names one dataset for each degenerate flag, each within three points')
    ceil_d, near_d, none_d = flag_of['ceiling'][0], flag_of['near_majority'][0], flag_of['no_learning'][0]
    mr = C['mean_rank']
    order = sorted(mr, key=mr.get)
    if order[0] != 'svc_rbf' or KNN not in C['separated'].get('svc_rbf', []) or C['p'] >= 0.001:
        raise AssertionError('Section 7.2 says the SVC ranks first, the critical difference separates it from ArrowFlow, and p<0.001')
    position = {2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}[order.index(KNN) + 1]
    if [b for a, bs in C['separated'].items() for b in bs if KNN in (a, b)] + [a for a, bs in C['separated'].items() if KNN in bs] != [KNN, 'svc_rbf']:
        raise AssertionError('the text says the critical difference separates ArrowFlow from the SVC and from no other model')
    ahead = order[:order.index(KNN)]
    # the rank facts of the Introduction and the Conclusion (Ruling (final review), FINAL-I2): the models ranked next to ArrowFlow, and the
    # SVC as the only model ahead of ArrowFlow beyond the critical difference, with none behind it
    rank_better, rank_worse = order[order.index(KNN) - 1], order[order.index(KNN) + 1]
    if [a for a, bs in C['separated'].items() if KNN in bs] != ['svc_rbf'] or C['separated'].get(KNN):
        raise AssertionError('the text says only the SVC ranks ahead of ArrowFlow beyond the critical difference, and no model behind it')
    parts = [f'{THE[a]} from {listing([PLAIN[b] for b in bs], "and")}' for a, bs in C['separated'].items()]
    separated_text = ', and '.join(parts) if len(parts) == 2 else listing(parts, 'and')
    R = referee

    # the training controls
    reg, h34, keys = C['reg'], C['h34'], C['keys']
    bench_sig = [x for x in C['significant'] if x[0] in DATASETS]
    further_sig = [x for x in C['significant'] if x[0] in NEWDATA]
    if len(bench_sig) != 1 or bench_sig[0][1] != INPUT_KNN or reg[bench_sig[0]]['diff'] <= 0:
        raise AssertionError('the text says one benchmark contrast is significant, a gain over tuned input footrule kNN')
    vd = bench_sig[0][0]
    pair = sorted({d for d, _ in further_sig}, key=NEWDATA.index)
    if len(pair) != 2 or sorted(further_sig) != sorted((d, m) for d in pair for m in CONTROLS) or any(reg[x]['diff'] <= 0 for x in further_sig):
        raise AssertionError('the text says the gains over both controls are significant on the same two further datasets')
    a2, b2 = pair
    if sorted(C['survivors']) != sorted(further_sig):
        raise AssertionError('the text says only the four gains on the two further datasets survive the adjustment over all contrasts')
    sig_datasets = sorted({d for d, _ in C['significant']}, key=COMBINED.index)
    if [d for d in COMBINED if d not in C['higher']] != [HCV] or 'no_learning' not in C['flags'][HCV]:
        raise AssertionError('the text says only HCV, where no model beats the majority class, has a negative training effect')
    T = train
    clear_b = [d for d in DATASETS if any(key[0] == d for key in T['excludes'])]
    if len(clear_b) != 2 or set(T['excludes']) != {(d, m) for d in clear_b for m in CONTROLS} or any(T['C'][x]['ci_low'] <= 0 for x in T['excludes']):
        raise AssertionError('the text says the unadjusted intervals exclude zero in favor of ArrowFlow on two benchmark datasets for both controls')
    effect_b = {d: T['C'][(d, UNTRAINED)]['mean_difference'] for d in DATASETS}
    within1 = [d for d in DATASETS if abs(effect_b[d]) <= 0.01]
    if sorted(within1) != sorted(d for d in DATASETS if d not in clear_b):
        raise AssertionError('the text says the untrained ArrowFlow is within one point exactly on the other benchmark datasets')
    close = C['close']
    if [d for d in close if d in DATASETS] != within1:
        raise AssertionError('the benchmark datasets within one point of the untrained ArrowFlow differ between the combined analysis '
                             'and the training run')
    rest = [x for x in keys if x[0] in NEWDATA and x not in further_sig]
    near = min(rest, key=lambda x: reg[x]['holm'])
    if near[1] != INPUT_KNN or reg[near]['diff'] <= 0 or reg[near]['holm'] >= 0.1:
        raise AssertionError('the text names one near miss, a gain over tuned input footrule kNN')
    mt, sp = ND['mt'], ND['sp']
    if not (mt['mean_h'] > mt['mean_c'] and mt['p_one_sided'] <= mt['alpha']):
        raise AssertionError('the text states the stratum result of the p <= 0.05 branch')
    two = C['two']
    two_all, two_b, two_f = sum(two.values()), sum(two[d] for d in DATASETS), sum(two[d] for d in NEWDATA)
    if two[a2] or two[b2]:
        raise AssertionError('the text says the significant further gains came from single-hidden-layer networks')
    rows_n = {d: ND['pin'][d]['shape'][0] for d in NEWDATA}
    width = {d: ND['F'][('primary', d)]['ci_high'] - ND['F'][('primary', d)]['ci_low'] for d in NEWDATA}
    small_marked = [d for d in marked if d not in pair]
    if sorted(small_marked) != sorted(sorted(NEWDATA, key=rows_n.get)[:2]) or \
            sorted(small_marked) != sorted(sorted(NEWDATA, key=width.get, reverse=True)[:2]) or not set(pair) < set(marked):
        raise AssertionError('the text says the other two marked datasets are the two smallest further datasets, with the widest intervals')
    favor = sum(reg[x]['diff'] > 0 for x in keys)
    largest_exception = max(-reg[x]['diff'] for x in keys if reg[x]['diff'] <= 0)
    single, double = C['depths']
    pooled = C['pooled']
    never = [d for d in COMBINED if two[d] == 0]

    # sorting
    P = sorting
    gain_sig = [d for d in DATASETS if P['C'][(d, PROJECTED_GAIN)]['holm_p_approximate'] < 0.05]
    if len(gain_sig) != 1 or P['C'][(gain_sig[0], PROJECTED_GAIN)]['mean_difference'] <= 0:
        raise AssertionError('the text says ArrowFlow beats numeric kNN on the projected scores significantly on one dataset')
    v = gain_sig[0]
    sortgap = {d: ND['E'][(d, INPUT_KNN)]['mean'] - ND['E'][(d, PROJECTED)]['mean'] for d in NEWDATA}
    worse, better = [d for d in NEWDATA if sortgap[d] > 0], [d for d in NEWDATA if sortgap[d] < 0]
    if len(worse) + len(better) != k:
        raise AssertionError('a sorting difference on a further dataset is exactly zero, but the text sorts every dataset')
    worst, reverse = max(NEWDATA, key=sortgap.get), min(NEWDATA, key=sortgap.get)

    # layers whose responses tie at every input
    if len(C['tie_layers']) != 1:
        raise AssertionError('the text names one layer size whose responses tie at every input')
    (tie_v, tie_n), = C['tie_layers']
    ties_all, ties_f = sum(C['ties'].values()), sum(C['ties'][d] for d in NEWDATA)

    # the selection of the further datasets
    doc = SELECTION_DOCUMENT.read_text()
    for rule in ('There is no target leakage, no time series, and no artificial generator outside CC18',
                 'The licence permits research use', 'If it has fewer, the standardized and range-normalized families are pooled',
                 'At 35 or more, mfeat-zernike loses its informative source'):
        if rule not in doc:
            raise AssertionError(f'selection document: no rule {rule!r}, which Section S3 states')
    zm = re.search(r'diabetes has (\d+) zero cells in ([a-z, ]+?), which is', doc)
    fam = next(f for f in ND['B'].values() if 'diabetes' in f.protocol['datasets'])
    feature_names = read_json(fam.dir / 'diabetes' / 'manifest.json')['feature_names']
    with np.load(fam.dir / 'diabetes' / 'data.npz', allow_pickle=False) as data:
        X = data['X']
    columns = [c.strip() for c in re.split(r',|\band\b', zm.group(2)) if c.strip()] if zm else []
    zeros = int((X[:, [feature_names.index(c) for c in columns]] == 0).sum()) if columns else -1
    if not zm or zeros != int(zm.group(1)):
        raise AssertionError('diabetes: the placeholder zeros of the prepared data differ from the selection document')
    all_zeros = int((X == 0).sum())
    other_zeros = {f: int((X[:, j] == 0).sum()) for j, f in enumerate(feature_names) if f not in columns and (X[:, j] == 0).any()}
    if list(other_zeros) != ['preg'] or all_zeros != zeros + other_zeros['preg']:
        raise AssertionError('diabetes: its zero cells outside the placeholder columns are not all pregnancy counts')
    informative = ND['informative']
    with_sources = [d for d in NEWDATA if ND['pin'][d]['external_gap']['sources']]
    if [d for d in with_sources if not any(informative[d].values())] != ['climate_model_simulation_crashes'] or not ND['rank_stable'] \
            or ND['pin'][HCV]['external_gap']['sources'] or not any(s.startswith('openml') and flag for s, flag in informative['mfeat_zernike'].items()):
        raise AssertionError('Section S3: the counted sources differ from the pooling, uncounted-source and HCV sentences')
    if (ND['dup']['label_conflicting_groups'] != 0).any():
        raise AssertionError('Table S-duplicates: a group of identical rows carries two labels, which the duplicate paragraph of S3.2 '
                             'does not discuss')

    # the combined analysis in S5
    ready = C['ready']
    S5_text = (HERE / 'supplement_sections' / 'S5_reproducibility.tex').read_text()
    holistic_src = (REPO / 'experiments' / 'make_revision' / 'holistic.py').read_text()
    # the commands pass the runs directory explicitly (Ruling (final review), FINAL-M3); without it, holistic.py and this renderer default
    # to the runs directory of the project workspace, a path on the author's machine, as Section S5 says (referee-panel revision of
    # 2026-09-25, item H2: the text now tells a reader to always pass --runs)
    runs_arg = ('--runs', '<runs', 'directory>')
    expected = [('experiments.make_revision.holistic', ('benchmark',) + runs_arg + ('--output', '<run>/benchmark')),
                ('experiments.make_revision.holistic', ('training',) + runs_arg + ('--output', '<run>/training')),
                ('experiments.make_revision.holistic', ('ready', '--root', '<run>'))]
    if verbatim_commands(S5_text, 'Combined analysis of the seventeen datasets') != expected or \
            "commands.add_parser('ready'" not in holistic_src or "('benchmark', 'the combined main benchmark'), ('training', 'the combined training controls')" not in holistic_src:
        raise AssertionError('Section S5: the commands of the combined analysis differ from its command line')
    verbatim = S5_text.split('\\begin{verbatim}', 1)[1].split('\\end{verbatim}', 1)[0]
    if 'python manuscript/v3/render_tables.py --runs <runs directory>\n' not in verbatim or \
            DEFAULT_RUNS != REPO.parent / '.superpowers' / 'sdd' / '2026-09-12-arrowflow-story-restoration-plan' / 'runs' or \
            "WORKSPACE_RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'" not in holistic_src or \
            holistic_src.count("add_argument('--runs', type=Path, default=WORKSPACE_RUNS") != 2:
        raise AssertionError('Section S5: the renderer command or the default runs directories differ from the text')

    # Section 7.1: the prototype readout ran as a separately tuned model only on the benchmark datasets
    if bridge.protocol.get('datasets') != DATASETS or any('arrowflow_full' in p['models'] for p in ND['protocols']):
        raise AssertionError('Section 7.1 says the prototype readout ran in a run of its own on the benchmark datasets only')
    # a count of datasets with significant gains names the control that carries them (post-restructure review, Critical)
    sig_untrained = [d for d in sig_datasets if (d, UNTRAINED) in C['significant']]
    intro_controls = 'an untrained ArrowFlow' if sig_untrained == sig_datasets else 'controls that train no filters'
    FACTS['attribution'] = dict(names=[NAME[d] for d in sig_untrained], all_names=[NAME[d] for d in COMBINED],
                                count={'all': len(sig_untrained), 'benchmark': sum(d in DATASETS for d in sig_untrained),
                                       'further': sum(d in NEWDATA for d in sig_untrained)})
    if sig_datasets != [vd, a2, b2]:
        raise AssertionError('the abstract names the datasets with significant gains in panel order: the benchmark dataset, then the two further ones')
    # post hoc labels (ruling P1 final 2, M2): the training protocol declared the benchmark depth split with its pool, and the
    # batch protocols declare none, so the rows of the further datasets and every pool over the seventeen datasets are post hoc
    if 'pooled across datasets' not in train['protocol']['training_controls']['depth_split']['definition'] or \
            any('depth' in json.dumps(p['analysis']).lower() for p in ND['protocols']):
        raise AssertionError('the post hoc labels of the depth split assume a declared benchmark split and no declared further split')
    # Section S3: the outer folds on which the two classifiers of the sorting control chose the same encoder settings
    same = sorting['same']
    per_dataset = [f'{same[d]} of the {n_outer} outer folds of {LABEL[d]}' if i == 0 else f'{same[d]} of {LABEL[d]}'
                   for i, d in enumerate(DATASETS)]
    fams = f'{words(k)} contrasts'
    # the simplified Section 7.1 (guarded here, where the manifests and pins are read) and the marked datasets of Section 7.3
    plain_fact(E, f'We use {words(k17)} classification datasets. {words(n).capitalize()} are standard benchmarks',
               f'We use {words(k17)}', f'We use {words(k17 + 1)}')
    cite = dict(balance_scale='uci_balance', mfeat_zernike='uci_mfeat', ionosphere='uci_ionosphere', vertebra_column='uci_vertebral',
                diabetes='smith1988diabetes', banknote_authentication='uci_banknote', qsar_biodeg='uci_qsar',
                steel_plates_fault='uci_steel', climate_model_simulation_crashes='uci_climate', hcv_egyptian_patients='uci_hcv')
    if sorted(cite) != sorted(NEWDATA):
        raise AssertionError('Section 7.1 cites a source for every further dataset')
    plain_fact(E, f'The other {words(k)}, the further datasets, were fixed in advance as every dataset of a predefined pool of '
                  f'OpenML datasets \\citep{{vanschoren2014openml}} that met prespecified eligibility rules, as interpreted during '
                  f'selection and before any model was fitted (Section~S3.1). They are '
                  + listing([f'{NEW_LABEL[d]} \\citep{{{cite[d]}}}' for d in NEWDATA[:-1]]
                            + [f'the hepatitis C virus (HCV) data \\citep{{{cite[HCV]}}}'], 'and') + '.',
               f'The other {words(k)}', f'The other {words(k - 1)}')
    plain_fact(E, f'The method, its candidate settings and the comparison models were fixed before these {words(k)} were selected, and '
                  f"none of the {words(k)} had been used in the method's earlier development (Section~S5).",
               f"none of the {words(k)} had", f"none of the {words(k - 1)} had")
    plain_fact(E, f'All {words(k17)} are small: {min(rows):,} to {max(rows):,} rows, {min(features)} to {max(features)} features and '
                  f'{min(classes)} to {max(classes)} classes (Section~S5.2)', f'{max(rows):,} rows', f'{max(rows) + 1:,} rows')
    plain_fact(E, f'Before the runs, {words(len(marked))} further datasets, {listing([NEW_LABEL[d] for d in marked], "and")}, were '
                  f'marked because the published results we counted \\citep{{vanschoren2014openml,fernandezdelgado2014hundreds}} put '
                  f'tuned kNN at least {ND["rules"]["gap"]:g} percentage points behind the best tuned model (Section~S3.1).',
               f'at least {ND["rules"]["gap"]:g} percentage points', f'at least {ND["rules"]["gap"] + 1:g} percentage points')
    return [
        # Section 7.1, simplified; the sentence on the prototype readout's own run moved to S2.5
        (S2, 'The prototype readout, the ablation of ArrowFlow that predicts with the nearest output filter, ran with the same grid in '
             'a separate run on the same partitions of the benchmark datasets.'),
        # the details of Section 7.2, moved to S3.2 by the simplification pass; the main-text phrases are guarded in simplify_claims
        (S3, f'Three post hoc flags mark degenerate comparisons: ceiling, when the best tuned classical model has less than '
            f'${COMBINED_RULES["ceiling_points"]:g}\\%$ error, and near-majority, when ArrowFlow is within '
            f'${COMBINED_RULES["near_majority_points"]:g}$ points of the majority class and at least '
            f'${COMBINED_RULES["near_majority_advantage_points"]:g}$ points behind the best tuned model.'),
        (S3, 'No-learning marks a dataset on which no model has lower mean error than the majority class.'),
        (S3, f'ArrowFlow is within three percentage points of the best tuned classical model on {len(C["within"])} of the {k17} datasets, '
            f'among them {name(ceil_d)} (ceiling), {name(near_d)} (near-majority) and {name(none_d)} (no-learning).'),
        (S3, 'The mean ranks are ' + listing([f'${mr[m]:.2f}$ for {PLAIN[m]}' for m in order], 'and')
             + f', so ArrowFlow ranks {position} of {words(len(mr))}.'),
        # the details of Section 7.3, moved to S3.3; the adjustment over all contrasts, the Friedman test, the Nemenyi separation,
        # the pooled depth split and the eight clear intervals were already guarded there
        (S3, f'Of the {two_all} two-layer folds, {two_b} are on the benchmark datasets and {two_f} on the further datasets, so the '
             f'significant gains on {name(a2)} and {name(b2)} came from single-hidden-layer networks.'),
        (S3, f'The Friedman test \\citep{{demsar2006statistical}}, post hoc and descriptive, gives $\\chi^2={C["chi2"]:.2f}$ on '
             f'{len(mr) - 1} degrees of freedom '
             f'($p={scientific(C["p"])}$), and the Nemenyi critical difference of ${C["cd"]:.2f}$ separates {separated_text}.'),
        (S3, 'ArrowFlow is separated from no model other than the SVC.'),
        (S3, f'On {name(HCV)} ArrowFlow is less accurate than the untrained ArrowFlow (${signed2(reg[(HCV, UNTRAINED)]["diff"])}$ pp, '
             f'interval $[{signed2(reg[(HCV, UNTRAINED)]["lo"])},{signed2(reg[(HCV, UNTRAINED)]["hi"])}]$). But there no model has '
             f'lower mean error than the majority class, so the contrast is wide and uninformative (Section~\\ref{{supp:controlled}}).'),
        (S3, f'On the benchmark datasets, only the gain over tuned input footrule kNN on {LABEL[vd]} is significant after Holm adjustment '
             f'(${signed(reg[(vd, INPUT_KNN)]["diff"], 2)}$ pp, Holm-adjusted $p={pval(reg[(vd, INPUT_KNN)]["holm"])}$), under an '
             f'approximate test (Section~\\ref{{supp:statistics}}).'),
        (S3, f'There the unadjusted intervals exclude zero on {LABEL[clear_b[0]]} and {LABEL[clear_b[1]]} for both controls.'),
        (S3, f'Descriptively, the untrained ArrowFlow is within one percentage point of ArrowFlow on {len(close)} of the {k17} datasets, '
             f'{words(len(within1))} of them benchmark datasets.'),
        (S3, f'On the further datasets, the gains over both controls are significant on {name(a2)} and {name(b2)}. The gain over tuned '
             f'input footrule kNN on {name(near[0])}, ${signed(reg[near]["diff"], 2)}$ pp, just misses significance (Holm-adjusted '
             f'$p={pval(reg[near]["holm"])}$).'),
        (S3, f'The prespecified exact one-sided stratum test finds a larger mean training effect on the {words(len(marked))} marked datasets '
             f'than on the other {words(len(others))} (${signed2(mt["mean_h"])}$ against ${signed2(mt["mean_c"])}$ pp, '
             f'$p={pval(mt["p_one_sided"])}$), but cannot separate a poor feature metric from room to improve and rests on {words(k)} '
             f'datasets.'),
        (S3, f'Its level is exact under exchangeability of the {words(k)} effects between the strata, and omitting any one of '
             f'{words(sum(v > mt["alpha"] for v in ND["loo"].values()))} datasets raises $p$ above ${mt["alpha"]:g}$.'),
        (S3, 'Repeating the same exact test on the nine remaining datasets gives '
             + listing([f'{pval(ND["loo"][d])} without {NAME[d]}' for d in sorted(NEWDATA, key=lambda x: (ND['loo'][x], x))],
                       'and') + '.'),
        (S3, f'Descriptively, the published kNN gap and the training effect have Spearman correlation ${sp["rho"]:.2f}$ over the '
             f'{words(sp["n"])} datasets with external evidence (exact $p={pval(sp["p_one_sided"])}$ one-sided, '
             f'${pval(sp["p_two_sided"])}$ two-sided).'),
        # abstract
        ('main.tex', 'Each layer holds learned rankings, the ranking filters, and outputs them sorted by footrule distance to its input.'),
        # the other abstract sentences are guarded in final_round_claims (the restructured abstract of 2026-09-23)
        # the Introduction and the Discussion are guarded in simplify_claims; the prototype readout's scope moved to S4.1
        (S4, "On the further datasets the prototype readout is evaluated only as a component variant at ArrowFlow's selections."),
        # Section S1: the full-order certificate is unattainable at three layer sizes, and forced ties at one of them
        (S1, f'Among the hidden-layer sizes of ArrowFlow, $V_\\ell\\in\\{{{",".join(str(v) for v in C["declared_v"])}\\}}$ '
             f'with $N_\\ell\\in\\{{{",".join(str(v) for v in C["declared_n"])}\\}}$ (Table~\\ref{{tab:settings}}), this '
             f'fails at $(V_\\ell,N_\\ell)\\in\\{{{",".join(f"({v},{nf})" for v, nf in sorted(C["vacuous_layers"]))}\\}}$, '
             f'so no input of such a layer carries a full-order certificate.'),
        (S1, f'Only $({tie_v},{tie_n})$ also has more filters than distance values, so only there does the pigeonhole principle '
             f'force ties and the hypothesis of (ii) fail outright.'),
        (S1, "ArrowFlow's first hidden layer was one of the " + words(len(C['vacuous_layers'])) + " on "
             + f"{sum(C['vacuous'].values())} of the {folds17} outer folds of the {words(k17)} datasets, "
             + listing([f"{C['vacuous_first'][pair]} at $({pair[0]},{pair[1]})$" for pair in sorted(C['vacuous_layers'])], "and")
             + f", and on {len([d for d in COMBINED if C['vacuous'][d]])} of the {k17} datasets."),
        # Conclusion
        (K, f'With every configuration chosen inside the training data, ArrowFlow trails the best tuned classical model on all '
            f'{words(k17)} tabular datasets (Table~\\ref{{tab:main}}).'),
        (K, f'Its average rank, {position} of {words(len(mr))}, lies between {PLAIN[rank_better]} and {PLAIN[rank_worse]};'),
        # S2
        (S2, f'the {words(T["size"])} benchmark training contrasts, the {words(P["size"])} contrasts of the sorting '
             f'control and the {words(len(NEW_FAMILIES))} families of {fams} on the further datasets. The {words(n)} contrasts of '
             f'ArrowFlow with the prototype readout and the {words(FACTS["family"])} contrasts fixed in the benchmark protocol form two '
             f'more families.'),
        (S2, f'One Holm adjustment over all {len(keys)} training contrasts, defined after every result was known, is reported as a post '
             f'hoc sensitivity analysis'),
        # S3: dataset selection
        (S3, 'It also had to be all numeric, free of target leakage, neither a time series nor an artificial generator outside '
             'OpenML-CC18, licensed for research use and neither a benchmark dataset nor a copy of one.'),
        (S3, f'Placeholder zeros were not counted as missing. Of the {all_zeros} zero cells of diabetes, {zeros}, '
             f'${100 * zeros / X.size:.1f}\\%$ of its cells, are placeholder zeros in {words(len(columns))} measurement columns and '
             f'{other_zeros["preg"]} are zero pregnancy counts.'),
        (S3, f'When a task had fewer than {words(ND["rules"]["runs"])} standardized kNN runs, range-normalized kNN runs were pooled, and '
             f'this pooling gives mfeat-zernike its counted source.'),
        (S3, 'The hepatitis C virus (HCV) data have no OpenML evaluations and no benchmark-table entry.'),
        (S3, f'The Gap column, which the correlation of Section~\\ref{{supp:training}} uses, averages every available source, '
             f'including OpenML values from fewer than '
             f'{words(ND["rules"]["runs"])} kNN runs. Counting only the sources of the stratum rule, and the single uncounted source of '
             f'{name("climate_model_simulation_crashes")}, which has no counted one, leaves the ranks of the {words(sp["n"])} gaps '
             f'unchanged.'),
        # S3: benchmark
        (S3, f'On the further datasets ArrowFlow chose such a first layer on {ties_f} of the {k * n_outer} outer folds.'),
        (S3, f'The {k17 * len(COMPARATORS)} differences are descriptive and carry no multiplicity adjustment'),
        (S3, f'Against that comparator ArrowFlow is less accurate on all {words(k17)} datasets, by ${gap[dmin]:.2f}$ ({name(dmin)}) to '
             f'${gap[dmax]:.2f}$ pp ({name(dmax)}).'),
        (S3, f'The interval excludes zero on {words(len(C["clear"]))} datasets: {listing([name(d) for d in C["clear"]], "and")}.'),
        (S3, f'Table~\\ref{{tab:s-ranks}} ranks the {words(len(mr))} tuned models within each of the {words(k17)} datasets.'),
        (S3, f'Table~\\ref{{tab:s-duplicates}} counts the exact duplicate rows of the {words(k17)} datasets'),
        # S3: training controls
        (S3, f'Table~\\ref{{tab:s-training}} gives the complete record of the {len(keys)} training contrasts of '
             f'Section~\\ref{{sec:training-controls}} in their three registered families'),
        (S3, f'In Table~\\ref{{tab:s-training}}, {favor} of the {len(keys)} point estimates favor ArrowFlow, and no exception exceeds '
             f'${pct(largest_exception, 2)}$ pp.'),
        (S3, f'After the adjustment over all {len(keys)} contrasts, only the {words(len(C["survivors"]))} gains on {name(a2)} and '
             f'{name(b2)} stay below 0.05. The gain over tuned input footrule kNN on {LABEL[vd]}, significant within the benchmark '
             f'family, has $p={pval(h34[(vd, INPUT_KNN)])}$.'),
        (S3, f'On the further datasets the difference between the strata comes mainly from {name(a2)} and {name(b2)}, the only datasets '
             f'with Holm-significant gains, while {listing([name(d) for d in small_marked], "and")}, the two smallest further datasets, '
             f'show small effects with wide intervals.'),
        (S3, f'Of the {mt["splits"]} splits of the {words(k)} further datasets into {words(len(marked))} and {words(len(others))}, '
             f'{mt["at_least_observed"]} reach the observed difference of ${signed2(mt["statistic"])}$ pp between the mean training '
             f'effects'),
        (S3, f'In Table~\\ref{{tab:s-training-depth}}, ArrowFlow chose {widths_label(single)} on {pooled[("all", single)]["n"]} outer '
             f'folds and {widths_label(double)} on {pooled[("all", double)]["n"]}. It never chose {widths_label(double)} on '
             f'{listing([name(d) for d in never], "or")}.'),
        (S3, f'The post hoc pooled training effect is ${signed2(pooled[("all", single)]["mean"])}$ pp at {widths_label(single)} and '
             f'${signed2(pooled[("all", double)]["mean"])}$ pp at {widths_label(double)}. But the two-layer folds come from '
             f'{words(len(pooled[("all", double)]["datasets"]))} datasets only, so depth and dataset are confounded.'),
        (S3, f'The untrained ArrowFlow chose the same widths as ArrowFlow on {pooled[("all", single)]["same"]} of the '
             f'{pooled[("all", single)]["n"]} single-layer folds and on {pooled[("all", double)]["same"]} of the '
             f'{pooled[("all", double)]["n"]} two-layer folds.'),
        (S3, f'On the further datasets, footrule kNN on the encoded rankings has higher mean error than numeric kNN on the projected '
             f'scores on {words(len(worse))} datasets, by up to ${pct(sortgap[worst])}$ pp on {name(worst)}, and lower on '
             f'{words(len(better))}, by up to ${pct(-sortgap[reverse])}$ pp on {name(reverse)}. This comparison is descriptive'),
        # S4
        (S4, f"Its within-fold seed SD lies between ${pct(bench['seed_low'][1])}$ ({LABEL[bench['seed_low'][0]]}) and "
             f"${pct(bench['seed_high'][1])}$ points ({LABEL[bench['seed_high'][0]]}) and is below its outer-fold SD on every dataset."),
        # S5
        (S5, f'The combined analysis of the {words(k17)} datasets, the source of their comparator, rank and duplicate-row tables, ran '
             f'at revision \\texttt{{{ready["commit"][:9]}}} and fits no model.'),
        (S5, f"A search of the parent project's working tree, git history and binary documents found no earlier use of the {words(k)} "
             f'datasets. The table renderer rescans ' + listing([f'\\texttt{{{x}/}}' for x in OLDER_WORK], 'and') + ' on every render.'),
        (S5, f'Table~\\ref{{tab:s-datasets}} lists the {words(k17)} datasets with their sources and content hashes.'),
        (S5, 'The selection document and the production scripts are the tracked copies in \\texttt{docs/superpowers/}.'),
        (S5, '\\texttt{<runs directory>} holds the run directories under their names. Without \\texttt{--runs}, \\texttt{holistic} and '
             "\\texttt{render\\_tables.py} default to a path on the author's machine, so always pass \\texttt{--runs}."),
        (S3, f'reports the matched study on the {words(len(matched_facts["datasets"]))} benchmark datasets at every architecture'),
        (S3, 'Per dataset, they chose the same settings on ' + listing(per_dataset, 'and') + '.'),
        (S3, f'the split is descriptive. Its rows for the benchmark datasets follow the training protocol, and those for the further '
             f'datasets and for all {words(k17)} are post hoc.'),
        (S5, 'Outside the repository the renderer reads only the workspace directory \\texttt{runs}. There it reads the run directories '
             + listing([f'\\texttt{{{x}}}' for x in FACTS['run_inputs'] if not x.endswith('.log')], 'and') + '. In \\texttt{runs} it also reads the production logs '
             + listing([f'\\texttt{{{x}}}' for x in FACTS['run_inputs'] if x.endswith('.log')], 'and') + '.'),
        (S5, f'The combined analysis of the {words(k17)} datasets, with its Friedman test and its Holm adjustment over all {len(keys)} '
             f'training contrasts, was written on {ready["written_utc"][:10]}, after every result above was known'),
    ]


def prose_texts():
    """Every reader-visible source: the abstract and sections, the supplement sections, and the figure and table files."""
    files = ['main.tex'] + [f'{sub}/{p.name}' for sub in ('sections', 'supplement_sections', 'figures', 'tables')
                            for p in sorted((HERE / sub).glob('*.tex'))]
    return {f: re.sub(r'\s+', ' ', (HERE / f).read_text()) for f in files}


# ruling (components fill concerns) 5: Segment's change without the checkpoint prints as -0.00, so no sentence may count the datasets
# on which ArrowFlow's variant without the checkpoint is more or less accurate
CHECKPOINT_COUNT = re.compile(r'without (?:the )?(?:validation )?checkpoint[^.]*?(?:less|more) accurate on (?:\d+|[a-z]+)\b')


# a count of datasets with significant gains names the control that carries it (post-restructure review, Critical; scoped review of the
# components fill and final fix round 2). In every reader-visible file (abstract, sections, supplement sections, tables and figure
# captions), a sentence that names the untrained ArrowFlow or the untrained networks and a significance claim, with no other control
# named between them, may give that claim only the datasets whose gains over the untrained ArrowFlow are significant. Adjectives, finite
# verbs, relative clauses, parentheses, a fronted control and a fronted count are all read the same way.
UNTRAINED_MENTION = re.compile(r'\b[Uu]ntrained (?:ArrowFlow|networks?)\b')
SIGNIFICANCE = re.compile(r'\b[Ss]ignificant(?:ly)?\b')
OTHER_CONTROLS = re.compile(r'\b(?:both|two|the|these) controls\b|\beither control\b|\bcontrols that train no filters\b|footrule kNN|'
                            r'nearest-neighbor classifier|\bthe classifier\b|numeric kNN|prototype readout|\bSVC\b|support vector')
SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z\\$(])')
COUNT_OF = {**{w: k for k, w in WORDS.items()}, 'none': 0}
# ruling (combined benchmark framing F-H) 2 and Ruling (final review), FINAL-I2: "competitive" appears at most once, next to the rank facts
COMPETITIVE = re.compile(r'\bcompetitive\b', re.I)
CITATION = re.compile(r'\\cite[a-z]*\*?(?:\[[^\]]*\])*\{[^}]*\}')


def significance_claim(span, dataset, fronted=False):
    """(count, dataset names) of the first 'on <count>' or 'on <datasets>' in span; a fronted count skips 'on k of n'."""
    for m in re.finditer(r'\b[Oo]n(?:~|\s)+(?:only\s+)?([^\s,.;:()~]+)', span):
        token = m.group(1).strip('${}').lower()
        if fronted and re.match(r'\s+of\b', span[m.end():]):
            continue
        if token in COUNT_OF or token.isdigit():
            return (COUNT_OF[token] if token in COUNT_OF else int(token)), []
        names, pos = [], m.start(1)
        while True:
            d = dataset.match(span, pos)
            if not d:
                break
            names.append(d.group(1))
            sep = re.match(r',?\s+(?:and|or)\s+|,\s+', span[d.end():])
            if not sep:
                break
            pos = d.end() + sep.end()
        if names:
            return None, names
    return None, []


def attribution_problems(texts):
    facts = FACTS.get('attribution')
    if not facts:
        return []
    dataset = re.compile('(' + '|'.join(re.escape(x) for x in sorted(facts['all_names'], key=len, reverse=True)) + r')(?![\w-])')
    problems = []
    for f, text in texts.items():
        for sentence in SENTENCE_BREAK.split(text):
            if not (UNTRAINED_MENTION.search(sentence) and SIGNIFICANCE.search(sentence)):
                continue
            has_b, has_f = 'benchmark' in sentence, 'further' in sentence
            scope = 'benchmark' if has_b and not has_f else 'further' if has_f and not has_b else 'all'
            for u in UNTRAINED_MENTION.finditer(sentence):
                for s in SIGNIFICANCE.finditer(sentence):
                    lo, hi = (u.end(), s.start()) if u.start() < s.start() else (s.end(), u.start())
                    if OTHER_CONTROLS.search(sentence, lo, hi):
                        continue
                    end = sentence.find(';', s.end())
                    end = len(sentence) if end < 0 else end
                    other = OTHER_CONTROLS.search(sentence, s.end(), end)
                    count, names = significance_claim(sentence[s.end():other.start() if other else end], dataset)
                    if count is None and not names:
                        count, names = significance_claim(sentence[sentence.rfind(';', 0, s.start()) + 1:s.start()], dataset, fronted=True)
                    if (count is not None and count != facts['count'][scope]) or (names and not set(names) <= set(facts['names'])):
                        problems.append((f, sentence[:160]))
    return sorted(set(problems))


def claim_files(f):
    """A claim names one reader-visible file, or several in which the phrase may stand (a rendered caption and the
    section whose prose repeats it); the phrase must be found in at least one of them."""
    return (f,) if isinstance(f, str) else tuple(f)


def check_prose_claims(texts, claims, quiet=False):
    missing = [(f, phrase) for f, phrase in claims if not any(phrase in texts[x] for x in claim_files(f))]
    forbidden = [(f, w) for f, text in texts.items() for w in FORBIDDEN if w in text.lower()]
    attribution = FACTS.get('attribution')
    misattributed = attribution_problems(texts)
    checkpoint_counts = [(f, m.group(0)) for f, text in texts.items() for m in CHECKPOINT_COUNT.finditer(text)]
    competitive = [(f, len(COMPETITIVE.findall(CITATION.sub(' ', text)))) for f, text in texts.items()]
    competitive = [x for x in competitive if x[1]]
    if missing or forbidden or misattributed or checkpoint_counts or sum(k for _, k in competitive) > 1:
        problems = [f'{len(missing)} phrases not found {missing[:2]}', f'forbidden wording {forbidden[:3]}']
        if misattributed:
            problems.append(f'significant gains over the untrained ArrowFlow on {words(attribution["count"]["all"])} datasets, but the '
                            f'text attributes another count to it {misattributed[:2]}')
        if checkpoint_counts:
            problems.append(f'a count of datasets by the sign of the change without the checkpoint, which printed precision does '
                            f'not support {checkpoint_counts[:2]}')
        if sum(k for _, k in competitive) > 1:
            problems.append(f'"competitive" appears {sum(k for _, k in competitive)} times, but at most once {competitive[:3]}')
        raise AssertionError('the text differs from the run files or uses a forbidden term: ' + '; '.join(problems))
    if not quiet:
        print(f'check guarded prose: {len(claims)} phrases rebuilt from the run files found in '
              f'{len({x for f, _ in claims for x in claim_files(f)})} files; no forbidden term in {len(texts)} files; '
              f'significant gains attributed to the '
              f'untrained ArrowFlow checked in every sentence of {len(texts)} files; "competitive" used '
              f'{sum(k for _, k in competitive)} time(s); no count by the sign of the change without the checkpoint')


def prose_mutation_test(texts, claims, extra=()):
    """A changed count or number in in-memory copies of the text, or a forbidden term, must make the check fail."""
    # the simplification pass moved the detail sentences of Section 7 to the supplement; their cases moved with them
    cases = [('supplement_sections/S3_results.tex', 'on 13 of the 17 datasets, among them', 'on 14 of the 17 datasets, among them'),
             ('supplement_sections/S4_ablations.tex', '($+13.7$ pp)', '($+13.8$ pp)'),
             ('supplement_sections/S4_ablations.tex', '17 of the 105 outer folds', '18 of the 105 outer folds'),
             ('supplement_sections/S3_results.tex', 'Of the 70 two-layer folds', 'Of the 71 two-layer folds'),
             ('supplement_sections/S4_ablations.tex', 'the target-aware first view, on twelve datasets',
              'the target-aware first view, on eleven datasets'),
             ('sections/07_experiments.tex', 'Seven views are more accurate than the first, target-aware view alone on 12 of the 17.',
              'Seven views are more accurate than the first, target-aware view alone on 11 of the 17.'),
             ('supplement_sections/S5_reproducibility.tex', 'then two training-only pilots', 'then three training-only pilots'),
             ('sections/09_conclusion.tex', 'remains open.', 'remains open; this is an independent test.'),
             ('supplement_sections/S3_results.tex', '$p=0.033$), but cannot separate', '$p=0.034$), but cannot separate'),
             # a count of significant datasets attributed to the untrained ArrowFlow alone, added to the Introduction with every
             # guarded phrase intact, must fail through the attribution check alone
             ('sections/01_introduction.tex', 'and Section~\\ref{sec:experiments} reports the experiments.',
              'and Section~\\ref{sec:experiments} reports the experiments. Training adds a modest gain over an untrained ArrowFlow, '
              'significant within the registered families on three datasets.', 'attributes another count'),
             ('sections/09_conclusion.tex', 'remains open.', 'remains open. Training adds a modest gain over an untrained '
              'ArrowFlow, significant within the registered families on three datasets.', 'attributes another count'),
             ('supplement_sections/S4_ablations.tex', 'No interval of either variant excludes zero.',
              'No interval of either variant excludes zero. The variant without the checkpoint is more accurate on six datasets.',
              'sign of the change without the checkpoint'),
             # the scoped review's paraphrases, each detected by the attribution check alone: a relative clause in the Conclusion, a
             # fronted control in the Discussion and the old Introduction wording in S3; then a parenthesis in a table caption, the
             # untrained networks with a finite verb in S4, and a fronted count in the abstract
             ('sections/09_conclusion.tex', 'remains open.', 'remains open. Training adds a modest gain over an untrained ArrowFlow '
              'that is significant within the registered families on three datasets.', 'attributes another count'),
             ('sections/08_discussion.tex', "and the voting-rule study is not a test of Arrow's theorem.",
              "and the voting-rule study is not a test of Arrow's theorem. Against the untrained ArrowFlow, training gains are "
              'significant within the registered families on three datasets.', 'attributes another count'),
             ('supplement_sections/S3_results.tex', 'ArrowFlow is separated from no model other than the SVC.',
              'ArrowFlow is separated from no model other than the SVC. Training adds a modest gain over an untrained ArrowFlow, '
              'significant within the registered families on three datasets.', 'attributes another count'),
             ('tables/tab_training.tex', 'which is not corrected for multiple comparisons.', 'which is not corrected for multiple comparisons. '
              'Training adds a modest gain over the untrained ArrowFlow (significant within the registered families on three datasets).',
              'attributes another count'),
             ('supplement_sections/S4_ablations.tex', 'so depth and dataset are confounded, and the split is descriptive.',
              'so depth and dataset are confounded, and the split is descriptive. The gains over the untrained networks are significant '
              'on three datasets.', 'attributes another count'),
             ('main.tex', 'than never moving its hidden filters.',
              'than never moving its hidden filters. On three datasets, its gains over the untrained ArrowFlow alone are '
              'significant.', 'attributes another count'),
             # the Introduction's one permitted use of "competitive" was removed by the corrections pass, so the mutation that
             # must trip the at-most-once rule now adds two occurrences
             ('sections/09_conclusion.tex', 'remains open.',
              'remains open. By mean rank ArrowFlow is competitive, and competitive with the tuned models.', 'competitive'),
             # the new guarded numbers: the closeness counts, the rank facts, the laboratory numbers and the Table S27 pointer
             ('sections/08_discussion.tex', 'on 12 of the 17 datasets.', 'on 13 of the 17 datasets.'),
             ('supplement_sections/S3_results.tex', 'on 12 of the 17 datasets, five of them', 'on 12 of the 17 datasets, six of them'),
             ('sections/01_introduction.tex', 'fourth among the six tuned models', 'third among the six tuned models'),
             ('sections/09_conclusion.tex', 'lies between random forest and gradient boosting;', 'lies between the MLP and random forest;'),
             ('supplement_sections/S4_ablations.tex', 'beat the footrule median by 14 to 21 points', 'beat the footrule median by 14 to 22 points'),
             ('supplement_sections/S4_ablations.tex', 'and Wine quality ($+14.15$ pp)', 'and Wine quality ($+14.16$ pp)')]
    cases += list(extra)
    for f, old, new, *alone in cases:
        if texts[f].count(old) != 1:
            raise SystemExit(f'prose mutation test: {old!r} occurs {texts[f].count(old)} times in {f}, expected once')
        mutated = dict(texts)
        mutated[f] = texts[f].replace(old, new, 1)
        try:
            check_prose_claims(mutated, claims, quiet=True)
        except AssertionError as exc:
            if alone and (': 0 phrases not found [];' not in str(exc) or alone[0] not in str(exc)):
                raise SystemExit(f'prose mutation test ({f}: {old} -> {new}): not detected by that check alone: {exc}')
            print(f'prose mutation test ({f}: {old} -> {new}): detected')
        else:
            raise SystemExit(f'prose mutation test ({f}: {old} -> {new}): the prose check did not fail')


# ----------------------------------------------------------------------------- production scripts of the later families (S5)
def production_script(runs, log_name, script_name, tag, replace):
    """Stage times, the pinned commit and the command lines of a production log and its launch script. `replace` maps a
    script word to the placeholder that Section S5 prints; other words are kept. The log is read from the runs directory and
    the script from its tracked copy in docs/superpowers."""
    text, script = (runs / log_name).read_text(), (PRODUCTION_SCRIPTS / script_name).read_text()
    start = re.search(rf'^{tag} start (\S+) HEAD (\w+)$', text, re.M)
    end = re.search(rf'^{tag} end (\S+) OK$', text, re.M)
    status = re.search(rf'^{tag} exit=(\d+)$', text, re.M)
    pinned = re.search(r'test "\$\(git rev-parse HEAD\)" = (\w+)', script)
    if not (start and end and status and pinned) or int(status.group(1)) != 0 or not pinned.group(1).startswith(start.group(2)):
        raise AssertionError(f'{log_name}: no start, end or exit status 0, or its script is not pinned to the logged commit')
    variables = dict(re.findall(r'^([A-Z][A-Z0-9]*)=([^$"\s]\S*)$', script, re.M))
    commands = []
    for module, args in re.findall(r'"\$PY" -m (experiments\.make_revision\.\w+) (.*?)(?: > \S+ 2>&1)?$', script, re.M):
        tokens = tuple(token for word in args.split() for token in replace(word, variables).split())
        if any('$' in token or '"' in token for token in tokens):
            raise AssertionError(f'{script_name}: an unresolved variable in {module} {args}')
        commands.append((module, tokens))
    return dict(start=start.group(1), head=start.group(2), pinned=pinned.group(1), end=end.group(1),
                exit=int(status.group(1)), stages=dict(re.findall(r'^== (.+?) (\d\d:\d\d:\d\d)Z$', text, re.M)),
                commands=commands)


def protocol_path(path):
    return 'P/' + path.split('experiments/make_revision/protocols/', 1)[1]


def run_source(block, fam):
    """A run that an analysis record names must be the run on disk: protocol, protocol file, code revision and summary."""
    if (block['protocol_id'], block['protocol_sha256'], block['code_revision'], block['summary_sha256']) != \
            (fam.protocol['protocol_id'], sha256_file(fam.dir / 'protocol.json'), fam.environment['code_revision'],
             sha256_file(fam.dir / 'summary.json')):
        raise AssertionError(f'{fam.name}: the run that the analysis record names is not the run on disk')


def fold_contrast(fam_a, model_a, fam_b, model_b, d):
    """Fold differences (fitting seeds averaged within each fold) and the exact mean difference of two models on one dataset."""
    a, b = fold_accuracy(fam_a, model_a, d), fold_accuracy(fam_b, model_b, d)
    if set(a) != set(b):
        raise AssertionError(f'{d}: {model_a} and {model_b} do not cover the same outer folds')
    return [float(a[x] - b[x]) for x in sorted(a)], sum(a.values()) / len(a) - sum(b.values()) / len(b)


def check_interval(r, diffs, exact, where, p_key='p_approximate'):
    """A recorded paired difference must equal the corrected resampled t over the fold differences of the model rows."""
    _, se, lo, hi, p = corrected_t(diffs, FACTS['confidence'])
    if abs(float(exact) - r['mean_difference']) > 1e-12 or \
            max(abs(se - r['standard_error']), abs(lo - r['ci_low']), abs(hi - r['ci_high'])) > 1e-9 or \
            (p_key is not None and abs(p - r[p_key]) > 1e-9):
        raise AssertionError(f'{where}: differs from the fold differences of the model rows')


# ----------------------------------------------------------------------------- sorting control (Section 7.3, S3)
PROJECTED = 'projected_numeric_knn'
SORTING, PROJECTED_GAIN, RAW_CONTRAST = f'{INPUT_KNN}_vs_{PROJECTED}', f'{KNN}_vs_{PROJECTED}', f'numeric_knn_vs_{PROJECTED}'
PROJECTED_FIRST = {SORTING: 'Tuned input footrule kNN', PROJECTED_GAIN: 'ArrowFlow', RAW_CONTRAST: 'Numeric kNN, raw features'}
LADDER = [('numeric_knn', 'knn'), (PROJECTED, 'projected'), (INPUT_KNN, 'training'), (UNTRAINED, 'training'), (KNN, 'knn')]
LADDER_HEAD = [two_lines('Numeric kNN,', 'raw features'), two_lines('Numeric kNN,', 'projected scores'),
               two_lines('Tuned input', 'footrule kNN'), two_lines('Untrained', 'ArrowFlow'), 'ArrowFlow']


def render_projected(projected, knn, training):
    """The sorting control (Section 7.3, S3): its family of fourteen from compare_runs projected with every member
    recomputed from the model rows of the three runs, the descriptive raw-feature rows, the error ladder from the three
    summaries, and the outer folds on which numeric kNN on the projected scores and tuned input footrule kNN chose the same
    encoder settings, from the selection records of both runs."""
    cdir = projected.dir / 'compare_projected'
    cj, lj = read_json(cdir / 'projected_contrasts.json'), read_json(cdir / 'projected_ladder_error_table.json')
    sealed(cj, cdir, ['projected_contrasts.csv', 'projected_raw_descriptive.csv'])
    c, raw = pd.read_csv(cdir / 'projected_contrasts.csv'), pd.read_csv(cdir / 'projected_raw_descriptive.csv')
    p, n, n_outer, conf = projected.protocol, len(DATASETS), FACTS['n_outer'], FACTS['confidence']
    size, fams = p['primary_family_size'], {'knn': knn, 'training': training, 'projected': projected}
    if sha256_file(PROTOCOLS_V3 / 'knn_projected.json') != sha256_file(projected.dir / 'protocol.json') or p.get('frozen') is not True:
        raise AssertionError('knn_projected: the protocol of the run is not the frozen protocol file')
    if p['datasets'] != DATASETS or p['primary_contrasts'] != [SORTING, PROJECTED_GAIN] or size != 2 * n:
        raise AssertionError('knn_projected protocol: datasets, contrasts or family size differ from the text')
    for key in ('split_seed', 'outer_folds', 'outer_repeats', 'inner_folds', 'fit_seeds', 'candidate_budget', 'candidate_tie_rule',
                'selection_metric', 'test_train_ratio', 'confidence'):
        if not p[key] == training.protocol[key] == knn.protocol[key]:
            raise AssertionError(f'knn_projected: {key} differs from the ArrowFlow and training runs')
    model = p['projected_control']['model']
    src = (REPO / 'experiments' / 'make_revision' / 'projected_knn.py').read_text()
    if "NUMERIC_READOUT_GRID = {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance'], 'p': [1, 2]}" not in src \
            or not model['representation'].endswith('no further standardization') or 'PresortMismatch' not in model['presort_check']:
        raise AssertionError('knn_projected: the control no longer reads the unstandardized array that each encoder sorts, '
                             'with the Minkowski exponent in its readout grid')
    for block, fam in fams.items():
        run_source(cj['provenance'][block], fam)
        run_source(lj['sources'][block], fam)
        if cj['provenance']['verification'][block]['jobs_verified'] != fam.planned or fam.completed != fam.planned:
            raise AssertionError(f'projected_contrasts.json: the {block} run was not re-verified on every planned job')
    f = cj['family']
    if (f['size'], f['multiplicity'], f['metric'], f['n_folds'], f['df'], f['test_train_ratio'], round(100 * f['confidence'])) != \
            (size, 'holm', 'accuracy', n_outer, FACTS['df'], FACTS['ratio'], conf):
        raise AssertionError('projected_contrasts.json: family settings differ from the captions')
    if list(zip(c['dataset'], c['contrast'])) != [(d, x) for d in DATASETS for x in (SORTING, PROJECTED_GAIN)] or \
            list(c['family_index']) != list(range(1, size + 1)) or set(c['model_b']) != {PROJECTED}:
        raise AssertionError('projected_contrasts.csv: members or order differ from the frozen family')
    if any(abs(a - b) > 1e-12 for a, b in zip(holm(list(c['p_approximate'])), c['holm_p_approximate'])):
        raise AssertionError('projected_contrasts.csv: the Holm column is not one adjustment over the family')
    by_member = {(r['dataset'], r['contrast']): r for r in cj['contrasts']}
    first = {SORTING: (training, INPUT_KNN), PROJECTED_GAIN: (knn, KNN)}
    C = {}
    for _, r in c.iterrows():
        d, x = r['dataset'], r['contrast']
        if any(abs(float(r[key]) - float(by_member[(d, x)][key])) > 1e-15 for key in
               ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate')):
            raise AssertionError(f'{d}/{x}: the contrasts CSV and JSON differ')
        check_interval(r, *fold_contrast(*first[x], projected, PROJECTED, d), f'projected_contrasts.csv/{d}/{x}')
        C[(d, x)] = r
    if list(raw['dataset']) != DATASETS or set(raw['contrast']) != {RAW_CONTRAST} or \
            {'p_approximate', 'holm_p_approximate'} & set(raw.columns) or not cj['descriptive']['status'].startswith('descriptive'):
        raise AssertionError('projected_raw_descriptive.csv: rows, columns or status differ from the descriptive record')
    RAW = {}
    for _, r in raw.iterrows():
        check_interval(r, *fold_contrast(knn, 'numeric_knn', projected, PROJECTED, r['dataset']),
                       f'projected_raw_descriptive.csv/{r["dataset"]}', p_key=None)
        RAW[r['dataset']] = r
    L = {}
    for d in DATASETS:
        if [(x['model_id'], x['source_run']) for x in lj['rows'][d]] != LADDER:
            raise AssertionError(f'projected_ladder_error_table.json/{d}: its rungs differ from the ladder')
        for x in lj['rows'][d]:
            s = lookup(fams[x['source_run']].summary['summaries'][d], x['model_id'], 'error')
            for field, key in (('mean_error', 'mean'), ('outer_fold_sd', 'outer_fold_sd'),
                               ('mean_within_fold_seed_sd', 'mean_within_fold_seed_sd')):
                if (x[field] is None) != (s.get(key) is None) or (x[field] is not None and abs(x[field] - s[key]) > 1e-12):
                    raise AssertionError(f'projected_ladder_error_table.json/{d}/{x["model_id"]}: {field} differs from its summary')
            L[(d, x['model_id'])] = x
    cp, ct = read_json(projected.dir / 'candidates.json')[PROJECTED], read_json(training.dir / 'candidates.json')[INPUT_KNN]
    if (cp['candidates'], cp['config_ids']) != (ct['candidates'], ct['config_ids']) or len(cp['config_ids']) != model['candidates']:
        raise AssertionError('numeric kNN on the projected scores and tuned input footrule kNN do not share their encoder candidates')
    rp, rt = selection_records(projected, PROJECTED), selection_records(training, INPUT_KNN)
    if sorted(rp) != sorted(rt):
        raise AssertionError('the selection records of the two classifiers do not cover the same outer folds')
    same = {d: sum(rp[x]['config_id'] == rt[x]['config_id'] for x in rp if x[0] == d) for d in DATASETS}

    err = lambda d, m: pm(L[(d, m)]['mean_error'], L[(d, m)]['outer_fold_sd'])
    rows, subheads = [], {0: f'Family of {words(size)} contrasts, one Holm adjustment'}
    for d in DATASETS:
        for x in (SORTING, PROJECTED_GAIN):
            r = C[(d, x)]
            rows.append([str(int(r['family_index'])), LABEL[d], PROJECTED_FIRST[x], err(d, first[x][1]), err(d, PROJECTED),
                         signed(r['mean_difference']), pct(r['standard_error']), interval(r['ci_low'], r['ci_high']),
                         pval(r['p_approximate']), pval(r['holm_p_approximate'])])
    subheads[len(rows)] = 'Descriptive: numeric kNN on the raw features $-$ numeric kNN on the projected scores'
    for d in DATASETS:
        r = RAW[d]
        rows.append(['--', LABEL[d], PROJECTED_FIRST[RAW_CONTRAST], err(d, 'numeric_knn'), err(d, PROJECTED),
                     signed(r['mean_difference']), pct(r['standard_error']), interval(r['ci_low'], r['ci_high']), '--', '--'])
    caption = (f'\\textbf{{Sorting control: the family of {words(size)} contrasts.}} Each row pairs a first-named classifier with '
               f'numeric kNN on the projected scores on one of the {words(n)} benchmark datasets. That classifier reads the scores of its '
               f'{words(FACTS["views"])} encoders before they are sorted, with their column scales, and its views vote by majority. It '
               f'chooses its encoder settings on inner folds from the {words(model["candidates"])} candidates of tuned input footrule '
               f'kNN, and its readout tunes the neighbor count, the weighting and the Minkowski exponent inside every fit. The '
               f'errors are mean outer-fold errors in percent (outer-fold SD). Each difference is the first-named classifier\'s '
               f'accuracy minus that of numeric kNN on the projected scores, in percentage points, with its standard error (SE). '
               f'The last columns give the {interval_note()}, the approximate $p$ value and the Holm-adjusted $p$ value over the '
               f'family. The last block is descriptive: numeric kNN on the raw features comes from the ArrowFlow run with one fitting '
               f'seed and also differs in its views and in where it is tuned. Protocol \\texttt{{{tex(p["protocol_id"])}}}.')
    write_table('tab_s_projected', caption, 'tab:s-projected',
                ['\\#', 'Dataset', 'First-named', 'Its error', two_lines('Error, numeric kNN', 'on projected scores'),
                 'Diff. (pp)', 'SE', f'{conf}\\% interval', '$p$', 'Holm $p$'], rows, 'rlllrrrrrr',
                subheads=subheads, size='\\scriptsize', wide=True)
    # the ladder of the benchmark datasets is printed once, with the further datasets, in Table S-ladder (render_combined); the outer
    # folds on which the two classifiers chose the same encoder settings are guarded in the text of Section S3
    return dict(C=C, RAW=RAW, L=L, same=same, size=size, protocol=p, candidates=model['candidates'],
                sorting={d: float(C[(d, SORTING)]['mean_difference']) for d in DATASETS},
                gain={d: C[(d, PROJECTED_GAIN)] for d in DATASETS})


# ----------------------------------------------------------------------------- additional datasets (Section 7.4, S3)
NEWDATA = ['balance_scale', 'mfeat_zernike', 'ionosphere', 'vertebra_column', 'diabetes', 'banknote_authentication',
           'qsar_biodeg', 'steel_plates_fault', 'climate_model_simulation_crashes', 'hcv_egyptian_patients']
HCV = 'hcv_egyptian_patients'
NEW_LABEL = {**{d: d.replace('_', '-') for d in NEWDATA}, HCV: 'HCV'}
NEW_MODELS = [KNN, UNTRAINED, INPUT_KNN, PROJECTED] + COMPARATORS + ['dummy']
NEW_RUNGS = [('raw_numeric_knn', 'numeric_knn'), ('unsorted_projected_knn', PROJECTED), ('encoded_ranking_knn', INPUT_KNN),
             ('untrained_arrowflow_knn', UNTRAINED), ('arrowflow_knn', KNN)]
NEW_COMPARATORS = ['numeric_knn', 'svc_rbf', 'random_forest', 'mlp', 'gradient_boosting', 'dummy']
NEW_FAMILIES = (('primary', UNTRAINED, 'primary_family'), ('secondary', INPUT_KNN, 'secondary_family'))
NEW_STRATUM = {'H': 'marked', 'C': 'other'}


def selection_rules(path, panel):
    """Eligibility rules, funnel counts and informative flags of the external evidence, read from the frozen selection
    document that the batch protocols name. Every pinned dataset must satisfy the printed rules, the funnel must end in the
    panel, the pool table must list every pinned source value, and the counted sources must reproduce the strata."""
    text = Path(path).read_text()
    funnel = {s: (crit, int(k.replace(',', ''))) for s, crit, k in re.findall(r'^\| ([A-D]) \| (.+?) \| (\d[\d,]*) \|$', text, re.M)}
    if sorted(funnel) != list('ABCD'):
        raise AssertionError('selection document: no funnel of steps A to D')
    a, b = funnel['A'][0], funnel['B'][0]
    found = dict(rows=re.search(r'NumberOfInstances (\d[\d,]*) to (\d[\d,]*)', a),
                 features=re.search(r'NumberOfFeatures (\d+) to (\d+) \(target included\)', a),
                 classes=re.search(r'NumberOfClasses (\d+) to (\d+)', a),
                 minority=re.search(r'MinorityClassSize at least (\d+)', b),
                 missing=re.search(r'missing values in at most (\d+)% of cells', b),
                 duplicates=re.search(r'Exact duplicate feature rows are at most (\d+)% of rows', text),
                 gap=re.search(r'at least one informative source shows a gap of ([\d.]+) points or more', text),
                 runs=re.search(r'S1 is informative when the standardized-kNN family has at least (\d+) runs', text),
                 frame=re.search(r'OpenML-CC18 \(study 99, 72 datasets\).*?OpenML100 \(study 14, 100 datasets\).*?OpenML copies of '
                                 r'UCI datasets', text, re.S))
    if not all(found.values()):
        raise AssertionError(f'selection document: rules not found: {[x for x, v in found.items() if not v]}')
    num = lambda m, i=1: int(m.group(i).replace(',', ''))
    rules = dict(rows=(num(found['rows']), num(found['rows'], 2)),
                 features=(num(found['features']) - 1, num(found['features'], 2) - 1),
                 classes=(num(found['classes']), num(found['classes'], 2)), minority=num(found['minority']),
                 missing=num(found['missing']), duplicates=num(found['duplicates']), gap=float(found['gap'].group(1)),
                 runs=num(found['runs']), funnel=[funnel[s][1] for s in 'ABCD'])
    for e in panel:
        n_rows, n_features = e['shape']
        if not (rules['rows'][0] <= n_rows <= rules['rows'][1] and rules['features'][0] <= n_features <= rules['features'][1]
                and rules['classes'][0] <= len(e['class_counts']) <= rules['classes'][1]
                and min(e['class_counts']) >= rules['minority'] and e['n_missing'] == 0
                and 100 * e['duplicates']['duplicate_rows'] <= rules['duplicates'] * n_rows):
            raise AssertionError(f'{e["name"]}: the pinned dataset breaks an eligibility rule of the selection document')
    if rules['funnel'][-1] != len(panel) or rules['funnel'] != sorted(rules['funnel'], reverse=True):
        raise AssertionError('selection document: the funnel does not end in the pinned panel')
    pool = text.split('## 2. Eligible pool', 1)[1].split('## 3.', 1)[0]
    informative = {}
    for e in panel:
        cells = [[c.strip() for c in line.split('|')[1:-1]] for line in pool.splitlines() if line.startswith(f'| {e["data_id"]} |')]
        cells = [x for x in cells if len(x) == 11]
        if len(cells) != 1:
            raise AssertionError(f'selection document: no single pool row for data id {e["data_id"]}')
        evidence, label = cells[0][9], cells[0][10]
        sources = e['external_gap']['sources']
        if (label == 'H') != (e['stratum'] == 'H') or (not sources) != evidence.startswith('None') or \
                any(f'{v:g}' not in evidence for v in sources.values()):
            raise AssertionError(f'{e["name"]}: its pool row disagrees with the pinned stratum or sources')
        informative[e['name']] = {x: not (x.startswith('openml') and 'S1 not informative' in evidence) for x in sources}
        counted = [v for x, v in sources.items() if informative[e['name']][x]]
        if (e['stratum'] == 'H') != (bool(counted) and min(counted) >= rules['gap']):
            raise AssertionError(f'{e["name"]}: the counted sources do not reproduce its stratum')
    return rules, informative


def render_newdata(runs, knn, training, projected):
    """The ten additional datasets (Section 7.4, S3): both Holm families recomputed from the model rows of the two batch
    runs, the stratum permutation test over every split and the exact Spearman correlation recomputed from the effects and
    the pinned gaps, the ladder, comparator intervals, balanced accuracy, duplicate audit and selected depths from the batch
    summaries and selection records, and the pins and eligibility rules from the frozen protocols and selection document."""
    from itertools import combinations, permutations
    from scipy import stats
    adir = runs / '2026-09-14-newdata-analysis'
    B = {1: Family('newdata batch 1', runs / '2026-09-14-newdata-batch1'),
         2: Family('newdata batch 2', runs / '2026-09-14-newdata-batch2')}
    aj = read_json(adir / 'newdata_analysis.json')
    sealed(aj, adir, ['newdata_families.csv', 'newdata_comparators.csv', 'newdata_ladder.csv', 'newdata_duplicates.csv'])
    n_outer, conf, k = FACTS['n_outer'], FACTS['confidence'], len(NEWDATA)
    for b, fam in B.items():
        frozen, prov = PROTOCOLS_V3 / f'newdata_batch{b}.json', aj['provenance'][f'batch{b}']
        if fam.summary is None or fam.protocol.get('frozen') is not True or sha256_file(frozen) != sha256_file(fam.dir / 'protocol.json'):
            raise AssertionError(f'{fam.name}: no verified summary, or its protocol is not the frozen batch protocol')
        if (prov['protocol_id'], prov['protocol_sha256'], prov['code_revision'], prov['summary_sha256'], prov['planned_jobs']) != \
                (fam.protocol['protocol_id'], sha256_file(frozen), fam.environment['code_revision'],
                 sha256_file(fam.dir / 'summary.json'), fam.planned) or fam.completed != fam.planned or \
                aj['provenance']['verification'][f'batch{b}']['jobs_verified'] != fam.planned:
            raise AssertionError(f'newdata_analysis.json: batch {b} is not the complete, re-verified run on disk')
    p1, p2 = B[1].protocol, B[2].protocol
    if {x for x in set(p1) | set(p2) if p1.get(x) != p2.get(x)} != {'protocol_id', 'batch', 'datasets'} or \
            aj['analysis_declaration'] != p1['analysis'] or p1['analysis']['requires_both_batches_complete'] is not True:
        raise AssertionError('the batch protocols differ outside their batch fields, or the analysis is not the frozen one')
    panel = p1['panel']
    if [e['name'] for e in panel] != NEWDATA or sorted(p1['datasets'] + p2['datasets']) != sorted(NEWDATA) or \
            any(e['name'] != HCV and e['openml_name'] != NEW_LABEL[e['name']] for e in panel):
        raise AssertionError('batch protocols: the panel, the batches or the dataset names differ from the tables')
    runs_of = {d: B[b] for b in B for d in B[b].protocol['datasets']}
    for e in panel:
        m = read_json(runs_of[e['name']].dir / e['name'] / 'manifest.json')
        if (m['dataset_hash'], m['shape'], m['class_counts'], m['openml']['data_id'], m['openml']['version'], m['stratum']) != \
                (e['dataset_hash'], e['shape'], e['class_counts'], e['data_id'], e['version'], e['stratum']):
            raise AssertionError(f'{e["name"]}: the prepared manifest differs from its pin')
    pin = {e['name']: e for e in panel}
    stratum = {d: pin[d]['stratum'] for d in NEWDATA}
    marked, others = [d for d in NEWDATA if stratum[d] == 'H'], [d for d in NEWDATA if stratum[d] != 'H']
    decl = p1['analysis']['moderator_test']
    if decl['strata'] != {'H': marked, 'C': others} or marked != NEWDATA[:len(marked)] or decl['alpha'] != 0.05 or \
            decl['splits'] != math.comb(k, len(marked)) or decl['sided'] != 'one-sided, H greater':
        raise AssertionError('batch protocols: the stratum test differs from the text')
    for key in ('split_seed', 'outer_folds', 'outer_repeats', 'inner_folds', 'fit_seeds', 'candidate_budget', 'candidate_tie_rule',
                'selection_metric', 'test_train_ratio', 'confidence', 'full_method'):
        if p1[key] != knn.protocol[key]:
            raise AssertionError(f'batch protocols: {key} differs from the ArrowFlow run')
    if p1['models'][KNN]['readout']['grid'] != knn.protocol['knn_readout']['grid']:
        raise AssertionError('batch protocols: the readout grid differs from the ArrowFlow run')
    reference = {m: read_json(knn.dir / 'candidates.json')[m] for m in [KNN] + COMPARATORS + ['dummy']}
    reference.update({m: read_json(training.dir / 'candidates.json')[m] for m in CONTROLS})
    reference[PROJECTED] = read_json(projected.dir / 'candidates.json')[PROJECTED]
    for b in B:
        cand = read_json(B[b].dir / 'candidates.json')
        if sorted(cand) != sorted(NEW_MODELS) or any(cand[m] != reference[m] for m in NEW_MODELS):
            raise AssertionError(f'batch {b}: its candidates differ from those of the benchmark runs')
    shown = {'"$OUT1"': '<batch 1 run>', '"$OUT2"': '<batch 2 run>', '"$OUTA"': '<analysis directory>'}
    log = production_script(runs, 'task23b-run.log', 'task23b-prod.sh', 'task23b',
                            lambda word, v: {'$B1': protocol_path(v['B1']), '$B2': protocol_path(v['B2']), **shown}.get(word, word))
    if not (knn.protocol['frozen_at_utc'] < p1['frozen_at_utc'] and p1['frozen_at_utc'][:19] < log['start'][:19]) or \
            not log['pinned'] == B[1].environment['code_revision'] == B[2].environment['code_revision']:
        raise AssertionError('the batch protocols were not frozen after the ArrowFlow protocol and before the batch runs, or the '
                             'batches ran at another revision than the script pins')
    rules, informative = selection_rules(SELECTION_DOCUMENT, panel)

    # both families, every member from the model rows
    fam_csv = pd.read_csv(adir / 'newdata_families.csv')
    F = {}
    for family, control, key in NEW_FAMILIES:
        declared = p1['analysis'][key]
        if (declared['model_a'], declared['model_b'], declared['size'], declared['alpha']) != (KNN, control, k, 0.05):
            raise AssertionError(f'batch protocols: the {family} family differs from the text')
        rows = fam_csv[fam_csv['family'] == family]
        if list(rows['dataset']) != NEWDATA or set(rows['model_a']) != {KNN} or set(rows['model_b']) != {control} or \
                set(rows['n_folds']) != {n_outer} or set(rows['df']) != {FACTS['df']} or list(rows['family_index']) != list(range(1, k + 1)):
            raise AssertionError(f'newdata_families.csv: the {family} family differs from the frozen family')
        if any(abs(x - y) > 1e-12 for x, y in zip(holm(list(rows['p_approximate'])), rows['holm_p_approximate'])):
            raise AssertionError(f'newdata_families.csv: the {family} Holm column is not one adjustment over the {k} datasets')
        js = {r['dataset']: r for r in aj[key]}
        for _, r in rows.iterrows():
            d = r['dataset']
            if r['stratum'] != stratum[d] or (js[d]['test_train_ratio'], js[d]['confidence']) != (FACTS['ratio'], conf / 100) or \
                    any(abs(float(r[x]) - float(js[d][x])) > 1e-15 for x in
                        ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate')):
                raise AssertionError(f'newdata {family}/{d}: the families CSV and the analysis record differ')
            check_interval(r, *fold_contrast(runs_of[d], KNN, runs_of[d], control, d), f'newdata_families.csv/{family}/{d}')
            F[(family, d)] = r

    # the prespecified stratum test over every split, and the descriptive Spearman correlation over every order
    effect = {d: float(F[('primary', d)]['mean_difference']) for d in NEWDATA}
    statistic = lambda group: float(np.mean([effect[d] for d in group]) - np.mean([effect[d] for d in NEWDATA if d not in group]))
    observed, null = statistic(marked), [statistic(g) for g in combinations(NEWDATA, len(marked))]
    at_least = sum(v >= observed - 1e-12 for v in null)
    mt = aj['moderator_test']
    if (mt['splits'], mt['at_least_observed'], mt['larger_than_observed'], mt['alpha']) != \
            (len(null), at_least, sum(v > observed + 1e-12 for v in null), 0.05) or \
            max(abs(mt['p_one_sided'] - at_least / len(null)), abs(mt['statistic'] - observed),
                abs(mt['mean_h'] - np.mean([effect[d] for d in marked])), abs(mt['mean_c'] - np.mean([effect[d] for d in others]))) > 1e-12 \
            or mt['p_at_most_alpha'] != (at_least / len(null) <= 0.05) or abs(decl['smallest_attainable_p'] - 1 / len(null)) > 1e-15:
        raise AssertionError('newdata_analysis.json: the stratum test differs from a recomputation over every split')
    for label, group in (('H', marked), ('C', others)):
        v, es = np.array([effect[d] for d in group]), mt['effect_sizes'][label]
        if es['n'] != len(group) or max(abs(es['mean'] - v.mean()), abs(es['sd'] - v.std(ddof=1)), abs(es['median'] - np.median(v)),
                                        abs(es['min'] - v.min()), abs(es['max'] - v.max())) > 1e-12:
            raise AssertionError(f'newdata_analysis.json: the {label} effect sizes differ from the effects')
    # every leave-one-dataset-out repetition of the same exact test, reported in full so that the sensitivity is not
    # presented one-sidedly (audit finding A32)
    loo = {}
    for d in NEWDATA:
        rest, rest_marked = [x for x in NEWDATA if x != d], [x for x in marked if x != d]
        stat = lambda group, rest=rest: float(np.mean([effect[x] for x in group])
                                              - np.mean([effect[x] for x in rest if x not in group]))
        obs, null_d = stat(rest_marked), [stat(g) for g in combinations(rest, len(rest_marked))]
        loo[d] = sum(v >= obs - 1e-12 for v in null_d) / len(null_d)
    gaps = {d: pin[d]['external_gap']['points'] for d in NEWDATA if pin[d]['external_gap']['points'] is not None}
    with_gap = [d for d in NEWDATA if d in gaps]
    if [d for d in NEWDATA if d not in gaps] != [HCV] or pin[HCV]['external_gap']['sources'] or \
            any(abs(np.mean(list(pin[d]['external_gap']['sources'].values())) - gaps[d]) > 1e-9 for d in with_gap):
        raise AssertionError('batch protocols: a gap is not the mean of its sources, or a dataset other than HCV lacks one')
    x, y = stats.rankdata([gaps[d] for d in with_gap]), stats.rankdata([effect[d] for d in with_gap])
    xc, yc = x - x.mean(), y - y.mean()
    rho = float(xc @ yc / math.sqrt((xc @ xc) * (yc @ yc)))
    orders = np.array(list(permutations(range(len(y)))), dtype=np.int16)
    ranks = y[orders]
    centred = ranks - ranks.mean(axis=1, keepdims=True)
    rhos = centred @ xc / np.sqrt((xc @ xc) * np.sum(centred ** 2, axis=1))
    sp = aj['spearman']
    if sp['datasets'] != with_gap or sp['excluded'] != [HCV] or sp['n'] != len(with_gap) or sp['permutations'] != len(orders) or \
            max(abs(sp['rho'] - rho), abs(sp['p_one_sided'] - float(np.mean(rhos >= rho - 1e-12))),
                abs(sp['p_two_sided'] - float(np.mean(np.abs(rhos) >= abs(rho) - 1e-12)))) > 1e-12 or sp['gaps'] != gaps:
        raise AssertionError('newdata_analysis.json: the Spearman correlation differs from a recomputation over every order')
    counted = {d: [v for s, v in pin[d]['external_gap']['sources'].items() if informative[d][s]] or
               list(pin[d]['external_gap']['sources'].values()) for d in with_gap}
    rank_stable = list(stats.rankdata([np.mean(counted[d]) for d in with_gap])) == list(x)

    # ladder, comparators, balanced accuracy, duplicates and selected depths
    E, BA = {}, {}
    for d in NEWDATA:
        S = runs_of[d].summary['summaries'][d]
        for m in NEW_MODELS:
            E[(d, m)], BA[(d, m)] = lookup(S, m, 'error'), lookup(S, m, 'balanced_accuracy')
    lad = pd.read_csv(adir / 'newdata_ladder.csv')
    if list(zip(lad['dataset'], lad['rung'], lad['model_id'])) != [(d, rung, m) for d in NEWDATA for rung, m in NEW_RUNGS]:
        raise AssertionError('newdata_ladder.csv: rows differ from the ladder')
    for _, r in lad.iterrows():
        s = E[(r['dataset'], r['model_id'])]
        within = r['mean_within_fold_seed_sd']
        if abs(r['mean_error'] - s['mean']) > 1e-12 or abs(r['outer_fold_sd'] - s['outer_fold_sd']) > 1e-12 or \
                (pd.isna(within) != (s['mean_within_fold_seed_sd'] is None)) or \
                (not pd.isna(within) and abs(within - s['mean_within_fold_seed_sd']) > 1e-12):
            raise AssertionError(f'newdata_ladder.csv/{r["dataset"]}/{r["model_id"]}: differs from the batch summary')
    comp = pd.read_csv(adir / 'newdata_comparators.csv')
    if list(zip(comp['dataset'], comp['model_b'])) != [(d, m) for d in NEWDATA for m in NEW_COMPARATORS] or set(comp['model_a']) != {KNN}:
        raise AssertionError('newdata_comparators.csv: rows differ from the comparator intervals')
    CMP = {}
    for _, r in comp.iterrows():
        d, m = r['dataset'], r['model_b']
        if abs(r['mean_error_a'] - E[(d, KNN)]['mean']) > 1e-12 or abs(r['mean_error_b'] - E[(d, m)]['mean']) > 1e-12 or \
                bool(r['best_comparator']) != (m == min(NEW_COMPARATORS, key=lambda c: E[(d, c)]['mean'])):
            raise AssertionError(f'newdata_comparators.csv/{d}/{m}: errors or best flag differ from the batch summary')
        check_interval(r, *fold_contrast(runs_of[d], KNN, runs_of[d], m, d), f'newdata_comparators.csv/{d}/{m}', p_key='p_unadjusted')
        CMP[(d, m)] = r
    best = {d: min(COMPARATORS, key=lambda m: E[(d, m)]['mean']) for d in NEWDATA}
    lowest = {d: min(NEW_MODELS, key=lambda m: E[(d, m)]['mean']) for d in NEWDATA}
    dup = pd.read_csv(adir / 'newdata_duplicates.csv').set_index('dataset')
    for d in NEWDATA:
        r, audit = dup.loc[d], aj['duplicate_audit'][d]
        if any(int(r[x]) != audit['counts'][x] for x in audit['counts']) or audit['matches_pinned_counts'] is not True or \
                audit['pinned_counts'] != pin[d]['duplicates'] or int(r['label_conflicting_groups']) != 0 or \
                abs(r['share_with_training_duplicate'] - r['test_rows_with_training_duplicate'] / r['test_rows']) > 1e-12:
            raise AssertionError(f'newdata_duplicates.csv/{d}: counts differ from the audit or the pins')
    grid_widths = [list(w) for w in knn.protocol['full_method']['candidate_grid']['widths']]
    two = {}
    for d in NEWDATA:
        widths = [list(rec['config']['widths']) for rec in selection_records(runs_of[d], KNN, [d]).values()]
        if len(widths) != n_outer or any(w not in grid_widths for w in widths):
            raise AssertionError(f'{d}: the selected widths are not outer-fold choices from the grid')
        two[d] = sum(len(w) == 2 for w in widths)

    # tables
    gap_cell = lambda d: f'{gaps[d]:.2f}' if d in gaps else '--'
    higher = {family: [d for d in NEWDATA if F[(family, d)]['mean_difference'] > 0] for family, _, _ in NEW_FAMILIES}
    significant = {family: [d for d in NEWDATA if F[(family, d)]['holm_p_approximate'] < 0.05] for family, _, _ in NEW_FAMILIES}
    if len(significant['primary']) != len(significant['secondary']):
        raise AssertionError('the caption gives one count of significant gains for both families')
    source_name = lambda s: f'OpenML task {s.rsplit("_", 1)[1]}' if s.startswith('openml_task_') else \
        ('2014 table' if s == 'fernandez_delgado_2014' else None)
    if any(source_name(s) is None for d in with_gap for s in pin[d]['external_gap']['sources']):
        raise AssertionError('batch protocols: an evidence source other than OpenML tasks and the 2014 table')

    def sources_cell(d):
        items = sorted(pin[d]['external_gap']['sources'].items(), key=lambda kv: (not kv[0].startswith('openml'), kv[0]))
        return '; '.join(f'{source_name(s)}: {v:g}' + ('' if informative[d][s] else '$^\\ddagger$') for s, v in items) or 'none'
    erows = [[NEW_LABEL[d], f"{pin[d]['data_id']} (v{pin[d]['version']})", f"{pin[d]['shape'][0]:,}", str(pin[d]['shape'][1]),
              str(len(pin[d]['class_counts'])), str(min(pin[d]['class_counts'])), NEW_STRATUM[stratum[d]], gap_cell(d), sources_cell(d),
              f"\\texttt{{{pin[d]['dataset_hash'][:HASH_DIGITS]}}}"] for d in NEWDATA]
    caption = (f'\\textbf{{Pins and published evidence of the further datasets.}} For every dataset: its OpenML data identifier '
               f'and version, the rows, features and classes and the size of the smallest class as prepared, its stratum and the first '
               f'{words(HASH_DIGITS)} hexadecimal digits of the content hash recorded at preparation. The gap is the best tuned random '
               f'forest, support vector machine or gradient boosting minus the best tuned kNN, in percentage points. Its sources are '
               f'the OpenML evaluations of the dataset\'s task \\citep{{vanschoren2014openml}} and the accuracy table of '
               f'\\citet{{fernandezdelgado2014hundreds}} (2014 table); the Gap column averages them. $^\\ddagger$: fewer than '
               f'{words(rules["runs"])} kNN runs, a source the stratum rule did not count. HCV: hepatitis C virus data of Egyptian '
               f'patients, with no OpenML evaluations or benchmark-table entry. The protocols pin every value together with the checksums of the OpenML files.')
    write_table('tab_s_newdata_evidence', caption, 'tab:s-newdata-evidence',
                ['Dataset', 'OpenML ID', 'Rows', 'Features', 'Classes', two_lines('Smallest', 'class'), 'Stratum', 'Gap (pp)',
                 'Sources (pp)', 'Hash'], erows, 'llrrrrlrll', size='\\scriptsize', wide=True)

    es = mt['effect_sizes']
    stratum_row = lambda label: (f"{signed(es[label]['mean'], 2)} (SD {pct(es[label]['sd'], 2)}; median {signed(es[label]['median'], 2)}; "
                                 f"{signed(es[label]['min'], 2)} to {signed(es[label]['max'], 2)})")
    trows = [[f'Mean training effect, {words(len(marked))} marked datasets (pp)', stratum_row('H')],
             [f'Mean training effect, other {words(len(others))} datasets (pp)', stratum_row('C')],
             ['Difference of the two means (pp)', signed(mt['statistic'], 2)],
             [f'Splits of the {words(k)} datasets into {words(len(marked))} and {words(len(others))}', str(mt['splits'])],
             ['Splits whose difference is at least the observed one', str(mt['at_least_observed'])],
             ['Exact one-sided $p$ (smallest attainable)', f"{pval(mt['p_one_sided'])} ({decl['smallest_attainable_p']:.3f})"],
             ['Spearman correlation of external gap and training effect', f"{sp['rho']:.2f} ({words(sp['n'])} datasets, HCV excluded)"],
             ['Exact $p$ of the correlation, one-sided and two-sided', f"{pval(sp['p_one_sided'])} and {pval(sp['p_two_sided'])}"],
             ['Orders of the effects enumerated', f"{sp['permutations']:,}"]]
    caption = (f'\\textbf{{Prespecified stratum test and descriptive correlation on the further datasets.}} The training effect of a '
               f'dataset is its mean paired accuracy difference, ArrowFlow minus the untrained ArrowFlow '
               f'(Table~\\ref{{tab:s-training}}). The exact one-sided permutation test compares the mean effect of the '
               f'{words(len(marked))} marked datasets with that of the other {words(len(others))} over every split of the '
               f'{words(k)} datasets into groups of those sizes, the effects held fixed. Its $p$ value is the share of splits whose '
               f'difference is at least the observed one, so $p\\le{decl["alpha"]:g}$ requires the observed difference to be among the '
               f'{int(decl["alpha"] * mt["splits"])} largest. The Spearman correlation, the Pearson correlation of the ranks of the '
               f'external gaps and the effects, is descriptive; its exact $p$ values enumerate every order of the effects against the '
               f'gaps. HCV has no gap.')
    write_table('tab_s_newdata_tests', caption, 'tab:s-newdata-tests', ['Quantity', 'Value'], trows, 'll', size='\\footnotesize')

    return dict(F=F, E=E, BA=BA, CMP=CMP, best=best, lowest=lowest, two=two, gaps=gaps, marked=marked, others=others,
                stratum=stratum, mt=mt, sp=sp, rules=rules, informative=informative, pin=pin, B=B, log=log, dup=dup,
                higher=higher, significant=significant, effect=effect, decl=decl, rank_stable=rank_stable, notes=aj['notes'],
                loo=loo, protocols=(p1, p2))


# ----------------------------------------------------------------------------- the seventeen datasets together (Sections 7.2-7.3, S3)
HOLISTIC = '2026-09-14-holistic'
COMBINED = DATASETS + NEWDATA
NAME = {**LABEL, **NEW_LABEL}
MAIN7 = [KNN] + COMPARATORS + ['dummy']
GROUP_BLOCK = {'benchmark': 'Benchmark datasets', 'further': 'Further datasets'}
FLAG_MARK = {'ceiling': 'c', 'near_majority': 'm', 'no_learning': 'n'}
COMBINED_RULES = {'within_points': 3.0, 'ceiling_points': 1.0, 'near_majority_points': 0.5, 'near_majority_advantage_points': 2.0}
MODEL_HEAD = {**{m: MODEL[m] for m in MAIN7}, 'gradient_boosting': two_lines('Gradient', 'boosting'),
              'dummy': two_lines('Majority', 'class')}
H34_HEAD = two_lines('Holm $p$', 'over 34')
ENCODER_KEYS, SHARED_KEYS = ('embed_scale', 'degree_offset'), ('aggregation', 'n_views', 'strategy')


def scientific(x):
    """A small p value as TeX scientific notation with two significant digits: 7.7\\times10^{-5}."""
    mantissa, exponent = f'{float(x):.1e}'.split('e')
    return f'{mantissa}\\times10^{{{int(exponent)}}}'


def blank(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def agrees(a, b, where, tol=1e-12):
    if blank(a) != blank(b) or (not blank(a) and abs(float(a) - float(b)) > tol):
        raise AssertionError(f'{where}: the combined analysis gives {a}, the verified source gives {b}')


def holistic_outputs(hdir):
    """The sealed outputs of the combined analysis: READY lists exactly the files on disk with their sha256, and both records
    were written from committed analysis sources at the commit that READY names."""
    ready = read_json(hdir / 'READY')
    listed = {e['path']: e['sha256'] for e in ready['outputs']}
    for sub in ('benchmark', 'training'):
        if sorted(f'{sub}/{p.name}' for p in (hdir / sub).iterdir()) != sorted(x for x in listed if x.startswith(f'{sub}/')):
            raise AssertionError(f'{hdir / sub}: the files on disk differ from the READY listing')
    for path, digest in listed.items():
        if sha256_file(hdir / path) != digest:
            raise AssertionError(f'{hdir / path}: differs from its READY hash')
    records = {x: read_json(hdir / x) for x in listed if x.endswith('.json')}
    for name in ('benchmark/benchmark.json', 'training/training.json'):
        provenance = records[name]['provenance']
        if provenance.get('sources_committed_at_revision') is not True or provenance.get('code_revision') != ready['commit']:
            raise AssertionError(f'{name}: not written from the committed analysis sources that READY names')
    return ready, {x: pd.read_csv(hdir / x) for x in listed if x.endswith('.csv')}, records


def render_combined(hdir, runs, bridge, knn, main_table, training, projected, train, sorting, newdata, referee):
    """The seventeen datasets together (Tables 2 and 3, S3). Every value is read from the sealed outputs of holistic benchmark
    and holistic training and must equal the verified source that the earlier renders read: the benchmark, training, sorting
    and batch runs and their analyses. The rank statistics, the competitiveness flags, the Holm adjustment over all training
    contrasts and the depth split of the further datasets are recomputed."""
    from scipy import stats
    ready, csvs, records = holistic_outputs(hdir)
    bj, tj = records['benchmark/benchmark.json'], records['training/training.json']
    if len({Path(r['run']).parent for record in (bj, tj) for r in record['provenance']['runs'].values()}) != 1:
        raise AssertionError('holistic benchmark and training read their source runs from more than one directory, but Section S5 passes '
                             'one runs directory')
    group = {**dict.fromkeys(DATASETS, 'benchmark'), **dict.fromkeys(NEWDATA, 'further')}
    runs_of = {d: fam for fam in newdata['B'].values() for d in fam.protocol['datasets']}
    conf, n_outer, k17 = FACTS['confidence'], FACTS['n_outer'], len(COMBINED)
    iv_head = f'{conf}\\% interval'
    blocks17 = {0: GROUP_BLOCK['benchmark'], len(DATASETS): GROUP_BLOCK['further']}

    def summary(d, m, metric='error'):
        fam = runs_of[d] if group[d] == 'further' else training if m in CONTROLS else projected if m == PROJECTED else knn
        return lookup(fam.summary['summaries'][d], m, metric)

    def indexed(name, keys, per_dataset):
        frame = csvs[name]
        if list(dict.fromkeys(frame['dataset'])) != COMBINED or len(frame) != per_dataset * k17 or frame.duplicated(keys).any() \
                or any((frame.loc[frame['dataset'] == d, 'group'] != group[d]).any() for d in COMBINED):
            raise AssertionError(f'{name}: rows or groups differ from the seventeen datasets in panel order')
        return frame.set_index(keys)

    # main table and complete metrics
    MT = indexed('benchmark/main_table.csv', ['dataset', 'model_id'], len(MAIN7))
    T = main_table_rows(main_table, bridge, knn)
    for d in COMBINED:
        for m in MAIN7:
            s = summary(d, m)
            for column, key in (('mean_error', 'mean'), ('outer_fold_sd', 'outer_fold_sd'), ('mean_within_fold_seed_sd', 'mean_within_fold_seed_sd')):
                agrees(MT.loc[(d, m), column], s.get(key), f'main_table.csv/{d}/{m}/{column}')
            if group[d] == 'benchmark':
                agrees(MT.loc[(d, m), 'mean_error'], T[d][m]['mean_error'], f'main_table.csv/{d}/{m} against main_table.json')
    err = lambda d, m: float(MT.loc[(d, m), 'mean_error'])
    CM = indexed('benchmark/complete_metrics.csv', ['dataset', 'model_id'], len(NEW_MODELS))
    for d in COMBINED:
        for m in NEW_MODELS:
            for metric in ('error', 'balanced_accuracy', 'macro_f1'):
                s = summary(d, m, metric)
                for suffix, key in (('', 'mean'), ('_outer_fold_sd', 'outer_fold_sd'), ('_within_fold_seed_sd', 'mean_within_fold_seed_sd')):
                    agrees(CM.loc[(d, m), metric + suffix], s.get(key), f'complete_metrics.csv/{d}/{m}/{metric}{suffix}')

    # comparator intervals, the best tuned model, the competitiveness flags
    CI = indexed('benchmark/comparator_intervals.csv', ['dataset', 'model_b'], len(COMPARATORS))
    published = pd.read_csv(runs / '2026-09-13-referee-analyses' / 'comparators' / 'comparator_contrasts.csv').set_index(['dataset', 'model_b'])
    best = {d: min(COMPARATORS, key=lambda m: err(d, m)) for d in COMBINED}
    for d in COMBINED:
        for m in COMPARATORS:
            r, src = CI.loc[(d, m)], (published.loc[(d, m)] if group[d] == 'benchmark' else newdata['CMP'][(d, m)])
            for column in ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_unadjusted'):
                agrees(r[column], src[column], f'comparator_intervals.csv/{d}/{m}/{column}')
            if r['model_a'] != KNN or bool(r['best_classical']) != (m == best[d]) or (int(r['n_folds']), int(r['df'])) != (n_outer, FACTS['df']):
                raise AssertionError(f'comparator_intervals.csv/{d}/{m}: the model, the best flag or the folds differ')
    CP = indexed('benchmark/competitiveness.csv', ['dataset'], 1)
    if bj['rules']['thresholds'] != COMBINED_RULES:
        raise AssertionError('benchmark.json: the fixed competitiveness rules differ from those the captions state')
    gap, flags = {}, {}
    for d in COMBINED:
        e = {m: round(100 * err(d, m), 9) for m in MAIN7}
        lowest = min(e[m] for m in COMPARATORS)
        gap[d] = round(e[KNN] - lowest, 9)
        mine = dict(within_three_points=gap[d] <= COMBINED_RULES['within_points'], best_on=e[KNN] < lowest,
                    ceiling=lowest < COMBINED_RULES['ceiling_points'],
                    near_majority=round(abs(e[KNN] - e['dummy']), 9) <= COMBINED_RULES['near_majority_points']
                    and gap[d] >= COMBINED_RULES['near_majority_advantage_points'],
                    no_learning=e['dummy'] <= min(e[m] for m in [KNN] + COMPARATORS))
        mine['within_three_points_as_printed'] = mine['within_three_points']
        if any(bool(CP.loc[d, column]) != value for column, value in mine.items()) or CP.loc[d, 'best_classical'] != best[d]:
            raise AssertionError(f'competitiveness.csv/{d}: a flag or the best tuned model differs from the fixed rules')
        agrees(CP.loc[d, 'gap_points'], gap[d], f'competitiveness.csv/{d}/gap_points', tol=1e-6)
        flags[d] = [flag for flag in FLAG_MARK if mine[flag]]
    if any(gap[d] <= 0 for d in COMBINED):
        raise AssertionError('the text says ArrowFlow trails the best tuned classical model on every dataset')
    clear = [d for d in COMBINED if CI.loc[(d, best[d]), 'ci_high'] < 0]
    if any(CI.loc[(d, best[d]), 'ci_low'] > 0 for d in COMBINED):
        raise AssertionError('an interval against the best tuned model favors ArrowFlow')

    # ranks over the seventeen datasets
    tuned = [KNN] + COMPARATORS
    matrix = np.array([[err(d, m) for m in tuned] for d in COMBINED])
    ranks = np.vstack([stats.rankdata(row) for row in matrix])
    kk = len(tuned)
    chi2, p = stats.friedmanchisquare(*matrix.T)
    F = (k17 - 1) * chi2 / (k17 * (kk - 1) - chi2)
    pF = stats.f.sf(F, kk - 1, (kk - 1) * (k17 - 1))
    cd = stats.studentized_range.ppf(0.95, kk, np.inf) / math.sqrt(2) * math.sqrt(kk * (kk + 1) / (6 * k17))
    fr, nm = bj['ranks']['friedman'], bj['ranks']['nemenyi']
    if max(abs(fr['chi2'] - chi2), abs(fr['p'] - p), abs(fr['iman_davenport_f'] - F), abs(fr['iman_davenport_p'] - pF),
           abs(nm['critical_difference'] - cd)) > 1e-9 or fr['df'] != kk - 1 or nm['alpha'] != 0.05:
        raise AssertionError('benchmark.json: the Friedman and Nemenyi statistics differ from a recomputation')
    if any(len(set(row)) < kk for row in matrix.tolist()) or any(len({pct(x, 2) for x in row}) < kk for row in matrix):
        raise AssertionError('two errors tie at full or two-decimal precision, but the rank caption says none do')
    mean_rank = dict(zip(tuned, ranks.mean(axis=0)))
    MR, RK = csvs['benchmark/mean_ranks.csv'].set_index('model_id'), csvs['benchmark/rank_matrix.csv'].set_index(['dataset', 'model_id'])
    for j, m in enumerate(tuned):
        agrees(MR.loc[m, 'mean_rank'], mean_rank[m], f'mean_ranks.csv/{m}')
        for i, d in enumerate(COMBINED):
            agrees(RK.loc[(d, m), 'rank'], ranks[i, j], f'rank_matrix.csv/{d}/{m}')
    order = sorted(tuned, key=mean_rank.get)
    separated = OrderedDict((a, [b for b in order if mean_rank[b] - mean_rank[a] > cd]) for a in order)
    separated = OrderedDict((a, bs) for a, bs in separated.items() if bs)
    if {frozenset((x['model_a'], x['model_b'])) for x in nm['pairs_exceeding_critical_difference']} != \
            {frozenset((a, b)) for a, bs in separated.items() for b in bs}:
        raise AssertionError('benchmark.json: the pairs beyond the Nemenyi critical difference differ from a recomputation')
    tied_1dp = [d for d, row in zip(COMBINED, matrix) if len({pct(x) for x in row}) < kk]
    if bj['ranks']['ties']['datasets_with_tied_errors_as_displayed'] != tied_1dp:
        raise AssertionError('benchmark.json: the datasets with tied errors at table precision differ')

    # duplicates
    DU = indexed('benchmark/duplicates.csv', ['dataset'], 1)
    audit = pd.read_csv(runs / '2026-09-13-referee-analyses' / 'duplicates' / 'duplicate_groups.csv').set_index('dataset')
    for d in COMBINED:
        src = audit.loc[d] if group[d] == 'benchmark' else newdata['dup'].loc[d]
        for column in ('n_rows', 'duplicate_rows', 'duplicate_groups', 'label_conflicting_groups', 'test_rows',
                       'test_rows_with_training_duplicate', 'share_with_training_duplicate', 'fold_share_min', 'fold_share_max'):
            agrees(DU.loc[d, column], src[column], f'duplicates.csv/{d}/{column}')

    # training contrasts in their registered families, and one Holm adjustment over all of them
    TC = indexed('training/training_controls.csv', ['dataset'], 1)
    prefix, reg = {UNTRAINED: 'untrained', INPUT_KNN: 'input'}, {}
    for d in COMBINED:
        for m in CONTROLS:
            if group[d] == 'benchmark':
                src, family, size = train['C'][(d, m)], 'knn_training_primary', train['size']
            else:
                label = 'primary' if m == UNTRAINED else 'secondary'
                src, family, size = newdata['F'][(label, d)], f'newdata_{label}', len(NEWDATA)
            r = TC.loc[d]
            for column, key in (('difference', 'mean_difference'), ('standard_error', 'standard_error'), ('ci_low', 'ci_low'),
                                ('ci_high', 'ci_high'), ('p_unadjusted', 'p_approximate'), ('registered_holm_p', 'holm_p_approximate')):
                agrees(r[f'{prefix[m]}_{column}'], src[key], f'training_controls.csv/{d}/{m}/{column}')
            if (r[f'{prefix[m]}_registered_family'], int(r[f'{prefix[m]}_registered_family_size'])) != (family, size):
                raise AssertionError(f'training_controls.csv/{d}/{m}: the registered family differs')
            reg[(d, m)] = dict(diff=float(src['mean_difference']), se=float(src['standard_error']), lo=float(src['ci_low']),
                               hi=float(src['ci_high']), p=float(src['p_approximate']), holm=float(src['holm_p_approximate']),
                               index=int(src['family_index']), family=family)
    keys = [(d, m) for d in COMBINED for m in CONTROLS]
    h34 = dict(zip(keys, holm([reg[x]['p'] for x in keys])))
    H = indexed('training/holm34_sensitivity.csv', ['dataset', 'control'], len(CONTROLS))
    for (d, m), value in h34.items():
        agrees(H.loc[(d, m), 'holm34_p'], value, f'holm34_sensitivity.csv/{d}/{m}')
        agrees(H.loc[(d, m), 'registered_holm_p'], reg[(d, m)]['holm'], f'holm34_sensitivity.csv/{d}/{m}/registered')
        if bool(H.loc[(d, m), 'holm34_below_alpha']) != (value < 0.05):
            raise AssertionError(f'holm34_sensitivity.csv/{d}/{m}: the flag differs from the adjusted p value')
    survivors = [x for x in keys if h34[x] < 0.05]
    significant = [x for x in keys if reg[x]['holm'] < 0.05]
    if {(x['dataset'], x['control']) for x in tj['holm34']['below_alpha']} != set(survivors) or tj['holm34']['members'] != len(keys):
        raise AssertionError('training.json: the Holm adjustment over all training contrasts differs from a recomputation')
    higher = [d for d in COMBINED if reg[(d, UNTRAINED)]['diff'] > 0]
    if any(signed(reg[(d, UNTRAINED)]['diff'], 2) in ('+0.00', '$-$0.00') for d in COMBINED):
        raise AssertionError('a difference from the untrained ArrowFlow prints as zero, but the text counts datasets by its sign')
    # the datasets on which the untrained ArrowFlow is within one percentage point of ArrowFlow (Ruling (final review), FINAL-I1), read
    # from training_controls.csv; the count must agree with the two-decimal cells of Table 3
    close = [d for d in COMBINED if abs(float(TC.loc[d, 'untrained_difference'])) <= 0.01]
    if any((abs(float(signed(reg[(d, UNTRAINED)]['diff'], 2).replace('$-$', '-'))) <= 1) != (d in close) for d in COMBINED):
        raise AssertionError('the datasets within one point of the untrained ArrowFlow differ at the two decimals of Table 3')

    # ladder
    LD, LW = indexed('training/ladder_detail.csv', ['dataset', 'rung'], len(NEW_RUNGS)), indexed('training/ladder.csv', ['dataset'], 1)
    for d in COMBINED:
        for rung, m in NEW_RUNGS:
            s = sorting['L'][(d, m)] if group[d] == 'benchmark' else newdata['E'][(d, m)]
            mean = s['mean_error'] if group[d] == 'benchmark' else s['mean']
            if LD.loc[(d, rung), 'model_id'] != m:
                raise AssertionError(f'ladder_detail.csv/{d}/{rung}: the rung holds another model')
            agrees(LD.loc[(d, rung), 'mean_error'], mean, f'ladder_detail.csv/{d}/{rung}')
            agrees(LD.loc[(d, rung), 'outer_fold_sd'], s['outer_fold_sd'], f'ladder_detail.csv/{d}/{rung}/outer_fold_sd')
            agrees(LW.loc[d, rung], mean, f'ladder.csv/{d}/{rung}')

    # depth split: the benchmark groups from the training analysis, the further groups recomputed from the batch runs
    depths = [tuple(w) for w in train['depths']]
    DS = csvs['training/depth_split.csv']
    if len(DS) != len(depths) * k17 or list(dict.fromkeys(DS['dataset'])) != COMBINED:
        raise AssertionError('depth_split.csv: rows differ from the seventeen datasets and the declared depths')
    DS = DS.set_index(['dataset', 'widths'])
    diffs, same_widths = {}, {}
    for d in COMBINED:
        if group[d] == 'further':
            fam = runs_of[d]
            recs, urecs = selection_records(fam, KNN, [d]), selection_records(fam, UNTRAINED, [d])
            a, b = fold_accuracy(fam, KNN, d), fold_accuracy(fam, UNTRAINED, d)
        for w in depths:
            if group[d] == 'benchmark':
                g = train['G'][(d, w)]
                values, same = [float(x) for x in g['fold_differences']], int(g['untrained_selected_the_same_widths'])
            else:
                folds = sorted(key[1:] for key, rec in recs.items() if tuple(rec['config']['widths']) == w)
                values = [float(a[f] - b[f]) for f in folds]
                same = sum(tuple(urecs[(d, *f)]['config']['widths']) == w for f in folds)
            diffs[(d, w)], same_widths[(d, w)] = values, same
            r = DS.loc[(d, json.dumps(list(w)))]
            if int(r['n_folds']) != len(values) or int(r['untrained_selected_the_same_widths']) != same:
                raise AssertionError(f'depth_split.csv/{d}/{w}: folds or same-width count differ from the selection records')
            if values:
                agrees(r['mean_difference'], statistics.mean(values), f'depth_split.csv/{d}/{w}/mean')
                agrees(r['min'], min(values), f'depth_split.csv/{d}/{w}/min')
                agrees(r['max'], max(values), f'depth_split.csv/{d}/{w}/max')
            if len(values) > 1:
                _, _, lo, hi, _ = corrected_t(values, conf)
                agrees(r['sd'], statistics.stdev(values), f'depth_split.csv/{d}/{w}/sd')
                agrees(r['ci_low'], lo, f'depth_split.csv/{d}/{w}/ci_low', tol=1e-9)
                agrees(r['ci_high'], hi, f'depth_split.csv/{d}/{w}/ci_high', tol=1e-9)
            elif not (blank(r['sd']) and blank(r['ci_low'])):
                raise AssertionError(f'depth_split.csv/{d}/{w}: an SD or interval over fewer than two folds')
    DP, pooled = csvs['training/depth_pooled.csv'], {}
    for scope, names in (('all', COMBINED), ('benchmark', DATASETS), ('further', NEWDATA)):
        for w in depths:
            values = [x for d in names for x in diffs[(d, w)]]
            row = DP[(DP['scope'] == scope) & (DP['widths'] == json.dumps(list(w)))]
            if len(row) != 1 or int(row.iloc[0]['n_dataset_folds']) != len(values):
                raise AssertionError(f'depth_pooled.csv/{scope}/{w}: the pooled fold count differs')
            agrees(row.iloc[0]['mean_difference'], statistics.mean(values), f'depth_pooled.csv/{scope}/{w}/mean')
            agrees(row.iloc[0]['sd'], statistics.stdev(values), f'depth_pooled.csv/{scope}/{w}/sd')
            pooled[(scope, w)] = dict(n=len(values), mean=statistics.mean(values), sd=statistics.stdev(values),
                                      same=sum(same_widths[(d, w)] for d in names), datasets=[d for d in names if diffs[(d, w)]])
    SW = csvs['benchmark/selected_widths.csv']
    if len(SW) != k17 * n_outer or list(dict.fromkeys(SW['dataset'])) != COMBINED:
        raise AssertionError('selected_widths.csv: rows differ from the outer folds of the seventeen datasets')
    two = {d: int((SW.loc[SW['dataset'] == d, 'hidden_layers'] == 2).sum()) for d in COMBINED}
    if any(two[d] != len(diffs[(d, depths[1])]) for d in COMBINED) or \
            bj['selected_widths']['totals']['all'][json.dumps(list(depths[1]))] != sum(two.values()):
        raise AssertionError('selected_widths.csv: the two-layer folds differ from the depth split')

    # the stratum test and the correlation, passed through
    st, mt, sp = tj['stratum_test'], newdata['mt'], newdata['sp']
    for key in ('statistic', 'mean_h', 'mean_c', 'p_one_sided'):
        agrees(st['moderator_test'][key], mt[key], f'training.json/moderator_test/{key}')
    for key in ('rho', 'p_one_sided', 'p_two_sided'):
        agrees(st['spearman'][key], sp[key], f'training.json/spearman/{key}')
    if (st['moderator_test']['splits'], st['moderator_test']['at_least_observed'], st['spearman']['n']) != \
            (mt['splits'], mt['at_least_observed'], sp['n']):
        raise AssertionError('training.json: the stratum test or the correlation differs from the frozen analysis')

    # layers whose responses tie at every input (at most V^2/4 + 1 distinct footrule values), and layers at which
    # Proposition 4(ii)'s full-order certificate is unattainable: a changed input lies at footrule distance at least 2,
    # so the certificate needs g_min >= 6, and packing N distinct responses into [0, floor(V^2/2)] with consecutive
    # gaps of at least 6 needs 6(N-1) <= floor(V^2/2)
    ties, tie_layers = {}, set()
    vacuous, vacuous_layers, vacuous_first = {}, set(), Counter()
    for d in COMBINED:
        count, vac = 0, 0
        for rec in selection_records(knn if group[d] == 'benchmark' else runs_of[d], KNN, [d]).values():
            e = {o['settings'].get('embed_dim') for o in rec['outer']}
            if len(e) != 1:
                raise AssertionError(f'{d}: the outer fits of one fold record different vocabulary sizes')
            layers = [int(e.pop())] + [int(x) for x in rec['config']['widths']]
            forced = [(v, nf) for v, nf in zip(layers, layers[1:]) if nf > v * v // 4 + 1]
            tie_layers.update(forced)
            count += bool(forced)
            unpacked = [(v, nf) for v, nf in zip(layers, layers[1:]) if 6 * (nf - 1) > v * v // 2]
            vacuous_layers.update(unpacked)
            vac += bool(unpacked)
            if unpacked:
                if unpacked != [(layers[0], layers[1])]:
                    raise AssertionError(f'{d}: the text says the unattainable certificate lies in the first hidden layer only')
                vacuous_first[unpacked[0]] += 1
        ties[d] = count
        vacuous[d] = vac
    # the same arithmetic over the declared layer sizes, read from the protocol rather than from the selections
    method = knn.protocol['full_method']
    declared_v = sorted({int(min(128, max(8, round(method['adaptive_encoding_defaults'][b]['embed_dim'] * s))))
                         for b in method['adaptive_encoding_defaults'] for s in method['candidate_grid']['embed_scale']})
    declared_n = sorted({int(w) for ws in method['candidate_grid']['widths'] for w in ws})
    declared_vacuous = [(v, nf) for v in declared_v for nf in declared_n if 6 * (nf - 1) > v * v // 2]
    declared_tied = [(v, nf) for v in declared_v for nf in declared_n if nf > v * v // 4 + 1]
    if set(declared_vacuous) != vacuous_layers or set(declared_tied) != tie_layers:
        raise AssertionError('the layer sizes of the grid and the layer sizes actually selected disagree on the certificate')

    # Table 2
    headline = [KNN] + COMPARATORS
    rows = []
    for d in COMBINED:
        low = min(err(d, m) for m in headline)
        marks = ''.join(FLAG_MARK[flag] for flag in flags[d])
        cells = [NAME[d] + (f'$^{{\\mathrm{{{marks}}}}}$' if marks else '')]
        for m in MAIN7:
            cell = pm(err(d, m), MT.loc[(d, m), 'outer_fold_sd'])
            cells.append(f'\\textbf{{{cell}}}' if m in headline and err(d, m) == low else cell)
        rows.append(cells + [signed(err(d, KNN) - err(d, best[d]), 2)])
    th = COMBINED_RULES
    if not all(err(d, KNN) > err(d, best[d]) for d in COMBINED):
        raise AssertionError('the caption of Table 2 says ArrowFlow trails the best tuned classical model on every dataset')
    # the simplified caption says what a reader needs to read the table; the protocol details it once carried are stated in
    # Section 7.1 and in S2.5 (fitting seeds averaged within folds, the candidate budget, the readout chosen inside every fit)
    caption = (f'\\textbf{{ArrowFlow against five tuned classical models and the majority class on the {words(k17)} datasets.}} Mean '
               f'test error in percent over the {n_outer} outer folds ({FACTS["outer_folds"]} folds $\\times$ {FACTS["outer_repeats"]} '
               f'repeats), with the standard deviation (SD) across folds in parentheses; the SD describes spread, not a confidence '
               f'interval. Every model chose its configuration on inner folds only (Section~\\ref{{sec:protocol}}). Bold marks the lowest '
               f'unrounded error among ArrowFlow and the five classical models. The last column gives ArrowFlow\'s error minus that of the best '
               f'tuned classical model, the majority class excluded, in percentage points, positive when ArrowFlow trails; it is '
               f'positive on all {words(k17)} datasets. It is computed from unrounded errors, so it can differ in the last digit from the '
               f'printed errors; Section~S3.2 lists the errors of the tuned models to two decimals. The marks are post hoc flags. c: the best tuned '
               f'classical model has less than '
               f'{th["ceiling_points"]:g}\\% error. m: ArrowFlow is within {th["near_majority_points"]:g} points of the majority class, '
               f'and the best tuned model is at least {th["near_majority_advantage_points"]:g} points better. n: no model has lower '
               f'error than the majority class. SVC: support vector classifier with a radial basis function '
               f'(RBF) kernel; MLP: multilayer perceptron; kNN: nearest-neighbor classifier; HCV: hepatitis C virus.')
    write_table('tab_main_benchmark', caption, 'tab:main', ['Dataset'] + [MODEL_HEAD[m] for m in MAIN7] + [two_lines('Gap to best', 'tuned (pp)')],
                rows, 'l' + 'r' * (len(MAIN7) + 1), size='\\scriptsize', subheads=blocks17, wide=True)

    # Table 3
    change = lambda x: f"{signed(reg[x]['diff'], 2)} {interval(reg[x]['lo'], reg[x]['hi'], 2)}"
    holm_cell = lambda x: pval(reg[x]['holm']) + ('$^\\ddagger$' if h34[x] < 0.05 else '')
    rows = [[NAME[d]] + [pct(LW.loc[d, rung]) for rung, _ in NEW_RUNGS] + [c for m in CONTROLS for c in (change((d, m)), holm_cell((d, m)))]
            for d in COMBINED]
    # the encoder settings (vocabulary scale and degree offset) that tuned input footrule kNN chose on the outer folds where ArrowFlow
    # chose the same ones (ruling P1 final 2, M4)
    same_encoder = {}
    for d in COMBINED:
        mine, theirs = (selection_records(knn, KNN, [d]), selection_records(training, INPUT_KNN, [d])) if group[d] == 'benchmark' \
            else (selection_records(runs_of[d], KNN, [d]), selection_records(runs_of[d], INPUT_KNN, [d]))
        if sorted(mine) != sorted(theirs) or any(set(theirs[x]['config']) != set(ENCODER_KEYS + SHARED_KEYS) or
                                                 any(mine[x]['config'][key] != theirs[x]['config'][key] for key in SHARED_KEYS)
                                                 for x in theirs):
            raise AssertionError(f'{d}: tuned input footrule kNN does not choose among the encoder settings of ArrowFlow on its folds')
        same_encoder[d] = sum(all(mine[x]['config'][key] == theirs[x]['config'][key] for key in ENCODER_KEYS) for x in mine)
    # the block headers name the Holm scope in plain words; each family was fixed in its run protocol before its controls were
    # scored, and the benchmark family after ArrowFlow's own benchmark run existed: the training protocol records the pairing check
    # against that run before its freeze (referee panel of 2026-09-23, item J1)
    if not re.search(r'training-pairing against arrowflow-v3-bridge-knn-1 passed', training.protocol['resource_decision']):
        raise AssertionError('the caption says the benchmark training family was fixed after ArrowFlow\'s benchmark results existed')
    subheads = {0: f'Benchmark datasets: Holm correction over {words(train["size"])} comparisons',
                len(DATASETS): f'Further datasets: Holm correction over {words(len(NEWDATA))} comparisons per control'}
    caption = (f'\\textbf{{What training adds, on the {words(k17)} datasets.}} Upper part: mean test error in percent, lower is '
               f'better, of five classifiers that all predict from nearest neighbors. From left to right they are numeric kNN on the standardized raw features, '
               f'numeric kNN on the projected scores before sorting, tuned input footrule kNN on the encoded input rankings, the untrained '
               f'ArrowFlow and ArrowFlow. Neighboring columns differ in more than one component, so this ladder describes and does not '
               f'decompose. Lower part: '
               f'ArrowFlow\'s accuracy minus that of each control in percentage points, positive when ArrowFlow is more accurate, with its '
               f'{conf}\\% interval, which is not corrected for multiple comparisons. Holm $p$ is corrected over the comparisons named in '
               f'the block header, which the run protocols fixed before the controls were scored; the benchmark family was fixed after '
               f'ArrowFlow\'s own benchmark results were known. $^\\ddagger$: still below 0.05 after one post '
               f'hoc correction over all {len(keys)} comparisons. Both controls share ArrowFlow\'s encoder family but choose their own '
               f'settings on inner folds. kNN: nearest-neighbor classifier; HCV: hepatitis C virus.')
    write_table('tab_training', caption, 'tab:training',
                ['Dataset'] + LADDER_HEAD + [two_lines('ArrowFlow $-$', 'untrained (pp)'), two_lines('Holm', '$p$'),
                                             two_lines('ArrowFlow $-$ tuned', 'input footrule kNN (pp)'), two_lines('Holm', '$p$')],
                rows, 'l' + 'r' * 9, size='\\scriptsize', subheads=[blocks17, subheads], stack=5)

    # S3: complete metrics
    heads = ['Dataset'] + [MODEL_HEAD[m] for m in MAIN7]

    def metric_rows(metric, with_seed):
        out = []
        for d in COMBINED:
            cells = [NAME[d]]
            for m in MAIN7:
                cell = pm(CM.loc[(d, m), metric], CM.loc[(d, m), f'{metric}_outer_fold_sd'])
                seed = CM.loc[(d, m), f'{metric}_within_fold_seed_sd']
                cells.append(cell + (f' [{pct(seed)}]' if with_seed and not blank(seed) else ''))
            out.append(cells)
        return out
    caption = (f'\\textbf{{Complete error of the models of Table~\\ref{{tab:main}}.}} Mean outer-fold error in percent (outer-fold SD) '
               f'[mean within-fold SD across the {words(FACTS["n_seeds"])} fitting seeds] on the {words(k17)} datasets. Deterministic '
               f'models have one fitting seed per fold and no seed SD. SVC: support vector classifier; MLP: multilayer perceptron; kNN: '
               f'nearest-neighbor classifier; HCV: hepatitis C virus.')
    write_table('tab_s_errors', caption, 'tab:s-errors', heads, metric_rows('error', True), 'l' + 'r' * len(MAIN7),
                size='\\scriptsize', subheads=blocks17, stack=4)
    caption = (f'\\textbf{{Balanced accuracy of the models of Table~\\ref{{tab:main}}.}} Mean outer-fold balanced accuracy in percent, '
               f'the unweighted mean of the per-class recalls, with the outer-fold SD in parentheses. Accuracy is the metric of every '
               f'registered analysis, so this table is descriptive.')
    write_table('tab_s_balanced', caption, 'tab:s-balanced', heads, metric_rows('balanced_accuracy', False), 'l' + 'r' * len(MAIN7),
                size='\\scriptsize', subheads=blocks17, stack=4)
    caption = (f'\\textbf{{Macro-F1 of the models of Table~\\ref{{tab:main}}.}} Mean outer-fold macro-F1 in percent, the unweighted mean '
               f'of the per-class F1 scores, where a class that is never predicted scores 0, with the outer-fold SD in parentheses. '
               f'Descriptive.')
    write_table('tab_s_macro_f1', caption, 'tab:s-macro-f1', heads, metric_rows('macro_f1', False), 'l' + 'r' * len(MAIN7),
                size='\\scriptsize', subheads=blocks17, stack=4)

    # S3: comparator intervals
    for g, name, label in (('benchmark', 'tab_s_comparators', 'tab:s-comparators'),
                           ('further', 'tab_s_comparators_further', 'tab:s-comparators-further')):
        crows, cblocks = [], []
        for d in [x for x in COMBINED if group[x] == g]:
            cblocks.append(len(crows))
            for i, m in enumerate(COMPARATORS):
                r = CI.loc[(d, m)]
                crows.append([NAME[d] if i == 0 else '', MODEL[m] + ('$^\\ast$' if m == best[d] else ''), signed(r['mean_difference'], 2),
                              interval(r['ci_low'], r['ci_high'], 2), pval(r['p_unadjusted'])])
        if g == 'benchmark':
            caption = (f'\\textbf{{ArrowFlow against every tuned classical model, benchmark datasets.}} Paired accuracy differences in '
                       f'percentage points, ArrowFlow minus the comparator (positive favors ArrowFlow), with fitting seeds averaged within '
                       f'each outer fold. Each difference carries the {interval_note()} and its unadjusted $p$ value. The {len(CI)} '
                       f'differences of this table and Table~\\ref{{tab:s-comparators-further}} are descriptive: they form no registered '
                       f'family, so some unadjusted $p$ values below 0.05 are expected by chance. $^\\ast$: the tuned classical model with the '
                       f'lowest mean error, picked after seeing the errors, which biases its difference against ArrowFlow. SVC: support '
                       f'vector classifier with a radial basis function (RBF) kernel; MLP: multilayer perceptron; kNN: nearest-neighbor '
                       f'classifier.')
        else:
            caption = (f'\\textbf{{ArrowFlow against every tuned classical model, further datasets.}} Paired accuracy differences in percentage '
                       f'points, ArrowFlow minus the comparator, positive when ArrowFlow is more accurate, with the intervals and unadjusted '
                       f'$p$ values of Table~\\ref{{tab:s-comparators}}. Against the best tuned model ArrowFlow is less accurate on all {words(k17)} '
                       f'datasets of both tables, and the interval excludes zero on {num_word(len(clear))}. HCV: hepatitis C virus.')
        write_table(name, caption, label, ['Dataset', 'Comparator', 'ArrowFlow $-$ comparator (pp)', iv_head, '$p$ (unadjusted)'],
                    crows, 'llrrr', block_rules=cblocks, size='\\scriptsize')

    # S3: ranks
    if any(r != int(r) for r in ranks.flatten()):
        raise AssertionError('a rank is fractional, but the rank table prints integers')
    rrows = [[NAME[d]] + [f'{int(ranks[i, j])} ({pct(matrix[i, j], 2)})' for j in range(kk)] for i, d in enumerate(COMBINED)]
    rrows.append(['Mean rank'] + [f'{mean_rank[m]:.2f}' for m in tuned])
    parts = [f'{THE[a]} from {listing([PLAIN[b] for b in bs], "and")}' for a, bs in separated.items()]
    separated_text = ', and '.join(parts) if len(parts) == 2 else listing(parts, 'and')
    plain_p = lambda x: 'p=' + scientific(x) if x < 0.001 else f'p={x:.3f}'
    caption = (f'\\textbf{{Ranks of ArrowFlow and the five tuned classical models on the {words(k17)} datasets.}} Within each dataset, '
               f'rank 1 marks the lowest mean outer-fold error of Table~\\ref{{tab:main}}, given in percent to two decimals in parentheses; '
               f'no two errors tie at that precision. The last row gives the mean rank. The Friedman test, post hoc and descriptive, gives $\\chi^2={chi2:.2f}$ on '
               f'{kk - 1} degrees of freedom (${plain_p(p)}$), and its Iman--Davenport form gives $F={F:.2f}$ (${plain_p(pF)}$). The Nemenyi '
               f'critical difference at $\\alpha=0.05$ is {cd:.2f} mean ranks; it separates {separated_text}. At the one-decimal precision '
               f'of Table~\\ref{{tab:main}}, {listing([NAME[d] for d in tied_1dp], "and")} have tied errors. The tests are descriptive, '
               f'and ranks discard the fold-level uncertainty of Tables~\\ref{{tab:s-comparators}} and~\\ref{{tab:s-comparators-further}}.')
    write_table('tab_s_ranks', caption, 'tab:s-ranks', ['Dataset'] + [MODEL_HEAD[m] for m in tuned], rrows, 'l' + 'r' * kk,
                size='\\scriptsize', subheads=blocks17, block_rules=[k17])

    # S3: duplicates
    drows = [[NAME[d], f'{int(DU.loc[d, "n_rows"]):,}', str(int(DU.loc[d, 'duplicate_rows'])), str(int(DU.loc[d, 'duplicate_groups'])),
              str(int(DU.loc[d, 'label_conflicting_groups'])),
              f"{pct(DU.loc[d, 'share_with_training_duplicate'], 2)} ({pct(DU.loc[d, 'fold_share_min'], 2)}--{pct(DU.loc[d, 'fold_share_max'], 2)})"]
             for d in COMBINED]
    caption = (f'\\textbf{{Exact duplicate rows of the {words(k17)} datasets.}} For every dataset: its rows, its duplicate rows (the copies '
               f'beyond the first of a raw feature vector repeated exactly in double precision), its groups of identical rows and the groups '
               f'whose rows carry different labels. The last column gives the percentage of test rows, pooled over the {n_outer} outer '
               f'folds, that have an exact copy in their training partition, with its range over folds. Descriptive. HCV: hepatitis C virus.')
    write_table('tab_s_duplicates', caption, 'tab:s-duplicates',
                ['Dataset', 'Rows', two_lines('Duplicate', 'rows'), 'Groups', two_lines('Conflicting', 'groups'),
                 two_lines('Test rows with a', 'training copy, \\%')], drows, 'lrrrrr', size='\\scriptsize', subheads=blocks17)

    # S3: the complete record of the training contrasts
    trows, tsub = [], {}
    for family, title in (('knn_training_primary', f'Benchmark datasets: registered family of {words(train["size"])} contrasts'),
                          ('newdata_primary', f'Further datasets, untrained ArrowFlow: registered family of {words(len(NEWDATA))}'),
                          ('newdata_secondary', f'Further datasets, tuned input footrule kNN: registered family of {words(len(NEWDATA))}')):
        tsub[len(trows)] = title
        members = sorted((x for x in keys if reg[x]['family'] == family), key=lambda x: reg[x]['index'])
        if [reg[x]['index'] for x in members] != list(range(1, len(members) + 1)):
            raise AssertionError(f'{family}: the family indices are not consecutive')
        trows += [[str(reg[x]['index']), NAME[x[0]], CONTROL[x[1]], signed(reg[x]['diff'], 2), pct(reg[x]['se'], 2),
                   interval(reg[x]['lo'], reg[x]['hi'], 2), pval(reg[x]['p']), pval(reg[x]['holm']), pval(h34[x])] for x in members]
    caption = (f'\\textbf{{Complete record of the {len(keys)} training contrasts.}} Each row gives the index of the contrast in its '
               f'registered family, the dataset and the control, and ArrowFlow\'s accuracy minus that of the control in percentage points, '
               f'positive when ArrowFlow is more accurate, with its standard error (SE). The next columns hold the {interval_note()}, not adjusted for multiplicity, and the '
               f'approximate $p$ value. The Holm column adjusts within the family that heads the block. The last column applies one Holm '
               f'adjustment over all {len(keys)} contrasts, a post hoc sensitivity analysis. The benchmark family ran under protocol '
               f'\\texttt{{{tex(train["protocol"]["protocol_id"])}}} and the further families under the two batch protocols; '
               f'\\texttt{{holistic training}} recomputed every member from the model rows. Both controls share ArrowFlow\'s encoder '
               f'family but choose their own settings on inner folds; tuned input footrule kNN chose the same encoder settings as ArrowFlow '
               f'on {sum(same_encoder.values())} of the {k17 * n_outer} outer folds. HCV: hepatitis C virus.')
    write_table('tab_s_training_complete', caption, 'tab:s-training',
                ['\\#', 'Dataset', 'Control', 'Diff. (pp)', 'SE', iv_head, '$p$', 'Holm $p$', H34_HEAD], trows, 'rllrrrrrr',
                subheads=tsub, size='\\scriptsize', wide=True)

    # S3: the depth split
    drows2, dsub = [], {0: GROUP_BLOCK['benchmark'], len(DATASETS) * len(depths): GROUP_BLOCK['further']}
    for d in COMBINED:
        for w in depths:
            r, values, label = DS.loc[(d, json.dumps(list(w)))], diffs[(d, w)], f'{NAME[d]}, {widths_label(w)}'
            if not values:
                drows2.append([label, '0', '--', '--', '--', '--', '--', '--'])
                continue
            many = len(values) > 1
            if many and any(abs(a - float(b)) > 1e-9 for a, b in zip(corrected_t(values, conf)[2:4], (r['ci_low'], r['ci_high']))):
                raise AssertionError(f'training by depth, {d} {w}: the group interval is not the t interval at n - 1 degrees of freedom')
            drows2.append([label, str(len(values)), str(same_widths[(d, w)]), signed(r['mean_difference'], 2),
                           pct(r['sd'], 2) if many else '--', signed(r['min'], 2), signed(r['max'], 2),
                           interval(r['ci_low'], r['ci_high'], 2) if many else '--'])
    dsub[len(drows2)] = 'Pooled over datasets, without range or interval'
    for scope, title in (('all', 'All datasets'), ('benchmark', 'Benchmark datasets'), ('further', 'Further datasets')):
        for w in depths:
            q = pooled[(scope, w)]
            drows2.append([f'{title}, {widths_label(w)}', str(q['n']), str(q['same']), signed(q['mean'], 2), pct(q['sd'], 2), '--', '--', '--'])
    caption = (f'\\textbf{{Training effect by selected depth on the {words(k17)} datasets.}} ArrowFlow\'s accuracy minus that of the '
               f'untrained ArrowFlow in percentage points, positive when ArrowFlow is more accurate, with fitting seeds averaged within each '
               f'outer fold, grouped by the hidden widths that ArrowFlow chose on the inner folds of that fold. For every dataset and depth, the table gives the number of outer folds and '
               f'the number on which the untrained ArrowFlow chose the same widths on its own inner folds. It then gives the mean, SD, '
               f'minimum and maximum of the fold differences and, for groups of at least two folds, an interval. Intervals are '
               f'{interval_note(subset=True)}, where $n$ is the number of folds of the group. The last block pools the folds of each '
               f'depth. The split is descriptive, lies outside '
               f'the Holm families and carries no $p$ values. The training protocol declared it for the benchmark datasets; the rows of the '
               f'further datasets and of all datasets are post hoc. HCV: hepatitis C virus.')
    write_table('tab_s_training_depth', caption, 'tab:s-training-depth',
                ['Dataset, depth', 'Folds', 'Same widths', 'Mean (pp)', 'SD', 'Min', 'Max', iv_head], drows2, 'lrrrrrrr',
                subheads=dsub, size='\\scriptsize')

    # S3: the ladder with its dispersion; the benchmark rungs come from three runs, and the further rungs from the batch runs above
    if sorted({source for _, source in LADDER}) != ['knn', 'projected', 'training']:
        raise AssertionError('Table S-ladder: the benchmark rungs no longer come from the ArrowFlow, sorting-control and training-control runs')
    lrows = [[NAME[d]] + [pm(LD.loc[(d, rung), 'mean_error'], LD.loc[(d, rung), 'outer_fold_sd']) for rung, _ in NEW_RUNGS] for d in COMBINED]
    caption = (f'\\textbf{{Nearest-neighbor ladder on the {words(k17)} datasets.}} Mean outer-fold error in percent (outer-fold SD) of '
               f'numeric kNN on the raw features, numeric kNN on the projected scores, tuned input footrule kNN, the untrained ArrowFlow '
               f'and ArrowFlow. Benchmark rungs come from the ArrowFlow run and the runs of the sorting and training controls, and the rungs '
               f'of each further dataset from its batch run. Neighboring rungs differ in more than one component, so '
               f'the ladder is descriptive. kNN: nearest-neighbor classifier; HCV: hepatitis C virus.')
    write_table('tab_s_ladder', caption, 'tab:s-ladder', ['Dataset'] + LADDER_HEAD, lrows, 'lrrrrr', size='\\scriptsize', subheads=blocks17)
    return dict(err=err, best=best, gap=gap, flags=flags, group=group, reg=reg, h34=h34, survivors=survivors, significant=significant,
                higher=higher, keys=keys, mean_rank=mean_rank, chi2=chi2, p=p, F=F, pF=pF, cd=cd, separated=separated,
                tied_1dp=tied_1dp, clear=clear, pooled=pooled, diffs=diffs, two=two, ties=ties, tie_layers=tie_layers, depths=depths,
                vacuous=vacuous, vacuous_layers=vacuous_layers, vacuous_first=vacuous_first,
                declared_v=declared_v, declared_n=declared_n,
                ready=ready, within=[d for d in COMBINED if gap[d] <= COMBINED_RULES['within_points']], same_encoder=same_encoder,
                close=close, cm=CM)


# ----------------------------------------------------------------------------- component ablation on the seventeen datasets (Table 4, S4.1, S5)
COMPONENT_RUN = '2026-09-14-newdata-ablation'
COMPONENT_OUTPUTS = ('components.csv', 'components_depth.csv', 'components_depth_pooled.csv', 'components.json')
COMPONENT_COPIED = ('aggregation', 'confidence', 'depth_split', 'dropped_variants', 'failure_policy', 'fit_reuse', 'fit_seeds',
                    'historical_results', 'inner_folds', 'max_workers', 'numeric_threads_per_worker', 'outer_folds', 'outer_repeats',
                    'parallelism', 'report_metrics', 'selection_metric', 'split_seed', 'test_train_ratio', 'variant_definitions',
                    'variants')
COMPONENT_HEAD = {'views7': two_lines('ArrowFlow', '(seven views)'), 'views1': two_lines('One', 'view'),
                  'views3': two_lines('Three', 'views'), 'no_checkpoint': two_lines('Without', 'checkpoint'),
                  'no_augment': two_lines('Without', 'augmentation'), 'prototype_readout': two_lines('Prototype', 'readout'),
                  'untrained': two_lines('Untrained', 'networks'), 'input_knn': two_lines('Footrule kNN,', 'same inputs')}
COMPONENT_OTHERS = {'untrained': 'the untrained networks', 'input_knn': 'footrule kNN on the same encoded inputs', 'views1': 'one view'}
COMPONENT_STATUS = 'descriptive; no p values; no multiplicity adjustment'
COMPONENT_COMMANDS = 'Component ablation on the further datasets, then the component table of all seventeen'


def render_components(runs, hdir, knn_ablation, kab, newdata, referee):
    """The component ablation of ArrowFlow on the seventeen datasets (Table 4, S4.1). The ablation of the further datasets is verified
    as the benchmark's is, at the selections of the batch runs and against its production log and script, and every value of the
    sealed outputs of holistic components must equal the verified summaries of both ablation runs. Changes are ArrowFlow minus the
    variant, positive when ArrowFlow is more accurate; all of it is descriptive."""
    fam = Family('newdata-ablation', runs / COMPONENT_RUN, 'newdata_ablation_summary.json')
    s, p, B = fam.summary, fam.protocol, newdata['B']
    n, k, n_outer, k17 = len(DATASETS), len(NEWDATA), FACTS['n_outer'], len(COMBINED)
    pairs, conf = n_outer * FACTS['n_seeds'], FACTS['confidence']
    if s is None or p.get('frozen') is not True or sha256_file(PROTOCOLS_V3 / 'newdata_ablation.json') != sha256_file(fam.dir / 'protocol.json'):
        raise AssertionError(f'{fam.name}: no verified summary, or the protocol of the run is not the frozen protocol file')
    if s['protocol_id'] != p['protocol_id'] or s['code_revision'] != fam.environment['code_revision'] or \
            s['inferential_significance_claims'] is not False or 'the summary refuses otherwise' not in s['views7_reproduces_reference']:
        raise AssertionError('newdata_ablation_summary.json: protocol, revision, descriptive status or reproduction rule differs')
    kp = kab['protocol']
    if p['source_template_sha256'] != sha256_file(PROTOCOLS_V3 / 'knn_ablation.json') or any(p[x] != kp[x] for x in COMPONENT_COPIED):
        raise AssertionError('newdata_ablation protocol: its design differs from that of the benchmark ablation, which Section S4 says it copies')
    if p['datasets'] != NEWDATA or p['variants'] != ['views7'] + KNN_VARIANTS or fam.planned != k * n_outer or fam.completed != fam.planned:
        raise AssertionError('newdata_ablation: its datasets, variants or completed jobs differ from the tables')
    for b, batch in B.items():
        ref = s['reference_sources'][str(b)]
        if (ref['protocol_id'], ref['protocol_sha256'], ref['code_revision'], ref['summary_sha256'], ref['model_id'], sorted(ref['datasets'])) != \
                (batch.protocol['protocol_id'], sha256_file(batch.dir / 'protocol.json'), batch.environment['code_revision'],
                 sha256_file(batch.dir / 'summary.json'), KNN, sorted(batch.protocol['datasets'])):
            raise AssertionError(f'newdata_ablation_summary.json: reference {b} is not the batch run on disk')
    shown = {'"$B1"': '<batch 1 run>', '"$B2"': '<batch 2 run>', '"$OUT"': '<run>', '"$W/runs"': '<runs directory>',
             f'"$W/runs/{HOLISTIC}/components"': '<combined analysis run>/components'}
    log = production_script(runs, 'task24-run.log', 'task24-prod.sh', 'task24',
                            lambda word, v: {'$P': protocol_path(v['P']), **shown}.get(word, word))
    text, script = (runs / 'task24-run.log').read_text(), (PRODUCTION_SCRIPTS / 'task24-prod.sh').read_text()
    gate = re.search(r'^views7 reproduced (\d+)/(\d+) lines: (\d+)$', text, re.M)
    if not gate or tuple(int(x) for x in gate.groups()) != (pairs, pairs, k) or f'grep -c "views7 reproduced {pairs}/{pairs}"' not in script \
            or f'test "$n" -eq {k}' not in script or not script.index(f'test "$n" -eq {k}') < script.index('holistic components'):
        raise AssertionError('task24 log and script: the seven-view reproduction was not checked on every further dataset before the table')
    if log['pinned'] != fam.environment['code_revision'] or list(log['stages']) != ['prepare', 'ablation', 'summary', 'components'] or \
            not p['frozen_at_utc'][:19] < log['start'][:19] or '.worktrees/arrowflow-v3-task24-run' not in script:
        raise AssertionError('task24 log and script: the stages, pinned revision, freeze order or worktree differ from Section S5')
    pilots = len(re.findall(r'training-only pilot', p['resource_decision']))
    if not re.search(r'synthetic [\w-]*\s*smoke', p['resource_decision']) or pilots < 1:
        raise AssertionError('newdata_ablation protocol: its resource decision records no synthetic smoke stage or training-only pilot')

    # the ablation of the further datasets, verified as the benchmark's
    runs_of = {d: batch for batch in B.values() for d in batch.protocol['datasets']}
    records = {key: rec for batch in B.values() for key, rec in selection_records(batch, KNN, batch.protocol['datasets']).items()}
    identical, _, G, depths = verify_ablation(fam.name, s, p, pd.read_csv(fam.dir / 'newdata_ablation_summary.csv'), NEWDATA,
                                              lambda d: lookup(runs_of[d].summary['summaries'][d], KNN, 'error'), records)
    if depths != kab['depths']:
        raise AssertionError('the two ablations group their outer folds by different depths')
    summary_of = {**dict.fromkeys(DATASETS, knn_ablation.summary), **dict.fromkeys(NEWDATA, s)}
    ident, groups = {**kab['identical'], **identical}, {**kab['G'], **G}
    variant = lambda d, v: summary_of[d]['summaries'][d]['variants'][v]
    change = lambda d, v: variant(d, v)['change_from_views7']['accuracy']
    gain = {(d, v): -change(d, v)['mean_difference'] for d in COMBINED for v in KNN_VARIANTS}
    bounds = {(d, v): (-change(d, v)['ci_high'], -change(d, v)['ci_low']) for d in COMBINED for v in KNN_VARIANTS}
    if any(gain[(d, v)] != 0 or bounds[(d, v)] != (0, 0) for d in COMBINED for v in ident[d]):
        raise AssertionError('a variant that coincides with ArrowFlow by construction has a nonzero change or interval')
    if any(gain[(d, v)] == 0 for d in COMBINED for v in KNN_VARIANTS if v not in ident[d]):
        raise AssertionError('a fitted variant has exactly the accuracy of ArrowFlow, but every count sorts the datasets by sign')

    # the sealed outputs of holistic components equal the verified summaries of both runs
    cdir = hdir / 'components'
    cj = read_json(cdir / 'components.json')
    if sorted(x.name for x in cdir.iterdir()) != sorted(COMPONENT_OUTPUTS) or sorted(cj['outputs']) != sorted(COMPONENT_OUTPUTS[:3]) or \
            any(sha256_file(cdir / name) != out['sha256'] for name, out in cj['outputs'].items()):
        raise AssertionError(f'{cdir}: the files on disk differ from the sealed outputs of holistic components')
    prov = cj['provenance']
    if prov['sources_committed_at_revision'] is not True or prov['code_revision'] != fam.environment['code_revision'] or \
            cj['datasets'] != {'all': COMBINED, 'benchmark': DATASETS, 'further': NEWDATA} or cj['variants'] != KNN_VARIANTS:
        raise AssertionError('components.json: not written from committed sources at the revision of the ablation run, or its panel differs')
    for key, run, stem in (('knn_ablation', knn_ablation, 'knn_ablation_summary'), ('newdata_ablation', fam, 'newdata_ablation_summary')):
        r = prov['runs'][key]
        if (r['protocol_id'], r['frozen'], r['code_revision'], r['jobs_verified']) != \
                (run.protocol['protocol_id'], True, run.summary['code_revision'], run.planned) or \
                any(r['files'][x]['sha256'] != sha256_file(run.dir / x) for x in ('protocol.json', 'manifest.json', f'{stem}.json', f'{stem}.csv')) or \
                r['views7_reproduces_reference'] != {d: e['views7_reproduces_reference'] for d, e in run.summary['summaries'].items()}:
            raise AssertionError(f'components.json: the {key} run it names is not the verified run on disk')
    frame, keys = pd.read_csv(cdir / 'components.csv'), [(d, v) for d in COMBINED for v in KNN_VARIANTS]
    group = {**dict.fromkeys(DATASETS, 'benchmark'), **dict.fromkeys(NEWDATA, 'further')}
    source = {**dict.fromkeys(DATASETS, 'knn_ablation'), **dict.fromkeys(NEWDATA, 'newdata_ablation')}
    if list(zip(frame['dataset'], frame['variant'])) != keys or [(j['dataset'], j['variant']) for j in cj['components']] != keys:
        raise AssertionError('components.csv and components.json: rows differ from the seventeen datasets and seven variants in panel order')
    for (_, r), j, (d, v) in zip(frame.iterrows(), cj['components'], keys):
        metrics, views7 = variant(d, v)['metrics'], variant(d, 'views7')['metrics']
        expected = dict(status=COMPONENT_STATUS, group=group[d], source_run=source[d], n_folds=n_outer, df=FACTS['df'],
                        identical_to_views7_folds=n_outer if v in ident[d] else 0)
        if any(r[c] != x or j[c] != x for c, x in expected.items()):
            raise AssertionError(f'components/{d}/{v}: status, group, source, folds or coinciding folds differ from the verified runs')
        for column, value in (('arrowflow_minus_variant', gain[(d, v)]), ('standard_error', change(d, v)['standard_error']),
                              ('ci_low', bounds[(d, v)][0]), ('ci_high', bounds[(d, v)][1]),
                              ('variant_accuracy', metrics['accuracy']['mean']), ('views7_accuracy', views7['accuracy']['mean']),
                              ('variant_error', metrics['error']['mean']), ('views7_error', views7['error']['mean'])):
            agrees(r[column], value, f'components.csv/{d}/{v}/{column}')
            agrees(j[column], value, f'components.json/{d}/{v}/{column}')
    fitted = lambda v: [d for d in COMBINED if v not in ident[d]]
    counts = {v: {'arrowflow_more_accurate': [d for d in fitted(v) if gain[(d, v)] > 0],
                  'variant_more_accurate': [d for d in fitted(v) if gain[(d, v)] < 0],
                  'identical_to_views7_in_every_fold': [d for d in COMBINED if v in ident[d]],
                  'interval_excludes_zero': [d for d in fitted(v) if bounds[(d, v)][0] > 0 or bounds[(d, v)][1] < 0]}
              for v in KNN_VARIANTS}
    if cj['counts'] != counts:
        raise AssertionError('components.json: the counts differ from the verified changes and intervals')
    DEP = pd.read_csv(cdir / 'components_depth.csv')
    listed = [(d, tuple(g['widths'])) for d in COMBINED for g in summary_of[d]['depth_split']['by_dataset'][d]]
    if sorted(listed) != sorted((d, w) for d in COMBINED for w in depths) or \
            list(zip(DEP['dataset'], DEP['widths'])) != [(d, json.dumps(list(w))) for d, w in listed]:
        raise AssertionError('components_depth.csv: rows differ from the depth groups of the seventeen datasets')
    negate = lambda x: None if x is None else -x
    for (_, r), (d, w) in zip(DEP.iterrows(), listed):
        values, iv = groups[(d, w)]['fold_differences'], groups[(d, w)]['interval'] or {}
        if r['group'] != group[d] or int(r['n_folds']) != len(values):
            raise AssertionError(f'components_depth.csv/{d}/{w}: the group or its folds differ from the verified depth split')
        middle = statistics.mean(values) if values else None
        for column, value in (('untrained_minus_views7', middle), ('sd', statistics.stdev(values) if len(values) > 1 else None),
                              ('min', min(values) if values else None), ('max', max(values) if values else None),
                              ('ci_low', iv.get('ci_low')), ('ci_high', iv.get('ci_high')), ('views7_minus_untrained', negate(middle)),
                              ('views7_minus_untrained_ci_low', negate(iv.get('ci_high'))),
                              ('views7_minus_untrained_ci_high', negate(iv.get('ci_low')))):
            agrees(r[column], value, f'components_depth.csv/{d}/{w}/{column}')
    POOL, pooled = pd.read_csv(cdir / 'components_depth_pooled.csv'), {}
    if len(POOL) != 3 * len(depths):
        raise AssertionError('components_depth_pooled.csv: rows differ from three scopes and the declared depths')
    for scope, names in (('all', COMBINED), ('benchmark', DATASETS), ('further', NEWDATA)):
        for w in depths:
            values = [-x for d in names for x in groups[(d, w)]['fold_differences']]    # ArrowFlow minus the untrained networks
            with_folds = [d for d in names if groups[(d, w)]['fold_differences']]
            row = POOL[(POOL['scope'] == scope) & (POOL['widths'] == json.dumps(list(w)))]
            if len(row) != 1 or int(row.iloc[0]['n_dataset_folds']) != len(values) or \
                    sorted(str(row.iloc[0]['datasets']).split()) != sorted(with_folds):
                raise AssertionError(f'components_depth_pooled.csv/{scope}/{w}: the pooled folds differ from the verified depth split')
            agrees(row.iloc[0]['views7_minus_untrained'], statistics.mean(values), f'components_depth_pooled.csv/{scope}/{w}/mean')
            agrees(row.iloc[0]['untrained_minus_views7'], -statistics.mean(values), f'components_depth_pooled.csv/{scope}/{w}/negated')
            agrees(row.iloc[0]['sd'], statistics.stdev(values), f'components_depth_pooled.csv/{scope}/{w}/sd')
            pooled[(scope, w)] = dict(n=len(values), mean=statistics.mean(values), sd=statistics.stdev(values), datasets=with_folds)

    # the facts that the text states (ruling F-A17)
    mean = {v: statistics.mean(gain[(d, v)] for d in COMBINED) for v in KNN_VARIANTS}
    largest = {}
    for d in COMBINED:
        effect = {v: gain[(d, v)] for v in KNN_VARIANTS if v not in ident[d]}
        ranked = sorted(effect, key=lambda v: abs(effect[v]), reverse=True)
        if effect[ranked[0]] <= 0 or pct(abs(effect[ranked[0]]), 2) == pct(abs(effect[ranked[1]]), 2):
            raise AssertionError(f'{d}: the largest change is no gain of ArrowFlow or ties at two decimals, but Table 4 bolds one gain')
        largest[d] = ranked[0]
    if max(mean, key=mean.get) != 'prototype_readout':
        raise AssertionError('the text says the prototype readout has the largest mean change')
    proto = {d: gain[(d, 'prototype_readout')] for d in COMBINED}
    top2 = sorted(COMBINED, key=proto.get, reverse=True)[:2]
    rest = [d for d in COMBINED if d not in top2]
    if top2[1] != referee['wq'] or proto[top2[0]] + proto[top2[1]] <= sum(proto.values()) / 2 or statistics.mean(proto[d] for d in rest) <= 0:
        raise AssertionError('the text says most of the prototype readout\'s mean change comes from two datasets, the second Wine quality, '
                             'and that the prototype readout is less accurate on average over the others')
    by_top = {v: [d for d in COMBINED if largest[d] == v] for v in KNN_VARIANTS}
    if {v for v, ds in by_top.items() if ds} != {'prototype_readout', *COMPONENT_OTHERS}:
        raise AssertionError('the text names the datasets on which the untrained networks, footrule kNN on the same encoded inputs or one '
                             'view has the largest change, and the prototype readout has it on the others')
    more = {v: counts[v]['arrowflow_more_accurate'] for v in KNN_VARIANTS}
    less = {v: counts[v]['variant_more_accurate'] for v in KNN_VARIANTS}
    worst = min(less['views1'], key=lambda d: gain[(d, 'views1')])
    if gain[(worst, 'views3')] <= 0 or len(less['views1']) < 2:
        raise AssertionError('the text says three views are less accurate than seven where one view is most ahead of seven')
    off = counts['no_augment']['identical_to_views7_in_every_fold']
    if any(v != 'no_augment' for d in COMBINED for v in ident[d]) or sorted(off) != sorted(d for d in COMBINED if ident[d]):
        raise AssertionError('Table 4 marks n/a only where the variant without augmentation coincides with ArrowFlow')
    if any(counts[v]['interval_excludes_zero'] for v in ('no_checkpoint', 'no_augment')) or mean['no_checkpoint'] <= 0 or \
            mean['no_augment'] <= 0:
        raise AssertionError('the text says removing the checkpoint or augmentation lowers mean accuracy, with no interval excluding zero')
    excl = counts['untrained']['interval_excludes_zero']
    top_u = max(COMBINED, key=lambda d: gain[(d, 'untrained')])
    if excl != counts['input_knn']['interval_excludes_zero'] or any(bounds[(d, v)][0] <= 0 for d in excl for v in ('untrained', 'input_knn')) \
            or top_u != max(COMBINED, key=lambda d: gain[(d, 'input_knn')]):
        raise AssertionError('the text says the intervals of both untrained variants exclude zero on the same datasets, in favor of '
                             'ArrowFlow, and that both gain most on one dataset')
    single, double = depths
    if len(pooled[('all', single)]['datasets']) != k17 or len(pooled[('all', double)]['datasets']) >= k17:
        raise AssertionError('the text says the one-layer folds come from every dataset and the two-layer folds from fewer')

    # tables
    blocks17 = {0: GROUP_BLOCK['benchmark'], n: GROUP_BLOCK['further']}
    heads = ['Dataset'] + [COMPONENT_HEAD[v] for v in KNN_VARIANTS]
    mean_row = [f'Mean over the {words(k17)} datasets'] + [signed(mean[v], 2) for v in KNN_VARIANTS]
    n_a = ('n/a: the variant coincides with ArrowFlow because the adaptive rule already switches augmentation off')

    def change_cell(d, v, bold=False, with_interval=False):
        if v in ident[d]:
            return 'n/a'
        cell = signed(gain[(d, v)], 2)
        cell = f'\\textbf{{{cell}}}' if bold and largest[d] == v else cell
        return cell + (f' {interval(*bounds[(d, v)], 2)}' if with_interval else '')

    rows = [[NAME[d]] + [change_cell(d, v, bold=True) for v in KNN_VARIANTS] for d in COMBINED] + [mean_row]
    caption = (f'\\textbf{{Component ablation of ArrowFlow on the {words(k17)} datasets.}} Mean paired difference over the {n_outer} '
               f'outer folds, ArrowFlow\'s accuracy minus the variant\'s in percentage points after averaging the fitting seeds within '
               f'each fold; positive when ArrowFlow is more accurate. Every variant is evaluated at the configuration that ArrowFlow chose '
               f'on the inner folds of each outer fold, which favors ArrowFlow against the prototype readout, the untrained networks and '
               f'footrule kNN on the same encoded inputs. One and three views are the first views of ArrowFlow\'s seven, and the '
               f'prototype readout predicts with the output layers of the same networks; the other variants are fitted separately. '
               f'{n_a}; the {words(len(off))} such datasets contribute zero to the mean without augmentation. Bold marks the largest change '
               f'of each dataset. The prototype readout changes most on '
               f'{NAME[top2[0]]} and {NAME[top2[1]]}. Everything is descriptive; '
               f'Section~S4 gives the intervals, the errors and the split by selected depth. kNN: nearest-neighbor classifier; HCV: '
               f'hepatitis C virus.')
    write_table('tab_ablation', caption, 'tab:ablation', heads, rows, 'l' + 'r' * len(KNN_VARIANTS), size='\\scriptsize',
                subheads=blocks17, block_rules=[k17], wide=True)

    def error_cell(d, v):
        if v in ident[d]:
            return 'n/a'
        e = variant(d, v)['metrics']['error']
        return f"{pm(e['mean'], e['outer_fold_sd'])} [{pct(e['mean_within_fold_seed_sd'])}]"
    erows = [[NAME[d]] + [error_cell(d, v) for v in ['views7'] + KNN_VARIANTS] for d in COMBINED]
    caption = (f'\\textbf{{Error of ArrowFlow and of every variant of the component ablation.}} Mean outer-fold error in percent '
               f'(outer-fold SD) [mean within-fold SD across the {words(FACTS["n_seeds"])} fitting seeds] on the {words(k17)} datasets, '
               f'every variant at the configuration that ArrowFlow chose on the inner folds of each outer fold. The seven-view refit '
               f'predicted every test example as ArrowFlow did in Table~\\ref{{tab:main}} on all {pairs} fold--seed pairs of every '
               f'dataset, so its errors are those of ArrowFlow. {n_a}. Descriptive. kNN: nearest-neighbor classifier; HCV: hepatitis C '
               f'virus.')
    write_table('tab_s_components_errors', caption, 'tab:s-components-errors', ['Dataset'] + [COMPONENT_HEAD[v] for v in ['views7'] + KNN_VARIANTS],
                erows, 'l' + 'r' * (1 + len(KNN_VARIANTS)), size='\\scriptsize', subheads=blocks17, stack=4, wide=True)

    crows = [[NAME[d]] + [change_cell(d, v, with_interval=True) for v in KNN_VARIANTS] for d in COMBINED] + [mean_row]
    caption = (f'\\textbf{{ArrowFlow minus every variant, with intervals.}} The differences of the component ablation of '
               f'Section~\\ref{{sec:components}} on the '
               f'{words(k17)} datasets, ArrowFlow\'s accuracy minus the variant\'s in percentage points with fitting seeds averaged within '
               f'each outer fold, positive when ArrowFlow is more accurate. Each carries the {interval_note()}, not adjusted for '
               f'multiplicity. {n_a}. The last row averages the datasets, counting n/a as zero, and has no interval. Everything is '
               f'descriptive. The benchmark datasets ran under protocol \\texttt{{{tex(kp["protocol_id"])}}} '
               f'and the further datasets under '
               f'\\texttt{{{tex(p["protocol_id"])}}}. kNN: nearest-neighbor classifier; HCV: hepatitis C virus.')
    write_table('tab_s_components_changes', caption, 'tab:s-components-changes', heads, crows, 'l' + 'r' * len(KNN_VARIANTS),
                size='\\scriptsize', subheads=blocks17, block_rules=[k17], stack=4, wide=True)

    drows, dsub = [], {0: GROUP_BLOCK['benchmark'], n * len(depths): GROUP_BLOCK['further']}
    for d in COMBINED:
        for w in depths:
            values, label = [-x for x in groups[(d, w)]['fold_differences']], f'{NAME[d]}, {widths_label(w)}'
            if not values:
                drows.append([label, '0', '--', '--', '--', '--', '--'])
                continue
            many, iv = len(values) > 1, groups[(d, w)]['interval']
            if many and any(abs(a - b) > 1e-9 for a, b in zip(corrected_t(values, conf)[2:4], (-iv['ci_high'], -iv['ci_low']))):
                raise AssertionError(f'components by depth, {d} {w}: the group interval is not the t interval at n - 1 degrees of freedom')
            drows.append([label, str(len(values)), signed(statistics.mean(values), 2), pct(statistics.stdev(values), 2) if many else '--',
                          signed(min(values), 2), signed(max(values), 2), interval(-iv['ci_high'], -iv['ci_low'], 2) if many else '--'])
    dsub[len(drows)] = 'Pooled over datasets, without range or interval'
    for scope, title in (('all', 'All datasets'), ('benchmark', 'Benchmark datasets'), ('further', 'Further datasets')):
        for w in depths:
            q = pooled[(scope, w)]
            drows.append([f'{title}, {widths_label(w)}', str(q['n']), signed(q['mean'], 2), pct(q['sd'], 2), '--', '--', '--'])
    caption = (f'\\textbf{{ArrowFlow minus the untrained networks by selected depth.}} ArrowFlow\'s accuracy minus that of the untrained '
               f'networks in percentage points, positive when ArrowFlow is more accurate, with fitting seeds averaged within each outer '
               f'fold. The folds are grouped by the hidden widths that ArrowFlow chose on their inner folds. For every dataset and depth, '
               f'the table gives the number of outer folds and the mean, SD, minimum and maximum of the fold differences, with an '
               f'interval for groups of at least two folds. Intervals are {interval_note(subset=True)}, where $n$ is the number of folds '
               f'of the group. The last block pools the folds of each '
               f'depth. These untrained networks sit at ArrowFlow\'s configuration, unlike the tuned untrained ArrowFlow of '
               f'Table~\\ref{{tab:s-training-depth}}. The split is descriptive and confounded with dataset. HCV: hepatitis C virus.')
    write_table('tab_s_components_depth', caption, 'tab:s-components-depth',
                ['Dataset, depth', 'Folds', 'Mean (pp)', 'SD', 'Min', 'Max', f'{conf}\\% interval'], drows, 'lrrrrrr', subheads=dsub,
                size='\\scriptsize')
    return dict(family=fam, log=log, pilots=pilots, gain=gain, bounds=bounds, ident=ident, mean=mean, largest=largest, by_top=by_top,
                top2=top2, rest=rest, proto=proto, counts=counts, more=more, less=less, worst=worst, off=off, excl=excl, top_u=top_u,
                pooled=pooled, depths=depths, pairs=pairs, record=cj, protocol=p, knn_protocol=kp)


def components_claims(cf, referee, ready):
    """(file, phrase) for the guarded numbers, counts and facts of the component ablation on the seventeen datasets (ruling F-A17) and
    of its production record in Section S5, rebuilt from the verified ablation runs, the production log and script of the further
    datasets' ablation and the sealed outputs of holistic components."""
    E, D, K = 'sections/07_experiments.tex', 'sections/08_discussion.tex', 'sections/09_conclusion.tex'
    S4, S5 = 'supplement_sections/S4_ablations.tex', 'supplement_sections/S5_reproducibility.tex'
    k17, k, R = len(COMBINED), len(NEWDATA), referee
    gain, mean, counts, pooled, more, less = cf['gain'], cf['mean'], cf['counts'], cf['pooled'], cf['more'], cf['less']
    p, kp, fam, log, st = cf['protocol'], cf['knn_protocol'], cf['family'], cf['log'], cf['log']['stages']
    (first, wq), proto, rest, worst = cf['top2'], cf['proto'], cf['rest'], cf['worst']
    rest_mean = statistics.mean(proto[d] for d in rest)
    plus, two = (lambda x: f'{100 * float(x):+.2f}'), (lambda x: f'{100 * float(x):.2f}')
    names = lambda ds: listing([NAME[d] for d in ds], 'and')
    others = [f'{COMPONENT_OTHERS[v]} on {names(cf["by_top"][v])}' for v in COMPONENT_OTHERS]
    max_other = max(-gain[(d, 'views1')] for d in less['views1'] if d != worst)
    one, two_layer = pooled[('all', cf['depths'][0])], pooled[('all', cf['depths'][1])]
    fitted_augment = [d for d in COMBINED if d not in cf['off']]
    rev, frozen = fam.environment['code_revision'][:9], p['frozen_at_utc']
    fitted_mean = statistics.mean(gain[(d, 'no_augment')] for d in fitted_augment)
    if len(fitted_augment) + len(cf['off']) != k17 or abs(mean['no_augment'] * k17 - fitted_mean * len(fitted_augment)) > 1e-12:
        raise AssertionError('the mean without augmentation over the seventeen datasets is not its fitted mean with zeros for the others')
    if not ready['written_utc'][:19] < frozen[:19]:
        raise AssertionError('Section S5 says the protocol of the further datasets\' ablation was frozen after the combined analysis was written')

    # Section S5: the commands must be those of the production script, and a changed argument must make the check fail
    S5_text = (HERE / 'supplement_sections' / 'S5_reproducibility.tex').read_text()
    if verbatim_commands(S5_text, COMPONENT_COMMANDS) != log['commands']:
        raise AssertionError('Section S5: the commands of the component ablation of the further datasets differ from its production script')
    block = S5_text.split('# ' + COMPONENT_COMMANDS, 1)[1].split('\n# ', 1)[0]
    for old, new in (('--workers 16', '--workers 12'), ('<batch 2 run> --output <run>', '<batch 1 run> --output <run>')):
        if old not in block or verbatim_commands(S5_text.replace(block, block.replace(old, new, 1), 1), COMPONENT_COMMANDS) == log['commands']:
            raise SystemExit(f'command guard mutation test ({old} -> {new}): not detected')
    print('command guard mutation test (component ablation of the further datasets): 2 value-only mutations detected')

    if max(mean, key=mean.get) != 'prototype_readout':
        raise AssertionError('S4.1 says the prototype readout has the largest mean change of any variant')
    return [
        # the details of Section 7.4, moved to S4.1 by the simplification pass (the main-text phrases are guarded in simplify_claims;
        # the depth split and the six datasets without augmentation were already guarded in S4.1)
        (S4, f'The prototype readout has the largest mean change of any variant, ${plus(mean["prototype_readout"])}$ pp over the '
             f'{words(k17)} datasets.'),
        (S4, f'Most of this change comes from {NAME[first]} (${plus(proto[first])}$ pp) and {NAME[wq]} (${plus(proto[wq])}$ pp), and '
             f'on the other {words(len(rest))} datasets the mean change is ${plus(rest_mean)}$ pp.'),
        (S4, f'The prototype readout also has the largest change on {words(len(cf["by_top"]["prototype_readout"]))} datasets.'),
        (S4, 'On the others, the largest change comes from ' + ', '.join(others[:-1]) + ', and ' + others[-1] + '.'),
        (S4, f'One view is more accurate on the other {words(len(less["views1"]))}, by ${two(-gain[(worst, "views1")])}$ pp on '
             f'{NAME[worst]} and by at most ${two(max_other)}$ pp on the others, and on {NAME[worst]} three views are '
             f'${two(gain[(worst, "views3")])}$ pp less accurate than seven.'),
        (S4, f'Removing the checkpoint, which selects filters by the prototype readout\'s validation error, lowers mean accuracy by '
             f'${two(mean["no_checkpoint"])}$ pp.'),
        # S4.1
        (S4, f'The benchmark datasets ran under protocol \\texttt{{{tex(kp["protocol_id"])}}} and the further datasets under '
             f'\\texttt{{{tex(p["protocol_id"])}}}, which copies its design and takes the configuration of each outer fold from the batch '
             f'run of the dataset.'),
        (S4, f'reproduced ArrowFlow\'s outer predictions exactly on all {cf["pairs"]} fold--seed pairs of every dataset'),
        (S4, f'So is the variant without augmentation, but only where the adaptive rule switches augmentation on (not on {names(cf["off"])}).'),
        (S4, f'In Table~\\ref{{tab:s-components-changes}}, ArrowFlow is more accurate than the prototype readout on '
             f'{len(more["prototype_readout"])} of the {k17} datasets, by up to ${two(proto[first])}$ pp ({NAME[first]}), and less accurate '
             f'on ' + listing([f'{NAME[d]} (${plus(proto[d])}$ pp)' for d in less['prototype_readout']], 'and')
             + f'. The interval excludes zero on {words(len(counts["prototype_readout"]["interval_excludes_zero"]))} datasets.'),
        (S4, f'ArrowFlow is more accurate than the untrained networks on {words(len(more["untrained"]))} datasets and than footrule kNN on '
             f'the same encoded inputs on {words(len(more["input_knn"]))}, by up to ${two(gain[(cf["top_u"], "untrained")])}$ and '
             f'${two(gain[(cf["top_u"], "input_knn")])}$ pp, both on {NAME[cf["top_u"]]}.'),
        (S4, f'For both variants, the interval excludes zero on {names(cf["excl"])}, each time in favor of ArrowFlow.'),
        (S4, f'Seven views are more accurate than one, the target-aware first view, on {words(len(more["views1"]))} datasets and '
             f'than three on '
             f'{words(len(more["views3"]))}.'),
        (S4, f'ArrowFlow is more accurate than the variant without the checkpoint on {words(len(more["no_checkpoint"]))} datasets, and '
             f'than the variant without augmentation on {len(more["no_augment"])} of the {len(fitted_augment)} datasets where that '
             f'variant is fitted.'),
        (S4, f'The mean change without augmentation, ${plus(mean["no_augment"])}$ pp over the {words(k17)} datasets, counts the '
             f'{words(len(cf["off"]))} datasets where the variant coincides with ArrowFlow as zero. Over the '
             f'{words(len(fitted_augment))} datasets where it is fitted, the mean is ${plus(fitted_mean)}$ pp.'),
        (S4, 'No interval of either variant excludes zero.'),
        (S4, f'By selected depth (Table~\\ref{{tab:s-components-depth}}), ArrowFlow minus the untrained networks is ${plus(one["mean"])}$ '
             f'pp over the {one["n"]} outer folds with one hidden layer and ${plus(two_layer["mean"])}$ pp over the {two_layer["n"]} with '
             f'two.'),
        (S4, f'The two-layer folds come from {words(len(two_layer["datasets"]))} datasets and the one-layer folds from all {words(k17)}, '
             f'so depth and dataset are confounded, and the split is descriptive.'),
        # S5
        (S5, f'Its component table was written at revision \\texttt{{{cf["record"]["provenance"]["code_revision"][:9]}}}, after both '
             f'component ablations were complete.'),
        (S5, f'The component ablation of the further datasets ran under its own frozen protocol, \\texttt{{{p["protocol_id"]}}} (file '
             f'\\texttt{{2026-09-12/newdata\\_ablation.json}}, whose \\texttt{{sha256}} digest begins with '
             f'\\texttt{{{sha256_file(PROTOCOLS_V3 / "newdata_ablation.json")[:HASH_DIGITS]}}}), at code revision \\texttt{{{rev}}}.'),
        (S5, f'It copies the design of \\texttt{{{kp["protocol_id"]}}} and takes the configuration of every outer fold from the batch run '
             f'holding the dataset. Its exact reproduction check is specific to the same machine and environment.'),
        (S5, f'The component ablation of the further datasets ran a synthetic smoke stage and {words(cf["pilots"])} training-only '
             f'pilot{"s" if cf["pilots"] > 1 else ""} before its protocol was frozen.'),
        (S5, f'Its production script ran from a git worktree detached at commit \\texttt{{{log["head"]}}} on {log["start"][:10]} and '
             f'logged \\texttt{{prepare}} at {st["prepare"]} UTC, \\texttt{{ablation}} at {st["ablation"]}, \\texttt{{summary}} at '
             f'{st["summary"]} and \\texttt{{holistic components}} at {st["components"]}. It ended at {log["end"][11:19]} UTC with exit '
             f'status {log["exit"]}.'),
        (S5, f'The summary refuses unless the seven-view refit reproduces ArrowFlow on every fold and seed. The script checked this for '
             f'all {words(k)} datasets before it wrote the component table.'),
        (S5, f'The protocol of the component ablation of the further datasets was frozen on {frozen[:10]} at {frozen[11:16]} UTC, after the '
             f'combined analysis was written. Its run executed at code revision \\texttt{{{rev}}}.'),
        (S5, f'For the component ablation of ArrowFlow it reads \\texttt{{knn\\_ablation\\_summary.json}} and \\texttt{{.csv}}, '
             f'\\texttt{{newdata\\_ablation\\_summary.json}} and \\texttt{{.csv}}, the selection records of the batch runs, and the '
             f'{words(len(COMPONENT_OUTPUTS))} outputs of \\texttt{{holistic components}}.'),
    ]


# ----------------------------------------------------------------------------- guarded prose of the simplified main text
def simplify_claims(combined, newdata, controlled, components, referee):
    """Every measured number and plain-language approximation that the simplified main text and its new supplement sentences
    state, rebuilt from the run data under the rules of plain_fact; each phrase registers its own mutation case."""
    I, E, D, K, S3 = INTRO, E7, 'sections/08_discussion.tex', 'sections/09_conclusion.tex', SUPP3
    C, ND, R, cf = combined, newdata, referee, components
    mo, dg, me = controlled['motion'], controlled['diagnostics'], controlled['mechanism']
    de, ag, dd, bl = controlled['depth'], controlled['aggregation'], controlled['dedup'], controlled['baselines']
    n, k, k17 = len(DATASETS), len(NEWDATA), len(COMBINED)
    tuned = len(COMPARATORS) + 1

    # the benchmark: ArrowFlow trails the best tuned classical model everywhere; its average rank and neighbors
    gap, mr = C['gap'], C['mean_rank']
    plain_rule(all(gap[d] > 0 for d in COMBINED), 'ArrowFlow trails the best tuned classical model on every dataset')
    all17, fewer = f'all {words(k17)}', f'{words(k17 - 1)} of the {words(k17)}'
    dmin, dmax = min(COMBINED, key=gap.get), max(COMBINED, key=gap.get)
    order = sorted(mr, key=mr.get)
    at = order.index(KNN)
    position, ahead_of, behind = ORDINAL[at + 1], order[at - 1], order[at + 1]
    # the mean rank over the ten further datasets alone (referee panel of 2026-09-23, item Q2): the within-dataset ranks of the
    # six tuned models, averaged over those datasets
    from itertools import combinations
    from scipy import stats as st
    six = [KNN] + COMPARATORS
    further_rank = {m: float(np.mean([st.rankdata([C['err'](x, y) for y in six])[six.index(m)] for x in NEWDATA])) for m in six}
    further_order = sorted(six, key=further_rank.get)
    further_position = 1 + sum(further_rank[m] < further_rank[KNN] for m in six)
    plain_rule([m for m in six if further_rank[m] > further_rank[KNN]] == [further_order[-1]] and further_order[-1] == 'numeric_knn'
               and further_position == ORDINAL_INDEX['fifth'], 'on the ten further datasets ArrowFlow is fifth, ahead of numeric kNN only')
    C['further_rank'], C['bench_rank'] = further_rank, {m: float(np.mean([st.rankdata([C['err'](x, y) for y in six])[six.index(m)]
                                                                          for x in DATASETS])) for m in six}
    # the post hoc stratum check without the three flagged datasets (item J2)
    flagged = [x for x in COMBINED if C['flags'][x]]
    if not set(flagged) <= set(ND['others']):
        raise AssertionError('the flagged datasets are not all in the unmarked stratum of the stratum test')
    kept_new = [x for x in NEWDATA if x not in flagged]
    stat = lambda group: float(np.mean([ND['effect'][x] for x in group]) - np.mean([ND['effect'][x] for x in kept_new if x not in group]))
    observed = stat([x for x in ND['marked'] if x in kept_new])
    null = [stat(g) for g in combinations(kept_new, len([x for x in ND['marked'] if x in kept_new]))]
    flagged_at_least, flagged_p = sum(v >= observed - 1e-12 for v in null), sum(v >= observed - 1e-12 for v in null) / len(null)
    C['flagged_test'] = dict(flagged=flagged, at_least=flagged_at_least, splits=len(null), p=flagged_p)
    plain_rule([a for a, bs in C['separated'].items() if KNN in bs] == ['svc_rbf'] and not C['separated'].get(KNN),
               'only the SVC ranks clearly ahead of ArrowFlow in the post hoc comparison')
    # training: more accurate than the untrained ArrowFlow almost everywhere, within one point on most datasets, three significant
    higher, close = len(C['higher']), len(C['close'])
    sig = {d: sorted({m for x, m in C['significant'] if x == d}) for d, _ in C['significant']}
    order_sig = sorted(sig, key=COMBINED.index)
    if len(order_sig) != 3 or sig[order_sig[0]] != [INPUT_KNN] or any(sig[d] != sorted(CONTROLS) for d in order_sig[1:]):
        raise AssertionError('the text says one dataset is significant against tuned input footrule kNN and two against both controls')
    vd, a2, b2 = order_sig
    if sorted(C['survivors']) != sorted((d, m) for d in (a2, b2) for m in CONTROLS):
        raise AssertionError('the text says only the last two datasets survive the post hoc adjustment over all contrasts')
    mt = ND['mt']
    plain_rule(mt['mean_h'] > mt['mean_c'] and mt['p_one_sided'] <= mt['alpha'], 'the marked datasets gained more, p <= 0.05')
    # the controlled experiments
    matches = sum(bl['keyed'][('kendall_svc', x)]['mean_difference'] < 0 for x in COMBINED)   # the kernel strictly more accurate
    if matches != k17 - bl['ahead']['kendall_svc']:
        raise AssertionError('a dataset on which ArrowFlow and the Kendall kernel are exactly equal, which "more accurate" would miscount')
    majority(matches / k17, 'the fixed Kendall kernel is more accurate than ArrowFlow on most datasets')
    plain_rule(not de['holm']['depth2'] and de['better']['depth2'] > k17 / 2,
               '"a second hidden layer did not help": no significant depth contrast and one layer ahead on most datasets')

    # ------------------------------------------------------------------ Introduction
    plain_fact(I, f'it trails the best tuned classical model on {all17} tabular datasets we test (Table~\\ref{{tab:main}})', all17, fewer)
    plain_fact(I, f'Its average rank is {position} among the {words(len(mr))} tuned models, between {PLAIN[ahead_of]} and '
                  f'{PLAIN[behind]}, and {ORDINAL[further_position]} on the {words(k)} further datasets alone.',
               f'between {PLAIN[ahead_of]} and {PLAIN[behind]}', f'between {PLAIN[behind]} and {PLAIN[ahead_of]}')
    plain_fact(I, f'The trained network is more accurate than an untrained ArrowFlow on {higher} of the {k17} datasets, although the '
                  f'untrained version comes within one percentage point on {close} of the {k17}', f'on {higher} of', f'on {higher - 1} of')
    plain_fact(I, f'A support vector classifier with a fixed Kendall kernel, a standard similarity between rankings, is more accurate than '
                  f'ArrowFlow on {matches} of the {k17} datasets when both read rankings from the same encoder family.', f'on {matches} of',
               f'on {matches - 1} of')

    # ------------------------------------------------------------------ Section 7: opener, 7.1 and 7.2
    plain_fact(E, f'In brief, ArrowFlow trails the best tuned classical model on {all17} datasets, and training adds only a modest '
                  f'gain.', all17, fewer)
    # "the same encoder family" (referee-panel revision of 2026-09-25, item B15): the Kendall classifier chooses its own encoder settings
    plain_fact(E, 'Yet a fixed kernel on rankings from the same encoder family is more accurate on most datasets', 'on most datasets',
               'on every dataset')
    plain_fact(E, f'Its {FACTS["grid_size"]} candidates vary the hidden-layer widths, the learning rate and two encoder settings',
               f'Its {FACTS["grid_size"]} candidates', f'Its {FACTS["grid_size"] + 1} candidates')
    plain_fact(E, f'On every one of the {words(k17)} datasets the best tuned classical model has lower error, by ${gap[dmin]:.2f}$ '
                  f'({NAME[dmin]}) to ${gap[dmax]:.2f}$ percentage points ({NAME[dmax]}).', f'${gap[dmax]:.2f}$', f'${gap[dmax] + 0.01:.2f}$')
    plain_fact(E, f'Ranked within each dataset and averaged, ArrowFlow comes {position} of the {words(len(mr))} tuned models, between '
                  f'{PLAIN[ahead_of]} and {PLAIN[behind]}. On the {words(k)} further datasets alone it comes {ORDINAL[further_position]}, '
                  f'ahead of {PLAIN[further_order[-1]]} only (Section~S3.2).', f'comes {position} of', f'comes {ORDINAL[at]} of')

    # ------------------------------------------------------------------ 7.3
    plain_fact(E, f'Yet the untrained version comes within one percentage point on {close} of the {k17}.', f'on {close} of the',
               f'on {close + 1} of the')
    plain_fact(E, f'The gain is significant against at least one control on {words(len(order_sig))} datasets: {LABEL[vd]}, against tuned '
                  f'input footrule kNN, and {NAME[a2]} and {NAME[b2]}, against both controls.',
               f'on {words(len(order_sig))} datasets', f'on {words(len(order_sig) + 1)} datasets')
    plain_fact(E, f'Only the last two survive a stricter, post hoc correction over all {len(C["keys"])} comparisons',
               f'all {len(C["keys"])}', f'all {len(C["keys"]) + 1}')
    plain_fact(E, f'Training gained more on these {words(len(ND["marked"]))} than on the other {words(len(ND["others"]))} '
                  f'($p={pval(mt["p_one_sided"])}$).', f'$p={pval(mt["p_one_sided"])}$', f'$p={float(pval(mt["p_one_sided"])) + 0.001:.3f}$')
    plain_fact(E, f'It also rests on only {words(k)} datasets, and without the {words(len(flagged))} datasets that '
                  f'Table~\\ref{{tab:main}} flags it gives $p={flagged_p:.2f}$, a post hoc check (Section~S3.3).',
               f'only {words(k)}', f'only {words(k + 1)}')
    # the sorting result on the further datasets, first stated in the Discussion, now also in Section 7.3 (referee-panel revision of
    # 2026-09-25, item F6): tuned input footrule kNN on the encoded rankings minus numeric kNN on the projected scores, in error
    sortgap = {x: ND['E'][(x, INPUT_KNN)]['mean'] - ND['E'][(x, PROJECTED)]['mean'] for x in NEWDATA}
    raised = [x for x in NEWDATA if sortgap[x] > 0]
    plain_fact(E, f"On the {words(k)} further datasets, where the comparison is descriptive, sorting raised the readout's error on "
                  f"{words(len(raised))} of them, by up to {pct(max(sortgap.values()))} points.", f'on {words(len(raised))} of them',
               f'on {words(len(raised) + 1)} of them', f'by up to {pct(max(sortgap.values()))} points',
               f'by up to {pct(max(sortgap.values()) + 0.001)} points')

    # ------------------------------------------------------------------ 7.4
    proto = len(cf['more']['prototype_readout'])
    nearly_all(proto / k17, 'the prototype readout is less accurate than ArrowFlow on almost every dataset')
    by_more = sum(cf['gain'][(x, 'prototype_readout')] > 0.01 for x in COMBINED)
    plain_rule(max(cf['mean'], key=lambda v: cf['mean'][v]) == 'prototype_readout', 'the readout has the largest mean change')
    plain_fact(E, f'The readout makes the largest difference. Predicting with the prototype readout, the nearest output filter, instead of the '
                  f'nearest stored training examples is less accurate on {proto} of the {k17} datasets, by more than one percentage '
                  f'point on {words(by_more)} of them.', f'on {proto} of', f'on {proto - 1} of')
    views = len(cf['more']['views1'])
    plain_fact(E, f'Seven views are more accurate than the first, target-aware view alone on {views} of the {k17}.', f'on {views} of',
               f'on {views + 1} of')
    below(max(abs(cf['mean']['no_checkpoint']), abs(cf['mean']['no_augment'])), 0.002,
          'mean changes without the checkpoint and without augmentation, as fractions')
    plain_fact(E, 'Removing the checkpoint or augmentation changes mean accuracy by less than $0.2$ percentage points.', '$0.2$', '$0.1$')

    # ------------------------------------------------------------------ 7.5
    la, worse = mo['less_accurate'], mo['worse']
    plain_rule(all(t['p'] < 0.001 for t in mo['tests'].values()), 'both post hoc signed-rank tests give p < 0.001')
    # the test is the one-sided test of Table S-motion-scramble (signed_rank_less in render_motion), cited at the claim (items B26, G2)
    plain_fact(E, f'Delivering the motions to the wrong filters does so on {la["permuted_alignment"]} of the {k17} datasets, and '
                  f'shuffling their directions on {la["random_direction"]} of the {k17}. These counts are descriptive, but a post hoc '
                  f'one-sided signed-rank test across the datasets supports both ($p<0.001$; Table~S28).',
               f'on {la["permuted_alignment"]} of', f'on {la["permuted_alignment"] - 1} of')
    for arm in SCRAMBLED:
        majority(worse[arm] / k17, f'{arm} leaves neighborhoods less pure than the frozen arm on most datasets')
    plain_fact(E, "On most datasets, an example's nearest training examples share its class less often after either scramble than "
                  "with frozen filters", 'On most datasets', 'On every dataset')

    # ------------------------------------------------------------------ 7.6: what training teaches, on balance-scale
    b, rec = me['balanced'], me['recall']
    level = me['level_rows'] / me['all_rows']
    plain_fact(E, f'A level balance is rare, about one example in {words(one_in(level))}', f'in {words(one_in(level))}',
               f'in {words(one_in(level) - 1)}')
    fewer_than_in_100(max(b[UNTRAINED], b[INPUT_KNN], b[PROJECTED]), 5,
                      'level balances recognized by the untrained ArrowFlow and the two nearest-neighbor methods')
    plain_fact(E, 'Before training, ArrowFlow almost never recognizes a level balance, and neither do simpler nearest-neighbor methods '
                  'on the same inputs. None of them gets more than 5 in 100 right.', '5 in 100', '3 in 100')
    about_half(b[KNN], 'level balances recognized by ArrowFlow')
    nearly_all(min(rec[(m, c)] for m in (KNN, UNTRAINED) for c in ('L', 'R')), 'tips to either side, before and after training')
    plain_fact(E, 'After training, ArrowFlow gets about half of them right.', 'about half', 'about a third')
    plain_fact(E, 'The two common outcomes are easy either way. ArrowFlow recognizes tips to either side nearly always, before '
                  'training and after.', 'nearly always', 'about half the time')
    majority(me['gain'][UNTRAINED] / me['total'][UNTRAINED], 'share of the extra correct predictions in the balanced bucket')
    plain_fact(E, 'So most of what training adds on this dataset comes from learning this one thin class.', 'So most of', 'So all of')
    nearly_all(b['svc_rbf'], 'level balances recognized by the support vector classifier')
    far_more(b['svc_rbf'], b[KNN], 'the support vector classifier against ArrowFlow on the level balance')
    plain_fact(E, 'The support vector classifier recognizes nearly every level balance, far more often than ArrowFlow', 'nearly every',
               'about half of each')

    # ------------------------------------------------------------------ 7.6: do the filters move, and ties
    pc, disp, ties = dg['pooled_checkpoints'], dg['displacement'], dg['ties']
    plain_rule(int(pc['n_views_checkpoint_0']) == 0, 'no view returned its initial filters')
    one_layer = disp[(disp['widths'] == '[128]') & (disp['layer'] != 'output')]
    nearly_all(float(one_layer['changed_share_mean_later'].min()), 'hidden filters of the one-layer networks in a new order')
    plain_fact(E, f"In {num_word(int(pc['n_views_checkpoint_0']))} of the {int(pc['n_views']):,} trained views of one fitting seed did "
                  f"the checkpoint keep the initial filters, and in the networks with one hidden layer nearly every hidden filter ended in "
                  f"a new order",
               f"{int(pc['n_views']):,} trained", f"{int(pc['n_views']) + 1:,} trained")
    row32 = ties[(ties['widths'] == '[128]') & (ties['n'] == 32)].iloc[0]
    four = in_five(float(row32['tied_response_share_checkpoint']))
    plain_rule(in_five(float(row32['tied_response_share_initial'])) == four, 'the same share of ties before and after training')
    plain_fact(E, f'In a hidden layer of 128 filters that ranks 32 items, about {words(four)} in five distances are shared with another '
                  f'filter, before training and after', f'about {words(four)} in five', f'about {words(four - 1)} in five')

    # ------------------------------------------------------------------ 7.6: depth, voting rule, baselines
    plain_fact(E, f'With the earlier relay, one hidden layer was more accurate than two on {de["better"]["depth2"]} of the '
                  f'{k17} datasets, although no difference was significant after correction.', f'on {de["better"]["depth2"]} of',
               f'on {de["better"]["depth2"] - 1} of')
    if len(ag['holm']) != 1 or ag['keyed'][ag['holm'][0]]['mean_difference'] <= 0:
        raise AssertionError('the text says the Borda count is significantly more accurate on exactly one dataset')
    plain_fact(E, f'The Borda count was more accurate on {ag["better"]} of the {k17} datasets, but significantly so only on '
                  f'{NAME[ag["holm"][0]]}.', f'on {ag["better"]} of', f'on {ag["better"] - 1} of')
    plain_fact(E, f"On most datasets, yes. Take a support vector classifier with a fixed Kendall kernel "
                  f"\\citep{{jiao2015kernels,jiao2018kendall}}. It compares two rankings by Kendall's $\\tau$, the share of item pairs "
                  f"that they order alike minus the share that they order differently. Applied to rankings from ArrowFlow's own encoder, "
                  f"with its encoder settings chosen on its own inner folds, the classifier is more accurate than ArrowFlow on "
                  f"{matches} of the {k17} datasets.", f'on {matches} of', f'on {matches - 1} of')
    for m in ('lda_knn', 'pca_knn'):
        majority(bl['ahead'][m] / k17, f'ArrowFlow is more accurate than {BASELINE[m]} on most datasets')
    plain_rule(bl['ahead']['nca_knn'] <= k17 / 2, 'ArrowFlow does not beat NCA kNN on most datasets')
    plain_fact(E, f"ArrowFlow has a higher mean accuracy than nearest neighbors after a linear discriminant projection on "
                  f"{bl['ahead']['lda_knn']} of the {k17} datasets, and after a principal component projection on {bl['ahead']['pca_knn']}. "
                  f"After a learned metric (neighborhood components analysis), it has the higher mean accuracy on only "
                  f"{bl['ahead']['nca_knn']} of the {k17}", f"only {bl['ahead']['nca_knn']} of", f"only {bl['ahead']['nca_knn'] + 1} of",
               f"projection on {bl['ahead']['lda_knn']} of", f"projection on {bl['ahead']['lda_knn'] + 1} of")

    # ------------------------------------------------------------------ Discussion
    plain_fact(D, f'But ArrowFlow trails the best tuned classical model on {all17} small tabular datasets, and training '
                  f'its ranking layers adds only a modest gain. A fixed kernel on rankings from the same encoder family is also more '
                  f'accurate on most datasets.', all17, fewer)
    plain_fact(D, "Instead, a scrambled signal is worse than freezing the filters altogether, as a post hoc test across the datasets "
                  "supports. And on most datasets it leaves an example's nearest neighbors less likely to share its class.",
               'on most datasets', 'on every dataset')
    plain_fact(D, f'With the same readout, the untrained ArrowFlow comes within one percentage point of ArrowFlow on {close} of the '
                  f'{k17} datasets.', f'on {close} of', f'on {close + 1} of')
    plain_fact(D, 'On balance-scale, most of it comes from the rare class of level balances, which the support vector classifier recognizes '
                  'far more often', 'most of it', 'all of it')
    majority(ag['better'] / k17, 'the Borda count is more accurate than the prior-scaled median on most datasets')
    # the median's movement (referee-panel revision of 2026-09-25, item C12): "on some datasets it barely moved the filters" needs at
    # least two datasets on which the prior-scaled median changes fewer than a tenth of the filter orders per update; "on most others
    # it moved them but ended within 0.2 points of untrained filters" needs a strict majority of the remaining datasets among those on
    # which the median ends within 0.2 points of the untrained networks (the post hoc match of S3.7, item F11); "learned little" is
    # that match itself, on most datasets
    moved_share = {x: ag['movement']['median_mass_matched'][x]['changed_share'] for x in COMBINED}
    barely = [x for x in COMBINED if moved_share[x] < 0.1]
    plain_rule(len(barely) >= 2, '"on some datasets it barely moved the filters": at least two datasets below a tenth of the slots')
    near_untrained = [x for x in COMBINED if abs(cf['gain'][(x, 'untrained')] - float(ag['keyed'][x]['mean_difference'])) <= 0.002]
    majority(len(near_untrained) / k17, '"the median learned little": it ends within 0.2 points of the untrained networks on most datasets')
    moved = [x for x in COMBINED if x not in barely]
    majority(sum(x in near_untrained for x in moved) / len(moved),
             '"on most others it moved them but ended within 0.2 points of untrained filters"')
    plain_fact(D, f"Against a footrule median inside ArrowFlow's own layers, the Borda count had the higher mean accuracy on "
                  f'{ag["better"]} of the {k17} datasets, significantly on {words(len(ag["holm"]))}. Post hoc, the median learned little. '
                  f'On some datasets it barely moved the filters. On most others it moved them but ended within 0.2 points of untrained '
                  f'filters', f'on {ag["better"]} of', f'on {ag["better"] - 1} of', 'On most others', 'On all others')
    # the Boundaries sentence on the Kendall kernel left the Discussion (item F10): its opening paragraph, the Conclusion and Section
    # 7.6 state the same count
    plain_fact(D, 'A second hidden layer did not improve accuracy at a matched configuration', 'did not improve', 'improved')

    # ------------------------------------------------------------------ Conclusion
    plain_fact(K, f'A fixed Kendall kernel on rankings from the same encoder family is more accurate than ArrowFlow on {matches} of '
                  f'the {k17} datasets', f'on {matches} of', f'on {matches - 1} of')
    plain_fact(K, f'Its average rank, {ORDINAL[at + 1]} of {words(len(mr))}, lies between {PLAIN[ahead_of]} and {PLAIN[behind]}; on the '
                  f'{words(k)} further datasets alone it is {ORDINAL[further_position]}.', f'it is {ORDINAL[further_position]}',
               f'it is {ORDINAL[further_position - 1]}')
    plain_fact(K, 'A second hidden layer did not improve accuracy.', 'did not improve', 'improved')

    # ------------------------------------------------------------------ sentences added to the supplement
    purity, table = mo['purity'], mo['table']
    purer = sum(purity[('frozen', x)] < purity[('views7', x)] for x in COMBINED)
    plain_fact(S3, f"ArrowFlow's neighborhoods are purer than those of the frozen arm on {purer} of the {k17} datasets, and the "
                   f"permuted-alignment and random-direction arms leave them less pure than the frozen arm on "
                   f"{worse['permuted_alignment']} and {worse['random_direction']} of the {k17}.", f'on {purer} of', f'on {purer + 1} of')
    hcv = HCV
    plain_rule(all(table[(arm, hcv)]['holm_p_approximate'] >= 0.05 for arm in MOTION_ARMS), 'no motion arm is significant on HCV')
    unchanged([purity[(arm, hcv)] for arm in ['views7'] + MOTION_ARMS], 'purities of the four arms on HCV')
    plain_rule('no_learning' in C['flags'][hcv],
               'HCV is flagged no-learning: no model has lower mean error than the majority class')
    plain_fact(S3, f'No arm differs detectably on {NAME[hcv]}, where no model beats the majority class, and its purities are essentially '
                   f'unchanged', 'No arm differs detectably', 'One arm differs detectably')
    reading = me['reading']['mfeat_zernike']
    majority(reading['training_gain_untrained_minus_arrowflow'] / reading['representation_cost_untrained_minus_projected'],
             'share of the mfeat-zernike representation cost that training returns')
    plain_rule(reading['arrowflow_error'] > reading['projected_error'], 'on mfeat-zernike ArrowFlow does not reach the pre-sort projection')
    plain_fact(S3, 'training recovers most of what sorting and random filters cost, without reaching the accuracy of numeric kNN on the '
                   'projected scores before sorting', 'recovers most of', 'recovers all of')
    plain_rule(dg['relabel']['changed_share_mean'].idxmax() == hcv, 'relabeling changes the most predictions on HCV')
    plain_fact(S3, f'Relabeling changes the most seven-view predictions on {NAME[hcv]}', f'on {NAME[hcv]}', 'on Vehicle')
    move = de['movement']
    similar(move['depth2']['mean_changed_share'], move['depth1']['mean_changed_share'], 'movement of the two fully trained depths')
    plain_rule(move['depth2']['mean_changed_share'] / move['depth1']['mean_changed_share'] >= 0.9,
               'the two-layer networks change at least nine tenths as many slots as the one-layer networks')
    under_third(move['depth2_untrained_second']['mean_changed_share'] / move['depth2']['mean_changed_share'],
                'movement of the untrained-second arm against the fully trained two-layer arm')
    crippled = de['better']['depth2_untrained_second']
    move_of = de['per_dataset_movement']
    ratio = [move_of['depth2'][x]['changed_share'] / move_of['depth1'][x]['changed_share'] for x in COMBINED]
    plain_rule(min(ratio) < 0.9 and max(ratio) > 1, '"although the ratio varies by dataset": below nine tenths on one dataset and '
                                                    'above one on another')
    plain_fact(S3, f'Pooled over all folds, both fully trained depths reorder a similar share of their hidden filters per update, the '
                   f'two-layer networks at least nine tenths as many as the one-layer networks, although the ratio varies by dataset. One '
                   f'hidden layer is more accurate than the untrained-second arm on {crippled} of the {k17} datasets, but that arm '
                   f'reorders less than a third as many filters as the fully trained two-layer arm.', f'on {crippled} of',
               f'on {crippled - 1} of')
    return list(PLAIN_CLAIMS)


# ----------------------------------------------------------------------------- closed forms of the referee-panel revision (2026-09-25)
def footrule_distribution(V):
    """Pr(D = d) for d = 0, 1, ..., floor(V^2/2), where D is the footrule distance between a fixed ranking of V items and a
    uniformly random one (S1.7). The recursion runs over positions and counts permutations by total displacement: after
    position t, k positions wait for a later value and k values for a later position, and D adds 2k at the boundary after t.
    From k, position t + 1 and value t + 1 close two waiting pairs in k^2 ways, keep k in 2k + 1 ways and open a new pair in
    one way. footrule_checks compares it with enumeration and with the moments of Diaconis and Graham."""
    top = V * V // 2
    prob = np.zeros((V // 2 + 2, top + 1))
    prob[0, 0] = 1.0
    for t in range(1, V + 1):
        new = np.zeros_like(prob)
        for k in range(min(t - 1, V // 2) + 1):
            new[k] += (2 * k + 1) * prob[k]
            if k:
                new[k - 1] += k * k * prob[k]
            new[k + 1] += prob[k]
        new /= t
        if t < V:
            for k in range(1, new.shape[0]):
                if 2 * k > top:
                    if new[k].any():
                        raise AssertionError(f'footrule distribution: {k} waiting pairs cannot occur at V = {V}')
                    continue
                shifted = new[k, :top + 1 - 2 * k].copy()
                new[k, :] = 0
                new[k, 2 * k:] = shifted
        prob = new
    return prob[0]


def footrule_checks(values):
    """The recursion against enumeration for V <= 6, and its mean (V^2 - 1)/3 and variance (V + 1)(2V^2 + 7)/45 at every V."""
    from itertools import permutations
    for V in range(1, 7):
        counts = Counter(sum(abs(i - q[i]) for i in range(V)) for q in permutations(range(V)))
        p = footrule_distribution(V)
        if any(abs(p[d] - counts.get(d, 0) / math.factorial(V)) > 1e-12 for d in range(len(p))):
            raise AssertionError(f'footrule distribution: the recursion differs from enumeration at V = {V}')
    for V in values:
        p = footrule_distribution(V)
        d = np.arange(len(p))
        mean, var = float((p * d).sum()), float((p * d * d).sum() - (p * d).sum() ** 2)
        if abs(p.sum() - 1) > 1e-9 or abs(mean - (V * V - 1) / 3) > 1e-6 * V * V or \
                abs(var - (V + 1) * (2 * V * V + 7) / 45) > 1e-6 * V ** 3:
            raise AssertionError(f'footrule distribution: the moments differ from Diaconis and Graham at V = {V}')


def all_distinct(p, N):
    """Pr(N independent draws from the distribution p are pairwise distinct) = N! e_N(p), e_N the elementary symmetric
    polynomial of degree N."""
    e = np.zeros(N + 1)
    e[0] = 1.0
    for pk in p[p > 0]:
        e[1:] = e[1:] + pk * e[:-1]
    return float(math.factorial(N) * e[N])


def cover_orderings(e, d):
    """Cover (1967): the number of orderings of e points in general position in R^d by a linear functional, Q(1, d) = 1,
    Q(e, 1) = 2 for e >= 2 and Q(e + 1, d) = Q(e, d) + e Q(e, d - 1)."""
    q = {(1, k): 1 for k in range(1, d + 1)}
    for n in range(2, e + 1):
        q[(n, 1)] = 2
        for k in range(2, d + 1):
            q[(n, k)] = q[(n - 1, k)] + (n - 1) * q[(n - 1, k - 1)]
    return q[(e, d)]


def borda_update(prior, ballots, ids):
    """The weighted Borda update of S1.1 on items a, b, c: the position sums (from 1) of the prior, weight one, and of the
    ballots with their weights, in the order a, b, c, and the output ranking by increasing sum with ties by the identifier
    order ids."""
    items = sorted(prior)
    P = {v: Fraction(prior.index(v) + 1) for v in items}
    for ballot, w in ballots:
        for v in items:
            P[v] += Fraction(w) * (ballot.index(v) + 1)
    return tuple(P[v] for v in items), ''.join(sorted(items, key=lambda v: (P[v], ids.index(v))))


def tie_rule_witness(ids):
    """An IIA violation at V = 3 and total vote weight one decided by the tie rule: the prior abc, and two single ballots of
    weight one that order a pair alike, whose outputs order it differently, one of them through tied position sums."""
    from itertools import permutations
    orders = [''.join(q) for q in permutations('abc')]
    for b1 in orders:
        for b2 in orders:
            (s1, o1), (s2, o2) = borda_update('abc', [(b1, 1)], ids), borda_update('abc', [(b2, 1)], ids)
            for x, y in (('a', 'b'), ('a', 'c'), ('b', 'c')):
                i, j = 'abc'.index(x), 'abc'.index(y)
                if (b1.index(x) < b1.index(y)) == (b2.index(x) < b2.index(y)) and \
                        (o1.index(x) < o1.index(y)) != (o2.index(x) < o2.index(y)) and (s1[i] == s1[j] or s2[i] == s2[j]):
                    return x, y, b1, b2
    return None


def post_hoc_interval(values, level):
    """Mean and t interval across datasets at the confidence level in percent, the datasets as units (post hoc; n - 1 degrees
    of freedom)."""
    from scipy import stats as st
    n = len(values)
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = float(st.t.ppf(0.5 + level / 200, n - 1)) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def referee_claims(combined, newdata, controlled, components, referee, selected, further_selected, round_off, knn, training, projected,
                   sorting, knn_facts, kab, compute):
    """The numbers and plain phrases that the corrections after the simulated referee panel of 2026-09-23 added to the text, each
    rebuilt from the run data, the protocols or a closed form, with its mutation case (plain_fact). Runs after simplify_claims,
    whose group ranks and flagged-dataset test it reuses."""
    start = len(PLAIN_CLAIMS)
    I, E, D, K = INTRO, E7, 'sections/08_discussion.tex', 'sections/09_conclusion.tex'
    S1, S2, S3, S4 = ('supplement_sections/S1_proofs.tex', 'supplement_sections/S2_protocol.tex', SUPP3,
                      'supplement_sections/S4_ablations.tex')
    C, ND, cf, R = combined, newdata, components, referee
    mo, dg, ag = controlled['motion'], controlled['diagnostics'], controlled['aggregation']
    n, k, k17, n_outer = len(DATASETS), len(NEWDATA), len(COMBINED), FACTS['n_outer']
    fm = knn.protocol['full_method']

    # the post hoc signed-rank tests of scrambled minus frozen (item P), wherever the text leans on them
    tests = mo['tests']
    plain_rule(all(t['p'] < 0.001 for t in tests.values()), 'both post hoc signed-rank tests give p < 0.001')
    plain_fact(I, 'A post hoc test across the datasets, one defined after the results were known, supports both', 'supports both',
               'rejects both')
    plain_fact(K, 'Matched controls and a post hoc test across the datasets indicate that the returned motions carry information',
               'indicate', 'refute')
    plain_fact(S3, f'Post hoc, an exact one-sided signed-rank test \\citep{{demsar2006statistical}} across the {words(k17)} datasets '
                   f'supports both scrambles being less '
                   f'accurate than frozen filters ($p<0.001$ for each)', '$p<0.001$ for each', '$p<0.01$ for each')

    # the frozen arm is the untrained networks of the component ablation (item H1), and the two counts of Section 7.5 (item H2)
    frozen = {x: mo['table'][('frozen', x)] for x in COMBINED}
    if any(abs(float(frozen[x]['mean_difference']) - cf['gain'][(x, 'untrained')]) > 1e-12 or
           abs(float(frozen[x]['ci_low']) - cf['bounds'][(x, 'untrained')][0]) > 1e-12 or
           abs(float(frozen[x]['ci_high']) - cf['bounds'][(x, 'untrained')][1]) > 1e-12 for x in COMBINED):
        raise AssertionError('the text says the frozen arm equals the untrained networks of the component ablation on every dataset')
    # Section 7.5 glosses the control in place (referee-panel revision of 2026-09-25, item F5); the check above holds it
    plain_fact(E, "the hidden filters stay in their random starting orders, and everything else is ArrowFlow's own, settings included "
                  "(Section~S3.6)", 'settings included', 'settings retuned')
    plain_fact(S3, 'the frozen arm predicts exactly as the untrained networks of the component ablation', 'predicts exactly as',
               'predicts unlike')
    above_frozen = sum(float(frozen[x]['mean_difference']) > 0 for x in COMBINED)
    exceptions = [x for x in COMBINED if float(frozen[x]['mean_difference']) <= 0]
    mean_frozen = 100 * statistics.mean(float(frozen[x]['mean_difference']) for x in COMBINED)
    mean_untrained = 100 * statistics.mean(C['reg'][(x, UNTRAINED)]['diff'] for x in COMBINED)
    below(max(abs(float(frozen[x]['mean_difference'])) for x in exceptions) * 100, 0.2, 'the exceptions against frozen filters, points')
    plain_rule(all(abs(v - 1) <= 0.1 for v in (mean_frozen, mean_untrained)), '"about one point": both mean gains within 0.1 of 1')
    plain_fact(E, f'Against frozen filters ArrowFlow is more accurate on {above_frozen} of the {k17} datasets, fewer than the '
                  f'{len(C["higher"])} against the untrained ArrowFlow of Section~\\ref{{sec:training-controls}}, which chooses its own '
                  f'settings. The {words(len(exceptions))} exceptions lie within 0.2 points of zero, and the mean gain is about one point '
                  f'against either.', f'on {above_frozen} of', f'on {above_frozen + 1} of')

    # what each scrambling arm changed (items C1 and X1)
    moved = [float(r['changed_share']) for _, r in mo['stats'].iterrows() if r['arm'] == 'permuted_alignment']
    nearly_all(min(moved), 'selected slots that permuted alignment moves to another filter')
    for key in mo['reversed']:
        about_half(mo['attract'][key], f'attractions among the random-direction arm\'s nonzero votes, {key}')
        about_half(mo['reversed'][key], f'nonzero votes reversed, {key}')
        about_half(mo['repulsion'][key], f'repulsions among ArrowFlow\'s nonzero votes, {key}')
    plain_fact(S3, 'About half of the nonzero votes attract, so the random-direction arm reverses the direction of about half of them. '
                   'Permuted alignment instead moves almost every selected slot, zero votes included, to another filter.',
               'reverses the direction of about half', 'reverses the direction of every one')
    plain_fact(S3, "the share of ArrowFlow's votes that repel, which is about half on every dataset and layer", 'about half on every',
               'most on every')

    # Section 7.1: the three quality classes of Wine quality, the rescoring of finalists and the readout budget (the compute sentence
    # left the main text by the author's decision of 2026-09-23: "No compute cost needed anywhere")
    wq_manifest = read_json(knn.dir / 'wine_quality' / 'manifest.json')
    if len(wq_manifest['label_map']) != 3 or wq_manifest['source'] != 'OpenML data_id=40691':
        raise AssertionError('Section 7.1 says Wine quality is the red-wine OpenML copy with three quality classes')
    plain_fact(E, 'Wine quality (red wines, three quality classes)', 'three quality', 'four quality')
    # the class cut points, read from the label map of the dataset manifest (referee-panel revision of 2026-09-25, item F11)
    cuts = {'<=': ' or less', '==': '', '>=': ' or more'}
    wq_classes = []
    for label in wq_manifest['label_map']:
        m_label = re.fullmatch(r'quality(<=|==|>=)(\d+)', label)
        if not m_label:
            raise AssertionError(f'Section 7.1: the Wine quality class {label!r} is not a cut on the quality score')
        wq_classes.append(m_label.group(2) + cuts[m_label.group(1)])
    plain_fact(E, f"Wine quality's {words(len(wq_classes))} classes are the quality scores {wq_classes[0]}, {wq_classes[1]}, and "
                  f"{wq_classes[2]}.", f'scores {wq_classes[0]},', f'scores {wq_classes[1]},')

    # Section 7.2 and the Limitations: accuracy hides a rare class (item C7). The dataset named is the one on which ArrowFlow's
    # balanced accuracy lies nearest above the majority class's, among the datasets on which some model beats the majority class
    # (HCV is flagged: none does). "barely above" means above, by less than two points; the majority class scores exactly one half
    # on a two-class dataset
    CM = C['cm']
    ba = lambda x, m: float(CM.loc[(x, m), 'balanced_accuracy'])
    learnable = [x for x in COMBINED if 'no_learning' not in C['flags'][x]]
    rare = min(learnable, key=lambda x: ba(x, KNN) - ba(x, 'dummy'))
    n_classes = len(read_json(knn.dir / rare / 'manifest.json')['class_counts'] if rare in DATASETS else ND['pin'][rare]['class_counts'])
    plain_rule(0 < ba(rare, KNN) - ba(rare, 'dummy') < 0.02 and ba(rare, 'dummy') == 0.5 and n_classes == 2,
               '"barely above the 50 percent of the majority class": a two-class dataset, less than two points above one half')
    plain_rule(ba(rare, 'svc_rbf') - ba(rare, KNN) > 0.2, 'the support vector classifier finds the rare class far more often there')
    plain_fact(E, f"On {NAME[rare]}, ArrowFlow's balanced accuracy is {pct(ba(rare, KNN))} percent, barely above the "
                  f"{pct(ba(rare, 'dummy'), 0)} percent of the majority class. The support vector classifier reaches "
                  f"{pct(ba(rare, 'svc_rbf'))} percent (Section~S3.2).", f'is {pct(ba(rare, KNN))} percent',
               f'is {pct(ba(rare, KNN) + 0.001)} percent', f"reaches {pct(ba(rare, 'svc_rbf'))} percent",
               f"reaches {pct(ba(rare, 'svc_rbf') + 0.001)} percent")
    readout_weights = set(knn.protocol['knn_readout']['grid']['weights'])
    plain_rule(knn.protocol['selection_metric'] == 'accuracy' and readout_weights == {'uniform', 'distance'},
               'settings are chosen by accuracy, and the readout weights neighbors uniformly or by distance, never by class')
    plain_fact(D, 'Settings were chosen by accuracy, and the readout does not weight classes. So a rare class can be neglected '
                  '(Section~\\ref{sec:main-benchmark}).', 'does not weight classes', 'weights classes')
    plain_fact(E, f'each model picks its configuration there from at most {FACTS["budget"]} candidates. A stochastic model, one whose '
                  f'fit depends on a random seed, has its {words(knn.protocol["stochastic_finalists"])} best candidates rescored with all '
                  f'{words(FACTS["n_seeds"])} fitting seeds', f'has its {words(knn.protocol["stochastic_finalists"])} best',
               f'has its {words(knn.protocol["stochastic_finalists"] + 1)} best')
    if not re.search(r'training-pairing against arrowflow-v3-bridge-knn-1 passed', training.protocol['resource_decision']):
        raise AssertionError('the text says the benchmark training family was fixed after ArrowFlow\'s benchmark results existed')
    plain_fact(E, f"On the {words(n)} benchmark datasets, the training comparisons were fixed after ArrowFlow's own results there "
                  f"were known (Section~S2.5).", 'were fixed after', 'were fixed before')

    # Section 7.3: sorting itself (item B5)
    plain_rule(all(float(sorting['C'][(x, SORTING)]['holm_p_approximate']) >= 0.05 for x in DATASETS),
               'no sorting contrast is significant after Holm adjustment on the benchmark datasets')
    plain_fact(E, f'On the {words(n)} benchmark datasets, sorting the projected scores made no detectable difference to a '
                  f'nearest-neighbor readout (Section~S3.4)', 'made no detectable difference', 'made a clear difference')

    # Section 7.6: ties as with random filters (item F) and the mass-matched median (item A22)
    row32 = dg['ties'][(dg['ties']['widths'] == '[128]') & (dg['ties']['n'] == 32)].iloc[0]
    plain_rule(in_five(float(row32['random_tied_response_share'])) == in_five(float(row32['tied_response_share_checkpoint'])),
               'the same share of ties for random filters')
    plain_fact(E, 'before training and after, as with random filters (Section~S3.7)', 'as with random filters', 'unlike random filters')
    inert = ag['inertness']
    plain_rule(inert['median_mass_matched']['mean_changed_share'] < inert['borda']['mean_changed_share'],
               'over all datasets the mass-matched median moves the filters less often than Borda')
    rel = [ag['movement']['median_mass_matched'][x]['changed_share'] / ag['movement']['borda'][x]['changed_share'] for x in COMBINED]
    plain_rule(max(rel) / min(rel) >= 4, f'"differs widely": the per-dataset movement ratios span a factor of at least four, got '
                                         f'{max(rel) / min(rel):.1f}')
    plain_fact(E, f"Its prior weight, the weight of the filter's current order, was set once, on training data from "
                  f'{words(len(ag["pilots"]))} datasets, so that it moved the filters about '
                  f'as often as the Borda count there. Over all datasets it moves them less often, and per dataset the movement '
                  f'differs widely (Section~S3.7).', f'from {words(len(ag["pilots"]))} datasets',
               f'from {words(len(ag["pilots"]) + 1)} datasets')
    plain_fact(S3, f'Its prior counts as {words(ag["multiplier"])} ballots of the mean incoming vote weight, where Borda\'s prior has '
                   f'weight one. The multiple was chosen on a training-only ladder over {words(len(ag["pilots"]))} pilot datasets',
               f'as {words(ag["multiplier"])} ballots', f'as {words(ag["multiplier"] + 1)} ballots')

    # Section 6.3 and S1.7: the concentration of footrule distances (item F), closed forms and the random-filter share
    V, N = int(row32['n']), int(row32['n_filters'])
    sd = math.sqrt((V + 1) * (2 * V * V + 7) / 45)
    values = V * V // 4 + 1
    share = 1 - (1 - 1 / values) ** (N - 1)
    plain_fact('sections/06_theory.tex', f'with a standard deviation near ${math.sqrt(2 / 45):.2f}\\,V^{{3/2}}$', f'{math.sqrt(2 / 45):.2f}',
               '0.31')
    plain_fact(S1, f'At $V={V}$ the standard deviation is ${sd:.1f}$ against a range of ${V * V // 2}$. If the ${values}$ attainable '
                   f'values were equally likely, ${N}$ random filters would share a distance with another filter with probability '
                   f'$1-(1-1/{values})^{{{N - 1}}}\\approx{share:.2f}$. The concentration of the distances raises the share to the '
                   f'${float(row32["random_tied_response_share"]):.3f}$ that the diagnostics report for uniform random filters',
               f'${sd:.1f}$', f'${sd + 0.1:.1f}$')

    # Section 6.2 and S1.5: the rankings a view of degree one reaches (referee-panel revision of 2026-09-25, item D2), Cover's count
    # for the example of e = 16 coordinates and d = 4 features. The recursion must give the n(n - 1) orderings of points in the plane
    # and all n! orderings when d >= n
    e_ex, d_ex = 16, 4
    plain_rule(all(cover_orderings(m, 2) == m * (m - 1) for m in range(2, 12))
               and all(cover_orderings(m, m) == math.factorial(m) for m in range(1, 9)), "Cover's recursion on its checkable cases")
    reached = cover_orderings(e_ex, d_ex)
    mantissa, exponent = f'{math.factorial(e_ex):.1e}'.split('e')
    fact_ex = f'{e_ex}!\\approx{mantissa}\\times10^{{{int(exponent)}}}'
    plain_fact('sections/06_theory.tex', f'For $e={e_ex}$ and $d={d_ex}$ there are {reached:,}, against ${fact_ex}$ (Section~S1.5).',
               f'{reached:,}', f'{reached + 1:,}')
    plain_fact(S1, f'For $e={e_ex}$ and $d={d_ex}$ that is ${reached:,}$ rankings, against ${fact_ex}$.'.replace(f'{reached:,}',
               f'{reached:,}'.replace(',', '{,}')), f'{reached:,}'.replace(',', '{,}'), f'{reached + 1:,}'.replace(',', '{,}'))

    # Section 6.3, S1.7 and S3.7: how often random filters tie (items A6 and B25), from the exact distribution of the footrule
    # distance between an input and a uniformly random filter (footrule_distribution). rho_V is the chance that two random
    # filters tie at an input; a filter's response is shared with probability sum_k p_k (1 - (1 - p_k)^(N - 1)); N filters give
    # N sum_k (1 - (1 - p_k)^N) distinct responses on average, C(N, 2) rho_V tied pairs, and all N responses distinct with
    # probability N! e_N(p). "Reproduces" means within 0.005 of every "Tied random" share of Table S-diag-ties (a simulation);
    # "rare" and "almost no input" mean below 5 in 100 at every hidden-layer size, as "almost never"
    sizes = sorted((v, m) for v in C['declared_v'] for m in C['declared_n'])
    ties = dg['ties']
    plain_rule(sorted({(int(r['n']), int(r['n_filters'])) for _, r in ties.iterrows()}) == sizes,
               'the tie diagnostics cover every hidden-layer size of Table S1')
    footrule_checks(sorted(C['declared_v']))
    dist = {v: footrule_distribution(v) for v in C['declared_v']}
    rho = {v: float((dist[v] ** 2).sum()) for v in C['declared_v']}
    tied = {(v, m): float((dist[v] * (1 - (1 - dist[v]) ** (m - 1))).sum()) for v, m in sizes}
    distinct = {(v, m): float((1 - (1 - dist[v]) ** m).sum()) / m for v, m in sizes}
    pairs = {(v, m): math.comb(m, 2) * rho[v] for v, m in sizes}
    alone = {(v, m): all_distinct(dist[v], m) for v, m in sizes}
    plain_rule(max(abs(tied[(int(r['n']), int(r['n_filters']))] - float(r['random_tied_response_share'])) for _, r in ties.iterrows())
               < 0.005, 'the exact tied shares reproduce the simulated shares of uniform random filters')
    fewest = min(sizes, key=pairs.get)
    plain_rule(fewest == min(sizes, key=tied.get) and min(pairs[x] for x in sizes if x != fewest) > 10,
               'the size with the fewest expected tied pairs also has the fewest tied responses, and every other size more than 10')
    fewer_than_in_100(max(alone.values()), 5, 'inputs at which all responses of random filters are distinct, every hidden-layer size')
    sig2 = lambda x: f'{x:#.2g}'
    vs = sorted(C['declared_v'])
    plain_fact(S1, 'yields $\\rho_V\\approx' + '$, $'.join(sig2(rho[v]) for v in vs[:-1]) + f'$ and ${sig2(rho[vs[-1]])}$ at $V='
                   + '$, $'.join(str(v) for v in vs[:-1]) + f'$ and ${vs[-1]}$', sig2(rho[vs[2]]), f'{rho[vs[2]] + 0.0001:#.2g}')
    plain_fact(S1, 'It also reproduces the tied shares of uniform random filters in Table~\\ref{tab:s-diag-ties}.', 'reproduces',
               'contradicts')
    v32, n128 = 32, max(C['declared_n'])
    majority(min(tied[(v32, m)] for m in C['declared_n']), f'most responses of random filters are shared at V = {v32}')
    about_half(distinct[(v32, n128)], f'distinct responses of {n128} random filters at V = {v32}')
    plain_rule(round_half_up(1 / distinct[(v32, n128)]) == 2, '"about two filters" in a group of equal distances')
    plain_fact(S1, f'So at $V={v32}$ most responses are shared, but only about {100 * rho[v32]:.1f} percent of filter pairs tie. With '
                   f'${n128}$ filters the number of distinct responses is about half the number of filters, so a group of filters at one '
                   f'distance holds about {words(round_half_up(1 / distinct[(v32, n128)]))} filters on average.',
               f'about {100 * rho[v32]:.1f} percent', f'about {100 * rho[v32] + 0.1:.1f} percent',
               'about two filters', 'about three filters')
    plain_fact(S1, f'This is about ${pairs[fewest]:.1f}$ at $(V,N)=({fewest[0]},{fewest[1]})$, the size with the fewest ties, and more '
                   f'than $10$ at each of the other {words(len(sizes) - 1)} sizes. So an input with all responses distinct is rare',
               f'${pairs[fewest]:.1f}$', f'${pairs[fewest] + 0.1:.1f}$', 'is rare', 'is common')
    plain_fact('sections/06_theory.tex', 'At the hidden-layer sizes used here, random filters therefore leave almost no input with all '
                                         'distances distinct, and the guarantee then covers at most the nearest filter (Section~S1.7).',
               'almost no input', 'most inputs')
    # S3.7: the groups of tied filters in the diagnosed hidden layers at V = 32 with 128 filters, before and after training and for
    # random filters ("less than half", the rule "less than B"; "about two", the group size 1/share rounded)
    row_groups = ties[(ties['n'] == v32) & (ties['n_filters'] == n128)].iloc[0]
    shares = [float(row_groups[c]) for c in ('distinct_response_ratio_initial', 'distinct_response_ratio_checkpoint',
                                             'random_distinct_response_ratio')]
    for x in shares:
        below(x, 0.5, f'distinct responses at V = {v32} with {n128} filters')
    plain_rule({round_half_up(1 / x) for x in shares} == {2}, 'a group of equal distances holds about two filters, every column')
    plain_fact(S3, f'At $V={v32}$ with {n128} filters the number of distinct responses is less than half the number of filters, so a '
                   f'group holds about {words(round_half_up(1 / shares[1]))} filters on average. For uniform random filters only about '
                   f'{100 * rho[v32]:.1f} percent of filter pairs tie there', 'less than half', 'more than half',
               f'about {100 * rho[v32]:.1f} percent', f'about {100 * rho[v32] + 0.1:.1f} percent')

    # S1.1: the vote weights of ArrowFlow's layers against the Pareto threshold (item C)
    lr, batch = max(fm['candidate_grid']['learning_rate']), fm['fixed']['batch_size']
    v_min = min(C['declared_v']) - 1
    plain_rule(2 * lr * batch < v_min and min(C['declared_n']) - 1 >= v_min,
               'every vote below 2 eta_max, a batch below 2 eta_max B, and every layer with V - 1 at least the smallest e - 1')
    plain_fact(S1, f'each vote weighs less than $2\\eta\\le{2 * lr:g}$ and a batch has ${batch}$ examples, so '
                   f'$\\Omega<{2 * lr * batch:g}<{v_min}\\le V-1$', f'${batch}$ examples', f'${batch + 1}$ examples')

    # S1.1: the boundary case V = 3 with total vote weight one, where IIA fails through the tie rule (referee-panel revision of
    # 2026-09-25, item A5): the prior abc, which is also the identifier order, and one ballot of weight one, cba or cab; and for every
    # other identifier order a witness of the same kind, found by exhaustive search over single ballots
    s1_text = ' '.join((HERE / S1).read_text().split())
    plain_rule('Take the prior $abc$, which is also the identifier order, and one ballot of weight one. The ballots $cba$ and $cab$ '
               'both place $c$ before $b$.' in s1_text, 'S1.1 states the setting of the tie-rule witness as computed here')
    (sums1, out1), (sums2, out2) = borda_update('abc', [('cba', 1)], 'abc'), borda_update('abc', [('cab', 1)], 'abc')
    plain_rule(sums1[1] == sums1[2] and out1.index('b') < out1.index('c') and out2.index('c') < out2.index('b')
               and all(tie_rule_witness(ids) for ids in ('abc', 'acb', 'bac', 'bca', 'cab', 'cba')),
               'the tie rule decides the first profile, the second reverses b and c, and every identifier order has such a witness')
    tup3 = lambda v: '(' + ','.join(str(x) for x in v) + ')'
    plain_fact(S1, f'The first gives the position sums ${tup3(sums1)}$, and the tie rule keeps $b$ before $c$. The second gives '
                   f'${tup3(sums2)}$ and places $c$ before $b$.', tup3(sums1), tup3(sums2))
    plain_fact(S1, 'For every other identifier order, one of the three pairs of the prior gives a witness of the same kind.',
               'For every other', 'For no other')
    # S1.8: the pairwise majorities of the limit example (item D8), from the example's own distribution as the paragraph states it
    example = {'bac': Fraction(3, 5), 'acb': Fraction(2, 5)}
    plain_rule('Let independent rankings of $a,b,c$ equal $bac$ with probability $0.6$ and $acb$ with probability $0.4$.' in s1_text
               and sum(example.values()) == 1, 'S1.8 states the distribution of the limit example as used here')
    before = lambda x, y: sum(w for r, w in example.items() if r.index(x) < r.index(y))
    plain_rule(before('b', 'a') > Fraction(1, 2) and before('a', 'c') > Fraction(1, 2) and before('b', 'c') > Fraction(1, 2),
               'the pairwise majorities order the items b, a, c, the limit bac of the footrule median')
    plain_fact(S1, f"A ranking places $b$ before $a$ with probability ${float(before('b', 'a')):g}$, and it places $a$ and $b$ before $c$ "
                   f"with probabilities ${float(before('a', 'c')):g}$ and ${float(before('b', 'c')):g}$.",
               f"probability ${float(before('b', 'a')):g}$", f"probability ${float(before('b', 'a')) - 0.1:g}$")

    # S2.5: the interval rule over all outer folds and over a subset of them (item G)
    plain_fact(S2, f'the sample variance of the ${n_outer}$ differences is multiplied by $1/{n_outer}+{FACTS["ratio"]:g}$, where '
                   f'${FACTS["ratio"]:g}$ is the test-to-training ratio of the protocol. The $t$ quantile has ${FACTS["df"]}$ degrees '
                   f'of freedom. A contrast over a subset of $n$ outer folds uses $1/n+{FACTS["ratio"]:g}$ and $n-1$ degrees of freedom.',
               'and $n-1$ degrees of freedom.', f'and ${FACTS["df"]}$ degrees of freedom.')

    # S2.3: degree one (item Q)
    degree_one = {**selected['degree_one'], **further_selected['degree_one']}
    folds_one, sets_one = sum(degree_one.values()), sum(1 for x in COMBINED if degree_one.get(x))
    plain_fact(S2, f'ArrowFlow selected degree one on {folds_one} of the {k17 * n_outer} outer folds, on {sets_one} of the {k17} datasets',
               f'on {folds_one} of', f'on {folds_one + 1} of')

    # S2.4: round-off in the selection of the finalists (item A16)
    fr = further_selected['round_off']
    if round_off['choice'] or not fr['choice'] or round_off['n_finalists'] != fr['n_finalists']:
        raise AssertionError('the text says exact arithmetic picks the chosen configuration on every benchmark fold and not on every '
                             'further fold')
    plain_fact(S2, f"In ArrowFlow's runs, comparing the one-seed means in double precision changed which "
                   f"{words(round_off['n_finalists'])} candidates were rescored on {round_off['finalists']} of the "
                   f"{round_off['n_folds']} outer folds of the benchmark datasets and on {fr['finalists']} of the {fr['n_folds']} of "
                   f"the further datasets, each time by one candidate. Among the candidates that were rescored, exact arithmetic with the "
                   f"ID rule picks the configuration that was chosen on every benchmark fold and on all but {words(fr['choice'])} of the "
                   f"further folds.", f"on {fr['finalists']} of the {fr['n_folds']}", f"on {fr['finalists'] + 1} of the {fr['n_folds']}")

    # Table S1: the class cap of the discriminant coordinates binds everywhere (item R12)
    src = (REPO / 'experiments' / 'make_revision' / 'models.py').read_text()
    m_ratio = re.search(r"def __init__\(self, strategy='random', embed_dim=16, degree=1, lda_ratio=([\d.]+), seed=8129\)", src)
    if not m_ratio or 'n_lda = max(1, min(int(self.embed_dim * self.lda_ratio),' not in src:
        raise AssertionError('models.py: the discriminant share or its rule differs from Table S1')
    lam, binds = float(m_ratio.group(1)), []
    shapes = {**{x: read_json(knn.dir / x / 'manifest.json') for x in DATASETS},
              **{x: dict(shape=ND['pin'][x]['shape'], class_counts=ND['pin'][x]['class_counts']) for x in NEWDATA}}
    for x in COMBINED:
        d, classes = int(shapes[x]['shape'][1]), len(shapes[x]['class_counts'])
        base = fm['adaptive_encoding_defaults']['n_features<=10' if d <= 10 else 'n_features<=30' if d <= 30 else 'n_features>30']
        for scale in fm['candidate_grid']['embed_scale']:
            e = int(min(128, max(8, round(base['embed_dim'] * scale))))
            binds.append(int(e * lam) >= classes - 1 and d >= classes - 1 and e >= classes - 1)
    plain_rule(all(binds), 'the class cap C - 1 binds for every dataset and every candidate vocabulary size')
    plain_fact(S2, f'on all {words(k17)} datasets the class cap binds, so a target-aware view takes $C-1$ coordinates from the LDA',
               'the class cap binds', 'the share cap binds')

    # S3.2: the mean ranks split by group (item Q2)
    br, fr_rank = C['bench_rank'], C['further_rank']
    six = [KNN] + COMPARATORS
    bench_position = 1 + sum(br[m] < br[KNN] - 1e-12 for m in six)
    tied = [m for m in six if m != KNN and abs(br[m] - br[KNN]) < 1e-12]
    behind_f = sorted(m for m in six if fr_rank[m] < fr_rank[KNN])
    plain_rule(tied == ['random_forest'] and bench_position == 3, 'on the benchmark datasets ArrowFlow ties random forest for third')
    plain_rule(abs(fr_rank['random_forest'] - fr_rank['gradient_boosting']) < 1e-12, 'random forest and gradient boosting tie on the '
                                                                                     'further datasets')
    plain_fact(S3, f"Split by group, ArrowFlow's mean rank is {br[KNN]:.2f} on the {words(n)} benchmark datasets, tied with random forest "
                   f"for {ORDINAL[bench_position]}, and {fr_rank[KNN]:.2f} on the {words(k)} further datasets, "
                   f"{ORDINAL[1 + len(behind_f)]}, behind random forest and gradient boosting at {fr_rank['random_forest']:.2f} each and "
                   f"ahead of numeric kNN only.", f'{fr_rank[KNN]:.2f} on the', f'{fr_rank[KNN] + 0.01:.2f} on the')

    # S3.3: the stratum test without the flagged datasets (item J2)
    ft = C['flagged_test']
    plain_fact(S3, f"Omitting together the {words(len(ft['flagged']))} datasets that Table~\\ref{{tab:main}} flags, "
                   f"{listing([NAME[x] for x in ft['flagged']], 'and')}, leaves {ft['at_least']} of the {ft['splits']} splits at or "
                   f"above the observed difference, $p={ft['p']:.2f}$. This check is post hoc.", f"$p={ft['p']:.2f}$",
               f"$p={ft['p'] + 0.01:.2f}$")

    # S3.2: "many" exact duplicate rows on Wine quality and Segment (item A15; the sentence moved from Section 7.6 when the duplicate
    # material left the main text on 2026-09-23): at least a hundred rows each
    plain_rule(min(R['duplicate_rows'][R['wq']], R['duplicate_rows'][R['seg']]) >= 100, '"many" duplicate rows: at least 100 each')
    plain_fact(S3, f'{LABEL[R["wq"]]} and {LABEL[R["seg"]]} contain many, and the splits did not keep copies together', 'contain many',
               'contain a few')

    # S3.7: the checkpoint on the smallest datasets (item R06-7)
    size_of = {x: int(read_json(knn.dir / x / 'manifest.json')['shape'][0]) for x in DATASETS}
    size_of.update({x: int(ND['pin'][x]['shape'][0]) for x in NEWDATA})
    plain_rule(sorted(size_of, key=size_of.get)[:2] == ['iris', 'wine'], 'Iris and Wine are the two smallest datasets')
    per = dg['per']
    plain_rule(per.loc['iris', 'checkpoint_median'] < 10 and per.loc['wine', 'checkpoint_median'] < 10,
               'the checkpoint comes early on Iris and Wine')
    plain_fact(S3, f"its median is update {per.loc['iris', 'checkpoint_median']:.0f} on Iris and update "
                   f"{per.loc['wine', 'checkpoint_median']:.0f} on Wine, judged on {int(per.loc['iris', 'n_validation_samples_min'])} and "
                   f"{int(per.loc['wine', 'n_validation_samples_min'])} validation rows",
               f"update {per.loc['iris', 'checkpoint_median']:.0f} on Iris", f"update {per.loc['iris', 'checkpoint_median'] + 1:.0f} on Iris")

    # S4.1: where ArrowFlow's own configuration favors it (item A13) and the target-aware single view (item R07-5)
    tuned_gap = {x: float(knn_facts['contrasts'].loc[x, 'mean_difference']) for x in DATASETS}
    larger = sum(cf['gain'][(x, 'prototype_readout')] > tuned_gap[x] for x in DATASETS)
    match = max(abs(cf['mean'][v] - statistics.mean(C['reg'][(x, c)]['diff'] for x in COMBINED))
                for v, c in (('untrained', UNTRAINED), ('input_knn', INPUT_KNN)))
    bound = math.ceil(round(100 * match, 6) * 100) / 100
    plain_fact(S4, f'the gap is larger than against the separately tuned prototype readout on {words(larger)} of the {words(n)} benchmark '
                   f'datasets. For the untrained networks and footrule kNN, the mean gains match those of Table~\\ref{{tab:training}} '
                   f'within {bound:.2f} points.', f'on {words(larger)} of the', f'on {words(larger - 1)} of the')
    views1 = kab['protocol']['variant_definitions']['views1']
    if 'the first per-view kNN vote of views7' not in views1 or fm['view_strategy_cycle'][0] != 'target_aware':
        raise AssertionError('the text says the one-view variant is the first, target-aware view of the seven-view fit')
    plain_fact(S4, 'so the single view is always the target-aware one', 'the target-aware one', 'a random one')

    # S5.1 and S5.2 (referee-panel revision of 2026-09-25, items H6 and H10): the unreported eighth dataset of the matched protocol,
    # and the Iris copy, which the runs loaded from scikit-learn, as scikit-learn's own description of that copy states it
    S5 = 'supplement_sections/S5_reproducibility.tex'
    matched_protocol = read_json(PROTOCOLS_V3 / 'matched_v3.json')
    listed, primary = matched_protocol['datasets'], matched_protocol['primary_datasets']
    extra = [x for x in listed if x not in primary]
    plain_rule(sorted(primary) == sorted(DATASETS) and len(extra) == 1 and extra[0] not in COMBINED,
               'the matched protocol lists the seven benchmark datasets as primary and one more dataset that this paper does not use')
    plain_fact(S5, f'The protocol of the matched study lists an {ORDINAL[len(listed)]} dataset beside its {words(len(primary))} primary '
                   f'ones, and this paper does not report it either.', f'an {ORDINAL[len(listed)]} dataset',
               f'a {ORDINAL[len(listed) + 1] if len(listed) + 1 in ORDINAL else "ninth"} dataset')
    from sklearn.datasets import load_iris
    iris_note = re.search(r"The dataset is taken from Fisher's paper\. Note that it's the same as in R, but not as in the UCI Machine "
                          r"Learning Repository, which has (\w+) wrong data points\.", ' '.join(load_iris().DESCR.split()))
    plain_rule(iris_note is not None and read_json(knn.dir / 'iris' / 'manifest.json')['source'] == 'sklearn load_iris',
               "the runs loaded scikit-learn's Iris, which its description says follows Fisher's paper and differs from UCI's")
    plain_fact(S5, f"The Iris copy of scikit-learn is taken from Fisher's paper, and it differs from the copy in the UCI repository in "
                   f"{iris_note.group(1)} data points.", f'in {iris_note.group(1)} data points', 'in three data points')
    return PLAIN_CLAIMS[start:]


# ----------------------------------------------------------------------------- tests across the datasets (S3.3, Section 7.3)
SIGNED_RANK_ROWS = (  # key, first-named model, second-named model: every row is the first minus the second
    ('untrained', 'ArrowFlow', 'untrained ArrowFlow'),
    ('input', 'ArrowFlow', 'tuned input footrule kNN'),
    ('kendall_svc', 'Kendall SVC', 'ArrowFlow'),
    ('nca_knn', 'NCA kNN', 'ArrowFlow'),
    ('lda_knn', 'LDA kNN', 'ArrowFlow'),
    ('pca_knn', 'PCA kNN', 'ArrowFlow'),
    # the depth row (final round of 2026-09-23): two hidden layers with the corrected relay against one hidden layer, from the
    # depth study at one fitting seed; exact zeros leave its test, as in Wilcoxon's signed-rank test
    ('depth', 'Two hidden layers, corrected relay', 'one hidden layer'),
)


def signed_rank_greater(differences):
    """Exact one-sided Wilcoxon signed-rank test that the differences lie above zero: signed_rank_less of the negated
    differences, whose p value SciPy's exact test for the other direction must reproduce."""
    from scipy import stats
    result = signed_rank_less([-float(x) for x in differences])
    reference = stats.wilcoxon([float(x) for x in differences], alternative='greater', method='exact')
    if abs(float(reference.pvalue) - result['p']) > 1e-12:
        raise AssertionError('signed-rank test: the exact count for the alternative "above zero" differs from SciPy')
    return result


def render_signed_rank_tests(combined, baselines, knn, training, newdata, relay):
    """Post hoc Wilcoxon signed-rank tests with the datasets as units (S3.3 and Section 7.3; referee panel of 2026-09-23, item
    R1). The two training rows are recomputed from the per-fold accuracies of the runs in exact rational arithmetic and must equal
    the unrounded differences of the combined analysis; the baseline rows reverse the sign of the sealed differences of the
    neighbor baselines. Every p value is exact and one-sided, for the alternative that the first-named model is more accurate."""
    runs_of = {x: fam for fam in newdata['B'].values() for x in fam.protocol['datasets']}
    differences = {}
    for key, control in (('untrained', UNTRAINED), ('input', INPUT_KNN)):
        for x in COMBINED:
            mine = fold_accuracy(knn if x in DATASETS else runs_of[x], KNN, x)
            theirs = fold_accuracy(training if x in DATASETS else runs_of[x], control, x)
            if sorted(mine) != sorted(theirs):
                raise AssertionError(f'signed-rank tests: {x}: ArrowFlow and {control} were not scored on the same outer folds')
            exact = sum(mine[f] - theirs[f] for f in mine) / len(mine)
            if abs(float(exact) - combined['reg'][(x, control)]['diff']) > 1e-12:
                raise AssertionError(f'signed-rank tests: {x}: the per-fold difference from {control} is not the one of Table S-training')
            differences[(key, x)] = exact
    for key in BASELINE_MODELS:
        for x in COMBINED:
            differences[(key, x)] = -float(baselines['keyed'][(key, x)]['mean_difference'])
    for x in COMBINED:
        differences[('depth', x)] = -relay['diff']['depth1_minus_depth2_signed'][x]
    sets = (('all', COMBINED), ('further', NEWDATA))
    tests = {}
    for key, _, _ in SIGNED_RANK_ROWS:
        for name, members in sets:
            values = [differences[(key, x)] for x in members]
            # only the depth row, fitted at one seed, may hold an exact zero; it leaves the test, and the test needs no tie
            nonzero = [v for v in values if v != 0] if key == 'depth' else values
            if any(v == 0 for v in nonzero) or len({abs(v) for v in nonzero}) != len(nonzero):
                raise AssertionError(f'signed-rank tests: {key}/{name} has a zero or tied difference, for which the exact null differs')
            result = signed_rank_greater(nonzero)
            tests[(key, name)] = dict(result, positive=sum(v > 0 for v in values), n=len(values),
                                      zeros=len(values) - len(nonzero), mean=float(sum(values)) / len(values))
    head = ['Contrast'] + [two_lines(f'{label}:', column) for label in (f'All {words(len(COMBINED))}', f'The {words(len(NEWDATA))} further')
                           for column in ('positive', 'mean (pp)', '$p$')]
    rows = [[f'{first} $-$ {second}'] + [cell for name, _ in sets for cell in
                                          (f"{tests[(key, name)]['positive']} of {tests[(key, name)]['n']}",
                                           signed(tests[(key, name)]['mean'], 2), pval(tests[(key, name)]['p']))]
            for key, first, second in SIGNED_RANK_ROWS]
    t17 = tests[('untrained', 'all')]
    zeros = {name: tests[('depth', name)]['zeros'] for name, _ in sets}
    zero_sets = [name for name in zeros if zeros[name]]
    if zero_sets != ['all'] or zeros['all'] != 1:
        raise AssertionError('signed-rank tests: the caption says one dataset of the seventeen has an exactly zero depth difference')
    zero_name = [NAME[x] for x in COMBINED if differences[('depth', x)] == 0]
    zero_note = (f'Its test over all {words(len(COMBINED))} leaves out {zero_name[0]}, whose difference is exactly zero, as '
                 "Wilcoxon's test does. ")
    fact = (f"Over all {words(len(COMBINED))} datasets ArrowFlow is more accurate than the untrained ArrowFlow on {t17['positive']} "
            f"of them, one-sided $p={scientific(t17['p'])}$.")
    caption = ('\\textbf{Post hoc tests across the datasets.} For each contrast and set of datasets, the table gives the number of '
               'datasets on which the first-named model has the higher mean accuracy and the mean of the per-dataset accuracy '
               'differences in percentage points. It also gives the exact one-sided $p$ value of the Wilcoxon signed-rank test with '
               'the datasets as units \\citep{demsar2006statistical}, for the alternative that the first-named model is more '
               'accurate. A large $p$ gives no '
               f'support to that direction. Each per-dataset difference is a mean over the {FACTS["n_outer"]} outer folds. The two '
               'training rows are recomputed from the per-fold accuracies of the runs and equal the differences of '
               'Table~\\ref{tab:s-training}, and the four baseline rows reverse the sign of Table~\\ref{tab:s-baseline-families}. '
               'The depth row reverses the sign of the corrected-relay column of Table~\\ref{tab:s-relay-contrasts}, which comes '
               f'from one fitting seed per fold. {zero_note}The tests were '
               f'defined after every result was known and carry no multiplicity adjustment. {fact} LDA: linear discriminant '
               'analysis; PCA: principal component analysis; NCA: neighborhood components analysis; SVC: support vector classifier; '
               'kNN: nearest-neighbor classifier.')
    write_table('tab_s_signed_rank', caption, 'tab:s-signed-rank', head, rows, 'l' + 'r' * 6, size='\\scriptsize', wide=True)
    return dict(tests=tests, differences=differences, fact=fact,
                fact_mutation=(f"$p={scientific(t17['p'])}$", f"$p={scientific(2 * t17['p'])}$"))


def approved_edits_claims(combined, newdata, controlled, components, tests, knn):
    """The numbers and plain phrases of the edits the author approved on 2026-09-23 from the lean shortlist of the simulated
    referee panel: the post hoc signed-rank tests (item R1), the voting-rule and modest-gain diagnoses (items F11 and F13a), the
    fixed settings of the rule (F19), the design of the motion controls (R12, R13, F12), the duplicate paragraph of S3.2 that
    replaces the duplicate material, and the verbatim generative-AI disclosure. Each is rebuilt from the run data, the protocols
    or the code, with its mutation case (plain_fact)."""
    from scipy import stats as st
    start = len(PLAIN_CLAIMS)
    E, S2, S3 = E7, 'supplement_sections/S2_protocol.tex', SUPP3
    C, cf, k17, k = combined, components, len(COMBINED), len(NEWDATA)
    mo, dg, ag, dd = controlled['motion'], controlled['diagnostics'], controlled['aggregation'], controlled['dedup']
    T = tests['tests']

    # Section 7.3 and S3.3: the post hoc signed-rank tests across the datasets (item R1)
    t17, t10 = T[('untrained', 'all')], T[('untrained', 'further')]
    plain_rule(t17['p'] < 0.001 and t17['positive'] == len(C['higher']),
               'the signed-rank test over all seventeen datasets gives p < 0.001 and counts the datasets of Table 3')
    p10 = pval(t10['p'])
    # the gloss now describes a signed-rank test and says one-sided (referee-panel revision of 2026-09-25, items B12 and B26)
    plain_fact(E, f'ArrowFlow is more accurate than the untrained ArrowFlow on {t17["positive"]} of the {k17} datasets. A post hoc '
                  f'one-sided signed-rank test supports the gain over all {words(k17)} ($p<0.001$) and over the {words(k)} further '
                  f'datasets alone ($p={p10}$; Section~S3.3). This test ranks the differences across datasets by size and asks whether '
                  f'the positive ones outweigh the negative ones more than chance allows \\citep{{demsar2006statistical}}.',
               f'($p={p10}$;', f'($p={float(p10) + 0.001:.3f}$;')
    plain_rule(all(T[(key, s)]['p'] < 0.05 for key in ('untrained', 'input', 'kendall_svc') for s in ('all', 'further')),
               '"support": one-sided p below 0.05 in both sets for the two training rows and for the Kendall SVC')
    plain_rule(all(T[(key, s)]['p'] >= 0.05 for key in ('nca_knn', 'lda_knn', 'pca_knn') for s in ('all', 'further')),
               '"not detectably more accurate": one-sided p of at least 0.05 in both sets for the three nearest-neighbor baselines')
    plain_fact(S3, f'They support the gain over the untrained ArrowFlow in both sets (one-sided $p<0.001$ and $p={p10}$), the gain over '
                   f'tuned input footrule kNN and the advantage of the Kendall SVC over ArrowFlow.',
               f'and $p={p10}$)', f'and $p={float(p10) + 0.001:.3f}$)')
    plain_fact(S3, 'None of the three nearest-neighbor baselines is detectably more accurate than ArrowFlow.', 'None of the three',
               'One of the three')
    plain_fact('tables/tab_s_signed_rank.tex', tests['fact'], *tests['fact_mutation'])

    # S3.7: the voting-rule study reads mostly as a median that barely learned (item F11); post hoc and approximate, since the
    # aggregation family fits one seed and the component ablation three
    near = [x for x in COMBINED if abs(cf['gain'][(x, 'untrained')] - float(ag['keyed'][x]['mean_difference'])) <= 0.002]
    majority(len(near) / k17, 'the prior-scaled median ends within 0.2 points of the untrained networks on most datasets')
    plain_fact(S3, f'the prior-scaled median ends within 0.2 points of the untrained networks of Section~\\ref{{supp:ablation-arrowflow}} '
                   f'on {len(near)} of the {k17} datasets', f'on {len(near)} of the', f'on {len(near) + 1} of the')

    # S3.7: why the gain is modest (item F13a), post hoc and descriptive
    worse = [x for x in COMBINED if cf['gain'][(x, 'prototype_readout')] > cf['gain'][(x, 'untrained')]]   # higher error
    majority(len(worse) / k17, 'the prototype readout has higher error than the untrained networks on most datasets')
    plain_fact(S3, f'has higher error than even the untrained networks read by nearest neighbors on {len(worse)} of the {k17} datasets',
               f'on {len(worse)} of the', f'on {len(worse) + 1} of the')
    advantage = [cf['gain'][(x, 'untrained')] - cf['gain'][(x, 'prototype_readout')] for x in COMBINED]
    rho = float(st.spearmanr(advantage, [cf['gain'][(x, 'untrained')] for x in COMBINED]).statistic)
    plain_fact(S3, f'with a Spearman correlation of {rho:.2f} over the {k17} datasets', f'{rho:.2f}', f'{rho + 0.01:.2f}')
    share = mo['nonzero']
    top = {x: max(v for (y, _), v in share.items() if y == x) for x in COMBINED}
    low = {x: min(v for (y, _), v in share.items() if y == x) for x in COMBINED}
    starved = [x for x in COMBINED if x in sorted(COMBINED, key=top.get)[:4]]
    others = [x for x in COMBINED if x not in starved]
    plain_rule(max(top[x] for x in starved) < min(low[x] for x in others),
               'the four datasets with the fewest votes are separated from every other dataset at every layer')
    plain_fact(S3, f'on {listing([NAME[x] for x in starved], "and")} only {pct(min(low[x] for x in starved), 1)} to '
                   f'{pct(max(top[x] for x in starved), 1)} percent of the selected slots carry a vote, against at least '
                   f'{pct(min(low[x] for x in others), 1)} percent on every other dataset',
               f'to {pct(max(top[x] for x in starved), 1)} percent', f'to {pct(max(top[x] for x in starved) + 0.001, 1)} percent')
    per = dg['per']
    late = [x for x in COMBINED if x in sorted(COMBINED, key=lambda y: -per.loc[y, 'checkpoint_median'])[:5]]
    first, last = min(per.loc[x, 'checkpoint_median'] for x in late), max(per.loc[x, 'checkpoint_median'] for x in late)
    plain_rule(first > max(per.loc[x, 'checkpoint_median'] for x in COMBINED if x not in late),
               'the five datasets with the latest median checkpoint are separated from the others')
    plain_fact(S3, f'on {listing([NAME[x] for x in late], "and")} the median checkpoint is update {first:.0f} to {last:.0f} of '
                   f'{dg["updates"]}, so at least half of the views there still improved their validation error after update '
                   f'{first - 1:.0f}', f'update {first:.0f} to', f'update {first + 1:.0f} to')

    # S2.4: the rule's own settings are fixed and never varied (item F19)
    fm = knn.protocol['full_method']
    models_src = (REPO / 'experiments' / 'make_revision' / 'models.py').read_text()
    s2_text = (HERE / S2).read_text()
    table_rows_s1 = ['$\\rho$ & selected fraction per hidden layer & $0.5$', '$c$ & signal scale & $0.125$',
                     '$p_c$ & acceptance probability of a correct example & $0.01$', '$T$ & iterations (batch updates) & $200$',
                     '$B$ & batch size & $32$']
    plain_rule(bool(re.search(r'ratio_data_backprop=\.5,\s*motion_normalization_mult=\.125,\s*p_correct=\.01', models_src))
               and (fm['fixed']['iterations'], fm['fixed']['batch_size']) == (200, 32)
               and sorted(fm['candidate_grid']) == ['degree_offset', 'embed_scale', 'learning_rate', 'widths']
               and all(row in s2_text for row in table_rows_s1),
               'rho, c and p_c are the estimator defaults and T and B the protocol constants of Table S1, and none is a candidate')
    plain_fact(S2, 'The settings of the learning rule itself, $\\rho$, $c$, $p_c$, $T$ and $B$, were fixed during development and are '
                   'not varied in any experiment reported here but one.', 'are not varied', 'are varied')

    # S3.6: the design of the motion controls, from the frozen protocol and the instrumentation (items R12, R13 and F12)
    mp = read_json(PROTOCOLS_G5 / 'motion_controls.json')
    arm_src = (REPO / 'experiments' / 'make_revision' / 'motion_controls.py').read_text()
    plain_rule('drawn once per layer and batch' in mp['arm_definitions']['permuted_alignment'],
               'the permuted-alignment protocol draws a new permutation for every layer and batch')
    plain_fact(S3, 'The permutation is drawn anew for every layer and batch. A fixed permutation, the permutation analog of feedback '
                   'alignment \\citep{lillicrap2016random}, was not tested.', 'drawn anew for every layer and batch',
               'drawn once for the whole fit')
    plain_rule('PRESERVED in every arm' in mp['supervision_disclosure']
               and 'and so are the forward pass, the eligibility gate, the checkpoint and the readout' in arm_src,
               'every arm keeps the gate and the label-driven output update, and runs its own forward pass')
    plain_fact(S3, "Each arm applies the gate's rule to its own prototype readout, so its gate decisions are recomputed, not replayed.",
               'recomputed, not replayed', 'replayed, not recomputed')
    plain_fact(S3, 'Every arm keeps the class-filter target and the error gate', 'keeps the class-filter target',
               'replaces the class-filter target')

    # S3.2: the one duplicate paragraph (author, 2026-09-23). "Changes no conclusion" is checked on every conclusion that the
    # rerun can touch: the standing, the mean rank and its neighbors, the training effect and its test across the datasets,
    # the undetectable effect on Wine quality, and the cost of the prototype readout
    fam, errors, flag, proto, rise = dd['families'], dd['errors'], dd['flag'], dd['prototype'], dd['rise']
    rerun_of = {source: x for x, source in DEDUP_SOURCE.items()}
    plain_rule(all(float(flag[x]['arrowflow_error']) > float(flag[x]['best_classical_error']) for x in DEDUP_ORDER),
               'ArrowFlow still trails the best tuned classical model on both duplicate-free reruns')
    six = [KNN] + COMPARATORS
    mean_ranks = lambda err: dict(zip(six, np.vstack([st.rankdata([err(x, m) for m in six]) for x in COMBINED]).mean(axis=0)))
    before = mean_ranks(C['err'])
    after = mean_ranks(lambda x, m: errors[(rerun_of[x], m)] if x in rerun_of else C['err'](x, m))
    place = lambda ranks: (lambda order: (order.index(KNN), order[order.index(KNN) - 1], order[order.index(KNN) + 1]))(
        sorted(six, key=ranks.get))
    plain_rule(place(before) == place(after), 'with the rerun rows in place ArrowFlow keeps its mean-rank place and neighbors')
    wq_full, wq_rerun = C['reg'][('wine_quality', UNTRAINED)], fam[('primary', rerun_of['wine_quality'])]
    could_be_chance(wq_full['lo'], wq_full['hi'], 'the full-data training effect on Wine quality')
    could_be_chance(float(wq_rerun['ci_low']), float(wq_rerun['ci_high']), 'the duplicate-free training effect on Wine quality')
    effects = {x: float(fam[('primary', rerun_of[x])]['mean_difference']) if x in rerun_of else C['reg'][(x, UNTRAINED)]['diff']
               for x in COMBINED}
    plain_rule(C['reg'][('segment', UNTRAINED)]['diff'] > 0 and effects['segment'] > 0,
               "Segment's training effect is positive with and without the duplicates")
    majority(sum(v > 0 for v in effects.values()) / k17, 'training helps on most datasets with the rerun effects in place')
    plain_rule(signed_rank_greater([effects[x] for x in COMBINED])['p'] < 0.001,
               'the signed-rank test with the rerun effects in place still gives p < 0.001')
    plain_rule(all(float(proto[x]['arrowflow_minus_variant']) > 0 and float(proto[x]['ci_low']) > 0 for x in DEDUP_ORDER),
               'the prototype readout stays clearly less accurate than ArrowFlow on both reruns')
    plain_fact(S3, 'A rerun of both datasets without the duplicates, under its own frozen protocol (Section~\\ref{supp:protocols}), '
                   'changes no conclusion of this paper.', 'changes no conclusion', 'changes one conclusion')
    rises = {m: v for m, v in rise[rerun_of['wine_quality']].items() if m != 'dummy'}
    largest = sorted(rises, key=rises.get, reverse=True)[:4]
    plain_rule(largest[0] == KNN and set(largest) <= {KNN, UNTRAINED, INPUT_KNN, PROJECTED, 'numeric_knn'},
               'the four largest error rises without duplicates are all nearest-neighbor models, ArrowFlow first')
    plain_fact(S3, "on Wine quality, removing them raises ArrowFlow's error the most, and the four largest rises all belong to "
                   "nearest-neighbor models", 'the four largest', 'the five largest')

    # the generative-AI disclosure of the back matter, verbatim as the author approved it on 2026-09-23 (British "analysing"
    # included): an edit that normalizes it must fail
    plain_fact('main.tex', 'During the preparation of this work the author used Claude Opus 5, Claude Opus 5.5, Claude Fable 5 and '
                           'Claude Fable 5.1 (Anthropic), and GPT 5.6 Sol and GPT 6 Astra (OpenAI) for software development, running '
                           'and analysing the experiments, and editing the manuscript. The author reviewed all output and takes full '
                           'responsibility for the content of the publication.', 'analysing', 'analyzing')
    return PLAIN_CLAIMS[start:]


def worked_relay():
    """The relay of the four-item worked example of Section S2.2, recomputed from its filters: the hidden input
    pi = [1,3,2,4], the filters h1 = [1,2,3,4] and h3 = [2,1,4,3] selected with the signs of the output layer's returned
    motion (h1 negative, h3 positive). Returns the corrected relayed motions by item, their mean, the earlier relay's motion
    of h1 toward the reversed input, its mean with h3, and the inner product of h1's two motions."""
    pi, h1, h3 = [1, 3, 2, 4], [1, 2, 3, 4], [2, 1, 4, 3]
    o1, hidden_out = ['h3', 'h2', 'h1'], ['h1', 'h2', 'h3']          # the class-1 filter and the hidden output ranking
    returned = {h: hidden_out.index(h) - o1.index(h) for h in o1}      # m_{o1 -> pi^(1)} keyed by hidden filter
    if returned != {'h1': -2, 'h2': 0, 'h3': 2}:
        raise AssertionError('worked example: the output layer no longer returns (h1: -2, h2: 0, h3: +2)')
    sign = {'h1': -1, 'h3': 1}
    by_item = lambda r, tau: {v: tau.index(v) - r.index(v) for v in r}   # m_{r -> tau} keyed by item (0-based positions)
    corrected = {h: {v: sign[h] * m for v, m in by_item(r, pi).items()} for h, r in (('h1', h1), ('h3', h3))}
    earlier_h1 = by_item(h1, pi[::-1])
    items = [1, 2, 3, 4]
    mean = [Fraction(corrected['h1'][v] + corrected['h3'][v], 2) for v in items]
    earlier_mean = [Fraction(earlier_h1[v] + corrected['h3'][v], 2) for v in items]
    inner = sum(earlier_h1[v] * corrected['h1'][v] for v in items)
    return dict(h1=[corrected['h1'][v] for v in items], h3=[corrected['h3'][v] for v in items], mean=mean,
                earlier_h1=[earlier_h1[v] for v in items], earlier_mean=earlier_mean, inner=inner,
                plain_h1=[-corrected['h1'][v] for v in items])


def final_round_claims(combined, newdata, controlled, components, tests, knn):
    """The numbers and plain phrases of the final writing round of 2026-09-23: the corrected relay of Section 3.5 with its
    disclosure, the restructured abstract, the depth results with both relays and the representation test (Sections 7.6,
    S3.3 and S3.7), the contribution statements of the Introduction, the Discussion and the Conclusion, the relay of the
    worked example (S2.2), the one varied setting of S2.4, the new experiment counts and the named tools of the disclosure.
    Each is rebuilt from the run data, the frozen protocols or the worked example, with its mutation case (plain_fact)."""
    start = len(PLAIN_CLAIMS)
    I, E, D, K = INTRO, E7, 'sections/08_discussion.tex', 'sections/09_conclusion.tex'
    L3, S2, S3, S5 = ('sections/03_ranking_layer.tex', 'supplement_sections/S2_protocol.tex', SUPP3,
                      'supplement_sections/S5_reproducibility.tex')
    C, k17, k, n = combined, len(COMBINED), len(NEWDATA), len(DATASETS)
    de, rl, rp, mo = controlled['depth'], controlled['relay'], controlled['representation'], controlled['motion']
    c1, c2, c3 = RELAY_CONTRASTS
    prim, sec = rp['primary'], rp['secondary']

    # the rules every plain statement below leans on
    # "no better than untrained ones": the prespecified primary reading is not met, no primary contrast is a Holm-significant
    # gain, the trained rankings are ahead on at most half of the further datasets and their mean difference is not positive
    no_better = not prim['met'] and not prim['gains'] and len(prim['higher']) <= k / 2 and prim['mean'] <= 0
    plain_rule(no_better, '"no better than untrained ones" under the Kendall-kernel readout')
    # "a second hidden layer did not help": the prespecified outcome with the corrected relay, and with the earlier relay one
    # hidden layer ahead on most datasets and no Holm-significant contrast in either family
    plain_rule(rl['outcome'] == 'does_not_help' and not de['holm']['depth2'] and de['better']['depth2'] > k17 / 2,
               '"a second hidden layer did not help" with either relay')
    # "about as accurate as one": no Holm-significant contrast either way and a mean difference below half a point
    plain_rule(not rl['holm'][c1] and abs(rl['mean'][c1]) < 0.005, '"about as accurate as one": |mean| < 0.5 points, none significant')
    # "improves the neighborhoods of the nearest-neighbor readout": the kNN readout of the trained rankings is more accurate than
    # that of the untrained rankings on most datasets
    knn_ahead = sum(float(rp['near'][('knn_untrained', x)]['mean_difference']) > 0 for x in COMBINED)
    majority(knn_ahead / k17, 'the nearest-neighbor readout of the trained rankings beats that of the untrained rankings')
    statement = prim['statement'].replace('neighbour', 'neighbor')
    plain_rule(statement == 'training improves the nearest-neighbor neighborhoods but not the representation read by a strong fixed '
                            'readout', 'the fixed statement of the representation protocol, in American spelling')

    # ------------------------------------------------------------------ the abstract (restructured on 2026-09-23, item S1)
    # the dataset clause qualifies the ranking-layer result since the referee-panel revision of 2026-09-25 (items B4, F1); the
    # modest-gain clause is guarded on its own in learned_encoder_claims
    plain_fact('main.tex', f'Under nested cross-validation on {words(k17)} datasets, {words(n)} used during development and {words(k)} '
                           f'chosen by prespecified rules, training the ranking layers helps modestly.', f'{words(n)} used',
               f'{words(n + 1)} used')
    # the abstract's training gains of the ranking layers moved to Section 7.3 when the learned encoder took their place (the
    # author's ruling of 2026-09-25); the learned-encoder numbers of the abstract are guarded in learned_encoder_claims
    for arm in SCRAMBLED:
        majority(mo['less_accurate'][arm] / k17, f'{arm} less accurate than frozen filters on most datasets')
    plain_fact('main.tex', 'leave the network less accurate on most datasets than never moving its hidden filters', 'on most datasets',
               'on every dataset')
    plain_fact('main.tex', 'But read by a Kendall-kernel support vector classifier, trained hidden rankings are no better than untrained '
                           'ones, and a second hidden layer did not help.', 'are no better than', 'are better than')
    plain_fact('main.tex', "Counting the filter's current order as one more voter, the update rule violates independence of irrelevant "
                           "alternatives by Arrow's theorem under a mild weight condition. So a filter's order of two items can change when a "
                           "third moves.", 'a mild weight condition', 'any weight condition')

    # ------------------------------------------------------------------ Section 3.5: the relay and its disclosure
    relay_protocol = rl['protocol']
    plain_rule('a network with one hidden layer is unchanged by construction' in relay_protocol['relay']['scope']
               and 'sign(a_j) m(r_j -> pi)' in relay_protocol['relay']['corrected'],
               'only networks with two hidden layers relay, and the corrected relay passes sign(a_j) m(r_j -> pi)')
    folds17 = k17 * FACTS['n_outer']
    plain_fact(L3, f"({de['two_layer']} of ArrowFlow's {folds17} outer folds)",
               f"({de['two_layer']} of", f"({de['two_layer'] + 1} of")

    # ------------------------------------------------------------------ Introduction, item 3
    # the null is scoped to the fixed statement (referee-panel revision of 2026-09-25, item C5)
    plain_fact(I, 'Read by that classifier, trained hidden rankings are no better than untrained ones. So training improves the '
                  'neighborhoods of the nearest-neighbor readout, but the representation read by a strong fixed readout shows no gain.',
               'are no better than', 'are better than')
    plain_fact(I, 'A second hidden layer did not help (Section~\\ref{sec:controlled}).', 'did not help', 'helped')

    # ------------------------------------------------------------------ Section 7: the disclosure, depth and the representation
    plain_fact(E, 'The tools were Claude Opus 5, Claude Opus 5.5, Claude Fable 5 and Claude Fable 5.1 (Anthropic), and GPT 5.6 Sol '
                  'and GPT 6 Astra (OpenAI).', 'GPT 6 Astra', 'GPT 6')
    plain_fact(E, 'With the corrected relay, two layers were about as accurate as one, and a second layer still did not help '
                  '(Section~S3.7).', 'about as accurate as one', 'more accurate than one')
    # the depth comparison's settings and its cross-dataset interval (referee-panel revision of 2026-09-25, items C10 and C5): the
    # widths of both depth protocols, the settings they keep at the fold's own selection, the folds whose selection has one hidden
    # layer, and a post hoc t interval across the seventeen datasets for two layers with the corrected relay minus one
    depth_protocols = [read_json(PROTOCOLS_G5 / 'depth.json'), rl['protocol']]
    widths = [x['arm_widths'] for x in depth_protocols]
    plain_rule(widths[0] == widths[1] and len(widths[0]['depth1']) == 1 and len(widths[0]['depth2']) == 2
               and all('the learning rate, the iteration count' in x['not_retuned'] for x in depth_protocols),
               'both depth families compare one hidden layer with two and keep the learning rate and the iteration count of each fold')
    w1, w2 = widths[0]['depth1'], widths[0]['depth2']
    plain_fact(E, f'We refitted every split with one hidden layer of {w1[0]} filters and with two, of {w2[0]} and then {w2[1]} filters.',
               f'of {w2[0]} and then', f'of {w2[1]} and then')
    updates = knn.protocol['full_method']['fixed']['iterations']
    plain_fact(E, f'Every other setting, including the {updates} updates and the learning rate, was the one chosen for that split.',
               f'the {updates} updates', f'the {2 * updates} updates')
    one_layer = folds17 - de['two_layer']
    plain_fact(E, f'On {one_layer} of the {folds17} splits, that choice was made for a network with one hidden layer.',
               f'On {one_layer} of', f'On {one_layer + 1} of')
    two_minus_one, lo21, hi21 = post_hoc_interval([-100 * rl['diff'][c1][x] for x in COMBINED], FACTS['confidence'])
    could_be_chance(lo21, hi21, 'two hidden layers with the corrected relay minus one, across the datasets')
    plain_fact(E, f'Over the datasets, two layers minus one averaged ${two_minus_one:+.2f}$ points, with a post hoc '
                  f'{FACTS["confidence"]}\\% interval from ${lo21:+.2f}$ to ${hi21:+.2f}$.', f'${two_minus_one:+.2f}$',
               f'${two_minus_one + 0.01:+.2f}$')
    plain_fact(E, f'On the {words(k)} further datasets, the scope fixed in advance, the trained rankings had the higher mean accuracy on '
                  f'only {len(prim["higher"])}, and no difference was significant.', f'on only {len(prim["higher"])}',
               f'on only {len(prim["higher"]) + 1}')
    rep_diffs = [100 * float(prim['keyed'][x]['mean_difference']) for x in NEWDATA]
    rep_mean, lo_rep, hi_rep = post_hoc_interval(rep_diffs, FACTS['confidence'])
    plain_rule(abs(rep_mean - 100 * prim['mean']) < 1e-9, 'the interval is centred on the mean difference of the primary family')
    could_be_chance(lo_rep, hi_rep, 'trained minus untrained rankings under the Kendall-kernel readout, across the further datasets')
    plain_fact(E, f'Over the {words(k)} datasets, trained minus untrained averaged ${rep_mean:+.2f}$ points, with a post hoc '
                  f'{FACTS["confidence"]}\\% interval from ${lo_rep:+.2f}$ to ${hi_rep:+.2f}$.', f'${rep_mean:+.2f}$',
               f'${rep_mean + 0.01:+.2f}$')
    plain_fact(E, 'For this outcome, the protocol had fixed the following statement before the test. Training improves the '
                  'nearest-neighbor neighborhoods but not the representation read by a strong fixed readout (Section~S3.7).',
               'but not the representation', 'and the representation')

    # ------------------------------------------------------------------ Discussion and Conclusion
    plain_fact(D, 'Read by a strong fixed readout, a Kendall-kernel classifier, the trained hidden rankings are no better than the '
                  'untrained ones (Section~\\ref{sec:controlled}).', 'are no better than', 'are better than')
    plain_fact(D, 'So training improves the neighborhoods that the nearest-neighbor readout uses. It does not improve the representation '
                  'that a strong readout sees.', 'It does not improve', 'It also improves')
    plain_fact(D, 'with the earlier relay or with the corrected one. The rule passes supervision through several ranking layers, but '
                  'the second layer brought no measured benefit.', 'no measured benefit', 'a measured benefit')
    plain_fact(K, 'Read with that kernel, the trained hidden rankings are no better than untrained ones. So training improves the '
                  'neighborhoods of the nearest-neighbor readout, but the representation read by a strong fixed readout shows no gain.',
               'are no better than', 'are better than')

    # ------------------------------------------------------------------ S2.2: the relay of the worked example
    wr = worked_relay()
    tup = lambda v: '(' + ','.join(str(x) for x in v) + ')'
    half = lambda f: ('-' if f < 0 else '') + '\\tfrac12' if abs(f) == Fraction(1, 2) else str(f)
    plain_rule(wr['inner'] == 0, 'the earlier relay of h1 is orthogonal to its corrected relay')
    plain_rule([v for v in range(4) if wr['h1'][v] > 0] == [2] and [v for v in range(4) if wr['h1'][v] < 0] == [1]
               and [v for v in range(4) if wr['h3'][v] > 0] == [1, 3] and [v for v in range(4) if wr['h3'][v] < 0] == [0, 2],
               'h1 asks item 3 earlier and item 2 later; h3 asks items 2 and 4 earlier and items 1 and 3 later')
    plain_fact(S2, f'For the repelled $h_1$ this is $-{tup(wr["plain_h1"])}={tup(wr["h1"])}$ for the items $1,2,3,4$',
               f'={tup(wr["h1"])}$', f'={tup(wr["plain_h1"])}$')
    plain_fact(S2, f'For the attracted $h_3$ it is ${tup(wr["h3"])}$', tup(wr['h3']), tup([-x for x in wr['h3']]))
    plain_fact(S2, 'Their unweighted mean is $u=(' + ','.join(half(f) for f in wr['mean']) + ')$ with $\\max_v|u_v|=\\tfrac12$',
               '\\max_v|u_v|=\\tfrac12', '\\max_v|u_v|=1')
    c_scale = float(relay_protocol['relay']['unchanged'].split('eq. (6) with c = ', 1)[1].split(',', 1)[0])
    amp = c_scale * 4                                                     # c V_l with the example's V_l = 4
    peak = max(abs(f) for f in wr['mean'])
    scaled = [amp * float(f) / (float(peak) + 1e-5) for f in wr['mean']]
    plain_rule(c_scale == 0.125 and peak == Fraction(1, 2) and all(abs(abs(v) - 0.5) < 1e-4 for v in scaled),
               'the worked signal: c = 0.125, V = 4 and max |u| = 1/2 give entries of about 0.5 in magnitude')
    plain_fact(S2, f'the signal is $\\tilde u={amp:g}\\,u/(\\tfrac12+10^{{-5}})\\approx('
                   + ','.join(f'{v:.1f}' for v in scaled) + ')$', f'{amp:g}\\,u', f'{amp * 2:g}\\,u')
    plain_fact(S2, f"passed down $h_1$'s motion toward the reversed input, ${tup(wr['earlier_h1'])}$, instead",
               tup(wr['earlier_h1']), tup(wr['h1']))
    plain_fact(S2, "The earlier relay's mean was $u=(" + ','.join(str(f) for f in wr['earlier_mean']) + ')$.',
               '(' + ','.join(str(f) for f in wr['earlier_mean']) + ')', '(' + ','.join(half(f) for f in wr['mean']) + ')')

    # ------------------------------------------------------------------ S2.4: the one arm that varies a setting of the rule
    plain_rule('multiplied by s before it is accumulated' in relay_protocol['vote_scale']['definition']
               and 'the first hidden layer' in relay_protocol['vote_scale']['definition']
               and 'eq. (6) with c = 0.125' in relay_protocol['relay']['unchanged']
               and 'only the hidden widths, the relay and the first layer\'s vote scale differ' in relay_protocol['arm_definitions']['note']
               and 'are not re-tuned for any arm' in relay_protocol['not_retuned']
               and 'at the per-fold selection sealed by the reference ablation run' in rp['protocol']['selection_statement'],
               'the scaled arm multiplies the first layer\'s votes, which c alone scales in a two-layer network, and nothing else of '
               'the rule varies in the two families of 2026-09-23')
    plain_fact(S2, f'makes the votes of the first hidden layer {words(rl["scale"])} times larger, as an {words(rl["scale"])}fold $c$ would '
                   f'in a network with two hidden layers', f'{words(rl["scale"])} times larger', f'{words(rl["scale"] // 2)} times larger')

    # ------------------------------------------------------------------ S3: counts, the depth row, the two families
    families = ['motion', 'diagnostics', 'mechanism', 'depth', 'relay', 'aggregation', 'baselines', 'representation']
    plain_rule(all(controlled.get(f) is not None for f in families), 'eight controlled experiments were rendered')
    plain_fact(S3, f'and the {words(len(families))} controlled experiments of Sections', f'the {words(len(families))} controlled',
               f'the {words(len(families) - 1)} controlled')
    plain_fact(S3, f'the complete numerical record of the {words(len(families))} controlled experiments summarized in',
               f'the {words(len(families))} controlled', f'the {words(len(families) - 2)} controlled')
    plain_fact(S3, f'apply to the {words(len(families) - 1)} studies below', f'the {words(len(families) - 1)} studies',
               f'the {words(len(families) - 3)} studies')
    frozen = sorted({rl['protocol']['frozen_at_utc'][:10], rp['protocol']['frozen_at_utc'][:10]})
    earlier = [read_json(PROTOCOLS_G5 / f)['frozen_at_utc'] for f in ('motion_controls.json', 'training_diagnostics.json', 'depth.json',
                                                                      'aggregation.json', 'dedup.json', 'neighbour_baselines.json')]
    plain_rule(len(frozen) == 1 and all(min(rl['protocol']['frozen_at_utc'], rp['protocol']['frozen_at_utc']) > e for e in earlier)
               and min(rl['protocol']['frozen_at_utc'], rp['protocol']['frozen_at_utc'])[:10] > C['ready']['written_utc'][:10],
               'both protocols of the final round were frozen on one day, after every earlier protocol and the combined analysis')
    plain_fact(S5, f'were frozen on {frozen[0]}, after every result above was known', frozen[0], '2026-09-14')
    plain_fact(S5, f'The {words(len(families))} controlled experiments of Sections', f'The {words(len(families))} controlled',
               f'The {words(len(families) - 2)} controlled')
    plain_fact(S5, f'For each of the {words(len(families))} controlled experiments and the duplicate-free rerun',
               f'the {words(len(families))} controlled', f'the {words(len(families) - 2)} controlled')
    T = tests['tests']
    t17, t10 = T[('depth', 'all')], T[('depth', 'further')]
    plain_rule(t17['p'] >= 0.05 and t10['p'] >= 0.05, '"not detectably more accurate": both depth p values at least 0.05')
    plain_fact(S3, f'Two hidden layers with the corrected relay are not detectably more accurate than one either (one-sided '
                   f'$p={pval(t17["p"])}$ over all {words(k17)} and $p={pval(t10["p"])}$ over the {words(k)} further datasets).',
               f'$p={pval(t17["p"])}$', f'$p={float(pval(t17["p"])) - 0.01:.3f}$')
    # S3.7: depth with the corrected relay
    plain_rule(rl['record']['check_totals']['reused_arms_reproduced']['passed'] == k17,
               'the reused arms were reproduced on the first outer fold of every dataset')
    plain_fact(S3, 'a refit of the first outer fold of every dataset reproduced both exactly', 'reproduced both exactly',
               'reproduced one of them')
    plain_fact(S3, f'one whose first-layer votes are also multiplied by {rl["scale"]}.', f'by {rl["scale"]}.', f'by {rl["scale"] * 2}.')
    target = rl['target']
    plain_rule(target == 0.5, '"at least half": the cleared-share target is one half')
    plain_fact(S3, f'The scale {rl["scale"]} is the smallest of {listing(rl["ladder"], "and")} at which at least half of the first '
                   f"layer's voted filter-batches reach that threshold, on training-only pilot fits of "
                   f'{listing([NAME[x] for x in rl["pilots"][::-1]], "and")}.', f'The scale {rl["scale"]}',
               f'The scale {rl["scale"] * 2}')
    most = set(rl['most'].values())
    plain_rule(most == {9}, 'the most-threshold of the depth rule is nine of the seventeen datasets')
    plain_fact(S3, f'a corrected arm helps only if it is ahead of one hidden layer on at least {most.pop()} of the {k17} datasets and '
                   f'significantly ahead, after Holm adjustment, on at least one', 'at least 9 of', 'at least 8 of')
    plain_fact(S3, f'One hidden layer is more accurate than two with the corrected relay on {rl["first_ahead"][c1]} of the {k17} '
                   f'datasets, by ${signed(rl["mean"][c1], 2)}$ points on average, and than the scaled arm on '
                   f'{rl["first_ahead"][c2]}. No contrast of either arm survives Holm adjustment.',
               f'on {rl["first_ahead"][c1]} of the', f'on {rl["first_ahead"][c1] + 1} of the')
    plain_fact(S3, f'The correction itself makes two-layer networks more accurate than the earlier relay on {rl["first_ahead"][c3]} of '
                   f'the {k17} datasets, by ${signed(rl["mean"][c3], 2)}$ points on average, again with no Holm-significant contrast.',
               f'on {rl["first_ahead"][c3]} of', f'on {rl["first_ahead"][c3] - 1} of')
    far_more(rl['reordered']['depth2_signed_scaled'], rl['reordered']['depth2_signed'],
             'first-layer reorders per update with the scaled votes against the corrected relay')
    plain_rule(rl['mean'][c2] > 0 and rl['first_ahead'][c2] > k17 / 2, '"without making two layers more accurate": one layer ahead of '
                                                                        'the scaled arm on most datasets and on average')
    plain_fact(S3, f"The scaled votes reorder the first layer's filters far more often, in {pct(rl['reordered']['depth2_signed_scaled'], 1)} "
                   f"against {pct(rl['reordered']['depth2_signed'], 1)} percent of the updates on average, without making two layers more "
                   f"accurate", f"in {pct(rl['reordered']['depth2_signed_scaled'], 1)} against",
               f"in {pct(rl['reordered']['depth2_signed_scaled'] + 0.001, 1)} against")
    # S1.1: how often a batch clears the prior's weight threshold (referee-panel revision of 2026-09-25, item A2), the mean row of
    # Table S-relay-movement. The record holds the share only for the two arms with the corrected relay (both hidden layers), none
    # for one hidden layer or the earlier relay, and the table reports the first hidden layer
    movement = rl['record']['movement']
    plain_rule(all(layer.get('cleared_share_of_voted') is None for a in ('depth1', 'depth2_printed') for x in COMBINED
                   for layer in movement[a][x]['by_layer'])
               and all(movement[a][x]['by_layer'][0].get('cleared_share_of_voted') is not None
                       for a in ('depth2_signed', 'depth2_signed_scaled') for x in COMBINED),
               'the clearing share is reported only for the first of two hidden layers under the corrected relay, and no other arm '
               'records it')
    plain_fact('supplement_sections/S1_proofs.tex',
               f"There, {pct(rl['cleared']['depth2_signed'], 1)} percent of the filter-batches with a nonzero vote reached it on average, "
               f"and {pct(rl['cleared']['depth2_signed_scaled'], 1)} percent when these votes were multiplied by {rl['scale']} "
               f"(Table~\\ref{{tab:s-relay-movement}}).", f"There, {pct(rl['cleared']['depth2_signed'], 1)} percent",
               f"There, {pct(rl['cleared']['depth2_signed'] + 0.001, 1)} percent", f"multiplied by {rl['scale']} (",
               f"multiplied by {2 * rl['scale']} (")
    # S3.7: the representation test
    reproduced = {v['matching_fold_seeds'] for x in COMBINED for v in rp['record']['reproduction'][x].values()}
    plain_rule(reproduced == {FACTS['n_outer'] * FACTS['n_seeds']}, 'every stored prediction reproduced on every fold and seed')
    plain_fact(S3, f'A refit of ArrowFlow reproduced its stored predictions on all {FACTS["n_outer"] * FACTS["n_seeds"]} fold-seeds of every '
                   f'dataset', f'all {FACTS["n_outer"] * FACTS["n_seeds"]} fold-seeds', f'all {FACTS["n_outer"] * FACTS["n_seeds"] - 1} fold-seeds')
    svc = rp['protocol']['readouts']['svc']
    grid_row = 'Kendall SVC & \\texttt{C}: ' + ', '.join(f'{c:g}' for c in svc['grid']['C'])
    plain_rule("kendall_svc baseline's grid" in svc['grid_source'] and grid_row in WRITTEN['tab_s_grids']
               and "the very splits on which the view's kNN readout is chosen" in svc['selection']
               and 'plurality vote with ArrowFlow\'s own tie rule' in rp['protocol']['aggregation'],
               'the SVC takes the Kendall SVC\'s C grid of Table S-grids, is tuned on the kNN readout\'s own splits, and the views vote '
               'by plurality with ArrowFlow\'s tie rule')
    plain_fact(S3, 'on the very splits of the training rows that choose each view\'s nearest-neighbor readout', 'the very splits',
               'other splits')
    plain_fact(S3, f'they have the higher mean accuracy on at least {prim["most"]} of the {k}', f'at least {prim["most"]} of',
               f'at least {prim["most"] - 1} of')
    plain_fact(S3, 'Otherwise it was to state that training improves the nearest-neighbor neighborhoods but not the representation read '
                   'by a strong fixed readout.', 'but not the representation', 'and the representation')
    plain_rule('screened for this question before this protocol was written' in rp['protocol']['screening_disclosure']['status'],
               'the development datasets were screened before the protocol')
    plain_fact(S3, 'an exploratory screen of the same question had used them before the protocol was written', 'before the protocol',
               'after the protocol')
    plain_fact(S3, f'The trained rankings have a higher mean accuracy than the untrained ones on {len(prim["higher"])} of the {k} further '
                   f'datasets, and the mean difference is ${signed(prim["mean"], 2).replace("$-$", "-")}$ points. Against the encoded '
                   f'inputs they are ahead on {len(sec["higher"])} as well. No contrast of either family survives Holm adjustment.',
               f'on {len(prim["higher"])} of the', f'on {len(prim["higher"]) + 1} of the')
    dev = [float(rp['development'][('svc_untrained', x)]['mean_difference']) for x in DATASETS]
    plain_rule(max(abs(v) for v in dev) < 0.01 and min(dev) < 0 < max(dev), '"below one point and of both signs" on the benchmark datasets')
    plain_fact(S3, f'On the {words(n)} benchmark datasets the differences are below one point and of both signs.', 'below one point',
               'above one point')
    means = rp['svc_mean']
    plain_rule(all(v > 0 for v in means.values()), 'the SVC is more accurate than the kNN readout on average on every representation')
    plain_fact(S3, 'by ' + ', '.join(f'${pct(means[rep], 2)}$' for rep in REP_REPRESENTATIONS[:2]) + f' and ${pct(means["trained"], 2)}$ '
                   'points on the input, untrained and trained rankings', f'${pct(means["trained"], 2)}$',
               f'${pct(means["trained"] + 0.0001, 2)}$')
    majority((means['untrained'] - means['trained']) / means['untrained'], 'training closes most of the kNN readout\'s gap to the SVC')
    plain_fact(S3, "Training thus closes most of the nearest-neighbor readout's gap to the SVC, while under the SVC the trained rankings "
                   'are no better than the untrained ones.', 'closes most of', 'closes all of')
    return PLAIN_CLAIMS[start:]


# ------------------------------------------------------------------------------------ the learned encoder: the guarded text
SANDBOX_RUNS = REPO / 'experiments' / 'tensor_rank_sandbox' / 'runs'
SANDBOX_KEY = {'steel_plates': 'steel_plates_fault'}   # the sandbox's name for a dataset of this paper, where it differs


def close_to(a, b, what):
    plain_rule(abs(a - b) <= 0.005, f'"close to" needs a difference of at most half a point, got {a:.4f}, {b:.4f} ({what})')


def learned_encoder_claims(controlled, knn):
    """The numbers and plain phrases of the learned encoder (the abstract, the Introduction, Sections 2, 5.5 and 7.7, the
    Discussion, the Conclusion and S6), each rebuilt from the two runs as render_learned_encoder verified them, from the
    development records in experiments/tensor_rank_sandbox/runs, which the freeze commits already held, or from the code,
    with its mutation case (plain_fact). Two plain rules join those listed above: "close to" (|a - b| <= 0.5 points) and
    "the lowest of all the tuned models" (below every tuned model of this paper on that dataset)."""
    start = len(PLAIN_CLAIMS)
    I, E, D, K, R = INTRO, E7, 'sections/08_discussion.tex', 'sections/09_conclusion.tex', 'sections/02_related.tex'
    enc, k17 = controlled['learned'], len(COMBINED)
    p1, p2, p3 = enc['P']['P1'], enc['P']['P2'], enc['P']['P3']
    prim, lr, rf = enc['S']['primary'], enc['S']['learned_random'], enc['S']['random_fixed']
    hyb, swap, registered, mean = enc['hyb'], enc['swap'], enc['registered'], enc['mean']
    error17 = lambda folds_of: 1 - statistics.mean(mean(folds_of(x)) for x in COMBINED)
    e_fixed, e_mlp = error17(lambda x: registered[(x, KNN)]), error17(lambda x: registered[(x, 'mlp')])
    e_learned, e_random = error17(lambda x: swap[(x, 'learned')]), error17(lambda x: swap[(x, 'random')])
    e_hyb = {(arm, r): error17(lambda x, arm=arm, r=r: hyb[(x, arm, r)]) for arm in HYBRID_ARMS for r in ('rank', 'knn')}
    names = lambda keys: listing([NAME[x] for x in keys], 'and')
    # the datasets on whose single splits the conversion and its settings were developed (S6.2)
    dev = sorted({SANDBOX_KEY.get(r['dataset'], r['dataset']) for r in read_json(SANDBOX_RUNS / 'core_confirm.json')['results']})
    plain_rule(set(dev) <= set(COMBINED), 'the development datasets are datasets of this paper')

    # the rules the plain statements lean on
    plain_rule(len(p1['higher']) == k17 and p1['up'] and not p1['down'], '"can train": the trained encoder is ahead of the untrained '
                                                                          'one on every dataset, significantly on some, behind on none')
    plain_rule(enc['verdict'] == 'improves ArrowFlow', 'the rule fixed before the swap reads that the trained encoder improves ArrowFlow')
    majority(len(prim['higher']) / k17, 'ArrowFlow with the trained encoder is more accurate than with the fixed one on most datasets')
    losses = [x for x in COMBINED if prim['rows'][x]['mean'] < 0]
    plain_rule(len(prim['higher']) + len(losses) == k17, 'no dataset without a difference in the encoder swap')
    worst = min(losses, key=lambda x: prim['rows'][x]['mean'])
    top = max(COMBINED, key=lambda x: p1['rows'][x]['mean'])
    below_fixed = [x for x in COMBINED if rf['rows'][x]['mean'] < 0]
    plain_rule(len(lr['higher']) > k17 / 2 and len(below_fixed) > k17 / 2 and lr['up'] and not lr['down'],
               '"the gain comes from the training, not from the network": the trained encoder beats the untrained one on most '
               'datasets, never significantly behind, and the untrained one is behind the fixed encoder on most')
    close_to(e_learned, e_mlp, 'ArrowFlow with the trained encoder and the MLP, mean error over the seventeen')
    close_to(e_hyb[('hybrid_target_mlp', 'knn')], e_fixed, 'the nearest-neighbor readout of the trained encoder and ArrowFlow')
    # balance-scale: ArrowFlow with the trained encoder has the lowest error of every tuned model of this paper there
    b = 'balance_scale'
    others = [registered[(b, m)] for m in [KNN] + LEARNED_COMPARATORS] + [swap[(b, 'random')]] + \
             [hyb[(b, arm, r)] for arm in HYBRID_ARMS for r in ('rank', 'knn')]
    plain_rule(all(mean(swap[(b, 'learned')]) > mean(o) for o in others), 'the lowest error of all the tuned models on balance-scale')
    equal = [x for x in COMBINED if mean(swap[(x, 'learned')]) == enc['best'][x]]
    ahead = [x for x in COMBINED if mean(swap[(x, 'learned')]) > enc['best'][x]]
    plain_rule(equal == ['banknote_authentication'] and enc['best'][equal[0]] == 1 and ahead == [b]
               and len(enc['trails']) + len(equal) + len(ahead) == k17,
               'behind the best tuned classical model on every dataset but banknote-authentication, where both are perfect, and '
               'balance-scale')

    # ------------------------------------------------------------------ the abstract
    # "motions of the same kind" and "by gradient steps" (referee-panel revision of 2026-09-25, items B1, B2, F1, F2)
    plain_fact('main.tex', 'Turned into target scores, motions of the same kind also train a neural encoder network by gradient steps, '
                           'without differentiating the sort.', 'also train', 'cannot train')
    plain_fact('main.tex', f'the trained encoder network is more accurate than the same network untrained on all {k17}, significantly on '
                           f'{len(p1["up"])}.', f'significantly on {len(p1["up"])}.', f'significantly on {len(p1["up"]) + 1}.')
    plain_fact('main.tex', f'Swapped into ArrowFlow, it raises mean accuracy on {len(prim["higher"])} datasets, significantly on '
                           f'{num_word(len(prim["up"]))}, and lowers it significantly on {num_word(len(prim["down"]))}.',
               f'on {len(prim["higher"])} datasets', f'on {len(prim["higher"]) + 1} datasets',
               f'significantly on {num_word(len(prim["up"]))},', f'significantly on {num_word(len(prim["up"]) + 1)},')
    # the modest-gain clause now ends the abstract's nested cross-validation sentence (items B4, F1); the matched-control finding is
    # carried by the guarded "leave the network less accurate on most datasets than never moving its hidden filters"
    plain_fact('main.tex', 'training the ranking layers helps modestly.', 'helps modestly', 'helps greatly')

    # ------------------------------------------------------------------ the Introduction
    plain_fact(I, 'It also asks whether the signal of those votes can train an ordinary neural network as well.', 'can train',
               'cannot train')
    plain_fact(I, 'can train ranking layers, and that the motions they return can train an ordinary neural network:',
               'can train an ordinary', 'cannot train an ordinary')
    plain_fact(I, f'The trained encoder network is more accurate than the same network untrained on all {k17} datasets, significantly '
                  f'on {len(p1["up"])}.', f'on all {k17}', f'on {len(p1["higher"]) - 1} of the {k17}')
    plain_rule([NAME[x] for x in prim['up']] == ['balance-scale'], 'the one significant gain of the encoder swap is on balance-scale')
    plain_fact(I, f'Swapped into ArrowFlow for the fixed encoder, it raises the mean accuracy on {len(prim["higher"])} of the {k17} '
                  f'datasets, significantly on {names(prim["up"])}, and lowers it significantly on {num_word(len(prim["down"]))} '
                  f'(Section~\\ref{{sec:learned}}).', f'significantly on {num_word(len(prim["down"]))}', 'significantly on one')
    # the development record where the results are claimed (referee-panel revision of 2026-09-25, item B4)
    plain_fact(I, f'The conversion of motions into target scores and its settings were developed on single splits of {words(len(dev))} of '
                  f'the {words(k17)} datasets (Section~S6.2).', f'of {words(len(dev))} of', f'of {words(len(dev) - 1)} of')

    # ------------------------------------------------------------------ Section 2 and Section 5.5
    plain_fact(R, 'Its trained encoder (Section~\\ref{sec:learned-encoder}) also learns through an exact sort, by turning the motions '
                  'into target scores', 'through an exact sort', 'through a relaxed sort')
    plain_rule(HYBRID_DESIGN['common']['contrast'] == 0.5 == SWAP_DESIGN['encoder']['contrast']
               and HYBRID_DESIGN['common']['beta'] == 1.0 == SWAP_DESIGN['encoder']['beta'],
               'the away term is one half and the attraction one in both designs, as Equation (eq:target) writes them')
    plain_fact(E5, 'q_p = p+m_y[p]-\\tfrac12\\,m_w[p]', '\\tfrac12', '\\tfrac14')
    # the verbal rule of Equation (eq:target), corrected in the referee-panel revision of 2026-09-25 (item A7)
    plain_fact(E5, "This is the coordinate's position in the true filter, minus half the displacement that the nearest wrong filter asks "
                   "of it.", 'minus half', 'minus all')
    # the network facts of Section 5.5, from the code (TensorNet: three linear layers of width e, the first two initialized by
    # nn.init.uniform on [0, 1)) and from both designs (60 passes; core_hybrid.build keeps no validation rows unless asked, and neither
    # study asks)
    net_code = (REPO / 'arrowflow' / 'arrowflow.py').read_text()
    net_code = net_code[net_code.index('class TensorNet'):net_code.index('def model_embedding_nlp_net')]
    hybrid_code = (REPO / LEARNED_MODULES['conversion']).read_text()
    plain_rule(all(f'self.{name} = nn.Linear({src}, number_of_hidden' in net_code
                   for name, src in (('fc1', 'self.input_dim'), ('fc2', 'number_of_hidden'), ('fc3', 'number_of_hidden')))
               and 'nn.init.uniform(self.fc1.weight)' in net_code and 'nn.init.uniform(self.fc2.weight)' in net_code
               and 'nn.init.uniform(self.fc3' not in net_code,
               'the encoder network: three fully connected layers of width e, the first two drawn uniformly from 0 to 1')
    plain_fact(E5, 'The encoder network has three fully connected layers of width $e$, and the first two start with weights drawn uniformly '
                   'from 0 to 1.', 'three fully connected', 'four fully connected')
    plain_rule(HYBRID_DESIGN['epochs'] == SWAP_DESIGN['encoder']['epochs'] and 'val_ratio=0.' in hybrid_code.split('def build(', 1)[1][:250]
               and all('val_ratio' not in (REPO / LEARNED_MODULES[w]).read_text() for w in ('nested', 'swap')),
               'both studies train the encoder network for the same number of passes, with no validation rows')
    plain_fact(E5, f'It trains for {HYBRID_DESIGN["epochs"]} passes over the training rows, with no validation checkpoint.',
               f'for {HYBRID_DESIGN["epochs"]} passes', f'for {2 * HYBRID_DESIGN["epochs"]} passes')

    # ------------------------------------------------------------------ Section 7: the questions, the summary and Section 7.7
    questions = E7_QUESTIONS.search((HERE / E).read_text())
    plain_rule(questions is not None and questions.group(1) == 'six' and questions.group(2).count('?') == 6,
               'the opening of Section 7 asks as many questions as it counts')
    # the summary gives the swap's counts (referee-panel revision of 2026-09-25, items B2 and C1), and the answer to the second question
    # is the reading of the rule fixed before the run (the verdict rule above)
    plain_fact(E, f'Motions of the same kind also train a real-valued encoder network. With the trained encoder, ArrowFlow has the higher '
                  f'mean accuracy on {counted(len(prim["higher"]), k17, "datasets")}, significantly on {num_word(len(prim["up"]))}.',
               f'on {len(prim["higher"])} of', f'on {len(prim["higher"]) - 1} of',
               f'significantly on {num_word(len(prim["up"]))}.', f'significantly on {num_word(len(prim["up"]) + 1)}.')
    plain_fact(E, 'whether ArrowFlow gains from the trained encoder. The answer to the first is yes, and the answer to the second is a '
                  'modest yes, by a rule fixed before the run (Figure~\\ref{fig:learned}).', 'a modest yes', 'a clear yes')
    # the development record in Section 7.1 and at the opening of Section 7.7 (item B4)
    plain_fact(E, f'Its target conversion and settings were developed on single splits of {words(len(dev))} of the {words(k17)} datasets, '
                  f'{words(sum(x in NEWDATA for x in dev))} of them further datasets (Section~S6.2).',
               f'{words(sum(x in NEWDATA for x in dev))} of them', f'{words(sum(x in NEWDATA for x in dev) + 1)} of them')
    plain_fact(E, f"Both studies below use the benchmark's test sets, but the target conversion and its settings were developed on single "
                  f"splits of {words(len(dev))} of its {words(k17)} datasets (Section~\\ref{{sec:protocol}}).",
               f'of {words(len(dev))} of its', f'of {words(len(dev) - 1)} of its')
    plain_fact(E, f'Trained by the converted motions, it was more accurate than the same network left at its random start on all {k17} '
                  f'datasets, significantly so on {len(p1["up"])} (Figure~\\ref{{fig:learned}}a).',
               f'significantly so on {len(p1["up"])}', f'significantly so on {len(p1["up"]) - 1}')
    # the weak reference (item B5): the untrained encoder starts from the code's uniform draws, checked above, and its mean error
    plain_fact(E, f"Its first two layers start with weights drawn uniformly from 0 to 1, and its mean error is "
                  f"{pct(e_hyb[('hybrid_frozen_mlp', 'rank')])} percent, against {pct(e_hyb[('hybrid_target_mlp', 'rank')])} percent "
                  f"after training.", f"is {pct(e_hyb[('hybrid_frozen_mlp', 'rank')])} percent",
               f"is {pct(e_hyb[('hybrid_frozen_mlp', 'rank')] + 0.001)} percent",
               f"against {pct(e_hyb[('hybrid_target_mlp', 'rank')])} percent", f"against {pct(e_hyb[('hybrid_target_mlp', 'rank')] + 0.001)} percent")
    # the largest gain against its weak starting point (item B5): the untrained encoder's error on that dataset, Table S-learned-nested
    top_start = 1 - mean(hyb[(top, 'hybrid_frozen_mlp', 'rank')])
    plain_fact(E, f'The largest gain, {pct(p1["rows"][top]["mean"])} points on {NAME[top]}, started from an error of {pct(top_start)} '
                  f'percent.', f'{pct(p1["rows"][top]["mean"])} points', f'{pct(p1["rows"][top]["mean"] + 0.001)} points',
               f'{pct(top_start)} percent', f'{pct(top_start + 0.001)} percent')
    # the other two contrasts fixed before the run (item B5)
    plain_fact(E, f'It is ahead on {len(p2["higher"])} datasets, significantly on {num_word(len(p2["up"]))}, and significantly behind on '
                  f'{len(p2["down"])}.', f'ahead on {len(p2["higher"])} datasets', f'ahead on {len(p2["higher"]) + 1} datasets',
               f'behind on {len(p2["down"])}.', f'behind on {len(p2["down"]) + 1}.')
    plain_fact(E, f"Read instead by a nearest-neighbor vote, a single trained encoder with no hidden ranking layer comes close to ArrowFlow, "
                  f"with a mean error of {pct(e_hyb[('hybrid_target_mlp', 'knn')])} against {pct(e_fixed)} percent (Section~S6.3).",
               f"of {pct(e_hyb[('hybrid_target_mlp', 'knn')])} against", f"of {pct(e_hyb[('hybrid_target_mlp', 'knn')] + 0.001)} against")
    plain_rule(len(enc['checked']) == len(SWAP_CHECK) * len(FACTS['bridge']['fit_seeds']),
               'every sampled refit of the fixed arm reproduced the stored predictions (render_learned_encoder fails on any differing '
               'row)')
    # the reproduction check as run (item B9): the declared sample of test sets, each at every fitting seed
    checked_sets = sorted({(x, index) for x, index, _, _ in enc['checked']})
    plain_rule(len(checked_sets) == len(SWAP_CHECK) and len(enc['checked']) == len(checked_sets) * len(FACTS['bridge']['fit_seeds']),
               'every test set of the reproduction check was refitted at every fitting seed')
    plain_fact(E, f'With the fixed encoder, the rebuild reproduced the stored predictions exactly on the {words(len(checked_sets))} test sets '
                  f'we checked, at all {words(len(FACTS["bridge"]["fit_seeds"]))} fitting seeds ({len(enc["checked"])} fits; '
                  f'Section~S6.4).', f'({len(enc["checked"])} fits;', f'({len(enc["checked"]) + 1} fits;',
               f'the {words(len(checked_sets))} test sets', f'the {words(len(checked_sets) + 1)} test sets')
    fm = knn.protocol['full_method']
    swap_header = ' '.join((REPO / LEARNED_MODULES['swap']).read_text().split('"""', 2)[1].split())
    plain_rule('(there are no projection strategies), and the polynomial degree does not apply' in swap_header
               and len(set(fm['view_strategy_cycle'])) == len(fm['view_strategy_cycle']),
               'the swap drops the polynomial expansion and every projection strategy of the fixed encoder')
    plain_fact(E, f"This also removed the polynomial expansion and the {words(len(fm['view_strategy_cycle']))} projection strategies.",
               f"the {words(len(fm['view_strategy_cycle']))} projection", f"the {words(len(fm['view_strategy_cycle']) + 1)} projection")
    plain_fact(E, f'With the trained encoder network, ArrowFlow had the higher mean accuracy on {len(prim["higher"])} of the {k17} '
                  f'datasets, significantly on {names(prim["up"])}, and was significantly less accurate on '
                  f'{num_word(len(prim["down"]))} (Figure~\\ref{{fig:learned}}b).', f'on {len(prim["higher"])} of',
               f'on {len(prim["higher"]) - 1} of')
    # the rule in full (item C1): its two conditions, as the design header of the swap module fixed them, both hold
    plain_rule('"improves ArrowFlow" if the mean over the 17 is higher and more datasets are significantly higher than significantly lower'
               in swap_header and prim['mean'] > 0 and len(prim['up']) > len(prim['down']),
               'the reading rule of the swap has two conditions, a higher mean and more significant gains than losses, and both hold')
    plain_fact(E, f'Its mean error over the datasets fell from {pct(e_fixed)} to {pct(e_learned)} percent. The rule fixed before the run has '
                  f'two conditions: a higher mean accuracy, and more datasets significantly higher than significantly lower. Both hold, so '
                  f'the rule reads that the trained encoder improves ArrowFlow.', f'to {pct(e_learned)} percent',
               f'to {pct(e_learned - 0.001)} percent', 'has two conditions', 'has three conditions')
    # the one significant gain carries the second condition, on a development dataset; across the datasets a post hoc exact one-sided
    # signed-rank test (as Table S-signed-rank) does not support the gain ("not established": p of at least 0.05)
    plain_rule(len(prim['up']) == 1 and not prim['down'] and prim['up'][0] in dev,
               'a single dataset, a development dataset, is significantly higher, and none significantly lower')
    plain_fact(E, f'A single dataset, {NAME[prim["up"][0]]}, meets the second condition, and the conversion was developed partly on it.',
               f'{NAME[prim["up"][0]]},', 'Vehicle,')
    swap_diffs = [prim['rows'][x]['mean'] for x in COMBINED]
    plain_rule(all(v != 0 for v in swap_diffs) and len({abs(v) for v in swap_diffs}) == len(swap_diffs),
               'the encoder swap has no zero and no tied mean difference, so the exact signed-rank test applies')
    swap_rank = signed_rank_greater(swap_diffs)
    plain_rule(swap_rank['p'] >= 0.05, '"not established": the one-sided signed-rank p value is at least 0.05')
    plain_fact(E, f'Across the datasets the gain is not established (post hoc one-sided signed-rank test, $p={pval(swap_rank["p"])}$).',
               f'$p={pval(swap_rank["p"])}$', f'$p={float(pval(swap_rank["p"])) - 0.001:.3f}$')
    plain_fact(E, f'Left untrained, the same network made ArrowFlow less accurate than the fixed encoder did on {len(below_fixed)} of the '
                  f'{k17} datasets.', f'on {len(below_fixed)} of', f'on {len(below_fixed) - 1} of')
    plain_fact(E, 'So the gain comes from training. Swapping in the network alone makes ArrowFlow worse.', 'comes from training',
               'comes from the network')
    plain_fact(E, f'The trained encoder made ArrowFlow less accurate on {len(losses)} of the {k17} datasets, by up to '
                  f'{pct(-prim["rows"][worst]["mean"])} points on {NAME[worst]}, although none of these losses was significant.',
               f'on {len(losses)} of', f'on {len(losses) - 1} of')
    plain_rule(not prim['down'], 'none of the losses of the encoder swap is significant')
    plain_fact(E, f"With the trained encoder, ArrowFlow's mean error of {pct(e_learned)} percent is close to the tuned MLP's "
                  f"{pct(e_mlp)}.", f"MLP's {pct(e_mlp)}", f"MLP's {pct(e_mlp + 0.001)}")
    plain_fact(E, f'On balance-scale its error of {pct(1 - mean(swap[(b, "learned")]))} percent is the lowest of all the tuned models. '
                  f'But it still trails the best tuned classical model on {len(enc["trails"])} of the {k17} datasets',
               f'on {len(enc["trails"])} of', f'on {len(enc["trails"]) - 1} of')
    lm = enc['S']['learned_mlp']
    plain_rule(lm['rows'][b]['mean'] > 0 and lm['rows'][b]['holm'] >= 0.05, 'on balance-scale the lead over the MLP is not significant')
    plain_fact(E, f'These comparisons are descriptive, and on {NAME[b]} the lead over the MLP is not significant.', 'is not significant',
               'is significant')
    mlp_warn = controlled['mlp_warnings']
    plain_fact(E, f"The MLP reached its iteration cap before convergence on {mlp_warn['warned']} of its {mlp_warn['fits']} outer fits "
                  f"(Section~S3.2).", f"on {mlp_warn['warned']} of", f"on {mlp_warn['warned'] + 1} of")
    hk = enc['S']['learned_hybrid_knn']
    plain_rule(hk['mean'] > 0 and not hk['up'] and not hk['down'], 'the fourth secondary contrast: a positive mean, and no dataset '
                                                                     'significant either way')
    plain_fact(E, f'Against the single trained encoder read by a nearest-neighbor vote, ArrowFlow with the trained encoder is ahead on '
                  f'{counted(len(hk["higher"]), k17, "datasets")}, by {pct(hk["mean"], 2)} points on average, and significantly on '
                  f'{num_word(len(hk["up"]))} (Section~S6.4).', f'ahead on {len(hk["higher"])} of', f'ahead on {len(hk["higher"]) - 1} of',
               f'by {pct(hk["mean"], 2)} points', f'by {pct(hk["mean"] + 0.0001, 2)} points')

    # ------------------------------------------------------------------ the Discussion and the Conclusion
    plain_fact(D, 'A learning rule that computes no derivative trains ranking layers, and the signal it sends down carries information. '
                  'Motions of the same kind can also train a real-valued encoder network.', 'can also train', 'cannot train')
    plain_fact(D, f'As a classifier, the trained encoder is more accurate than the same network untrained on every dataset, significantly '
                  f'on {len(p1["up"])} (Section~\\ref{{sec:learned}}).', 'on every dataset', 'on most datasets',
               f'significantly on {len(p1["up"])} (', f'significantly on {len(p1["up"]) + 1} (')
    plain_fact(D, f'Inside ArrowFlow the gain is modest and uneven. {num_word(len(prim["up"])).capitalize()} dataset improves '
                  f'significantly and {num_word(len(prim["down"]))} worsens significantly, but {words(len(losses))} lose some accuracy.',
               f'but {words(len(losses))} lose', f'but {words(len(losses) + 1)} lose')
    plain_fact(D, f'the conversion and its settings were developed on single splits of {words(len(dev))} of the {words(k17)} datasets '
                  f'before these runs (Section~S6.2).', f'of {words(len(dev))} of', f'of {words(len(dev) - 1)} of')
    plain_fact(K, 'Motions of the same kind also train an ordinary neural network.', 'also train', 'cannot train')
    plain_fact(K, f'Swapped into ArrowFlow for its fixed encoder, the trained encoder raised the mean accuracy on {len(prim["higher"])} of '
                  f'the {k17} datasets, significantly on {num_word(len(prim["up"]))}, and lowered it significantly on '
                  f'{num_word(len(prim["down"]))} (Section~\\ref{{sec:learned}}).',
               f'on {len(prim["higher"])} of', f'on {len(prim["higher"]) + 1} of',
               f'significantly on {num_word(len(prim["up"]))},', f'significantly on {num_word(len(prim["up"]) + 1)},')

    # ------------------------------------------------------------------ S6.1: the network and its training, from the code
    code = (REPO / 'arrowflow' / 'arrowflow.py').read_text()
    net = code[code.index('class TensorNet'):code.index('def model_embedding_nlp_net')]
    plain_rule('nn.init.uniform(self.fc1.weight)' in net and 'nn.init.uniform(self.fc2.weight)' in net and 'self.fc3 = nn.Linear' in net
               and re.search(r'self\.fc1,\s*self\.instance_norm,\s*nn\.ReLU\(\),\s*self\.fc2,\s*self\.instance_norm,\s*nn\.ReLU\(\),\s*'
                             r'self\.fc3\s*\)', net) is not None
               and 'forward_input_np = -1 * forward_input_np' in code and 'MultiStepLR' in code,
               'the code\'s encoder network: three layers, instance normalization and a ReLU after the first two, uniform weights in '
               'those two, a negated input and a step schedule')
    for revision in (knn.environment['code_revision'], LEARNED_FREEZE['nested'], LEARNED_FREEZE['swap']):
        if git_blob(revision, 'arrowflow/arrowflow.py') != (REPO / 'arrowflow' / 'arrowflow.py').read_bytes():
            raise AssertionError(f'S6.1 says arrowflow.py is unchanged, but it differs from its version at {revision[:9]}')
    plain_fact(S6, 'and the weights of those two layers start uniform on $[0,1)$, as the code draws them.', '$[0,1)$', '$[-1,1)$')
    plain_fact(S6, 'The file \\texttt{arrowflow.py} is unchanged.', 'is unchanged', 'is edited')
    build = HYBRID_DESIGN['build']
    plain_rule(build['p_correct'] == SWAP_DESIGN['encoder']['p_correct'] and build['lr_rank'] == SWAP_DESIGN['encoder']['lr_rank']
               and HYBRID_DESIGN['epochs'] == SWAP_DESIGN['encoder']['epochs'] and SWAP_DESIGN['encoder']['lr_tensor'] == 0.01
               and sorted({c['lr_tensor'] for c in HYBRID_GRID}) == [0.003, 0.01],
               'one set of training settings in both designs; the nested grid adds a learning rate of 0.003')
    plain_fact(S6, f'A misclassified example is always accepted, and a correctly classified one with probability $p_c={build["p_correct"]:g}$. '
                   f'The class filters learn by the output layer\'s rule at learning rate $\\eta={build["lr_rank"]:g}$.',
               f'$p_c={build["p_correct"]:g}$', f'$p_c={2 * build["p_correct"]:g}$')
    plain_fact(S6, f'For the accepted examples of a batch of {build["batch_size"]}, the target conversion', f'a batch of {build["batch_size"]}',
               f'a batch of {2 * build["batch_size"]}')
    plain_fact(S6, f'at learning rate ${SWAP_DESIGN["encoder"]["lr_tensor"]:g}$ (Section~\\ref{{supp:learned-nested}} also tries '
                   f'${min(c["lr_tensor"] for c in HYBRID_GRID):g}$)', '$0.003$', '$0.001$')
    plain_fact(S6, f'Training runs for {HYBRID_DESIGN["epochs"]} passes over the training rows in random batches, with no validation '
                   f'checkpoint.', f'for {HYBRID_DESIGN["epochs"]} passes', f'for {HYBRID_DESIGN["epochs"] * 2} passes')
    # the step schedule as the code builds it (referee-panel revision of 2026-09-25, item H8): TensorNet.define_optimizer takes its
    # milestones from the module-level default configuration, whose update count no module of this paper changes, so the steps fall
    # at fixed updates whatever the length of training
    config_src = (REPO / 'arrowflow' / 'config.py').read_text()
    for revision in (LEARNED_FREEZE['nested'], LEARNED_FREEZE['swap']):
        if git_blob(revision, 'arrowflow/config.py') != config_src.encode():
            raise AssertionError(f'S6.1: arrowflow/config.py differs from its version at {revision[:9]}')
    optimizer = code[code.index('def define_optimizer'):code.index('def step_optimizer')]
    m_default = re.search(r'^class SortNetConfig:.*?^    no_of_iters = (\d+)$', config_src, re.S | re.M)
    m_steps = re.search(r'milestones=\s*\[no_of_iters/(\d+), no_of_iters/(\d+), no_of_iters/(\d+), no_of_iters/(\d+)\]', optimizer)
    m_gamma = re.search(r'gamma=([\d.]+)\)', optimizer)
    assigned = [f.name for sub in ('arrowflow', 'experiments/make_revision', 'experiments/tensor_rank_sandbox')
                for f in sorted((REPO / sub).glob('*.py')) if re.search(r'sortnet_config\.no_of_iters\s*=[^=]', f.read_text())]
    plain_rule(m_default is not None and m_steps is not None and m_gamma is not None and not assigned
               and 'no_of_iters = sortnet_config.no_of_iters' in optimizer and 'from arrowflow.config import sortnet_config' in code
               and re.search(r'^sortnet_config = SortNetConfig\(\)$', config_src, re.M) is not None,
               'the schedule divides the default update count of the module-level configuration, which no module reassigns')
    default_updates, gamma = int(m_default.group(1)), float(m_gamma.group(1))
    milestones = [default_updates / int(g) for g in m_steps.groups()]
    plain_rule(all(v == int(v) for v in milestones) and milestones == sorted(milestones), 'the milestones fall on whole updates, in order')
    milestones = [int(v) for v in milestones]
    plain_fact(S6, f"The code's step schedule multiplies this learning rate by ${gamma:g}$ after {listing(milestones, 'and')} updates.",
               f"after {listing(milestones, 'and')} updates", f"after {listing([2 * v for v in milestones], 'and')} updates")
    plain_fact(S6, f'So after the first {milestones[-1]} updates it stays at ${gamma:g}^{len(milestones)}\\approx'
                   f'{gamma ** len(milestones):.2f}$ of its starting value.', f'\\approx{gamma ** len(milestones):.2f}$',
               f'\\approx{gamma ** (len(milestones) - 1):.2f}$')
    plain_fact(S6, f"The schedule takes these steps from the code's default of {default_updates} updates, whatever the length of training.",
               f'default of {default_updates} updates', f'default of {2 * default_updates} updates')
    nested_src, swap_src = ((REPO / LEARNED_MODULES[w]).read_text() for w in ('nested', 'swap'))
    batch = HYBRID_DESIGN['build']['batch_size']
    plain_rule(f'iterations = EPOCHS * int(np.ceil(len(yi) / {batch}))' in nested_src and f'EPOCHS = {HYBRID_DESIGN["epochs"]}' in nested_src
               and f"iterations = ENCODER['epochs'] * int(np.ceil(len(yi) / {batch}))" in swap_src
               and 'np.random.permutation(len(data_train))[0:batch_size]' in code,
               'both studies run epochs times ceil(n / 32) updates, each on a batch of rows drawn at random')
    plain_fact(S6, f'That is ${HYBRID_DESIGN["epochs"]}\\lceil n_{{\\mathrm{{tr}}}}/{batch}\\rceil$ updates for $n_{{\\mathrm{{tr}}}}$ '
                   f'training rows, each on {batch} rows drawn at random.', f'each on {batch} rows', f'each on {2 * batch} rows')
    sandbox_src = (REPO / 'experiments' / 'tensor_rank_sandbox' / 'sandbox.py').read_text()
    plain_rule('linear.weight.copy_(torch.randn(V, n_features) / np.sqrt(n_features))' in hybrid_code
               and 'linear.bias.copy_(torch.randn(V) * .1)' in hybrid_code
               and 'self.W = self.rng.randn(V, d) / np.sqrt(d)' in sandbox_src
               and "self.b = self.rng.randn(V) * .1 if cfg['bias'] else np.zeros(V)" in sandbox_src
               and re.search(r"DEFAULT = dict\(V=32, phi='linear', bias=True,", sandbox_src) is not None,
               'the linear layer of the module and of the sandbox starts from standard normal weights over sqrt(d) and biases times 0.1')
    plain_fact(S6, 'Where a single linear layer replaces the encoder network, as in Sections~\\ref{supp:learned-development} '
                   'and~\\ref{supp:learned-nested}, its weights start as standard normal draws divided by $\\sqrt d$. Its biases start '
                   'as standard normal draws times $0.1$.', 'divided by $\\sqrt d$', 'divided by $d$')

    # ------------------------------------------------------------------ S6.2: the development record, held by the freeze commits
    record = {}
    for name in ('conversion_check.json', 'search.jsonl', 'confirm.json', 'core_confirm.json', 'mlp_reference.json'):
        data = (SANDBOX_RUNS / name).read_bytes()
        if name != 'conversion_check.json' and git_blob(LEARNED_FREEZE['nested'], f'experiments/tensor_rank_sandbox/runs/{name}') != data:
            raise AssertionError(f'S6.2 says the development came before the runs, but {name} is not the file the nested freeze held')
        record[name] = [json.loads(line) for line in data.decode().splitlines()] if name.endswith('.jsonl') else json.loads(data)
    cc = record['conversion_check.json']
    plain_rule(all(cc['gradient_identity'].values()), 'the gradient of the code\'s conversion ignores the sign of the motions and equals '
                                                      'eps |m| sign(h) / numel')
    st = {k.split('/', 1)[1]: v for k, v in cc['sign_test'].items() if k.startswith('linear/')}
    plain_rule(len({v['before'] for v in st.values()}) == 1 and st['core_original']['after'] < st['core_original']['before']
               < st['core_detached']['after'] and st['target']['after'] < st['core_original']['after']
               and 'five updates' in cc['setting'] and 'class filters fixed' in cc['setting'] and 'digits' in cc['setting'],
               'the sign test: one start, five updates on Digits with the class filters fixed; the code\'s conversion closer, the '
               'detached formula farther, the target conversion closest')
    plain_fact(S6, 'We applied five updates of each conversion to one batch of Digits, from the same start and with the class filters '
                   'held fixed.', 'five updates', 'ten updates')
    # the check's gate and away term, as its record states its setting (referee-panel revision of 2026-09-25, item B19)
    plain_rule('gate all' in cc['setting'] and 'no away term' in cc['setting'],
               'the conversion check accepted every example and used no away term')
    plain_fact(S6, 'The check accepted every example and used no away term.', 'accepted every example', 'accepted the misclassified examples')
    plain_fact(S6, f"The code's conversion moved the rankings of a linear encoder from a mean footrule distance of "
                   f"{st['target']['before']:.0f} to their true class filters to {st['core_original']['after']:.0f}, through the pull "
                   f"toward zero alone. The detached formula pushed them out to {st['core_detached']['after']:.0f}, and the target "
                   f"conversion brought them to {st['target']['after']:.0f}", f"to {st['target']['after']:.0f}",
               f"to {st['target']['after'] + 1:.0f}")
    search, final = record['search.jsonl'], record['confirm.json']['final']
    by_config = defaultdict(list)
    for r in search:
        by_config[(r['stage'], json.dumps(r['cfg'], sort_keys=True))].append(r['rank_acc'])
    best_config = max(by_config, key=lambda key: statistics.mean(by_config[key]))
    plain_rule(json.loads(best_config[1]) == final and sorted({r['dataset'] for r in search}) == sorted(['iris', 'wine', 'breast_cancer',
                                                                                                          'digits'])
               and {r['cfg'].get('convert') for r in search} >= {'target', 'sign', 'linear'},
               'the sandbox search: four development datasets, three conversions, and the configuration with the highest mean '
               'development accuracy is the one confirmed')
    sandbox_code = (REPO / 'experiments' / 'tensor_rank_sandbox' / 'sandbox.py').read_text()
    plain_rule(final['convert'] == 'target' and final['contrast'] == 0.5 and final['eta_decay'] is True
               and "self.eta_scale = 1. - epoch / cfg['epochs']" in sandbox_code,
               'the confirmed configuration: the target conversion, the away term at one half and a linearly decaying learning rate')
    plain_fact(S6, f"a vocabulary of {final['V']}, the away term at one half and a learning rate of {final['eta']:g} that decays linearly "
                   f"over {final['epochs']} passes", f"a vocabulary of {final['V']}", f"a vocabulary of {final['V'] // 2}")
    conf = defaultdict(list)
    for r in record['confirm.json']['results']:
        conf[(r['dataset'], r['arm'])].append(r['rank_acc'])
    confirm_sets = sorted({d for d, _ in conf})
    arm_mean = lambda d, arm: statistics.mean(conf[(d, arm)])
    plain_rule(len(confirm_sets) == len(dev) and sorted(SANDBOX_KEY.get(d, d) for d in confirm_sets) == dev
               and {len(conf[(d, 'trained')]) for d in confirm_sets} == {5},
               'the sandbox confirmation covers the twelve development datasets with five seeds each')
    trained_ahead = sum(arm_mean(d, 'trained') > arm_mean(d, 'frozen') for d in confirm_sets)
    scrambled_behind = sum(arm_mean(d, 'scrambled') < arm_mean(d, 'frozen') for d in confirm_sets)
    order = list(dict.fromkeys(r['dataset'] for r in record['confirm.json']['results']))
    search_sets = [d for d in order if d in {r['dataset'] for r in search}]
    plain_rule(order[:len(search_sets)] == search_sets and len(search_sets) == 4, 'the four search datasets come first')
    sandbox_name = lambda d: NAME[SANDBOX_KEY.get(d, d)]
    plain_fact(S6, f'by cross-validation on the 70\\% development portions of {listing([sandbox_name(d) for d in search_sets], "and")}.',
               'the 70\\%', 'the 80\\%')
    # the search's stages and its first stage, where each conversion tried the same learning rates without the away term (the sandbox's
    # default contrast is zero); the mean development accuracy of each conversion at its best learning rate (item B19)
    stages = sorted({r['stage'] for r in search})
    first = [r for r in search if r['stage'] == stages[0] and 'convert' in r['cfg']]
    rates = {c: sorted({r['cfg']['eta'] for r in first if r['cfg']['convert'] == c}) for c in ('target', 'sign', 'linear')}
    plain_rule(stages == list(range(len(stages))) and len({tuple(v) for v in rates.values()}) == 1
               and all('contrast' not in r['cfg'] for r in first) and re.search(r'DEFAULT = dict\(.*?contrast=0\.,', sandbox_src) is not None,
               'the first stage tried the same learning rates for each conversion, with the away term at its default of zero')
    stage_mean = defaultdict(list)
    for r in first:
        stage_mean[(r['cfg']['convert'], r['cfg']['eta'])].append(r['rank_acc'])
    plain_rule({len(v) for v in stage_mean.values()} == {len(search_sets)}, 'every first-stage configuration ran on the four search datasets')
    best_rate = {c: max(statistics.mean(stage_mean[(c, eta)]) for eta in rates[c]) for c in rates}
    plain_rule(best_rate['sign'] > max(best_rate['linear'], best_rate['target']),
               'at the first stage the sign of the motion has the highest mean development accuracy')
    plain_fact(S6, f'The search ran in {words(len(stages))} stages. The first tried {words(len(rates["target"]))} learning rates for each '
                   f'conversion, without the away term.', f'in {words(len(stages))} stages', f'in {words(len(stages) + 1)} stages',
               f'tried {words(len(rates["target"]))} learning', f'tried {words(len(rates["target"]) + 1)} learning')
    plain_fact(S6, f"There, each at its best learning rate, the sign of the motion reached the highest mean development accuracy, "
                   f"{pct(best_rate['sign'])}\\%, against {pct(best_rate['linear'])}\\% for the scaled motion and "
                   f"{pct(best_rate['target'])}\\% for the target conversion.", f"accuracy, {pct(best_rate['sign'])}\\%",
               f"accuracy, {pct(best_rate['sign'] + 0.001)}\\%", f"and {pct(best_rate['target'])}\\% for",
               f"and {pct(best_rate['target'] + 0.001)}\\% for")
    # "more datasets", not "further datasets", which names the ten of Section 7.1 (item B18)
    plain_fact(S6, f'on 70/30 splits of {words(len(order) - len(search_sets))} more datasets of this paper: '
                   f'{listing([sandbox_name(d) for d in order[len(search_sets):]], "and")}.',
               f'of {words(len(order) - len(search_sets))} more', f'of {words(len(order) - len(search_sets) + 1)} more')
    plain_fact(S6, f'The trained linear encoder was more accurate than the frozen one on all {trained_ahead} datasets. The same signal '
                   f'with its coordinates shuffled was less accurate than the frozen encoder on {scrambled_behind} of the '
                   f'{len(confirm_sets)}.', f'on {scrambled_behind} of the', f'on {scrambled_behind - 1} of the')
    plain_rule(trained_ahead == len(confirm_sets), 'the trained linear encoder is ahead on all twelve sandbox datasets')
    core = defaultdict(list)
    for r in record['core_confirm.json']['results']:
        core[(r['arch'], r['arm'])].append(r['acc'])
    cm = {key: statistics.mean(v) for key, v in core.items()}
    plain_rule({len(v) for v in core.values()} == {len(dev) * 5}, 'every core arm ran on the twelve splits with five seeds')
    mlp_ref = statistics.mean(r['acc'] for r in record['mlp_reference.json'])
    plain_rule(len(record['mlp_reference.json']) == len(dev) * 5, 'the MLP reference ran on the twelve splits with five seeds')
    plain_fact(S6, f"the code's own encoder network reached a mean accuracy of {pct(cm[('core_mlp', 'target')])}\\% with the target "
                   f"conversion. It reached {pct(cm[('core_mlp', 'frozen')])}\\% when left at its random start, "
                   f"{pct(cm[('core_mlp', 'core_original')])}\\% with the code's own conversion, "
                   f"{pct(cm[('core_mlp', 'core_detached_flipped')])}\\% with that conversion detached and its sign reversed, and "
                   f"{pct(cm[('core_mlp', 'target_scrambled')])}\\% with the shuffled signal.",
               f"reached {pct(cm[('core_mlp', 'frozen')])}", f"reached {pct(cm[('core_mlp', 'frozen')] + 0.001)}")
    lag, lead = cm[('core_mlp', 'core_detached_flipped')] - cm[('core_mlp', 'target')], \
        cm[('linear', 'target')] - cm[('linear', 'core_detached_flipped')]
    plain_rule(lag > 0 and lead > 0, 'the target conversion trails the repaired code conversion with the network and leads it with a '
                                     'linear encoder')
    plain_fact(S6, f"It trails the repaired code conversion by {pct(lag)} points with the code's encoder network, and it leads by "
                   f"{pct(lead)} points with a linear encoder, {pct(cm[('linear', 'target')])}\\% against "
                   f"{pct(cm[('linear', 'core_detached_flipped')])}\\%.", f'by {pct(lag)} points', f'by {pct(lag + 0.001)} points')
    plain_fact(S6, f'The MLP comparison model of this paper, tuned in the same way, reached {pct(mlp_ref)}\\%.', f'{pct(mlp_ref)}\\%',
               f'{pct(mlp_ref + 0.001)}\\%')
    plain_fact(S6, f'These single splits cover {words(len(dev))} of the {words(k17)} datasets of this paper',
               f'cover {words(len(dev))} of', f'cover {words(len(dev) + 1)} of')
    # the port's settings (core_confirm.py: vocabulary 64, 60 passes, the target arm's gate and away term, and the defaults of
    # core_hybrid.build) against the two designs: the nested grid adds a vocabulary and a learning rate, and the swap takes its
    # vocabulary from ArrowFlow's selected configuration; nothing else differs (item B19)
    port_src = (REPO / 'experiments' / 'tensor_rank_sandbox' / 'core_confirm.py').read_text()
    build_head = hybrid_code.split('def build(', 1)[1].split(')', 1)[0]
    m_port = re.search(r'V=(\d+), iterations=iterations', port_src)
    m_epochs = re.search(r'^EPOCHS = (\d+)$', port_src, re.M)
    m_lr = re.search(r'lr_tensor=([\d.]+)', build_head)
    port = dict(V=int(m_port.group(1)), epochs=int(m_epochs.group(1)), lr_tensor=float(m_lr.group(1)),
                lr_rank=float(re.search(r'lr_rank=([\d.]+)', build_head).group(1)),
                p_correct=float(re.search(r'p_correct=([\d.]+)', build_head).group(1)),
                batch_size=int(re.search(r'batch_size=(\d+)', build_head).group(1)))
    plain_rule("'target': {'gate': 'accepted', 'contrast': .5}" in port_src
               and HYBRID_DESIGN['common']['gate'] == SWAP_DESIGN['encoder']['gate'] == 'accepted'
               and HYBRID_DESIGN['common']['contrast'] == SWAP_DESIGN['encoder']['contrast'] == 0.5
               and HYBRID_DESIGN['epochs'] == SWAP_DESIGN['encoder']['epochs'] == port['epochs']
               and HYBRID_DESIGN['build'] == {k: port[k] for k in ('lr_rank', 'p_correct', 'batch_size')}
               and SWAP_DESIGN['encoder']['lr_rank'] == port['lr_rank'] and SWAP_DESIGN['encoder']['p_correct'] == port['p_correct']
               and SWAP_DESIGN['encoder']['lr_tensor'] == port['lr_tensor'] and 'V=self.embed_dim' in swap_src,
               'the two runs keep the settings of the port except the vocabulary and, in the nested grid, a second learning rate')
    extra_v = sorted({c['V'] for c in HYBRID_GRID} - {port['V']})
    extra_lr = sorted({c['lr_tensor'] for c in HYBRID_GRID} - {port['lr_tensor']})
    plain_rule(len(extra_v) == 1 and len(extra_lr) == 1 and port['V'] in {c['V'] for c in HYBRID_GRID}
               and port['lr_tensor'] in {c['lr_tensor'] for c in HYBRID_GRID}, 'the nested grid adds one vocabulary and one learning rate')
    plain_fact(S6, f"Those runs kept the settings of the port, with two exceptions. The nested study also tried a vocabulary of "
                   f"{extra_v[0]} and a learning rate of {extra_lr[0]:g}. The encoder swap took its vocabulary from ArrowFlow's "
                   f"configuration.", f'a vocabulary of {extra_v[0]} and', f'a vocabulary of {2 * extra_v[0]} and', 'with two exceptions',
               'with three exceptions')

    # ------------------------------------------------------------------ S6.3 and S6.4
    plain_fact(S6, f"with vocabulary size $e\\in\\{{{','.join(str(v) for v in sorted({c['V'] for c in HYBRID_GRID}))}\\}}$ and learning rate "
                   f"${min(c['lr_tensor'] for c in HYBRID_GRID):g}$ or ${max(c['lr_tensor'] for c in HYBRID_GRID):g}$ chosen on the inner "
                   f"folds", '$0.003$ or', '$0.001$ or')
    plain_fact(S6, f"and its mean error over the datasets is {pct(e_hyb[('hybrid_target_mlp', 'rank')])}\\%, against "
                   f"{pct(e_hyb[('hybrid_frozen_mlp', 'rank')])}\\% untrained.", f"against {pct(e_hyb[('hybrid_frozen_mlp', 'rank')])}",
               f"against {pct(e_hyb[('hybrid_frozen_mlp', 'rank')] - 0.001)}")
    # how the nested study selects (item B8; the design header is checked in render_learned_encoder): at the first fitting seed, with
    # no rescoring of the finalists that the stochastic models of the main benchmark receive
    plain_fact(S6, f"Unlike the stochastic models of Table~\\ref{{tab:main}}, this study does not rescore its "
                   f"{words(knn.protocol['stochastic_finalists'])} best candidates with all {words(len(HYBRID_DESIGN['fit_seeds']))} seeds.",
               'does not rescore', 'does rescore')
    plain_fact(S6, f'The trained encoder is more accurate than the untrained one on all {len(p1["higher"])} datasets, significantly on '
                   f'{len(p1["up"])} (Table~\\ref{{tab:s-learned-nested-contrasts}})', f'significantly on {len(p1["up"])} (',
               f'significantly on {len(p1["up"]) - 1} (')
    plain_fact(S6, f'Read by its class filters, the classifier trails the MLP. It is ahead on {len(p2["higher"])} datasets, significantly '
                   f'on {num_word(len(p2["up"]))}, and significantly behind on {len(p2["down"])}, {names(p2["down"])}.',
               f'behind on {len(p2["down"])},', f'behind on {len(p2["down"]) + 1},')
    plain_fact(S6, f"its mean error of {pct(e_hyb[('hybrid_target_mlp', 'knn')])}\\% is close to ArrowFlow's {pct(e_fixed)}\\%.",
               f"ArrowFlow's {pct(e_fixed)}", f"ArrowFlow's {pct(e_fixed + 0.001)}")
    plain_fact(S6, f'It is ahead of ArrowFlow on {len(p3["higher"])} datasets, significantly on {names(p3["up"])}, and significantly behind '
                   f'on {names(p3["down"])}.', f'ahead of ArrowFlow on {len(p3["higher"])}', f'ahead of ArrowFlow on {len(p3["higher"]) + 1}')
    e_lin = e_hyb[('hybrid_target_linear', 'rank')]
    plain_rule(e_hyb[('hybrid_target_mlp', 'rank')] < e_lin < e_hyb[('hybrid_frozen_mlp', 'rank')],
               'the trained linear layer lies between the trained and the untrained network')
    plain_fact(S6, f'The trained single linear layer reaches a mean error of {pct(e_lin)}\\%, between the trained and the untrained '
                   f'network.', f'{pct(e_lin)}\\%', f'{pct(e_lin + 0.001)}\\%')
    check_sets = sorted({x for x, _, _, _ in enc['checked']})
    plain_fact(S6, f'the module refitted the fixed arm on {len(check_sets)} datasets at all three fitting seeds and reproduced the stored '
                   f'predictions exactly. None of the {len(enc["checked"])} fits differed on any test row.',
               f'None of the {len(enc["checked"])} fits', f'None of the {len(enc["checked"]) + 1} fits')
    plain_fact(S6, f'During the run, {enc["recomputed"]} jobs were computed twice, once by each of two worker pools, and their records '
                   f'were identical.', f'{enc["recomputed"]} jobs', f'{enc["recomputed"] + 1} jobs')
    # S6.4 (referee-panel revision of 2026-09-25, items B9, B3, B7, B14 and C1): the reproduction sample, one test set per dataset;
    # the encoder network's width over the swap's splits against the nested grid; the design's secondary contrasts, three of them in
    # Table S-learned-swap-controls; the post hoc signed-rank test of the primary contrast; and the fourth secondary contrast
    plain_rule(len({x for x, _ in SWAP_CHECK}) == len(SWAP_CHECK) and {x for x, _ in SWAP_CHECK} == set(check_sets),
               'the reproduction check used one test set of each of its datasets')
    plain_fact(S6, f'It used one test set of each of {listing([NAME[x] for x, _ in SWAP_CHECK], "and")}.', 'one test set of each',
               'two test sets of each')
    swap_widths = enc['widths']
    plain_rule(sorted(swap_widths['learned']) == sorted(swap_widths['random']) and len(swap_widths['learned']) == k17 * FACTS['n_outer'],
               'both network arms of the swap use the same width on every split')
    grid_v = sorted({c['V'] for c in HYBRID_GRID})
    plain_fact(S6, f"Its width therefore ranges from {min(swap_widths['learned'])} to {max(swap_widths['learned'])} over the splits, while "
                   f"the nested study tried only {listing(grid_v, 'and')}.", f"ranges from {min(swap_widths['learned'])} to",
               f"ranges from {2 * min(swap_widths['learned'])} to")
    m_secondary = re.search(r'Secondary, descriptive: (.+?)\. python', swap_header)
    secondary = m_secondary.group(1).split(', ') if m_secondary else []
    in_table = ['learned_random', 'random_fixed', 'learned_mlp']
    plain_rule(len(secondary) == len(enc['secondary']) and "learned - the nested hybrid's kNN readout" in secondary
               and all(key in enc['S'] for key in in_table), 'the design declares the secondary contrasts the run analysis holds')
    plain_fact(S6, f'The design also declared {words(len(secondary))} secondary contrasts, all descriptive, each with its own Holm adjustment '
                   f'across the {words(k17)} datasets. {words(len(in_table)).capitalize()} compare the arms with each other and with the MLP',
               f'declared {words(len(secondary))} secondary', f'declared {words(len(secondary) + 1)} secondary')
    plain_fact(S6, f'A post hoc Wilcoxon signed-rank test over the {k17} mean differences, one-sided as in '
                   f'Table~\\ref{{tab:s-signed-rank}}, gives $p={pval(swap_rank["p"])}$.', f'$p={pval(swap_rank["p"])}$',
               f'$p={float(pval(swap_rank["p"])) - 0.001:.3f}$')
    plain_fact(S6, f'In the fourth secondary contrast, ArrowFlow with the trained encoder has the higher mean accuracy on '
                   f'{counted(len(hk["higher"]), k17, "datasets")}, by {pct(hk["mean"], 2)} points on average, and no difference is '
                   f'significant.', f'on {len(hk["higher"])} of', f'on {len(hk["higher"]) - 1} of', f'by {pct(hk["mean"], 2)} points',
               f'by {pct(hk["mean"] + 0.0001, 2)} points')

    # ------------------------------------------------------------------ S5.1: the two runs of the learned encoder (items H9 and B6)
    S5 = 'supplement_sections/S5_reproducibility.tex'
    pools = enc['pools']
    plain_rule(pools['nested'] == pools['swap'], 'both modules run the same default pool')
    plain_fact(S5, f"They ran with single-thread workers, {pools['nested']} at once in the nested study. The encoder swap added a second "
                   f"pool of {pools['second']} workers, {pools['swap'] + pools['second']} in all.",
               f"{pools['nested']} at once", f"{pools['nested'] + 2} at once", f"{pools['swap'] + pools['second']} in all",
               f"{pools['swap'] + pools['second'] + 2} in all")
    committed = {w: git_time_utc(LEARNED_FREEZE[w]) for w in ('nested', 'swap')}
    plain_rule(committed['nested'] < committed['swap'], 'the nested design was committed before the swap design')
    plain_fact(S5, f"The nested study's design was committed on {committed['nested'][:10]} at {committed['nested'][11:]} UTC and the "
                   f"encoder swap's on {committed['swap'][:10]} at {committed['swap'][11:]} UTC (Table~\\ref{{tab:s-protocols}}).",
               f"at {committed['nested'][11:]} UTC", 'at 00:00 UTC')
    # the caption's sentence, repeated in S6.4, guarded there on its own (a phrase shared by two files survives a mutation of one)
    plain_fact(S6, f"With the trained encoder, ArrowFlow has the higher mean accuracy on {counted(len(prim['higher']), k17, 'datasets')}, "
                   f"significantly on {num_word(len(prim['up']))} and significantly lower on {num_word(len(prim['down']))}.",
               f"on {len(prim['higher'])} of", f"on {len(prim['higher']) - 1} of")
    plain_fact(S6, 'The rule fixed before the run therefore reads that the trained encoder improves ArrowFlow.', 'improves ArrowFlow',
               'worsens ArrowFlow')
    plain_fact(S6, f'Its mean error over the datasets falls from {pct(e_fixed)}\\% to {pct(e_learned)}\\%, against {pct(e_random)}\\% with '
                   f'the untrained network and {pct(e_mlp)}\\% for the MLP.', f'against {pct(e_random)}', f'against {pct(e_random + 0.001)}')
    plain_fact(S6, f'The trained encoder beats the untrained one on {len(lr["higher"])} of the {k17} datasets, significantly on '
                   f'{len(lr["up"])}. The untrained encoder is behind the fixed encoder on {len(below_fixed)}, significantly on '
                   f'{len(rf["down"])} (Table~\\ref{{tab:s-learned-swap-controls}}).', f'significantly on {len(rf["down"])} (',
               f'significantly on {len(rf["down"]) + 1} (')
    plain_fact(S6, f'ArrowFlow with the trained encoder is behind on {len(enc["trails"])} datasets and equal on {NAME[equal[0]]}, where '
                   f'both are perfect. It is ahead on {NAME[ahead[0]]}, where its error of {pct(1 - mean(swap[(b, "learned")]))}\\% is the lowest '
                   f'of all the tuned models.', f'behind on {len(enc["trails"])} datasets', f'behind on {len(enc["trails"]) - 1} datasets')
    return PLAIN_CLAIMS[start:]


E7_QUESTIONS = re.compile(r'The experiments ask (\w+) questions\.(.*?)\n\n', re.S)


def cells_ablation(t):
    """Table 4 against the complete record of its changes (leading value of every cell, bold removed)."""
    header, rows = table_rows(t['tab_ablation'])
    hs, rs = table_rows(t['tab_s_components_changes'])
    if header != hs:
        raise AssertionError('tab_ablation and tab_s_components_changes differ in their columns')
    main = {(r[0], h): unbold(c) for _, r in labelled(rows) for h, c in zip(header[1:], r[1:])}
    supp = {(r[0], h): lead(c) for _, r in labelled(rs) for h, c in zip(hs[1:], r[1:])}
    return main, supp


def cells_main_benchmark(t):
    """Table 2 against the complete error table (leading value of every cell); the gap column is checked where it is computed."""
    header, rows = table_rows(t['tab_main_benchmark'])
    strip = lambda label: re.sub(r'\$\^\{\\mathrm\{[cmn]+\}\}\$$', '', label)
    main = {(strip(r[0]), h): unbold(c) for _, r in labelled(rows) for h, c in zip(header[1:-1], r[1:-1])}
    hs, rs = table_rows(t['tab_s_errors'])
    supp = {(r[0], h): lead(c) for _, r in labelled(rs) for h, c in zip(hs[1:], r[1:])}
    return main, supp


def cells_training(t):
    """Table 3 against the combined ladder and the complete record of the training contrasts."""
    header, rows = table_rows(t['tab_training'])
    main = {(r[0], j): c for _, r in labelled(rows) for j, c in enumerate(r[1:], start=1)}
    hs, rs = table_rows(t['tab_s_ladder'])
    supp = {(r[0], j): c.split(' (')[0] for _, r in labelled(rs) for j, c in enumerate(r[1:], start=1)}
    hs, rs = table_rows(t['tab_s_training_complete'])
    col, place = {h: i for i, h in enumerate(hs)}, {CONTROL[UNTRAINED]: 6, CONTROL[INPUT_KNN]: 8}
    below = lambda cell: cell == '$<$0.001' or float(cell) < 0.05
    for _, r in labelled(rs):
        d, j = r[col['Dataset']], place[r[col['Control']]]
        supp[(d, j)] = f"{r[col['Diff. (pp)']]} {r[col[hs[5]]]}"
        supp[(d, j + 1)] = r[col['Holm $p$']] + ('$^\\ddagger$' if below(r[col[H34_HEAD]]) else '')
    return main, supp


def swap_columns(a, b):
    def change(cells, header):
        i, j = header.index(a), header.index(b)
        vi, vj = unbold(cells[i]), unbold(cells[j])
        cells[i], cells[j] = cells[i].replace(vi, vj), cells[j].replace(vj, vi)
        return cells
    return change


# ----------------------------------------------------------------------------- controlled experiments (Section 7.3, S3)
# The seven controlled experiments of 2026-09-14: matched motion controls, training diagnostics, mechanism analysis,
# fixed-configuration depth, aggregation, the duplicate-free rerun and the neighbor baselines. The rerun's tables left the
# supplement by the author's decision of 2026-09-23, so six of them are reported in S3.6 and S3.7.  Every table is read
# from a verified analysis output, checked against the sha256 and row count its analysis record seals (or, where an
# analysis writes no seal, against the records the same analysis JSON holds), and nothing numeric is typed here.
# The diagnostics tables come from experiments/make_revision/diagnostics_tables.py, which applies the three
# corrections the raw diagnostics summary does not make (iteration-0 points, tie grouping, checkpoint-zero views).
MOTION = '2026-09-14-motion'
DIAGNOSTICS_TABLES = '2026-09-14-training-diagnostics-tables'
MECHANISM = '2026-09-14-mechanism-analysis'
DEPTH_AGGREGATION = '2026-09-14-depth-aggregation'
DEDUP = '2026-09-14-dedup'
BASELINES = '2026-09-14-baselines'

MOTION_ARMS = ['frozen', 'permuted_alignment', 'random_direction']
SCRAMBLED = MOTION_ARMS[1:]
SINGLE_LAYER_ARMS = ['single_layer_first', 'single_layer_last']
ARM = {'views7': 'ArrowFlow', 'frozen': 'Frozen filters', 'permuted_alignment': 'Permuted alignment',
       'random_direction': 'Random directions', 'single_layer_first': 'First hidden layer only',
       'single_layer_last': 'Last hidden layer only'}
ARM_SHORT = {'frozen': 'Frozen', 'permuted_alignment': 'Permuted', 'random_direction': 'Random'}
DEPTH_ARMS = ['depth2', 'depth2_untrained_second', 'depth2_first_only']
DEPTH_LABEL = {'depth1': 'One hidden layer', 'depth2': 'Two hidden layers',
               'depth2_untrained_second': 'Untrained second layer', 'depth2_first_only': 'First layer trained only'}
DEPTH_CONTRAST = {'depth1_minus_depth2': 'depth2', 'depth1_minus_depth2_untrained_second': 'depth2_untrained_second',
                  'depth1_minus_depth2_first_only': 'depth2_first_only'}
AGG_LABEL = {'borda': 'Borda count', 'median': 'Plain median', 'median_mass_matched': 'Prior-scaled median'}
BASELINE_MODELS = ['lda_knn', 'pca_knn', 'nca_knn', 'kendall_svc']
BASELINE = {'lda_knn': 'LDA kNN', 'pca_knn': 'PCA kNN', 'nca_knn': 'NCA kNN', 'kendall_svc': 'Kendall SVC'}
RANK_PANEL = [KNN] + BRIDGE_MODELS[1:] + BASELINE_MODELS
RANK_HEAD = {**{m: MODEL[m] for m in [KNN, 'svc_rbf', 'mlp', 'numeric_knn', 'dummy']},
             'random_forest': two_lines('Random', 'forest'), 'gradient_boosting': two_lines('Gradient', 'boosting'),
             **BASELINE}
DEDUP_SOURCE = {'wine_quality_dedup': 'wine_quality', 'segment_dedup': 'segment'}
DEDUP_NAME = {'wine_quality_dedup': 'Wine quality', 'segment_dedup': 'Segment'}
DEDUP_MODELS = [KNN, UNTRAINED, INPUT_KNN, PROJECTED] + BRIDGE_MODELS[1:]
DEDUP_MODEL = {**MODEL, UNTRAINED: 'Untrained ArrowFlow', INPUT_KNN: 'Tuned input footrule kNN',
               PROJECTED: 'Numeric kNN, projected scores', 'arrowflow_full': 'Prototype readout'}
DEDUP_VARIANT = {'views1': 'One view', 'views3': 'Three views', 'no_checkpoint': 'Without validation checkpoint',
                 'no_augment': 'Without augmentation', 'prototype_readout': 'Prototype readout',
                 'untrained': 'Untrained networks', 'input_knn': 'Footrule kNN, same inputs'}   # the component variants at ArrowFlow's
                 # configuration, not the separately tuned controls (referee panel of 2026-09-23, item C2)
LADDER_RUNG = {'raw_numeric_knn': two_lines('Numeric kNN,', 'raw features'),
               'unsorted_projected_knn': two_lines('Numeric kNN,', 'projected scores'),
               'encoded_ranking_knn': two_lines('Tuned input', 'footrule kNN'),
               'untrained_arrowflow_knn': two_lines('Untrained', 'ArrowFlow'), 'arrowflow_knn': 'ArrowFlow'}
RUNG_PLAIN = {'raw_numeric_knn': 'numeric kNN on the raw features',
              'unsorted_projected_knn': 'numeric kNN on the projected scores',
              'encoded_ranking_knn': 'tuned input footrule kNN', 'untrained_arrowflow_knn': 'the untrained ArrowFlow',
              'arrowflow_knn': 'ArrowFlow'}

G5_CLAIMS, G5_MUTATIONS = [], []
E7 = 'sections/07_experiments.tex'
INTRO = 'sections/01_introduction.tex'
SUPP3 = 'supplement_sections/S3_results.tex'


def g5_fact(files, phrase, old, new, home=None):
    """Register a guarded phrase, rebuilt here from the run files, and the mutation case that must break it.

    ``files`` are the reader-visible files in which the phrase may stand: the rendered caption that carries it now
    and, where the prose will repeat it, the section file.  The mutation rewrites the phrase in ``home`` (the
    caption), so a drifting number is caught wherever the sentence lives."""
    files = (files,) if isinstance(files, str) else tuple(files)
    mutated = phrase.replace(old, new, 1)
    if mutated == phrase:
        raise AssertionError(f'guarded phrase: the mutation {old!r} -> {new!r} changes nothing in {phrase!r}')
    G5_CLAIMS.append((files, phrase))
    G5_MUTATIONS.append((home or files[0], phrase, mutated))
    return phrase


def g5_read(directory, name, record, key='outputs'):
    """One analysis output, checked against the sha256 and row count its analysis record seals."""
    path = Path(directory) / name
    frame = pd.read_csv(path)
    seal = record[key][name]
    if sha256_file(path) != seal['sha256'] or len(frame) != seal['rows']:
        raise AssertionError(f'{path}: differs from the seal in its analysis record')
    return frame


def g5_rows(frame, record, keys, columns, where):
    """A CSV an analysis writes without a seal, checked row for row against the records the same analysis JSON holds."""
    held = pd.DataFrame(record)
    if len(held) != len(frame) or sorted(held.columns) != sorted(frame.columns):
        raise AssertionError(f'{where}: {len(frame)} rows and {len(frame.columns)} columns on disk, '
                             f'{len(held)} and {len(held.columns)} in the analysis record')
    a, b = frame.sort_values(keys).reset_index(drop=True), held.sort_values(keys).reset_index(drop=True)
    for column in columns:
        for x, y in zip(a[column].tolist(), b[column].tolist()):
            if isinstance(x, str) or isinstance(y, str):
                if x != y:
                    raise AssertionError(f'{where}: column {column} differs from the analysis record')
            else:
                agrees(x, y, f'{where}, column {column}')
    return frame


def counted(values, total, noun):
    """'16 of the 17 datasets'."""
    return f'{int(values)} of the {total} {noun}'


def g5_status(record, *path):
    """The status line of an analysis, as its record states it."""
    node = record
    for key in path:
        node = node[key]
    return str(node)


def g5_dash(value, render):
    return '--' if blank(value) else render(value)


# --------------------------------------------------------------------------------- plain-language numbers
# The simplified main text (ledger, "RULING (simplicity ...)", item 5) states some measured quantities in plain words. Each
# such phrase is rebuilt here from the run data under one of the fixed rules below, and the rule is asserted, so a plain word
# cannot drift from the number it stands for. plain_fact registers the phrase with a mutation case, as g5_fact does; its
# claims and mutations join the prose check after every other claim has been built.
#   "about half"                                   0.45 <= x <= 0.55
#   "most", "mostly"                               x > 0.5, a strict majority
#   "nearly all", "nearly always", "nearly every", "almost every"            x >= 0.90
#   "almost never", "no more than N in 100"        x < N / 100, with N = 5 for "almost never"
#   "about one in N"                               N = round(1 / x), halves rounded up
#   "about K in five"                              K = round(5 x), halves rounded up
#   "almost a quarter"                             0.20 <= x < 0.25
#   "less than B"                                  |x| < B
#   "far more often"                               a >= 1.5 b and a - b >= 0.25
#   "could be chance"                              the 95% interval contains zero
#   "similar", "at least nine tenths as many"      smaller / larger >= 0.9
#   "less than a third"                            x < 1/3
#   "essentially unchanged"                        the values span less than half a point (0.005)
# The referee-panel revision of 2026-09-25 adds three, each asserted where it is used:
#   "barely above" (a balanced accuracy)           above the other value by less than two points (0 < x - y < 0.02)
#   "reproduces" (simulated shares)                the exact value lies within 0.005 of every simulated share
#   "not established" (a post hoc test)            the one-sided p value is at least 0.05
# and reads "rare" and "almost no input" by the rule of "almost never" (below 5 in 100)
PLAIN_CLAIMS, PLAIN_MUTATIONS = [], []
ORDINAL = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth'}
ORDINAL_INDEX = {v: k for k, v in ORDINAL.items()}


def plain_fact(files, phrase, old, new, *more):
    """Register a guarded phrase of the simplified text and the mutation case that must break it. Further (old, new) pairs
    in ``more`` register one more mutation case each, for a phrase that carries more than one guarded number (the
    referee-panel revision of 2026-09-25 added counts to several guarded sentences)."""
    files = (files,) if isinstance(files, str) else tuple(files)
    if len(more) % 2:
        raise AssertionError(f'guarded phrase: mutation pairs must come as (old, new), got {more!r}')
    PLAIN_CLAIMS.append((files, phrase))
    for a, b in [(old, new)] + list(zip(more[::2], more[1::2])):
        mutated = phrase.replace(a, b, 1)
        if mutated == phrase:
            raise AssertionError(f'guarded phrase: the mutation {a!r} -> {b!r} changes nothing in {phrase!r}')
        PLAIN_MUTATIONS.append((files[0], phrase, mutated))
    return phrase


def plain_rule(ok, what):
    if not ok:
        raise AssertionError(f'plain-language rule broken: {what}')


def round_half_up(v):
    return int(math.floor(v + 0.5))


def about_half(x, what):
    plain_rule(0.45 <= x <= 0.55, f'"about half" needs 0.45 <= x <= 0.55, got {x:.4f} ({what})')


def majority(x, what):
    plain_rule(x > 0.5, f'"most" needs more than half, got {x:.4f} ({what})')


def nearly_all(x, what):
    plain_rule(x >= 0.9, f'"nearly all" needs at least 0.9, got {x:.4f} ({what})')


def fewer_than_in_100(x, n, what):
    plain_rule(100 * x < n, f'"no more than {n} in 100" needs 100 x < {n}, got {100 * x:.3f} ({what})')


def one_in(x):
    """N of "about one in N"."""
    return round_half_up(1 / x)


def in_five(x):
    """K of "about K in five"."""
    return round_half_up(5 * x)


def almost_quarter(x, what):
    plain_rule(0.2 <= x < 0.25, f'"almost a quarter" needs 0.20 <= x < 0.25, got {x:.4f} ({what})')


def below(x, bound, what):
    plain_rule(abs(x) < bound, f'"less than {bound}" needs |x| < {bound}, got {x} ({what})')


def far_more(a, b, what):
    plain_rule(a >= 1.5 * b and a - b >= 0.25, f'"far more often" needs a >= 1.5 b and a - b >= 0.25, got {a:.4f}, {b:.4f} ({what})')


def could_be_chance(lo, hi, what):
    plain_rule(lo <= 0 <= hi, f'"could be chance" needs an interval that contains zero, got [{lo}, {hi}] ({what})')


def similar(a, b, what):
    plain_rule(min(a, b) / max(a, b) >= 0.9, f'"similar" needs a ratio of at least 0.9, got {a:.4f}, {b:.4f} ({what})')


def under_third(x, what):
    plain_rule(x < 1 / 3, f'"less than a third" needs x < 1/3, got {x:.4f} ({what})')


def unchanged(values, what):
    plain_rule(max(values) - min(values) < 0.005, f'"essentially unchanged" needs a span below 0.005, got {values} ({what})')


# --------------------------------------------------------------------------------- matched motion-signal controls
def render_motion(runs):
    """The matched motion controls (Section 7.3, S3): the three-arm primary family, neighborhood purity, the motion
    statistics that say how much each scrambling arm actually changed, and the two single-layer arms."""
    d = runs / MOTION / 'analysis'
    record = read_json(d / 'motion_analysis.json')
    fam = g5_read(d, 'motion_primary_family.csv', record)
    purity = g5_read(d, 'motion_purity.csv', record)
    stats = g5_read(d, 'motion_motion_statistics.csv', record)
    single = g5_read(d, 'motion_single_layer.csv', record)
    subset = g5_read(d, 'motion_named_subset.csv', record)
    declaration = record['analysis_declaration']
    n_outer, conf = FACTS['n_outer'], FACTS['confidence']
    if sorted(record['provenance']['verification']['arms']) != sorted(['views7'] + MOTION_ARMS + SINGLE_LAYER_ARMS) or \
            sorted(fam['arm'].unique()) != sorted(MOTION_ARMS) or sorted(fam['dataset'].unique()) != sorted(COMBINED) or \
            set(fam['n_folds']) != {n_outer} or set(fam['df']) != {FACTS['df']} or set(fam['test_train_ratio']) != {FACTS['ratio']}:
        raise AssertionError('motion controls: the primary family is not the three arms over the seventeen datasets')
    if declaration['primary_family']['contrast'] != 'views7 minus the control arm' or \
            'Holm across the seventeen datasets within each arm' not in declaration['primary_family']['adjustment']:
        raise AssertionError('motion controls: the declared family is not Holm-adjusted within each arm')
    if any(v['matching_fold_seeds'] != v['total_fold_seeds'] for v in record['views7_reproduces_reference'].values()):
        raise AssertionError('motion controls: the unmodified arm does not reproduce the reference run everywhere')
    checks = {name: sorted(k for k, v in counts.items() if k != 'None' and v) for name, counts in record['job_checks'].items()}
    if any(passed != ['True'] for passed in checks.values()):
        raise AssertionError('motion controls: a job check did not pass on every job')
    table = {(r['arm'], r['dataset']): r for _, r in fam.iterrows()}
    level = {(r['arm'], r['dataset']): r['mean'] for _, r in purity[purity['rows'] == 'outer_test'].iterrows()}
    train_level = {(r['arm'], r['dataset']): r['mean'] for _, r in purity[purity['rows'] == 'training'].iterrows()}
    arrow = {d: table[(MOTION_ARMS[0], d)]['mean_accuracy_a'] for d in COMBINED}
    for arm in MOTION_ARMS:
        if any(not math.isclose(table[(arm, d)]['mean_accuracy_a'], arrow[d], rel_tol=0, abs_tol=1e-12) for d in COMBINED):
            raise AssertionError('motion controls: the unmodified arm differs between the three contrasts')
    worse = {arm: sum(level[(arm, d)] < level[('frozen', d)] for d in COMBINED) for arm in SCRAMBLED}
    less_accurate = {arm: sum(table[(arm, d)]['mean_accuracy_b'] < table[('frozen', d)]['mean_accuracy_b'] for d in COMBINED)
                     for arm in SCRAMBLED}
    holm = {arm: [d for d in COMBINED if table[(arm, d)]['holm_p_approximate'] < 0.05] for arm in MOTION_ARMS}
    named = sorted({d for arm in MOTION_ARMS for d in holm[arm]}, key=COMBINED.index)
    diff = lambda arm, d: signed(table[(arm, d)]['mean_difference'], 2)
    span = lambda arm, d: interval(table[(arm, d)]['ci_low'], table[(arm, d)]['ci_high'], 2)
    adjusted = lambda arm, d: pval(table[(arm, d)]['holm_p_approximate'])

    # the complete three-arm family
    head = ['Dataset'] + [two_lines(ARM_SHORT[a], t) for a in MOTION_ARMS for t in ('Diff. (pp)', f'{conf}\\% interval', 'Holm $p$')]
    rows = [[NAME[d]] + [f(a, d) for a in MOTION_ARMS for f in (diff, span, adjusted)] for d in COMBINED]
    scrambled_fact = g5_fact('tables/tab_s_motion_family.tex',
                             f'The permuted-alignment arm is less accurate than the frozen arm on '
                             f'{counted(less_accurate["permuted_alignment"], len(COMBINED), "datasets")} and the '
                             f'random-direction arm on {counted(less_accurate["random_direction"], len(COMBINED), "")}'.rstrip() + '.',
                             counted(less_accurate['permuted_alignment'], len(COMBINED), 'datasets'),
                             counted(less_accurate['permuted_alignment'] - 1, len(COMBINED), 'datasets'))
    holm_fact = g5_fact('tables/tab_s_motion_family.tex',
                        f'Holm adjustment leaves {len(holm[MOTION_ARMS[0]])} datasets significant in the '
                        f'{ARM[MOTION_ARMS[0]].lower()} arm, '
                        + listing([f'{len(holm[a])} in the {ARM[a].lower()} arm' for a in MOTION_ARMS[1:]], 'and') + '.',
                        f'leaves {len(holm[MOTION_ARMS[0]])} datasets', f'leaves {len(holm[MOTION_ARMS[0]]) + 1} datasets')
    # the frozen arm keeps every hidden filter at its seeded initial order and the readout reads only the hidden rankings, so it
    # predicts exactly as the untrained networks of the component ablation (checked cell by cell in simplify_claims); both
    # component ablations had written their results before this protocol was frozen (referee panel of 2026-09-23, item H1)
    protocol = runs / MOTION / 'run' / 'protocol.json'
    if sha256_file(protocol) != sha256_file(PROTOCOLS_G5 / 'motion_controls.json'):
        raise AssertionError('motion controls: the run protocol differs from the frozen protocol file')
    frozen_at = read_json(protocol)['frozen_at_utc'][:19]
    ablation_ends = [re.search(rf'^{tag} end (\S+) OK$', (runs / log).read_text(), re.M) for tag, log in
                     (('task20', 'task20-run.log'), ('task24', 'task24-run.log'))]
    if not all(m and m.group(1)[:19] < frozen_at for m in ablation_ends):
        raise AssertionError('the text says both component ablations had finished before the motion protocol was frozen')
    write_table('tab_s_motion_family',
                '\\textbf{Matched motion-signal controls: the primary family.} ArrowFlow minus the control arm, accuracy, in '
                f'percentage points, on the {words(len(COMBINED))} datasets. Fitting seeds are averaged within each outer fold; '
                f'intervals are {interval_note()}. Positive favors ArrowFlow. Holm adjusts across the '
                f'{words(len(COMBINED))} datasets within each arm, the three arms separately, as the run protocol declared '
                'before its own outer folds were scored. The frozen arm predicts exactly as the untrained networks of the '
                'component ablation (Table~\\ref{tab:s-components-changes}), whose results existed before this protocol was '
                f'frozen, so only the two scrambling arms were scored blind. {scrambled_fact} {holm_fact} '
                'kNN: nearest-neighbor classifier; HCV: hepatitis C virus.',
                'tab:s-motion-family', head, rows, 'l' + 'rrr' * len(MOTION_ARMS), size='\\scriptsize', wide=True,
                stack=2 * len(MOTION_ARMS))

    # the main-text extract: every dataset significant in at least one arm
    main_head = ['Dataset'] + [two_lines(ARM_SHORT[a], t) for a in MOTION_ARMS for t in ('Diff. (pp)', 'Holm $p$')]
    main_rows = [[NAME[d]] + [f(a, d) for a in MOTION_ARMS for f in (diff, adjusted)] for d in named]
    headline_fact = g5_fact(INTRO,
                            'This happens on '
                            f'{counted(less_accurate["permuted_alignment"], len(COMBINED), "datasets")} when each motion reaches '
                            f'the wrong filter, and on {counted(less_accurate["random_direction"], len(COMBINED), "")}'.rstrip()
                            + ' when its direction is random',
                            counted(less_accurate['permuted_alignment'], len(COMBINED), 'datasets'),
                            counted(less_accurate['permuted_alignment'] - 1, len(COMBINED), 'datasets'))
    write_table('tab_motion',
                '\\textbf{Matched motion-signal controls.} ArrowFlow minus each control arm, accuracy, in percentage points, on '
                f'the {words(len(named))} datasets where at least one arm is significant after Holm adjustment. The frozen arm '
                'never moves its hidden filters; the two scrambling arms keep the accepted slots, the vote magnitudes and the '
                'eligibility gate and change only which filter a motion reaches, or which way it points. Holm adjusts across the '
                f'{words(len(COMBINED))} datasets within each arm. Positive favors ArrowFlow. Section~S3 gives the complete '
                f'family with intervals. kNN: nearest-neighbor classifier.',
                'tab:motion', main_head, main_rows, 'l' + 'rr' * len(MOTION_ARMS), size='\\footnotesize')

    # neighborhood purity
    purity_arms = ['views7'] + MOTION_ARMS
    phead = ['Dataset'] + [two_lines(ARM.get(a, a).split()[0] if a != 'views7' else 'ArrowFlow', t)
                           for a in purity_arms for t in ('Test', 'Training')]
    prows = [[NAME[d]] + [pct(src[(a, d)], 2) for a in purity_arms for src in (level, train_level)] for d in COMBINED]
    purity_fact = g5_fact('tables/tab_s_motion_purity.tex',
                          'On balance-scale the purity of the outer test rows is '
                          f"{pct(level[('views7', 'balance_scale')], 2)} percent for ArrowFlow, "
                          + listing([f"{pct(level[(a, 'balance_scale')], 2)} for the {ARM[a].lower()}" for a in MOTION_ARMS],
                                    'and') + '.',
                          pct(level[('views7', 'balance_scale')], 2), pct(level[('views7', 'balance_scale')] + 0.001, 2))
    write_table('tab_s_motion_purity',
                '\\textbf{Matched motion-signal controls: neighborhood purity.} Same-class purity of the last hidden ranking, '
                'in percent. For each row it is the share of its nearest stored training rankings that carry its class, at the '
                'neighbor count the view selected. Rows are averaged first, then the seven views, then the outer folds. '
                'Training rows drop their own stored ranking and are subsampled above 512 rows. Descriptive: no test and no '
                'interval. '
                f'{purity_fact} HCV: hepatitis C virus.',
                'tab:s-motion-purity', phead, prows, 'l' + 'rr' * len(purity_arms), size='\\scriptsize', wide=True)

    # what each scrambling arm changed. The analysis output counts every selected slot, the zero votes of the examples the
    # gate rejects included, so its attraction share tracks the error rate and not the balance of signs; seven referees read
    # it as a repulsion share (referee panel of 2026-09-23, items C1 and X1). The nonzero votes and their signs are summed from
    # the per-fold result records of the run, which must reproduce every sealed count of the analysis output first.
    if float(stats['mass_deviation'].abs().max()) != 0.0:
        raise AssertionError('motion controls: a scrambling arm changed the accepted vote mass')
    keyed = {(r['arm'], r['dataset'], int(r['layer'])): r for _, r in stats.iterrows()}
    layers = sorted({(r['dataset'], int(r['layer'])) for _, r in stats.iterrows()}, key=lambda k: (COMBINED.index(k[0]), k[1]))
    counted_arms = ['views7', 'permuted_alignment', 'random_direction']
    votes, result_files = defaultdict(Counter), sorted((runs / MOTION / 'run' / 'results').glob('*.json'))
    if len(result_files) != len(COMBINED) * n_outer:
        raise AssertionError('motion controls: the run does not hold one result record per outer fold')
    for path in result_files:
        rec = read_json(path)
        for arm in counted_arms:
            for layer, e in rec['checks']['mass_matched']['by_arm'][arm].items():
                votes[(arm, rec['identity']['dataset_id'], int(layer))].update(
                    {k: e[k] for k in ('accepted_slots', 'changed_slots', 'effective_before', 'effective_after',
                                       'positive_before', 'positive_after', 'mass_before', 'mass_after')})
    for arm in counted_arms:
        for dataset, layer in layers:
            v, r = votes[(arm, dataset, layer)], keyed[(arm, dataset, layer)]
            if (v['accepted_slots'], v['changed_slots']) != (r['accepted_slots'], r['changed_slots']) or \
                    abs(v['positive_after'] / v['accepted_slots'] - r['positive_share_after']) > 1e-12 or \
                    abs(v['mass_before'] - r['mass_before']) > 1e-6 * max(1.0, r['mass_before']) or \
                    v['effective_before'] != v['effective_after'] or \
                    (arm == 'random_direction' and v['positive_before'] != v['positive_after']):
                raise AssertionError(f'motion controls: the result records of {arm}/{dataset}/{layer} differ from the sealed '
                                     'motion statistics, or a scrambling arm created, destroyed or re-signed a vote')
    nonzero = lambda arm, dataset, layer: votes[(arm, dataset, layer)]['effective_before'] / votes[(arm, dataset, layer)]['accepted_slots']
    repulsion = {(d, l): 1 - votes[('views7', d, l)]['positive_before'] / votes[('views7', d, l)]['effective_before'] for d, l in layers}
    reversed_ = {(d, l): votes[('random_direction', d, l)]['changed_slots'] / votes[('random_direction', d, l)]['effective_before']
                 for d, l in layers}
    attract_r = {(d, l): votes[('random_direction', d, l)]['positive_before'] / votes[('random_direction', d, l)]['effective_before']
                 for d, l in layers}
    for key in layers:
        about_half(repulsion[key], f'repulsions among the nonzero votes of ArrowFlow, {key}')
        about_half(reversed_[key], f'nonzero votes reversed by the random-direction arm, {key}')
        if abs(reversed_[key] - 2 * attract_r[key] * (1 - attract_r[key])) > 0.01:
            raise AssertionError(f'the caption says a sign permutation reverses about 2q(1-q) of the nonzero votes, {key}')
    shead = ['Dataset', 'Layer', two_lines('Selected', 'slots (M)'), two_lines('ArrowFlow:', 'nonzero (\\%)'),
             two_lines('ArrowFlow:', 'repulsions (\\%)'), two_lines('Permuted:', 'moved (\\%)'), two_lines('Random:', 'nonzero (\\%)'),
             two_lines('Random:', 'reversed (\\%)')]
    srows = []
    for dataset, layer in layers:
        p = keyed[('permuted_alignment', dataset, layer)]
        srows.append([NAME[dataset], str(layer + 1), f"{p['accepted_slots'] / 1e6:.1f}", pct(nonzero('views7', dataset, layer), 2),
                      pct(repulsion[(dataset, layer)], 2), pct(p['changed_share'], 2), pct(nonzero('random_direction', dataset, layer), 2),
                      pct(reversed_[(dataset, layer)], 2)])
    lo_rep, hi_rep = min(repulsion.values()), max(repulsion.values())
    lo_rev, hi_rev = min(reversed_.values()), max(reversed_.values())
    halves = g5_fact('tables/tab_s_motion_statistics.tex',
                     f'Among ArrowFlow\'s nonzero votes, repulsions make up about half, {pct(lo_rep, 2)} to {pct(hi_rep, 2)} '
                     f'percent, on every dataset and layer, and the random-direction arm reverses {pct(lo_rev, 2)} to '
                     f'{pct(hi_rev, 2)} percent of its nonzero votes.', f'about half, {pct(lo_rep, 2)} to',
                     f'about half, {pct(lo_rep - 0.02, 2)} to')
    bs = ('balance_scale', 0)
    strength = g5_fact('tables/tab_s_motion_statistics.tex',
                       'On balance-scale permuted alignment moves '
                       f"{pct(keyed[('permuted_alignment', *bs)]['changed_share'], 2)} percent of the selected slots to another "
                       f"filter, and random directions reverses {pct(reversed_[bs], 2)} percent of its nonzero votes, which are "
                       f"{pct(nonzero('random_direction', *bs), 2)} percent of the selected slots, so the two arms are not equally "
                       'strong.', pct(keyed[('permuted_alignment', *bs)]['changed_share'], 2), '98.23')
    write_table('tab_s_motion_statistics',
                '\\textbf{Matched motion-signal controls: what each arm changed.} Counts over every fold, seed and view, per '
                'dataset and hidden layer; layer 1 is the first hidden layer. A slot is one filter that one example selects in '
                'one batch: every example of every batch selects $\\lceil\\rho N_\\ell\\rceil$ filters of each hidden layer, '
                'and an example that the gate rejects gives its slots zero votes. Selected slots: their number in millions, the '
                'same in every arm. Nonzero: the share of the selected slots that carry a nonzero vote, in the arm\'s own run. '
                'Repulsions: the share of ArrowFlow\'s nonzero votes that push a filter away from the batch input rather than '
                'pull it toward it. Moved: the share of all selected slots, zero votes included, that permuted alignment '
                'delivers to another filter. Reversed: the share of the random-direction arm\'s nonzero votes whose direction '
                'it reverses; permuting the signs of votes of which a share $q$ attract reverses a share $2q(1-q)$ on average. '
                'Both arms preserve the slots, the vote magnitudes and the vote mass of every batch exactly. '
                f'{halves} {strength}',
                'tab:s-motion-statistics', shead, srows, 'llrrrrrr', size='\\scriptsize',
                block_rules=tuple(i for i, (dataset, layer) in enumerate(layers) if layer == 0 and i))

    # the post hoc comparison of each scramble with frozen filters (referee panel of 2026-09-23, item P): the counts of
    # Section 7.5 are descriptive, and an exact one-sided signed-rank test across the datasets, not declared in the protocol,
    # asks whether a scramble leaves the network less accurate than frozen filters
    accuracy = {(arm, d): float(table[(arm, d)]['mean_accuracy_b']) for arm in MOTION_ARMS for d in COMBINED}
    scramble = {arm: [accuracy[(arm, d)] - accuracy[('frozen', d)] for d in COMBINED] for arm in SCRAMBLED}
    tests = {arm: signed_rank_less(scramble[arm]) for arm in SCRAMBLED}
    below_frozen = {arm: sum(x < 0 for x in scramble[arm]) for arm in SCRAMBLED}
    if below_frozen != less_accurate:
        raise AssertionError('motion controls: the scrambled-minus-frozen signs differ from the counts of Section 7.5')
    crows = [[NAME[d]] + [signed(scramble[arm][i], 2) for arm in SCRAMBLED] for i, d in enumerate(COMBINED)]
    test_fact = g5_fact('tables/tab_s_motion_scramble.tex',
                        'Post hoc, an exact one-sided Wilcoxon signed-rank test across the '
                        f'{words(len(COMBINED))} datasets gives $p={scientific(tests["permuted_alignment"]["p"])}$ for permuted '
                        f'alignment, below the frozen arm on {counted(below_frozen["permuted_alignment"], len(COMBINED), "datasets")}, '
                        f'and $p={scientific(tests["random_direction"]["p"])}$ for random directions, below it on '
                        f'{counted(below_frozen["random_direction"], len(COMBINED), "")}'.rstrip() + '.',
                        f'$p={scientific(tests["permuted_alignment"]["p"])}$', '$p=2.4\\times10^{-5}$')
    write_table('tab_s_motion_scramble',
                '\\textbf{Matched motion-signal controls: each scramble against frozen filters (post hoc).} The mean accuracy '
                'of each scrambling arm minus that of the frozen arm, in percentage points, per dataset; negative means that the '
                'scrambled signal left the network less accurate than frozen filters. The values are differences of the arm '
                'accuracies of Table~\\ref{tab:s-motion-family}, so they carry no interval. The protocol declared ArrowFlow '
                f'minus each arm, not this comparison. The test ranks the {words(len(COMBINED))} absolute differences, and its $p$ '
                f'value is the share of the $2^{{{len(COMBINED)}}}$ sign assignments whose sum of positive ranks is at most the '
                f'observed one. {test_fact} HCV: hepatitis C virus.',
                'tab:s-motion-scramble', ['Dataset', two_lines('Permuted', '$-$ frozen (pp)'), two_lines('Random', '$-$ frozen (pp)')],
                crows, 'lrr', size='\\scriptsize')

    # the two single-layer arms: every row covers only the folds whose selection has two hidden layers, so its interval and
    # p value use n - 1 degrees of freedom (referee panel of 2026-09-23, item G), and a row whose fold differences are all
    # equal has no interval and no test
    kept = {(r['arm'], r['dataset']): r for _, r in single.iterrows()}
    flat = check_subset_intervals(single, 'motion controls, single-layer arms', 'p_unadjusted')
    flat_keys = {(r['arm'], r['dataset']) for r in flat}
    fold = {d: int(kept[(SINGLE_LAYER_ARMS[0], d)]['two_hidden_layer_folds']) for d in COMBINED}
    lhead = ['Dataset', two_lines('Two-layer', 'folds')] + [two_lines('First only' if a.endswith('first') else 'Last only', t)
                                                            for a in SINGLE_LAYER_ARMS for t in ('Diff. (pp)', f'{conf}\\% interval', '$p$')]
    lrows = []
    for dataset in COMBINED:
        cells = [NAME[dataset], f'{fold[dataset]} of {n_outer}']
        for arm in SINGLE_LAYER_ARMS:
            r = kept[(arm, dataset)]
            cells += [g5_dash(r['mean_difference'], lambda v: signed(v, 2)),
                      '--' if blank(r['ci_low']) or (arm, dataset) in flat_keys else interval(r['ci_low'], r['ci_high'], 2),
                      '--' if (arm, dataset) in flat_keys else g5_dash(r['p_unadjusted'], pval)]
        lrows.append(cells)
    none_selected = sum(1 for d in COMBINED if not fold[d])
    single_fact = g5_fact('tables/tab_s_motion_single_layer.tex',
                          f'{of_total(none_selected, words(len(COMBINED)), "datasets", "selected", "selected")} two hidden '
                          f'layers on no outer fold, so their rows are empty.',
                          of_total(none_selected, words(len(COMBINED)), 'datasets', 'selected', 'selected'),
                          of_total(none_selected + 1, words(len(COMBINED)), 'datasets', 'selected', 'selected'))
    df_fact = g5_fact('tables/tab_s_motion_single_layer.tex', SUBSET_DF, '$n-1$ degrees', f'{FACTS["df"]} degrees')
    flat_note = ('' if not flat else ' Every fold difference is zero in ' + listing(
        [f'the {"first-only" if r["arm"].endswith("first") else "last-only"} row of {NAME[r["dataset"]]}' for r in flat], 'and')
        + ', which therefore has no interval and no test.')
    caption = ('\\textbf{Matched motion-signal controls: one hidden layer trained.} ArrowFlow minus an arm in which only the '
               'first, or only the last, hidden layer applies its accumulated motion, accuracy, in percentage points. At a '
               'one-hidden-layer selection the arm is ArrowFlow by construction, so each row uses only the outer folds whose '
               f'selection has two hidden layers; the second column gives their number $n$. Intervals are '
               f'{interval_note(subset=True)}; the $p$ values use the same $t$ distribution.{flat_note} Descriptive: not part of the primary '
               f'family and not Holm-adjusted. {single_fact}')
    if df_fact not in caption or f'{FACTS["df"]} degrees of freedom' in caption:
        raise AssertionError('the single-layer table must state the n - 1 rule its rows use, and no full-fold count')
    write_table('tab_s_motion_single_layer', caption, 'tab:s-motion-single-layer', lhead, lrows, 'llrrrrrr', size='\\scriptsize',
                wide=True)
    return dict(table=table, holm=holm, named=named, worse=worse, less_accurate=less_accurate, purity=level,
                subset=subset, declaration=declaration, arrow=arrow, tests=tests, scramble=scramble, repulsion=repulsion,
                reversed=reversed_, attract=attract_r, flat=flat, stats=stats,
                nonzero={key: nonzero('views7', *key) for key in layers})


# --------------------------------------------------------------------------------- training diagnostics
def layer_name(layer):
    """A diagnostics layer key as printed: the main text numbers hidden layers from one (referee panel item C5)."""
    m = re.fullmatch(r'hidden_(\d+)', str(layer))
    if m:
        return f'Hidden {int(m.group(1)) + 1}'
    if layer != 'output':
        raise AssertionError(f'diagnostics tables: an unknown layer key {layer!r}')
    return 'Output'


def render_diagnostics(runs, combined):
    """The training diagnostics (Section 7.3, S3), read from the report-ready tables of
    experiments/make_revision/diagnostics_tables.py: checkpoints, displacement at the checkpoint, response ties by
    permutation length against a random-filter reference, and the sensitivity of the readout to filter-ID relabeling."""
    d = runs / DIAGNOSTICS_TABLES
    record = read_json(d / 'diagnostics_tables.json')
    module = REPO / record['module']['path']
    if record['synthetic_smoke_only'] is not False or record['allow_smoke'] is not False or \
            sha256_file(module) != record['module']['sha256']:
        raise AssertionError('diagnostics tables: a smoke run, or not the committed post-processing module')
    checks = record['checks']
    if checks['checkpoint_zero_views_with_zero_checkpoint_displacement'] != 0 or \
            checks['folds_with_checkpoint_accuracy_equal_to_the_reference'] != len(COMBINED) * FACTS['n_outer'] or \
            checks['views_with_checkpoint_accuracy_equal_to_the_ablation_votes'] != len(COMBINED) * FACTS['n_outer'] * FACTS['views']:
        raise AssertionError('diagnostics tables: a verification count does not cover every fold and view')
    t1 = g5_read(d, 't1_checkpoints.csv', record, 'tables')
    t2 = g5_read(d, 't2_displacement.csv', record, 'tables')
    t3 = g5_read(d, 't3_ties.csv', record, 'tables')
    t4 = g5_read(d, 't4_relabel.csv', record, 'tables')
    views = FACTS['n_outer'] * FACTS['views']
    pooled1 = t1[t1['scope'] == 'pooled'].iloc[0]
    per = t1[t1['scope'] == 'dataset'].set_index('dataset_id')
    if sorted(per.index) != sorted(COMBINED) or int(pooled1['n_views']) != len(COMBINED) * views or \
            set(per['n_views']) != {views} or set(t1['iterations_min']) != set(t1['iterations_max']):
        raise AssertionError('diagnostics tables: the checkpoint table does not cover every view at one update count')
    updates = int(pooled1['iterations_max'])

    # checkpoints
    rows = [[NAME[x], f"{int(per.loc[x, 'n_validation_samples_min'])}"
             + ('' if per.loc[x, 'n_validation_samples_min'] == per.loc[x, 'n_validation_samples_max']
                else f"--{int(per.loc[x, 'n_validation_samples_max'])}"),
             str(int(per.loc[x, 'n_views_checkpoint_0'])), f"{per.loc[x, 'checkpoint_median']:.0f}",
             f"{per.loc[x, 'checkpoint_q25']:.0f}", f"{per.loc[x, 'checkpoint_q75']:.0f}"] for x in COMBINED]
    rows.append(['All ' + words(len(COMBINED)) + ' together',
                 f"{int(pooled1['n_validation_samples_min'])}--{int(pooled1['n_validation_samples_max'])}",
                 str(int(pooled1['n_views_checkpoint_0'])), f"{pooled1['checkpoint_median']:.0f}",
                 f"{pooled1['checkpoint_q25']:.0f}", f"{pooled1['checkpoint_q75']:.0f}"])
    slowest, fastest = per['checkpoint_median'].idxmax(), per['checkpoint_median'].idxmin()
    checkpoint_fact = g5_fact('tables/tab_s_diag_checkpoints.tex',
                              f"No view returned its initial filters: {num_word(int(pooled1['n_views_checkpoint_0']))} of the "
                              f"{int(pooled1['n_views']):,} views has checkpoint zero, and the median checkpoint runs from "
                              f"{per.loc[fastest, 'checkpoint_median']:.0f} on {NAME[fastest]} to "
                              f"{per.loc[slowest, 'checkpoint_median']:.0f} on {NAME[slowest]} of {updates} updates.",
                              f"{per.loc[slowest, 'checkpoint_median']:.0f} on {NAME[slowest]}",
                              f"{per.loc[slowest, 'checkpoint_median'] + 1:.0f} on {NAME[slowest]}")
    write_table('tab_s_diag_checkpoints',
                '\\textbf{Training diagnostics: checkpoints.} Every dataset contributes '
                f'{FACTS["n_outer"]} outer folds $\\times$ {words(FACTS["views"])} views $= {views}$ views, each trained for '
                f'{updates} updates. The checkpoint is the last update whose error on the core\'s held-out validation rows is '
                'strictly below the running minimum before it; zero means the core returned the initial filters. Quartiles are '
                'over all views of the group. Descriptive: one fitting seed, nothing selected. '
                f'{checkpoint_fact} HCV: hepatitis C virus.',
                'tab:s-diag-checkpoints', ['Dataset', two_lines('Validation', 'rows'), two_lines('Views with', 'checkpoint 0'),
                                           two_lines('Median', 'checkpoint'), 'Q1', 'Q3'],
                rows, 'lrrrrr', size='\\scriptsize', block_rules=(len(COMBINED),))

    # displacement at the checkpoint
    p2 = t2[t2['scope'] == 'pooled']
    if int(p2[p2['layer'] == 'output']['n_views'].sum()) != len(COMBINED) * views or \
            float(p2['share_checkpoint_0'].max()) != 0.0:
        raise AssertionError('diagnostics tables: the displacement table does not cover every view')
    drows = [[widths_label(json.loads(r['widths'])), layer_name(r['layer']), str(int(r['n'])),
              str(int(r['n_views'])), f"{r['displacement_mean_later']:.3f}", f"{r['displacement_sd_later']:.3f}",
              pct(r['changed_share_mean_later'], 1), f"{r['random_displacement_reference']:.3f}"]
             for _, r in p2.iterrows()]
    hidden = p2[p2['layer'] != 'output']
    first = hidden[hidden['widths'] == '[128]']
    displacement_fact = g5_fact('tables/tab_s_diag_displacement.tex',
                                'In the single-hidden-layer networks the mean displacement at the checkpoint runs from '
                                f"{first['displacement_mean_later'].min():.3f} to {first['displacement_mean_later'].max():.3f}, "
                                f"against {hidden['random_displacement_reference'].min():.3f} to "
                                f"{hidden['random_displacement_reference'].max():.3f} for two independent uniform "
                                'permutations.', f"{first['displacement_mean_later'].max():.3f}, against",
                                f"{first['displacement_mean_later'].max() + 0.001:.3f}, against")
    write_table('tab_s_diag_displacement',
                '\\textbf{Training diagnostics: displacement at the checkpoint.} Per view, the mean over the filters of a layer '
                'of the footrule distance between the checkpoint filter and its initial filter, divided by the largest footrule '
                '$\\lfloor V^2/2 \\rfloor$, where $V$ is the number of items the layer orders. Hidden layers are numbered from one. '
                'Rows pool every view of every dataset with that architecture and $V$. The reference '
                '$\\bigl((V^2-1)/3\\bigr)/\\lfloor V^2/2 \\rfloor$ is the expected normalized footrule between two independent uniform '
                f'permutations. Descriptive. {displacement_fact}',
                'tab:s-diag-displacement',
                ['Widths', 'Layer', '$V$', 'Views', 'Mean', 'SD', two_lines('Filters', 'changed (\\%)'),
                 two_lines('Random', 'reference')],
                drows, 'llrrrrrr', size='\\scriptsize')

    # response ties by permutation length
    p3 = t3[t3['scope'] == 'pooled']
    hid, out = p3[p3['layer'] != 'output'], p3[p3['layer'] == 'output']
    trows = [[widths_label(json.loads(r['widths'])), layer_name(r['layer']), str(int(r['n'])),
              str(int(r['n_filters'])), f"{r['tied_response_share_initial']:.3f}", f"{r['tied_response_share_checkpoint']:.3f}",
              f"{r['random_tied_response_share']:.3f}", f"{r['distinct_response_ratio_initial']:.3f}",
              f"{r['distinct_response_ratio_checkpoint']:.3f}", f"{r['random_distinct_response_ratio']:.3f}",
              f"{int(r['distinct_values_bound']):,}"] for _, r in hid.iterrows()]
    row32 = hid[(hid['widths'] == '[128]') & (hid['n'] == 32)].iloc[0]
    ties_fact = g5_fact('tables/tab_s_diag_ties.tex',
                        'At the first hidden layer of a $[128]$ network ordering 32 items the tied-response share is '
                        f"{row32['tied_response_share_initial']:.3f} at the initial filters, "
                        f"{row32['tied_response_share_checkpoint']:.3f} at the checkpoint and "
                        f"{row32['random_tied_response_share']:.3f} for uniform random filters.",
                        f"{row32['tied_response_share_checkpoint']:.3f} at the checkpoint",
                        f"{row32['tied_response_share_checkpoint'] + 0.001:.3f} at the checkpoint")
    write_table('tab_s_diag_ties',
                '\\textbf{Training diagnostics: response ties of the hidden layers.} Share of the responses on the outer test '
                'rows that another filter of the same layer attains exactly, and the mean share of distinct responses among the '
                '$N$ filters of a row; hidden layers are numbered from one. The random column reads uniform random filters of the '
                'same shape through the same forward pass and the same tie definition, over 200 filter sets. A footrule between '
                'permutations of $V$ items is even and at most $\\lfloor V^2/2 \\rfloor$, so a response takes at most the last '
                f'column\'s number of values. Descriptive. {ties_fact}',
                'tab:s-diag-ties',
                ['Widths', 'Layer', '$V$', '$N$', two_lines('Tied', 'initial'), two_lines('Tied', 'checkpoint'),
                 two_lines('Tied', 'random'), two_lines('Distinct', 'initial'), two_lines('Distinct', 'checkpoint'),
                 two_lines('Distinct', 'random'), two_lines('Distinct', 'values')],
                trows, 'llrrrrrrrrr', size='\\scriptsize', wide=True)
    orows = [[widths_label(json.loads(r['widths'])), str(int(r['n_filters'])), str(int(r['n_datasets'])), str(int(r['n_views'])),
              f"{r['tied_nearest_share_initial']:.4f}", f"{r['tied_nearest_share_checkpoint']:.4f}"] for _, r in out.iterrows()]
    write_table('tab_s_diag_output_ties',
                '\\textbf{Training diagnostics: ties at the output layer.} Share of outer test rows whose smallest response is '
                'attained by more than one class filter, at the initial filters and at the checkpoint. Rows group the views by '
                'architecture and class count, since the output layer holds one filter per class. Descriptive.',
                'tab:s-diag-output-ties',
                ['Widths', 'Classes', 'Datasets', 'Views', two_lines('Tied nearest,', 'initial'),
                 two_lines('Tied nearest,', 'checkpoint')],
                orows, 'lrrrrr', size='\\scriptsize')

    # sensitivity of the readout to filter-ID relabeling
    view_rows = t4[(t4['scope'] == 'dataset') & (t4['level'] == 'view')].set_index('dataset_id')
    maj_rows = t4[(t4['scope'] == 'dataset') & (t4['level'] == 'seven_view_majority')].set_index('dataset_id')
    pooled4 = t4[(t4['scope'] == 'pooled') & (t4['level'] == 'seven_view_majority')].iloc[0]
    draws = int(pooled4['draws_per_unit'])
    if set(t4['draws_per_unit']) != {draws} or sorted(maj_rows.index) != sorted(COMBINED):
        raise AssertionError('diagnostics tables: the relabeling table does not cover every dataset at one draw count')
    rrows = [[NAME[x], pct(maj_rows.loc[x, 'checkpoint_accuracy_mean'], 1), pct(view_rows.loc[x, 'changed_share_mean'], 2),
              pct(view_rows.loc[x, 'changed_share_max'], 2), pct(maj_rows.loc[x, 'changed_share_mean'], 2),
              pct(maj_rows.loc[x, 'changed_share_max'], 2), signed(maj_rows.loc[x, 'accuracy_change_min'], 2),
              signed(maj_rows.loc[x, 'accuracy_change_max'], 2)] for x in COMBINED]
    cut = 0.0031
    quiet = [x for x in COMBINED if maj_rows.loc[x, 'changed_share_mean'] < cut]
    worst = maj_rows['changed_share_mean'].idxmax()
    relabel_fact = g5_fact('tables/tab_s_diag_relabel.tex',
                           f'Relabeling changes under {pct(cut, 2)} percent of the seven-view predictions on '
                           f'{counted(len(quiet), len(COMBINED), "datasets")} and {pct(maj_rows.loc[worst, "changed_share_mean"], 2)} '
                           f'percent on {NAME[worst]}.', counted(len(quiet), len(COMBINED), 'datasets'),
                           counted(len(quiet) + 1, len(COMBINED), 'datasets'))
    # one fitting seed, so the accuracy column differs from Table 3's three-seed mean (referee panel of 2026-09-23, item E2)
    off = max(abs(maj_rows.loc[x, 'checkpoint_accuracy_mean'] - (1 - combined['err'](x, KNN))) for x in COMBINED)
    seed_note = g5_fact('tables/tab_s_diag_relabel.tex',
                        f'one fitting seed, so its accuracy differs from the three-seed accuracy behind Table~\\ref{{tab:main}} by '
                        f'up to {pct(off, 1)} points', f'up to {pct(off, 1)} points', f'up to {pct(off + 0.001, 1)} points')
    write_table('tab_s_diag_relabel',
                '\\textbf{Training diagnostics: sensitivity to filter-ID tie-breaking.} At inference on the checkpoint network '
                'the hidden filter identities are permuted, with the next layer relabeled to match; the output layer is not '
                f'relabeled. Each view and each seven-view majority is drawn {draws} times. Columns give the share of outer test '
                'predictions that change, in percent, and the smallest and largest change of the seven-view accuracy in '
                f'percentage points. Descriptive: {seed_note}. {relabel_fact} HCV: hepatitis C virus.',
                'tab:s-diag-relabel',
                ['Dataset', two_lines('Accuracy', '(\\%)'), two_lines('View:', 'mean (\\%)'), two_lines('View:', 'max (\\%)'),
                 two_lines('Majority:', 'mean (\\%)'), two_lines('Majority:', 'max (\\%)'),
                 two_lines('Accuracy', 'change min'), two_lines('Accuracy', 'change max')],
                rrows, 'lrrrrrrr', size='\\scriptsize', wide=True)
    # learning curves (Figure S5 since the editing round; referee panel of 2026-09-23, item R9): test error every ten updates of ArrowFlow's seven-view
    # nearest-neighbor readout and of the nearest-neighbor and prototype readouts of one view, from the sealed report-ready table
    t5 = g5_read(d, 't5_learning_curves.csv', record, 'tables')
    curves = t5[(t5['scope'] == 'dataset') & (t5['snapshot'] == 'scheduled')]
    step = 10
    series = {'seven_knn': ('seven_view_majority', 'knn_test_accuracy', 1, 'initial filters, final readout setting'),
              'one_knn': ('per_view', 'knn_test_accuracy', FACTS['views'], 'initial filters, final readout setting'),
              'one_proto': ('per_view', 'output_rule_test_accuracy', FACTS['views'], 'initial filters')}
    points = {}
    for key, (level, measure, per_fold, first_point) in series.items():
        part = curves[(curves['level'] == level) & (curves['measure'] == measure)]
        for x in COMBINED:
            rows_x = part[part['dataset_id'] == x].sort_values('iteration')
            if [int(v) for v in rows_x['iteration']] != list(range(0, updates + 1, step)) or \
                    set(rows_x['n_folds']) != {FACTS['n_outer']} or set(rows_x['views_per_fold']) != {per_fold} or \
                    rows_x['point'].iloc[0] != first_point:
                raise AssertionError(f'diagnostics tables: the learning curve {key} of {x} is not a curve over every tenth update, '
                                     'every outer fold and every view, starting from the initial filters')
            points[(key, x)] = [100 * (1 - float(v)) for v in rows_x['mean']]
    last_above = [x for x in COMBINED if points[('one_proto', x)][-1] > points[('one_knn', x)][-1]]
    curve_fact = g5_fact('figures/figS_learning_curves.tex',
                         f'At update {updates} the prototype readout of one view has higher test error than the nearest-neighbor '
                         f'readout of one view on {counted(len(last_above), len(COMBINED), "datasets")}.',
                         counted(len(last_above), len(COMBINED), 'datasets'), counted(len(last_above) - 1, len(COMBINED), 'datasets'))
    curve_caption = ('\\textbf{Training diagnostics: learning curves.} Test error in percent on the outer test rows against the '
                     f'number of updates, every {words(step)} updates from the initial filters, update 0, to update {updates}, for one '
                     f'fitting seed; each point is the mean over the {FACTS["n_outer"]} outer folds. Solid: ArrowFlow\'s '
                     f'{words(FACTS["views"])}-view nearest-neighbor readout, combined by majority vote. Dashed: the nearest-neighbor '
                     f'readout of one view, and dotted: the prototype readout of one view, each averaged over the '
                     f'{words(FACTS["views"])} views. The curves follow the filters '
                     'through training, while the network that ArrowFlow returns is the one at its checkpoint '
                     '(Table~\\ref{tab:s-diag-checkpoints}). Each panel has its own error scale. Descriptive: nothing is selected or '
                     f'tested. {curve_fact} HCV: hepatitis C virus.')
    return dict(pooled_checkpoints=pooled1, per=per, displacement=p2, ties=hid, relabel=maj_rows, updates=updates, views=views,
                curves=points, curve_caption=curve_caption, curve_step=step)


# --------------------------------------------------------------------------------- mechanism analysis
MECH_MODELS = [KNN, UNTRAINED, INPUT_KNN, PROJECTED, 'svc_rbf', 'mlp', 'random_forest', 'gradient_boosting',
               'numeric_knn', 'dummy']
MECH_MODEL = {**DEDUP_MODEL, 'dummy': 'Majority class'}
BALANCE_CLASS = {'B': 'Balanced', 'L': 'Left', 'R': 'Right'}


def render_mechanism(runs):
    """What the two datasets with a surviving training gain give up (Section 7.3, S3): balance-scale per-class recall,
    the representation ladder per class, the torque buckets and the reading of mfeat-zernike."""
    d = runs / MECHANISM
    record = read_json(d / 'mechanism_analysis.json')
    rates = g5_read(d, 'class_rates.csv', record)
    ladder = g5_read(d, 'representation_ladder.csv', record)
    torque = g5_read(d, 'torque_structure.csv', record)
    reading = g5_read(d, 'mfeat_reading.csv', record)
    arch = g5_read(d, 'architecture_matching.csv', record)
    paired = g5_read(d, 'paired_contrasts.csv', record)
    if sorted(record['datasets']) != ['balance_scale', 'mfeat_zernike'] or \
            any(r['agrees_with_published'] is not True for _, r in paired[paired['registered']].iterrows()):
        raise AssertionError('mechanism analysis: it does not reproduce the registered contrasts of the two datasets')
    pooled = rates[(rates['scope'] == 'pooled') & (rates['dataset'] == 'balance_scale')]
    diag = {(r['model_id'], r['true_label']): r for _, r in pooled[pooled['true_label'] == pooled['predicted_label']].iterrows()}
    counts = {(r['model_id'], r['true_label']): r for _, r in pooled.iterrows()}
    classes = sorted(BALANCE_CLASS)
    if sorted({m for m, _ in diag}) != sorted(MECH_MODELS) or sorted({c for _, c in diag}) != classes:
        raise AssertionError('mechanism analysis: the balance-scale class rates do not cover every model and class')

    # per-class recall on balance-scale
    rows = []
    for model in MECH_MODELS:
        r = counts[(model, 'B')]
        cells = [MECH_MODEL[model], str(int(r['n_fitting_seeds'])), str(int(r['n_outer_fits']))]
        for c in classes:
            cells += [f"{int(counts[(model, c)]['n_true_in_scope']):,}", pct(diag[(model, c)]['share_of_true'], 3)]
        rows.append(cells)
    balanced = {m: diag[(m, 'B')]['share_of_true'] for m in MECH_MODELS}
    recall_fact = g5_fact('tables/tab_s_mech_class_rates.tex',
                          'ArrowFlow returns the balanced class on ' + pct(balanced[KNN], 3) + ' percent of its '
                          f"{int(counts[(KNN, 'B')]['n_true_in_scope']):,} balanced-class rows, the untrained ArrowFlow on "
                          + pct(balanced[UNTRAINED], 3) + ', tuned input footrule kNN on ' + pct(balanced[INPUT_KNN], 3)
                          + ' and numeric kNN on the projected scores on ' + pct(balanced[PROJECTED], 3) + '.',
                          pct(balanced[KNN], 3), pct(balanced[KNN] + 0.001, 3))
    svc_fact = g5_fact('tables/tab_s_mech_class_rates.tex',
                       'A tuned support vector classifier, fitted at one seed, returns the balanced class on '
                       + pct(balanced['svc_rbf'], 3) + ' percent of its '
                       + f"{int(counts[('svc_rbf', 'B')]['n_true_in_scope']):,} balanced-class rows.",
                       pct(balanced['svc_rbf'], 3), pct(balanced['svc_rbf'] + 0.001, 3))
    if int(counts[('svc_rbf', 'B')]['n_fitting_seeds']) != 1:
        raise AssertionError('mechanism analysis: the support vector classifier did not run at one fitting seed')
    write_table('tab_s_mech_class_rates',
                '\\textbf{Mechanism: per-class recall on balance-scale.} Share of the rows of each class that the model returns, '
                'in percent, pooled over the outer folds, repeats and fitting seeds. The three classes are the balanced scale '
                '(B), and the tips to the left (L) and to the right (R). The row count of a deterministic model is a third of a '
                'stochastic model\'s, because it is fitted at one seed. Descriptive: the pooled rows repeat the same underlying '
                f'test rows, so no rate here carries an interval or a test. {recall_fact} {svc_fact}',
                'tab:s-mech-class-rates',
                ['Model', 'Seeds', 'Fits'] + [two_lines(BALANCE_CLASS[c], t) for c in classes for t in ('rows', 'recall (\\%)')],
                rows, 'lrr' + 'rr' * len(classes), size='\\scriptsize', wide=True)

    # the representation ladder, per class
    rungs = list(dict.fromkeys(ladder['rung']))
    for dataset, name, label in (('balance_scale', 'tab_s_mech_ladder_balance', 'balance-scale'),
                                 ('mfeat_zernike', 'tab_s_mech_ladder_mfeat', 'mfeat-zernike')):
        sub = ladder[ladder['dataset'] == dataset]
        keyed = {(r['rung'], r['true_label']): r for _, r in sub.iterrows()}
        labels = sorted({c for _, c in keyed}, key=lambda c: int(c) if c.isdigit() else c)
        lrows = []
        for rung in rungs:
            first = keyed[(rung, labels[0])]
            lrows.append([RUNG_PLAIN[rung][0].upper() + RUNG_PLAIN[rung][1:], str(int(first['n_fitting_seeds'])),
                          pm(first['mean_error'], first['error_outer_fold_sd'])]
                         + [pct(keyed[(rung, c)]['pooled_recall'], 1) for c in labels])
        head = ['Rung', 'Seeds', two_lines('Error \\%', '(SD)')] + [(BALANCE_CLASS[c] if dataset == 'balance_scale' else c)
                                                                    for c in labels]
        write_table(name,
                    f'\\textbf{{Mechanism: the representation ladder on {label}.}} Mean outer-fold error in percent '
                    '(outer-fold SD) of the five rungs, and the recall of each class in percent, pooled over the outer folds, '
                    'repeats and fitting seeds. Neighboring rungs differ in more than one component. Descriptive: the pooled '
                    'recalls repeat the same test rows, so none of them carries an interval or a test.',
                    f'tab:s-mech-ladder-{"balance" if dataset == "balance_scale" else "mfeat"}',
                    head, lrows, 'lrr' + 'r' * len(labels), size='\\scriptsize', wide=True)

    # the torque buckets
    bucket_rows = torque[torque['record'] == 'bucket_recall']
    test_rows = torque[torque['record'] == 'test_row']
    if not bool(test_rows['rule_agrees'].all()):
        raise AssertionError('mechanism analysis: the torque rule does not reproduce every balance-scale row')
    buckets = list(dict.fromkeys(bucket_rows['bucket']))
    keyed = {(r['model_id'], r['bucket']): r for _, r in bucket_rows.iterrows()}
    bucket_head = [two_lines('$|\\Delta| = ' + (b.replace('-', '$ to $') if b[0].isdigit() else '') + '$'
                             if b[0].isdigit() else '$|\\Delta| \\geq 11$',
                             f"{int(keyed[(KNN, b)]['rows_in_bucket'])} rows") for b in buckets]
    brows = [[MECH_MODEL[m]] + [pct(keyed[(m, b)]['recall'], 1) for b in buckets] for m in MECH_MODELS]
    zero = buckets[0]
    gain = {m: int(keyed[(KNN, zero)]['correct_in_bucket']) - int(keyed[(m, zero)]['correct_in_bucket'])
            for m in (UNTRAINED, PROJECTED)}
    total = {m: int(bucket_rows[bucket_rows['model_id'] == KNN]['correct_in_bucket'].sum())
             - int(bucket_rows[bucket_rows['model_id'] == m]['correct_in_bucket'].sum()) for m in (UNTRAINED, PROJECTED)}
    torque_fact = g5_fact('tables/tab_s_mech_torque.tex',
                          'The rule reproduces the label of all '
                          f"{len(test_rows):,} balance-scale rows, and the {int(keyed[(KNN, zero)]['rows_in_bucket'])} rows of "
                          f'the balanced bucket carry {gain[UNTRAINED]} of the {total[UNTRAINED]} extra correct predictions '
                          f'ArrowFlow makes over the untrained ArrowFlow.', f'{gain[UNTRAINED]} of the {total[UNTRAINED]}',
                          f'{gain[UNTRAINED] + 1} of the {total[UNTRAINED]}')
    write_table('tab_s_mech_torque',
                '\\textbf{Mechanism: balance-scale recall by torque gap.} Recall in percent within each bucket of $|\\Delta|$, '
                'the absolute difference of the two torques (weight $\\times$ distance); $\\Delta = 0$ is exactly the balanced '
                'class. The header counts the distinct test rows of the bucket; each is predicted once per outer fold and '
                'fitting seed. Descriptive: pooled rates over repeated rows, with no interval and no test. The physical rule is '
                f'read from the four features and is not learned by any model here. {torque_fact}',
                'tab:s-mech-torque', ['Model'] + bucket_head, brows, 'l' + 'r' * len(buckets), size='\\scriptsize',
                wide=True)

    # reading the two datasets: where the error goes, and which architecture each control selected
    read = {r['dataset']: r for _, r in reading.iterrows()}
    sets = {dataset: arch[arch['dataset'] == dataset] for dataset in read}
    order = ['balance_scale', 'mfeat_zernike']
    points = lambda key: [f"{100 * read[x][key]:.4f}" for x in order]
    quantities = [('Numeric kNN on the raw features', points('raw_error')),
                  ('Numeric kNN on the projected scores', points('projected_error')),
                  ('Tuned input footrule kNN', points('input_error')),
                  ('Untrained ArrowFlow', points('untrained_error')),
                  ('ArrowFlow', points('arrowflow_error')),
                  ('Sorting cost: input $-$ projected', points('sorting_cost_input_minus_projected')),
                  ('Random layers: untrained $-$ input', points('random_layer_cost_untrained_minus_input')),
                  ('Representation cost: untrained $-$ projected', points('representation_cost_untrained_minus_projected')),
                  ('Training returns: untrained $-$ ArrowFlow', points('training_gain_untrained_minus_arrowflow')),
                  ('ArrowFlow $-$ projected, accuracy (pp)', [signed(read[x]['contrast_mean_difference'], 4) for x in order]),
                  (f'{FACTS["confidence"]}\\% interval of that difference',
                   [interval(read[x]['contrast_ci_low'], read[x]['contrast_ci_high'], 2) for x in order]),
                  ('$p$ of that difference, unadjusted', [pval(read[x]['contrast_p_unadjusted']) for x in order])]
    architecture = [('Outer folds', [str(len(sets[x])) for x in order]),
                    ('ArrowFlow chose two hidden layers on',
                     [str(int((sets[x]['arrowflow_hidden_layers'] == 2).sum())) for x in order]),
                    ('The untrained control chose two hidden layers on',
                     [str(int((sets[x]['untrained_hidden_layers'] == 2).sum())) for x in order]),
                    ('Both chose the same widths on', [str(int(sets[x]['widths_match'].sum())) for x in order])]
    mrows = [[name] + cells for name, cells in quantities + architecture]
    depth_counts = {x: int((sets[x]['untrained_hidden_layers'] == 2).sum()) for x in order}
    reading_fact = g5_fact('tables/tab_s_mech_reading.tex',
                           'On mfeat-zernike sorting costs '
                           f"{100 * read['mfeat_zernike']['sorting_cost_input_minus_projected']:.4f} points and random layers "
                           f"{100 * read['mfeat_zernike']['random_layer_cost_untrained_minus_input']:.4f}, and training returns "
                           f"{100 * read['mfeat_zernike']['training_gain_untrained_minus_arrowflow']:.4f} of the "
                           f"{100 * read['mfeat_zernike']['representation_cost_untrained_minus_projected']:.4f} points.",
                           f"{100 * read['mfeat_zernike']['sorting_cost_input_minus_projected']:.4f} points",
                           f"{100 * read['mfeat_zernike']['sorting_cost_input_minus_projected'] + 0.001:.4f} points")
    arch_fact = g5_fact('tables/tab_s_mech_reading.tex',
                        'The untrained control chose two hidden layers on '
                        f"{depth_counts['balance_scale']} of the {len(sets['balance_scale'])} balance-scale and "
                        f"{depth_counts['mfeat_zernike']} of the {len(sets['mfeat_zernike'])} mfeat-zernike outer folds.",
                        f"{depth_counts['balance_scale']} of the", f"{depth_counts['balance_scale'] + 1} of the")
    write_table('tab_s_mech_reading',
                '\\textbf{Mechanism: where the error goes, and which architecture was selected.} The first block gives mean '
                'outer-fold error in percent and the differences between neighboring rungs in points; a positive difference is '
                'a cost paid, and the training line is what training gives back. The last three lines of the block are the '
                f'paired ArrowFlow minus projected kNN difference with its {interval_note()}, post hoc and unadjusted. The '
                'second block counts outer folds. '
                f'{reading_fact} {arch_fact}',
                'tab:s-mech-reading', ['Quantity'] + [NAME[x] for x in order], mrows, 'lrr', size='\\scriptsize',
                subheads={0: 'Error and its differences (percent, points)', len(quantities): 'Selected architecture (outer folds)'})
    recall = {(m, c): float(diag[(m, c)]['share_of_true']) for m in MECH_MODELS for c in classes}
    return dict(reading=read, balanced=balanced, gain=gain, total=total, buckets=keyed, depth_counts=depth_counts, recall=recall,
                level_rows=int(keyed[(KNN, zero)]['rows_in_bucket']), all_rows=len(test_rows))


# --------------------------------------------------------------------------------- fixed-configuration depth
def render_depth(runs):
    """Depth at a matched configuration (Section 7.3, S3): the three variant arms, the movement rates that must be read
    beside them, and the split by the architecture each fold's own selection chose."""
    d = runs / DEPTH_AGGREGATION / 'depth' / 'analysis'
    record = read_json(d / 'depth_analysis.json')
    frame = g5_rows(pd.read_csv(d / 'depth_contrasts.csv'), record['contrasts'] + record['named_subsets'],
                    ['contrast', 'subset', 'dataset_id'],
                    ['mean_difference', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate', 'n_folds'],
                    'depth contrasts')
    declaration = record['analysis_declaration']
    n_outer, folds = FACTS['n_outer'], len(COMBINED) * FACTS['n_outer']
    if declaration['reference_arm'] != 'depth1' or sorted(declaration['variant_arms']) != sorted(DEPTH_ARMS) or \
            any(v['performed'] != v['passed'] for v in record['check_totals'].values()) or \
            sorted(record['datasets']) != sorted(COMBINED):
        raise AssertionError('depth: the run is not the three variant arms against one hidden layer on every dataset')
    if any(record['inertness'][a]['folds'] != folds for a in ['depth1'] + DEPTH_ARMS):
        raise AssertionError('depth: an arm does not cover every outer fold')
    rows = frame[frame['subset'] == 'all_folds']
    keyed = {(DEPTH_CONTRAST[r['contrast']], r['dataset_id']): r for _, r in rows.iterrows()}
    better = {a: sum(keyed[(a, x)]['mean_difference'] > 0 for x in COMBINED) for a in DEPTH_ARMS}
    holm = {a: [x for x in COMBINED if keyed[(a, x)]['holm_p_approximate'] < 0.05] for a in DEPTH_ARMS}
    head = ['Dataset'] + [two_lines(t, DEPTH_LABEL[a].replace(' hidden layers', '').replace(' layer', ''))
                          for a in DEPTH_ARMS for t in ('Diff. (pp)', 'Holm $p$')]
    head = ['Dataset'] + [two_lines(DEPTH_LABEL[a].split()[0] if a == 'depth2' else ('Untrained' if 'untrained' in a else 'First'),
                                    t) for a in DEPTH_ARMS for t in ('Diff. (pp) [interval]', 'Holm $p$')]
    drows = [[NAME[x]] + [f'{signed(keyed[(a, x)]["mean_difference"], 2)} {interval(keyed[(a, x)]["ci_low"], keyed[(a, x)]["ci_high"], 2)}'
                          if t else pval(keyed[(a, x)]['holm_p_approximate']) for a in DEPTH_ARMS for t in (1, 0)]
             for x in COMBINED]
    largest = max(COMBINED, key=lambda x: keyed[('depth2', x)]['mean_difference'])
    depth_fact = g5_fact('tables/tab_s_depth_contrasts.tex',
                         'One hidden layer is more accurate than two on '
                         f'{counted(better["depth2"], len(COMBINED), "datasets")} at a matched configuration, '
                         f'{of_total(len(holm["depth2"]), words(len(COMBINED)), "contrasts", "survives", "survive", capital=False)} '
                         f'Holm adjustment, and the largest difference is {NAME[largest]} '
                         f'{signed(keyed[("depth2", largest)]["mean_difference"], 2)} points '
                         f'(Holm {pval(keyed[("depth2", largest)]["holm_p_approximate"])}).',
                         counted(better['depth2'], len(COMBINED), 'datasets'),
                         counted(better['depth2'] - 1, len(COMBINED), 'datasets'))
    write_table('tab_s_depth_contrasts',
                '\\textbf{Depth at a matched configuration.} One hidden layer minus each two-layer arm, accuracy, in percentage '
                f'points with the {interval_note()}; positive favors one layer. Every arm reads the same per-fold selection, '
                'the same splits and one fitting seed, and neither the learning rate nor the update count is retuned. The '
                'untrained-second arm never moves its second hidden layer; the first-only arm never moves the layer the readout '
                'reads. Every two-layer arm used the earlier relay of Section~\\ref{sec:training}; '
                'Table~\\ref{tab:s-relay-contrasts} repeats the primary contrast with the corrected relay. Holm adjusts across the '
                f'{words(len(COMBINED))} datasets within each arm. {depth_fact} '
                'HCV: hepatitis C virus.',
                'tab:s-depth-contrasts', head, drows, 'l' + 'rr' * len(DEPTH_ARMS), size='\\scriptsize', wide=True)

    # movement, without which the crippled arms cannot be read
    arms = ['depth1'] + DEPTH_ARMS
    move = record['movement']
    mrows = [[NAME[x]] + [pct(move[a][x]['changed_share'], 1) for a in arms]
             + [str(int(move[a][x]['effectively_inert_folds'])) for a in DEPTH_ARMS] for x in COMBINED]
    mrows.append(['All ' + words(len(COMBINED)) + ' together'] + [pct(record['inertness'][a]['mean_changed_share'], 1) for a in arms]
                 + [str(int(record['inertness'][a]['effectively_inert_folds'])) for a in DEPTH_ARMS])
    threshold = move[arms[0]][COMBINED[0]]['inert_threshold']
    movement_fact = g5_fact('tables/tab_s_depth_movement.tex',
                            'Over all folds a hidden filter update reorders the filter in '
                            + ' and '.join(pct(record['inertness'][a]['mean_changed_share'], 1) for a in arms[:2])
                            + ' percent of the cases in the two fully trained arms, in '
                            + pct(record['inertness']['depth2_untrained_second']['mean_changed_share'], 1)
                            + ' percent in the untrained-second arm and in '
                            + pct(record['inertness']['depth2_first_only']['mean_changed_share'], 1) + ' percent in the first-only arm.',
                            pct(record['inertness']['depth2_untrained_second']['mean_changed_share'], 1) + ' percent in the',
                            pct(record['inertness']['depth2_untrained_second']['mean_changed_share'] + 0.001, 1) + ' percent in the')
    write_table('tab_s_depth_movement',
                '\\textbf{Depth: how much each arm moved.} Share of the hidden filter updates that change the filter\'s order, '
                'in percent. Every batch updates each hidden filter once, and the share pools the batches, the hidden layers and '
                'the seven views of a fit and is averaged over the outer folds. It is a rate per update, not the displacement at '
                'the checkpoint of Table~\\ref{tab:s-diag-displacement}. The last columns '
                f'count the outer folds on which an arm is effectively inert, that is below a changed share of {threshold:g}. '
                f'The last row pools all {len(COMBINED) * FACTS["n_outer"]} folds. A contrast against an arm that barely moves '
                f'is evidence about movement, not about depth. Descriptive. {movement_fact} HCV: hepatitis C virus.',
                'tab:s-depth-movement',
                ['Dataset'] + [two_lines(DEPTH_LABEL[a].split()[0] if a in ('depth1', 'depth2') else
                                         ('Untrained' if 'untrained' in a else 'First'), 'changed (\\%)') for a in arms]
                + [two_lines(('Untrained' if 'untrained' in a else 'First' if 'first' in a else 'Two'), 'inert folds')
                   for a in DEPTH_ARMS],
                mrows, 'l' + 'r' * (len(arms) + len(DEPTH_ARMS)), size='\\scriptsize', wide=True,
                block_rules=(len(COMBINED),))

    # the split by the architecture each fold's own selection chose
    subsets = [s for s in dict.fromkeys(frame['subset']) if s != 'all_folds']
    # every subset row covers only the folds of its column, so its interval uses n - 1 degrees of freedom (referee panel of
    # 2026-09-23, item G); a row whose fold differences are all equal has no interval
    flat = check_subset_intervals(frame[frame['subset'] != 'all_folds'], 'depth subsets')
    flat_keys = {(r['subset'], r['dataset_id']) for r in flat if r['contrast'] == 'depth1_minus_depth2'}
    sub = {(s, x): r for s in subsets for _, r in frame[(frame['subset'] == s) & (frame['contrast'] == 'depth1_minus_depth2')].iterrows()
           for x in [r['dataset_id']] }
    two_layer = {x: int(sub[(subsets[1], x)]['n_folds']) for x in COMBINED}
    srows = [[NAME[x], f'{two_layer[x]} of {n_outer}']
             + [g5_dash(sub[(s, x)]['mean_difference'], lambda v: signed(v, 2)) for s in subsets]
             + ['--' if blank(sub[(s, x)]['ci_low']) or (s, x) in flat_keys else
                interval(sub[(s, x)]['ci_low'], sub[(s, x)]['ci_high'], 2) for s in subsets] for x in COMBINED]
    selected_two = sum(two_layer.values())
    none_two = sum(1 for x in COMBINED if not two_layer[x])
    split_fact = g5_fact('tables/tab_s_depth_subsets.tex',
                         f'The fold\'s own selection chose two hidden layers on {selected_two} of the '
                         f'{len(COMBINED) * n_outer} outer folds, and on none of the fifteen on '
                         f'{counted(none_two, len(COMBINED), "datasets")}.',
                         f'{selected_two} of the', f'{selected_two + 1} of the')
    depth_df_fact = g5_fact('tables/tab_s_depth_subsets.tex', SUBSET_DF, '$n-1$ degrees', f'{FACTS["df"]} degrees')
    subset_caption = ('\\textbf{Depth: the same contrast on the folds each selection chose.} One hidden layer minus two, '
                      'accuracy, in percentage points, restricted to the outer folds whose own reconstructed selection was one '
                      'hidden layer and to those whose selection was two. The fold sets differ by dataset and are chosen by the '
                      'selection itself, so these columns describe where each architecture was selected and carry no test. '
                      f'Intervals are {interval_note(subset=True)}; $n$ is the number of folds of the row\'s column. '
                      f'{split_fact} HCV: hepatitis C virus.')
    if depth_df_fact not in subset_caption or f'{FACTS["df"]} degrees of freedom' in subset_caption:
        raise AssertionError('the depth subset table must state the n - 1 rule its rows use, and no full-fold count')
    write_table('tab_s_depth_subsets', subset_caption,
                'tab:s-depth-subsets',
                ['Dataset', two_lines('Two-layer', 'folds')]
                + [two_lines(s.replace('own_selection_', '').replace('_', ' '), t) for t in ('Diff. (pp)',) for s in subsets]
                + [two_lines(s.replace('own_selection_', '').replace('_', ' '), f'{FACTS["confidence"]}\\% interval') for s in subsets],
                srows, 'lrrrrr', size='\\scriptsize', wide=True)
    return dict(keyed=keyed, better=better, holm=holm, movement=record['inertness'], largest=largest, two_layer=selected_two,
                per_dataset_movement=move)


# --------------------------------------------------------------------------------- aggregation of the view votes
def render_aggregation(runs):
    """Borda against the median (Section 7.3, S3): the mass-matched contrast, and the movement table that shows why the
    plain median's contrast is not a comparison of aggregation rules."""
    d = runs / DEPTH_AGGREGATION / 'aggregation' / 'analysis'
    record = read_json(d / 'aggregation_analysis.json')
    frame = g5_rows(pd.read_csv(d / 'aggregation_contrasts.csv'), record['contrasts'] + record['named_subsets'],
                    ['contrast', 'dataset_id'],
                    ['mean_difference', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate', 'n_folds'],
                    'aggregation contrasts')
    arms = ['borda', 'median', 'median_mass_matched']
    folds = len(COMBINED) * FACTS['n_outer']
    if record['reference_arm'] != 'borda' or sorted(record['arms']) != sorted(arms) or \
            any(v['performed'] != v['passed'] for v in record['check_totals'].values()) or \
            any(record['inertness'][a]['folds'] != folds for a in arms):
        raise AssertionError('aggregation: the run is not Borda against the two median arms on every fold')
    matched = frame[frame['contrast'] == 'borda_minus_median_mass_matched']
    keyed = {r['dataset_id']: r for _, r in matched.iterrows()}
    better = sum(keyed[x]['mean_difference'] > 0 for x in COMBINED)
    holm = [x for x in COMBINED if keyed[x]['holm_p_approximate'] < 0.05]
    rows = [[NAME[x], signed(keyed[x]['mean_difference'], 2), interval(keyed[x]['ci_low'], keyed[x]['ci_high'], 2),
             pval(keyed[x]['p_approximate']), pval(keyed[x]['holm_p_approximate'])] for x in COMBINED]
    survivor = holm[0] if holm else None
    agg_fact = g5_fact('tables/tab_s_aggregation_contrasts.tex',
                       'Against the prior-scaled median, Borda is more accurate on '
                       f'{counted(better, len(COMBINED), "datasets")}, and '
                       + (f'{NAME[survivor]} {signed(keyed[survivor]["mean_difference"], 2)} points '
                          f'(Holm {pval(keyed[survivor]["holm_p_approximate"])}) is the only contrast that survives Holm '
                          'adjustment.' if survivor else 'no contrast survives Holm adjustment.'),
                       counted(better, len(COMBINED), 'datasets'), counted(better - 1, len(COMBINED), 'datasets'))
    # the mass-matched median receives the same ballots as Borda but not the same prior weight (referee panel of 2026-09-23,
    # items R8 and A22): its prior counts as m ballots of the mean incoming weight, m chosen on training-only pilot fits
    protocol = runs / DEPTH_AGGREGATION / 'aggregation' / 'run' / 'protocol.json'
    if sha256_file(protocol) != sha256_file(PROTOCOLS_G5 / 'aggregation.json'):
        raise AssertionError('aggregation: the run protocol differs from the frozen protocol file')
    choice = read_json(protocol)['mass_matching_choice']
    multiplier, pilots = int(choice['prior_multiplier']), sorted(choice['ladders'])
    if [x['prior_multiplier'] for x in choice['ladders'][pilots[0]]] != read_json(protocol)['mass_matching']['ladder']:
        raise AssertionError('aggregation: the recorded ladder differs from the declared one')
    prior_fact = g5_fact('tables/tab_s_aggregation_contrasts.tex',
                         f'its prior counts as {words(multiplier)} ballots of the mean incoming vote weight, where Borda\'s prior '
                         f'has weight one. The multiple was chosen on training-only pilot fits of {words(len(pilots))} datasets',
                         f'as {words(multiplier)} ballots', f'as {words(multiplier + 1)} ballots')
    write_table('tab_s_aggregation_contrasts',
                '\\textbf{Aggregation: Borda minus the prior-scaled median.} Accuracy difference in percentage points with the '
                f'{interval_note()}; positive favors the Borda count. The prior-scaled median receives the same ballots as the '
                f'Borda count, but {prior_fact} so that it moved the filters about as often as Borda on those fits. The arms therefore '
                'differ in how the votes are combined and in the weight of the prior. One fitting seed and one selection per '
                f'fold; Holm adjusts across the {words(len(COMBINED))} datasets. This is a comparison of two vote-combining rules '
                f'and not a test of any axiom. {agg_fact} HCV: hepatitis C virus.',
                'tab:s-aggregation-contrasts',
                ['Dataset', 'Diff. (pp)', f'{FACTS["confidence"]}\\% interval', '$p$', 'Holm $p$'], rows, 'lrrrr',
                size='\\scriptsize')

    move = record['movement']
    threshold = move[arms[0]][COMBINED[0]]['inert_threshold']
    mrows = [[NAME[x]] + [pct(move[a][x]['changed_share'], 1) for a in arms]
             + [f"{int(move[a][x]['effectively_inert_folds'])} of {FACTS['n_outer']}" for a in arms[1:]] for x in COMBINED]
    mrows.append(['All ' + words(len(COMBINED)) + ' together'] + [pct(record['inertness'][a]['mean_changed_share'], 1) for a in arms]
                 + [f"{int(record['inertness'][a]['effectively_inert_folds'])} of {folds}" for a in arms[1:]])
    inert_fact = g5_fact('tables/tab_s_aggregation_movement.tex',
                         'The plain median is effectively inert on '
                         f"{int(record['inertness']['median']['effectively_inert_folds'])} of the {folds} folds, with a mean "
                         f"changed share of {record['inertness']['median']['mean_changed_share']:.3f} against "
                         f"{record['inertness']['borda']['mean_changed_share']:.3f} for Borda, so its contrast measures "
                         'movement and not the aggregation rule.',
                         f"{int(record['inertness']['median']['effectively_inert_folds'])} of the {folds} folds",
                         f"{int(record['inertness']['median']['effectively_inert_folds']) + 1} of the {folds} folds")
    write_table('tab_s_aggregation_movement',
                '\\textbf{Aggregation: how much each rule moved.} Share of the hidden filter updates that change the filter\'s '
                'order, in percent, as in Table~\\ref{tab:s-depth-movement}: a rate per batch update, averaged over the outer '
                'folds. The last '
                f'columns count the outer folds on which the rule is effectively inert, that is below a changed share of '
                f'{threshold:g}. The last row pools all {folds} folds. {inert_fact} HCV: hepatitis C virus.',
                'tab:s-aggregation-movement',
                ['Dataset'] + [two_lines(AGG_LABEL[a].split()[0], 'changed (\\%)') for a in arms]
                + [two_lines(AGG_LABEL[a].split()[0], 'inert folds') for a in arms[1:]],
                mrows, 'l' + 'r' * (len(arms) + 2), size='\\scriptsize', wide=True, block_rules=(len(COMBINED),))
    return dict(keyed=keyed, better=better, holm=holm, inertness=record['inertness'], folds=folds, multiplier=multiplier,
                pilots=pilots, movement=move)


# --------------------------------------------------------------------------------- the duplicate-free rerun
DEDUP_ORDER = ['wine_quality_dedup', 'segment_dedup']
DEDUP_FULL_MODELS = [KNN, UNTRAINED, INPUT_KNN, PROJECTED, 'svc_rbf', 'random_forest', 'mlp', 'numeric_knn',
                     'gradient_boosting', 'dummy']


def render_dedup(runs):
    """The duplicate-free rerun of Wine quality and Segment. Its seven detailed tables left the supplement by the author's
    decision of 2026-09-23: the duplicate material is now one paragraph of S3.2 beside the audit table. Every sealed output of
    the rerun's analysis is still read and verified, and the facts behind that paragraph are returned for approved_edits_claims:
    the two registered training families, ArrowFlow's standing and rank among the tuned models, the error of every model on the
    distinct rows, the prototype readout of the component ablation, and the unpaired rise of every model's error when the
    duplicates are dropped."""
    d = runs / DEDUP / 'analysis'
    record = read_json(d / 'dedup_analysis.json')
    families = g5_read(d, 'dedup_families.csv', record)
    for name in ('dedup_ladder.csv', 'dedup_comparators.csv', 'dedup_complete_metrics.csv'):
        g5_read(d, name, record)   # sealed outputs of the record that no text quotes any more, verified all the same
    components = g5_read(d, 'dedup_components.csv', record)
    main = g5_read(d, 'dedup_main_table.csv', record)
    full = g5_read(d, 'dedup_full_data.csv', record)
    duplicates = g5_read(d, 'dedup_duplicates.csv', record)
    standing = g5_read(d, 'dedup_competitiveness.csv', record)
    declaration = record['analysis_declaration']
    if sorted(declaration['primary_family']['datasets']) != sorted(DEDUP_ORDER) or \
            declaration['primary_family']['model_b'] != UNTRAINED or declaration['secondary_family']['model_b'] != INPUT_KNN or \
            any(v['matches_expected_counts'] is not True or int(v['duplicate_rows']) for _, v in duplicates.iterrows()):
        raise AssertionError('dedup: the reruns are not duplicate-free, or the two families are not the registered ones')
    if declaration['full_data_comparison']['status'] != 'unpaired; descriptive; no interval and no test':
        raise AssertionError('dedup: the full-data comparison is not declared unpaired and descriptive')
    keyed = {(r['family'], r['dataset']): r for _, r in families.iterrows()}
    if sorted(keyed) != sorted((f, x) for f in ('primary', 'secondary') for x in DEDUP_ORDER):
        raise AssertionError('dedup: the rerun does not hold one contrast per registered family and dataset')
    errors = {(r['dataset'], r['model_id']): float(r['mean_error']) for _, r in main.iterrows()}
    flag = {r['dataset']: r for _, r in standing.iterrows()}
    if any(abs(float(flag[x]['arrowflow_error']) - errors[(x, KNN)]) > 1e-12 for x in DEDUP_ORDER):
        raise AssertionError('dedup: the standing record and the main table disagree on ArrowFlow\'s error')
    rank = {x: 1 + sum(errors[(x, m)] < errors[(x, KNN)] for m in COMPARATORS) for x in DEDUP_ORDER}
    prototype = {r['dataset']: r for _, r in components.iterrows() if r['variant'] == 'prototype_readout'}
    rise = {x: {r['model_id']: float(r['dedup_minus_full_points']) for _, r in full[full['dataset'] == x].iterrows()}
            for x in DEDUP_ORDER}
    if any(sorted(rise[x]) != sorted(DEDUP_FULL_MODELS) for x in DEDUP_ORDER):
        raise AssertionError('dedup: the full-data comparison does not cover the models of the rerun')
    return dict(families=keyed, rank=rank, flag=flag, errors=errors, prototype=prototype, rise=rise)


# --------------------------------------------------------------------------------- neighbor baselines
def render_baselines(runs):
    """Four established neighborhood baselines on the same folds (Section 7.4, S3): the four Holm families, their
    intervals, the mean errors, the rank panel and the convergence warnings."""
    d = runs / BASELINES / 'analysis'
    record = read_json(d / 'baseline_analysis.json')
    families = g5_read(d, 'baseline_families.csv', record)
    metrics = g5_read(d, 'baseline_metrics.csv', record)
    ranks = g5_read(d, 'baseline_ranks.csv', record)
    warnings = g5_read(d, 'baseline_warnings.csv', record)
    declaration = record['analysis_declaration']
    n_outer, conf = FACTS['n_outer'], FACTS['confidence']
    if declaration['reference_model'] != KNN or sorted(declaration['models']) != sorted(BASELINE_MODELS) or \
            any(declaration['families'][m]['size'] != len(COMBINED) for m in BASELINE_MODELS) or \
            sorted(families['family'].unique()) != sorted(BASELINE_MODELS) or set(families['n_folds']) != {n_outer}:
        raise AssertionError('baselines: the four families are not ArrowFlow against each baseline over the seventeen datasets')
    if 'is not refitted for this family' not in declaration['reference_not_refitted']:
        raise AssertionError('baselines: ArrowFlow was refitted for this family')
    keyed = {(r['family'], r['dataset']): r for _, r in families.iterrows()}
    ahead = {m: sum(keyed[(m, x)]['mean_difference'] > 0 for x in COMBINED) for m in BASELINE_MODELS}
    mean_gap = {m: float(families[families['family'] == m]['mean_difference'].mean()) for m in BASELINE_MODELS}
    holm = {m: [x for x in COMBINED if keyed[(m, x)]['holm_p_approximate'] < 0.05] for m in BASELINE_MODELS}

    head = ['Dataset'] + [two_lines(BASELINE[m], 'Diff. (pp)') for m in BASELINE_MODELS] \
        + [two_lines(BASELINE[m], 'Holm $p$') for m in BASELINE_MODELS]
    rows = [[NAME[x]] + [signed(keyed[(m, x)]['mean_difference'], 2) for m in BASELINE_MODELS]
            + [pval(keyed[(m, x)]['holm_p_approximate']) for m in BASELINE_MODELS] for x in COMBINED]
    kernel = 'kendall_svc'
    # the kernel's lead counted by the unrounded sign (referee-panel revision of 2026-09-25, items B27 and B15): no dataset ties, so
    # "has a higher mean accuracy" counts the same datasets that "matches or beats" did
    kernel_ahead = sum(keyed[(kernel, x)]['mean_difference'] < 0 for x in COMBINED)
    if kernel_ahead != len(COMBINED) - ahead[kernel]:
        raise AssertionError('baselines: a dataset on which ArrowFlow and the Kendall SVC are exactly equal')
    kernel_fact = g5_fact('tables/tab_s_baseline_families.tex',
                          'A fixed Kendall kernel on rankings from the same encoder family has a higher mean accuracy than ArrowFlow on '
                          f'{counted(kernel_ahead, len(COMBINED), "datasets")}, with a mean difference of '
                          f'{signed(mean_gap[kernel], 2)} points.',
                          counted(kernel_ahead, len(COMBINED), 'datasets'),
                          counted(kernel_ahead - 1, len(COMBINED), 'datasets'))
    metric_fact = g5_fact('tables/tab_s_baseline_families.tex',
                          'ArrowFlow has the higher mean accuracy on '
                          + ', '.join(f'{ahead[m]} of the {len(COMBINED)} against {BASELINE[m]}' for m in BASELINE_MODELS[:3])
                          + '.', f'{ahead[BASELINE_MODELS[0]]} of the', f'{ahead[BASELINE_MODELS[0]] + 1} of the')
    write_table('tab_s_baseline_families',
                '\\textbf{Neighbor baselines: the four Holm families.} ArrowFlow minus the baseline, accuracy, in percentage '
                'points; positive favors ArrowFlow. Every contrast is paired: the same rows, the same outer folds and the same '
                'fitting seeds as the registered runs, whose ArrowFlow predictions are reused and not refitted. Holm adjusts '
                f'across the {words(len(COMBINED))} datasets within each baseline, the four baselines separately. LDA: linear '
                'discriminant analysis; PCA: principal component analysis; NCA: neighborhood components analysis. The Kendall '
                "SVC is a support vector classifier with a fixed Kendall kernel on rankings from ArrowFlow's encoder family, at "
                'encoder settings it chooses on its own inner folds. '
                f'{kernel_fact} {metric_fact} HCV: hepatitis C virus.',
                'tab:s-baseline-families', head, rows, 'l' + 'r' * (2 * len(BASELINE_MODELS)), size='\\scriptsize', wide=True,
                stack=len(BASELINE_MODELS))

    ihead = ['Dataset'] + [two_lines(BASELINE[m], f'{conf}\\% interval') for m in BASELINE_MODELS] \
        + [two_lines(BASELINE[m], '$p$') for m in BASELINE_MODELS]
    irows = [[NAME[x]] + [interval(keyed[(m, x)]['ci_low'], keyed[(m, x)]['ci_high'], 2) for m in BASELINE_MODELS]
             + [pval(keyed[(m, x)]['p_approximate']) for m in BASELINE_MODELS] for x in COMBINED]
    write_table('tab_s_baseline_intervals',
                '\\textbf{Neighbor baselines: intervals and unadjusted $p$ values.} The same contrasts as the preceding table '
                f'with their {interval_note()} and their unadjusted $p$ values. The intervals are not simultaneous. '
                'HCV: hepatitis C virus.',
                'tab:s-baseline-intervals', ihead, irows, 'l' + 'r' * (2 * len(BASELINE_MODELS)), size='\\scriptsize',
                wide=True, stack=len(BASELINE_MODELS))

    errors = {(r['dataset'], r['model_id']): r for _, r in metrics[metrics['metric'] == 'error'].iterrows()}
    models = [KNN] + BASELINE_MODELS
    erows = [[NAME[x]] + [pm(errors[(x, m)]['mean'], errors[(x, m)]['outer_fold_sd'], 2) for m in models] for x in COMBINED]
    write_table('tab_s_baseline_errors',
                '\\textbf{Neighbor baselines: mean outer-fold error.} Error in percent (outer-fold SD); fitting seeds are '
                'averaged within each outer fold. The three pipelines read the imputed and scaled features with no polynomial '
                'expansion; only the Kendall SVC reads ArrowFlow\'s own encoder. Descriptive. HCV: hepatitis C virus.',
                'tab:s-baseline-errors', ['Dataset'] + [MODEL[KNN]] + [BASELINE[m] for m in BASELINE_MODELS], erows,
                'l' + 'r' * len(models), size='\\scriptsize')

    place = {(r['dataset'], r['model_id']): r for _, r in ranks.iterrows()}
    ranked = int(ranks['models_ranked'].iloc[0])
    if set(ranks['models_ranked']) != {ranked} or sorted({m for _, m in place}) != sorted(RANK_PANEL):
        raise AssertionError('baselines: the rank panel does not hold the same models on every dataset')
    rrows = [[NAME[x]] + [f"{place[(x, m)]['rank']:g}" for m in RANK_PANEL] for x in COMBINED]
    mean_rank = {m: sum(float(place[(x, m)]['rank']) for x in COMBINED) / len(COMBINED) for m in RANK_PANEL}
    rrows.append(['Mean rank'] + [f'{mean_rank[m]:.1f}' for m in RANK_PANEL])
    write_table('tab_s_baseline_ranks',
                f'\\textbf{{Neighbor baselines: the rank panel.}} Rank of each of the {words(ranked)} models by mean outer-fold '
                'error within a dataset, 1 the lowest error; tied models share the average rank. The panel holds ArrowFlow, the '
                'five tuned comparators, the majority class and the four baselines. Descriptive: the last row is the mean rank '
                'over the datasets and carries no test. HCV: hepatitis C virus.',
                'tab:s-baseline-ranks', ['Dataset'] + [RANK_HEAD[m] for m in RANK_PANEL], rrows,
                'l' + 'r' * len(RANK_PANEL), size='\\scriptsize', wide=True, rotate=True, block_rules=(len(COMBINED),))

    outer = {(r['dataset'], r['model_id']): r for _, r in warnings[warnings['scope'] == 'outer_fits'].iterrows()}
    every = {(r['dataset'], r['model_id']): r for _, r in warnings[warnings['scope'] == 'all_fits'].iterrows()}
    noisy = [m for m in BASELINE_MODELS if int(sum(outer[(x, m)]['fits_with_warning'] for x in COMBINED))
             or int(sum(every[(x, m)]['fits_with_warning'] for x in COMBINED))]
    wrows = [[NAME[x]] + [f"{int(src[(x, m)]['fits_with_warning'])} of {int(src[(x, m)]['fits'])}"
                          for m in noisy for src in (outer, every)] for x in COMBINED]
    wrows.append(['All ' + words(len(COMBINED)) + ' together']
                 + [f"{int(sum(src[(x, m)]['fits_with_warning'] for x in COMBINED))} of "
                    f"{int(sum(src[(x, m)]['fits'] for x in COMBINED))}" for m in noisy for src in (outer, every)])
    quiet = [BASELINE[m] for m in BASELINE_MODELS if m not in noisy]
    warn_fact = g5_fact('tables/tab_s_baseline_warnings.tex',
                        'Only ' + listing([BASELINE[m] for m in noisy], 'and') + ' raised a convergence warning: '
                        f"{int(sum(outer[(x, noisy[0])]['fits_with_warning'] for x in COMBINED))} of its "
                        f"{int(sum(outer[(x, noisy[0])]['fits'] for x in COMBINED))} outer fits reached the iteration cap.",
                        f"{int(sum(outer[(x, noisy[0])]['fits_with_warning'] for x in COMBINED))} of its",
                        f"{int(sum(outer[(x, noisy[0])]['fits_with_warning'] for x in COMBINED)) + 1} of its")
    write_table('tab_s_baseline_warnings',
                '\\textbf{Neighbor baselines: convergence warnings.} Fits that raised a warning, out of all fits, over the outer '
                'fits and over every fit of the complete logs, inner folds included. '
                + listing(quiet, 'and') + ' raised none, so they have no column. Descriptive. '
                f'{warn_fact} HCV: hepatitis C virus.',
                'tab:s-baseline-warnings',
                ['Dataset'] + [two_lines(BASELINE[m], 'outer fits' if src is outer else 'all fits')
                               for m in noisy for src in (outer, every)],
                wrows, 'l' + 'r' * (2 * len(noisy)), size='\\scriptsize', block_rules=(len(COMBINED),))
    return dict(keyed=keyed, ahead=ahead, mean_gap=mean_gap, holm=holm, mean_rank=mean_rank, ranked=ranked,
                noisy=noisy, errors=errors)


# --------------------------------------------------------------------------------- the relay and representation studies
# The two families of 2026-09-23 (final round): depth with the corrected, signed relay, and the readout-matched representation
# test. The relay that every earlier run used passed each hidden vote's motion toward the target actually used down to the
# layer below, unsigned; for a repelling vote that is the motion toward the reversed input, which is orthogonal to the
# push-away direction. The corrected relay passes sign(a_j) m(r_j -> pi). Both families are read from their sealed analyses,
# with their frozen protocols, and nothing numeric is typed here.
PROTOCOLS_23 = REPO / 'experiments' / 'make_revision' / 'protocols' / '2026-09-23'
SIGNED_RELAY = '2026-09-23-signed-relay-depth'
REPRESENTATION = '2026-09-23-representation-test'
RELAY_ARMS = ['depth1', 'depth2_printed', 'depth2_signed', 'depth2_signed_scaled']
RELAY_CONTRASTS = ['depth1_minus_depth2_signed', 'depth1_minus_depth2_signed_scaled', 'depth2_signed_minus_depth2_printed']
RELAY_HEAD = {'depth1_minus_depth2_signed': 'One $-$ corrected', 'depth1_minus_depth2_signed_scaled': 'One $-$ scaled',
              'depth2_signed_minus_depth2_printed': 'Corrected $-$ earlier'}
REP_ARMS = ['svc_input', 'svc_untrained', 'svc_trained', 'knn_input', 'knn_untrained', 'knn_trained']
REP_REPRESENTATIONS = ['input', 'untrained', 'trained']
DEVELOPMENT_LABEL = 'screened before this protocol'


def frozen_run_protocol(run, name, label):
    """A run of 2026-09-23 must have executed exactly its committed frozen protocol."""
    protocol = read_json(run / 'protocol.json')
    if sha256_file(run / 'protocol.json') != sha256_file(PROTOCOLS_23 / name) or protocol.get('frozen') is not True:
        raise AssertionError(f'{label}: the run did not execute the frozen protocol {name}')
    return protocol


def render_signed_relay(runs):
    """Depth with the corrected relay (Section 7.6, S3.7): one hidden layer against two hidden layers with the corrected relay
    and with the corrected relay and first-layer votes scaled by s, and the corrected minus the earlier relay in two-layer
    networks; the first hidden layer's movement per arm. The one-layer arm and the two-layer arm with the earlier relay are
    the depth family's own arms, which this renderer checks value for value against that family."""
    d = runs / SIGNED_RELAY / 'analysis'
    record = read_json(d / 'signed_relay_depth_analysis.json')
    frame = g5_rows(pd.read_csv(d / 'signed_relay_depth_contrasts.csv'), record['contrasts'] + record['named_subsets'],
                    ['contrast', 'subset', 'dataset_id'],
                    ['mean_difference', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate', 'n_folds'],
                    'signed-relay depth contrasts')
    protocol = frozen_run_protocol(runs / SIGNED_RELAY / 'run', 'signed_relay_depth.json', 'signed-relay depth')
    n_outer, k17 = FACTS['n_outer'], len(COMBINED)
    declaration = record['analysis_declaration']
    if record['protocol_id'] != protocol['protocol_id'] or record['arms'] != RELAY_ARMS or protocol['arms'] != RELAY_ARMS or \
            sorted(record['datasets']) != sorted(COMBINED) or \
            any(v['performed'] != v['passed'] for v in record['check_totals'].values()) or \
            record['check_totals']['reused_arms_reproduced']['passed'] != k17 or \
            sorted(f['contrast'] for f in declaration['families']) != sorted(RELAY_CONTRASTS) or \
            any(f['size'] != k17 for f in declaration['families']):
        raise AssertionError('signed-relay depth: the run is not the four arms and three families of its protocol on every dataset')
    if any(record['summaries'][x]['folds'] != n_outer or not all(record['summaries'][x]['reuse_reproduced'].values())
           for x in COMBINED):
        raise AssertionError('signed-relay depth: a dataset lacks a fold, or a reused arm did not reproduce its stored record')
    # the one-layer arm and the arm with the earlier relay are the depth family's arms, value for value
    old = read_json(runs / DEPTH_AGGREGATION / 'depth' / 'analysis' / 'depth_analysis.json')
    for x in COMBINED:
        for new_arm, old_arm in (('depth1', 'depth1'), ('depth2_printed', 'depth2')):
            a = record['summaries'][x]['arms'][new_arm]['metrics']['accuracy']['mean']
            b = old['summaries'][x]['arms'][old_arm]['metrics']['accuracy']['mean']
            if abs(a - b) > 1e-12 or abs(record['movement'][new_arm][x]['changed_share'] - old['movement'][old_arm][x]['changed_share']) > 1e-12:
                raise AssertionError(f'signed-relay depth: {x}/{new_arm} differs from the depth family\'s {old_arm}')
    rows = frame[frame['subset'] == 'all_folds']
    keyed = {(r['contrast'], r['dataset_id']): r for _, r in rows.iterrows()}
    if sorted(keyed) != sorted((c, x) for c in RELAY_CONTRASTS for x in COMBINED) or set(rows['n_folds']) != {n_outer}:
        raise AssertionError('signed-relay depth: the three contrasts do not cover every dataset over every outer fold')
    diff = {c: {x: float(keyed[(c, x)]['mean_difference']) for x in COMBINED} for c in RELAY_CONTRASTS}
    holm = {c: [x for x in COMBINED if keyed[(c, x)]['holm_p_approximate'] < 0.05] for c in RELAY_CONTRASTS}
    first_ahead = {c: sum(v > 0 for v in diff[c].values()) for c in RELAY_CONTRASTS}
    second_ahead = {c: sum(v < 0 for v in diff[c].values()) for c in RELAY_CONTRASTS}
    mean = {c: statistics.mean(diff[c].values()) for c in RELAY_CONTRASTS}
    # the prespecified rule, recomputed: a corrected arm helps only if it is ahead of one layer on most datasets and
    # Holm-significantly ahead on at least one
    rule = record['interpretation']
    most = {arm: int(rule['arms'][arm]['most_threshold']) for arm in ('depth2_signed', 'depth2_signed_scaled')}
    helps = {arm: second_ahead[f'depth1_minus_{arm}'] >= most[arm] and
             any(diff[f'depth1_minus_{arm}'][x] < 0 for x in holm[f'depth1_minus_{arm}']) for arm in most}
    outcome = 'helps' if any(helps.values()) else 'does_not_help'
    if outcome != rule['outcome'] or any(helps[a] != rule['arms'][a]['helps'] or second_ahead[f'depth1_minus_{a}'] != rule['arms'][a]['n_higher_mean']
                                         for a in helps) or most != {a: 9 for a in most} or \
            protocol['analysis']['interpretation'][outcome] != rule['statement']:
        raise AssertionError('signed-relay depth: the recomputed prespecified outcome differs from the analysis record')
    # the vote scale of the fourth arm: the smallest ladder value whose cleared share reaches the target on every pilot dataset
    choice, scale = protocol['scale_choice'], protocol['vote_scale']
    pilots = sorted(choice['cleared_share'])
    reached = [s for s in scale['ladder'] if all(choice['cleared_share'][p][str(s)] >= scale['target'] for p in pilots)]
    s = int(choice['lower_vote_scale'])
    if not reached or s != min(reached) or s != int(record['scale_choice']['lower_vote_scale']) or pilots != ['ionosphere', 'iris']:
        raise AssertionError('signed-relay depth: the recorded vote scale is not the smallest ladder value that reaches its target')
    if 'multiplied by s before it is accumulated' not in scale['definition'] or \
            'only the hidden widths, the relay and the first layer\'s vote scale differ' not in protocol['arm_definitions']['note']:
        raise AssertionError('signed-relay depth: the scaled arm is no longer defined as a first-layer vote scale alone')

    head = ['Dataset'] + [two_lines(RELAY_HEAD[c], t) for c in RELAY_CONTRASTS for t in ('Diff. (pp) [interval]', 'Holm $p$')]
    trows = [[NAME[x]] + [cell for c in RELAY_CONTRASTS for cell in
                          (f'{signed(diff[c][x], 2)} {interval(keyed[(c, x)]["ci_low"], keyed[(c, x)]["ci_high"], 2)}',
                           pval(keyed[(c, x)]['holm_p_approximate']))] for x in COMBINED]
    c1, c2, c3 = RELAY_CONTRASTS
    if holm[c1] or holm[c2] or holm[c3]:
        raise AssertionError('signed-relay depth: the caption says no contrast survives Holm adjustment')
    relay_fact = g5_fact('tables/tab_s_relay_contrasts.tex',
                         f'One hidden layer is more accurate than two on {counted(first_ahead[c1], k17, "datasets")} with the '
                         f'corrected relay, by {signed(mean[c1], 2)} points on average, and on {first_ahead[c2]} with the '
                         f'scaled votes; the correction makes two-layer networks more accurate on {first_ahead[c3]}, by '
                         f'{signed(mean[c3], 2)} points on average. No contrast survives Holm adjustment.',
                         counted(first_ahead[c1], k17, 'datasets'), counted(first_ahead[c1] + 1, k17, 'datasets'))
    write_table('tab_s_relay_contrasts',
                '\\textbf{Depth with the corrected relay.} One hidden layer minus two hidden layers with the corrected relay '
                f'(corrected), and minus two hidden layers whose first-layer votes are also multiplied by {s} (scaled); then two '
                'hidden layers with the corrected relay minus two with the earlier relay. Accuracy, in percentage points, with '
                f'the {interval_note()}; positive favors the first-named. The four arms share each fold\'s selected '
                'configuration, splits and one fitting seed; the one-layer arm and the arm with the earlier relay are those of '
                'Table~\\ref{tab:s-depth-contrasts}. Holm adjusts across the '
                f'{words(k17)} datasets within each contrast. {relay_fact} HCV: hepatitis C virus.',
                'tab:s-relay-contrasts', head, trows, 'l' + 'rr' * len(RELAY_CONTRASTS), size='\\scriptsize', wide=True)

    # the first hidden layer's movement: a rate per batch update, and the share of voted filter-batches that clear the prior
    move = record['movement']
    two = ['depth2_printed', 'depth2_signed', 'depth2_signed_scaled']
    first = {(a, x): move[a][x]['by_layer'][0] for a in two for x in COMBINED}
    if any(first[(a, x)]['layer'] != 0 or len(move[a][x]['by_layer']) != 2 for a in two for x in COMBINED) or \
            any(first[(a, x)].get('cleared_share_of_voted') is None for a in two[1:] for x in COMBINED):
        raise AssertionError('signed-relay depth: the movement record lacks the first hidden layer of a two-layer arm')
    reordered = {a: statistics.mean(first[(a, x)]['changed_share'] for x in COMBINED) for a in two}
    cleared = {a: statistics.mean(first[(a, x)]['cleared_share_of_voted'] for x in COMBINED) for a in two[1:]}
    mrows = [[NAME[x]] + [pct(first[(a, x)]['changed_share'], 1) for a in two]
             + [pct(first[(a, x)]['cleared_share_of_voted'], 1) for a in two[1:]] for x in COMBINED]
    mrows.append([f'Mean over the {words(k17)}'] + [pct(reordered[a], 1) for a in two] + [pct(cleared[a], 1) for a in two[1:]])
    move_fact = g5_fact('tables/tab_s_relay_movement.tex',
                        f'On average over the {words(k17)} datasets a batch update reorders {pct(reordered["depth2_printed"], 1)}, '
                        f'{pct(reordered["depth2_signed"], 1)} and {pct(reordered["depth2_signed_scaled"], 1)} percent of the '
                        'first-layer filters with the earlier relay, the corrected relay and the scaled votes. The scaled '
                        f'votes clear the prior in {pct(cleared["depth2_signed_scaled"], 1)} percent of the voted filter-batches, '
                        f'against {pct(cleared["depth2_signed"], 1)} with the corrected relay.',
                        f'{pct(reordered["depth2_signed_scaled"], 1)} percent', f'{pct(reordered["depth2_signed_scaled"] + 0.001, 1)} percent')
    write_table('tab_s_relay_movement',
                '\\textbf{Depth with the corrected relay: how much the first hidden layer moved.} Reordered: the share of the '
                'first hidden layer\'s filter updates that change the filter\'s order, in percent. Every batch updates each '
                'filter once, and the share pools the batches and the seven views of a fit and is averaged over the outer '
                'folds. Clearing: among the first layer\'s filter-batches with at least one nonzero vote, the share whose '
                'vote mass $M$ reaches the prior\'s threshold, $M(e-1)\\ge1$, below which the prior dictates the filter\'s '
                'order (Section~\\ref{supp:iia}); the record of the earlier relay does not hold it. Descriptive. '
                f'{move_fact} HCV: hepatitis C virus.',
                'tab:s-relay-movement',
                ['Dataset'] + [two_lines(t, 'reordered (\\%)') for t in ('Earlier', 'Corrected', 'Scaled')]
                + [two_lines(t, 'clearing (\\%)') for t in ('Corrected', 'Scaled')],
                mrows, 'l' + 'r' * 5, size='\\scriptsize', block_rules=(k17,))
    return dict(diff=diff, holm=holm, first_ahead=first_ahead, second_ahead=second_ahead, mean=mean, outcome=outcome,
                statement=rule['statement'], scale=s, pilots=pilots, ladder=list(scale['ladder']), target=scale['target'],
                most=most, reordered=reordered, cleared=cleared, protocol=protocol, record=record)


def render_representation(runs, components):
    """The readout-matched representation test (Section 7.6, S3.7): a Kendall-kernel support vector classifier reads the
    encoded input, untrained hidden and trained hidden rankings of every view at ArrowFlow's own selections. The primary and
    secondary families on the ten further datasets, the seven development datasets as descriptive rows, and the comparison of
    the strong readout with the nearest-neighbor readout on each representation."""
    d = runs / REPRESENTATION / 'analysis'
    record = read_json(d / 'representation_analysis.json')
    primary = g5_read(d, 'representation_primary_family.csv', record)
    secondary = g5_read(d, 'representation_secondary_family.csv', record)
    development = g5_read(d, 'representation_development.csv', record)
    readouts = g5_read(d, 'representation_readout_comparison.csv', record)
    neighbors = g5_read(d, 'representation_knn_contrasts.csv', record)
    run = runs / REPRESENTATION / 'run'
    protocol = frozen_run_protocol(run, 'representation_test.json', 'representation test')
    n_outer, k17, k = FACTS['n_outer'], len(COMBINED), len(NEWDATA)
    if record['provenance']['run']['protocol_sha256'] != sha256_file(run / 'protocol.json') or \
            record['provenance']['run']['protocol_id'] != protocol['protocol_id'] or protocol['arms'] != REP_ARMS or \
            any(checks != {'True': k17 * n_outer} for checks in record['job_checks'].values()) or \
            any(v['matching_fold_seeds'] != v['total_fold_seeds'] or v['total_fold_seeds'] != n_outer * FACTS['n_seeds']
                for x in COMBINED for v in record['reproduction'][x].values()) or sorted(record['reproduction']) != sorted(COMBINED):
        raise AssertionError('representation test: the run, its job checks or its reproduction of the stored predictions differ')
    interp = protocol['analysis']['interpretation']
    if sorted(interp['development_datasets']) != sorted(DATASETS) or sorted(protocol['further_datasets']) != sorted(NEWDATA):
        raise AssertionError('representation test: the development and further datasets are not the benchmark and further datasets')
    fam = {}
    for name, frame, contrast in (('primary', primary, 'svc_untrained'), ('secondary', secondary, 'svc_input')):
        if sorted(frame['dataset']) != sorted(NEWDATA) or set(frame['model_a']) != {'svc_trained'} or \
                set(frame['model_b']) != {contrast} or set(frame['n_folds']) != {n_outer} or set(frame['panel']) != {'further'}:
            raise AssertionError(f'representation test: the {name} family is not trained minus {contrast} on the further datasets')
        keyed = {r['dataset']: r for _, r in frame.iterrows()}
        higher = [x for x in NEWDATA if keyed[x]['mean_difference'] > 0]
        gains = [x for x in NEWDATA if keyed[x]['holm_p_approximate'] < 0.05 and keyed[x]['mean_difference'] > 0]
        losses = [x for x in NEWDATA if keyed[x]['holm_p_approximate'] < 0.05 and keyed[x]['mean_difference'] < 0]
        rule = interp[name]
        met = bool(gains) and len(higher) >= int(rule['most_threshold'])
        held = record['interpretation'][name]
        if met != held['met'] or len(higher) != held['higher_mean_count'] or held['statement'] != rule['statements']['met' if met else 'not_met']:
            raise AssertionError(f'representation test: the recomputed {name} interpretation differs from the analysis record')
        fam[name] = dict(keyed=keyed, higher=higher, gains=gains, losses=losses, met=met, statement=held['statement'],
                         mean=statistics.mean(float(keyed[x]['mean_difference']) for x in NEWDATA),
                         most=int(rule['most_threshold']))
    dev = {(r['model_b'], r['dataset']): r for _, r in development.iterrows()}
    if sorted(dev) != sorted((b, x) for b in ('svc_untrained', 'svc_input') for x in DATASETS) or \
            set(development['label']) != {DEVELOPMENT_LABEL} or set(development['model_a']) != {'svc_trained'}:
        raise AssertionError('representation test: the development rows are not both contrasts on the seven benchmark datasets')
    # the neighborhood side: under the nearest-neighbor readout, trained minus untrained is ArrowFlow minus the untrained networks
    # of the component ablation, prediction for prediction
    near = {(r['model_b'], r['dataset']): r for _, r in neighbors.iterrows()}
    if any(abs(float(near[('knn_untrained', x)]['mean_difference']) - components['gain'][(x, 'untrained')]) > 1e-12 for x in COMBINED):
        raise AssertionError('representation test: the kNN readout of the trained and untrained rankings is not the component ablation')
    comparison = {(r['model_a'].split('_', 1)[1], r['dataset']): r for _, r in readouts.iterrows()}
    if sorted(comparison) != sorted((rep, x) for rep in REP_REPRESENTATIONS for x in COMBINED) or \
            any(r['model_b'] != 'knn_' + r['model_a'].split('_', 1)[1] for _, r in readouts.iterrows()):
        raise AssertionError('representation test: the readout comparison is not the SVC minus the kNN readout on each representation')
    svc_ahead = {rep: sum(float(comparison[(rep, x)]['mean_difference']) > 0 for x in COMBINED) for rep in REP_REPRESENTATIONS}
    svc_mean = {rep: statistics.mean(float(comparison[(rep, x)]['mean_difference']) for x in COMBINED) for rep in REP_REPRESENTATIONS}

    p, s2 = fam['primary'], fam['secondary']
    head = ['Dataset'] + [two_lines('Trained $-$ untrained', 'Diff. (pp) [interval]'), two_lines('Trained $-$ untrained', '$p$'),
                          two_lines('Trained $-$ untrained', 'Holm $p$'), two_lines('Trained $-$ input', 'Diff. (pp) [interval]'),
                          two_lines('Trained $-$ input', 'Holm $p$')]
    cell = lambda r: f'{signed(r["mean_difference"], 2)} {interval(r["ci_low"], r["ci_high"], 2)}'
    frows = [[NAME[x], cell(p['keyed'][x]), pval(p['keyed'][x]['p_approximate']), pval(p['keyed'][x]['holm_p_approximate']),
              cell(s2['keyed'][x]), pval(s2['keyed'][x]['holm_p_approximate'])] for x in NEWDATA]
    frows += [[NAME[x], cell(dev[('svc_untrained', x)]), '--', '--', cell(dev[('svc_input', x)]), '--'] for x in DATASETS]
    if p['gains'] or p['losses'] or s2['gains'] or s2['losses']:
        raise AssertionError('representation test: the caption says no contrast of either family survives Holm adjustment')
    family_fact = g5_fact('tables/tab_s_rep_family.tex',
                          f'The trained rankings have the higher mean accuracy on {counted(len(p["higher"]), k, "further datasets")} '
                          f'against the untrained rankings, with a mean difference of {signed(p["mean"], 2)} points, and on '
                          f'{len(s2["higher"])} against the encoded inputs; no contrast of either family survives Holm adjustment.',
                          counted(len(p['higher']), k, 'further datasets'), counted(len(p['higher']) + 1, k, 'further datasets'))
    write_table('tab_s_rep_family',
                '\\textbf{Representation test: the trained rankings under a fixed Kendall-kernel readout.} Accuracy differences, '
                f'in percentage points, with the {interval_note()}; positive favors the trained hidden rankings. A support '
                'vector classifier on the Kendall kernel reads each representation of every view at ArrowFlow\'s own selected '
                'configuration and fitting seeds, and the seven views vote by plurality. The primary family is trained minus '
                'untrained rankings and the secondary family trained rankings minus the encoded input rankings, each on the '
                f'{words(k)} further datasets with its own Holm adjustment. The {words(len(DATASETS))} benchmark datasets were '
                'screened for this question before the protocol was written, so their rows carry no $p$ value. '
                f'{family_fact} HCV: hepatitis C virus.',
                'tab:s-rep-family', head, frows, 'lrrrrr', size='\\scriptsize', wide=True,
                subheads={0: f'The {words(k)} further datasets: the registered families',
                          k: f'The {words(len(DATASETS))} benchmark datasets, {DEVELOPMENT_LABEL}: descriptive'})

    rhead = ['Dataset'] + [two_lines('SVC $-$ kNN,', f'{rep} (pp)') for rep in REP_REPRESENTATIONS] \
        + [two_lines('kNN: trained $-$', 'untrained (pp)')]
    rrows = [[NAME[x]] + [signed(comparison[(rep, x)]['mean_difference'], 2) for rep in REP_REPRESENTATIONS]
             + [signed(near[('knn_untrained', x)]['mean_difference'], 2)] for x in COMBINED]
    rrows.append([f'Mean over the {words(k17)}'] + [signed(svc_mean[rep], 2) for rep in REP_REPRESENTATIONS]
                 + [signed(statistics.mean(float(near[('knn_untrained', x)]['mean_difference']) for x in COMBINED), 2)])
    readout_fact = g5_fact('tables/tab_s_rep_readouts.tex',
                           'The support vector classifier is more accurate than the nearest-neighbor readout on '
                           + ', '.join(f'{svc_ahead[rep]}' for rep in REP_REPRESENTATIONS[:2]) + f' and {svc_ahead["trained"]} of '
                           f'the {k17} datasets on the input, untrained and trained rankings, by '
                           + ', '.join(f'{signed(svc_mean[rep], 2)}' for rep in REP_REPRESENTATIONS[:2])
                           + f' and {signed(svc_mean["trained"], 2)} points on average.',
                           f'and {svc_ahead["trained"]} of', f'and {svc_ahead["trained"] - 1} of')
    write_table('tab_s_rep_readouts',
                '\\textbf{Representation test: the strong readout against the nearest-neighbor readout.} The Kendall-kernel '
                'support vector classifier (SVC) minus ArrowFlow\'s nearest-neighbor (kNN) readout on the same representation, '
                'accuracy, in percentage points; positive favors the SVC. The last column is the kNN readout of the trained '
                'minus the untrained rankings, which is ArrowFlow minus the untrained networks of '
                'Table~\\ref{tab:s-components-changes}. Descriptive: no interval, no $p$ value and no adjustment. '
                f'{readout_fact} HCV: hepatitis C virus.',
                'tab:s-rep-readouts', rhead, rrows, 'l' + 'r' * 4, size='\\scriptsize', block_rules=(k17,))
    return dict(primary=p, secondary=s2, development=dev, svc_ahead=svc_ahead, svc_mean=svc_mean, near=near,
                protocol=protocol, record=record)


# --------------------------------------------------------------------------------- the learned encoder (Sections 5.5 and 7.7, S6)
# The two studies of the learned encoder fixed their designs in the headers of their modules and committed them before any score
# existed; the run directories record the revision they ran at. Everything below is recomputed from their result files and from
# the registered runs of ArrowFlow and the comparison models, which cover the same seventeen datasets and the same test sets.
LEARNED_NESTED = '2026-09-24-tensor-rank-nested'
LEARNED_SWAP = '2026-09-25-encoder-swap'
LEARNED_FREEZE = {'nested': '4d26573e0fd7296197e7c39ac049b959815c747a', 'swap': 'de380080c373d514abd5e11d5232266cea822693'}
LEARNED_MODULES = {'nested': 'experiments/tensor_rank_sandbox/nested.py', 'swap': 'experiments/tensor_rank_sandbox/encoder_swap.py',
                   'conversion': 'experiments/tensor_rank_sandbox/core_hybrid.py'}
HYBRID_ARMS = ('hybrid_target_mlp', 'hybrid_frozen_mlp', 'hybrid_target_linear')
HYBRID_GRID = [{'V': v, 'lr_tensor': lr} for v in (64, 128) for lr in (0.003, 0.01)]
HYBRID_DESIGN = {'arms': {'hybrid_target_mlp': ['target', 'core_mlp', HYBRID_GRID],
                          'hybrid_frozen_mlp': ['frozen', 'core_mlp', [{'V': v, 'lr_tensor': 0.01} for v in (64, 128)]],
                          'hybrid_target_linear': ['target', 'linear', HYBRID_GRID]},
                 'epochs': 60, 'common': {'contrast': 0.5, 'beta': 1.0, 'gate': 'accepted'},
                 'build': {'lr_rank': 0.1, 'p_correct': 0.1, 'batch_size': 32}, 'fit_seeds': [8129, 19391, 39019]}
SWAP_DESIGN = {'arms': ['learned', 'random'], 'encoder': {'epochs': 60, 'lr_tensor': 0.01, 'lr_rank': 0.1, 'p_correct': 0.1,
                                                          'contrast': 0.5, 'beta': 1.0, 'gate': 'accepted', 'arch': 'core_mlp'},
               'fit_seeds': [8129, 19391, 39019], 'reference_model': KNN}
SWAP_CHECK = [('iris', 0), ('segment', 0), ('balance_scale', 0), ('mfeat_zernike', 3)]
LEARNED_COMPARATORS = BRIDGE_MODELS[1:-1]
S6 = 'supplement_sections/S6_learned_encoder.tex'
E5 = 'sections/05_encoding_ensemble.tex'
CHECK_LINE = re.compile(r'^(\w+)\s+fold\s+(\d+) seed (\d+): fixed refit vs registered: (\d+) of (\d+) differ')


def git_blob(revision, path):
    import subprocess
    return subprocess.run(['git', 'show', f'{revision}:{REPO.name}/{path}'], cwd=REPO.parent, capture_output=True,
                          check=True).stdout


def git_time_utc(revision):
    import subprocess
    from datetime import datetime, timezone
    stamp = subprocess.run(['git', 'show', '-s', '--format=%cI', revision], cwd=REPO.parent, capture_output=True, text=True,
                           check=True).stdout.strip()
    return datetime.fromisoformat(stamp).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')


def exact_fit_accuracy(accuracy, n, where):
    acc = Fraction(round(accuracy * n), n)
    if abs(float(acc) - accuracy) > 1e-12:
        raise AssertionError(f'{where}: accuracy {accuracy} is not a count over {n} rows')
    return acc


def learned_folds(run, arm, x, test_rows, key='acc'):
    """Seed-averaged accuracy per outer fold of one arm on one dataset, exact, checked against the registered test sets."""
    out = {}
    for path in sorted((run / 'results').glob(f'{x}__{arm}__i*.json')):
        record = read_json(path)
        k = (record['outer_repeat'], record['outer_fold'])
        fits = record['fits']
        seeds = [f['seed'] for f in fits]
        if seeds != FACTS['bridge']['fit_seeds'] or k in out:
            raise AssertionError(f'{path.name}: fit seeds {seeds} or a repeated fold')
        pred = 'pred' if key == 'acc' else key.replace('_acc', '_pred')
        if any(len(f[pred]) != test_rows[k] for f in fits):
            raise AssertionError(f'{path.name}: predictions for another test set than the registered one')
        out[k] = sum(exact_fit_accuracy(f[key], test_rows[k], path.name) for f in fits) / len(fits)
    if sorted(out) != sorted(test_rows):
        raise AssertionError(f'{run.name}/{x}/{arm}: {len(out)} outer folds, not the registered {len(test_rows)}')
    return out


def paired_contrast(a, b):
    keys = sorted(a)
    if keys != sorted(b):
        raise AssertionError('a paired contrast over different outer folds')
    diffs = [float(a[k] - b[k]) for k in keys]
    if len(set(diffs)) == 1:
        # identical differences on every test set (on banknote-authentication both models are perfect on every test set): no
        # variance, so no interval; the p value is one for a zero difference, as the runs' own analyses record it
        return dict(mean=diffs[0], lo=diffs[0], hi=diffs[0], p=1.0 if diffs[0] == 0 else 0.0, better=sum(v > 0 for v in diffs))
    mean, _, lo, hi, p = corrected_t(diffs, FACTS['confidence'])
    return dict(mean=mean, lo=lo, hi=hi, p=p, better=sum(v > 0 for v in diffs))


def contrast_family(pairs):
    """Per-dataset paired contrasts with the Holm adjustment across the seventeen datasets, and their counts."""
    rows = {x: paired_contrast(*pairs[x]) for x in COMBINED}
    for x, adjusted in zip(COMBINED, holm([rows[x]['p'] for x in COMBINED])):
        rows[x]['holm'] = adjusted
    higher = [x for x in COMBINED if rows[x]['mean'] > 0]
    up = [x for x in COMBINED if rows[x]['mean'] > 0 and rows[x]['holm'] < 0.05]
    down = [x for x in COMBINED if rows[x]['mean'] < 0 and rows[x]['holm'] < 0.05]
    return dict(rows=rows, higher=higher, up=up, down=down, mean=statistics.mean(rows[x]['mean'] for x in COMBINED))


def learned_worker_pools(runs):
    """The worker pools of the two studies of the learned encoder (S5.1, Table S-environment): single-thread workers with no
    graphics processor visible, the modules' default pool, and the swap's second pool, which its helper in the run directory
    states."""
    pools = {}
    threads = "for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):\n    os.environ[_n] = '1'"
    for which in ('nested', 'swap'):
        src = (REPO / LEARNED_MODULES[which]).read_text()
        m_workers = re.search(r"parser\.add_argument\('--workers', type=int, default=(\d+)\)", src)
        if not m_workers or threads not in src or "os.environ['CUDA_VISIBLE_DEVICES'] = ''" not in src:
            raise AssertionError(f'learned encoder: the {which} module no longer runs a default pool of single-thread CPU workers')
        pools[which] = int(m_workers.group(1))
    second = (runs / LEARNED_SWAP / 'second_pool.py').read_text()
    m_second = re.search(r"a second pool of (\d+) workers beside the main\s+pool's (\d+)", second)
    if not m_second or int(m_second.group(2)) != pools['swap'] or threads not in second or \
            "os.environ['CUDA_VISIBLE_DEVICES'] = ''" not in second:
        raise AssertionError('learned encoder: the second pool of the swap is not single-threaded on the CPU beside the main pool')
    pools['second'] = int(m_second.group(1))
    return pools


def render_learned_encoder(runs, knn, newdata):
    """The learned encoder (Sections 5.5 and 7.7, S6): the hybrid classifier under nested cross-validation and the encoder swap
    inside ArrowFlow, both verified against their frozen designs, their committed modules and the registered runs."""
    nested_run, swap_run = runs / LEARNED_NESTED, runs / LEARNED_SWAP
    # ---- the frozen designs and the code that ran
    for which, run in (('nested', nested_run), ('swap', swap_run)):
        if (run / 'code_revision.txt').read_text().strip() != LEARNED_FREEZE[which]:
            raise AssertionError(f'learned encoder: the {which} run did not run at its freeze commit')
        for module in (LEARNED_MODULES[which], LEARNED_MODULES['conversion']):
            if git_blob(LEARNED_FREEZE[which], module) != (REPO / module).read_bytes():
                raise AssertionError(f'learned encoder: {module} differs from its version at the {which} freeze')
    nested_design, swap_design = read_json(nested_run / 'design.json'), read_json(swap_run / 'design.json')
    if {k: v for k, v in nested_design.items() if k != 'sources'} != HYBRID_DESIGN or swap_design != SWAP_DESIGN:
        raise AssertionError('learned encoder: a run design differs from the design the text states')
    runs_of = {x: knn for x in DATASETS} | {x: fam for fam in newdata['B'].values() for x in fam.protocol['datasets']}
    if sorted(runs_of) != sorted(COMBINED):
        raise AssertionError('learned encoder: the registered runs do not cover the seventeen datasets')
    # ---- the registered models on the same test sets
    test_rows, registered = {}, {}
    for x in COMBINED:
        fam = runs_of[x]
        rows = [r for r in fam.summary['model_rows'][x] if r['model_id'] == KNN]
        test_rows[x] = {(r['outer_repeat'], r['outer_fold']): len(r['test_rows']) for r in rows}
        for m in [KNN] + LEARNED_COMPARATORS:
            registered[(x, m)] = fold_accuracy(fam, m, x)
    # ---- the hybrid classifier under nested cross-validation
    hyb = {}
    for x in COMBINED:
        for arm in HYBRID_ARMS:
            for readout in ('rank', 'knn'):
                hyb[(x, arm, readout)] = learned_folds(nested_run, arm, x, test_rows[x], f'{readout}_acc')
    P = {'P1': contrast_family({x: (hyb[(x, 'hybrid_target_mlp', 'rank')], hyb[(x, 'hybrid_frozen_mlp', 'rank')]) for x in COMBINED}),
         'P2': contrast_family({x: (hyb[(x, 'hybrid_target_mlp', 'rank')], registered[(x, 'mlp')]) for x in COMBINED}),
         'P3': contrast_family({x: (hyb[(x, 'hybrid_target_mlp', 'knn')], registered[(x, KNN)]) for x in COMBINED})}
    held = read_json(nested_run / 'analysis.json')
    for name in P:
        for row in held['contrasts'][name]:
            mine = P[name]['rows'][row['dataset']]
            if abs(100 * mine['mean'] - row['mean_difference']) > 1e-9 or abs(mine['p'] - row['p']) > 1e-9 \
                    or abs(mine['holm'] - row['p_holm']) > 1e-9:
                raise AssertionError(f'learned encoder: {name} on {row["dataset"]} differs from the run analysis')
    # ---- the encoder swap inside ArrowFlow, at ArrowFlow's own selections
    swap, widths = {}, defaultdict(list)
    for x in COMBINED:
        for arm in SWAP_DESIGN['arms']:
            swap[(x, arm)] = learned_folds(swap_run, arm, x, test_rows[x])
            for path in (swap_run / 'results').glob(f'{x}__{arm}__i*.json'):
                record = read_json(path)
                reference = read_json(runs_of[x].dir / 'results' / f"{x}__{KNN}__r{record['outer_repeat']}f{record['outer_fold']}.json")
                if record['config'] != reference['selection']['config']:
                    raise AssertionError(f'{path.name}: not the configuration ArrowFlow selected on this test set')
                # the encoder network's width is the encoded vocabulary size resolved from that configuration (S6.4)
                widths[arm].append(int(record['params']['embed_dim']))
    S = {'primary': contrast_family({x: (swap[(x, 'learned')], registered[(x, KNN)]) for x in COMBINED}),
         'learned_random': contrast_family({x: (swap[(x, 'learned')], swap[(x, 'random')]) for x in COMBINED}),
         'random_fixed': contrast_family({x: (swap[(x, 'random')], registered[(x, KNN)]) for x in COMBINED}),
         'learned_mlp': contrast_family({x: (swap[(x, 'learned')], registered[(x, 'mlp')]) for x in COMBINED}),
         # the design's fourth secondary contrast (referee-panel revision of 2026-09-25, item B7): ArrowFlow with the trained encoder
         # minus the trained encoder of the nested study read alone by its nearest-neighbor vote, on the same test sets
         'learned_hybrid_knn': contrast_family({x: (swap[(x, 'learned')], hyb[(x, 'hybrid_target_mlp', 'knn')]) for x in COMBINED})}
    prim = S['primary']
    verdict = ('improves ArrowFlow' if prim['mean'] > 0 and len(prim['up']) > len(prim['down'])
               else 'worsens ArrowFlow' if prim['mean'] < 0 and len(prim['down']) > len(prim['up']) else 'no difference shown')
    held = read_json(swap_run / 'analysis.json')
    names = {'PRIMARY': 'primary', 'learned - random': 'learned_random', 'random - fixed': 'random_fixed', 'learned - mlp': 'learned_mlp',
             'learned - hybrid_knn': 'learned_hybrid_knn'}
    if sorted(held['contrasts']) != sorted(names):
        raise AssertionError('learned encoder: the run analysis holds other contrasts than the design declared')
    for name, key in names.items():
        for row in held['contrasts'][name]['rows']:
            mine = S[key]['rows'][row['dataset']]
            if abs(100 * mine['mean'] - row['mean_difference']) > 1e-9 or abs(mine['p'] - row['p']) > 1e-9 \
                    or abs(mine['holm'] - row['p_holm']) > 1e-9:
                raise AssertionError(f'learned encoder: the swap contrast {name} on {row["dataset"]} differs from the run analysis')
        summary = held['contrasts'][name]
        if abs(100 * S[key]['mean'] - summary['mean_difference']) > 1e-9 or summary['significantly_higher'] != len(S[key]['up']) \
                or summary['significantly_lower'] != len(S[key]['down']):
            raise AssertionError(f'learned encoder: the summary of the swap contrast {name} differs from the run analysis')
    if held['contrasts']['PRIMARY']['verdict'] != verdict:
        raise AssertionError('learned encoder: the recomputed verdict differs from the verdict the run analysis recorded')
    # the reproduction check before the run: the fixed arm refitted by the module reproduced the registered predictions exactly
    checked = []
    for line in (swap_run / 'check.log').read_text().splitlines():
        m = CHECK_LINE.match(line)
        if m:
            x, index, seed, differing, n = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            split = read_json(runs_of[x].dir / x / 'splits.json')[index]
            if differing or n != test_rows[x][(split['outer_repeat'], split['outer_fold'])]:
                raise AssertionError(f'learned encoder: the reproduction check of {x} fold {index} seed {seed} failed')
            checked.append((x, index, seed, n))
    if sorted((x, i, s) for x, i, s, _ in checked) != sorted((x, i, s) for x, i in SWAP_CHECK for s in FACTS['bridge']['fit_seeds']):
        raise AssertionError('learned encoder: the reproduction check did not cover its declared sample')
    # jobs computed twice by the run's two worker pools, from a snapshot taken before the second pool ended: identical records
    recomputed = 0
    for path in sorted((swap_run / 'snapshot_before_second_pool_end').glob('*.json')):
        a, b = read_json(path), read_json(swap_run / 'results' / path.name)
        if {k: v for k, v in a.items() if k != 'seconds'} != {k: v for k, v in b.items() if k != 'seconds'}:
            raise AssertionError(f'learned encoder: {path.name} changed when it was computed a second time')
        recomputed += a['seconds'] != b['seconds']
    pools = learned_worker_pools(runs)
    # the best tuned comparison model per dataset, against ArrowFlow with the trained encoder
    mean = lambda folds: float(sum(folds.values()) / len(folds))
    best = {x: max(mean(registered[(x, m)]) for m in LEARNED_COMPARATORS) for x in COMBINED}
    trails = [x for x in COMBINED if mean(swap[(x, 'learned')]) < best[x]]

    # ---- tables
    err = lambda folds: pct(1 - mean(folds))
    k17, n_outer = len(COMBINED), FACTS['n_outer']
    cell = lambda r: f"{signed(r['mean'], 2)} {interval(r['lo'], r['hi'], 2)}"
    blocks = {0: 'The seven benchmark datasets', len(DATASETS): 'The ten further datasets'}
    head1 = ['Dataset', two_lines('Trained encoder,', 'class filters'), two_lines('Untrained encoder,', 'class filters'),
             two_lines('Trained linear enc.,', 'class filters'), two_lines('Trained encoder,', 'kNN readout'), 'ArrowFlow', 'MLP']
    rows1 = [[NAME[x], err(hyb[(x, 'hybrid_target_mlp', 'rank')]), err(hyb[(x, 'hybrid_frozen_mlp', 'rank')]),
              err(hyb[(x, 'hybrid_target_linear', 'rank')]), err(hyb[(x, 'hybrid_target_mlp', 'knn')]),
              err(registered[(x, KNN)]), err(registered[(x, 'mlp')])] for x in COMBINED]
    columns1 = [lambda x: hyb[(x, 'hybrid_target_mlp', 'rank')], lambda x: hyb[(x, 'hybrid_frozen_mlp', 'rank')],
                lambda x: hyb[(x, 'hybrid_target_linear', 'rank')], lambda x: hyb[(x, 'hybrid_target_mlp', 'knn')],
                lambda x: registered[(x, KNN)], lambda x: registered[(x, 'mlp')]]
    rows1.append([f'Mean over the {words(k17)}'] + [pct(1 - statistics.mean(mean(c(x)) for x in COMBINED)) for c in columns1])
    # how the nested study selects (referee-panel revision of 2026-09-25, item B8): at the first fitting seed, as its design header
    # says, and without the rescoring of the finalists that the stochastic models of the main benchmark receive
    nested_header = ' '.join((REPO / LEARNED_MODULES['nested']).read_text().split('"""', 2)[1].split())
    seeds = HYBRID_DESIGN['fit_seeds']
    if f"inner splits at fit seed {seeds[0]}" not in nested_header or 'rescor' in nested_header or seeds != FACTS['bridge']['fit_seeds']:
        raise AssertionError('Table S-learned-nested: the nested study no longer selects at the first fitting seed without rescoring')
    finalists = knn.protocol['stochastic_finalists']
    write_table('tab_s_learned_nested',
                '\\textbf{The learned encoder under nested cross-validation: error.} Mean error (\\%) over the '
                f'{n_outer} test sets of the main benchmark. The first four columns are the classifier of '
                'Section~\\ref{supp:learned-method}, an encoder '
                'network followed by one ranking layer of class filters. It is read by the class filters or by a nearest-neighbor vote '
                'on the encoder\'s own ranking, and its encoder is the trained network, the same network never trained, or a trained '
                f'single linear layer. Each is tuned on the inner folds at the first fitting seed and refitted with the {words(len(seeds))} '
                f'fitting seeds. Unlike the stochastic models of the main benchmark, it does not rescore its {words(finalists)} best '
                f'candidates with all {words(len(seeds))} seeds. The untrained encoder keeps its initial weights, uniform on $[0,1)$ in '
                'its first two layers. ArrowFlow and the MLP are the registered results on the same test sets. HCV: hepatitis C virus.',
                'tab:s-learned-nested', head1, rows1, 'lrrrrrr', size='\\scriptsize', wide=True,
                subheads={**blocks, k17: 'All datasets'})
    p1, p2, p3 = P['P1'], P['P2'], P['P3']
    higher_p1 = f'all {k17} datasets' if len(p1['higher']) == k17 else counted(len(p1['higher']), k17, 'datasets')
    fact2 = g5_fact('tables/tab_s_learned_nested_contrasts.tex',
                    f"The trained encoder is more accurate than the untrained one on {higher_p1}, "
                    f"significantly on {len(p1['up'])}. Against the MLP it is ahead on {len(p2['higher'])}, significantly on "
                    f"{num_word(len(p2['up']))}, and significantly behind on {num_word(len(p2['down']))}; with the nearest-neighbor readout "
                    f"it is ahead of ArrowFlow on {len(p3['higher'])}, significantly on {num_word(len(p3['up']))}, and significantly "
                    f"behind on {num_word(len(p3['down']))}.",
                    f"significantly on {len(p1['up'])}.", f"significantly on {len(p1['up']) - 1}.")
    head2 = ['Dataset', two_lines('Trained $-$ untrained', 'encoder (pp)'), 'Holm $p$', two_lines('Trained encoder', '$-$ MLP (pp)'),
             'Holm $p$', two_lines('kNN readout $-$', 'ArrowFlow (pp)'), 'Holm $p$']
    rows2 = [[NAME[x], cell(p1['rows'][x]), pval(p1['rows'][x]['holm']), cell(p2['rows'][x]), pval(p2['rows'][x]['holm']),
              cell(p3['rows'][x]), pval(p3['rows'][x]['holm'])] for x in COMBINED]
    write_table('tab_s_learned_nested_contrasts',
                '\\textbf{The learned encoder under nested cross-validation: the three contrasts fixed before the run.} Accuracy '
                f'differences, in percentage points, with the {interval_note()}; positive favors the first-named model, and each '
                f'contrast has its own Holm adjustment across the {words(k17)} datasets. The first two contrasts read the classifier '
                'of Section~\\ref{supp:learned-method} by its class filters; the third reads its trained encoder\'s ranking with a nearest-neighbor vote, '
                f'the readout ArrowFlow uses. {fact2} HCV: hepatitis C virus.',
                'tab:s-learned-nested-contrasts', head2, rows2, 'lrrrrrr', size='\\scriptsize', wide=True, subheads=blocks)
    fact3 = g5_fact(('tables/tab_s_learned_swap.tex',),
                    f"With the trained encoder, ArrowFlow has the higher mean accuracy on {counted(len(prim['higher']), k17, 'datasets')}, "
                    f"significantly on {num_word(len(prim['up']))} and significantly lower on {num_word(len(prim['down']))}.",
                    f"on {len(prim['higher'])} of", f"on {len(prim['higher']) - 1} of", home='tables/tab_s_learned_swap.tex')
    head3 = ['Dataset', two_lines('Fixed encoder', '(ArrowFlow)'), two_lines('Trained', 'encoder'), two_lines('Untrained', 'encoder'),
             'MLP', two_lines('Trained $-$ fixed', 'Diff. (pp) [interval]'), two_lines('Trained $-$ fixed', '$p$'),
             two_lines('Trained $-$ fixed', 'Holm $p$')]
    rows3 = [[NAME[x], err(registered[(x, KNN)]), err(swap[(x, 'learned')]), err(swap[(x, 'random')]), err(registered[(x, 'mlp')]),
              cell(prim['rows'][x]), pval(prim['rows'][x]['p']), pval(prim['rows'][x]['holm'])] for x in COMBINED]
    columns3 = [lambda x: registered[(x, KNN)], lambda x: swap[(x, 'learned')], lambda x: swap[(x, 'random')],
                lambda x: registered[(x, 'mlp')]]
    rows3.append([f'Mean over the {words(k17)}'] + [pct(1 - statistics.mean(mean(c(x)) for x in COMBINED)) for c in columns3]
                 + [signed(prim['mean'], 2), '--', '--'])
    write_table('tab_s_learned_swap',
                '\\textbf{The encoder swap: ArrowFlow with the trained encoder.} Mean error (\\%) over the '
                f'{n_outer} test sets, and the contrast fixed before the run, in percentage points: ArrowFlow with the trained encoder '
                f'minus ArrowFlow with its fixed encoder. The contrast has the {interval_note()} and a Holm adjustment across the '
                f'{words(k17)} datasets. For every outer split, ArrowFlow keeps the configuration that its inner-fold tuning chose on '
                "that split's training partition, and nothing is retuned. Each view's encoder is the fixed encoder of "
                'Section~\\ref{sec:encoder} (the registered results), the trained encoder network, or the same network never trained. An '
                'encoder network reads the features without the polynomial expansion and uses no projection strategy, so its views '
                'differ only by their seeds. The MLP is the registered comparison model. '
                f'{fact3} HCV: hepatitis C virus.',
                'tab:s-learned-swap', head3, rows3, 'lrrrrrrr', size='\\scriptsize', wide=True,
                subheads={**blocks, k17: 'All datasets'})
    lr, rf, lm = S['learned_random'], S['random_fixed'], S['learned_mlp']
    fact4 = g5_fact('tables/tab_s_learned_swap_controls.tex',
                    f"The trained encoder beats the untrained one on {counted(len(lr['higher']), k17, 'datasets')}, significantly on "
                    f"{num_word(len(lr['up']))}. The untrained encoder is behind the fixed one on {k17 - len(rf['higher'])}, significantly "
                    f"on {num_word(len(rf['down']))}.",
                    f"on {len(lr['higher'])} of", f"on {len(lr['higher']) - 1} of")
    head4 = ['Dataset', two_lines('Trained $-$ untrained', '(pp) [interval]'), 'Holm $p$', two_lines('Untrained $-$ fixed', '(pp) [interval]'),
             'Holm $p$', two_lines('Trained $-$ MLP', '(pp) [interval]'), 'Holm $p$']
    rows4 = [[NAME[x], cell(lr['rows'][x]), pval(lr['rows'][x]['holm']), cell(rf['rows'][x]), pval(rf['rows'][x]['holm']),
              cell(lm['rows'][x]), pval(lm['rows'][x]['holm'])] for x in COMBINED]
    write_table('tab_s_learned_swap_controls',
                '\\textbf{The encoder swap: controls.} Accuracy differences between the arms of Table~\\ref{tab:s-learned-swap}, '
                f'in percentage points, with the {interval_note()}; positive favors the first-named arm. The design declared these '
                'contrasts secondary and descriptive, and only the contrast of Table~\\ref{tab:s-learned-swap} primary. Each has its own '
                f'Holm adjustment across the {words(k17)} datasets, and a Holm $p$ below 0.05 counts as significant within it. {fact4} '
                'HCV: hepatitis C virus.',
                'tab:s-learned-swap-controls', head4, rows4, 'lrrrrrr', size='\\scriptsize', wide=True, subheads=blocks)
    return dict(hyb=hyb, P=P, swap=swap, S=S, verdict=verdict, registered=registered, best=best, trails=trails, mean=mean,
                checked=checked, recomputed=recomputed, test_rows=test_rows, widths=dict(widths), pools=pools,
                secondary=[name for name in held['contrasts'] if name != 'PRIMARY'])


def learned_protocol_rows(runs):
    """Table S-protocols rows of the two studies of the learned encoder: the module whose header fixes the design, the freeze
    commit and its time, and the completed jobs."""
    rows = []
    for which, label, run, jobs in (('nested', 'learned encoder: hybrid classifier, nested cross-validation', LEARNED_NESTED,
                                     len(COMBINED) * FACTS['n_outer'] * len(HYBRID_ARMS)),
                                    ('swap', 'learned encoder: encoder swap inside ArrowFlow', LEARNED_SWAP,
                                     len(COMBINED) * FACTS['n_outer'] * len(SWAP_DESIGN['arms']))):
        completed = len(list((runs / run / 'results').glob('*.json')))
        if completed != jobs:
            raise AssertionError(f'{label}: {completed} completed jobs against {jobs} planned')
        rows.append([label, 'module header', tex(LEARNED_MODULES[which].split('/', 1)[1]), git_time_utc(LEARNED_FREEZE[which]),
                     f"\\texttt{{{LEARNED_FREEZE[which][:HASH_DIGITS]}}}", '--', f'{completed} / {jobs}', 'complete'])
    return rows


def render_learned_figure(enc):
    """Figure 9: the two contrasts of the learned encoder, drawn by render_figures.encoder_forest from the rows computed above;
    every drawn value must equal the row it was drawn from."""
    sys.path.insert(0, str(HERE))
    from render_figures import encoder_forest  # noqa: E402
    rows = [dict(panel=key, dataset=x, mean_difference=fam['rows'][x]['mean'], ci_low=fam['rows'][x]['lo'],
                 ci_high=fam['rows'][x]['hi'], holm_p=fam['rows'][x]['holm'])
            for key, fam in (('hybrid', enc['P']['P1']), ('swap', enc['S']['primary'])) for x in COMBINED]
    panels = (('hybrid', '(a) Trained $-$ untrained encoder', 'Accuracy difference (pp)'),
              ('swap', '(b) ArrowFlow: trained $-$ fixed', 'Accuracy difference (pp)'))
    drawn = encoder_forest(rows, FIG_DATA / 'fig9_learned_encoder.pdf', datasets=COMBINED, labels={x: NAME[x] for x in COMBINED},
                           panels=panels)
    for r in rows:
        d = drawn[(drawn['panel'] == r['panel']) & (drawn['dataset'] == r['dataset'])].iloc[0]
        if abs(d['difference_pp'] - 100 * r['mean_difference']) > 1e-9 or (d['holm_significant'] != (r['holm_p'] < 0.05)):
            raise AssertionError(f"learned encoder figure: the drawn mark of {r['panel']}/{r['dataset']} differs from its row")
    p1, prim, k17 = enc['P']['P1'], enc['S']['primary'], len(COMBINED)
    higher_a = f'all {k17} datasets' if len(p1['higher']) == k17 else counted(len(p1['higher']), k17, 'datasets')
    # the referee-panel revision of 2026-09-25 (items G1, B5, B10, B3, C1, B4): the scales and the unadjusted intervals, the
    # reference arm and the mean errors of both panels, the split wording, what the swap drops, the marks of each panel and the
    # datasets not used to develop the conversion
    from render_figures import ENCODER_STYLE  # noqa: E402
    if [marker for _, marker in ENCODER_STYLE] != ['o', 's']:
        raise AssertionError('learned encoder figure: the caption says circles in panel (a) and squares in panel (b)')
    error17 = lambda folds_of: pct(1 - statistics.mean(enc['mean'](folds_of(x)) for x in COMBINED))
    dev = {SANDBOX_KEY.get(r['dataset'], r['dataset']) for r in read_json(SANDBOX_RUNS / 'core_confirm.json')['results']}
    unused = [x for x in COMBINED if x not in dev]
    refs = supplement_refs(E7, ('supp:learned-swap', 'supp:learned-development'))
    caption = ('\\textbf{Can the ranking signal train the encoder?} Accuracy differences, in percentage points, with '
               f'{FACTS["confidence"]}\\% intervals (not simultaneous); positive favors the trained encoder. The two panels have '
               'different scales. The intervals are not adjusted for multiple comparisons, so an open mark\'s interval can exclude '
               'zero. (a)~The classifier of '
               'Section~\\ref{sec:learned-encoder}, an encoder network and one ranking layer of class filters, with the encoder '
               'trained by the converted motions, minus the same classifier with the encoder left at its random start. At that start, '
               'the weights of the encoder\'s first two layers are drawn uniformly from 0 to 1. Training lowers the mean error over '
               f"the datasets from {error17(lambda x: enc['hyb'][(x, 'hybrid_frozen_mlp', 'rank')])} to "
               f"{error17(lambda x: enc['hyb'][(x, 'hybrid_target_mlp', 'rank')])} percent. "
               '(b)~ArrowFlow with each view\'s fixed encoder replaced by the trained encoder network, minus ArrowFlow with its fixed '
               'encoder. For every split, both use the configuration that ArrowFlow\'s own tuning chose on the inner folds of that '
               "split's training partition. The swap also drops the polynomial expansion and the projection strategies "
               f"(Section~{refs['supp:learned-swap']}). The mean error falls from {error17(lambda x: enc['registered'][(x, KNN)])} to "
               f"{error17(lambda x: enc['swap'][(x, 'learned')])} percent. Marks are circles in panel~(a) and squares in panel~(b). "
               'Filled marks remain significant '
               f'after a Holm correction across the {words(k17)} datasets: {len(p1["up"])} in panel~(a) and {len(prim["up"])} in '
               f'panel~(b). The trained encoder has the higher mean accuracy on {higher_a} in '
               f'panel~(a) and on {len(prim["higher"])} of the {k17} in panel~(b). The target conversion was developed on '
               f"{words(len(dev))} of these datasets, all but {listing([NAME[x] for x in unused], 'and')} "
               f"(Section~{refs['supp:learned-development']}). HCV: hepatitis C virus.")
    (HERE / 'figures' / 'fig9_learned_encoder.tex').write_text(
        '% Figure: the learned encoder. Rendered by render_tables.py via render_figures.encoder_forest from the run files of the\n'
        '% two studies of the learned encoder; nothing numeric is typed here.\n'
        + ''.join(f'% supplement-ref {label}={number}\n' for label, number in refs.items())
        + '\\begin{figure}[!htbp]\n\\centering\n'
        f'\\includegraphics[width={FIGURE_WIDTH}\\textwidth]{{figures/data/fig9_learned_encoder.pdf}}\n'
        f'\\caption{{{caption}}}\n\\label{{fig:learned}}\n\\end{{figure}}\n')
    print(f'wrote {HERE / "figures" / "fig9_learned_encoder.tex"}')


# --------------------------------------------------------------------------------- the main-text figure
def cells_motion(t):
    """The main-text extract of the motion controls against the complete three-arm family."""
    header, rows = table_rows(t['tab_motion'])
    main = {(r[0], h): c for _, r in labelled(rows) for h, c in zip(header[1:], r[1:])}
    hs, rs = table_rows(t['tab_s_motion_family'])
    supp = {(r[0], h): c for _, r in labelled(rs) for h, c in zip(hs[1:], r[1:]) if h in header}
    return main, supp


def flip_motion(column):
    def change(cells, header):
        j = header.index(column)
        cells[j] = '+' + cells[j][3:] if cells[j].startswith('$-$') else '$-$' + cells[j].lstrip('+')
        return cells
    return change


# the two main-text data figures: with the captions of the referee-panel revision of 2026-09-25, an image at the full text width
# and its caption no longer fit one page, so both images take 92 percent of it
FIGURE_WIDTH = 0.92


def supplement_refs(section, labels):
    """The supplement numbers that a main-text section declares for these labels ('% supplement-ref label=number' lines), for a
    rendered caption that quotes them; build.sh checks every declared number against the built supplement."""
    declared = dict(re.findall(r'^% supplement-ref ([^=\s]+)=(\S+)$', (HERE / section).read_text(), re.M))
    missing = [x for x in labels if x not in declared]
    if missing:
        raise AssertionError(f'{section} declares no supplement number for {missing}')
    return {x: declared[x] for x in labels}


def motion_forest_wrapper(motion):
    table, holm = motion['table'], motion['holm']
    marked = sorted({x for arm in MOTION_ARMS for x in holm[arm]}, key=COMBINED.index)
    above = {arm: sum(table[(arm, x)]['mean_difference'] > 0 for x in COMBINED) for arm in MOTION_ARMS}
    purity_above = sum(motion['purity'][('views7', x)] > max(motion['purity'][(arm, x)] for arm in MOTION_ARMS)
                       for x in COMBINED)
    # the caption explains each control in the plain words of Section 7.5; S3.6 keeps the exact construction of every arm. The
    # referee-panel revision of 2026-09-25 (items B23, C4, G2) says "motions", draws permuted alignment afresh in every batch, notes
    # the unadjusted intervals, reads a scramble's mark against the frozen mark, and points to the direct tables of Section 7.5
    refs = supplement_refs(E7, ('tab:s-motion-scramble', 'tab:s-motion-purity'))
    caption = ('\\textbf{Is it the signal that helps?} (a)~ArrowFlow\'s accuracy minus that of each control, in percentage points, '
               f'with {FACTS["confidence"]}\\% intervals (not simultaneous); positive favors ArrowFlow. Frozen filters: the hidden '
               'filters never move. Permuted alignment: each motion reaches a filter of the same layer drawn at random, afresh in every '
               'batch. Random directions: the directions of the motions are shuffled. Both scrambled controls keep the number of '
               'motions, their sizes and '
               f'their total weight. Filled marks in panel (a) remain significant after a Holm correction across the {words(len(COMBINED))} '
               f'datasets within each control: {len(holm[MOTION_ARMS[0]])} datasets for {ARM[MOTION_ARMS[0]].lower()}, '
               + listing([f'{len(holm[a])} for {ARM[a].lower()}' for a in MOTION_ARMS[1:]], 'and')
               + '. The intervals are not adjusted for multiple comparisons, so an open mark\'s interval can exclude zero.'
               + f' In panel (a), ArrowFlow is more accurate than {ARM[MOTION_ARMS[0]].lower()} on {above[MOTION_ARMS[0]]} of the '
               f'{len(COMBINED)} datasets, '
               + listing([f'than {ARM[a].lower()} on {above[a]}' for a in MOTION_ARMS[1:]], 'and')
               + '. (b)~Neighborhood purity, how often an example\'s nearest training examples share its class: ArrowFlow minus each '
               f'control, in percentage points, on the test rows; descriptive, with no interval and no test, so its marks carry no fill '
               f'coding. ArrowFlow is purer than all three controls on '
               f'{purity_above} of the {len(COMBINED)} datasets. In both panels, a scramble\'s mark to the right of the mark of frozen '
               f'filters means that the scramble is less accurate than frozen filters, or in panel~(b) less pure. '
               f"Tables~{refs['tab:s-motion-scramble']} and~{refs['tab:s-motion-purity']} give these comparisons directly. "
               f'HCV: hepatitis C virus.')
    return ('% Figure: matched motion-signal controls. Rendered by render_tables.py via render_figures.motion_forest from the\n'
            '% motion-controls analysis outputs; nothing numeric is typed here.\n'
            + ''.join(f'% supplement-ref {label}={number}\n' for label, number in refs.items())
            + '\\begin{figure}[!htbp]\n\\centering\n'
            # the image takes FIGURE_WIDTH of the text width so that it and its caption fit one page (build.sh fails otherwise)
            f'\\includegraphics[width={FIGURE_WIDTH}\\textwidth]{{figures/data/fig8_motion_forest.pdf}}\n'
            f'\\caption{{{caption}}}\n\\label{{fig:motion-forest}}\n\\end{{figure}}\n')


def render_motion_figure(runs, motion):
    sys.path.insert(0, str(HERE))
    from render_figures import motion_forest  # noqa: E402
    d = runs / MOTION / 'analysis'
    motion_forest(d / 'motion_primary_family.csv', d / 'motion_purity.csv', FIG_DATA / 'fig8_motion_forest.pdf',
                  datasets=COMBINED, labels={x: NAME[x] for x in COMBINED},
                  arms=MOTION_ARMS, arm_labels={a: ARM[a] for a in MOTION_ARMS})
    (HERE / 'figures' / 'fig8_motion_forest.tex').write_text(motion_forest_wrapper(motion))
    print(f'wrote {HERE / "figures" / "fig8_motion_forest.tex"}')


def render_learning_curves(runs, diagnostics):
    """Figure S5: the learning curves of the training diagnostics, drawn by render_figures.learning_curves from the sealed table
    that render_diagnostics has verified; every plotted value must equal the value the renderer read."""
    sys.path.insert(0, str(HERE))
    from render_figures import learning_curves  # noqa: E402
    plotted = learning_curves(runs / DIAGNOSTICS_TABLES / 't5_learning_curves.csv', FIG_DATA / 'figS_learning_curves.pdf',
                              datasets=COMBINED, labels={x: NAME[x] for x in COMBINED}, updates=diagnostics['updates'],
                              step=diagnostics['curve_step'])
    for (key, x), values in diagnostics['curves'].items():
        drawn = plotted[(plotted['series'] == key) & (plotted['dataset'] == x)].sort_values('iteration')['error_pct'].tolist()
        if len(drawn) != len(values) or any(abs(a - b) > 1e-12 for a, b in zip(drawn, values)):
            raise AssertionError(f'learning curves: the figure draws another curve for {key} on {x} than the verified table holds')
    (HERE / 'figures' / 'figS_learning_curves.tex').write_text(
        '% Figure S5: learning curves. Rendered by render_tables.py via render_figures.learning_curves from the sealed\n'
        '% report-ready tables of the training diagnostics; nothing numeric is typed here.\n'
        '\\begin{figure}[!htbp]\n\\centering\n'
        '\\includegraphics[width=\\textwidth]{figures/data/figS_learning_curves.pdf}\n'
        f'\\caption{{{diagnostics["curve_caption"]}}}\n\\label{{fig:learning-curves}}\n\\end{{figure}}\n')
    print(f'wrote {HERE / "figures" / "figS_learning_curves.tex"}')


def render_hcv_task(runs):
    """HCV's target and the treatment-time measurements among its features, read from the pinned run manifest (E15)."""
    m = read_json(runs / '2026-09-14-newdata-batch2' / HCV / 'manifest.json')
    o, names = m['openml'], list(m['feature_names'])
    during = [f for f in names if re.match(r'(ALT_?(4|12|24|36|48)|ALT_after|RNA_(4|12|EOT|EF))', f)]
    if not during or o['default_target'] != 'Baselinehistological_staging':
        raise AssertionError('the HCV manifest no longer records a baseline target with treatment-time measurements')
    g5_fact(SUPP3,
            f"The HCV target is the baseline histological staging of the OpenML copy (data ID {o['data_id']}, version "
            f"{o['version']}), with class counts " + listing([str(c) for c in m['class_counts']], 'and') + '. Its '
            f"{len(names)} features include {len(during)} laboratory measurements taken during and after treatment, so the "
            f"task is a retrospective classification of the recorded stage rather than a prediction from information "
            f"available at baseline.",
            f"data ID {o['data_id']}", f"data ID {o['data_id'] + 1}", home=SUPP3)
    return dict(target=o['default_target'], during=during)


def render_comparator_warnings(runs):
    """Outer fits of the multilayer perceptron that reached their iteration cap before converging, counted from the
    ``fit_warnings`` of every result record of the three ArrowFlow runs (E09 of the external review)."""
    groups = {'benchmark': [runs / '2026-09-12-bridge-knn'],
              'further': [runs / '2026-09-14-newdata-batch1', runs / '2026-09-14-newdata-batch2']}
    fits, warned, per_fits, per_warned, kinds = Counter(), Counter(), Counter(), Counter(), set()
    for scope, dirs in groups.items():
        for d in dirs:
            for path in sorted((d / 'results').glob('*__mlp__r*f*.json')):
                dataset = path.name.split('__')[0]
                for m in read_json(path)['models']:
                    if m.get('stage', 'outer') != 'outer':
                        continue
                    fits[scope] += 1
                    per_fits[dataset] += 1
                    hits = m.get('fit_warnings') or []
                    kinds.update(w.get('category') for w in hits)
                    if hits:
                        warned[scope] += 1
                        per_warned[dataset] += 1
    total_fits, total_warned = sum(fits.values()), sum(warned.values())
    if total_fits != len(COMBINED) * FACTS['n_outer'] * 3 or kinds - {'ConvergenceWarning'}:
        raise AssertionError('the MLP fits counted for the convergence disclosure are not the three seeds of every outer '
                             'fold, or carry a warning that is not a ConvergenceWarning')
    always = [d for d in COMBINED if per_warned[d] == per_fits[d] and per_fits[d]]
    g5_fact(SUPP3,
            f'The multilayer perceptron reached its iteration cap before convergence on {total_warned} of its {total_fits} '
            f'outer fits, {warned["benchmark"]} of the {fits["benchmark"]} on the benchmark datasets and '
            f'{warned["further"]} of the {fits["further"]} on the further datasets, and on every fit of '
            + listing([NAME[d] for d in always], 'and') + '.',
            f'on {total_warned} of its', f'on {total_warned + 1} of its', home=SUPP3)
    return dict(fits=total_fits, warned=total_warned, always=always)


# --------------------------------------------------------------------------------- assembly
def render_controlled_experiments(runs, combined, components, knn=None, newdata=None):
    """Every table and the figure of the controlled experiments, and the guarded phrases they carry; the duplicate-free rerun is
    verified and returns facts but writes no table. The two families of 2026-09-23, depth with the corrected relay and the
    representation test, follow the families they extend."""
    render_hcv_task(runs)
    mlp_warnings = render_comparator_warnings(runs)
    motion = render_motion(runs)
    diagnostics = render_diagnostics(runs, combined)
    mechanism = render_mechanism(runs)
    depth = render_depth(runs)
    relay = render_signed_relay(runs)
    aggregation = render_aggregation(runs)
    dedup = render_dedup(runs)
    baselines = render_baselines(runs)
    representation = render_representation(runs, components)
    learned = render_learned_encoder(runs, knn, newdata)
    return dict(motion=motion, diagnostics=diagnostics, mechanism=mechanism, depth=depth, relay=relay, aggregation=aggregation,
                dedup=dedup, baselines=baselines, representation=representation, learned=learned, mlp_warnings=mlp_warnings,
                claims=list(G5_CLAIMS), mutations=list(G5_MUTATIONS))


# ----------------------------------------------------------------------------- Figure 7
def figure7_wrapper(contrasts):
    ok = contrasts[contrasts['kind'] == 'output_vs_knn']
    tr = contrasts[contrasts['kind'] == 'trained_vs_initial']
    above = int((ok['mean_difference'] >= 0).sum())
    trained_above = int((tr['mean_difference'] > 0).sum())
    probe_excl = int(((tr['ci_low'] > 0) | (tr['ci_high'] < 0)).sum())
    holm = contrasts[contrasts['holm_p_approximate'] < 0.05]
    kinds = {'output_vs_knn': 'prototype readout', 'trained_vs_initial': 'probe'}
    holm_names = listing([f"{LABEL[r['dataset_id']]} {kinds[r['kind']]}" for _, r in holm.iterrows()], 'and') if len(holm) else ''
    probe = FACTS['matched']['architectures'][FACTS['matched']['primary_architecture_index']]
    n, nd = len(contrasts), len(ok)
    significant = (of_total(len(holm), n, 'contrasts', 'remains', 'remain', capital=False) + ' significant'
                   + (f' ({holm_names})' if holm_names else ''))
    caption = (f'\\textbf{{Prototype-readout contrasts.}} (a)~Mean outer-fold test error in percent on the {words(nd)} benchmark '
               f'datasets of four classifiers. A footrule kNN probe reads the hidden ranking of a single-view '
               f'{widths_label(probe)} network before training (initial filters) and after training, with neighbor count and '
               f'weighting chosen on inner folds (matched study). The others are the prototype readout with $K={FACTS["views"]}$ '
               f'views and the seven-view footrule kNN control, footrule kNN on the same encoded views tuned on inner '
               f'folds (component ablation). Whiskers are $\\pm1$ outer-fold SD. (b)~The paired contrasts of '
               f'Table~\\ref{{tab:contrasts}}, trained minus initial probe and prototype readout minus kNN control, as accuracy '
               f'differences in percentage points with {FACTS["confidence"]}\\% corrected resampled-$t$ intervals (not '
               f'simultaneous); positive favors the first-named. The prototype readout is at or above the kNN control on '
               f'{num_word(above)} datasets and below it on {num_word(nd - above)}. The trained probe is above the initial '
               f'probe on {num_word(trained_above)}, and {of_total(probe_excl, words(nd), "probe intervals", "excludes", "exclude", capital=False)} '
               f'zero. After Holm adjustment, {significant}.')
    return ('% Figure 7: learning controls. Rendered by render_tables.py via render_figures.figure7 from the v3\n'
            '% contrasts, the matched v3 summary and the ablation summary; nothing numeric is typed here.\n'
            '\\begin{figure}[!htbp]\n\\centering\n'
            '\\includegraphics[width=\\textwidth]{figures/data/fig7_learning_controls.pdf}\n'
            f'\\caption{{{caption}}}\n\\label{{fig:learning-controls}}\n\\end{{figure}}\n')


def render_figure7(contrasts_csv, matched, ablation):
    sys.path.insert(0, str(HERE))
    from render_figures import figure7  # noqa: E402
    levels = ablation_levels_for_figure7(ablation)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump(levels, fh)
        extra = fh.name
    figure7(contrasts_csv, matched.dir / 'summary.json', FIG_DATA / 'fig7_learning_controls.pdf', extra_summary_json=[extra])
    Path(extra).unlink()
    contrasts = pd.read_csv(contrasts_csv)
    (HERE / 'figures' / 'fig7_learning_controls.tex').write_text(figure7_wrapper(contrasts))
    print(f'wrote {HERE / "figures" / "fig7_learning_controls.tex"}')


# ----------------------------------------------------------------------------- main
def main(argv=None):
    global TABLES
    opened = set()   # every file the render opens; S5 lists those outside the repository
    sys.addaudithook(lambda event, hook_args: opened.add(os.fspath(hook_args[0]))
                     if event == 'open' and hook_args and isinstance(hook_args[0], (str, os.PathLike)) else None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    ap.add_argument('--out', type=Path, default=TABLES)
    ap.add_argument('--holistic', type=Path, default=None, help='outputs of the combined analysis (default: RUNS/2026-09-14-holistic)')
    ap.add_argument('--no-figure7', action='store_true', help='skip the figure (tables only)')
    args = ap.parse_args(argv)
    TABLES = args.out
    TABLES.mkdir(parents=True, exist_ok=True)

    bridge = Family('bridge', args.runs / '2026-09-12-bridge')
    matched = Family('matched v3', args.runs / '2026-09-12-matched')
    ablation = Family('ablation', args.runs / '2026-09-12-ablation', 'ablation_summary.json')
    knn = Family('bridge-knn', args.runs / '2026-09-12-bridge-knn')
    training = Family('knn-training', args.runs / '2026-09-13-knn-training')
    knn_ablation = Family('knn-ablation', args.runs / '2026-09-13-knn-ablation', 'knn_ablation_summary.json')
    projected = Family('knn-projected', args.runs / '2026-09-13-knn-projected')
    knn_compare = args.runs / '2026-09-12-knn-vs-full'
    contrasts_csv = args.runs / '2026-09-12-contrasts' / 'primary_contrasts_v3.csv'
    for fam in (bridge, knn, matched, ablation, training, knn_ablation, projected):
        if fam.summary is None:
            raise SystemExit(f'{fam.name}: no verified summary in {fam.dir}')
    selections = read_json(ablation.dir / 'bridge_selections.json')
    main_table = read_json(knn_compare / 'main_table.json')
    load_facts(bridge, matched)
    referee = render_referee(args.runs / '2026-09-13-referee-analyses', bridge, knn, main_table, knn_compare)
    render_grids(knn, bridge, training, projected, args.runs)

    bench = render_bridge(bridge, knn, main_table)
    render_selected_configurations(ablation, selections)
    render_contrasts(contrasts_csv)
    knn_facts = render_knn_contrasts(knn_compare, bridge, knn, main_table)
    knn_records = selection_records(knn, KNN)
    selected = render_knn_selected_configurations(knn, knn_records)
    round_off = selection_round_off(knn_records, knn.protocol['fit_seeds'])
    train = render_training(training, knn, knn_records)
    sorting = render_projected(projected, knn, training)
    newdata = render_newdata(args.runs, knn, training, projected)
    further_selected = render_further_selected_configurations(knn, newdata)
    combined = render_combined(args.holistic or args.runs / HOLISTIC, args.runs, bridge, knn, main_table, training, projected, train,
                               sorting, newdata, referee)
    production = production_log(args.runs)
    kab = render_knn_ablation(knn_ablation, knn, knn_records, production)
    components = render_components(args.runs, args.holistic or args.runs / HOLISTIC, knn_ablation, kab, newdata, referee)
    matched_facts = render_matched(matched, contrasts_csv)
    ablation_facts = render_ablation(ablation)
    render_datasets(bridge, newdata)
    devlab_dir = args.runs / '2026-09-12-devlab'
    if (devlab_dir / 'readouts_summary.json').is_file() and (devlab_dir / 'permlvq_summary.json').is_file():
        render_laboratory(devlab_dir)
    render_protocols([(bridge, 'first benchmark (prototype readout)', '2026-09-12/bridge.json'),
                      (knn, 'ArrowFlow (nearest-neighbor readout)', '2026-09-12/bridge_knn.json'),
                      (matched, 'matched learning controls', '2026-09-12/matched_v3.json', DATASETS),
                      (ablation, 'component ablation and contrasts', '2026-09-12/ablation.json'),
                      (training, 'training controls', '2026-09-12/knn_training.json'),
                      (knn_ablation, 'component ablation of ArrowFlow, benchmark datasets', '2026-09-12/knn_ablation.json'),
                      (projected, 'sorting control', '2026-09-12/knn_projected.json'),
                      (newdata['B'][1], 'further datasets, batch 1', '2026-09-12/newdata_batch1.json'),
                      (newdata['B'][2], 'further datasets, batch 2', '2026-09-12/newdata_batch2.json'),
                      (components['family'], 'component ablation of ArrowFlow, further datasets', '2026-09-12/newdata_ablation.json')],
                     devlab_dir=devlab_dir, extra_rows=controlled_protocol_rows(args.runs) + learned_protocol_rows(args.runs))
    environments = [(fam.name, fam.environment) for fam in (matched, ablation)]
    if (devlab_dir / 'readouts_summary.json').is_file() and (devlab_dir / 'permlvq_summary.json').is_file():
        environments += [(f'laboratory {part}', read_json(path) if path.is_file() else {}) for part, path in
                         (('readouts', devlab_dir / 'environment.json'),
                          ('permutation LVQ', devlab_dir / 'permlvq' / 'environment.json'))]
    environments += [(knn.name, knn.environment), (training.name, training.environment),
                     (knn_ablation.name, knn_ablation.environment), (projected.name, projected.environment)]
    environments += [(fam.name, fam.environment) for fam in newdata['B'].values()]
    environments += [(components['family'].name, components['family'].environment)]
    render_environment(bridge.environment, environments, args.runs)
    compute = render_compute([(knn, 'ArrowFlow run', [KNN] + BRIDGE_MODELS[1:]), (bridge, 'First benchmark run', BRIDGE_MODELS)])
    controlled = render_controlled_experiments(args.runs, combined, components, knn, newdata)
    tests = render_signed_rank_tests(combined, controlled['baselines'], knn, training, newdata, controlled['relay'])

    # every cell of a main-text table equals the cell of the same row and column in its complete supplement
    # table(s); the mutation test proves on in-memory copies that the check fails on drift
    checks = drift_checks()
    run_drift_checks(WRITTEN, checks)
    mutation_self_test(checks)

    if not args.no_figure7:
        render_figure7(contrasts_csv, matched, ablation)
        render_motion_figure(args.runs, controlled['motion'])
        render_learning_curves(args.runs, controlled['diagnostics'])
        render_learned_figure(controlled['learned'])

    # every guarded number and count in the text is rebuilt from the run files and must be found, and no reader-visible
    # file may use a forbidden term; the check runs after every table and figure wrapper has been written
    claims = prose_claims(bench, knn_facts, knn, ablation_facts, train, selected, round_off, production, training, kab,
                          knn_ablation, referee, matched_facts)
    claims += results_fill2_claims(train, knn_records, sorting, newdata, ablation_facts, args.runs, projected, training, knn)
    runs_root, workspace = args.runs.resolve(), REPO.parent.resolve()
    reads = {Path(p).resolve() for p in opened}
    strays = sorted(str(q) for q in reads if workspace in q.parents and REPO not in q.parents and runs_root not in q.parents and q != runs_root)
    if strays:
        raise AssertionError(f'the renderer read workspace files outside the repository and the runs directory: {strays[:3]}')
    FACTS['run_inputs'] = sorted({q.relative_to(runs_root).parts[0] for q in reads if runs_root in q.parents})
    if any(not (runs_root / x).is_dir() and not x.endswith('.log') for x in FACTS['run_inputs']):
        raise AssertionError('the renderer read a file in the runs directory that is neither a run directory nor a production log; '
                             'Section S5 says the production scripts are read from their tracked copies')
    claims += restructure2_claims(bridge, newdata, combined, train, sorting, referee, bench, matched_facts)
    claims += components_claims(components, referee, combined['ready'])
    claims += controlled['claims']
    claims += simplify_claims(combined, newdata, controlled, components, referee)
    claims += referee_claims(combined, newdata, controlled, components, referee, selected, further_selected, round_off, knn, training,
                             projected, sorting, knn_facts, kab, compute)
    claims += approved_edits_claims(combined, newdata, controlled, components, tests, knn)
    claims += final_round_claims(combined, newdata, controlled, components, tests, knn)
    claims += learned_encoder_claims(controlled, knn)
    texts = prose_texts()
    check_prose_claims(texts, claims)
    prose_mutation_test(texts, claims, extra=controlled['mutations'] + list(PLAIN_MUTATIONS))
    print(f'{len(WRITTEN)} tables written to {TABLES}')


if __name__ == '__main__':
    main()
