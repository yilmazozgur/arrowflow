#!/usr/bin/env python3
"""Render the data figures of the ArrowFlow v3 manuscript from run summaries.

Every mark is read from a summary written by the experiment harness; nothing
numeric is typed in this file or in the captions.  Output is vector PDF sized to
the 6.5-inch text width of the article format (letter paper, 1-inch margins) so
that ``\\includegraphics[width=\\textwidth]`` applies no scaling and every glyph
is at least 8 pt on the printed page.

Function (summary paths in, PDF path out):

    figure7(contrasts_csv, matched_summary_json, out_pdf, extra_summary_json=())
        Learning controls: per dataset the error of the initial-prototype probe,
        the trained-prototype probe, the prototype readout and the footrule
        kNN control, plus the corrected intervals of the two paired contrasts.
        The four model ids come from the contrasts file; their error levels are
        looked up in the matched display summary and, optionally, in further
        summaries of the same shape (for example an ablation summary).

    learning_curves(curves_csv, out_pdf, datasets, labels, updates)
        Training diagnostics: test error against the update count, one panel per
        dataset, for ArrowFlow's seven-view nearest-neighbor readout and for the
        nearest-neighbor and prototype readouts of one view (Figure S2).

Command line::

    python render_figures.py --fig7 CONTRASTS.csv MATCHED_SUMMARY.json OUT.pdf

render_tables.py calls figure7, motion_forest and learning_curves itself when it
renders the tables.

Style: the dataviz reference palette in its colour-blind-safe slot order
(checked with the skill's validator on a white surface), one palette across
all data figures, thin marks, a hairline horizontal grid, legends plus marker
shape or line style as a second identity channel, no chart junk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TEXT_WIDTH_IN = 6.5

# dataviz reference palette, light mode, fixed slot order (assigned in sequence,
# never cycled).  Slots 1-4 pass the adjacent-pair checks on white; slots 1-3
# pass all pairs.
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
INK, INK_SOFT, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#ffffff'

DATASET_LABELS = {
    'iris': 'Iris', 'wine': 'Wine', 'digits': 'Digits', 'breast_cancer': 'Breast cancer',
    'wine_quality': 'Wine quality', 'vehicle': 'Vehicle', 'segment': 'Segment',
}

RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5,
    'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.frameon': False, 'legend.handlelength': 2.4,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK_SOFT, 'ytick.color': INK_SOFT, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'lines.linewidth': 1.4, 'lines.markersize': 4.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
}
WHISKER = dict(elinewidth=0.6, capsize=1.5, capthick=0.6)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _label(dataset_id):
    return DATASET_LABELS.get(dataset_id, dataset_id.replace('_', ' ').capitalize())


def _require(frame, columns, source):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f'{source}: missing columns {missing}')


def _mark(color, marker, filled, linestyle='-'):
    """Line/marker style with the surface ring (filled) or an open marker (hollow)."""
    style = dict(color=color, marker=marker, linestyle=linestyle, markeredgewidth=0.8)
    if filled:
        style.update(markerfacecolor=color, markeredgecolor=SURFACE)
    else:
        style.update(markerfacecolor=SURFACE, markeredgecolor=color)
    return style


def _save(fig, out_pdf):
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # No creation date: identical inputs give byte-identical PDFs.
    fig.savefig(out_pdf, format='pdf', metadata={'CreationDate': None, 'Creator': 'render_figures.py'})
    plt.close(fig)
    print(f'wrote {out_pdf}')
    return out_pdf


def _print(table, title):
    print(f'--- {title}')
    print(table.to_string(index=False))


# ----------------------------------------------------------------------------
# Figure 7: learning controls (primary contrasts + matched summary)
# ----------------------------------------------------------------------------
ROLES = (  # role -> (legend label or None, colour, marker, filled, x offset)
    ('initial', 'Probe on the initial filters (one view)', PALETTE[0], 'o', False, -0.27),
    ('trained', 'Probe on the trained filters (one view)', PALETTE[0], 'o', True, -0.09),
    ('output', 'Prototype readout (seven views)', PALETTE[1], 's', True, 0.09),
    ('knn', None, PALETTE[2], '^', True, 0.27),
)
CONTRAST_STYLE = {'probe': (PALETTE[0], 'o', -0.12), 'readout': (PALETTE[1], 's', 0.12)}


def _contrast_kind(model_a, model_b):
    b = str(model_b).lower()
    if 'untrained' in b or 'initial' in b:
        return 'probe'
    if 'knn' in b or 'footrule' in b:
        return 'readout'
    raise ValueError(f'cannot classify the contrast {model_a} - {model_b}: '
                     'expected the second model to be an initial/untrained probe or a footrule kNN')


def _load_error_levels(paths):
    """(dataset_id, model_id) -> (mean error, outer-fold SD or None) from display summaries.

    Accepts the matched display summary shape ``{"summaries": {dataset: [rows]}}``,
    a bare ``{dataset: [rows]}`` mapping, or a list of rows carrying ``dataset_id``.
    Rows are metric records with ``model_id``, ``metric``, ``mean`` and
    ``outer_fold_sd``; rows whose metric is not ``error`` are ignored.
    """
    levels = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        summaries = payload.get('summaries', payload) if isinstance(payload, dict) else payload
        if isinstance(summaries, dict):
            items = [(dataset, row) for dataset, rows in summaries.items() for row in rows]
        elif isinstance(summaries, list):
            items = [(row['dataset_id'], row) for row in summaries]
        else:
            raise ValueError(f'{path}: unrecognised summary shape')
        for dataset, row in items:
            if row.get('metric', 'error') != 'error':
                continue
            sd = row.get('outer_fold_sd')
            levels[(dataset, row['model_id'])] = (float(row['mean']), None if sd is None else float(sd))
    return levels


def figure7(contrasts_csv, matched_summary_json, out_pdf, extra_summary_json=()):
    """Four readouts per dataset (a) and the two paired contrasts with corrected intervals (b)."""
    contrasts = pd.read_csv(contrasts_csv)
    _require(contrasts, ['dataset_id', 'model_a', 'model_b', 'mean_difference', 'ci_low', 'ci_high'], contrasts_csv)
    contrasts = contrasts.assign(kind=[_contrast_kind(a, b) for a, b in zip(contrasts['model_a'], contrasts['model_b'])])
    levels = _load_error_levels([matched_summary_json, *extra_summary_json])
    datasets = list(dict.fromkeys(contrasts['dataset_id']))
    per = {}
    for dataset in datasets:
        sub = contrasts[contrasts['dataset_id'] == dataset]
        if sorted(sub['kind']) != ['probe', 'readout']:
            raise ValueError(f'{dataset}: expected one probe and one readout contrast, found {list(sub["kind"])}')
        probe = sub[sub['kind'] == 'probe'].iloc[0]
        readout = sub[sub['kind'] == 'readout'].iloc[0]
        per[dataset] = {'initial': probe['model_b'], 'trained': probe['model_a'],
                        'output': readout['model_a'], 'knn': readout['model_b'],
                        'probe': probe, 'readout': readout}
    knn_ids = {per[d]['knn'] for d in datasets}
    multiview = all(('multiview' in k) or ('multi_view' in k) for k in knn_ids)
    knn_short = 'seven-view footrule kNN control' if multiview else 'input footrule kNN'
    knn_label = 'Seven-view footrule kNN control' if multiview else 'Input footrule kNN (single view)'
    rows = []
    with plt.rc_context(RC):
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.9), sharex=True,
                                         layout='constrained', height_ratios=[3, 2])
        for i, dataset in enumerate(datasets):
            for role, label, color, marker, filled, dx in ROLES:
                model = per[dataset][role]
                if (dataset, model) not in levels:
                    raise KeyError(f'no error level for {dataset}/{model} in the summaries given')
                mean, sd = levels[(dataset, model)]
                ax_a.errorbar([i + dx], [100 * mean], yerr=None if sd is None else [100 * sd],
                              **_mark(color, marker, filled, 'none'), **WHISKER)
                rows.append(dict(dataset=dataset, role=role, model_id=model, error_pct=100 * mean,
                                 fold_sd_pct=None if sd is None else 100 * sd))
            for kind, (color, marker, dx) in CONTRAST_STYLE.items():
                row = per[dataset][kind]
                d = 100 * row['mean_difference']
                ax_b.errorbar([i + dx], [d], yerr=[[d - 100 * row['ci_low']], [100 * row['ci_high'] - d]],
                              **_mark(color, marker, True, 'none'), **WHISKER)
                rows.append(dict(dataset=dataset, role=f'contrast:{kind}', model_id=f'{row["model_a"]} - {row["model_b"]}',
                                 error_pct=d, fold_sd_pct=None, ci_low_pp=100 * row['ci_low'], ci_high_pp=100 * row['ci_high']))
        ax_b.axhline(0, color=AXIS, linewidth=0.6, zorder=0)
        ax_a.set_ylim(bottom=0)
        ax_a.margins(y=0.15)
        ax_b.margins(y=0.3)
        ax_a.set_ylabel('Mean error (%)')
        ax_b.set_ylabel('Accuracy difference (pp)')
        ax_a.set_title('(a) Error of the four classifiers (whiskers: $\\pm 1$ outer-fold SD)', loc='left')
        ax_b.set_title('(b) Paired accuracy differences with corrected 95% intervals; positive favors the first-named',
                       loc='left')
        ax_b.set_xticks(range(len(datasets)))
        ax_b.set_xticklabels([_label(d) for d in datasets])
        ax_b.set_xlim(-0.6, len(datasets) - 0.4)
        handles_a = [Line2D([], [], label=label or knn_label, **_mark(color, marker, filled, 'none'))
                     for _, label, color, marker, filled, _ in ROLES]
        ax_a.legend(handles=handles_a, loc='upper left', ncol=2)
        handles_b = [Line2D([], [], label='Trained $-$ initial probe', **_mark(PALETTE[0], 'o', True, 'none')),
                     Line2D([], [], label=f'Prototype readout $-$ {knn_short}', **_mark(PALETTE[1], 's', True, 'none'))]
        ax_b.legend(handles=handles_b, loc='upper left', ncol=2)
        _save(fig, out_pdf)
    table = pd.DataFrame(rows)
    _print(table, f'figure7 <- {contrasts_csv} + {matched_summary_json}')
    return table


# ----------------------------------------------------------------------------
# Motion controls: the three-arm forest and the neighborhood-purity differences
# ----------------------------------------------------------------------------
ARM_STYLE = (  # arm -> (colour, marker)
    (PALETTE[0], 'o'),
    (PALETTE[1], 's'),
    (PALETTE[2], '^'),
)
ARM_OFFSET = (0.26, 0.0, -0.26)


def motion_forest(primary_family_csv, purity_csv, out_pdf, datasets, labels, arms, arm_labels, alpha=0.05):
    """(a) ArrowFlow minus each control arm with corrected intervals, (b) the purity difference of the same arms.

    Every mark is read from the two analysis outputs: the primary-family contrasts (difference, interval and the
    Holm-adjusted p value that decides whether a mark is filled) and the neighborhood purity of the outer test rows.
    """
    family = pd.read_csv(primary_family_csv)
    _require(family, ['arm', 'dataset', 'mean_difference', 'ci_low', 'ci_high', 'holm_p_approximate'], primary_family_csv)
    purity = pd.read_csv(purity_csv)
    _require(purity, ['dataset', 'arm', 'rows', 'mean'], purity_csv)
    purity = purity[purity['rows'] == 'outer_test']
    contrast = {(r['arm'], r['dataset']): r for _, r in family.iterrows()}
    level = {(r['arm'], r['dataset']): float(r['mean']) for _, r in purity.iterrows()}
    missing = [k for a in arms for d in datasets for k in [(a, d)] if k not in contrast or k not in level]
    if missing or any(('views7', d) not in level for d in datasets):
        raise KeyError(f'motion_forest: no contrast or purity for {missing[:3]}')
    rows = []
    with plt.rc_context(RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 5.7), sharey=True, layout='constrained',
                                         width_ratios=[3, 2.1])
        for j, arm in enumerate(arms):
            color, marker = ARM_STYLE[j]
            for i, dataset in enumerate(datasets):
                r = contrast[(arm, dataset)]
                y = len(datasets) - 1 - i + ARM_OFFSET[j]
                d = 100 * float(r['mean_difference'])
                filled = float(r['holm_p_approximate']) < alpha
                ax_a.errorbar([d], [y], xerr=[[d - 100 * float(r['ci_low'])], [100 * float(r['ci_high']) - d]],
                              **_mark(color, marker, filled, 'none'), **WHISKER)
                gap = 100 * (level[('views7', dataset)] - level[(arm, dataset)])
                # panel (b) carries no test, so every mark has one style: no significance coding (referee panel of 2026-09-23,
                # item A23: the panel once reused panel (a)'s fills)
                ax_b.plot([gap], [y], **_mark(color, marker, True, 'none'))
                rows.append(dict(dataset=dataset, arm=arm, difference_pp=d, ci_low_pp=100 * float(r['ci_low']),
                                 ci_high_pp=100 * float(r['ci_high']), holm_p=float(r['holm_p_approximate']),
                                 holm_significant=filled, purity_difference_points=gap))
        for ax, title in ((ax_a, '(a) Accuracy difference (pp)'), (ax_b, '(b) Purity difference')):
            ax.axvline(0, color=AXIS, linewidth=0.6, zorder=0)
            ax.set_title(title, loc='left')
            ax.grid(axis='y', color=GRID, linewidth=0.5)
            ax.grid(axis='x', visible=False)
            ax.set_ylim(-0.7, len(datasets) - 0.3)
        ax_a.set_yticks(range(len(datasets)))
        ax_a.set_yticklabels([labels[d] for d in reversed(datasets)])
        ax_a.set_xlabel('ArrowFlow $-$ control (pp)')
        ax_b.set_xlabel('ArrowFlow $-$ control (points)')
        handles = [Line2D([], [], label=arm_labels[a], **_mark(*ARM_STYLE[j], True, 'none')) for j, a in enumerate(arms)]
        handles.append(Line2D([], [], label='(a) open: not significant after correction', **_mark(INK_SOFT, 'o', False, 'none')))
        fig.legend(handles=handles, loc='outside lower center', ncol=2)
        _save(fig, out_pdf)
    table = pd.DataFrame(rows)
    _print(table, f'motion_forest <- {primary_family_csv} + {purity_csv}')
    return table


# ----------------------------------------------------------------------------
# The learned encoder (Figure 9)
# ----------------------------------------------------------------------------
ENCODER_STYLE = ((PALETTE[0], 'o'), (PALETTE[1], 's'))   # panel -> (colour, marker); colour and panel both carry identity


def encoder_forest(rows, out_pdf, datasets, labels, panels, alpha=0.05):
    """Two paired contrasts of the learned encoder, one panel each, with corrected intervals: (a) the trained minus the untrained
    encoder under the same class-filter layer, (b) ArrowFlow with the trained encoder minus ArrowFlow with its fixed encoder.

    ``rows`` holds one record per (panel, dataset) with the mean accuracy difference, its interval and the Holm-adjusted p value
    that decides whether a mark is filled; render_tables.py computes them from the run files and checks the drawn table back.
    ``panels`` is a sequence of (key, title, x-axis label)."""
    frame = pd.DataFrame(rows)
    _require(frame, ['panel', 'dataset', 'mean_difference', 'ci_low', 'ci_high', 'holm_p'], 'encoder_forest rows')
    contrast = {(r['panel'], r['dataset']): r for _, r in frame.iterrows()}
    missing = [(k, d) for k, _, _ in panels for d in datasets if (k, d) not in contrast]
    if missing:
        raise KeyError(f'encoder_forest: no contrast for {missing[:3]}')
    drawn = []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH_IN, 5.4), sharey=True, layout='constrained')
        for j, (ax, (key, title, xlabel)) in enumerate(zip(axes, panels)):
            color, marker = ENCODER_STYLE[j]
            for i, dataset in enumerate(datasets):
                r = contrast[(key, dataset)]
                y = len(datasets) - 1 - i
                d = 100 * float(r['mean_difference'])
                lo, hi = 100 * float(r['ci_low']), 100 * float(r['ci_high'])
                filled = float(r['holm_p']) < alpha
                ax.errorbar([d], [y], xerr=[[d - lo], [hi - d]], **_mark(color, marker, filled, 'none'), **WHISKER)
                drawn.append(dict(panel=key, dataset=dataset, difference_pp=d, ci_low_pp=lo, ci_high_pp=hi,
                                  holm_p=float(r['holm_p']), holm_significant=filled))
            ax.axvline(0, color=AXIS, linewidth=0.6, zorder=0)
            ax.set_title(title, loc='left')
            ax.set_xlabel(xlabel)
            ax.grid(axis='y', color=GRID, linewidth=0.5)
            ax.grid(axis='x', visible=False)
            ax.set_ylim(-0.7, len(datasets) - 0.3)
        axes[0].set_yticks(range(len(datasets)))
        axes[0].set_yticklabels([labels[d] for d in reversed(datasets)])
        handles = [Line2D([], [], label='filled: significant after the Holm correction', **_mark(INK_SOFT, 'o', True, 'none')),
                   Line2D([], [], label='open: not significant', **_mark(INK_SOFT, 'o', False, 'none'))]
        fig.legend(handles=handles, loc='outside lower center', ncol=2)
        _save(fig, out_pdf)
    table = pd.DataFrame(drawn)
    _print(table, 'encoder_forest <- render_tables.render_learned_encoder')
    return table


# ----------------------------------------------------------------------------
# Training diagnostics: learning curves (Figure S2)
# ----------------------------------------------------------------------------
LEARNING_SERIES = (  # key -> (level, measure, legend label, colour, line style); colour and style both carry identity
    ('seven_knn', 'seven_view_majority', 'knn_test_accuracy',
     'ArrowFlow: nearest-neighbor readout, seven views, majority vote', PALETTE[0], '-'),
    ('one_knn', 'per_view', 'knn_test_accuracy',
     'Nearest-neighbor readout of one view, mean over the seven views', PALETTE[2], '--'),
    ('one_proto', 'per_view', 'output_rule_test_accuracy',
     'Prototype readout of one view, mean over the seven views', PALETTE[1], ':'),
)


def _panel_title(label, width=20):
    """A dataset name that is too long for a narrow panel breaks after a hyphen instead of running into its neighbor."""
    if len(label) <= width or '-' not in label:
        return label
    cut = label.rfind('-', 0, width) + 1 or label.find('-') + 1
    return label[:cut] + '\n' + label[cut:]


def learning_curves(curves_csv, out_pdf, datasets, labels, updates, step=10, ncols=4):
    """Test error against the update count, one panel per dataset, for three readouts of the same fitted networks.

    Every mark is read from the report-ready learning-curve table of the training diagnostics (scheduled snapshots only;
    the checkpoint rows are separate points and are not part of a curve). Error is one minus the mean test accuracy over
    the outer folds, and for the per-view readouts also over the views. Each panel keeps its own error scale.
    """
    frame = pd.read_csv(curves_csv)
    _require(frame, ['scope', 'dataset_id', 'snapshot', 'iteration', 'level', 'measure', 'mean'], curves_csv)
    frame = frame[(frame['scope'] == 'dataset') & (frame['snapshot'] == 'scheduled')]
    expected = list(range(0, updates + 1, step))
    rows = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_WIDTH_IN, 6.6), layout='constrained')
        grid = fig.add_gridspec(-(-len(datasets) // ncols), ncols)
        for i, dataset in enumerate(datasets):
            ax = fig.add_subplot(grid[i // ncols, i % ncols])
            for key, level, measure, _, color, style in LEARNING_SERIES:
                part = frame[(frame['dataset_id'] == dataset) & (frame['level'] == level) & (frame['measure'] == measure)]
                part = part.sort_values('iteration')
                steps = [int(v) for v in part['iteration']]
                if steps != expected:
                    raise ValueError(f'learning_curves: {dataset}/{key} has snapshots {steps[:3]}..., expected every {step} '
                                     f'updates from 0 to {updates}')
                error = [100 * (1 - float(v)) for v in part['mean']]
                ax.plot(steps, error, color=color, linestyle=style, linewidth=1.3, solid_capstyle='round')
                rows += [dict(dataset=dataset, series=key, iteration=s, error_pct=e) for s, e in zip(steps, error)]
            ax.set_title(_panel_title(labels[dataset]), loc='left', fontsize=8.5)
            ax.set_xlim(0, updates)
            ax.set_xticks([0, updates // 2, updates])
            ax.margins(y=0.15)
            if i % ncols == 0:
                ax.set_ylabel('Test error (%)')
            if i >= len(datasets) - ncols:
                ax.set_xlabel('Update')
        handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.3, label=label)
                   for _, _, _, label, color, style in LEARNING_SERIES]
        if len(datasets) % ncols:
            spare = fig.add_subplot(grid[len(datasets) // ncols, len(datasets) % ncols:])
            spare.axis('off')
            spare.legend(handles=handles, loc='center left', handlelength=3.0)
        else:
            fig.legend(handles=handles, loc='outside lower center', handlelength=3.0)
        _save(fig, out_pdf)
    table = pd.DataFrame(rows)
    _print(table.pivot_table(index=['dataset', 'series'], columns='iteration', values='error_pct').iloc[:, [0, -1]].reset_index(),
           f'learning_curves <- {curves_csv}')
    return table


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fig7', nargs=3, metavar=('CONTRASTS_CSV', 'MATCHED_SUMMARY_JSON', 'OUT_PDF'), required=True,
                        help='render the learning-controls figure to OUT_PDF')
    parser.add_argument('--fig7-extra-summary', type=Path, action='append', default=[],
                        help='additional summary JSON(s) holding error levels for figure 7 model ids')
    args = parser.parse_args(argv)
    contrasts_csv, summary_json, out_pdf = args.fig7
    figure7(contrasts_csv, summary_json, out_pdf, extra_summary_json=args.fig7_extra_summary)


if __name__ == '__main__':
    main()
