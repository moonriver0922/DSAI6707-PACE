"""Generate paper figures from the frozen PACE records, without model inference."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
BLUE, GRAY, INK = '#4C72B0', '#8A9199', '#30343B'
LIGHT, GRID = '#F2F4F7', '#E8EAED'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'axes.titleweight': 'normal', 'axes.edgecolor': '#B8BDC4',
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    'legend.frameon': False,
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=.10)
    fig.savefig(OUT / f'{name}.png', dpi=240, bbox_inches='tight', pad_inches=.10)
    plt.close(fig)


def xgrid(ax):
    ax.grid(axis='x', color=GRID, lw=.6)
    ax.set_axisbelow(True)


online = ROOT / 'outputs/pace/gate2_cap_align_15seed'
with (online / 'pace_gate2_15seed_paired.csv').open() as stream:
    rows = list(csv.DictReader(stream))
summary = json.loads((online / 'pace_gate2_15seed_summary.json').read_text())
with (ROOT / 'outputs/pace/gate1/pace_gate1_results.csv').open() as stream:
    replay = list(csv.DictReader(stream))

runs = {}
for method, prefix in [('pace', 'pace'), ('b0_archive', 'b0')]:
    runs[method] = []
    for row in rows:
        run = json.loads((online / f"{method}_s{row['seed']}.json").read_text())
        for metric in ['test_acc', 'test_cost']:
            assert abs(float(row[f'{prefix}_{metric}']) - run[metric]) < 1e-12
        assert int(row[f'{prefix}_calls']) == run['calls_used']
        runs[method].append(run)
d = np.array([float(row['delta_acc']) * 100 for row in rows])
assert len(rows) == 15 and abs(d.mean() - 1.7) < 1e-10
assert (sum(d > 1e-10), sum(abs(d) < 1e-10), sum(d < -1e-10)) == (8, 4, 3)
lo, hi = np.array(summary['ci95_delta_acc']) * 100

# Main online result: ordinary horizontal bars and a single mean interval.
fig, (ax, ci) = plt.subplots(
    1, 2, figsize=(6.7, 4.4), layout='constrained',
    gridspec_kw={'width_ratios': [2.5, 1.4]},
)
y = np.arange(15)
ax.barh(y, d, color=BLUE, height=.58)
ax.axvline(0, color=INK, lw=.8)
ax.axhline(2.5, color='#C9CDD3', lw=.7, ls='--')
for i, value in enumerate(d):
    ax.text(value + (.22 if value >= 0 else -.22), i,
            f'{value:+.1f}' if abs(value) > 1e-10 else '0.0',
            ha='left' if value >= 0 else 'right', va='center', fontsize=8)
ax.set_yticks(y, [row['seed'] + (' *' if i < 3 else '') for i, row in enumerate(rows)], fontsize=8)
ax.set_ylim(14.6, -.6)
ax.set_xlim(-7.5, 14)
ax.set_xlabel('Accuracy difference (pp)')
ax.set_ylabel('Search seed')
ax.set_title('(a) PACE minus B0+Archive', loc='left', pad=10)
xgrid(ax)
ci.axvline(0, ymin=.36, ymax=.67, color=GRAY, lw=.8)
ci.errorbar(d.mean(), .52, xerr=[[d.mean() - lo], [hi - d.mean()]],
            fmt='o', color=BLUE, ecolor=BLUE, capsize=5, ms=5, lw=1.5)
ci.text(.5, .80, f'Mean: +{d.mean():.2f} pp', transform=ci.transAxes,
        ha='center', fontsize=10)
ci.text(.5, .22, f'95% CI [{lo:.2f}, {hi:.2f}]\np = 0.150\n8 wins / 4 ties / 3 losses',
        transform=ci.transAxes, ha='center', fontsize=8.5, linespacing=1.6)
ci.set_xlim(-1.8, 5.5)
ci.set_ylim(0, 1)
ci.set_yticks([])
ci.set_xticks([-1, 0, 2, 4])
ci.set_xlabel('Accuracy difference (pp)')
ci.set_title('(b) Mean and 95% CI', loc='left', pad=10)
ci.spines['left'].set_visible(False)
save(fig, 'online_paired')

# Offline summary: two direct comparisons, with no decorative point clouds.
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.6), layout='constrained')
for ax, metric, scale, title, xlabel in [
    (axes[0], 'false_elim_rate', 100, '(a) False elimination', 'Reference-Pareto prompts removed (%)'),
    (axes[1], 'n_retained', 1, '(b) Retained candidates', 'Number of retained prompts'),
]:
    for level, method, color in [(1, 'b0_point_nsga', GRAY), (0, 'pace', BLUE)]:
        values = np.array([float(row[metric]) * scale for row in replay
                           if row['method'] == method and int(row['budget']) == 640])
        assert len(values) == 20
        mean = values.mean()
        ax.barh(level, mean, color=color, height=.48)
        ax.text(mean + (1.0 if scale == 100 else .25), level,
                f'{mean:.1f}' + ('%' if scale == 100 else ''), va='center', fontsize=9)
    ax.set_yticks([1, 0], ['B0 (capacity 8)', 'PACE (capacity 16)'], fontsize=8)
    ax.set_ylim(-.7, 1.7)
    ax.set_xlim(0, 25 if scale == 100 else 16.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, loc='left', pad=10)
    xgrid(ax)
save(fig, 'offline_replay')

# Endpoint accuracy and cost: conventional scatter plot without crossing lines.
fig, ax = plt.subplots(figsize=(6.7, 2.9), layout='constrained')
for prefix, label, color, marker in [('b0', 'B0+Archive', GRAY, 'o'), ('pace', 'PACE', BLUE, 's')]:
    x = np.array([float(row[f'{prefix}_test_cost']) for row in rows])
    y = np.array([float(row[f'{prefix}_test_acc']) * 100 for row in rows])
    ax.scatter(x, y, color=color, marker=marker, s=24, alpha=.72, label=label)
    ax.scatter(x.mean(), y.mean(), color=color, marker=marker, s=85,
               edgecolors=INK, linewidths=1, zorder=4)
    ax.annotate(f'{label} mean\n{y.mean():.1f}%, {x.mean():.1f} tokens',
                (x.mean(), y.mean()), xytext=(-82, -42) if prefix == 'b0' else (20, 18),
                textcoords='offset points', fontsize=8, color=INK,
                arrowprops={'arrowstyle': '-', 'color': color, 'lw': .8})
ax.set_xlim(100, 130)
ax.set_ylim(62, 91)
ax.set_xlabel('Mean deployment tokens per example')
ax.set_ylabel('Held-out accuracy (%)')
ax.grid(color=GRID, lw=.6)
ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=9)
save(fig, 'accuracy_cost')

# Method flow uses the same muted palette and plain boxes.
fig, ax = plt.subplots(figsize=(6.7, 3.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

def box(x, y, title, body, highlight=False):
    ax.add_patch(FancyBboxPatch((x, y), 2.7, 1.1,
        boxstyle='round,pad=0.05,rounding_size=0.05',
        ec=BLUE if highlight else '#ADB3BB', fc='#EDF2F9' if highlight else LIGHT, lw=.9))
    ax.text(x + 1.35, y + .78, title, ha='center', va='center', fontsize=9, color=INK)
    ax.text(x + 1.35, y + .34, body, ha='center', va='center', fontsize=7.5, linespacing=1.3)

def arrow(start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
        mutation_scale=10, lw=1, color=GRAY))

box(.1, 3.5, '1. Candidate pool', 'Retained prompts + offspring')
box(3.6, 3.5, '2. Paired Racing', 'Shared-sample observations')
box(7.1, 3.5, '3. Evidence gate', r'$n_{pq}\geq 8,\ L_{pq}>0$' + '\n' + r'$\widehat c(p)\leq\widehat c(q)$', True)
box(7.1, 1.5, '4. Survivor selection', r'Capacity: $\mu+K=16$' + '\n' + r'Active: $\mu\leq 8$; archive: $K\leq 8$')
box(3.6, 1.5, '5. Variation', 'Active prompts become parents\nCrossover + mutation')
arrow((2.9, 4.05), (3.5, 4.05))
arrow((6.4, 4.05), (7, 4.05))
arrow((8.45, 3.4), (8.45, 2.7))
arrow((7, 2.05), (6.4, 2.05))
ax.plot([3.5, 1.45, 1.45], [2.05, 2.05, 3.25], color=GRAY, lw=1)
arrow((1.45, 3.25), (1.45, 3.4))
ax.text(.1, .55, 'B0+Archive: the same capacity and Racing policy, with the evidence gate disabled.', fontsize=8.5)
save(fig, 'method_overview')

# Both recorded offline budgets, with descriptive standard deviations.
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), layout='constrained')
for method, label, color, marker, linestyle in [
    ('b0_point_nsga', 'B0 (capacity 8)', GRAY, 'o', '--'),
    ('pace', 'PACE (capacity 16)', BLUE, 's', '-'),
]:
    for ax, metric, scale in [(axes[0], 'false_elim_rate', 100), (axes[1], 'n_retained', 1)]:
        values = [np.array([float(row[metric]) * scale for row in replay
                            if row['method'] == method and int(row['budget']) == budget])
                  for budget in [320, 640]]
        assert all(len(v) == 20 for v in values)
        ax.errorbar([320, 640], [v.mean() for v in values],
                    yerr=[v.std(ddof=1) for v in values], color=color,
                    marker=marker, linestyle=linestyle, capsize=4, lw=1.3, label=label)
for ax, title, ylabel in [(axes[0], '(a) False elimination', 'False elimination (%)'),
                           (axes[1], '(b) Retained-set size', 'Retained prompts')]:
    ax.set_title(title, loc='left')
    ax.set_xticks([320, 640])
    ax.set_xlabel('Search-observation budget')
    ax.set_ylabel(ylabel)
    ax.set_xlim(270, 690)
    ax.grid(axis='y', color=GRID, lw=.6)
    ax.set_axisbelow(True)
axes[0].set_ylim(-8, 48)
axes[1].set_ylim(5, 17)
axes[0].legend(fontsize=8, loc='upper right')
save(fig, 'replay_budgets')

# Mean search trajectories: two lines, with different markers for grayscale use.
fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8), layout='constrained')
for method, label, color, marker, style in [('b0_archive', 'B0+Archive', GRAY, 'o', '--'),
                                           ('pace', 'PACE', BLUE, 's', '-')]:
    histories = [[h for h in run['history'] if 'retained' in h] for run in runs[method]]
    assert all(len(hist) == 7 for hist in histories)
    for ax, metric in [(axes[0], 'calls_used'), (axes[1], 'retained')]:
        values = np.array([[len(h[metric]) if metric == 'retained' else h[metric]
                            for h in hist] for hist in histories])
        ax.plot(np.arange(7), values.mean(axis=0), color=color, marker=marker,
                linestyle=style, ms=3.5, lw=1.3, label=label)
for ax, title, ylabel in [(axes[0], '(a) Observation acquisition', 'Cumulative search observations'),
                           (axes[1], '(b) Retained prompts', 'Retained prompts')]:
    ax.set_title(title, loc='left')
    ax.set_xticks(np.arange(7), ['G0', 'O1', 'G1', 'O2', 'G2', 'O3', 'G3'], fontsize=8)
    ax.set_xlabel('Survivor checkpoint')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=GRID, lw=.6)
    ax.set_axisbelow(True)
axes[1].axhline(16, color='#B5BAC1', lw=.7, ls=':')
axes[1].set_ylim(6, 17)
axes[0].legend(fontsize=8, loc='upper left')
save(fig, 'search_dynamics')

# Sensitivity summaries use ordinary bars and retain every seed in the analysis.
fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.8), layout='constrained',
                          gridspec_kw={'width_ratios': [1, 1.7]})
means = [d[:3].mean(), d[3:].mean(), d.mean()]
axes[0].barh([2, 1, 0], means, color=BLUE, height=.5)
for level, value in zip([2, 1, 0], means):
    axes[0].text(value + .12, level, f'+{value:.2f}', va='center', fontsize=8)
axes[0].set_yticks([2, 1, 0], ['Exploratory\n(n = 3)', 'Extension\n(n = 12)', 'All seeds\n(n = 15)'], fontsize=8)
axes[0].set_xlim(0, 6)
axes[0].set_xlabel('Mean difference (pp)')
axes[0].set_title('(a) Cohort means', loc='left')
loo = (d.sum() - d) / 14
axes[1].barh(np.arange(15), loo, color=BLUE, height=.58)
axes[1].axvline(d.mean(), color=INK, lw=.8, ls='--')
axes[1].set_yticks(np.arange(15), [row['seed'] for row in rows], fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlim(0, 2.6)
axes[1].set_xlabel('Mean after omission (pp)')
axes[1].set_ylabel('Omitted seed', fontsize=9)
axes[1].set_title('(b) Leave-one-seed-out means', loc='left')
xgrid(axes[0])
xgrid(axes[1])
save(fig, 'seed_sensitivity')
print('Generated 7 vector PDFs and PNG previews; 30 raw online records verified.')
