"""Build deterministic PDF figures for the correlated test-time scaling note.

Every figure is generated from a closed form or a fixed-seed Monte Carlo
simulation. Regenerate them with

    uv run python scripts/make_figures.py

The simulation figures use seeds set inside their functions, so the results are
bit-for-bit reproducible. The remaining figures are exact.

Style: clean editorial / consulting exhibits. Every figure carries a single
horizontal key placed ABOVE the plot area; no labels overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.special import betaln
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "paper" / "data" / "empirical_results.json"
RHOW = ROOT / "paper" / "data" / "rhow_results.json"

# Corporate / editorial palette: navy-anchored, desaturated, one warm accent.
INK = "#16202B"        # text and strong baselines
NAVY = "#1F4E79"       # primary / lead series
STEEL = "#5B86A8"      # secondary blue
GREEN = "#3F7A56"      # desaturated green (gain / coverage / independent ideal)
CLAY = "#A6473B"       # muted brick red (loss / correlation tax / argued against)
MUTE = "#566573"       # secondary text and guide lines
SPINE = "#AEB6BF"      # axis line
GRID = "#E6E9EC"       # horizontal grid

# A sequential navy ramp for ordinal series (correlation level), light to deep.
NAVY_RAMP = ["#AFC3D6", "#7CA0C0", "#47749C", "#1F4E79"]

# Native size equals the on-page width so type renders at true 11pt with no
# rescaling when included at width=0.9\linewidth.
FIGSIZE = (6.0, 3.9)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "axes.labelpad": 7,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "butt",
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.0,
    }
)


def _style_axes(ax) -> None:
    """Floating axis: no top, right, or left spine, light horizontal grid."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.grid(False, axis="x")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=4, width=1.0, color=SPINE)


def _legend_above(ax, handles, labels, *, ncol=None, y=1.16):
    """Place a single horizontal key above the plot area, no frame."""
    if ncol is None:
        ncol = len(labels)
    leg = ax.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=False, handlelength=1.7, columnspacing=1.6,
        handletextpad=0.6, borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core analytic objects.
# ---------------------------------------------------------------------------
def n_eff(n, rho):
    """Effective number of samples, n / [1 + (n-1) rho]."""
    return n / (1.0 + (n - 1.0) * rho)


def beta_params(s, rho):
    """(a,b) of a Beta with mean s and intra-cluster correlation rho.

    Under the de Finetti representation theta ~ Beta(a,b), Y_i | theta ~ iid
    Bernoulli(theta): the marginal mean is s = a/(a+b) and the same-cluster
    correlation is rho = 1/(a+b+1).
    """
    c = (1.0 - rho) / rho
    return s * c, (1.0 - s) * c


def coverage_corr(n, s, rho):
    """pass@n = 1 - E_theta[(1-theta)^n] for Beta(a,b) heterogeneity.

    P(K=0) = B(a, b+n)/B(a, b), evaluated in log space for stability.
    """
    a, b = beta_params(s, rho)
    return 1.0 - np.exp(betaln(a, b + n) - betaln(a, b))


def line(color, **kw):
    return Line2D([0], [0], color=color, **kw)


# ---------------------------------------------------------------------------
# 1. Effective number of samples and the correlation ceiling.
# ---------------------------------------------------------------------------
def plot_effective_samples() -> None:
    n = np.linspace(1, 64, 400)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.11, right=0.91, top=0.83, bottom=0.14)
    _style_axes(ax)

    ax.plot(n, n, color=MUTE, linestyle=(0, (1, 2.5)), linewidth=1.5, zorder=1)

    rhos = [0.05, 0.10, 0.25, 0.50]
    for rho, color in zip(rhos, NAVY_RAMP):
        y = n_eff(n, rho)
        ax.plot(n, y, color=color, zorder=3)
        ceiling = 1.0 / rho
        ax.axhline(ceiling, color=color, linestyle=(0, (5, 4)), linewidth=1.0,
                   zorder=2)
        # tiny right-edge tag on each well-separated ceiling line
        ax.annotate(f"$1/\\rho={int(ceiling)}$", xy=(64, ceiling),
                    xytext=(3, 0), textcoords="offset points", color=color,
                    fontsize=9, ha="left", va="center", annotation_clip=False)

    handles = [line(MUTE, linestyle=(0, (1, 2.5)), linewidth=1.5)] + \
              [line(c) for c in NAVY_RAMP]
    labels = ["independent ideal $n_{\\mathrm{eff}}{=}n$"] + \
             [f"$\\rho={r:g}$" for r in rhos]
    _legend_above(ax, handles, labels, ncol=5)

    ax.set_xlabel("samples drawn  $n$")
    ax.set_ylabel("effective number of samples  $n_{\\mathrm{eff}}$")
    ax.set_xlim(1, 64)
    ax.set_ylim(0, 40)
    ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
    _save(fig, "effective_samples")


# ---------------------------------------------------------------------------
# Shared helper: majority-vote accuracy under exchangeable correlated attempts.
# ---------------------------------------------------------------------------
def _majority_correct(s, rho, ns, draws, seed):
    """Probability the majority vote is correct, under exchangeable
    Bernoulli(s) with intra-cluster correlation rho (Beta latent).

    Ties on even n are broken at random (half credit), the standard convention,
    so the curve is monotone rather than zig-zagging on parity.
    """
    rng = np.random.default_rng(seed)
    a, b = beta_params(s, rho)
    out = []
    for n in ns:
        theta = rng.beta(a, b, size=draws)
        k = rng.binomial(n, theta)
        win = np.mean(k > n / 2.0)
        tie = np.mean(k == n / 2.0) if n % 2 == 0 else 0.0
        out.append(win + 0.5 * tie)
    return np.array(out)


# ---------------------------------------------------------------------------
# Self-consistency: correlated voters plateau below certainty.
# ---------------------------------------------------------------------------
def plot_majority_vote() -> None:
    s = 0.65
    ns = np.array([1, 3, 5, 9, 17, 33, 65, 129, 257, 513])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.83, bottom=0.14)
    _style_axes(ax)

    acc_indep = _majority_correct(s, 1e-6, ns, draws=200_000, seed=1)
    ax.plot(ns, acc_indep, marker="o", color=GREEN, zorder=4)

    cmap = {0.10: STEEL, 0.30: NAVY}
    for rho, color in cmap.items():
        acc = _majority_correct(s, rho, ns, draws=200_000, seed=2)
        ax.plot(ns, acc, marker="o", color=color, zorder=3)
        plateau = norm.cdf((s - 0.5) / np.sqrt(rho * s * (1 - s)))
        ax.axhline(plateau, color=color, linestyle=(0, (5, 4)), linewidth=1.0,
                   zorder=2)
        ax.annotate(f"$\\Phi\\!\\left(\\frac{{s-1/2}}{{\\sqrt{{\\rho s(1-s)}}}}\\right)$"
                    if rho == 0.30 else "", xy=(513, plateau), xytext=(4, 0),
                    textcoords="offset points", color=color, fontsize=10,
                    ha="left", va="center", annotation_clip=False)

    handles = [line(GREEN, marker="o"), line(STEEL, marker="o"),
               line(NAVY, marker="o")]
    labels = ["independent (Condorcet) $\\to 1$", "$\\rho=0.1$", "$\\rho=0.3$"]
    _legend_above(ax, handles, labels, ncol=3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("votes  $n$  (per-vote correctness $s=0.65$)")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_xlim(1, 513)
    ax.set_ylim(0.6, 1.02)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${int(v):d}$"))
    _save(fig, "majority_vote")


# ---------------------------------------------------------------------------
# 4b. Anti-scaling of selection: coverage rises in both panels, but the
#     plurality vote rises to 1 when the mode is correct and falls to 0 when it
#     is wrong. The mechanism behind "more compute can make selection worse."
# ---------------------------------------------------------------------------
def _plurality_acc(p, ns, draws, seed):
    """Expected plurality-vote accuracy (answer 0 is correct) over draws Monte
    Carlo sessions, with random tie-breaking, vectorized across sessions."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    p = p / p.sum()
    k = len(p)
    out = []
    for n in ns:
        d = rng.choice(k, size=(draws, int(n)), p=p).astype(np.int8)
        counts = np.stack([(d == j).sum(1) for j in range(k)], axis=1)
        is_max = counts == counts.max(1, keepdims=True)
        out.append(float((is_max[:, 0] / is_max.sum(1)).mean()))
    return np.array(out)


def plot_anti_scaling() -> None:
    ns = np.array([1, 3, 5, 9, 17, 33, 65, 129, 257, 513])
    cases = [
        ("most likely answer correct", [0.45, 0.35, 0.20]),
        ("most likely answer wrong", [0.30, 0.50, 0.20]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.78, bottom=0.155,
                        wspace=0.10)

    for ax, (title_txt, p) in zip(axes, cases):
        _style_axes(ax)
        cov = 1.0 - (1.0 - p[0]) ** ns
        sel = _plurality_acc(p, ns, draws=60_000, seed=7)
        ax.plot(ns, cov, marker="o", color=GREEN, zorder=4)
        ax.plot(ns, sel, marker="o", color=NAVY, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 513)
        ax.set_ylim(-0.03, 1.05)
        ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${int(v):d}$"))
        ax.set_title(title_txt, fontsize=10, color=INK, pad=5)
        ax.set_xlabel("attempts  $n$")

    axes[0].set_ylabel("success rate")
    axes[1].annotate("coverage $\\to 1$", xy=(513, 1.0), xytext=(-2, -3),
                     textcoords="offset points", color=GREEN, fontsize=9,
                     ha="right", va="top", annotation_clip=False)
    axes[1].annotate("selection $\\to 0$", xy=(513, 0.0), xytext=(-2, 12),
                     textcoords="offset points", color=NAVY, fontsize=9,
                     ha="right", va="bottom", annotation_clip=False)

    handles = [line(GREEN, marker="o"), line(NAVY, marker="o")]
    labels = ["coverage  pass@$n$  (any attempt correct)",
              "self-consistency  (plurality answer)"]
    leg = fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                     bbox_to_anchor=(0.5, 1.0), handlelength=1.7,
                     columnspacing=1.6, handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK)
    _save(fig, "anti_scaling")


# ---------------------------------------------------------------------------
# 5. The functional form of coverage: heterogeneity gives a power law.
# ---------------------------------------------------------------------------
def plot_coverage_powerlaw() -> None:
    n = np.unique(np.round(np.logspace(0, 4, 60)).astype(int)).astype(float)
    s = 0.5
    ymin = 1e-6
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.13, right=0.90, top=0.83, bottom=0.14)
    _style_axes(ax)

    miss_indep = (1.0 - s) ** n
    ax.plot(n, miss_indep, color=GREEN, zorder=3)

    series = [(0.10, STEEL), (0.30, NAVY)]
    slopes = {}
    for rho, color in series:
        a, _ = beta_params(s, rho)
        slopes[rho] = a
        miss = 1.0 - coverage_corr(n, s, rho)
        ax.plot(n, miss, color=color, zorder=4)

    handles = [line(GREEN)] + [line(c) for _, c in series]
    labels = ["independent  $(1-s)^n$"] + \
             [f"$\\rho={r:g}$  (slope $-{slopes[r]:.2g}$)" for r, _ in series]
    _legend_above(ax, handles, labels, ncol=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("samples  $n$  (per-attempt correctness $s=0.5$)")
    ax.set_ylabel("miss rate  $1-$pass@$n$")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(ymin, 1.4)
    _save(fig, "coverage_powerlaw")


# ---------------------------------------------------------------------------
# 6. The marginal sample and the break-even point near n = 1/rho.
# ---------------------------------------------------------------------------
def plot_break_even() -> None:
    n = np.arange(1, 65)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.83, bottom=0.14)
    _style_axes(ax)

    blues = ["#7CA0C0", "#47749C", "#1F4E79"]
    rhos = [0.05, 0.10, 0.25]
    for i, (rho, color) in enumerate(zip(rhos, blues)):
        marg = (1.0 - rho) / (1.0 + (n - 1.0) * rho) ** 2
        ax.plot(n, marg, color=color, zorder=3)
        knee = 1.0 / rho
        ax.axvline(knee, color=color, linestyle=(0, (2, 3)), linewidth=1.0,
                   zorder=2)
        # Knee tags sit just above the top of the plot, centered over each
        # vertical line, all three at the same height.
        ax.annotate(f"$1/\\rho={int(round(knee))}$", xy=(knee, 1.0),
                    xytext=(0, 5), textcoords="offset points", color=color,
                    fontsize=9, ha="center", va="bottom", annotation_clip=False)

    handles = [line(c) for c in blues]
    labels = [f"$\\rho={r:g}$" for r in rhos]
    _legend_above(ax, handles, labels, ncol=3, y=1.24)

    ax.set_xlabel("sample index  $n$")
    ax.set_ylabel("value of the $n$-th sample\n($\\Delta n_{\\mathrm{eff}}$ per draw)")
    ax.set_xlim(1, 64)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
    _save(fig, "break_even")


# ---------------------------------------------------------------------------
# 7. Empirical: the coverage-selection gap on real sampling logs.
# ---------------------------------------------------------------------------
def plot_empirical() -> None:
    if not DATA.exists():
        print("  (skipping empirical figure: run scripts/analyze_brown.py first)")
        return
    results = json.loads(DATA.read_text())
    r = next((x for x in results if x["config"] == "GSM8K_Llama-3-8B-Instruct"),
             results[0])
    cov_x, cov = np.array(r["cov_grid"], float), np.array(r["coverage"])
    sel_x, sel = np.array(r["sel_grid"], float), np.array(r["selection"])
    plateau = sel[-1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.11, right=0.90, top=0.83, bottom=0.14)
    _style_axes(ax)

    # Shade the gap over the range where both curves are measured.
    cov_on_sel = np.interp(sel_x, cov_x, cov)
    ax.fill_between(sel_x, sel, cov_on_sel, color=CLAY, alpha=0.10, zorder=1, lw=0)
    # Selection plateau carried to the end of the coverage axis.
    ax.plot([sel_x[-1], cov_x[-1]], [plateau, plateau], color=NAVY,
            linestyle=(0, (5, 4)), linewidth=1.0, zorder=2)

    ax.plot(cov_x, cov, marker="o", color=GREEN, zorder=4)
    ax.plot(sel_x, sel, marker="o", color=NAVY, zorder=4)

    ax.annotate(f"coverage $\\to {r['coverage_inf']:.2f}$", xy=(cov_x[-1], cov[-1]),
                xytext=(3, 6), textcoords="offset points", color=GREEN,
                fontsize=9, ha="right", va="bottom", annotation_clip=False)
    ax.annotate(f"self-consistency $\\to {plateau:.2f}$", xy=(cov_x[-1], plateau),
                xytext=(3, -3), textcoords="offset points", color=NAVY,
                fontsize=9, ha="right", va="top", annotation_clip=False)
    ax.annotate("identifiability gap", xy=(180, 0.5 * (plateau + 1.0)),
                xytext=(0, 0), textcoords="offset points", color=CLAY,
                fontsize=9.5, ha="center", va="center")

    handles = [line(GREEN, marker="o"), line(NAVY, marker="o")]
    labels = ["coverage  pass@$n$  (any sample correct)",
              "self-consistency  (majority vote)"]
    _legend_above(ax, handles, labels, ncol=2)

    ax.set_xscale("log")
    ax.set_xlabel("samples  $n$  (GSM8K, Llama-3-8B-Instruct; "
                  f"$\\hat\\rho={r['rho']:.2f}$)")
    ax.set_ylabel("success rate")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0.72, 1.012)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    _save(fig, "empirical_gap")


# ---------------------------------------------------------------------------
# 8. Within-session: the within-problem (rho_w) coverage-selection gap, measured
#    on one real sampling session per problem (dependent draws).
# ---------------------------------------------------------------------------
def plot_within_session() -> None:
    if not RHOW.exists():
        print("  (skipping within-session figure: run scripts/analyze_rhow.py first)")
        return
    r = json.loads(RHOW.read_text())
    w = r["within_session"]
    x = np.array(w["sel_grid"], float)
    cov = np.array(w["coverage"])
    sel = np.array(w["selection"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.11, right=0.90, top=0.83, bottom=0.14)
    _style_axes(ax)

    ax.fill_between(x, sel, cov, color=CLAY, alpha=0.10, zorder=1, lw=0)
    ax.plot(x, cov, marker="o", color=GREEN, zorder=4)
    ax.plot(x, sel, marker="o", color=NAVY, zorder=4)

    ax.annotate(f"coverage $\\to {cov[-1]:.2f}$", xy=(x[-1], cov[-1]),
                xytext=(3, 7), textcoords="offset points", color=GREEN,
                fontsize=9, ha="right", va="bottom", annotation_clip=False)
    ax.annotate(f"self-consistency $\\to {sel[-1]:.2f}$", xy=(x[-1], sel[-1]),
                xytext=(3, -4), textcoords="offset points", color=NAVY,
                fontsize=9, ha="right", va="top", annotation_clip=False)
    ax.annotate("within-problem\nidentifiability gap", xy=(40, 0.605),
                xytext=(0, 0), textcoords="offset points", color=CLAY,
                fontsize=9.5, ha="center", va="center")

    handles = [line(GREEN, marker="o"), line(NAVY, marker="o")]
    labels = ["coverage  pass@$n$  (any attempt correct)",
              "self-consistency  (plurality vote)"]
    _legend_above(ax, handles, labels, ncol=2)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("attempts in one session  $n$  "
                  "(MATH-500, Llama-3.2-1B-Instruct, $T{=}0.8$)")
    ax.set_ylabel("success rate")
    ax.set_xlim(1, x[-1])
    ax.set_ylim(0.15, 0.85)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${int(v):d}$"))
    _save(fig, "within_session_gap")


def plot_concept() -> None:
    """One hard problem's answer pool: the correct answer is present (coverage
    finds it) but is not the most common, so the plurality vote returns a
    confident wrong answer. The picture of 'generated but not selected'."""
    freqs = [0.34, 0.18, 0.15, 0.12, 0.09, 0.07, 0.05]
    grey = "#C9CED4"
    colors = [CLAY, grey, GREEN, grey, grey, grey, grey]  # 0 wrong mode, 2 correct
    y = np.arange(len(freqs))[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.83, bottom=0.18)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=4, width=1.0, color=SPINE)

    ax.barh(y, freqs, color=colors, height=0.62, zorder=3)
    ax.set_yticks([])
    ax.set_xlim(0, 0.70)
    ax.set_ylim(-0.7, len(freqs) - 0.3)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of the $n$ sampled answers", labelpad=6)

    ax.annotate("the vote returns this  (wrong)", xy=(0.345, y[0]),
                xytext=(0.365, y[0]), va="center", ha="left", color=CLAY,
                fontsize=9.5)
    ax.annotate("the correct answer is here, not the mode", xy=(0.155, y[2]),
                xytext=(0.175, y[2]), va="center", ha="left", color=GREEN,
                fontsize=9.5)
    ax.set_title("One hard problem: the answers the model samples",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    _save(fig, "concept_gap")


def plot_two_ceilings() -> None:
    """Summary exhibit. A nominal sample budget meets two different ceilings.
    Left: selection (self-consistency) saturates at the modal ceiling while
    coverage keeps climbing, and the wedge between them is the identifiability
    gap. Right: the effective number of samples saturates at the correlation
    ceiling 1/rho, so correlated draws are worth far fewer than their count.
    Schematic, drawn from the closed forms."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.7, 3.45))
    fig.subplots_adjust(left=0.088, right=0.965, top=0.70, bottom=0.145,
                        wspace=0.28)
    dash = (0, (4, 3))
    dot = (0, (1, 2.5))

    # ---- (a) selection vs coverage: the modal ceiling ---------------------
    n = np.logspace(0, np.log2(512), 240, base=2.0)
    s, pi_mode = 0.40, 0.62
    cov = coverage_corr(n, s=s, rho=0.06)
    sel = pi_mode - (pi_mode - s) / (1.0 + 0.5 * (n - 1.0))

    axL.fill_between(n, sel, cov, color=NAVY, alpha=0.10, zorder=1)
    axL.plot(n, cov, color=GREEN, zorder=4)
    axL.plot(n, sel, color=NAVY, zorder=4)
    axL.axhline(pi_mode, color=NAVY, linestyle=dash, linewidth=1.0, zorder=2)

    axL.set_xscale("log", base=2)
    _style_axes(axL)
    axL.minorticks_off()
    axL.set_xlim(1, 512)
    axL.set_ylim(0, 1.0)
    axL.set_xticks([1, 4, 16, 64, 256])
    axL.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    axL.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axL.set_xlabel("samples drawn  $n$")
    axL.set_ylabel("problems solved")
    axL.set_title("(a)  Selection hits the modal ceiling", fontsize=10.5,
                  color=INK, loc="left", y=1.20)

    # The only in-plot label sits in the wide, empty interior of the wedge,
    # well clear of both curves and the dashed ceiling.
    axL.annotate("identifiability gap", xy=(28.0, 0.82), color=MUTE,
                 fontsize=9.5, ha="center", va="center", style="italic")
    _legend_above(
        axL,
        [line(GREEN), line(NAVY), line(NAVY, linestyle=dash, linewidth=1.0)],
        ["coverage", "self-consistency", "modal ceiling"],
        ncol=3, y=1.02,
    )

    # ---- (b) effective samples: the correlation ceiling -------------------
    m = np.logspace(0, np.log2(512), 240, base=2.0)
    axR.plot(m, m, color=MUTE, linestyle=dot, linewidth=1.5, zorder=1)
    for rho, color, tag_y, tag_at in ((0.1, NAVY, 10.0, 15.0),
                                      (0.5, CLAY, 2.0, 3.05)):
        axR.plot(m, n_eff(m, rho), color=color, zorder=4)
        axR.axhline(tag_y, color=color, linestyle=dash, linewidth=1.0, zorder=2)
        # Ceiling value tagged in the clear band just above each plateau.
        axR.annotate(f"$1/\\rho={int(tag_y)}$", xy=(330.0, tag_at), color=color,
                     fontsize=9.5, ha="center", va="bottom")

    axR.set_xscale("log", base=2)
    axR.set_yscale("log", base=10)
    _style_axes(axR)
    axR.minorticks_off()
    axR.set_xlim(1, 512)
    axR.set_ylim(1, 600)
    axR.set_xticks([1, 4, 16, 64, 256])
    axR.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    axR.set_yticks([1, 10, 100])
    axR.set_xlabel("samples drawn  $n$")
    axR.set_ylabel("effective samples  $n_{\\mathrm{eff}}$")
    axR.set_title("(b)  Estimation hits the correlation ceiling",
                  fontsize=10.5, color=INK, loc="left", y=1.20)
    _legend_above(
        axR,
        [line(MUTE, linestyle=dot, linewidth=1.5), line(NAVY), line(CLAY)],
        ["independent", "$\\rho=0.1$", "$\\rho=0.5$"],
        ncol=3, y=1.02,
    )

    _save(fig, "two_ceilings")


def main() -> None:
    plot_two_ceilings()
    plot_concept()
    plot_effective_samples()
    plot_majority_vote()
    plot_anti_scaling()
    plot_coverage_powerlaw()
    plot_break_even()
    plot_empirical()
    plot_within_session()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
