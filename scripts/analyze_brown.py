"""Estimate the difficulty-correlation rho and the coverage-vs-selection gap on
the released sampling logs of Brown et al. (Large Language Monkeys, 2024).

For each configuration (benchmark x model) the public dataset gives, per
problem, 10,000 independently sampled solutions and a boolean correctness flag.
Because the attempts to a problem are sampled independently, the within-problem
correlation is zero by construction; what these logs reveal is the
between-problem difficulty distribution and the resulting benchmark-level gap
between coverage (any sample correct) and selection (self-consistency).

We compute, per configuration:
  - the per-problem success rates theta_i = c_i / n_i,
  - the difficulty mean s and intraclass correlation rho = Var(theta)/(s(1-s)),
  - coverage pass@k via the unbiased estimator of Chen et al. (2021),
  - self-consistency accuracy by majority (plurality) vote of extracted answers.

Run:  uv run python scripts/analyze_brown.py
Writes a small JSON summary to paper/data/empirical_results.json so the figures
build without re-downloading the multi-hundred-MB logs. The logs themselves are
cached under a local directory and never committed.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "empirical_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
CACHE = Path("/tmp/monkey_business")
CACHE.mkdir(exist_ok=True)

BASE = ("https://huggingface.co/datasets/ScalingIntelligence/"
        "monkey_business/resolve/main/")

# (config name, short label) -- ordered easy-to-hard for the table.
CONFIGS = [
    ("GSM8K_Llama-3-8B-Instruct", "GSM8K, Llama-3-8B-Instruct"),
    ("GSM8K_Llama-3-70B-Instruct", "GSM8K, Llama-3-70B-Instruct"),
    ("MATH_Llama-3-8B-Instruct", "MATH, Llama-3-8B-Instruct"),
    ("MATH_Llama-3-70B-Instruct", "MATH, Llama-3-70B-Instruct"),
]

COV_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000]
SEL_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEL_REPS = 400


def fetch(config: str) -> Path:
    path = CACHE / f"{config}.json"
    if not path.exists():
        print(f"  downloading {config} ...", flush=True)
        urllib.request.urlretrieve(BASE + f"{config}.json", path)
    return path


def final_answer(text: str):
    """Extract the final boxed numeric answer after the last '####' marker."""
    if "####" not in text:
        return None
    tail = text.split("####")[-1].strip().split("\n")[0]
    tail = tail.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", tail)
    return m.group(0).rstrip(".") if m else None


def coverage_at_k(correct_counts, n_samples, k):
    """Unbiased pass@k = mean_i [1 - C(n-c,k)/C(n,k)] (Chen et al. 2021)."""
    out = []
    for k_ in k:
        vals = []
        for c, n in zip(correct_counts, n_samples):
            if n - c < k_:
                vals.append(1.0)
            else:
                logp = (gammaln(n - c + 1) - gammaln(n - c - k_ + 1)
                        - gammaln(n + 1) + gammaln(n - k_ + 1))
                vals.append(1.0 - np.exp(logp))
        out.append(float(np.mean(vals)))
    return out


def selection_at_n(answers_per_problem, gt_per_problem, grid, reps, rng):
    """Self-consistency accuracy: plurality vote of extracted answers, averaged
    over random subsets and over problems. Answers are factorized to integer
    codes so the plurality is a fast bincount; ties go to the lowest code."""
    coded = []
    for ans, gt in zip(answers_per_problem, gt_per_problem):
        codes = {}
        arr = np.fromiter((codes.setdefault(a, len(codes)) for a in ans),
                          dtype=np.int64, count=len(ans))
        gt_code = codes.get(gt, -1)
        coded.append((arr, len(codes), gt_code))

    out = []
    for n in grid:
        correct = 0.0
        for arr, ncodes, gt_code in coded:
            hits = 0
            for _ in range(reps):
                idx = rng.integers(0, arr.shape[0], size=n)
                winner = np.argmax(np.bincount(arr[idx], minlength=ncodes))
                hits += int(winner == gt_code)
            correct += hits / reps
        out.append(correct / len(coded))
    return out


def rho_bootstrap_ci(theta, reps=10000, level=0.95, seed=0):
    """Clustered (problem-level) bootstrap CI for the between-problem intraclass
    correlation rho = Var(theta)/(s(1-s)). Each resample draws M problems with
    replacement -- the clustered bootstrap appropriate for cluster-sampled data
    -- and recomputes rho, so the interval reflects the finite number of
    problems and defuses the equicorrelation-fragility concern of the note."""
    theta = np.asarray(theta, dtype=float)
    M = theta.shape[0]
    rng = np.random.default_rng(seed)
    rhos = np.empty(reps)
    for b in range(reps):
        t = theta[rng.integers(0, M, size=M)]
        sb = t.mean()
        rhos[b] = t.var() / (sb * (1 - sb))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(rhos, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def analyze(config: str, label: str):
    data = json.load(open(fetch(config)))
    n_samples = [len(p["is_corrects"]) for p in data]
    correct = [int(np.sum(p["is_corrects"])) for p in data]
    theta = np.array([c / n for c, n in zip(correct, n_samples)])
    s = float(theta.mean())
    rho = float(theta.var() / (s * (1 - s)))
    rho_lo, rho_hi = rho_bootstrap_ci(theta)

    # Extracted answers for plurality voting; sentinel "" for unparsed samples.
    gt = [final_answer(p["gt_answer"]) for p in data]
    answers = [[final_answer(t) or "" for t in p["samples"]] for p in data]
    # Faithfulness of the extractor against the dataset's own labels.
    agree = tot = 0
    for ans, g, p in zip(answers, gt, data):
        for j in range(0, len(ans), 100):
            agree += int((ans[j] == g) == bool(p["is_corrects"][j]))
            tot += 1

    rng = np.random.default_rng(0)
    cov = coverage_at_k(correct, n_samples, COV_GRID)
    # Plurality voting needs reliable answer extraction; only trust it where the
    # extractor reproduces the dataset's own correctness labels. (MATH answers
    # are boxed LaTeX, which a numeric extractor cannot parse, so its
    # self-consistency curve is omitted rather than reported wrongly.)
    reliable = agree / tot >= 0.95
    sel = selection_at_n(answers, gt, SEL_GRID, SEL_REPS, rng) if reliable else None

    return {
        "config": config, "label": label,
        "n_problems": len(data), "samples_per_problem": int(np.median(n_samples)),
        "pass_at_1": s, "rho": rho, "ceiling": 1.0 / rho,
        "rho_ci": [rho_lo, rho_hi],
        "ceiling_ci": [1.0 / rho_hi, 1.0 / rho_lo],
        "plateau_majority_correct": float(np.mean(theta > 0.5)),
        "coverage_inf": cov[-1], "selection_inf": (sel[-1] if sel else None),
        "extractor_agreement": agree / tot,
        "cov_grid": COV_GRID, "coverage": cov,
        "sel_grid": SEL_GRID if reliable else None, "selection": sel,
        "theta": theta.round(4).tolist(),
    }


def main():
    results = []
    for config, label in CONFIGS:
        print(f"analyzing {config}", flush=True)
        try:
            r = analyze(config, label)
        except Exception as e:  # a config may be unavailable; skip cleanly
            print(f"  skipped ({e})")
            continue
        results.append(r)
        sel = f"{r['selection'][-1]:.3f}" if r["selection"] else "n/a"
        print(f"  pass@1={r['pass_at_1']:.3f}  rho={r['rho']:.3f}  "
              f"rho 95% CI=[{r['rho_ci'][0]:.3f},{r['rho_ci'][1]:.3f}]  "
              f"1/rho={r['ceiling']:.1f}  cov@inf={r['coverage_inf']:.3f}  "
              f"sel plateau={sel}  extractor={r['extractor_agreement']:.3f}")
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT} ({len(results)} configs)")


if __name__ == "__main__":
    main()
