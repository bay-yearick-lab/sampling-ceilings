"""Measure the within-problem correlation rho_w on a real dependent-draw log.

The selection ceiling 1/rho_w is governed by the WITHIN-problem correlation: the
run-to-run dispersion of a model's success rate on a fixed problem. The released
logs of Brown et al. (Large Language Monkeys) draw attempts independently, so
their rho_w is zero by construction and they can only expose the between-problem
term rho_b. To measure rho_w directly this script uses the best-of-N completion
logs released with HuggingFace's "Scaling test-time compute with open models"
(Beeching, Tunstall, and Rush, 2024): Llama-3.2-1B-Instruct sampled on the 500
MATH-500 problems at a fixed decoding configuration (temperature 0.8, top_p 1.0),
with FIVE independent sampling sessions (seeds 0--4), each drawing n=256 raw,
un-deduplicated completions per problem. Five sessions x 500 problems x 256 draws
is exactly the design the note's pre-registered protocol calls for: distinct
sessions of a fixed problem, every attempt verified.

Correctness per completion is graded with math-verify (the same library
HuggingFace used to produce the dataset's own labels), recovering the dataset's
reported single-sample accuracy of 27.2% to within about a point.

rho_w is the between-session, within-problem intraclass correlation. Two attempts
in the SAME session share the session latent rate theta; two attempts in
DIFFERENT sessions of the same problem do not. So the seed-to-seed spread of the
per-session success fraction, corrected for the binomial sampling noise of a
finite session, estimates the latent run-rate variance Var_session(theta), and

    rho_w = Var_session(theta) / [ s(1-s) ]

is the within-problem ICC, with a problem-clustered bootstrap CI.

Run:  uv run python scripts/analyze_rhow.py
Caches the graded (problem x seed) correct-count matrix under data/ so the figure
and table build without re-downloading or re-grading the multi-hundred-MB logs.
The raw parquet logs are cached under a local temp directory and never committed.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "rhow_results.json"
COUNTS = ROOT / "paper" / "data" / "rhow_counts.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
CACHE = Path("/tmp/h4_bon")
CACHE.mkdir(exist_ok=True)

DATASET = "HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions"
BASE = (f"https://huggingface.co/api/datasets/{DATASET}/parquet/"
        "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--max_tokens-2048"
        "--bsz-8--seed-{seed}--agg_strategy-last/train/0.parquet")
SEEDS = [0, 1, 2, 3, 4]
M = 256  # completions per problem per session


# ---------------------------------------------------------------------------
# Grading: extract the last \boxed{...} and verify against gold with math-verify.
# ---------------------------------------------------------------------------
def last_boxed(s: str):
    """Content of the final \\boxed{...} via brace matching, or None."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s) or s[i] != "{":
        return None
    depth, start = 0, i
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
    return None


_VERIFY = None  # lazily imported math-verify verify(), bound per worker

_SUBS = [(r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""),
         (r"\\ ", ""), (r"\\;", ""), (r"\\dfrac", r"\\frac"),
         (r"\\tfrac", r"\\frac"), (r"\\cdot", "*"), (r"\\times", "*")]


def _norm(a):
    """Cheap LaTeX-answer normalization for an exact-match fast path."""
    import re
    if a is None:
        return None
    a = a.strip().rstrip(" .")
    for pat, rep in _SUBS:
        a = re.sub(pat, rep, a)
    a = a.replace(" ", "").replace("$", "")
    a = a.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "")
    a = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", a)
    a = a.replace("{", "").replace("}", "").replace("\\\\", "\\").lower()
    return a


def _grade_boxed(boxed, gold, _cache={}, _gold_norm={}):
    """Cached boxed-vs-gold grade. A normalized-string equality fast path settles
    the large majority of attempts instantly; only residual non-matches fall
    through to math-verify (the library that produced the dataset's own labels),
    with a hard 2s cap so a pathological sympy comparison cannot stall the run."""
    global _VERIFY
    if boxed is None:
        return False
    key = (boxed, gold)
    if key in _cache:
        return _cache[key]
    gn = _gold_norm.get(gold)
    if gn is None:
        gn = _gold_norm[gold] = _norm(gold)
    bn = _norm(boxed)
    if bn is not None and bn == gn:
        _cache[key] = True
        return True
    if _VERIFY is None:
        from math_verify import parse, verify
        import warnings
        warnings.filterwarnings("ignore")
        _VERIFY = (parse, verify)
    parse, verify = _VERIFY
    try:
        g = parse("\\boxed{" + gold + "}", parsing_timeout=2)
        p = parse("\\boxed{" + boxed + "}", parsing_timeout=2)
        r = bool(verify(g, p, timeout_seconds=2))
    except Exception:
        r = False
    _cache[key] = r
    return r


def fetch(seed: int) -> Path:
    path = CACHE / f"seed{seed}.parquet"
    if not path.exists():
        url = BASE.format(seed=seed)
        print(f"  downloading seed {seed} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def _grade_seed(seed: int):
    """Grade one sampling session (seed); cache its per-problem counts so the
    run is resumable. Returns {uid: (n_correct, n_total)} and the per-problem
    first-completion correctness for validation against the dataset's labels."""
    seed_cache = ROOT / "paper" / "data" / f"rhow_seed{seed}.json"
    if seed_cache.exists():
        d = json.loads(seed_cache.read_text())
        return seed, d["counts"], d["naive_hits"], d["naive_tot"]
    import pyarrow.parquet as pq
    rows = pq.read_table(fetch(seed)).to_pylist()
    counts = {}
    naive_hits = naive_tot = 0
    for row in rows:
        gold = row["answer"]
        c = sum(_grade_boxed(last_boxed(x), gold) for x in row["completions"])
        counts[row["unique_id"]] = [c, len(row["completions"])]
        naive_hits += int(_grade_boxed(last_boxed(row["pred_naive@1"]), gold))
        naive_tot += 1
    seed_cache.write_text(json.dumps(
        {"counts": counts, "naive_hits": naive_hits, "naive_tot": naive_tot}))
    print(f"  seed {seed}: graded {len(counts)} problems", flush=True)
    return seed, counts, naive_hits, naive_tot


def grade_all():
    """Grade every completion in every seed (sessions graded in parallel);
    return aligned (P, R) count and sample-size matrices and the problem ids."""
    if COUNTS.exists():
        d = json.loads(COUNTS.read_text())
        return (np.array(d["counts"], float), np.array(d["ns"], float),
                d["uids"], d["single_sample_acc"])

    from multiprocessing import Pool
    with Pool(processes=len(SEEDS)) as pool:
        graded = pool.map(_grade_seed, SEEDS)

    per_seed = {}
    naive_hits = naive_tot = 0
    for seed, counts, nh, nt in graded:
        per_seed[seed] = counts
        naive_hits += nh
        naive_tot += nt

    uids = sorted(set.intersection(*[set(per_seed[r]) for r in SEEDS]))
    counts = np.array([[per_seed[r][u][0] for r in SEEDS] for u in uids], float)
    ns = np.array([[per_seed[r][u][1] for r in SEEDS] for u in uids], float)
    single = naive_hits / naive_tot
    COUNTS.write_text(json.dumps({
        "dataset": DATASET, "seeds": SEEDS, "m_per_session": M,
        "counts": counts.astype(int).tolist(), "ns": ns.astype(int).tolist(),
        "uids": uids, "single_sample_acc": single,
    }))
    # the consolidated cache is the committed artifact; drop the per-seed caches
    for r in SEEDS:
        (ROOT / "paper" / "data" / f"rhow_seed{r}.json").unlink(missing_ok=True)
    print(f"  wrote {COUNTS}  (single-sample acc {single:.3f})")
    return counts, ns, uids, single


# ---------------------------------------------------------------------------
# Estimators.
# ---------------------------------------------------------------------------
def rho_w_estimate(counts, ns):
    """Noise-corrected within-problem (between-session) ICC.

    For problem i and session r, theta_hat_{ir} = c_{ir}/m. The seed-to-seed
    variance of theta_hat within a problem equals the latent run-rate variance
    Var_session(theta) PLUS the binomial sampling variance theta(1-theta)/m of a
    finite session. Subtracting the (unbiased) binomial term problem by problem
    and averaging gives Var_session(theta); dividing by s(1-s) gives rho_w.
    """
    p = counts / ns                               # (P, R) session fractions
    R = p.shape[1]
    s = float(p.mean())
    # observed within-problem seed variance (unbiased, R-1)
    within_obs = p.var(axis=1, ddof=1)            # per problem
    # binomial sampling-noise term, unbiased estimate per session: phat(1-phat)/(m-1)
    noise = (p * (1.0 - p) / (ns - 1.0)).mean(axis=1)   # per problem
    var_session = np.maximum(within_obs - noise, 0.0).mean()
    rho_w = var_session / (s * (1.0 - s))
    return float(rho_w), float(var_session), s


def rho_b_estimate(counts, ns):
    """Between-problem ICC from the per-problem mean success rate (pooling the
    five sessions): rho_b = Var_i(theta_bar_i)/[s(1-s)], the difficulty spread."""
    theta_bar = counts.sum(axis=1) / ns.sum(axis=1)
    s = float(theta_bar.mean())
    return float(theta_bar.var() / (s * (1.0 - s))), theta_bar


def rho_pooled_estimate(counts, ns):
    """Pooled same-session ICC: two attempts drawn in one session of a randomly
    chosen problem. Equals rho_b + (1-rho_b) rho_w in expectation."""
    p = counts / ns                               # session fractions
    s = float(p.mean())
    # E[ theta(1-theta) ] estimated by within-session moment, then
    # Var(theta_session) = s(1-s) - E[theta(1-theta)]; pooled ICC = Var/[s(1-s)].
    # Use the design-effect identity on session-level fractions directly:
    # same-session covariance of two attempts = Var_total(theta_session).
    var_theta = p.var()                           # total variance of session rates
    # remove binomial noise of finite sessions
    noise = (p * (1.0 - p) / (ns - 1.0)).mean()
    var_theta = max(var_theta - noise, 0.0)
    return float(var_theta / (s * (1.0 - s)))


def bootstrap_ci(counts, ns, fn, reps=10000, level=0.95, seed=0):
    """Problem-clustered bootstrap CI: resample whole problems (all five of their
    sessions together) with replacement, recompute the statistic."""
    P = counts.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, P, size=P)
        vals[b] = fn(counts[idx], ns[idx])
    a = (1.0 - level) / 2.0
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Within-session answer-mode collapse: the within-problem (rho_w) selection
# ceiling, measured on real attempts that Brown's independent-draw logs cannot
# show. Within one sampling session the attempts' answers collapse onto a few
# modes, so self-consistency plateaus far below coverage.
# ---------------------------------------------------------------------------
SEL_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _norm_answer(a):
    """Reuse the fast LaTeX normalization to bucket answers for vote counting."""
    return _norm(a)


def within_session_curves(seed=0):
    """Per-problem answer distributions from ONE real sampling session; return
    the coverage and plurality (self-consistency) curves averaged over problems
    and random attempt subsets, the per-problem effective number of answers
    (1 / sum_a p_a^2), and the per-problem plurality-correct flag at full n.

    The answers are bucketed by the same normalization the grader's fast path
    uses; correctness of a bucket is decided by the math-verify grader so a
    bucket counts as the gold answer exactly when the grader says so."""
    import pyarrow.parquet as pq
    from collections import Counter
    rows = pq.read_table(fetch(seed)).to_pylist()
    rng = np.random.default_rng(0)

    cov = np.zeros(len(SEL_GRID))
    plur = np.zeros(len(SEL_GRID))
    eff_answers = []
    plateau_correct = []          # per problem: plurality at full n correct?
    s_list = []
    used = 0
    for row in rows:
        gold = row["answer"]
        raw = [last_boxed(c) for c in row["completions"]]
        # bucket by normalized string; unparsed answers are distinct singletons
        buckets = [(_norm_answer(b) if b is not None else f"__none{k}")
                   for k, b in enumerate(raw)]
        if len(buckets) < M:
            continue
        used += 1
        arr = np.array(buckets, dtype=object)
        # correctness of each attempt via the cached math-verify grader
        corr = np.array([_grade_boxed(b, gold) for b in raw])
        s_list.append(float(corr.mean()))
        cnt = Counter(buckets)
        ps = np.array([v / len(buckets) for v in cnt.values()])
        eff_answers.append(float(1.0 / np.square(ps).sum()))
        # which bucket is the gold answer (the grader's verdict on the bucket)
        gold_bucket = None
        for b, raw_b in zip(buckets, raw):
            if _grade_boxed(raw_b, gold):
                gold_bucket = b
                break
        for gi, n in enumerate(SEL_GRID):
            reps = 200
            cc = pp = 0
            for _ in range(reps):
                idx = rng.integers(0, len(buckets), size=n)
                cc += int(corr[idx].any())
                win = Counter(arr[idx].tolist()).most_common(1)[0][0]
                pp += int(win == gold_bucket)
            cov[gi] += cc / reps
            plur[gi] += pp / reps
        # full-n plurality correctness
        win_full = cnt.most_common(1)[0][0]
        plateau_correct.append(int(win_full == gold_bucket))
    cov /= used
    plur /= used
    return {
        "seed": seed, "n_problems": used,
        "sel_grid": SEL_GRID,
        "coverage": cov.round(4).tolist(),
        "selection": plur.round(4).tolist(),
        "eff_answers_median": float(np.median(eff_answers)),
        "eff_answers_mean": float(np.mean(eff_answers)),
        "plateau_selection": float(plur[-1]),
        "plateau_coverage": float(cov[-1]),
        "within_session_gap": float(cov[-1] - plur[-1]),
        "s_mean": float(np.mean(s_list)),
    }


def main():
    counts, ns, uids, single = grade_all()
    P = counts.shape[0]
    print(f"\nproblems aligned across {len(SEEDS)} sessions: {P}")
    print(f"single-sample accuracy (validation vs official 0.272): {single:.3f}")

    rho_w, var_sess, s = rho_w_estimate(counts, ns)
    rho_b, theta_bar = rho_b_estimate(counts, ns)
    rho_pool = rho_pooled_estimate(counts, ns)

    rho_w_ci = bootstrap_ci(counts, ns, lambda c, n: rho_w_estimate(c, n)[0])
    rho_b_ci = bootstrap_ci(counts, ns, lambda c, n: rho_b_estimate(c, n)[0])
    rho_pool_ci = bootstrap_ci(counts, ns, rho_pooled_estimate)

    # decomposition check: rho_pool ?= rho_b + (1-rho_b) rho_w
    decomp = rho_b + (1.0 - rho_b) * rho_w

    print(f"\ns (mean per-attempt success) = {s:.4f}")
    print(f"rho_w (within-problem)  = {rho_w:.4f}   95% CI [{rho_w_ci[0]:.4f},{rho_w_ci[1]:.4f}]")
    print(f"  1/rho_w (selection ceiling) = {1/rho_w:.1f}   "
          f"CI [{1/rho_w_ci[1]:.1f}, {1/rho_w_ci[0]:.1f}]")
    print(f"rho_b (between-problem) = {rho_b:.4f}   95% CI [{rho_b_ci[0]:.4f},{rho_b_ci[1]:.4f}]")
    print(f"rho_pool (same-session) = {rho_pool:.4f}   95% CI [{rho_pool_ci[0]:.4f},{rho_pool_ci[1]:.4f}]")
    print(f"decomposition rho_b+(1-rho_b)rho_w = {decomp:.4f}  vs pooled {rho_pool:.4f}")

    # Within-session answer-collapse: the within-problem selection ceiling on
    # real attempts (averaged over the five sessions for stability).
    ws_seeds = [within_session_curves(seed) for seed in SEEDS]
    cov = np.mean([w["coverage"] for w in ws_seeds], axis=0)
    sel = np.mean([w["selection"] for w in ws_seeds], axis=0)
    ws = {
        "sel_grid": SEL_GRID,
        "coverage": cov.round(4).tolist(),
        "selection": sel.round(4).tolist(),
        "plateau_coverage": float(cov[-1]),
        "plateau_selection": float(sel[-1]),
        "within_session_gap": float(cov[-1] - sel[-1]),
        "eff_answers_median": float(np.mean([w["eff_answers_median"] for w in ws_seeds])),
        "eff_answers_mean": float(np.mean([w["eff_answers_mean"] for w in ws_seeds])),
        "n_problems": ws_seeds[0]["n_problems"],
    }
    print(f"\nwithin-session (averaged over {len(SEEDS)} sessions):")
    print(f"  coverage@{M} = {ws['plateau_coverage']:.3f}   "
          f"self-consistency@{M} = {ws['plateau_selection']:.3f}   "
          f"gap = {ws['within_session_gap']:.3f}")
    print(f"  effective # answers (median 1/sum p^2) = {ws['eff_answers_median']:.1f} of {M}")

    results = {
        "dataset": DATASET,
        "source": "HuggingFace Scaling test-time compute (Beeching, Tunstall, Rush, 2024)",
        "model": "Llama-3.2-1B-Instruct", "benchmark": "MATH-500",
        "temperature": 0.8, "top_p": 1.0,
        "sessions": len(SEEDS), "m_per_session": M, "n_problems": P,
        "single_sample_acc": single,
        "s": s,
        "rho_w": rho_w, "rho_w_ci": list(rho_w_ci),
        "ceiling_w": 1.0 / rho_w,
        "ceiling_w_ci": [1.0 / rho_w_ci[1], 1.0 / rho_w_ci[0]],
        "var_session": var_sess,
        "rho_b": rho_b, "rho_b_ci": list(rho_b_ci),
        "rho_pooled": rho_pool, "rho_pooled_ci": list(rho_pool_ci),
        "decomp_check": decomp,
        "within_session": ws,
        "theta_bar": theta_bar.round(4).tolist(),
    }
    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
