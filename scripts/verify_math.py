"""Numerical verification of every analytic claim in the paper.

Run with:  uv run python scripts/verify_math.py
Exits 0 only if all checks pass.  Nothing here is plotted; this is the proof
harness that gates the paper's propositions.
"""
import numpy as np
from scipy.stats import norm, betabinom
from scipy.special import betaln

rng = np.random.default_rng(0)
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
# Exchangeable correlated Bernoulli via the latent (de Finetti) representation:
# theta ~ Beta(a,b);  Y_i | theta ~ iid Bernoulli(theta).
# Then  mean s = a/(a+b),  ICC rho = 1/(a+b+1).
# ---------------------------------------------------------------------------
def beta_params(s, rho):
    """Return (a,b) of a Beta with mean s and intra-cluster correlation rho."""
    c = (1.0 - rho) / rho          # a+b = (1-rho)/rho
    return s * c, (1.0 - s) * c


def simulate_counts(s, rho, n, draws):
    a, b = beta_params(s, rho)
    theta = rng.beta(a, b, size=draws)
    return rng.binomial(n, theta)   # K | theta ~ Binomial(n, theta)


# 1) ICC of the Beta-Binomial latent model -----------------------------------
for s, rho in [(0.3, 0.2), (0.5, 0.05), (0.1, 0.4)]:
    a, b = beta_params(s, rho)
    check(f"Beta mean s={s}", np.isclose(a / (a + b), s))
    check(f"Beta ICC rho={rho}", np.isclose(1.0 / (a + b + 1.0), rho))

# 2) Var(K) = n s(1-s)[1+(n-1)rho]  (design effect) --------------------------
for s, rho, n in [(0.3, 0.2, 8), (0.5, 0.1, 16), (0.2, 0.3, 32)]:
    K = simulate_counts(s, rho, n, draws=4_000_000)
    var_emp = K.var()
    var_thy = n * s * (1 - s) * (1 + (n - 1) * rho)
    check(f"Var(K) design effect  s={s},rho={rho},n={n}",
          np.isclose(var_emp, var_thy, rtol=2e-2),
          f"emp={var_emp:.3f} thy={var_thy:.3f}")

# 3) Var(mean) = s(1-s)/n_eff,  n_eff = n/(1+(n-1)rho) -----------------------
for s, rho, n in [(0.3, 0.2, 8), (0.4, 0.1, 50)]:
    K = simulate_counts(s, rho, n, draws=4_000_000)
    var_mean_emp = (K / n).var()
    n_eff = n / (1 + (n - 1) * rho)
    var_mean_thy = s * (1 - s) / n_eff
    check(f"Var(mean)=s(1-s)/n_eff  s={s},rho={rho},n={n}",
          np.isclose(var_mean_emp, var_mean_thy, rtol=2e-2),
          f"n_eff={n_eff:.3f} emp={var_mean_emp:.5f} thy={var_mean_thy:.5f}")

# 4) Correlation ceiling n_eff -> 1/rho --------------------------------------
for rho in [0.05, 0.2, 0.5]:
    n = 10_000_000
    n_eff = n / (1 + (n - 1) * rho)
    check(f"n_eff -> 1/rho  rho={rho}", np.isclose(n_eff, 1 / rho, rtol=1e-3),
          f"n_eff={n_eff:.4f} 1/rho={1/rho:.4f}")

# 5) Half-saturation knee at n = (1-rho)/rho  (n_eff = 1/(2 rho)) ------------
for rho in [0.05, 0.1, 0.25]:
    n_knee = (1 - rho) / rho
    n_eff = n_knee / (1 + (n_knee - 1) * rho)
    check(f"knee n_eff=1/(2rho) at n=(1-rho)/rho  rho={rho}",
          np.isclose(n_eff, 1 / (2 * rho)), f"n_knee={n_knee:.2f}")

# 6) Marginal value of n-th sample: dn_eff/dn = (1-rho)/(1+(n-1)rho)^2 -------
for rho, n in [(0.2, 5), (0.1, 20)]:
    f = lambda m: m / (1 + (m - 1) * rho)
    num = (f(n + 1e-6) - f(n - 1e-6)) / 2e-6
    ana = (1 - rho) / (1 + (n - 1) * rho) ** 2
    check(f"dn_eff/dn  rho={rho},n={n}", np.isclose(num, ana, rtol=1e-4),
          f"num={num:.5f} ana={ana:.5f}")

# 7) Coverage Jensen bound:  pass@n <= 1-(1-s)^n  (equality iff rho=0) -------
for s, rho, n in [(0.3, 0.2, 8), (0.2, 0.4, 16), (0.5, 0.1, 32)]:
    a, b = beta_params(s, rho)
    p0 = np.exp(betaln(a, b + n) - betaln(a, b))     # P(K=0) exact
    passn = 1 - p0
    indep = 1 - (1 - s) ** n
    check(f"pass@n <= independent  s={s},rho={rho},n={n}",
          passn <= indep + 1e-12, f"corr={passn:.4f} indep={indep:.4f}")

# independence limit rho->0 recovers the exponential exactly
s, n = 0.3, 12
rho = 1e-7
a, b = beta_params(s, rho)
p0 = np.exp(betaln(a, b + n) - betaln(a, b))
check("rho->0 recovers 1-(1-s)^n",
      np.isclose(1 - p0, 1 - (1 - s) ** n, atol=1e-3),
      f"corr={1-p0:.5f} indep={1-(1-s)**n:.5f}")

# 8) Power-law tail:  P(K=0) = B(a,b+n)/B(a,b) ~ C n^{-a}  -------------------
#    Fit slope of log P(K=0) vs log n at large n; should approach -a.
s, rho = 0.5, 0.25
a, b = beta_params(s, rho)
ns = np.array([2 ** k for k in range(8, 16)])      # 256 .. 32768
p0 = np.exp(betaln(a, b + ns) - betaln(a, b))
slope = np.polyfit(np.log(ns), np.log(p0), 1)[0]
check("power-law coverage tail exponent ~ -a",
      np.isclose(slope, -a, atol=0.03), f"slope={slope:.4f}  -a={-a:.4f}")

# also confirm exact Beta-Binomial P(K=0) matches scipy betabinom pmf at k=0
check("P(K=0) matches scipy betabinom",
      np.allclose(p0, betabinom.pmf(0, ns, a, b)),
      "")

# 8b) Coverage rises with n: pass@n non-decreasing, no within-problem ceiling.
s2, rho2 = 0.4, 0.2
a2, b2 = beta_params(s2, rho2)
ns2 = np.arange(1, 400)
passn = 1 - np.exp(betaln(a2, b2 + ns2) - betaln(a2, b2))
check("coverage pass@n non-decreasing in n (mixture)",
      np.all(np.diff(passn) >= -1e-12), f"min step={np.diff(passn).min():.2e}")
for pi in [0.05, 0.3]:
    nn = np.arange(1, 50)                       # range where float64 still resolves
    cov = 1 - (1 - pi) ** nn
    check(f"per-problem coverage strictly increasing  pi={pi}",
          bool(np.all(np.diff(cov) > 0)))
    check(f"per-problem coverage -> 1 as n grows  pi={pi}",
          bool(1 - (1 - pi) ** 5000 > 1 - 1e-9))

# 9) Hard ceiling from an atom at theta=0 (mixture: mass pi0 unreachable) ----
#    pass@inf = 1 - pi0.  Mix Beta(a,b) reachable mass with an atom at 0.
pi0 = 0.15
s_reach, rho_reach = 0.4, 0.2
a, b = beta_params(s_reach, rho_reach)
n = 5_000_000
p0_reach = np.exp(betaln(a, b + n) - betaln(a, b))   # ~0 at huge n
passinf = (1 - pi0) * (1 - p0_reach) + pi0 * 0.0
check("hard ceiling pass@inf = 1 - pi0",
      np.isclose(passinf, 1 - pi0, atol=1e-3), f"pass@inf={passinf:.4f}")

# 10) Correlated Condorcet plateau:
#     P(majority correct) -> Phi((s-1/2)/sqrt(rho s(1-s)))  for s>1/2 -------
def majority_acc(s, rho, n, draws=400_000):
    K = simulate_counts(s, rho, n, draws=draws)
    return np.mean(K > n / 2)


for s, rho in [(0.65, 0.1), (0.7, 0.25)]:
    acc_big = majority_acc(s, rho, n=2001)
    plateau = norm.cdf((s - 0.5) / np.sqrt(rho * s * (1 - s)))
    check(f"majority-vote plateau  s={s},rho={rho}",
          np.isclose(acc_big, plateau, atol=2e-2),
          f"acc(n=2001)={acc_big:.4f} plateau={plateau:.4f}")
    # independent voters (rho=0) would instead give ~1.0
    acc_indep = np.mean(rng.binomial(2001, s, size=200_000) > 2001 / 2)
    check(f"independent Condorcet -> ~1  s={s}", acc_indep > 0.999,
          f"acc_indep={acc_indep:.4f}")

# 11) Two-stage decomposition: pooled ICC rho = rho_b + (1-rho_b) rho_w. -----
#     Problem difficulty mu ~ Beta (between, rho_b); within a problem, a session
#     latent phi ~ Beta(mean mu, within-correlation rho_w); attempts ~ Bern(phi).
def two_level_icc(s, rho_b, rho_w, problems=4_000_000):
    cb = (1.0 - rho_b) / rho_b
    mu = rng.beta(s * cb, (1 - s) * cb, size=problems)
    cw = (1.0 - rho_w) / rho_w
    phi = rng.beta(mu * cw, (1 - mu) * cw)          # E[phi|mu]=mu, ICC_w=rho_w
    y1 = (rng.random(problems) < phi).astype(float)  # two same-problem attempts
    y2 = (rng.random(problems) < phi).astype(float)
    m = 0.5 * (y1.mean() + y2.mean())
    cov = np.mean((y1 - m) * (y2 - m))
    return cov / (m * (1 - m))


for s, rho_b, rho_w in [(0.5, 0.1, 0.3), (0.4, 0.2, 0.15)]:
    emp = two_level_icc(s, rho_b, rho_w)
    thy = rho_b + (1 - rho_b) * rho_w
    check(f"two-stage ICC  s={s},rho_b={rho_b},rho_w={rho_w}",
          np.isclose(emp, thy, atol=2e-3), f"emp={emp:.4f} thy={thy:.4f}")

# 12) Categorical selection ceiling: plurality -> mode; anti-scaling. --------
#     For a fixed problem with answer distribution p (index 0 = correct), the
#     plurality vote converges to the mode; selection accuracy -> 1{mode correct}.
def plurality_acc(p, correct, n, draws=8000):
    p = np.asarray(p, float)
    p = p / p.sum()
    d = rng.choice(len(p), size=(draws, n), p=p)
    hits = 0
    for row in d:
        c = np.bincount(row, minlength=len(p))
        cand = np.flatnonzero(c == c.max())
        hits += int(rng.choice(cand) == correct)
    return hits / draws


# (a) correct-mode problem: plurality accuracy -> 1
check("selection ceiling: correct is mode -> plurality ~1",
      plurality_acc([0.5, 0.3, 0.2], 0, n=401) > 0.99)

# (b) wrong-mode problem: plurality accuracy decays toward 0 (anti-scaling)
acc_lo = plurality_acc([0.3, 0.5, 0.2], 0, n=9)
acc_hi = plurality_acc([0.3, 0.5, 0.2], 0, n=401)
check("anti-scaling: wrong mode -> plurality -> 0",
      acc_hi < acc_lo and acc_hi < 0.02, f"n=9:{acc_lo:.3f} n=401:{acc_hi:.3f}")
# coverage on the same problem -> 1
cov = 1 - (1 - 0.3) ** 401
check("anti-scaling: coverage -> 1 on the same problem", cov > 0.999,
      f"cov={cov:.4f}")

# (c) plurality beats majority: theta = p_correct = 0.3 < 1/2, yet diffuse
#     errors leave the correct answer as the mode, so plurality -> 1.
check("plurality beats majority: theta=0.3<1/2 yet plurality ~1",
      plurality_acc([0.3] + [0.07] * 10, 0, n=401) > 0.99)

# (d) lemma behind P[theta>1/2] <= pi_mode: p_correct>1/2 forces correct=mode.
viol = 0
for _ in range(200_000):
    k = rng.integers(2, 8)
    w = rng.random(k)
    w = w / w.sum()
    if w[0] > 0.5 and np.argmax(w) != 0:
        viol += 1
check("lemma P[theta>1/2]<=pi_mode: p_correct>1/2 => correct is the mode",
      viol == 0, f"violations={viol}")

print(f"\n{sum(OK)}/{len(OK)} checks passed.")
raise SystemExit(0 if all(OK) else 1)
