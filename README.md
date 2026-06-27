# Correlated Test-Time Scaling

**When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling**

Yong Yi Bay and Kathleen A. Yearick

![Coverage climbs while self-consistency saturates at the modal ceiling; the effective number of samples saturates at the correlation ceiling 1/rho.](paper/figures/two_ceilings.png)

*The two ceilings. Left: selection (self-consistency) saturates at the modal ceiling while coverage keeps climbing, and the wedge between them is the identifiability gap. Right: the effective number of samples `n_eff = n/[1+(n-1)ρ]` saturates at the correlation ceiling `1/ρ`, so a problem sampled with intraclass correlation `ρ` is worth at most `1/ρ` independent draws, however large the budget.*

A short, self-contained note. Test-time scaling draws many samples from one
model and reports performance against the sample count `n`, accounting for the
samples as if they were independent. They are not: samples from one model at a
fixed temperature share a prompt, a decoding distribution, and recurring
reasoning templates, so they are positively correlated. We make this precise
with one borrowed instrument.

**Test-time sampling is cluster sampling.** A problem is a cluster and its `n`
attempts are correlated draws within it, so every same-problem quantity inherits
the survey-sampling *design effect* `d_eff = 1 + (n-1)ρ`, where `ρ` is the
intraclass correlation of the per-attempt success indicators. The usable count
is therefore not `n` but the **effective number of samples**

```
n_eff = n / [ 1 + (n-1) ρ ]   →   1/ρ   as n → ∞.
```

The limit `1/ρ` is a hard **correlation ceiling**: beyond about `1/ρ` samples,
extra draws are statistically redundant. But one nominal count buys different
amounts of three different things, and each meets a different ceiling:

- **Estimating** a benchmark mean is what the correlation ceiling binds: `n`
  correlated attempts carry the information of at most `1/ρ_b` independent ones
  (about two on released logs), so evaluation should buy more problems, not more
  samples.
- **Selecting** an answer (self-consistency, best-of-`n`) reads the *mode* of the
  answer distribution, not a sample mean, so it meets a separate ceiling: the
  **modal-hit rate** `π_mode`, the fraction of problems whose most common answer
  is correct. Past a few samples it plateaus, and where the mode is wrong it
  *anti-scales*, sharpening a confident error even as coverage rises.
- **Covering** (finding one correct sample for a verifier) has no within-problem
  ceiling at all and keeps paying.

So the widely reported gap, in which coverage scales over orders of magnitude
while majority voting and reward models plateau beyond a few hundred samples, is
two ceilings pulling apart, an identifiability limit rather than the design
effect. The difficulty-heterogeneity power law for coverage is the within-problem
`ρ_w = 0` case.

Both correlations are measured on released logs. The **between-problem** spread
`ρ_b ≈ 0.4–0.6` comes from the independent-draw logs of Brown et al. (*Large
Language Monkeys*), so ten thousand samples per problem carry the
benchmark-mean information of about two. The **within-problem** ceiling is read
off a dependent-draw log (the best-of-`n` release of Beeching, Tunstall, and
Rush): 500 MATH-500 problems sampled 256 times each by one model at a fixed
temperature, where one session's answers collapse onto a median of ~13 modes,
coverage reaches 0.88, and self-consistency plateaus at 0.45. The two-stage
identity `ρ = ρ_b + (1 − ρ_b)ρ_w` holds on real data to within 0.001
(`0.401` pooled versus `0.402` from the separate terms).

## Build

```
make verify    # numerically check every proposition against Monte Carlo (uv)
make figures   # regenerate the seven figures: five model-based + two empirical (uv + matplotlib, fixed seeds)
make data      # re-download and re-grade the public logs (uv sync --extra data); optional, cached JSON is committed
make all       # figures + compile paper/main.pdf (tectonic or latexmk)
make arxiv     # assemble arXiv source under build/arxiv-source
```

The Python environment is pinned in `pyproject.toml` and installed with
`uv sync` (add `--extra data` to regenerate the empirical summaries). Simulations
use fixed seeds, and the graded per-problem counts are cached as committed JSON,
so figures and checks are bit-for-bit reproducible without re-downloading the
multi-hundred-MB logs.

## Layout

```
paper/main.tex          the note
paper/references.bib     bibliography
paper/figures/           generated PDF figures
scripts/make_figures.py  figure generation
scripts/verify_math.py   numerical verification of the propositions
scripts/analyze_brown.py estimate between-problem rho_b (clustered-bootstrap CI) on Brown et al. logs
scripts/analyze_rhow.py  measure within-problem rho_w and the within-session gap on the Beeching et al. best-of-n log
```

## Reusing the result

Report the effective number of samples alongside the nominal count:

> Following Bay and Yearick, we report the effective number of samples
> `n_eff = n/[1+(n-1)ρ]` alongside the nominal sample count.

## Citation

```bibtex
% Add eprint / archivePrefix / doi once the arXiv preprint is posted.
@misc{bay2026ceilings,
  title         = {When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling},
  author        = {Bay, Yong Yi and Yearick, Kathleen A.},
  year          = {2026}
}
```

## License

The source code (`scripts/`, `Makefile`) is released under the MIT License
([LICENSE](LICENSE)). The paper text and figures (`paper/`) are licensed under
CC BY 4.0 ([paper/LICENSE](paper/LICENSE)).

---

## Part of a series

This repository is one of **Closed-Form Laws for Reasoning and Agentic Models**, a coordinated
series by Yong Yi Bay and Kathleen A. Yearick. Each note turns a freshly-observed phenomenon in
LLM/RL systems into a law with a named order parameter, using one classical mathematical instrument.

- `passk-crossover` — why pass@k curves cross exactly once (single-crossing law)
- `agent-horizon` — long-horizon agent reliability as a criticality law
- `correlated-test-time-scaling` — the effective number of samples and the correlation ceiling
- `grpo-explained` — GRPO as a variance-stabilized REINFORCE
- `outcome-concentration` — why large networks train and generalize reliably
- `reasoning-half-life` — a discovery–corruption law for overthinking
- `effective-group-size-rlvr` — a GRPO group is a cluster sample
- `reward-normalization-fairness` — reward normalization as inequality aversion
- `gspo-trust-region` — GSPO as GRPO with a mean trust region
- `verifier-density` — verifier density as the order parameter of agentic RLVR
- `latent-collapse-power-iteration` — latent collapse is power iteration
- `rope-condition-number` — RoPE extrapolation as ill-posed super-resolution
- `moe-routing-collapse` — routing collapse as a Pólya-urn monopoly

Foundational notes: *Solve for the Hyperparameter, Skip the Search* (arXiv:2606.23575) and
*Machine Learning vs Deep Learning: The Generalization Problem* (arXiv:2403.01621).
