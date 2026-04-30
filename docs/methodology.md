# Methodology

This document is the math-first companion to the four notebooks. It covers
the model specification, the simulated maximum likelihood (SML) estimator,
the elasticity derivation, and the willingness-to-pay (WTP) interpretation.
The intended reader is someone with a working knowledge of discrete choice
who wants to verify what this library does and why.

The synthetic-data approach is central. Every claim in this writeup is
*checked* in the test suite by recovering parameters we set ourselves —
not validated against an external benchmark, which is the usual setup, but
against ground truth. That is a stronger guarantee than "the model fits
the data."

---

## 1. Model specification

### 1.1 Random utility

We adopt the standard random-coefficient discrete-choice model. For
decision-maker $n$ in choice situation $s$, alternative $j$ has utility

$$
U_{nsj} = \beta_n' \, x_{nsj} + \varepsilon_{nsj},
$$

where $x_{nsj}$ is a $K$-vector of attributes (price, quality, brand, etc.),
$\beta_n$ is an individual-specific coefficient vector, and $\varepsilon_{nsj}$
is iid Type-I Extreme Value (Gumbel). The individual chooses the alternative
that maximizes their utility.

The Gumbel error gives the familiar logit choice probability *conditional*
on $\beta_n$:

$$
P_{nsj}(\beta_n) = \frac{\exp(\beta_n' x_{nsj})}{\sum_{k} \exp(\beta_n' x_{nsk})}.
$$

### 1.2 Heterogeneity

The crucial step in Mixed Logit is treating $\beta_n$ as a draw from a
parametric population distribution:

$$
\beta_n \sim f(\beta \mid \theta).
$$

In this library we support three forms per coefficient:

| Distribution | Parameters | Use when |
|---|---|---|
| **Fixed** | $\beta_k$ | Universal preference (homogeneous attribute) |
| **Normal** | $(\mu_k, \sigma_k)$ | Heterogeneous tastes that can flip sign |
| **Log-normal** | $(\mu_k, \sigma_k)$ of underlying normal | Sign-constrained (e.g., always negative price coefficient) |

For a normal coefficient, $\beta_{n,k} = \mu_k + \sigma_k \cdot z_{n,k}$
with $z_{n,k} \sim N(0,1)$. For a log-normal, $\beta_{n,k} = \exp(\mu_k + \sigma_k \cdot z_{n,k})$.
A fixed coefficient is the limiting case $\sigma_k = 0$.

### 1.3 The likelihood

The probability that individual $n$ produces their observed sequence of
choices $\{y_{ns}\}_s$ given $\beta_n$ is the product over their situations:

$$
L_n(\beta_n) = \prod_s P_{ns,\,y_{ns}}(\beta_n).
$$

Because $\beta_n$ is unobserved, the *unconditional* probability — the one
that enters the likelihood we actually want to maximize — integrates out
$\beta_n$ over its population distribution:

$$
L_n(\theta) = \int L_n(\beta) \, f(\beta \mid \theta) \, d\beta.
$$

The full sample log-likelihood is $\sum_n \log L_n(\theta)$.

The integral has no closed form. Section 2 explains how we approximate it.

### 1.4 The DGP module

The library includes a synthetic data generator that takes $\theta$ as
input and produces a long-format choice dataset. The point of synthetic
data is correctness: we *set* $\theta$, run the estimator, and check that
we recover it. The recovery test
(`test_mxl_recovers_default_dgp_across_seeds`) does exactly this on every
weekly CI run, asserting all parameters land within ~2.5 standard errors
of the truth across multiple seeds.

---

## 2. The simulated maximum likelihood estimator

### 2.1 Approximating the integral

We approximate $L_n(\theta)$ by Monte Carlo:

$$
\hat L_n(\theta) = \frac{1}{R} \sum_{r=1}^{R} L_n(\beta_n^{(r)}),
\qquad \beta_n^{(r)} \sim f(\,\cdot \mid \theta).
$$

Three implementation details matter, and each is the kind of mistake that
will silently produce nonsense if you get it wrong.

**The product over situations is *inside* the average over draws.** The
integral is at the *individual* level — each draw represents one
synthetic individual, who then makes their full sequence of choices.
Writing $\prod_s \frac{1}{R}\sum_r P_{r,s}$ is a different (and biased)
estimator. The correct expression is

$$
\hat L_n(\theta) = \frac{1}{R} \sum_{r=1}^{R} \prod_s P_{ns}(\beta_n^{(r)}).
$$

**The log is taken *after* averaging.** The simulated log-likelihood is
$\log \hat L_n$, not $\frac{1}{R} \sum_r \log L_n(\beta^{(r)})$. The
latter is biased downward by Jensen's inequality and converges to a
different quantity. To prevent underflow in the inner products, we
implement the average using log-sum-exp on the per-draw log
sequence-likelihoods.

**Halton draws are held fixed across iterations.** Re-drawing
$\beta^{(r)}$ at every likelihood evaluation injects sampling noise into
the gradient and breaks BFGS. We pre-compute the standard-normal draws
once per fit and reuse them. This is what `xlogit`, `mlogit`, `Apollo`,
and `gmnl` all do internally.

### 2.2 Halton sequences

Pseudo-random draws give the right answer eventually but require many of
them. **Halton sequences** are quasi-random low-discrepancy sequences
that fill the unit hypercube more uniformly. Empirically, a Halton-based
simulator with $R = 200$ matches the precision of a pseudo-random
simulator with $R = 2{,}000$ or more.

The library implements the **scrambled Halton variant** of Bhat (2003).
Plain Halton sequences develop spurious correlation in dimensions tied to
large primes; scrambling permutes the digits per dimension and breaks
that correlation. The smoke test in `halton.py` shows scrambled Halton
draws fill a 10×10 grid with bin-count variance ~1.0, vs ~2.3 for uniform
random — a 2.3× improvement in coverage uniformity.

We transform the unit-hypercube draws to standard normal via the inverse
normal CDF, clipping inputs to $(\epsilon, 1-\epsilon)$ to avoid $\pm\infty$.

### 2.3 Optimization

We maximize $\sum_n \log \hat L_n(\theta)$ using `scipy.optimize.minimize`
with BFGS. The numerical gradient (finite-difference) is sufficient for
the typical 5–10 parameters in a Mixed Logit model; an analytic gradient
is a v0.2 enhancement. Convergence is checked via the gradient norm
rather than scipy's `success` flag, because BFGS occasionally returns
`success=False` despite producing numerically excellent estimates (a
known interaction between line-search tolerance and `gtol`).

### 2.4 Standard errors

Asymptotic standard errors come from the inverse of the numerical Hessian
of the negative log-likelihood at the optimum. The Hessian is computed
with central finite differences ($O(p^2)$ likelihood evaluations).

Two caveats worth knowing:

- The SEs are asymptotic. With ~500 individuals and ~4,000 choice
  situations they are well-calibrated, as confirmed by the recovery test.
  For much smaller samples consider bootstrap.
- Numerical Hessian inversion can occasionally produce tiny negative
  diagonal entries (rounding error around the optimum). The library
  reports $\sqrt{\max(0, \text{var})}$ to keep SEs real, which means in
  the rare problematic case you'll see an SE of `0.000`. Treat that as
  "Hessian was non-PD here, this number is suspect" — not as "this
  estimate is exact."

### 2.5 Why MNL is biased on heterogeneous data

When the true population has $\sigma_k > 0$ for some attribute but you
fit a homogeneous MNL, the estimated coefficient is **biased toward zero**
— a phenomenon called *attenuation bias from omitted heterogeneity*.

The intuition: the mean of $P_{nsj}(\beta_n)$ over the population is not
the same function of the population-mean $\beta$ as $P_{nsj}$ evaluated
at the mean. Jensen's inequality applies to the choice probabilities,
which are non-linear in $\beta$, and the curvature pulls the
representative coefficient toward zero. Notebook 01 makes this concrete:
on the default DGP, MNL recovers `brand_known` (the homogeneous
attribute) within 0.2 SE of truth, but `price` and `quality` (the
heterogeneous ones) are off by 2.0 and 2.9 SE, both attenuated.

MXL recovers all five population parameters within 0.8 SE of the truth on
the same data, because it targets the population mean and SD directly
rather than a representative coefficient.

---

## 3. Elasticities and the IIA failure

### 3.1 MNL elasticities (closed form)

For an alternative $i$ at the representative design with price $p_i$ and
share $P_i$, the own- and cross-price elasticities under MNL are

$$
\eta_{ii}^{\text{MNL}} = \beta_p \, p_i \, (1 - P_i),
\qquad
\eta_{ij}^{\text{MNL}} = -\beta_p \, p_j \, P_j \quad \text{for } i \ne j.
$$

(For normal goods $\beta_p < 0$, so the diagonal is negative and the
off-diagonal is positive, as expected.)

### 3.2 The IIA structure

Look at the cross-elasticity: it depends only on $j$, not on $i$. **All
off-diagonal entries in a given column of the cross-elasticity matrix
are identical.** That is the signature of the Independence of Irrelevant
Alternatives (IIA) property.

In words: when alternative $j$ raises its price, every other alternative
gains share at the same proportional rate, regardless of how similar
each one is to $j$. That is wrong in essentially every real market — close
substitutes pick up more of the lost share than distant ones.

The library tests this algebraic fact directly: `test_mnl_satisfies_iia`
asserts that off-diagonal entries in each column match to numerical
precision (1e-10).

### 3.3 MXL elasticities (simulation)

Under MXL, the aggregate share is

$$
\bar P_i = \mathbb{E}_\beta\!\left[ P_i(\beta) \right],
$$

and the elasticity of *aggregate* share with respect to price $j$ is

$$
\eta_{ij}^{\text{MXL}} = \frac{p_j}{\bar P_i} \cdot \frac{\partial \bar P_i}{\partial p_j}.
$$

For the individual logit, $\partial P_{ni}/\partial p_j = \beta_{p,n} P_{ni}(\delta_{ij} - P_{nj})$,
so aggregating gives

$$
\eta_{ii}^{\text{MXL}} = \frac{p_i}{\bar P_i} \, \mathbb{E}_\beta\!\left[ \beta_p \, P_i (1 - P_i) \right],
$$

$$
\eta_{ij}^{\text{MXL}} = -\frac{p_j}{\bar P_i} \, \mathbb{E}_\beta\!\left[ \beta_p \, P_i \, P_j \right]
\quad \text{for } i \ne j.
$$

The expectations have no closed form; the library computes them by
simulation, reusing scrambled Halton draws for consistency with
estimation.

### 3.4 Why MXL breaks IIA

Look at the cross-elasticity expression. The expectation $\mathbb{E}_\beta[\beta_p P_i P_j]$
depends on the joint distribution of $(P_i, P_j)$ across draws — that is,
on which segments of the population tend to consider $i$ and $j$ together.
If $i$ and $j$ share a high-price-sensitivity segment, $\beta_p$ and
$P_i P_j$ are jointly large in that segment, inflating the cross-elasticity
relative to a less-correlated pair. The cross-elasticity now depends on $i$,
not just $j$. IIA is broken.

The test `test_mxl_violates_iia_on_differentiated_design` checks that on
the Premium / Mid / Budget design, MXL produces row spreads $> 0.05$ in
at least one column. The notebook 03 walkthrough shows the spreads are
typically 0.05–0.14, with the largest occurring for the Premium and Mid
columns where preference correlation is highest.

### 3.5 Practical consequence

The pricing-projection example in notebook 03 illustrates what this means
in dollars-and-share terms. A 5% Premium price increase, projected with
the two models on the Premium / Mid / Budget design:

| Alternative | MNL share change | MXL share change | MNL bias |
|---|---|---|---|
| Premium | −2.55 pp | −2.23 pp | overshoots loss |
| Mid | +1.54 pp | +1.56 pp | matches |
| Budget | +0.80 pp | +0.67 pp | overshoots gain by ~20% |

MNL systematically misallocates the spillover share. The error doesn't
show up in any goodness-of-fit metric — MNL's log-likelihood and pseudo-$R^2$
look fine — because the bias is in the *shape* of the substitution
pattern, not in fit on the observed choices.

---

## 4. Willingness to pay

### 4.1 Definition

The marginal willingness to pay for attribute $k$ is

$$
\text{WTP}_k = -\frac{\partial U / \partial x_k}{\partial U / \partial p}
\;=\; -\frac{\beta_k}{\beta_p}.
$$

Under MNL, this is a single number per attribute. Under MXL, both
numerator and denominator are random across the population, so WTP is
itself a *distribution*.

### 4.2 The Cauchy trap

The ratio of two normals has a known dirty secret: when the denominator's
distribution can cross zero, **the ratio's mean and variance are
undefined** (the distribution is Cauchy-like, with tails heavy enough
that sample moments do not converge).

The library detects this case by checking whether the price coefficient's
distribution puts non-trivial mass near zero, and warns the user when
$|\mu_p / \sigma_p| < 3$. The user is then directed to **median, IQR,
and trimmed mean** as robust summary statistics. Notebook 04 shows what
goes wrong: on the default DGP, the sample mean of WTP is plausible
(0.61 for quality) but the sample SD is 26.9 — that is not a meaningful
SD, it is the Cauchy tail talking.

### 4.3 Two ways to side-step the trap

There are two clean fixes for the Cauchy problem, both standard in
practice:

**Lognormal price coefficient.** If $-\beta_p \sim \text{Lognormal}$, then
$-\beta_p > 0$ always and the ratio is well-behaved. The library's
`AttributeSpec` supports `dist="lognormal"` for exactly this purpose.
The drawback is interpretive — the lognormal mean is not the same as the
mean of the underlying normal, and reporting and convergence both get
slightly trickier.

**WTP-space parameterization (Train & Weeks 2005).** Reparameterize the
utility as $U = \alpha (p - w' x)$, where $\alpha$ is a price scaling
parameter and $w_k$ is the WTP for attribute $k$. The WTPs are then
*primary* parameters of the model, not derived ratios, and have
well-defined estimates and standard errors regardless of the price
coefficient distribution. This is a v0.2 enhancement for this library.

### 4.4 Feature preference ranking

The library exposes WTP distributions per non-price attribute and a
ranking helper that sorts by population median (the robust default).
The summary table reports:

- Mean (with caveat when warning fires)
- Median (always reported)
- Trimmed mean at 5% (robust)
- Standard deviation (with caveat)
- Quantiles at 5/25/50/75/95
- Interquartile range

The IQR is the most useful summary of preference heterogeneity. On the
default DGP, the recovered IQRs are roughly $[0.36, 1.05]$ for quality
and $[0.42, 0.70]$ for brand. Read those numbers as: 50% of the
population has quality WTP somewhere in $[0.36, 1.05]$, but 50% of the
population has brand WTP in the much narrower $[0.42, 0.70]$. **Quality
is a feature where preferences segment the population; brand recognition
is broadly liked at a similar level.** That distinction is invisible in
MNL output and is the kind of thing that MXL is supposed to surface.

### 4.5 Population-level questions

Once you have the WTP distribution, you can answer questions that are
ill-posed under MNL:

- *What share of the population has quality WTP above some threshold $T$?*
  $\Pr(\text{WTP}_q > T)$ — direct from the empirical CDF.
- *What is the 90th-percentile WTP?* The empirical 90% quantile.
- *What share of customers fall in a given WTP band $[a, b]$?*
  Direct from the samples.

The `WTPDistribution` class exposes these as `share_above`, `quantile`,
and `share_between`. Notebook 04 ends with a panel of seven such
questions answered for the default DGP. None has a meaningful answer in
a plain MNL world — and that is the project's bottom line. **The value
of MXL over MNL is not better fit, and not even better point estimates
of the means; it is access to a class of questions that MNL fundamentally
cannot represent.**

## 5. Cross-language parity

The R companion package (`r/`) is not a re-implementation of the
estimator. It reads the canonical synthetic dataset from CSV — the same
data the Python tests use — and fits MNL and MXL with R's `mlogit` package
(Croissant, 2020), which is the standard reference implementation in the
field. Both languages produce results, both write to JSON, and a
parity comparison surfaces any cross-language disagreement.

The parity tests assert:

| Quantity | Tolerance | Why this much |
|---|---|---|
| MNL coefficients | within 5% relative | MNL is deterministic; gap should be <1% in practice (different optimizers, same optimum) |
| MXL means | within 5% relative | SML simulation noise from independently-seeded Halton draws |
| MXL standard deviations | within 10% relative | SDs are estimated from per-draw spread, so they're more sensitive to draw quality than means |

**Why this matters.** Two independent implementations agreeing on the
same data is a stronger correctness signal than either one in
isolation. Bugs typically don't survive cross-language testing — they
either show up as systematic gaps that flag the bug, or they happen to
exist in only one implementation, which is also informative. This
project ships the full pipeline: shared synthetic data, two estimators,
side-by-side comparison, and CI that runs the comparison on every
weekly schedule.

It is also a credibility lever for portfolio review. "I built an MXL
estimator" is a routine claim; "I built an MXL estimator and verified
its output matches `mlogit` to within 5% on the same synthetic data"
is harder to dispute.

---

## 6. References

Bhat, C.R. (2003). "Simulation Estimation of Mixed Discrete Choice Models
Using Randomized and Scrambled Halton Sequences." *Transportation Research
Part B*, 37(9), 837–855.

Croissant, Y. (2020). "Estimation of Random Utility Models in R: The
mlogit Package." *Journal of Statistical Software*, 95(11), 1–41.

Hensher, D.A. & Greene, W.H. (2003). "The Mixed Logit Model: The State
of Practice." *Transportation*, 30, 133–176.

Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
Cambridge University Press.

Train, K.E. & Weeks, M. (2005). "Discrete Choice Models in Preference
Space and Willingness-to-Pay Space." In *Applications of Simulation
Methods in Environmental and Resource Economics* (pp. 1–16). Springer.

---

## Appendix: design choices worth defending

A few things in the library that deviate from textbook treatments. Each
is intentional:

**Variable names follow math notation.** `X`, `V`, `P`, `H`, `LL` and so
on are standard in econometric papers. The ruff lint config disables the
lowercase-variable rule (`N806`) inside the package and tests so the code
reads like the math. Renaming `X` to `design_matrix` would not improve
clarity; it would obscure the connection to the equations above.

**Fast tests vs slow tests are separate pytest markers.** The fast suite
runs in ~30s and gates every push. The slow recovery suite runs weekly
on schedule (or on-demand via `workflow_dispatch`). This keeps PR
feedback under a minute while still validating the most important
property of the library — that the estimator recovers the truth — on a
regular cadence.

**Notebooks are built from `_build_*.py` scripts.** The `.ipynb` files
are byproducts. This means edits go to the script, the notebook
regenerates and re-executes, and there is no Jupyter-JSON merge conflict
to ever worry about. Every notebook is also guaranteed to actually run
end-to-end because that is checked at build time.

**Standard errors come from the numerical Hessian, not BHHH or
sandwich.** The numerical Hessian is more expensive but more honest about
local curvature in finite samples. For the model sizes targeted by this
library (5–15 parameters), the cost is irrelevant. BHHH and
robust-sandwich variants are reasonable v0.2 additions for users who
know they want them.
