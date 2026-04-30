"""Mixed Logit (MXL) model via Simulated Maximum Likelihood.

For decision-maker n in choice situation s with chosen alternative y_{ns}:

    P_n(observed | beta_n) = prod_s [exp(beta_n' x_{n,s,y_{ns}})
                                    / sum_j exp(beta_n' x_{n,s,j})]

where beta_n is drawn once per individual from a parametric distribution
f(beta | theta). The unconditional probability has no closed form; we
approximate it with R simulation draws from f(. | theta):

    P_hat_n(theta) = (1/R) sum_{r=1..R} P_n(observed | beta_n^{(r)})

The simulated log-likelihood is sum_n log P_hat_n(theta), maximized over
theta = (means, sds) of the random coefficients (plus any fixed coefficients).

Three implementation details that matter:

1. **Halton draws are held fixed.** They are generated once in fit_mxl() and
   reused at every likelihood evaluation; resampling per call would inject
   noise into the gradient and break BFGS.

2. **Product over situations is INSIDE the average over draws**, not outside.
   The integral is at the individual level, so each draw r gives one
   sequence-likelihood per individual.

3. **Numerically stable averaging.** We compute log P_hat_n via the
   log-sum-exp trick on the per-draw log sequence-likelihoods, otherwise the
   inner products underflow to zero.

Constraints
-----------
- Lognormal random coefficients are estimated in their underlying-normal
  parameterization: the user-facing estimates are the (mu, sigma) of the
  *log* coefficient, and the implied coefficient is exp(mu + sigma * z).
- Fixed coefficients are constants estimated alongside the random ones.

Standard errors come from the inverse of the numerical Hessian at the
optimum (BHHH-style sandwich estimators are a v0.2 enhancement).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from mixedlogit.dgp import AttributeSpec
from mixedlogit.halton import standard_normal_draws
from mixedlogit.mnl import _reshape_long_to_arrays


@dataclass
class MXLResult:
    """Estimation output for a Mixed Logit model.

    The ``coefficients`` and ``std_errors`` arrays are flat: for each random
    attribute, two entries (mean, sd); for each fixed attribute, one entry
    (the value). The corresponding labels live in ``coef_names``.
    """

    attr_specs: list[AttributeSpec]
    coef_names: list[str]
    coefficients: np.ndarray
    std_errors: np.ndarray
    loglik: float
    loglik_null: float
    converged: bool
    n_iterations: int
    n_individuals: int
    n_observations: int  # number of choice situations
    n_draws: int
    hessian_inv: np.ndarray

    @property
    def t_values(self) -> np.ndarray:
        return self.coefficients / self.std_errors

    @property
    def mcfadden_r2(self) -> float:
        return 1.0 - self.loglik / self.loglik_null

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.coefficients,
                "std_err": self.std_errors,
                "t_value": self.t_values,
            },
            index=self.coef_names,
        )

    def __repr__(self) -> str:
        lines = [
            f"MXLResult(n_indiv={self.n_individuals}, n_obs={self.n_observations}, "
            f"R={self.n_draws}, LL={self.loglik:.3f}, R2={self.mcfadden_r2:.4f}, "
            f"converged={self.converged})",
            self.summary().round(4).to_string(),
        ]
        return "\n".join(lines)


# --- Parameter packing/unpacking ---------------------------------------------


def _build_param_layout(specs: list[AttributeSpec]) -> tuple[list[str], list[tuple]]:
    """Build the flat parameter vector layout for the optimizer.

    For each AttributeSpec, decide which params it contributes:
      - "fixed":     1 param  (the coefficient)
      - "normal":    2 params  (mean, sd)
      - "lognormal": 2 params  (mu, sigma of the underlying normal)

    Returns
    -------
    coef_names : list[str]
        Human-readable names for each flat parameter, in order.
    layout : list[tuple]
        For each attribute, a tuple describing its slice in the flat vector
        and its kind. Schema:
            (attr_index_in_specs, kind, slice(start, stop))
        where kind is "fixed", "normal", or "lognormal".
    """
    coef_names: list[str] = []
    layout: list[tuple] = []
    cursor = 0
    for k, spec in enumerate(specs):
        if spec.dist == "fixed":
            coef_names.append(spec.name)
            layout.append((k, "fixed", slice(cursor, cursor + 1)))
            cursor += 1
        elif spec.dist == "normal":
            coef_names += [f"{spec.name} [mean]", f"{spec.name} [sd]"]
            layout.append((k, "normal", slice(cursor, cursor + 2)))
            cursor += 2
        elif spec.dist == "lognormal":
            coef_names += [f"{spec.name} [mu]", f"{spec.name} [sigma]"]
            layout.append((k, "lognormal", slice(cursor, cursor + 2)))
            cursor += 2
        else:
            raise ValueError(f"Unknown distribution: {spec.dist}")
    return coef_names, layout


def _draw_betas(
    theta: np.ndarray,
    layout: list[tuple],
    n_individuals: int,
    n_attr: int,
    z_random: np.ndarray,
) -> np.ndarray:
    """Construct the (n_individuals, n_draws, n_attr) array of beta draws.

    ``z_random`` has shape (n_individuals, n_draws, n_random_dims) and contains
    pre-computed standard-normal Halton draws. Random coefficients are
    rescaled with the relevant (mean, sd) or (mu, sigma) from ``theta``;
    fixed coefficients are broadcast across all (n, r) cells.
    """
    n_draws = z_random.shape[1]
    betas = np.empty((n_individuals, n_draws, n_attr))
    z_cursor = 0  # index into the random-only z dimension
    for k, kind, sl in layout:
        if kind == "fixed":
            betas[:, :, k] = theta[sl][0]
        elif kind == "normal":
            mean, sd = theta[sl]
            # Halton draws z_random[:, :, z_cursor] -> N(mean, sd) coefficient
            betas[:, :, k] = mean + sd * z_random[:, :, z_cursor]
            z_cursor += 1
        else:  # lognormal
            mu, sigma = theta[sl]
            betas[:, :, k] = np.exp(mu + sigma * z_random[:, :, z_cursor])
            z_cursor += 1
    return betas


# --- Likelihood --------------------------------------------------------------


def _mxl_neg_loglik(
    theta: np.ndarray,
    X_per_indiv: np.ndarray,
    y_per_indiv: np.ndarray,
    layout: list[tuple],
    z_random: np.ndarray,
) -> float:
    """Negative simulated log-likelihood for MXL.

    Parameters
    ----------
    theta : (n_params,) array
        Flat parameter vector (means, sds, fixed values).
    X_per_indiv : (n_indiv, n_situations_per_indiv, n_alts, n_attr) array
        Design tensor, reshaped from long format.
    y_per_indiv : (n_indiv, n_situations_per_indiv) integer array
        Chosen-alternative index for each (individual, situation).
    layout : list of (attr_idx, kind, slice)
        Output of _build_param_layout.
    z_random : (n_indiv, n_draws, n_random_dims) array
        Pre-computed standard-normal draws (held fixed across iterations).

    Returns
    -------
    nll : float
    """
    n_indiv, n_sit, n_alt, n_attr = X_per_indiv.shape
    n_draws = z_random.shape[1]

    # 1. Build betas: (n_indiv, n_draws, n_attr)
    betas = _draw_betas(theta, layout, n_indiv, n_attr, z_random)

    # 2. Utilities V[n, r, s, j] = sum_k X[n, s, j, k] * beta[n, r, k]
    # einsum over the k (attribute) axis only.
    V = np.einsum("nsjk,nrk->nrsj", X_per_indiv, betas)

    # 3. Log-prob of chosen alt at each (n, r, s):
    #    log P = V_chosen - logsumexp(V over j)
    V_max = V.max(axis=3, keepdims=True)
    log_denom = V_max.squeeze(-1) + np.log(np.exp(V - V_max).sum(axis=3))
    # V_chosen via fancy indexing along the j axis
    n_idx = np.arange(n_indiv)[:, None, None]   # (n,1,1)
    r_idx = np.arange(n_draws)[None, :, None]   # (1,r,1)
    s_idx = np.arange(n_sit)[None, None, :]     # (1,1,s)
    V_chosen = V[n_idx, r_idx, s_idx, y_per_indiv[:, None, :]]  # (n_indiv, n_draws, n_sit)
    log_P_per_situation = V_chosen - log_denom

    # 4. Log-product over situations within each (n, r):
    log_P_seq = log_P_per_situation.sum(axis=2)  # (n_indiv, n_draws)

    # log P_hat_n = log( (1/R) sum_r exp(log_P_seq[n, r]) )
    # log-sum-exp over the draw axis, then subtract log R
    log_max = log_P_seq.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(log_P_seq - log_max).sum(axis=1, keepdims=True))
    log_P_hat = (log_max + log_sum).squeeze(-1) - np.log(n_draws)

    # 6. Sum over individuals
    return -float(log_P_hat.sum())


# --- Estimator --------------------------------------------------------------


def fit_mxl(
    df: pd.DataFrame,
    attr_specs: list[AttributeSpec],
    *,
    n_draws: int = 200,
    init: np.ndarray | None = None,
    tol: float = 1e-5,
    max_iter: int = 300,
    halton_seed: int = 0,
    compute_se: bool = True,
    verbose: bool = False,
) -> MXLResult:
    """Fit a Mixed Logit model on long-format choice data via SML.

    Parameters
    ----------
    df : pandas.DataFrame
        Long format: individual_id, situation_id, alt_id, chosen, plus one
        column per attribute named in ``attr_specs``.
    attr_specs : list of AttributeSpec
        Specifications for each attribute. Only the ``name`` and ``dist``
        fields are read; ``mean`` and ``sd`` provide starting values when
        ``init`` is None.
    n_draws : int, default 200
        Number of Halton draws per individual. 200 is a sensible default for
        models with up to ~5 random coefficients; bump to 500-1000 for more.
    init : (n_params,) array, optional
        Custom starting values. If None, defaults are derived from
        ``attr_specs`` (sds initialised to 0.3 if zero).
    tol : float, default 1e-5
        Gradient-norm tolerance for convergence.
    max_iter : int, default 300
        Maximum BFGS iterations.
    halton_seed : int, default 0
        Seed for Halton scrambling; held fixed during optimization.
    compute_se : bool, default True
        Whether to compute the numerical Hessian for standard errors. Set
        False to halve the runtime when only point estimates are needed
        (e.g., during cross-validation or scenario sweeps).
    verbose : bool, default False
        Print optimizer progress.

    Returns
    -------
    MXLResult
    """
    attr_names = [s.name for s in attr_specs]
    X_long, y_long = _reshape_long_to_arrays(df, attr_names)
    # X_long is (n_situations, n_alts, n_attr); we need to group by individual
    # and reshape to (n_indiv, n_sit_per_indiv, n_alts, n_attr).

    # The DGP guarantees balanced panels: every individual has the same number
    # of situations. We discover that from the data rather than assuming.
    df_sorted = df.sort_values(["individual_id", "situation_id", "alt_id"])
    sits_per_indiv = (
        df_sorted.drop_duplicates(["individual_id", "situation_id"])
        .groupby("individual_id")
        .size()
    )
    if sits_per_indiv.nunique() != 1:
        raise ValueError(
            "Unbalanced individuals: each individual must have the same number "
            "of choice situations."
        )
    n_sit_per_indiv = int(sits_per_indiv.iloc[0])
    n_indiv = sits_per_indiv.shape[0]
    n_alt = X_long.shape[1]
    n_attr = X_long.shape[2]

    # Sanity: total situations must factor cleanly
    if X_long.shape[0] != n_indiv * n_sit_per_indiv:
        raise ValueError("Reshape sanity check failed: situations don't divide evenly.")

    X_per_indiv = X_long.reshape(n_indiv, n_sit_per_indiv, n_alt, n_attr)
    y_per_indiv = y_long.reshape(n_indiv, n_sit_per_indiv)

    # Build flat parameter layout
    coef_names, layout = _build_param_layout(attr_specs)
    n_params = sum(sl.stop - sl.start for _, _, sl in layout)
    n_random = sum(1 for _, kind, _ in layout if kind != "fixed")

    # Pre-compute Halton draws — these are held fixed across iterations.
    # Shape: (n_indiv, n_draws, n_random)
    if n_random > 0:
        flat_z = standard_normal_draws(n_indiv * n_draws, n_random, seed=halton_seed)
        z_random = flat_z.reshape(n_indiv, n_draws, n_random)
    else:
        # Pure MNL — no random coefficients, but the API still works
        z_random = np.empty((n_indiv, n_draws, 0))

    # Starting values
    if init is None:
        theta0 = np.empty(n_params)
        for k, kind, sl in layout:
            spec = attr_specs[k]
            if kind == "fixed":
                theta0[sl] = spec.mean
            else:
                # spec.mean as starting mean; default sd of 0.3 if user gave 0
                theta0[sl] = (spec.mean, spec.sd if spec.sd > 0 else 0.3)
    else:
        theta0 = np.asarray(init, dtype=np.float64)
        if theta0.shape != (n_params,):
            raise ValueError(f"init must have shape ({n_params},), got {theta0.shape}")

    # Null log-likelihood (equal shares)
    n_total_situations = n_indiv * n_sit_per_indiv
    loglik_null = -n_total_situations * np.log(n_alt)

    # Optimize. Numerical gradient: with n_params typically <= 10 and Halton
    # draws fixed, finite-difference BFGS converges fine. An analytical
    # gradient is a v0.2 enhancement.
    callback = (lambda xk: print(f"  theta = {np.round(xk, 4)}")) if verbose else None
    res = optimize.minimize(
        _mxl_neg_loglik,
        theta0,
        args=(X_per_indiv, y_per_indiv, layout, z_random),
        method="BFGS",
        options={"maxiter": max_iter, "gtol": tol, "disp": verbose},
        callback=callback,
    )

    final_grad_norm = float(np.linalg.norm(res.jac))
    converged = final_grad_norm < max(1e-3, 100 * tol)

    # Standard errors from numerical Hessian (skipped if compute_se=False)
    if compute_se:
        cov = _numerical_hessian_inverse(
            res.x, X_per_indiv, y_per_indiv, layout, z_random
        )
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        cov = np.full((n_params, n_params), np.nan)
        se = np.full(n_params, np.nan)

    # Convention: report SDs of normal random coefficients as positive
    for _, kind, sl in layout:
        if kind in ("normal", "lognormal"):
            sd_idx = sl.start + 1
            res.x[sd_idx] = abs(res.x[sd_idx])

    return MXLResult(
        attr_specs=list(attr_specs),
        coef_names=coef_names,
        coefficients=res.x,
        std_errors=se,
        loglik=-res.fun,
        loglik_null=loglik_null,
        converged=converged,
        n_iterations=int(res.nit),
        n_individuals=n_indiv,
        n_observations=n_total_situations,
        n_draws=n_draws,
        hessian_inv=cov,
    )


def _numerical_hessian_inverse(
    theta: np.ndarray,
    X_per_indiv: np.ndarray,
    y_per_indiv: np.ndarray,
    layout: list[tuple],
    z_random: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """Central-difference numerical Hessian of the negative log-likelihood;
    return its inverse as the covariance estimate."""
    n = len(theta)
    H = np.zeros((n, n))
    f0 = _mxl_neg_loglik(theta, X_per_indiv, y_per_indiv, layout, z_random)

    # Diagonal: f(x+h) - 2f(x) + f(x-h) over h^2
    for i in range(n):
        tp = theta.copy()
        tp[i] += h
        tm = theta.copy()
        tm[i] -= h
        fp = _mxl_neg_loglik(tp, X_per_indiv, y_per_indiv, layout, z_random)
        fm = _mxl_neg_loglik(tm, X_per_indiv, y_per_indiv, layout, z_random)
        H[i, i] = (fp - 2 * f0 + fm) / (h * h)

    # Off-diagonal: (f(+,+) - f(+,-) - f(-,+) + f(-,-)) / (4 h^2)
    for i in range(n):
        for j in range(i + 1, n):
            tpp = theta.copy()
            tpp[i] += h
            tpp[j] += h
            tpm = theta.copy()
            tpm[i] += h
            tpm[j] -= h
            tmp = theta.copy()
            tmp[i] -= h
            tmp[j] += h
            tmm = theta.copy()
            tmm[i] -= h
            tmm[j] -= h
            fpp = _mxl_neg_loglik(tpp, X_per_indiv, y_per_indiv, layout, z_random)
            fpm = _mxl_neg_loglik(tpm, X_per_indiv, y_per_indiv, layout, z_random)
            fmp = _mxl_neg_loglik(tmp, X_per_indiv, y_per_indiv, layout, z_random)
            fmm = _mxl_neg_loglik(tmm, X_per_indiv, y_per_indiv, layout, z_random)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)

    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full_like(H, np.nan)
