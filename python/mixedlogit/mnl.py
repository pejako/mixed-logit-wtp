"""Multinomial Logit (MNL) model: baseline before Mixed Logit.

For decision-maker n in choice situation s with J alternatives indexed j and
chosen alternative y_{ns}:

    P(y_{ns} | beta) = exp(beta' x_{ns,y_{ns}}) / sum_j exp(beta' x_{ns,j})

The log-likelihood is convex in beta and admits a closed-form gradient and
Hessian, so estimation is fast and reliable.

This module exists for two reasons:

1. **Baseline for comparison.** Running MNL first lets us show how its
   substitution patterns (IIA) and lack of heterogeneity differ from MXL.

2. **Sanity check on the MXL machinery.** When all MXL coefficient SDs are
   constrained to zero, the MXL likelihood must reduce to the MNL likelihood
   exactly. Writing MNL standalone first gives us a reference implementation
   to test that against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class MNLResult:
    """Estimation output for an MNL model."""

    coef_names: list[str]
    coefficients: np.ndarray  # shape (n_attr,)
    std_errors: np.ndarray  # shape (n_attr,)
    loglik: float
    loglik_null: float  # log-likelihood at beta = 0 (equal shares)
    converged: bool
    n_iterations: int
    n_observations: int  # number of choice situations
    hessian_inv: np.ndarray  # shape (n_attr, n_attr)

    @property
    def t_values(self) -> np.ndarray:
        return self.coefficients / self.std_errors

    @property
    def mcfadden_r2(self) -> float:
        """McFadden's pseudo R-squared: 1 - LL(model) / LL(null)."""
        return 1.0 - self.loglik / self.loglik_null

    def summary(self) -> pd.DataFrame:
        """Coefficient table for printing or logging."""
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
            f"MNLResult(n_obs={self.n_observations}, "
            f"LL={self.loglik:.3f}, LL0={self.loglik_null:.3f}, "
            f"R2={self.mcfadden_r2:.4f}, converged={self.converged})",
            self.summary().round(4).to_string(),
        ]
        return "\n".join(lines)


def _reshape_long_to_arrays(
    df: pd.DataFrame, attr_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert long-format choice data to the (n_situations, n_alts, n_attr)
    design tensor and the (n_situations,) chosen-index vector consumed by the
    likelihood.

    Assumes balanced panels: every situation has the same number of alternatives.
    """
    # Sort to guarantee a consistent ordering
    df = df.sort_values(["situation_id", "alt_id"]).reset_index(drop=True)

    n_situations = df["situation_id"].nunique()
    n_alts_per_sit = df.groupby("situation_id").size()
    if n_alts_per_sit.nunique() != 1:
        raise ValueError(
            "Unbalanced panels: situations have different numbers of alternatives. "
            "All situations must have the same J."
        )
    n_alts = int(n_alts_per_sit.iloc[0])

    X = df[attr_names].to_numpy(dtype=np.float64)
    X = X.reshape(n_situations, n_alts, len(attr_names))

    chosen_long = df["chosen"].to_numpy()
    chosen_per_situation = chosen_long.reshape(n_situations, n_alts).argmax(axis=1)
    return X, chosen_per_situation


def _mnl_neg_loglik_and_grad(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and gradient for MNL.

    Parameters
    ----------
    beta : (n_attr,) array
    X : (n_situations, n_alts, n_attr) array
        Design tensor.
    y : (n_situations,) integer array
        Index of the chosen alternative in each situation.

    Returns
    -------
    nll : float
    grad : (n_attr,) array
    """
    # V[s, j] = sum_k X[s, j, k] * beta[k]
    V = X @ beta  # shape (n_situations, n_alts)

    # Numerically stable log-sum-exp: subtract row-wise max
    V_max = V.max(axis=1, keepdims=True)
    V_shifted = V - V_max
    exp_V = np.exp(V_shifted)
    denom = exp_V.sum(axis=1, keepdims=True)  # shape (n_situations, 1)
    log_denom = np.log(denom).squeeze(-1) + V_max.squeeze(-1)  # shape (n_situations,)

    # Log-prob of the chosen alt
    n_situations = X.shape[0]
    sit_idx = np.arange(n_situations)
    V_chosen = V[sit_idx, y]
    log_lik = (V_chosen - log_denom).sum()

    # Gradient: sum_s (x_{s, y_s} - sum_j P_{sj} x_{s,j})
    P = exp_V / denom  # shape (n_situations, n_alts)
    # Expected attributes per situation under the model:
    expected_X = np.einsum("sj,sjk->sk", P, X)  # shape (n_situations, n_attr)
    chosen_X = X[sit_idx, y, :]  # shape (n_situations, n_attr)
    grad_ll = (chosen_X - expected_X).sum(axis=0)  # shape (n_attr,)

    return -log_lik, -grad_ll


def _mnl_hessian(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Analytic Hessian of the MNL log-likelihood.

    H = -sum_s sum_j P_{sj} (x_{sj} - x_bar_s)(x_{sj} - x_bar_s)'
    where x_bar_s = sum_j P_{sj} x_{sj}.

    The negative of this is the observed information matrix; its inverse is
    the asymptotic variance-covariance of beta.
    """
    V = X @ beta
    V -= V.max(axis=1, keepdims=True)
    exp_V = np.exp(V)
    P = exp_V / exp_V.sum(axis=1, keepdims=True)  # (n_situations, n_alts)

    expected_X = np.einsum("sj,sjk->sk", P, X)  # (n_situations, n_attr)
    centered = X - expected_X[:, None, :]  # (n_situations, n_alts, n_attr)
    # Weighted outer product summed over (s, j)
    H = -np.einsum("sj,sjk,sjl->kl", P, centered, centered)
    return H


def fit_mnl(
    df: pd.DataFrame,
    attr_names: list[str],
    *,
    init: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> MNLResult:
    """Fit a multinomial logit model on long-format choice data.

    Parameters
    ----------
    df : pandas.DataFrame
        Long format with columns: situation_id, alt_id, chosen, plus the
        attribute columns named in ``attr_names``.
    attr_names : list of str
        Attribute columns to include as fixed-coefficient regressors.
    init : (n_attr,) array, optional
        Starting values for beta. Defaults to zeros.
    tol : float
        Convergence tolerance passed to scipy.optimize.minimize.
    max_iter : int
        Maximum BFGS iterations.

    Returns
    -------
    MNLResult
    """
    X, y = _reshape_long_to_arrays(df, attr_names)
    n_situations, n_alts, n_attr = X.shape

    beta0 = np.zeros(n_attr) if init is None else np.asarray(init, dtype=np.float64)
    if beta0.shape != (n_attr,):
        raise ValueError(f"init must have shape ({n_attr},), got {beta0.shape}")

    # Null log-likelihood: equal shares 1/J at every situation
    loglik_null = -n_situations * np.log(n_alts)

    res = optimize.minimize(
        _mnl_neg_loglik_and_grad,
        beta0,
        args=(X, y),
        jac=True,
        method="BFGS",
        options={"maxiter": max_iter, "gtol": tol},
    )

    # BFGS occasionally reports success=False even when the solution is
    # numerically correct (e.g., ``gtol`` is hit but the line-search step is
    # tiny). Use the gradient norm directly as the trustworthy convergence
    # signal; this matches what mlogit / xlogit do in practice.
    final_grad_norm = float(np.linalg.norm(res.jac))
    converged = final_grad_norm < max(1e-4, 100 * tol)

    H = _mnl_hessian(res.x, X)
    # Asymptotic covariance: -inverse of the Hessian of the log-likelihood
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = np.full((n_attr, n_attr), np.nan)
    se = np.sqrt(np.diag(cov))

    return MNLResult(
        coef_names=list(attr_names),
        coefficients=res.x,
        std_errors=se,
        loglik=-res.fun,
        loglik_null=loglik_null,
        converged=converged,
        n_iterations=int(res.nit),
        n_observations=n_situations,
        hessian_inv=cov,
    )
