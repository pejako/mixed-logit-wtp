"""Own- and cross-price elasticities under MNL and Mixed Logit.

Why this module exists
----------------------
Plain MNL imposes Independence of Irrelevant Alternatives (IIA): the
cross-price elasticity of share i w.r.t. price j depends only on j, not on i.
Every other alternative loses share proportionally when j's price rises,
which is unrealistic — buyers of premium products don't substitute toward
budget products at the same rate as other premium-product buyers do.

Mixed Logit breaks IIA *because* preferences vary across individuals: when
two products share a high-price-sensitivity segment, that segment will
shift between them in response to price changes, while less sensitive
segments barely react. The result is realistic substitution patterns that
matter directly for revenue management and competitive response modeling.

Conventions
-----------
- Elasticities are computed at the **mean attribute design** of the data,
  i.e. the average choice situation. This is the standard reference point
  for reporting; sample-level elasticities are also available via
  :func:`sample_aggregate_elasticities`.
- "Price" is identified by name. Pass ``price_attr="price"`` (default) or
  override.
- For MXL, elasticities use the **same Halton draws** as estimation so the
  results are internally consistent with the reported coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mixedlogit.dgp import AttributeSpec
from mixedlogit.halton import standard_normal_draws
from mixedlogit.mnl import MNLResult
from mixedlogit.mxl import MXLResult, _build_param_layout, _draw_betas


@dataclass
class ElasticityMatrix:
    """Aggregate elasticity matrix.

    The element [i, j] is the elasticity of share i with respect to price j:
    a 1% increase in alt j's price changes alt i's share by ``E[i, j]`` percent.

    The diagonal entries are own-price elasticities (typically negative for
    normal goods); off-diagonal entries are cross-price elasticities (typically
    positive, since substitutes gain share when a competitor's price rises).
    """

    matrix: np.ndarray  # shape (n_alts, n_alts)
    alt_labels: list[str]
    model: str  # "MNL" or "MXL"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.matrix, index=self.alt_labels, columns=self.alt_labels
        )

    def __repr__(self) -> str:
        df = self.to_dataframe().round(4)
        return f"ElasticityMatrix [{self.model}]\n{df.to_string()}"


# --- Helpers -----------------------------------------------------------------


def _representative_design(
    df: pd.DataFrame,
    attr_names: list[str],
    n_alts: int,
) -> np.ndarray:
    """Build a representative (n_alts, n_attr) design at the data's averages.

    For each alt_id, take the mean attribute value across all situations.
    The result is a single representative choice situation that captures
    the typical attribute mix for each alternative position.

    This is the standard reference design for reporting elasticities; if you
    want share-weighted aggregates over the actual sample, use
    :func:`sample_aggregate_elasticities` instead.
    """
    avg = (
        df.sort_values(["situation_id", "alt_id"])
        .groupby("alt_id")[attr_names]
        .mean()
        .to_numpy()
    )
    if avg.shape != (n_alts, len(attr_names)):
        raise ValueError(f"Expected ({n_alts}, {len(attr_names)}) design, got {avg.shape}")
    return avg


def _coef_index_for_price(
    coef_names: list[str], price_attr: str, model: str
) -> int | tuple[int, int]:
    """Find the index (or indices for MXL random) of the price coefficient.

    Returns
    -------
    int | tuple
        - For MNL: a single int.
        - For MXL fixed price: a single int.
        - For MXL random price: a (mean_idx, sd_idx) pair.
    """
    if model == "MNL":
        if price_attr not in coef_names:
            raise ValueError(f"'{price_attr}' not found in MNL coef_names: {coef_names}")
        return coef_names.index(price_attr)
    # MXL: look for either bare name (fixed) or "<name> [mean]" / "[sd]"
    if price_attr in coef_names:
        return coef_names.index(price_attr)
    mean_label = f"{price_attr} [mean]"
    sd_label = f"{price_attr} [sd]"
    mu_label = f"{price_attr} [mu]"
    sigma_label = f"{price_attr} [sigma]"
    if mean_label in coef_names and sd_label in coef_names:
        return coef_names.index(mean_label), coef_names.index(sd_label)
    if mu_label in coef_names and sigma_label in coef_names:
        return coef_names.index(mu_label), coef_names.index(sigma_label)
    raise ValueError(f"Could not locate price coefficient for '{price_attr}' in {coef_names}")


# --- MNL elasticities (closed form) -----------------------------------------


def mnl_aggregate_elasticities(
    result: MNLResult,
    df: pd.DataFrame,
    *,
    price_attr: str = "price",
    alt_labels: list[str] | None = None,
    design: np.ndarray | None = None,
) -> ElasticityMatrix:
    """Aggregate MNL elasticities at a representative design.

    Closed-form expressions (with beta_p typically < 0 for normal goods):
        eta_ii =  beta_p * p_i * (1 - P_i)            (own-price, negative)
        eta_ij = -beta_p * p_j * P_j      for i != j  (cross-price, positive)

    Note the IIA structure: cross-elasticity depends only on the column
    (j), so all off-diagonal entries in a given column are identical.
    That's the limitation we expose by comparing against MXL.

    Parameters
    ----------
    design : (n_alts, n_attr) array, optional
        Custom representative design. If None, uses the data's per-alt
        mean attribute values. Pass a differentiated design to expose
        IIA violation more clearly when comparing against MXL.
    """
    n_alts = df["alt_id"].nunique()
    attr_names = result.coef_names
    if design is None:
        X_rep = _representative_design(df, attr_names, n_alts)
    else:
        X_rep = np.asarray(design, dtype=np.float64)
        if X_rep.shape != (n_alts, len(attr_names)):
            raise ValueError(f"design must be ({n_alts}, {len(attr_names)})")

    p_idx = _coef_index_for_price(result.coef_names, price_attr, "MNL")
    beta = result.coefficients
    beta_p = beta[p_idx]
    prices = X_rep[:, p_idx]  # (n_alts,)

    # Compute representative shares
    V = X_rep @ beta  # (n_alts,)
    V -= V.max()
    expV = np.exp(V)
    P = expV / expV.sum()

    # Build elasticity matrix
    E = np.zeros((n_alts, n_alts))
    for i in range(n_alts):
        for j in range(n_alts):
            if i == j:
                E[i, j] = beta_p * prices[i] * (1.0 - P[i])
            else:
                E[i, j] = -beta_p * prices[j] * P[j]

    labels = alt_labels or [f"Alt {j}" for j in range(n_alts)]
    return ElasticityMatrix(matrix=E, alt_labels=labels, model="MNL")


# --- MXL elasticities (simulation-based) ------------------------------------


def mxl_aggregate_elasticities(
    result: MXLResult,
    df: pd.DataFrame,
    *,
    price_attr: str = "price",
    n_draws: int = 1000,
    halton_seed: int = 0,
    alt_labels: list[str] | None = None,
    design: np.ndarray | None = None,
) -> ElasticityMatrix:
    """Aggregate MXL elasticities at a representative design.

    Computed by simulation (with beta_p < 0 typically):

        eta_ii = (p_i / Pbar_i) * E[ beta_p * P_i * (1 - P_i) ]
        eta_ij = -(p_j / Pbar_i) * E[ beta_p * P_i * P_j ]   for i != j

    where the expectation is over the random-coefficient distribution and
    Pbar_i = E[P_i(beta)] is the aggregate share. Uses Halton draws for
    fast convergence; ``n_draws=1000`` is plenty given the smoothness of
    the integrand.

    Parameters
    ----------
    design : (n_alts, n_attr) array, optional
        Custom representative design. If None, uses the data's per-alt
        mean attribute values. Pass a differentiated design to expose
        IIA violation more clearly when comparing against MNL.
    """
    attr_specs: list[AttributeSpec] = result.attr_specs
    attr_names = [s.name for s in attr_specs]
    n_alts = df["alt_id"].nunique()
    n_attr = len(attr_names)

    if design is None:
        X_rep = _representative_design(df, attr_names, n_alts)
    else:
        X_rep = np.asarray(design, dtype=np.float64)
        if X_rep.shape != (n_alts, n_attr):
            raise ValueError(f"design must be ({n_alts}, {n_attr})")

    p_idx = attr_names.index(price_attr)
    prices = X_rep[:, p_idx]

    # Build betas at the estimated theta -------------------------------------
    _, layout = _build_param_layout(attr_specs)
    n_random = sum(1 for _, kind, _ in layout if kind != "fixed")
    if n_random == 0:
        z_random = np.empty((1, n_draws, 0))
    else:
        z_random = standard_normal_draws(n_draws, n_random, seed=halton_seed)
        z_random = z_random[None, :, :]
    betas = _draw_betas(result.coefficients, layout, 1, n_attr, z_random)
    betas = betas[0]  # (n_draws, n_attr)

    # Per-draw choice probabilities at the representative design
    V = X_rep @ betas.T  # (n_alts, n_draws)
    V = V.T  # (n_draws, n_alts)
    V -= V.max(axis=1, keepdims=True)
    expV = np.exp(V)
    P = expV / expV.sum(axis=1, keepdims=True)  # (n_draws, n_alts)
    Pbar = P.mean(axis=0)  # (n_alts,)

    beta_p_per_draw = betas[:, p_idx]  # (n_draws,)

    # Build elasticity matrix
    E = np.zeros((n_alts, n_alts))
    for i in range(n_alts):
        for j in range(n_alts):
            if i == j:
                inner = beta_p_per_draw * P[:, i] * (1.0 - P[:, i])
                E[i, j] = (prices[i] / Pbar[i]) * inner.mean()
            else:
                inner = beta_p_per_draw * P[:, i] * P[:, j]
                E[i, j] = -(prices[j] / Pbar[i]) * inner.mean()

    labels = alt_labels or [f"Alt {j}" for j in range(n_alts)]
    return ElasticityMatrix(matrix=E, alt_labels=labels, model="MXL")


# --- Higher-order helper: substitution-pattern comparison -------------------


def substitution_pattern_summary(
    mnl_elast: ElasticityMatrix,
    mxl_elast: ElasticityMatrix,
) -> pd.DataFrame:
    """Side-by-side: MNL cross-elasticity columns vs MXL cross-elasticity columns.

    Under IIA, every off-diagonal entry in a column of the MNL matrix is
    identical. The MXL matrix should show across-row variation in each
    column proportional to how much the rows share preference structure
    with the column. This summary surfaces that difference numerically.

    Returns
    -------
    pandas.DataFrame
        For each column j (price source), reports:
          - MNL cross-elasticity (single value, since rows are constant)
          - MXL min and max cross-elasticity across rows
          - MXL row-spread (max - min): a direct measure of IIA violation
    """
    if mnl_elast.alt_labels != mxl_elast.alt_labels:
        raise ValueError("MNL and MXL elasticity matrices must share alt_labels")

    n = len(mnl_elast.alt_labels)
    rows = []
    for j in range(n):
        col_mnl = mnl_elast.matrix[:, j]
        col_mxl = mxl_elast.matrix[:, j]
        # Off-diagonal entries (i != j)
        off_mask = np.arange(n) != j
        mnl_off = col_mnl[off_mask]
        mxl_off = col_mxl[off_mask]
        rows.append(
            {
                "alt_j": mnl_elast.alt_labels[j],
                "MNL cross-eta (constant)": mnl_off.mean(),
                "MXL cross-eta min": mxl_off.min(),
                "MXL cross-eta max": mxl_off.max(),
                "MXL row spread": mxl_off.max() - mxl_off.min(),
            }
        )
    return pd.DataFrame(rows).set_index("alt_j")
