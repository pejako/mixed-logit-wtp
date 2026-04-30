"""Willingness-to-pay (WTP) distributions and feature preference ranking.

For a random utility model U_n = beta_n' x + epsilon, the marginal
willingness to pay for attribute k is

    WTP_k = -beta_{n,k} / beta_{n,p}

— the rate at which an individual will trade money for one unit of attribute
k, holding utility constant. The leading minus sign is because beta_p is
typically negative for normal goods.

Why this is interesting under MXL
----------------------------------
In MNL, WTP collapses to a single number per attribute. In Mixed Logit,
both numerator and denominator are random variables, so WTP is itself a
*distribution across the population*. That distribution lets us answer
questions plain MNL cannot:

  - "What does the median customer pay for quality?"
  - "What share of customers value brand recognition above $50?"
  - "Which feature shows the most preference heterogeneity?"

The Cauchy trap
---------------
The ratio of two normals can have undefined mean and variance when the
denominator's distribution crosses zero (Cauchy-like behavior). This is
the textbook reason to either (a) use a log-normal price coefficient so
-beta_p > 0 always, or (b) reparameterize in WTP-space (Train & Weeks 2005).

This module flags the dangerous case, reports robust summary statistics
(median, IQR, trimmed mean) alongside the mean, and explicitly warns when
the price coefficient distribution allows sign reversals.

References
----------
Train, K.E., & Weeks, M. (2005). "Discrete Choice Models in Preference
    Space and Willingness-to-Pay Space." In *Applications of Simulation
    Methods in Environmental and Resource Economics* (pp. 1-16). Springer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mixedlogit.dgp import AttributeSpec
from mixedlogit.halton import standard_normal_draws
from mixedlogit.mxl import MXLResult, _build_param_layout, _draw_betas


@dataclass
class WTPDistribution:
    """Simulated WTP distribution for a single attribute.

    Attributes
    ----------
    attr_name : str
    samples : numpy.ndarray
        Simulated WTP values, one per Monte Carlo draw.
    price_can_be_zero : bool
        True if the price coefficient distribution allows values close to
        zero (which makes the ratio unstable). Triggers a robust-statistics
        warning in summary().
    """

    attr_name: str
    samples: np.ndarray
    price_can_be_zero: bool

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples, ddof=1))

    def trimmed_mean(self, proportion: float = 0.05) -> float:
        """Mean after dropping the top and bottom ``proportion`` of samples."""
        from scipy import stats

        return float(stats.trim_mean(self.samples, proportion))

    def quantile(self, q: float | list[float]) -> float | np.ndarray:
        return np.quantile(self.samples, q)

    def share_above(self, threshold: float) -> float:
        """Fraction of the population with WTP above ``threshold``."""
        return float((self.samples > threshold).mean())

    def share_between(self, low: float, high: float) -> float:
        return float(((self.samples >= low) & (self.samples <= high)).mean())

    def summary(self) -> dict:
        """Compact summary suitable for a row in a feature-ranking table."""
        q = self.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        out = {
            "attribute": self.attr_name,
            "mean": self.mean,
            "median": self.median,
            "trimmed_mean_5pct": self.trimmed_mean(0.05),
            "std": self.std,
            "p05": q[0],
            "p25": q[1],
            "p50": q[2],
            "p75": q[3],
            "p95": q[4],
            "iqr": q[3] - q[1],
        }
        if self.price_can_be_zero:
            out["warning"] = "price coef can be near zero; mean/std unreliable"
        return out


# --- Compute WTP samples -----------------------------------------------------


def _price_distribution_can_be_zero(
    result: MXLResult, price_attr: str
) -> bool:
    """Return True if the price coef distribution allows values near zero.

    Conditions that DON'T allow zero crossings:
      - Lognormal: always >0 by construction.
      - Fixed: a constant; if it's nonzero, no zero crossings.
      - Normal: only safe if mean is many SDs from zero.
    """
    for spec in result.attr_specs:
        if spec.name != price_attr:
            continue
        if spec.dist == "lognormal":
            return False
        if spec.dist == "fixed":
            # Look up the fixed value in the result
            idx = result.coef_names.index(spec.name)
            return abs(result.coefficients[idx]) < 1e-3
        if spec.dist == "normal":
            mean_idx = result.coef_names.index(f"{spec.name} [mean]")
            sd_idx = result.coef_names.index(f"{spec.name} [sd]")
            mean = result.coefficients[mean_idx]
            sd = result.coefficients[sd_idx]
            # "Many SDs from zero" = roughly |mean/sd| > 3 means <0.15% mass crosses
            if sd <= 0:
                return abs(mean) < 1e-3
            return abs(mean / sd) < 3.0
        return True
    raise ValueError(f"Price attribute '{price_attr}' not in attr_specs")


def compute_wtp_samples(
    result: MXLResult,
    *,
    price_attr: str = "price",
    n_draws: int = 10_000,
    halton_seed: int = 0,
) -> dict[str, WTPDistribution]:
    """Simulate WTP distributions for every non-price attribute.

    Each draw represents one synthetic individual: we sample a beta vector
    from the estimated population distribution and compute -beta_k / beta_p
    for every non-price attribute k. The marginal distribution of these
    ratios is the population WTP distribution.

    Parameters
    ----------
    result : MXLResult
    price_attr : str, default "price"
        Name of the price attribute (its sign determines WTP sign).
    n_draws : int, default 10_000
        Number of MC samples. WTP distributions are well-summarized by 10k
        draws; bump higher for tail quantile precision.
    halton_seed : int, default 0
        Seed for the Halton scrambling RNG (shared with the elasticity module
        for reproducibility).

    Returns
    -------
    dict[str, WTPDistribution]
        Keyed by attribute name. Excludes the price attribute itself.
    """
    attr_specs: list[AttributeSpec] = result.attr_specs
    attr_names = [s.name for s in attr_specs]
    n_attr = len(attr_names)

    if price_attr not in attr_names:
        raise ValueError(f"Price attribute '{price_attr}' not in {attr_names}")
    p_idx = attr_names.index(price_attr)

    # Draw betas for n_draws synthetic individuals --------------------------
    _, layout = _build_param_layout(attr_specs)
    n_random = sum(1 for _, kind, _ in layout if kind != "fixed")
    if n_random == 0:
        z_random = np.empty((1, n_draws, 0))
    else:
        z_random = standard_normal_draws(n_draws, n_random, seed=halton_seed)
        z_random = z_random[None, :, :]
    betas = _draw_betas(result.coefficients, layout, 1, n_attr, z_random)
    betas = betas[0]  # (n_draws, n_attr)

    beta_p_samples = betas[:, p_idx]
    price_can_be_zero = _price_distribution_can_be_zero(result, price_attr)

    if price_can_be_zero:
        warnings.warn(
            f"Price coefficient '{price_attr}' has a distribution that allows "
            "values near zero. WTP ratios may have undefined mean/variance "
            "(Cauchy-like). Use median, IQR, and trimmed_mean instead. "
            "Consider re-estimating with a lognormal price coefficient.",
            stacklevel=2,
        )

    # Compute WTP per attribute ---------------------------------------------
    out: dict[str, WTPDistribution] = {}
    for k, name in enumerate(attr_names):
        if k == p_idx:
            continue
        wtp = -betas[:, k] / beta_p_samples
        out[name] = WTPDistribution(
            attr_name=name,
            samples=wtp,
            price_can_be_zero=price_can_be_zero,
        )
    return out


# --- Feature preference ranking ---------------------------------------------


def feature_preference_ranking(
    wtps: dict[str, WTPDistribution],
    *,
    by: str = "median",
) -> pd.DataFrame:
    """Rank attributes by central WTP, with heterogeneity diagnostics.

    Default ranking is by median because it's well-defined even when the
    underlying ratio distribution has heavy tails. Pass ``by="mean"`` for
    arithmetic-mean ranking (use only when the price coef is bounded away
    from zero).

    Parameters
    ----------
    wtps : dict[str, WTPDistribution]
        Output of :func:`compute_wtp_samples`.
    by : {"median", "mean", "trimmed_mean_5pct"}
        Statistic to sort on (descending: highest WTP first).

    Returns
    -------
    pandas.DataFrame
        One row per attribute with summary stats. Sorted by ``by``.
    """
    if by not in ("median", "mean", "trimmed_mean_5pct"):
        raise ValueError(f"by must be 'median', 'mean', or 'trimmed_mean_5pct', got '{by}'")

    rows = [w.summary() for w in wtps.values()]
    df = pd.DataFrame(rows).set_index("attribute")
    return df.sort_values(by, ascending=False)
