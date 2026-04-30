"""Tests for the WTP module.

Cover four kinds of properties:

1. **Median recovery on safe configs** — when the price coefficient is
   bounded away from zero, the median WTP should match -beta_k_true / beta_p_true.
2. **Cauchy warning** — fires when the price coef distribution allows zero
   crossings; silent when it doesn't.
3. **API correctness** — feature ranking sorts as expected, validation works.
4. **Robust statistics behaviour** — when the warning fires, summary stats
   are still well-defined.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mixedlogit.dgp import (
    AttributeSpec,
    DGPConfig,
    default_config,
    simulate_choices,
)
from mixedlogit.mxl import fit_mxl
from mixedlogit.wtp import (
    WTPDistribution,
    _price_distribution_can_be_zero,
    compute_wtp_samples,
    feature_preference_ranking,
)

# Fixtures --------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_mxl_default():
    """Default config: price ~ N(-1.2, 0.4), |mean/sd|=3.0, AT the threshold."""
    cfg = default_config()
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=100, halton_seed=0, compute_se=False)
    return res


@pytest.fixture(scope="module")
def fitted_mxl_safe():
    """Safe config: price ~ N(-2.0, 0.3), |mean/sd|>3, no zero crossings."""
    cfg = DGPConfig(
        n_individuals=300,
        n_situations_per_individual=8,
        n_alternatives=3,
        attributes=[
            AttributeSpec("price", "normal", mean=-2.0, sd=0.3,
                          levels=(0.5, 1.0, 1.5, 2.0)),
            AttributeSpec("quality", "normal", mean=0.8, sd=0.5,
                          levels=(-1.0, 0.0, 1.0)),
            AttributeSpec("brand_known", "fixed", mean=0.6,
                          levels=(0.0, 1.0)),
        ],
        seed=42,
    )
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=100, halton_seed=0, compute_se=False)
    return res


# --- Median recovery ---------------------------------------------------------


def test_median_wtp_recovers_truth_default(fitted_mxl_default):
    """Default DGP has true population WTP = 0.667 for quality, 0.5 for brand.
    The median is robust even when the Cauchy warning fires."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wtps = compute_wtp_samples(
            fitted_mxl_default, n_draws=20_000, halton_seed=0
        )

    # quality WTP median ~ -0.8 / -1.2 = 0.667 (true population value)
    assert abs(wtps["quality"].median - 0.667) < 0.05
    # brand_known WTP median ~ 0.5
    assert abs(wtps["brand_known"].median - 0.5) < 0.10


def test_median_wtp_recovers_truth_safe(fitted_mxl_safe):
    """Safe DGP: true quality WTP = -0.8 / -2.0 = 0.4, brand = 0.3."""
    wtps = compute_wtp_samples(fitted_mxl_safe, n_draws=10_000, halton_seed=0)
    assert abs(wtps["quality"].median - 0.4) < 0.10
    assert abs(wtps["brand_known"].median - 0.3) < 0.10


# --- Cauchy warning behaviour -----------------------------------------------


def test_warning_fires_for_threshold_config(fitted_mxl_default):
    """Default DGP has price coef estimated near |mean/sd|=2.7, below the
    safety threshold of 3.0. The warning should fire."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_wtp_samples(fitted_mxl_default, n_draws=1000, halton_seed=0)
    cauchy_warns = [
        w for w in caught
        if "near zero" in str(w.message) or "Cauchy" in str(w.message)
    ]
    assert len(cauchy_warns) >= 1


def test_no_warning_for_safe_config(fitted_mxl_safe):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_wtp_samples(fitted_mxl_safe, n_draws=1000, halton_seed=0)
    cauchy_warns = [
        w for w in caught
        if "near zero" in str(w.message) or "Cauchy" in str(w.message)
    ]
    assert len(cauchy_warns) == 0


def test_price_distribution_helper_lognormal_safe():
    """Lognormal price coefs are by construction never zero."""
    cfg = DGPConfig(
        attributes=[
            AttributeSpec("price", "lognormal", mean=0.0, sd=0.5),
            AttributeSpec("x", "normal", mean=1.0, sd=0.3),
        ],
        seed=0,
    )
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=50, halton_seed=0, compute_se=False)
    assert _price_distribution_can_be_zero(res, "price") is False


# --- WTPDistribution methods -------------------------------------------------


def test_wtp_distribution_summary_keys():
    samples = np.random.default_rng(0).normal(0.5, 0.2, size=1000)
    w = WTPDistribution("x", samples=samples, price_can_be_zero=False)
    s = w.summary()
    expected_keys = {
        "attribute", "mean", "median", "trimmed_mean_5pct", "std",
        "p05", "p25", "p50", "p75", "p95", "iqr",
    }
    assert expected_keys.issubset(s.keys())
    # No warning key when price_can_be_zero=False
    assert "warning" not in s


def test_wtp_summary_has_warning_key_when_unsafe():
    samples = np.random.default_rng(0).normal(0.5, 0.2, size=1000)
    w = WTPDistribution("x", samples=samples, price_can_be_zero=True)
    s = w.summary()
    assert "warning" in s


def test_share_above_and_between():
    samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    w = WTPDistribution("x", samples=samples, price_can_be_zero=False)
    assert w.share_above(0.4) == pytest.approx(0.6)
    assert w.share_between(0.2, 0.8) == pytest.approx(0.6)


def test_quantile_returns_array_for_list_input():
    samples = np.linspace(0.0, 1.0, 1001)
    w = WTPDistribution("x", samples=samples, price_can_be_zero=False)
    q = w.quantile([0.25, 0.5, 0.75])
    assert q.shape == (3,)
    assert q[1] == pytest.approx(0.5, abs=0.001)


# --- Feature ranking ---------------------------------------------------------


def test_feature_ranking_default_sort_descending(fitted_mxl_safe):
    wtps = compute_wtp_samples(fitted_mxl_safe, n_draws=5000, halton_seed=0)
    ranking = feature_preference_ranking(wtps, by="median")
    medians = ranking["median"].to_numpy()
    # quality (true 0.4) should rank above brand (true 0.3)
    assert ranking.index[0] == "quality"
    assert (np.diff(medians) <= 0).all()


def test_feature_ranking_rejects_unknown_by(fitted_mxl_safe):
    wtps = compute_wtp_samples(fitted_mxl_safe, n_draws=500, halton_seed=0)
    with pytest.raises(ValueError, match="by must be"):
        feature_preference_ranking(wtps, by="banana")


def test_feature_ranking_excludes_price():
    """The price attribute itself must not appear in the WTP table."""
    cfg = default_config()
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=50, halton_seed=0, compute_se=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wtps = compute_wtp_samples(res, n_draws=500, halton_seed=0)
    assert "price" not in wtps


# --- Validation --------------------------------------------------------------


def test_unknown_price_attr_raises(fitted_mxl_default):
    with pytest.raises(ValueError, match="not in"):
        compute_wtp_samples(fitted_mxl_default, price_attr="nonexistent")
