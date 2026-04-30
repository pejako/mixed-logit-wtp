"""Tests for the MNL baseline model."""

from __future__ import annotations

import numpy as np
import pytest

from mixedlogit.dgp import (
    AttributeSpec,
    DGPConfig,
    default_config,
    simulate_choices,
)
from mixedlogit.mnl import _mnl_neg_loglik_and_grad, fit_mnl

# --- Reshape helper / fit smoke ----------------------------------------------


def test_fit_runs_on_default_data():
    df = simulate_choices(seed=0)
    res = fit_mnl(df, ["price", "quality", "brand_known"])
    assert res.converged
    assert res.coefficients.shape == (3,)
    assert res.std_errors.shape == (3,)
    assert np.isfinite(res.loglik)


def test_unbalanced_panels_rejected():
    df = simulate_choices(seed=0)
    # Drop one alternative from a single situation -> unbalanced
    bad = df.drop(df.index[0]).reset_index(drop=True)
    with pytest.raises(ValueError, match="Unbalanced"):
        fit_mnl(bad, ["price", "quality", "brand_known"])


def test_init_shape_validated():
    df = simulate_choices(seed=0)
    with pytest.raises(ValueError, match="init"):
        fit_mnl(df, ["price", "quality"], init=np.array([0.0]))


# --- Sign and consistency tests ----------------------------------------------


def test_sign_recovery_on_default_dgp():
    """MNL on data generated with random coefs is biased toward zero, but
    the sign of each coefficient should be correct."""
    df = simulate_choices(seed=42)
    res = fit_mnl(df, ["price", "quality", "brand_known"])
    coefs = dict(zip(res.coef_names, res.coefficients, strict=True))
    assert coefs["price"] < 0
    assert coefs["quality"] > 0
    assert coefs["brand_known"] > 0


def test_fixed_coefficient_recovered_when_no_heterogeneity():
    """If all attributes have fixed coefficients, MNL is the correct model and
    should recover the truth within statistical noise."""
    cfg = DGPConfig(
        n_individuals=2000,
        n_situations_per_individual=8,
        n_alternatives=3,
        attributes=[
            AttributeSpec("price", "fixed", mean=-1.0, levels=(0.5, 1.0, 1.5, 2.0)),
            AttributeSpec("quality", "fixed", mean=0.5, levels=(-1.0, 0.0, 1.0)),
        ],
        seed=99,
    )
    df = simulate_choices(cfg)
    res = fit_mnl(df, ["price", "quality"])
    assert res.converged
    # With 16k situations, SE will be small. 4 SE is generous.
    coefs = dict(zip(res.coef_names, res.coefficients, strict=True))
    ses = dict(zip(res.coef_names, res.std_errors, strict=True))
    assert abs(coefs["price"] - (-1.0)) < 4 * ses["price"]
    assert abs(coefs["quality"] - 0.5) < 4 * ses["quality"]


# --- Likelihood properties ---------------------------------------------------


def test_loglik_at_zero_equals_null():
    """LL at beta=0 must equal the equal-shares null log-likelihood."""
    df = simulate_choices(seed=0)
    cfg = default_config()
    expected_null = -cfg.n_individuals * cfg.n_situations_per_individual * np.log(
        cfg.n_alternatives
    )
    # Fit but with init = zero & max_iter = 0 isn't clean; just call the LL fn directly
    from mixedlogit.mnl import _reshape_long_to_arrays

    X, y = _reshape_long_to_arrays(df, ["price", "quality", "brand_known"])
    nll, _ = _mnl_neg_loglik_and_grad(np.zeros(3), X, y)
    assert np.isclose(-nll, expected_null)


def test_gradient_matches_finite_difference():
    """Analytic gradient must match a numerical estimate."""
    df = simulate_choices(seed=0)
    from mixedlogit.mnl import _reshape_long_to_arrays

    X, y = _reshape_long_to_arrays(df, ["price", "quality", "brand_known"])

    beta = np.array([-0.5, 0.3, 0.2])
    _, grad_analytic = _mnl_neg_loglik_and_grad(beta, X, y)

    eps = 1e-5
    grad_numeric = np.zeros_like(beta)
    for k in range(len(beta)):
        beta_up = beta.copy()
        beta_dn = beta.copy()
        beta_up[k] += eps
        beta_dn[k] -= eps
        f_up, _ = _mnl_neg_loglik_and_grad(beta_up, X, y)
        f_dn, _ = _mnl_neg_loglik_and_grad(beta_dn, X, y)
        grad_numeric[k] = (f_up - f_dn) / (2 * eps)

    assert np.allclose(grad_analytic, grad_numeric, rtol=1e-4, atol=1e-4)


def test_mcfadden_r2_in_unit_interval():
    df = simulate_choices(seed=0)
    res = fit_mnl(df, ["price", "quality", "brand_known"])
    assert 0.0 < res.mcfadden_r2 < 1.0


def test_t_values_significant_with_strong_signal():
    """All three default coefficients should be highly significant (|t| > 5)."""
    df = simulate_choices(seed=0)
    res = fit_mnl(df, ["price", "quality", "brand_known"])
    assert (np.abs(res.t_values) > 5).all()
