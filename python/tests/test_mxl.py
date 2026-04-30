"""Tests for the Mixed Logit estimator.

The headline test, ``test_mxl_recovers_default_dgp_across_seeds``, is the
correctness anchor for the entire project: it generates synthetic data with
known parameters across several seeds and asserts MXL recovers each one
within 2 standard errors of truth.

Recovery tests are slower than the others (each fit is ~30s on the default
config). They run in CI but you can skip them locally with
``pytest -m 'not slow'``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixedlogit.dgp import (
    AttributeSpec,
    DGPConfig,
    default_config,
    simulate_choices,
)
from mixedlogit.mxl import (
    MXLResult,
    _build_param_layout,
    _draw_betas,
    _mxl_neg_loglik,
    fit_mxl,
)

# --- Parameter packing -------------------------------------------------------


def test_param_layout_for_mixed_specs():
    specs = [
        AttributeSpec("price", "normal", mean=-1.0, sd=0.3),
        AttributeSpec("brand", "fixed", mean=0.5, sd=0.0),
        AttributeSpec("speed", "lognormal", mean=0.0, sd=0.5),
    ]
    names, layout = _build_param_layout(specs)
    assert names == [
        "price [mean]", "price [sd]",
        "brand",
        "speed [mu]", "speed [sigma]",
    ]
    assert len(layout) == 3
    # 2 + 1 + 2 = 5 parameters
    total_params = sum(sl.stop - sl.start for _, _, sl in layout)
    assert total_params == 5


def test_draw_betas_shapes_and_signs():
    specs = [
        AttributeSpec("a", "normal", mean=1.0, sd=0.5),
        AttributeSpec("b", "lognormal", mean=0.0, sd=0.3),
        AttributeSpec("c", "fixed", mean=2.5),
    ]
    _, layout = _build_param_layout(specs)
    theta = np.array([1.0, 0.5, 0.0, 0.3, 2.5])
    rng = np.random.default_rng(0)
    z = rng.standard_normal(size=(10, 5, 2))  # 2 random dims (a and b)
    betas = _draw_betas(theta, layout, n_individuals=10, n_attr=3, z_random=z)

    assert betas.shape == (10, 5, 3)
    # b is lognormal -> all positive
    assert (betas[:, :, 1] > 0).all()
    # c is fixed -> constant
    assert np.allclose(betas[:, :, 2], 2.5)


# --- Likelihood structural tests --------------------------------------------


def test_mxl_reduces_to_mnl_with_zero_sds():
    """When all random-coefficient SDs are 0, MXL log-likelihood == MNL."""
    df = simulate_choices(seed=0)
    specs = [
        AttributeSpec("price", "normal", mean=-1.0, sd=0.0),
        AttributeSpec("quality", "normal", mean=0.5, sd=0.0),
        AttributeSpec("brand_known", "fixed", mean=0.5),
    ]
    # Build the X/y/z structures the way fit_mxl does
    from mixedlogit.mnl import _mnl_neg_loglik_and_grad, _reshape_long_to_arrays

    X, y = _reshape_long_to_arrays(df, ["price", "quality", "brand_known"])
    n_alts = X.shape[1]
    n_attr = X.shape[2]
    n_indiv = df["individual_id"].nunique()
    n_sit_per = X.shape[0] // n_indiv
    X4 = X.reshape(n_indiv, n_sit_per, n_alts, n_attr)
    y2 = y.reshape(n_indiv, n_sit_per)

    _, layout = _build_param_layout(specs)
    theta = np.array([-0.5, 0.0, 0.3, 0.0, 0.4])  # mean,sd, mean,sd, fixed
    # z values irrelevant when sds are 0, but shape must match
    z = np.zeros((n_indiv, 50, 2))
    nll_mxl = _mxl_neg_loglik(theta, X4, y2, layout, z)

    # MNL with the same coefficients (means + the fixed value)
    beta_mnl = np.array([-0.5, 0.3, 0.4])
    nll_mnl, _ = _mnl_neg_loglik_and_grad(beta_mnl, X, y)

    # Should match to numerical precision
    assert np.isclose(nll_mxl, nll_mnl, rtol=1e-8)


def test_likelihood_is_finite_at_initial_values():
    df = simulate_choices(seed=0)
    cfg = default_config()
    # Just check the LL evaluates without overflow / NaN at sensible values
    from mixedlogit.halton import standard_normal_draws
    from mixedlogit.mnl import _reshape_long_to_arrays

    attr_names = [s.name for s in cfg.attributes]
    X, y = _reshape_long_to_arrays(df, attr_names)
    n_indiv = df["individual_id"].nunique()
    n_sit_per = X.shape[0] // n_indiv
    X4 = X.reshape(n_indiv, n_sit_per, X.shape[1], X.shape[2])
    y2 = y.reshape(n_indiv, n_sit_per)

    _, layout = _build_param_layout(cfg.attributes)
    theta = np.array([-1.2, 0.4, 0.8, 0.5, 0.6])
    z = standard_normal_draws(n_indiv * 50, 2, seed=0).reshape(n_indiv, 50, 2)

    nll = _mxl_neg_loglik(theta, X4, y2, layout, z)
    assert np.isfinite(nll)
    assert nll > 0  # negative log-likelihood is positive


# --- API-level tests ---------------------------------------------------------


def test_fit_returns_correct_result_shape():
    cfg = default_config()
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=50)
    assert isinstance(res, MXLResult)
    # 5 params: 2 normals (2 each) + 1 fixed = 5
    assert res.coefficients.shape == (5,)
    assert res.std_errors.shape == (5,)
    assert res.coef_names == [
        "price [mean]", "price [sd]",
        "quality [mean]", "quality [sd]",
        "brand_known",
    ]


def test_fit_rejects_unbalanced_individuals():
    df = simulate_choices(seed=0)
    # Drop one situation entirely from one individual
    drop_sit = df[df["individual_id"] == 0]["situation_id"].iloc[0]
    bad = df[df["situation_id"] != drop_sit].reset_index(drop=True)
    cfg = default_config()
    with pytest.raises(ValueError, match="Unbalanced"):
        fit_mxl(bad, cfg.attributes, n_draws=20)


def test_init_shape_validated():
    cfg = default_config()
    df = simulate_choices(cfg)
    with pytest.raises(ValueError, match="init"):
        fit_mxl(df, cfg.attributes, n_draws=20, init=np.zeros(2))


# --- THE recovery test ------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [42, 123, 2024])
def test_mxl_recovers_default_dgp_across_seeds(seed: int):
    """Generate data with known truth, fit MXL, check recovery within ~2.5 SE.

    This is the *correctness anchor* for the whole project. If this fails,
    something is wrong in the DGP, the Halton machinery, or the likelihood.

    Runs at a smaller scale (n=300, R=100) than a real production fit for
    test speed; CI passing here plus the full-scale demo in notebook 02
    together establish correctness.
    """
    cfg = DGPConfig(
        n_individuals=300,
        n_situations_per_individual=8,
        n_alternatives=3,
        attributes=[
            AttributeSpec("price", "normal", mean=-1.2, sd=0.4,
                          levels=(0.5, 1.0, 1.5, 2.0)),
            AttributeSpec("quality", "normal", mean=0.8, sd=0.5,
                          levels=(-1.0, 0.0, 1.0)),
            AttributeSpec("brand_known", "fixed", mean=0.6, sd=0.0,
                          levels=(0.0, 1.0)),
        ],
        seed=seed,
    )
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=100, halton_seed=seed)

    assert res.converged, f"MXL did not converge for seed={seed}"

    truth = {
        "price [mean]": -1.2,
        "price [sd]": 0.4,
        "quality [mean]": 0.8,
        "quality [sd]": 0.5,
        "brand_known": 0.6,
    }
    estimates = dict(zip(res.coef_names, res.coefficients, strict=True))
    ses = dict(zip(res.coef_names, res.std_errors, strict=True))

    failures = []
    for name, true_val in truth.items():
        est = estimates[name]
        se = ses[name]
        z = (est - true_val) / se
        # 2.5 SE -> ~99% confidence; allows occasional sampling noise without
        # producing flaky tests.
        if abs(z) > 2.5:
            failures.append(
                f"  {name}: true={true_val:+.3f} est={est:+.3f} "
                f"se={se:.3f} z={z:+.2f}"
            )

    if failures:
        msg = f"Recovery failed for seed={seed}:\n" + "\n".join(failures)
        pytest.fail(msg)


@pytest.mark.slow
def test_mxl_recovers_pure_mnl_data():
    """When the DGP has no heterogeneity, MXL should still recover the truth
    and the estimated SDs should be small (close to zero)."""
    cfg = DGPConfig(
        n_individuals=400,
        n_situations_per_individual=8,
        n_alternatives=3,
        attributes=[
            AttributeSpec("price", "normal", mean=-1.0, sd=0.0,
                          levels=(0.5, 1.0, 1.5, 2.0)),
            AttributeSpec("quality", "fixed", mean=0.5, levels=(-1.0, 0.0, 1.0)),
        ],
        seed=7,
    )
    df = simulate_choices(cfg)
    specs_for_fit = [
        AttributeSpec("price", "normal", mean=-1.0, sd=0.3,
                      levels=(0.5, 1.0, 1.5, 2.0)),
        AttributeSpec("quality", "fixed", mean=0.5, levels=(-1.0, 0.0, 1.0)),
    ]
    res = fit_mxl(df, specs_for_fit, n_draws=100, halton_seed=7)
    assert res.converged

    estimates = dict(zip(res.coef_names, res.coefficients, strict=True))
    ses = dict(zip(res.coef_names, res.std_errors, strict=True))

    # price mean should be near -1.0
    assert abs(estimates["price [mean]"] - (-1.0)) < 4 * ses["price [mean]"]
    # price sd should be small (true value is 0). Bounded below 0.3.
    assert estimates["price [sd]"] < 0.3
    # quality fixed near 0.5
    assert abs(estimates["quality"] - 0.5) < 4 * ses["quality"]
