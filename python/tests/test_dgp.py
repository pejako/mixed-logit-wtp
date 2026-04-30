"""Tests for the synthetic DGP.

These tests verify the *structure* of the data (one chosen per situation,
correct shapes, etc.) and the *direction* of effects implied by the true
coefficients. Parameter-recovery tests, which actually check that an
estimator recovers the true betas, live in test_recovery.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixedlogit.dgp import AttributeSpec, DGPConfig, default_config, simulate_choices

# --- Shape and structural tests -----------------------------------------------


def test_default_config_runs():
    df = simulate_choices(seed=0)
    cfg = default_config()
    expected_rows = (
        cfg.n_individuals * cfg.n_situations_per_individual * cfg.n_alternatives
    )
    assert len(df) == expected_rows


def test_required_columns_present():
    df = simulate_choices(seed=0)
    for col in ("individual_id", "situation_id", "alt_id", "chosen"):
        assert col in df.columns


def test_exactly_one_choice_per_situation():
    df = simulate_choices(seed=0)
    counts = df.groupby("situation_id")["chosen"].sum()
    assert (counts == 1).all(), "Each situation must have exactly one chosen alternative"


def test_chosen_is_binary():
    df = simulate_choices(seed=0)
    assert set(df["chosen"].unique()).issubset({0, 1})


def test_alt_ids_are_consecutive_within_situation():
    df = simulate_choices(seed=0)
    cfg = default_config()
    for _, group in df.groupby("situation_id"):
        assert sorted(group["alt_id"].tolist()) == list(range(cfg.n_alternatives))


def test_seed_reproducibility():
    df1 = simulate_choices(seed=123)
    df2 = simulate_choices(seed=123)
    assert df1.equals(df2)


def test_different_seeds_produce_different_data():
    df1 = simulate_choices(seed=1)
    df2 = simulate_choices(seed=2)
    # Choice patterns should differ
    assert not (df1["chosen"].values == df2["chosen"].values).all()


def test_ground_truth_metadata_attached():
    df = simulate_choices(seed=0)
    truth = df.attrs.get("true_params")
    assert truth is not None
    assert "attributes" in truth
    names = [a["name"] for a in truth["attributes"]]
    assert names == ["price", "quality", "brand_known"]


# --- Outside option ----------------------------------------------------------


def test_outside_option_has_zero_attributes():
    cfg = DGPConfig(
        n_individuals=50,
        n_situations_per_individual=4,
        n_alternatives=3,
        include_outside_option=True,
        attributes=[
            AttributeSpec("price", "normal", mean=-1.0, sd=0.3, levels=(0.5, 1.0, 1.5)),
            AttributeSpec("quality", "normal", mean=0.5, sd=0.4, levels=(-1.0, 0.0, 1.0)),
        ],
        seed=7,
    )
    df = simulate_choices(cfg)
    outside = df[df["alt_id"] == 0]
    assert (outside["price"] == 0.0).all()
    assert (outside["quality"] == 0.0).all()


# --- Behavioural / sign tests -------------------------------------------------


def test_chosen_alts_have_lower_price_when_price_coef_negative():
    df = simulate_choices(seed=42)
    chosen_price = df.loc[df["chosen"] == 1, "price"].mean()
    overall_price = df["price"].mean()
    assert chosen_price < overall_price, (
        f"chosen mean price {chosen_price:.3f} should be < overall {overall_price:.3f}"
    )


def test_chosen_alts_have_higher_quality_when_quality_coef_positive():
    df = simulate_choices(seed=42)
    chosen_q = df.loc[df["chosen"] == 1, "quality"].mean()
    overall_q = df["quality"].mean()
    assert chosen_q > overall_q


# --- Validation of AttributeSpec / DGPConfig ---------------------------------


def test_attribute_spec_rejects_sd_for_fixed():
    with pytest.raises(ValueError, match="fixed"):
        AttributeSpec("x", "fixed", mean=1.0, sd=0.5)


def test_attribute_spec_rejects_negative_sd():
    with pytest.raises(ValueError, match="non-negative"):
        AttributeSpec("x", "normal", mean=0.0, sd=-1.0)


def test_attribute_spec_rejects_unknown_dist():
    with pytest.raises(ValueError, match="dist"):
        AttributeSpec("x", "weibull", mean=0.0, sd=1.0)  # type: ignore[arg-type]


def test_dgpconfig_rejects_duplicate_attribute_names():
    with pytest.raises(ValueError, match="Duplicate"):
        DGPConfig(
            attributes=[
                AttributeSpec("price", "normal", mean=-1.0, sd=0.2),
                AttributeSpec("price", "normal", mean=-0.5, sd=0.1),
            ]
        )


def test_dgpconfig_rejects_too_few_alternatives():
    with pytest.raises(ValueError, match="n_alternatives"):
        DGPConfig(
            n_alternatives=1,
            attributes=[AttributeSpec("price", "normal", mean=-1.0, sd=0.2)],
        )


def test_dgpconfig_rejects_empty_attributes():
    with pytest.raises(ValueError, match="at least one attribute"):
        DGPConfig(attributes=[])


# --- Coefficient draws --------------------------------------------------------


def test_lognormal_coefficients_are_positive():
    spec = AttributeSpec("x", "lognormal", mean=0.0, sd=0.5)
    rng = np.random.default_rng(0)
    draws = spec.draw_coefficients(10_000, rng)
    assert (draws > 0).all()


def test_fixed_coefficients_are_constant():
    spec = AttributeSpec("x", "fixed", mean=2.5, sd=0.0)
    rng = np.random.default_rng(0)
    draws = spec.draw_coefficients(100, rng)
    assert np.allclose(draws, 2.5)


def test_normal_coefficients_match_population_moments():
    spec = AttributeSpec("x", "normal", mean=1.0, sd=0.5)
    rng = np.random.default_rng(0)
    draws = spec.draw_coefficients(50_000, rng)
    # Loose tolerances — these are sample moments
    assert abs(draws.mean() - 1.0) < 0.02
    assert abs(draws.std() - 0.5) < 0.02
