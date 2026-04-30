"""Tests for the elasticity module.

Verify two kinds of properties:

1. **Mathematical structure** — IIA in MNL (rows of cross-elasticity columns
   are constant), correct signs, sensible magnitudes.
2. **Behavioural** — MXL produces row-varying cross-elasticities on a
   differentiated design, demonstrating IIA failure as expected.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixedlogit.dgp import (
    default_config,
    simulate_choices,
)
from mixedlogit.elasticity import (
    ElasticityMatrix,
    mnl_aggregate_elasticities,
    mxl_aggregate_elasticities,
    substitution_pattern_summary,
)
from mixedlogit.mnl import fit_mnl
from mixedlogit.mxl import fit_mxl

# Fixtures --------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_mnl():
    cfg = default_config()
    df = simulate_choices(cfg)
    return df, fit_mnl(df, ["price", "quality", "brand_known"])


@pytest.fixture(scope="module")
def fitted_mxl():
    """Module-scoped: fit MXL once, reuse across tests. Saves ~30s per test."""
    cfg = default_config()
    df = simulate_choices(cfg)
    res = fit_mxl(df, cfg.attributes, n_draws=100, halton_seed=0, compute_se=False)
    return df, res


@pytest.fixture
def differentiated_design():
    """Premium / Mid / Budget design used for several tests."""
    return np.array(
        [
            [2.0, 1.0, 1.0],   # Premium
            [1.0, 0.0, 1.0],   # Mid
            [0.5, -1.0, 0.0],  # Budget
        ]
    )


# --- Structural / matrix-shape tests ----------------------------------------


def test_mnl_elasticity_matrix_shape(fitted_mnl):
    df, mnl = fitted_mnl
    E = mnl_aggregate_elasticities(mnl, df)
    assert E.matrix.shape == (3, 3)
    assert E.model == "MNL"
    assert len(E.alt_labels) == 3


def test_mxl_elasticity_matrix_shape(fitted_mxl):
    df, mxl = fitted_mxl
    E = mxl_aggregate_elasticities(mxl, df, n_draws=200)
    assert E.matrix.shape == (3, 3)
    assert E.model == "MXL"


def test_to_dataframe(fitted_mnl):
    df, mnl = fitted_mnl
    E = mnl_aggregate_elasticities(mnl, df)
    edf = E.to_dataframe()
    assert edf.shape == (3, 3)
    assert list(edf.columns) == E.alt_labels
    assert list(edf.index) == E.alt_labels


# --- Sign tests --------------------------------------------------------------


def test_mnl_own_elasticity_negative(fitted_mnl, differentiated_design):
    """For normal goods (beta_p < 0), own-price elasticity must be negative."""
    df, mnl = fitted_mnl
    E = mnl_aggregate_elasticities(mnl, df, design=differentiated_design)
    for i in range(3):
        assert E.matrix[i, i] < 0, f"Own-elasticity at row {i} not negative: {E.matrix[i,i]}"


def test_mnl_cross_elasticity_positive(fitted_mnl, differentiated_design):
    """For substitutes, cross-price elasticities are positive."""
    df, mnl = fitted_mnl
    E = mnl_aggregate_elasticities(mnl, df, design=differentiated_design)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert E.matrix[i, j] > 0, (
                    f"Cross-elasticity ({i},{j}) not positive: {E.matrix[i,j]}"
                )


def test_mxl_own_elasticity_negative(fitted_mxl, differentiated_design):
    df, mxl = fitted_mxl
    E = mxl_aggregate_elasticities(
        mxl, df, n_draws=500, design=differentiated_design
    )
    for i in range(3):
        assert E.matrix[i, i] < 0


def test_mxl_cross_elasticity_positive(fitted_mxl, differentiated_design):
    df, mxl = fitted_mxl
    E = mxl_aggregate_elasticities(
        mxl, df, n_draws=500, design=differentiated_design
    )
    for i in range(3):
        for j in range(3):
            if i != j:
                assert E.matrix[i, j] > 0


# --- IIA structural test (the headline test) --------------------------------


def test_mnl_satisfies_iia(fitted_mnl, differentiated_design):
    """In MNL, every column j of the cross-elasticity matrix has identical
    off-diagonal entries: that is the IIA property."""
    df, mnl = fitted_mnl
    E = mnl_aggregate_elasticities(mnl, df, design=differentiated_design)
    for j in range(3):
        col = E.matrix[:, j]
        off_diag = np.delete(col, j)
        assert np.allclose(off_diag, off_diag[0], atol=1e-10), (
            f"MNL column {j} off-diagonals not constant: {off_diag}"
        )


def test_mxl_violates_iia_on_differentiated_design(
    fitted_mxl, differentiated_design
):
    """The point of MXL: cross-elasticities should differ by row when the
    design is differentiated and preferences are heterogeneous."""
    df, mxl = fitted_mxl
    E = mxl_aggregate_elasticities(
        mxl, df, n_draws=2000, halton_seed=0, design=differentiated_design
    )
    # For at least one column, the off-diagonal entries should NOT all match.
    found_violation = False
    for j in range(3):
        col = E.matrix[:, j]
        off_diag = np.delete(col, j)
        if (off_diag.max() - off_diag.min()) > 0.05:
            found_violation = True
            break
    assert found_violation, (
        "MXL did not produce row-varying cross-elasticities — IIA appears intact, "
        "which suggests a problem in the elasticity calculation."
    )


# --- Comparison helper -------------------------------------------------------


def test_substitution_pattern_summary_shape(
    fitted_mnl, fitted_mxl, differentiated_design
):
    df, mnl = fitted_mnl
    _, mxl = fitted_mxl
    mnl_E = mnl_aggregate_elasticities(mnl, df, design=differentiated_design)
    mxl_E = mxl_aggregate_elasticities(
        mxl, df, n_draws=500, design=differentiated_design
    )
    summary = substitution_pattern_summary(mnl_E, mxl_E)
    assert summary.shape == (3, 4)
    assert "MXL row spread" in summary.columns
    # MNL row spread is implicit zero, so MXL spread should be the only nonzero
    assert (summary["MXL row spread"] >= 0).all()


def test_summary_rejects_mismatched_labels():
    a = ElasticityMatrix(
        matrix=np.zeros((2, 2)), alt_labels=["A", "B"], model="MNL"
    )
    b = ElasticityMatrix(
        matrix=np.zeros((2, 2)), alt_labels=["X", "Y"], model="MXL"
    )
    with pytest.raises(ValueError, match="alt_labels"):
        substitution_pattern_summary(a, b)


# --- Validation --------------------------------------------------------------


def test_mnl_rejects_wrong_design_shape(fitted_mnl):
    df, mnl = fitted_mnl
    with pytest.raises(ValueError, match="design"):
        mnl_aggregate_elasticities(mnl, df, design=np.zeros((4, 3)))


def test_mxl_rejects_wrong_design_shape(fitted_mxl):
    df, mxl = fitted_mxl
    with pytest.raises(ValueError, match="design"):
        mxl_aggregate_elasticities(mxl, df, n_draws=50, design=np.zeros((2, 3)))


def test_unknown_price_attr_raises(fitted_mnl):
    df, mnl = fitted_mnl
    with pytest.raises(ValueError, match="not found|Could not"):
        mnl_aggregate_elasticities(mnl, df, price_attr="nonexistent")
