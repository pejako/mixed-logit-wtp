"""Tests for the Halton draws module."""

from __future__ import annotations

import numpy as np
import pytest

from mixedlogit.halton import halton_draws, standard_normal_draws

# --- Shape & range -----------------------------------------------------------


def test_shape():
    draws = halton_draws(50, 4, seed=0)
    assert draws.shape == (50, 4)


def test_values_in_unit_interval():
    draws = halton_draws(200, 5, seed=0)
    assert (draws >= 0).all()
    assert (draws < 1).all()


def test_each_dimension_uses_different_base():
    # With scrambled=False and seed irrelevant, dimensions 0 and 1 use bases
    # 2 and 3 -> they should not be identical sequences.
    draws = halton_draws(50, 2, scrambled=False)
    assert not np.allclose(draws[:, 0], draws[:, 1])


def test_too_many_dimensions_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        halton_draws(10, 200)


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        halton_draws(0, 2)
    with pytest.raises(ValueError):
        halton_draws(10, 0)


# --- Reproducibility ---------------------------------------------------------


def test_unscrambled_is_deterministic():
    a = halton_draws(50, 3, scrambled=False)
    b = halton_draws(50, 3, scrambled=False)
    assert np.array_equal(a, b)


def test_scrambled_is_seed_reproducible():
    a = halton_draws(50, 3, scrambled=True, seed=42)
    b = halton_draws(50, 3, scrambled=True, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_scrambling():
    a = halton_draws(50, 3, scrambled=True, seed=1)
    b = halton_draws(50, 3, scrambled=True, seed=2)
    assert not np.array_equal(a, b)


# --- Low-discrepancy property ------------------------------------------------


def _grid_variance(pts: np.ndarray, bins: int = 10) -> float:
    """Variance of point counts in a ``bins x bins`` grid; lower = more uniform."""
    h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=[[0, 1], [0, 1]])
    return float(h.var())


def test_halton_lower_discrepancy_than_uniform():
    """Halton should distribute more evenly than pseudo-random in 2D.

    On a 10x10 grid with 200 points, expected count per cell is 2; uniform
    random has high cell-count variance, Halton has much lower.
    """
    rng = np.random.default_rng(0)
    uniform = rng.uniform(size=(200, 2))
    halton = halton_draws(200, 2, scrambled=False)
    assert _grid_variance(halton) < _grid_variance(uniform)


# --- Standard normal transform ----------------------------------------------


def test_standard_normal_moments():
    z = standard_normal_draws(20_000, 3, seed=0)
    # Halton-based draws give very tight sample moments
    assert np.all(np.abs(z.mean(axis=0)) < 0.02)
    assert np.all(np.abs(z.std(axis=0) - 1.0) < 0.02)


def test_standard_normal_no_inf():
    """Inverse-normal at u=0 or u=1 returns +/-inf; we clip to avoid that."""
    z = standard_normal_draws(500, 5, seed=0)
    assert np.isfinite(z).all()


def test_standard_normal_shape():
    z = standard_normal_draws(123, 7, seed=0)
    assert z.shape == (123, 7)
