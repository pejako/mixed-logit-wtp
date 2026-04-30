"""Synthetic data generation for Mixed Logit choice models.

The DGP generates choice data from a known random-utility specification:

    U_{nij} = beta_n' * x_{ij} + epsilon_{nij}

where:
  - n indexes individuals (decision-makers)
  - i indexes choice situations (panels per individual)
  - j indexes alternatives within a choice situation
  - beta_n is an individual-specific coefficient vector drawn from a known
    parametric distribution (normal, log-normal, or fixed)
  - epsilon_{nij} is iid Type-I Extreme Value (Gumbel)

Because we set the population parameters (means and standard deviations of
beta), every estimator downstream can be validated by recovering them.

Conventions
-----------
- Output is in *long format*: one row per alternative per choice situation.
- Required columns: individual_id, situation_id, alt_id, chosen, plus one
  column per attribute. This matches the input format expected by xlogit
  and R's mlogit/gmnl.
- Price is just another attribute. By convention, name it 'price' so the
  elasticity and WTP modules can find it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

CoefDist = Literal["fixed", "normal", "lognormal"]


@dataclass
class AttributeSpec:
    """Specification for a single attribute in the choice experiment.

    Parameters
    ----------
    name : str
        Column name for the attribute in the output dataframe.
    dist : {"fixed", "normal", "lognormal"}
        Distribution of the coefficient across individuals.
        - "fixed": all individuals share the same coefficient (= mean)
        - "normal": coefficient ~ N(mean, sd)
        - "lognormal": coefficient ~ exp(N(mean, sd))  -- strictly positive
    mean : float
        Population mean of the coefficient (or, for log-normal, the mean
        of the underlying normal).
    sd : float
        Population standard deviation. Must be 0 for "fixed".
    levels : tuple[float, ...] | None
        If given, attribute values are sampled uniformly from these levels
        (typical for designed choice experiments). If None, attribute values
        are drawn from a standard normal.
    """

    name: str
    dist: CoefDist
    mean: float
    sd: float = 0.0
    levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.dist == "fixed" and self.sd != 0.0:
            raise ValueError(f"Attribute '{self.name}': sd must be 0 for fixed coefficient")
        if self.sd < 0.0:
            raise ValueError(f"Attribute '{self.name}': sd must be non-negative")
        if self.dist not in ("fixed", "normal", "lognormal"):
            raise ValueError(f"Attribute '{self.name}': unknown dist '{self.dist}'")

    def draw_coefficients(self, n_individuals: int, rng: np.random.Generator) -> np.ndarray:
        """Draw individual-level coefficients for this attribute."""
        if self.dist == "fixed":
            return np.full(n_individuals, self.mean)
        if self.dist == "normal":
            return rng.normal(self.mean, self.sd, size=n_individuals)
        # lognormal
        return np.exp(rng.normal(self.mean, self.sd, size=n_individuals))

    def draw_values(
        self, n_situations: int, n_alternatives: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw attribute values across (situation, alternative) cells."""
        size = (n_situations, n_alternatives)
        if self.levels is not None:
            idx = rng.integers(0, len(self.levels), size=size)
            return np.asarray(self.levels)[idx]
        return rng.standard_normal(size=size)


@dataclass
class DGPConfig:
    """Top-level configuration for the synthetic choice experiment.

    Parameters
    ----------
    n_individuals : int
        Number of decision-makers.
    n_situations_per_individual : int
        Choice situations (panels) per individual.
    n_alternatives : int
        Alternatives per choice situation.
    attributes : list[AttributeSpec]
        Attribute specifications. Include one named 'price' for elasticity
        and WTP modules to work out of the box.
    include_outside_option : bool
        If True, alternative 0 is an outside option with all-zero attributes
        (utility = epsilon only). Useful for share-of-zero situations.
    seed : int | None
        Master seed for reproducibility.
    """

    n_individuals: int = 500
    n_situations_per_individual: int = 8
    n_alternatives: int = 3
    attributes: list[AttributeSpec] = field(default_factory=list)
    include_outside_option: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.attributes:
            raise ValueError("DGPConfig requires at least one attribute")
        names = [a.name for a in self.attributes]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate attribute names: {names}")
        if self.n_individuals <= 0 or self.n_situations_per_individual <= 0:
            raise ValueError("n_individuals and n_situations_per_individual must be positive")
        if self.n_alternatives < 2:
            raise ValueError("n_alternatives must be >= 2")


def simulate_choices(
    config: DGPConfig | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Simulate a Mixed Logit choice dataset.

    Parameters
    ----------
    config : DGPConfig, optional
        Full DGP configuration. If None, a sensible default is used (see
        :func:`default_config`) so the function can be called with just a seed.
    seed : int, optional
        If given, overrides config.seed. Convenience for one-line calls in
        notebooks and tests.

    Returns
    -------
    pandas.DataFrame
        Long-format choice data with columns:
            individual_id, situation_id, alt_id, chosen, <attribute columns...>
        plus a metadata attribute ``df.attrs["true_params"]`` containing the
        ground-truth coefficient distributions (for parameter-recovery tests).
    """
    if config is None:
        config = default_config()
    if seed is not None:
        config = DGPConfig(
            n_individuals=config.n_individuals,
            n_situations_per_individual=config.n_situations_per_individual,
            n_alternatives=config.n_alternatives,
            attributes=config.attributes,
            include_outside_option=config.include_outside_option,
            seed=seed,
        )

    rng = np.random.default_rng(config.seed)

    n_n = config.n_individuals
    n_s = config.n_situations_per_individual
    n_j = config.n_alternatives
    n_attr = len(config.attributes)

    # --- 1. Draw individual-level coefficients: shape (n_individuals, n_attr)
    betas = np.column_stack(
        [a.draw_coefficients(n_n, rng) for a in config.attributes]
    )

    # --- 2. Draw attribute values: shape (n_individuals, n_situations, n_alts, n_attr)
    # We draw a fresh (situation, alternative) grid per individual so the
    # design varies across people, as in a real stated-preference study.
    X = np.empty((n_n, n_s, n_j, n_attr))
    for k, attr in enumerate(config.attributes):
        # One value array per individual to keep memory layout simple
        for n in range(n_n):
            X[n, :, :, k] = attr.draw_values(n_s, n_j, rng)

    # If we have an outside option, zero out its attributes so its
    # systematic utility is 0.
    if config.include_outside_option:
        X[:, :, 0, :] = 0.0

    # --- 3. Compute deterministic utilities V = X @ beta_n
    # einsum: for each (n, s, j), sum_k X[n,s,j,k] * betas[n,k]
    V = np.einsum("nsjk,nk->nsj", X, betas)

    # --- 4. Add Gumbel(0,1) errors and pick the argmax
    # Gumbel via inverse CDF: -log(-log(U)), U ~ Uniform(0,1)
    U = rng.uniform(size=(n_n, n_s, n_j))
    epsilon = -np.log(-np.log(U))
    chosen_alt = np.argmax(V + epsilon, axis=2)  # shape (n_individuals, n_situations)

    # --- 5. Pivot to long format
    rows = []
    attr_names = [a.name for a in config.attributes]
    for n in range(n_n):
        for s in range(n_s):
            for j in range(n_j):
                row = {
                    "individual_id": n,
                    "situation_id": n * n_s + s,
                    "alt_id": j,
                    "chosen": int(chosen_alt[n, s] == j),
                }
                for k, name in enumerate(attr_names):
                    row[name] = X[n, s, j, k]
                rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["true_params"] = _ground_truth_dict(config)
    return df


def _ground_truth_dict(config: DGPConfig) -> dict:
    """Package the true population parameters for downstream validation."""
    return {
        "n_individuals": config.n_individuals,
        "n_situations_per_individual": config.n_situations_per_individual,
        "n_alternatives": config.n_alternatives,
        "include_outside_option": config.include_outside_option,
        "seed": config.seed,
        "attributes": [
            {"name": a.name, "dist": a.dist, "mean": a.mean, "sd": a.sd}
            for a in config.attributes
        ],
    }


def default_config() -> DGPConfig:
    """A sensible default DGP for quick demos and smoke tests.

    Three attributes:
      - price       : log-normal coefficient (negative utility, sign-constrained)
                      Note: we model -price implicitly by giving the lognormal
                      coefficient a negative role downstream. Here we just keep
                      the convention that 'price' enters utility with a negative
                      mean.
      - quality     : normal coefficient (heterogeneous tastes)
      - brand_known : fixed coefficient (homogeneous brand premium)
    """
    return DGPConfig(
        n_individuals=500,
        n_situations_per_individual=8,
        n_alternatives=3,
        attributes=[
            AttributeSpec(name="price", dist="normal", mean=-1.2, sd=0.4,
                          levels=(0.5, 1.0, 1.5, 2.0)),
            AttributeSpec(name="quality", dist="normal", mean=0.8, sd=0.5,
                          levels=(-1.0, 0.0, 1.0)),
            AttributeSpec(name="brand_known", dist="fixed", mean=0.6, sd=0.0,
                          levels=(0.0, 1.0)),
        ],
        seed=42,
    )
