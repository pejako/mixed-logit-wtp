"""Mixed Logit choice modeling toolkit.

A reproducible implementation of Mixed Logit (random-coefficient) discrete
choice models, with synthetic data generation and parameter-recovery tests.
"""

from mixedlogit.dgp import AttributeSpec, DGPConfig, simulate_choices
from mixedlogit.elasticity import (
    ElasticityMatrix,
    mnl_aggregate_elasticities,
    mxl_aggregate_elasticities,
    substitution_pattern_summary,
)
from mixedlogit.halton import halton_draws, standard_normal_draws
from mixedlogit.mnl import MNLResult, fit_mnl
from mixedlogit.mxl import MXLResult, fit_mxl
from mixedlogit.wtp import (
    WTPDistribution,
    compute_wtp_samples,
    feature_preference_ranking,
)

__version__ = "0.1.0"

__all__ = [
    "AttributeSpec",
    "DGPConfig",
    "ElasticityMatrix",
    "MNLResult",
    "MXLResult",
    "WTPDistribution",
    "compute_wtp_samples",
    "feature_preference_ranking",
    "fit_mnl",
    "fit_mxl",
    "halton_draws",
    "mnl_aggregate_elasticities",
    "mxl_aggregate_elasticities",
    "simulate_choices",
    "standard_normal_draws",
    "substitution_pattern_summary",
]
