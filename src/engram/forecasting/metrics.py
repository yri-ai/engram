"""Pure forecast metric seam.

The current implementation re-exports the canonical scoring functions from
``engram.services.forecast_scoring`` so the codebase has one Brier/log/top-k
implementation while downstream R&D code can depend on ``engram.forecasting``.
"""

from engram.services.forecast_scoring import (
    assign_calibration_bucket,
    binary_brier_score,
    log_score,
    multiclass_brier_score,
    probability_assigned_to_resolved_branch,
    top_1_accuracy,
    top_k_accuracy,
)

__all__ = [
    "assign_calibration_bucket",
    "binary_brier_score",
    "log_score",
    "multiclass_brier_score",
    "probability_assigned_to_resolved_branch",
    "top_1_accuracy",
    "top_k_accuracy",
]
