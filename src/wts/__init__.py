"""OT-Shield: the Wasserstein Trust Score for GenAI, RAG and agent security."""
from .conformal import ConformalCalibrator, conformal_pvalue, conformal_threshold, coverage_study
from .guards import AgentGuard, ChunkGuard, Decision, PromptGuard, split_reference_calibration
from .measures import LOCATION_ONLY, MEASURES, ReferenceModel, bures_wasserstein2, fit_gaussian, sliced_w2

__all__ = [
    "AgentGuard", "ChunkGuard", "ConformalCalibrator", "Decision", "LOCATION_ONLY", "MEASURES", "PromptGuard",
    "ReferenceModel", "bures_wasserstein2", "conformal_pvalue", "conformal_threshold", "coverage_study",
    "fit_gaussian", "sliced_w2", "split_reference_calibration",
]
__version__ = "0.1.0"
