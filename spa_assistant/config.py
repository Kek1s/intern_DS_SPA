"""Explicit business assumptions and tunable heuristic thresholds."""

from dataclasses import dataclass

DAYS_PER_WEEK = 7
MONTH_DAYS = 30
QUARTER_DAYS = 90
HALF_YEAR_DAYS = 180
YEAR_DAYS = 365
CENTURY = 2000
MAX_TEXT_LENGTH = 10_000
GIGACHAT_TOKEN_REFRESH_MARGIN_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Confidence is a quality score, not a calibrated probability."""

    window_weeks: int = 4
    reference_weeks: int = 12
    min_regime_weeks: int = 4
    regime_ratio: float = 1.8
    regime_max_cv: float = 0.25
    outlier_mad_factor: float = 4.5
    mad_scale: float = 1.4826
    outlier_relative_floor: float = 0.5
    confidence_threshold: float = 0.60
    unknown_eta_penalty: float = 0.90
    regime_penalty: float = 0.85
    horizon_penalty_rate: float = 0.08
    max_horizon_days: int = 3660
    excess_cover_days: int = 180


@dataclass(frozen=True, slots=True)
class QueryConfig:
    """Rules use scores as evidence strengths, not statistical probabilities."""

    intent_threshold: float = 0.60
    intent_margin: float = 0.15
    explicit_score: float = 0.95
    generic_score: float = 0.70
    weak_score: float = 0.40
    clarification_confidence_cap: float = 0.49
    llm_timeout_seconds: float = 5.0
    max_llm_response_bytes: int = 65_536


FORECAST = ForecastConfig()
QUERY = QueryConfig()
