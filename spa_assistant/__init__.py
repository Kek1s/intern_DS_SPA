"""Public interface required by the DS assignment."""

from .forecast import forecast_demand
from .normalize import normalize_movement
from .questions import answer_question

__all__ = ["normalize_movement", "forecast_demand", "answer_question"]
