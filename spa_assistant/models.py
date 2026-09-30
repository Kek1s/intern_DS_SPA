"""Structured contracts and validation at public input boundaries."""

from datetime import date
from typing import Any, NotRequired, TypedDict

from .catalog import decimal_number


class Delivery(TypedDict):
    """Receipt at the beginning of a calendar day."""

    date: str
    qty: float


class History(TypedDict):
    """Inventory snapshot and weekly consumption in catalog base units."""

    as_of: str
    weekly_consumption: dict[str, list[float | None]]
    current_stock: dict[str, float]
    incoming_qty: dict[str, float]
    incoming_deliveries: NotRequired[dict[str, list[Delivery]]]


class ForecastResult(TypedDict):
    """Required calculation fields; diagnostics remain explicit extensions."""

    sku: str
    unit: str
    as_of: str
    horizon_days: int
    avg_daily_consumption: float
    forecast_demand: float
    current_stock: float
    incoming_qty: float
    eligible_incoming_qty: float
    safety_stock: float
    reorder_point: float
    recommended_qty: float
    estimated_cost: float
    stockout_date: str | None
    stockout_basis: str
    recommended_order_date: str | None
    confidence: float
    requires_clarification: bool
    deficit_before_lead_time: bool
    low_stock: bool
    excess_stock: bool
    no_consumption: bool
    budget_limit: float | None
    budget_gap: float | None
    diagnostics: dict[str, Any]
    assumptions: dict[str, Any]
    warnings: list[str]
    explanation: str


class Period(TypedDict):
    """A fixed horizon or a delivery event requiring a date."""

    kind: str
    days: int | None
    start: NotRequired[str]
    end: NotRequired[str]


class ParsedQuery(TypedDict):
    """Validated query entities and auditable routing metadata."""

    intent: str
    sku: str | None
    sku_candidates: list[str]
    location: str | None
    period: Period | None
    budget_limit: float | None
    demand_multiplier: float
    intent_scores: dict[str, float]
    intent_margin: float
    router: str
    llm_status: str
    status: str
    results: list[ForecastResult]
    reason: NotRequired[str]
    missing_skus: NotRequired[list[str]]
    estimated_cost: NotRequired[float]
    budget_gap: NotRequired[float | None]


def validate_history(history: object) -> date:
    """Raise ValueError consistently for malformed snapshot containers."""
    if not isinstance(history, dict):
        raise ValueError("history: требуется словарь")
    try:
        as_of = date.fromisoformat(history["as_of"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("history.as_of: требуется дата ISO") from error
    for field in ("weekly_consumption", "current_stock", "incoming_qty"):
        if not isinstance(history.get(field), dict):
            raise ValueError(f"history.{field}: требуется словарь по SKU")
    schedules = history.get("incoming_deliveries", {})
    if not isinstance(schedules, dict):
        raise ValueError("incoming_deliveries: требуется словарь по SKU")
    for schedule in schedules.values():
        if not isinstance(schedule, list):
            raise ValueError("График поставок должен быть списком")
        for row in schedule:
            if not isinstance(row, dict) or not {"date", "qty"} <= row.keys():
                raise ValueError("Поставка должна содержать date и qty")
            try:
                date.fromisoformat(row["date"])
            except (TypeError, ValueError) as error:
                raise ValueError("delivery.date: требуется дата ISO") from error
    return as_of


def validate_previous_query(previous: dict[str, Any]) -> None:
    """Validate persisted follow-up entities before inheriting them."""
    for field in ("sku", "location"):
        if previous.get(field) is not None and not isinstance(previous[field], str):
            raise ValueError(f"previous_params.{field}: требуется строка")
    for field in ("budget_limit", "demand_multiplier"):
        if previous.get(field) is not None:
            decimal_number(previous[field], f"previous_params.{field}")
    period = previous.get("period")
    if period is not None:
        if not isinstance(period, dict) or not {"kind", "days"} <= period.keys():
            raise ValueError("previous_params.period: нужны kind и days")
        if period["kind"] not in {
            "rolling_days",
            "calendar_month",
            "until_next_delivery",
        }:
            raise ValueError("previous_params.period: неизвестный вид периода")
        if period["kind"] != "until_next_delivery" and (
            type(period["days"]) is not int or period["days"] <= 0
        ):
            raise ValueError("previous_params.period.days: нужно положительное целое")
