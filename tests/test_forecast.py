"""Arithmetic oracles, demand-shift treatment and supply timing invariants."""

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import pytest

from spa_assistant import forecast_demand
from spa_assistant.forecast import (
    depletion_date,
    estimate_level,
    purchase_plan,
    round_order,
)


@pytest.mark.parametrize(
    ("sku", "days", "qty", "cost"),
    [
        ("OIL-001", 30, 10, 12590.50),
        ("OIL-001", 90, 120, 151086.00),
        ("SCRB-020", 30, 37, 66211.50),
        ("SCRB-020", 90, 98, 175371.00),
        ("WRAP-030", 30, 0, 0),
        ("WRAP-030", 90, 36, 87732.00),
    ],
)
def test_assignment_numeric_oracles(
    history: dict, sku: str, days: int, qty: float, cost: float
) -> None:
    """Hand-calculated expected quantities and money, independent of the code."""
    result = forecast_demand(history, sku, days, {})
    assert result["recommended_qty"] == qty
    assert result["estimated_cost"] == cost
    assert result["explanation"]
    assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize(
    ("need", "pack", "minimum", "expected"),
    [
        ("0", "5", "10", "0"),
        ("-1", "5", "10", "0"),
        ("0.01", "5", "10", "10"),
        ("10.000001", "5", "10", "15"),
        ("1", "6", "10", "12"),
        ("0.3", "0.1", "0.2", "0.3"),
    ],
)
def test_exact_pack_rounding(need: str, pack: str, minimum: str, expected: str) -> None:
    """Decimal arithmetic prevents float-driven extra packs at boundaries."""
    assert round_order(Decimal(need), Decimal(pack), Decimal(minimum)) == Decimal(
        expected
    )


def test_regime_shift_and_gap(history: dict) -> None:
    """Sustained demand doubles; the missing week remains explicitly missing."""
    scrub = forecast_demand(history, "SCRB-020", 30, {})
    wrap = forecast_demand(history, "WRAP-030", 30, {})
    assert scrub["avg_daily_consumption"] == pytest.approx(28.9 / 28)
    assert scrub["diagnostics"]["level_shift"]
    assert scrub["requires_clarification"]
    assert wrap["diagnostics"]["missing_weeks"] == 1
    assert wrap["avg_daily_consumption"] == pytest.approx(25.7 / 28)


def test_confidence_quality_and_horizon() -> None:
    """Shorter, gappy, volatile histories score lower than complete stable ones."""
    stable = estimate_level([7.0] * 12)["confidence"]
    assert estimate_level([7.0] * 3)["confidence"] < stable
    assert estimate_level([None] + [7.0] * 11)["confidence"] < stable
    assert estimate_level([1.0, 13.0] * 6)["confidence"] < stable
    with pytest.raises(ValueError):
        estimate_level([7.0] * 8 + [None] * 4)


def test_isolated_outlier_is_not_new_regime() -> None:
    """A single event must not double every future week's consumption."""
    result = estimate_level([7.0] * 11 + [700.0])
    assert not result["level_shift"]
    assert result["outliers_adjusted"] == 1
    assert result["weekly_level"] < Decimal(9)


def test_timing_and_conservative_mode(history: dict) -> None:
    """Undated supply may offset volume but cannot conceal a prior shortage."""
    default = forecast_demand(history, "OIL-001", 30, {})
    cautious = forecast_demand(
        history, "OIL-001", 30, {"credit_undated_incoming": False}
    )
    assert default["stockout_date"] == cautious["stockout_date"] == "2026-10-13"
    assert cautious["recommended_qty"] == 30
    history["incoming_deliveries"] = {"OIL-001": [{"date": "2026-12-01", "qty": 20}]}
    late = forecast_demand(history, "OIL-001", 30, {})
    assert late["eligible_incoming_qty"] == 0
    assert late["stockout_date"] == "2026-10-13"
    history["incoming_deliveries"]["OIL-001"][0]["date"] = "2026-09-20"
    early = forecast_demand(history, "OIL-001", 30, {})
    assert early["eligible_incoming_qty"] == 20
    assert early["stockout_date"] > late["stockout_date"]


def test_exact_depletion_and_same_day_receipts() -> None:
    """A receipt at opening prevents a shortage after yesterday's exact depletion."""
    start = date(2026, 1, 1)
    assert (
        depletion_date(Decimal(2), Decimal(1), start, [(date(2026, 1, 3), Decimal(2))])
        == "2026-01-05"
    )
    assert (
        depletion_date(Decimal(2), Decimal(1), start, [(date(2026, 1, 4), Decimal(2))])
        == "2026-01-03"
    )


def test_zero_demand_and_no_mutation(history: dict) -> None:
    """Zero demand yields neither a purchase nor a fabricated depletion date."""
    history["weekly_consumption"]["OIL-001"] = [0] * 12
    snapshot = deepcopy(history)
    result = forecast_demand(history, "OIL-001", 90, {})
    assert history == snapshot
    assert result["stockout_date"] is None
    assert result["recommended_qty"] == 0
    assert result["no_consumption"]
    assert result["excess_stock"]


def test_budget_never_silently_cuts_demand(history: dict) -> None:
    """A low budget is reported as a gap, not a falsely sufficient order."""
    result = forecast_demand(history, "SCRB-020", 180, {"budget_limit": 200_000})
    assert result["recommended_qty"] == 191
    assert result["budget_gap"] == 141794.5
    plan = purchase_plan(history, 90)
    assert plan["estimated_cost"] == 414189.0
    assert len(plan["missing_skus"]) == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "garbage"])
def test_invalid_observations(history: dict, bad: object) -> None:
    """Reject malformed data rather than propagating invalid totals."""
    history["weekly_consumption"]["OIL-001"][0] = bad
    with pytest.raises(ValueError):
        forecast_demand(history, "OIL-001", 30, {})


@pytest.mark.parametrize("days", [0, -1, 1.5, True, 4000])
def test_invalid_horizon(history: dict, days: object) -> None:
    """The horizon is a bounded positive integer."""
    with pytest.raises(ValueError):
        forecast_demand(history, "OIL-001", days, {})


def test_missing_data_and_inconsistent_schedule(history: dict) -> None:
    """Incomplete source data cannot become a zero quantity by default."""
    with pytest.raises(ValueError):
        forecast_demand(history, "OIL-002", 30, {})
    history["incoming_deliveries"] = {"OIL-001": [{"date": "2026-09-20", "qty": 19}]}
    with pytest.raises(ValueError):
        forecast_demand(history, "OIL-001", 30, {})


def test_scenario_scaling(history: dict) -> None:
    """Demand scaling affects consumption and reserve, not observed stock."""
    base = forecast_demand(history, "OIL-001", 30, {})
    growth = forecast_demand(history, "OIL-001", 30, {"demand_multiplier": 1.2})
    assert growth["forecast_demand"] == pytest.approx(base["forecast_demand"] * 1.2)
    assert growth["current_stock"] == base["current_stock"]


@pytest.mark.parametrize("weekly", [1, 2, 3, 6, 11, 19])
@pytest.mark.parametrize("weeks", [1, 2, 9])
def test_recurring_daily_fraction_exact_boundaries(weekly: int, weeks: int) -> None:
    """Repeating 1/7 rates must neither add a pack nor shift shortage a day early."""
    stock = weekly * weeks
    history = {
        "as_of": "2026-09-15",
        "weekly_consumption": {"OIL-001": [weekly] * 12},
        "current_stock": {"OIL-001": stock},
        "incoming_qty": {"OIL-001": 0},
    }
    product = {
        "sku": "OIL-001",
        "name": "Test",
        "unit": "л",
        "pack_size": 1,
        "min_order_qty": 1,
        "safety_stock_days": 0,
        "lead_time_days": 0,
        "price": 1,
    }
    result = forecast_demand(history, "OIL-001", weeks * 7, {"product": product})
    assert result["recommended_qty"] == 0
    assert (
        result["stockout_date"]
        == (date(2026, 9, 15) + timedelta(days=weeks * 7)).isoformat()
    )
