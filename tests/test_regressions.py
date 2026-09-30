"""Review counterexamples and invariants outside the assignment examples."""

from copy import deepcopy
from datetime import date
from decimal import Decimal
from fractions import Fraction
from unittest.mock import patch

import pytest

from spa_assistant import answer_question, forecast_demand, normalize_movement
from spa_assistant.evaluation import backtest
from spa_assistant.forecast import estimate_level, reorder_date
from spa_assistant.llm import INTENTS, _valid_scores


@pytest.mark.parametrize("separator", [" ", "\u00a0", "\u202f"])
def test_grouped_quantity(separator: str) -> None:
    """Thousands separators preserve the whole quantity, including decimals."""
    assert normalize_movement(f"OIL-001 приход 1{separator}000 мл")["qty"] == 1
    assert normalize_movement(f"SCRB-020 расход 1{separator}000,5 г")["qty"] == 1.0005


@pytest.mark.parametrize(
    "amount",
    [
        "1 00 мл",
        "-2 канистры по -5 л",
        "2 канистры по -5 л",
        "-2 канистры по 5 л",
        "1,5 канистры по 5 л",
    ],
)
def test_invalid_quantity_abstains(amount: str) -> None:
    """Invalid components cannot cancel each other or produce a partial parse."""
    assert normalize_movement(f"OIL-001 приход {amount}")["qty"] is None


@pytest.mark.parametrize(
    "intent", ["бюджет закупок", "риск дефицита", "список закупок"]
)
def test_multiple_skus_never_expand_to_catalog(history: dict, intent: str) -> None:
    """MVP asks for a single product rather than silently adding other products."""
    parsed, _, _ = answer_question(
        f"Покажи {intent} OIL-001 и WRAP-030 на квартал", {"history": history}
    )
    assert parsed["intent"] == "unknown"
    assert parsed["reason"] == "missing_or_ambiguous_sku"
    assert not parsed["results"]


@pytest.mark.parametrize("provider", ["gigachat", "ollama"])
def test_models_cannot_erase_explicit_conflicts(history: dict, provider: str) -> None:
    """The clarification gate runs before either optional provider."""
    context = {"history": history}
    if provider == "ollama":
        context["ollama_model"] = "test"
    with (
        patch("spa_assistant.questions.gigachat_scores_with_status") as giga,
        patch("spa_assistant.questions.ollama_scores") as ollama,
    ):
        parsed, _, _ = answer_question(
            "Покажи бюджет на месяц и какие партии сгорят", context
        )
    assert parsed["reason"] == "ambiguous_intent"
    giga.assert_not_called()
    ollama.assert_not_called()


def test_model_margin_is_not_a_one_hot_label(history: dict) -> None:
    """Model uncertainty also causes abstention when rules have no evidence."""
    scores = dict.fromkeys(INTENTS, 0.0)
    scores.update(budget=0.8, reorder_list=0.75)
    with patch(
        "spa_assistant.questions.gigachat_scores_with_status",
        return_value=(scores, "ok"),
    ):
        parsed, _, _ = answer_question(
            "На квартал сколько денег нужно?", {"history": history}
        )
    assert parsed["reason"] == "ambiguous_intent"
    assert parsed["intent_margin"] == 0.05
    assert _valid_scores({"intent": "budget"}) is None


def test_followup_preserves_and_overrides_constraints(history: dict) -> None:
    """Absent, replaced and explicitly cancelled constraints differ."""
    first, _, _ = answer_question(
        "Посчитай закупку скраба на месяц с бюджетом 10000 рублей, "
        "если загрузка вырастет на 20%",
        {"history": history},
    )
    assert first["budget_limit"] == 10000
    assert first["demand_multiplier"] == 1.2
    context = {"history": history, "previous_params": first}
    following, _, _ = answer_question("А на 90 дней?", context)
    assert following["budget_limit"] == 10000
    assert following["demand_multiplier"] == 1.2
    assert following["budget_gap"] > 0
    replaced, _, _ = answer_question("А на 90 дней при лимите 20000?", context)
    assert replaced["budget_limit"] == 20000
    cancelled, _, _ = answer_question(
        "А на 90 дней без лимита и без изменения загрузки?", context
    )
    assert cancelled["budget_limit"] is None
    assert cancelled["demand_multiplier"] == 1
    assert first["budget_limit"] == 10000


def test_order_date_uses_dated_receipts(history: dict) -> None:
    """A known receipt postpones the threshold crossing and stockout together."""
    history["weekly_consumption"]["OIL-001"] = [7] * 12
    history["current_stock"]["OIL-001"] = 30
    history["incoming_qty"]["OIL-001"] = 100
    history["incoming_deliveries"] = {"OIL-001": [{"date": "2026-09-16", "qty": 100}]}
    result = forecast_demand(history, "OIL-001", 180, {})
    assert result["recommended_order_date"] == "2027-01-02"
    assert result["stockout_date"] == "2027-01-23"


def test_receipt_on_trigger_day_and_late_receipt() -> None:
    """A later receipt cannot retroactively remove an earlier order trigger."""
    args = (Decimal(30), Fraction(1), Fraction(21), date(2026, 9, 15))
    assert reorder_date(*args, [(date(2026, 9, 24), Decimal(10))]) == "2026-10-04"
    assert reorder_date(*args, [(date(2026, 9, 25), Decimal(10))]) == "2026-09-24"
    assert reorder_date(*args, [(date(2026, 9, 15), Decimal(10))]) == "2026-10-04"


def test_intermittent_demand_is_not_erased(history: dict) -> None:
    """Repeated positive consumption must produce positive demand and a warning."""
    series = [0, 0, 0, 7] * 3
    level = estimate_level(series)
    assert level["weekly_level"] == Decimal("1.75")
    assert level["outliers_adjusted"] == 0
    history["weekly_consumption"]["OIL-001"] = series
    result = forecast_demand(history, "OIL-001", 28, {})
    assert result["forecast_demand"] == 7
    assert result["requires_clarification"]
    assert any("Редкий спрос" in warning for warning in result["warnings"])
    assert estimate_level([0] * 12)["weekly_level"] == 0


@pytest.mark.parametrize("bad", [None, [], {}, {"as_of": 123}])
def test_invalid_history_has_consistent_error(bad: object) -> None:
    """Public boundaries expose ValueError rather than implementation exceptions."""
    with pytest.raises(ValueError):
        forecast_demand(bad, "OIL-001", 30, {})
    with pytest.raises(ValueError):
        answer_question("На месяц", {"history": bad})


@pytest.mark.parametrize("schedule", [None, {}, [None], [{"date": 123, "qty": 1}]])
def test_invalid_delivery_container(history: dict, schedule: object) -> None:
    """Nested malformed delivery input has the same public error contract."""
    history["incoming_deliveries"] = {"OIL-001": schedule}
    with pytest.raises(ValueError):
        forecast_demand(history, "OIL-001", 30, {})


def test_location_snapshot_controls_calendar_period(history: dict) -> None:
    """Changing the snapshot date also changes remaining calendar-month demand."""
    scoped = deepcopy(history)
    scoped["as_of"] = "2026-10-01"
    parsed, _, _ = answer_question(
        "Какой бюджет закупок в этом месяце в Сочи?",
        {"history": history, "location_history": {"Сочи": scoped}},
    )
    assert parsed["period"]["start"] == "2026-10-01"
    assert parsed["period"]["days"] == 31
    assert all(row["as_of"] == "2026-10-01" for row in parsed["results"])


def test_multistep_metrics_use_only_training_data() -> None:
    """A future jump must remain an error, with independent arithmetic oracles."""
    results = backtest({"weekly_consumption": {"OIL-001": [7] * 4 + [70] * 4}})
    for row in results:
        assert row["horizon_folds"] == 1
        assert row["mae_horizon_total"] == 252
        assert row["order_revision_pairs"] == 3
    naive = next(row for row in results if row["method"] == "last_observation")
    # Rounded four-week orders: 30, 280, 280, 280.
    assert naive["mean_absolute_order_revision"] == pytest.approx(250 / 3)


def test_multistep_missing_targets_are_not_zeroes() -> None:
    """Incomplete target windows cannot count as complete observations."""
    results = backtest({"weekly_consumption": {"OIL-001": [7] * 4 + [7, None, 7, 7]}})
    assert all(row["horizon_folds"] == 0 for row in results)
    assert all(row["mae_horizon_total"] is None for row in results)


def test_offline_mode_skips_both_models(history: dict) -> None:
    """Reports and benchmarks stay offline even when a model is configured."""
    with (
        patch("spa_assistant.questions.gigachat_scores_with_status") as giga,
        patch("spa_assistant.questions.ollama_scores") as ollama,
    ):
        answer_question(
            "Какой бюджет закупок на месяц?",
            {"history": history, "llm_enabled": False, "ollama_model": "test"},
        )
    giga.assert_not_called()
    ollama.assert_not_called()


@pytest.mark.parametrize(
    "prior", [{"period": []}, {"sku": []}, {"period": {"kind": "rolling_days"}}]
)
def test_invalid_previous_context(history: dict, prior: dict) -> None:
    """Stored dialogue state obeys the same explicit input error contract."""
    with pytest.raises(ValueError):
        answer_question(
            "Сколько это стоит?", {"history": history, "previous_params": prior}
        )
