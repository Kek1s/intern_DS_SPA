"""Deterministic demand, procurement and stockout calculations; no LLM calls."""

from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from statistics import fmean, median, pstdev
from typing import Any

from .catalog import Product, decimal_number, get_catalog
from .config import DAYS_PER_WEEK, FORECAST, ForecastConfig
from .models import ForecastResult, History, validate_history

ZERO = Decimal(0)
MONEY_STEP = Decimal("0.01")


def round_order(need: Decimal | Fraction, pack: Decimal, minimum: Decimal) -> Decimal:
    """MOQ is a lower bound, pack size is a multiple; no order for zero need."""
    if pack <= 0 or minimum < 0:
        raise ValueError("Недопустимые параметры упаковки")
    if need <= 0:
        return ZERO
    packs = max(Fraction(need), Fraction(minimum)) / Fraction(pack)
    return Decimal(-(-packs.numerator // packs.denominator)) * pack


def _decimal(value: Fraction) -> Decimal:
    """Convert exact ratios for display/JSON, after discrete decisions are made."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def estimate_level(
    weekly: list[float | None],
    config: ForecastConfig = FORECAST,
) -> dict[str, Any]:
    """Use the last calendar window, never compact gaps into a shorter timeline.

    Sustained shifts remain intact. Isolated extremes in the recent window are
    winsorized with a median/MAD bound. No interpolation or zero imputation.
    """
    if not weekly or not isinstance(weekly, list):
        raise ValueError("Нужна непустая недельная история")
    clean = [
        None if x is None else float(decimal_number(x, "weekly_consumption"))
        for x in weekly
    ]
    recent = [x for x in clean[-config.window_weeks :] if x is not None]
    valid = [x for x in clean if x is not None]
    if not recent:
        raise ValueError("В последнем окне нет наблюдений; требуется уточнение")
    previous = [x for x in clean[: -config.window_weeks] if x is not None]
    center = median(recent)
    mad = median(abs(x - center) for x in recent)
    bound = max(
        config.outlier_mad_factor * config.mad_scale * mad,
        abs(center) * config.outlier_relative_floor,
    )
    clipped = [min(max(x, max(0.0, center - bound)), center + bound) for x in recent]
    adjusted = sum(x != y for x, y in zip(recent, clipped, strict=True))
    recent_mean = fmean(recent)
    cv_recent = pstdev(recent) / recent_mean if recent_mean else 0.0
    shift = bool(
        len(recent) >= config.min_regime_weeks
        and len(previous) >= config.min_regime_weeks
        and fmean(previous) > 0
        and recent_mean / fmean(previous) >= config.regime_ratio
        and min(recent) > max(previous)
        and cv_recent <= config.regime_max_cv
    )
    # Regime change is supported by every recent observation, not a single spike.
    intermittent = center == 0 and any(x > 0 for x in recent)
    chosen = recent if shift or intermittent else clipped
    weekly_sum = sum((Decimal(str(x)) for x in chosen), ZERO)
    level = weekly_sum / len(chosen)
    mean_all = fmean(valid)
    cv = pstdev(valid) / mean_all if mean_all else 0.0
    coverage = len(valid) / len(clean)
    recent_coverage = len(recent) / min(config.window_weeks, len(clean))
    length_factor = min(1.0, len(valid) / config.reference_weeks)
    confidence = length_factor * coverage * recent_coverage / (1.0 + cv)
    if shift:
        confidence *= config.regime_penalty
    return {
        "weekly_level": level,
        "recent_adjusted_sum": weekly_sum,
        "confidence": confidence,
        "weeks_total": len(clean),
        "weeks_observed": len(valid),
        "recent_observed": len(recent),
        "missing_weeks": len(clean) - len(valid),
        "coefficient_of_variation": cv,
        "level_shift": shift,
        "outliers_adjusted": 0 if shift or intermittent else adjusted,
        "intermittent_demand": intermittent,
        "recent_weekly_values": recent,
    }


def _deliveries(
    history: dict[str, Any],
    sku: str,
    as_of: date,
    horizon: int,
    credit_undated: bool,
) -> tuple[Decimal, Decimal, list[tuple[date, Decimal]], bool]:
    """Return total, credited quantity, dated events, unknown-date flag.

    When a schedule is supplied it must reconcile with the aggregate. Late
    deliveries do not silently reduce the requirement for the selected horizon.
    """
    total = decimal_number(history["incoming_qty"][sku], "incoming_qty")
    schedule = history.get("incoming_deliveries", {}).get(sku)
    if schedule is None:
        return total, total if credit_undated else ZERO, [], total > 0
    events: list[tuple[date, Decimal]] = []
    for row in schedule:
        arrival = date.fromisoformat(row["date"])
        qty = decimal_number(row["qty"], "delivery.qty")
        if arrival < as_of:
            raise ValueError("Просроченная поставка: нужна актуальная дата")
        events.append((arrival, qty))
    if sum((qty for _, qty in events), ZERO) != total:
        raise ValueError("График поставок не сходится с incoming_qty")
    end = as_of + timedelta(days=horizon)
    credited = sum((qty for arrival, qty in events if arrival < end), ZERO)
    return total, credited, sorted(events), False


def depletion_date(
    stock: Decimal,
    daily: Decimal | Fraction,
    as_of: date,
    deliveries: list[tuple[date, Decimal]],
) -> str | None:
    """First day with unmet demand; receipts arrive before that day's demand.

    Exact depletion at midnight allows a receipt on the following day. Delivery
    events after the first shortage cannot retroactively eliminate that shortage.
    """
    if daily == 0:
        return None
    cursor = as_of
    balance = Fraction(stock)
    daily = Fraction(daily)
    for arrival, qty in deliveries:
        days = (arrival - cursor).days
        if daily * days > balance:
            break
        balance += Fraction(qty) - daily * days
        cursor = arrival
    days_covered = int(balance // daily)
    try:
        return (cursor + timedelta(days=days_covered)).isoformat()
    except (OverflowError, ValueError):
        return None


def reorder_date(
    stock: Decimal,
    daily: Fraction,
    threshold: Fraction,
    as_of: date,
    deliveries: list[tuple[date, Decimal]],
) -> str | None:
    """Conservative day of threshold crossing; receipts precede consumption.

    A receipt on the trigger day can postpone ordering, but a later receipt
    cannot erase an earlier trigger. Combine same-day receipts before testing.
    """
    if not daily:
        return None
    balance = Fraction(stock)
    cursor = as_of
    events: dict[date, Decimal] = {}
    for arrival, qty in deliveries:
        events[arrival] = events.get(arrival, ZERO) + qty
    for arrival, qty in sorted(events.items()):
        offset = max(0, (balance - threshold) // daily)
        elapsed = (arrival - cursor).days
        if offset < elapsed:
            return (cursor + timedelta(days=offset)).isoformat()
        balance += Fraction(qty) - daily * elapsed
        cursor = arrival
    offset = max(0, (balance - threshold) // daily)
    if offset > (date.max - cursor).days:
        return None
    return (cursor + timedelta(days=offset)).isoformat()


def forecast_demand(
    history: History,
    sku: str,
    horizon_days: int,
    params: dict[str, Any],
) -> ForecastResult:
    """Forecast using only structured history and catalog values.

    params: product (optional catalog row), demand_multiplier (default 1),
    budget_limit (optional), credit_undated_incoming (default True).
    Unknown incoming dates are explicitly a volume-planning assumption and are
    never used to postpone the stockout date. Horizon is [as_of, as_of + days).
    """
    validate_history(history)
    if not isinstance(params, dict):
        raise ValueError("params: требуется словарь")
    if not isinstance(sku, str):
        raise ValueError("sku: требуется строка")
    if (
        type(horizon_days) is not int
        or not 0 < horizon_days <= FORECAST.max_horizon_days
    ):
        raise ValueError("horizon_days вне допустимого диапазона")
    allowed = {
        "product",
        "demand_multiplier",
        "budget_limit",
        "credit_undated_incoming",
    }
    if set(params) - allowed:
        raise ValueError(f"Неизвестные параметры: {sorted(set(params) - allowed)}")
    product = (
        Product.from_dict(params["product"])
        if "product" in params
        else get_catalog().get(sku)
    )
    if product is None or product.sku != sku:
        raise ValueError(f"SKU отсутствует в справочнике: {sku}")
    try:
        weekly = history["weekly_consumption"][sku]
        stock = decimal_number(history["current_stock"][sku], "current_stock")
        as_of = date.fromisoformat(history["as_of"])
        if sku not in history["incoming_qty"]:
            raise KeyError(sku)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Недостаточно структурированных данных для {sku}") from error
    level = estimate_level(weekly)
    multiplier = decimal_number(params.get("demand_multiplier", 1), "demand_multiplier")
    daily_exact = (
        Fraction(level["recent_adjusted_sum"])
        / level["recent_observed"]
        / DAYS_PER_WEEK
        * Fraction(multiplier)
    )
    daily = _decimal(daily_exact)
    credit = params.get("credit_undated_incoming", True)
    if type(credit) is not bool:
        raise ValueError("credit_undated_incoming должен быть bool")
    incoming, eligible, deliveries, unknown_eta = _deliveries(
        history,
        sku,
        as_of,
        horizon_days,
        credit,
    )
    demand = _decimal(daily_exact * horizon_days)
    safety = _decimal(daily_exact * product.safety_stock_days)
    reorder_exact = daily_exact * (product.lead_time_days + product.safety_stock_days)
    reorder = _decimal(reorder_exact)
    net_exact = max(
        Fraction(0),
        daily_exact * (horizon_days + product.safety_stock_days)
        - Fraction(stock)
        - Fraction(eligible),
    )
    net = _decimal(net_exact)
    recommended = round_order(net_exact, product.pack_size, product.min_order_qty)
    cost = (recommended * product.price).quantize(MONEY_STEP, rounding=ROUND_HALF_UP)
    stockout = depletion_date(stock, daily_exact, as_of, deliveries)
    nominal_arrival = as_of + timedelta(days=product.lead_time_days)
    deficit = stockout is not None and date.fromisoformat(stockout) < nominal_arrival
    horizon_factor = 1 + FORECAST.horizon_penalty_rate * max(
        0,
        horizon_days / (len(weekly) * DAYS_PER_WEEK) - 1,
    )
    confidence = level["confidence"] / horizon_factor
    warnings: list[str] = []
    if unknown_eta:
        confidence *= FORECAST.unknown_eta_penalty
        warnings.append(
            "Поставка без даты: объём в пути "
            + ("условно учтён на период." if credit else "не учтён в закупке.")
            + " Дата дефицита его не учитывает."
        )
    if level["missing_weeks"]:
        warnings.append("Пропуски исключены из среднего.")
    if level["level_shift"]:
        warnings.append("Рост последних недель принят за новый уровень спроса.")
    if level["outliers_adjusted"]:
        warnings.append("Одиночные выбросы ограничены медианой/MAD.")
    if level["intermittent_demand"]:
        warnings.append(
            "Редкий спрос: использовано среднее окна; нужен контроль прогноза."
        )
    if deficit:
        warnings.append("Дефицит возможен до поставки: нужна срочная закупка.")
    if multiplier != 1:
        warnings.append("Спрос изменён пропорционально загрузке (допущение).")
    budget = (
        decimal_number(params["budget_limit"], "budget_limit")
        if params.get("budget_limit") is not None
        else None
    )
    gap = max(ZERO, cost - budget) if budget is not None else None
    if gap:
        warnings.append("Стоимость выше лимита; объём не сокращён.")
    needs_clarification = (
        confidence < FORECAST.confidence_threshold or level["intermittent_demand"]
    )
    if needs_clarification:
        warnings.append("Требуется уточнение: низкая надёжность прогноза.")
    order_date = reorder_date(stock, daily_exact, reorder_exact, as_of, deliveries)
    order_hint = f"заказать до {order_date}" if order_date else "заказ не требуется"
    stockout_hint = f"дефицит с {stockout}" if stockout else "дефицит не ожидается"
    explanation = (
        f"{sku} — {product.name} | на {as_of} | {horizon_days} дн.\n"
        f"Итог: {recommended} {product.unit} × {product.price:.2f} = "
        f"{cost:.2f} руб.; {order_hint}; {stockout_hint}.\n"
        "Расчёт (значения округлены для показа):\n"
        f"• Расход: {level['weekly_level']:.6f} / {DAYS_PER_WEEK} × "
        f"{multiplier} ≈ {daily:.6f} {product.unit}/день "
        f"(окно до {FORECAST.window_weeks} нед.).\n"
        f"• Потребность: {daily:.6f} × {horizon_days} ≈ "
        f"{demand:.6f} {product.unit}; запас: {daily:.6f} × "
        f"{product.safety_stock_days} ≈ {safety:.6f} {product.unit}.\n"
        f"• Точка заказа: {daily:.6f} × ({product.lead_time_days} + "
        f"{product.safety_stock_days}) ≈ {reorder:.6f} {product.unit}.\n"
        f"• К закупке: max(0, {demand:.6f} + {safety:.6f} − {stock} − "
        f"{eligible}) ≈ {net:.6f} {product.unit} → {recommended} "
        f"{product.unit} (уп. {product.pack_size}; минимум {product.min_order_qty}).\n"
        f"Данные: остаток {stock}; в пути {incoming}, учтено {eligible} "
        f"{product.unit}."
    )
    return {
        "sku": sku,
        "unit": product.unit,
        "as_of": as_of.isoformat(),
        "horizon_days": horizon_days,
        "avg_daily_consumption": float(daily),
        "forecast_demand": float(demand),
        "current_stock": float(stock),
        "incoming_qty": float(incoming),
        "eligible_incoming_qty": float(eligible),
        "safety_stock": float(safety),
        "reorder_point": float(reorder),
        "recommended_qty": float(recommended),
        "estimated_cost": float(cost),
        "stockout_date": stockout,
        "stockout_basis": "dated_deliveries_only",
        "recommended_order_date": order_date,
        "confidence": round(confidence, 6),
        "requires_clarification": needs_clarification,
        "deficit_before_lead_time": deficit,
        "low_stock": Fraction(stock) <= reorder_exact,
        "excess_stock": Fraction(stock) > daily_exact * FORECAST.excess_cover_days
        if daily_exact
        else stock > 0,
        "no_consumption": daily == 0,
        "budget_limit": float(budget) if budget is not None else None,
        "budget_gap": float(gap) if gap is not None else None,
        "diagnostics": {
            key: float(value) if isinstance(value, Decimal) else value
            for key, value in level.items()
        },
        "assumptions": {
            "credit_undated_incoming": credit,
            "demand_multiplier": float(multiplier),
            "price_constant": True,
        },
        "warnings": warnings,
        "explanation": explanation
        + ("\nПримечания: " + " ".join(warnings) if warnings else ""),
    }


def purchase_plan(
    history: dict[str, Any],
    horizon_days: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate only calculated results; expose SKUs missing from the dataset."""
    params = params or {}
    validate_history(history)
    rows = [
        forecast_demand(history, sku, horizon_days, params)
        for sku in sorted(history["weekly_consumption"])
    ]
    return {
        "items": rows,
        **aggregate_forecasts(rows, params.get("budget_limit")),
        "confidence": min((row["confidence"] for row in rows), default=0.0),
        "missing_skus": sorted(set(get_catalog()) - set(history["weekly_consumption"])),
        "explanation": (
            "Сумма рассчитанных закупок; без истории отсутствующие позиции не оценены."
        ),
    }


def aggregate_forecasts(
    rows: list[dict[str, Any]],
    budget_limit: float | None = None,
) -> dict[str, float | None]:
    """Sum purchase costs centrally; responses only render these calculated totals."""
    total = sum((Decimal(str(row["estimated_cost"])) for row in rows), ZERO)
    budget = (
        decimal_number(budget_limit, "budget_limit")
        if budget_limit is not None
        else None
    )
    return {
        "estimated_cost": float(total),
        "budget_limit": float(budget) if budget is not None else None,
        "budget_gap": float(max(ZERO, total - budget)) if budget is not None else None,
    }
