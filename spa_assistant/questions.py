"""Russian intent/entity extraction and answer rendering from trusted calculations."""

import calendar
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from .catalog import canonical_text, locations, sku_candidates
from .config import (
    DAYS_PER_WEEK,
    HALF_YEAR_DAYS,
    MAX_TEXT_LENGTH,
    MONTH_DAYS,
    QUARTER_DAYS,
    QUERY,
    YEAR_DAYS,
)
from .forecast import aggregate_forecasts, forecast_demand, purchase_plan
from .llm import INTENTS, gigachat_scores_with_status, ollama_scores
from .models import (
    History,
    ParsedQuery,
    Period,
    validate_history,
    validate_previous_query,
)

WORD_NUMBERS = {
    "один": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "шесть": 6,
    "двенадцать": 12,
}
PERIOD_RE = re.compile(
    r"\b(\d+|" + "|".join(WORD_NUMBERS) + r")\s+(дн\w*|день|недел\w*|месяц\w*)"
)
BUDGET_RE = re.compile(
    r"\b(?:лимит\w*|бюджет\w*)\s+(?:до\s+)?(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)"
    r"\s*(тыс\w*|млн|миллион\w*)?"
)
PURCHASE_PATTERN = (
    r"\b(?:закуп\w*|заказ\w*|докуп\w*|покуп\w*|приобр\w*|"
    r"куп(?:ить|и|ите|им|лю|ят))\b"
)


def rule_scores(text: str) -> dict[str, float]:
    """Rank independent intent evidence, retaining genuine multi-intent ties."""
    scores = dict.fromkeys(INTENTS, 0.0)
    patterns = {
        "expiry_risk": r"сгор|срок\w*\s+годност|истека|просроч",
        "price_dynamics": r"подорож|динамик\w*\s+цен|рост\w*\s+цен|измен\w*\s+цен",
        "deficit_risk": r"дефицит|закончит|хватит.*постав",
        "budget": r"бюджет|план\w*\s+закуп\w*\s+вырос",
        "reorder_list": r"что\s+нужно\s+заказ|что\s+заказ|список\w*\s+закуп",
    }
    for intent, pattern in patterns.items():
        if re.search(pattern, text):
            scores[intent] = QUERY.explicit_score
    if re.search(PURCHASE_PATTERN, text) or re.search(
        r"уйдет|потребност|расход|прогноз", text
    ):
        scores["forecast_purchase"] = QUERY.generic_score
    if re.search(
        rf"сколько.*(?:{PURCHASE_PATTERN}|уйдет)|посчитай\s+{PURCHASE_PATTERN}", text
    ):
        scores["forecast_purchase"] = QUERY.explicit_score
    # A budget constraint is an entity, not a second task.
    if BUDGET_RE.search(text) and scores["forecast_purchase"]:
        scores["budget"] = 0.0
    # Asking the price of a purchase is one composite business intent.
    if scores["deficit_risk"]:
        scores["forecast_purchase"] = 0.0
    if re.search(r"погод|анекдот|курс\s+валют", text):
        scores["unknown"] = 1.0
    return scores


def extract_period(text: str, as_of: date) -> tuple[Period | None, bool]:
    """Separate fixed planning horizons from calendar expiry periods."""
    horizons = set()
    for match in PERIOD_RE.finditer(text):
        count, unit = match.groups()
        number = int(count) if count.isdigit() else WORD_NUMBERS[count]
        scale = (
            DAYS_PER_WEEK
            if unit.startswith("недел")
            else MONTH_DAYS
            if unit.startswith("месяц")
            else 1
        )
        horizons.add(number * scale)
    for pattern, days in (
        (r"\bквартал\w*", QUARTER_DAYS),
        (r"\bполгод\w*", HALF_YEAR_DAYS),
        (r"\b(?:на|за)\s+год\b", YEAR_DAYS),
        (r"\b(?:на|за)\s+месяц\b", MONTH_DAYS),
    ):
        if re.search(pattern, text):
            horizons.add(days)
    if len(horizons) > 1:
        return None, True
    if horizons:
        return {
            "kind": "rolling_days",
            "days": horizons.pop(),
            "start": as_of.isoformat(),
        }, False
    if "этом месяце" in text:
        last = calendar.monthrange(as_of.year, as_of.month)[1]
        return {
            "kind": "calendar_month",
            "start": as_of.isoformat(),
            "end": as_of.replace(day=last).isoformat(),
            "days": last - as_of.day + 1,
        }, False
    if re.search(r"до\s+следующ\w*\s+постав", text):
        return {"kind": "until_next_delivery", "days": None}, False
    return None, False


def _clarify(
    parsed: ParsedQuery,
    confidence: float,
    reason: str,
    message: str,
) -> tuple[dict[str, Any], float, str]:
    """Preserve extracted entities while making abstention machine-readable."""
    parsed.update(intent="unknown", status="clarification_required", reason=reason)
    return (
        parsed,
        min(confidence, QUERY.clarification_confidence_cap),
        "Требуется уточнение. " + message,
    )


def _extract_budget(text: str) -> float | None:
    """Read an explicit ruble limit with optional thousand/million suffix."""
    budget_match = BUDGET_RE.search(text)
    budget = None
    if budget_match:
        number, scale = budget_match.groups()
        factor = (
            1_000_000
            if scale and scale.startswith(("млн", "миллион"))
            else 1_000
            if scale
            else 1
        )
        budget = float(
            Decimal(number.replace(" ", "").replace("\u00a0", "").replace(",", "."))
            * factor
        )
    return budget


def _route_question(
    question: str, text: str, context: dict[str, Any]
) -> tuple[dict[str, float], str, str]:
    """Preserve explicit conflicts before consulting optional model evidence."""
    scores = rule_scores(text)
    evidence = sorted(scores.values(), reverse=True)
    conflict = (
        evidence[0] >= QUERY.intent_threshold
        and evidence[0] - evidence[1] < QUERY.intent_margin
    )
    if context.get("llm_enabled", True) is False or conflict:
        return scores, "rules", "disabled" if not conflict else "rules_conflict"
    router = "rules"
    if context.get("ollama_model"):
        proposed = ollama_scores(question, context["ollama_model"])
        model_router = "ollama"
        llm_status = "ollama_ok" if proposed is not None else "ollama_unavailable"
        if proposed is not None and max(scores.values()) < QUERY.intent_threshold:
            scores = proposed
            router = model_router
    else:
        proposed, llm_status = gigachat_scores_with_status(question)
        if proposed is not None:
            scores = proposed
            router = "gigachat"
    return scores, router, llm_status


def _validate_request(
    parsed: ParsedQuery, text: str, confidence: float, entity_conflict: bool
) -> tuple[dict[str, Any], float, str] | None:
    """Check required entities and intent ambiguity before performing calculations."""
    intent = parsed["intent"]
    gap = parsed["intent_margin"]
    if (
        intent == "unknown"
        or confidence < QUERY.intent_threshold
        or gap < QUERY.intent_margin
    ):
        return _clarify(
            parsed,
            confidence,
            "ambiguous_intent",
            "Уточните складскую задачу: закупка, бюджет, дефицит, "
            "сроки годности или цены?",
        )
    if entity_conflict:
        return _clarify(
            parsed,
            confidence,
            "conflicting_entities",
            "Выберите один объект и один период.",
        )
    if re.search(r"по сравнению|прошл\w*\s+квартал", text):
        return _clarify(
            parsed,
            confidence,
            "comparison_data_required",
            "Укажите сравниваемые периоды и предоставьте прошлый план, цены и расход.",
        )
    if len(parsed["sku_candidates"]) > 1 or (
        intent in {"forecast_purchase", "price_dynamics"} and not parsed["sku"]
    ):
        return _clarify(
            parsed,
            confidence,
            "missing_or_ambiguous_sku",
            "Укажите один SKU для расчёта. "
            + (
                "Найдены: " + ", ".join(parsed["sku_candidates"]) + "."
                if parsed["sku_candidates"]
                else "Название товара не распознано."
            ),
        )
    if intent != "price_dynamics" and parsed["period"] is None:
        return _clarify(
            parsed, confidence, "missing_period", "На какой период выполнить расчёт?"
        )
    if intent == "price_dynamics":
        return _clarify(
            parsed,
            confidence,
            "price_history_required",
            "Укажите поставщика и два сравниваемых периода; нужны исторические цены.",
        )
    if re.search(r"загруз\w*.*(?:выраст|увелич|сниз|уменьш)", text):
        change = re.search(r"на\s+(\d+(?:[.,]\d+)?)\s*%", text)
        if change is None:
            return _clarify(
                parsed,
                confidence,
                "missing_load_change",
                "Укажите изменение загрузки в процентах.",
            )
        delta = Decimal(change[1].replace(",", ".")) / 100
        sign = -1 if re.search(r"сниз|уменьш", text) else 1
        parsed["demand_multiplier"] = float(1 + sign * delta)
    return None


def answer_question(
    question: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any], float, str]:
    """Return parsed parameters, joint quality score and a grounded Russian answer.

    context requires history; location_history may supply independent per-site
    histories. previous_params only participates in explicitly anaphoric follow-ups.
    Optional Ollama or configured GigaChat uses automatic rules fallback.
    """
    if not isinstance(question, str) or len(question) > MAX_TEXT_LENGTH:
        raise ValueError("Ожидается вопрос допустимой длины")
    if not isinstance(context, dict):
        raise ValueError("context: требуется словарь")
    history = context.get("history")
    as_of = validate_history(history)
    for field in ("location_history", "previous_params"):
        if context.get(field) is not None and not isinstance(context[field], dict):
            raise ValueError(f"context.{field}: требуется словарь")
    text = canonical_text(question)
    scores, router, llm_status = _route_question(question, text, context)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    intent, confidence = ranked[0]
    gap = confidence - ranked[1][1]
    candidates = sku_candidates(text)
    sites = locations(text)
    period, period_conflict = extract_period(text, as_of)
    budget = _extract_budget(text)
    parsed: ParsedQuery = {
        "intent": intent,
        "sku": candidates[0] if len(candidates) == 1 else None,
        "sku_candidates": candidates,
        "location": sites[0] if len(sites) == 1 else None,
        "period": period,
        "budget_limit": budget,
        "demand_multiplier": 1.0,
        "intent_scores": scores,
        "intent_margin": round(gap, 6),
        "router": router,
        "llm_status": llm_status,
        "status": "ok",
        "results": [],
    }
    prior = context.get("previous_params")
    if prior and re.search(r"\b(?:это|этого|тогда|теперь|а на)\b", text):
        validate_previous_query(prior)
        for field in ("sku", "location", "period", "budget_limit"):
            if parsed[field] is None and not (field == "sku" and candidates):
                parsed[field] = prior.get(field)
        if not re.search(r"загруз", text):
            parsed["demand_multiplier"] = prior.get("demand_multiplier", 1.0)
        if confidence < QUERY.intent_threshold:
            intent = prior.get("intent", "unknown")
            parsed["intent"] = intent
            confidence = QUERY.generic_score
            gap = QUERY.generic_score
    if re.search(r"без\s+(?:лимит\w*|ограничени\w*|бюджет\w*)", text):
        parsed["budget_limit"] = None
    if re.search(r"без\s+(?:роста|изменения)\s+загруз", text):
        parsed["demand_multiplier"] = 1.0
    parsed["intent_margin"] = round(gap, 6)
    clarification = _validate_request(
        parsed, text, confidence, len(sites) > 1 or period_conflict
    )
    if clarification is not None:
        return clarification
    if parsed["location"]:
        scoped = (context.get("location_history") or {}).get(parsed["location"])
        if scoped is None:
            parsed.update(status="data_unavailable", reason="missing_location_history")
            return (
                parsed,
                0.0,
                "Нет остатков и расхода по указанному объекту. "
                "Общий склад нельзя выдать за филиал.",
            )
        history = scoped
        as_of = validate_history(history)
        # Calendar periods and risk filtering must use the selected snapshot.
        if period is not None:
            parsed["period"], _ = extract_period(text, as_of)
    if intent == "expiry_risk":
        parsed.update(status="data_unavailable", reason="missing_expiry_data")
        return (
            parsed,
            0.0,
            "В тестовом датасете нет сроков годности и остатков партий. "
            "Нужен реестр партий для расчёта риска списания.",
        )
    if parsed["period"]["kind"] == "until_next_delivery":
        parsed.update(status="data_unavailable", reason="missing_delivery_dates")
        return (
            parsed,
            0.0,
            "Укажите даты ближайших поставок: срок поставки в справочнике "
            "не является датой открытого заказа.",
        )
    return _execute_query(parsed, history, as_of, confidence)


def _execute_query(
    parsed: ParsedQuery, history: History, as_of: date, confidence: float
) -> tuple[dict[str, Any], float, str]:
    """Calculate and filter using the selected snapshot only."""
    intent = parsed["intent"]
    budget = parsed["budget_limit"]
    days = parsed["period"]["days"]
    params = {"demand_multiplier": parsed["demand_multiplier"], "budget_limit": budget}
    try:
        if parsed["sku"]:
            rows = [forecast_demand(history, parsed["sku"], days, params)]
            missing = []
        else:
            plan = purchase_plan(history, days, params)
            rows, missing = plan["items"], plan["missing_skus"]
        if intent == "deficit_risk":
            rows = [
                row
                for row in rows
                if row["stockout_date"] is not None
                and date.fromisoformat(row["stockout_date"])
                < as_of + timedelta(days=days)
            ]
        elif intent == "reorder_list":
            rows = [
                row
                for row in rows
                if row["recommended_order_date"] is not None
                and row["recommended_qty"] > 0
                and date.fromisoformat(row["recommended_order_date"])
                < as_of + timedelta(days=days)
            ]
    except (ValueError, OverflowError) as error:
        parsed.update(status="data_unavailable", reason="invalid_or_missing_data")
        return parsed, 0.0, f"Расчёт невозможен: {error}."
    parsed["results"] = rows
    parsed["missing_skus"] = missing
    # Aggregation is part of the deterministic calculation layer, never an LLM output.
    parsed.update(aggregate_forecasts(rows, budget))
    return _render_answer(parsed, confidence)


def _render_answer(
    parsed: ParsedQuery, confidence: float
) -> tuple[dict[str, Any], float, str]:
    """Render facts exclusively from the deterministic calculation layer."""
    rows = parsed["results"]
    missing = parsed["missing_skus"]
    budget = parsed["budget_limit"]
    total = parsed["estimated_cost"]
    quality = min([confidence, *(row["confidence"] for row in rows)])
    if any(row["requires_clarification"] for row in rows):
        parsed["status"] = "provisional"
    lines = [f"ИТОГО\nСуммарная стоимость рассчитанных позиций: {total:.2f} руб."]
    if not rows:
        lines.append(
            "По выбранному условию позиции не найдены; "
            "это не проверка отсутствующих данных."
        )
    if budget is not None:
        lines.append(
            f"Лимит {budget:.2f} руб.; превышение {parsed['budget_gap']:.2f} руб."
        )
    if missing:
        lines.append("Неполный охват: нет истории для " + ", ".join(missing) + ".")
    lines.extend(row["explanation"] for row in rows)
    return parsed, quality, "\n\n".join(lines)
