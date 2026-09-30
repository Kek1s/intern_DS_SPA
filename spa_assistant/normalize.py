"""Conservative extraction of inventory movements; unknown fields stay None."""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from .catalog import SKU_RE, canonical_text, get_catalog, locations, sku_candidates
from .config import CENTURY, MAX_TEXT_LENGTH

MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ),
        start=1,
    )
}
DATE_RE = re.compile(
    r"\b(?:(\d{4})-(\d{2})-(\d{2})|(\d{1,2})[./](\d{1,2})[./]"
    r"(\d{4}|\d{2})|(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4}))\b"
)
NUMBER = r"[+-]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
UNIT = r"(?:миллилитр\w*|литр\w*|килограмм\w*|грамм\w*|мл|кг|л|г|пар\w*|шт\w*)"
QUANTITY_RE = re.compile(rf"(?<![\w\d.,-])({NUMBER})\s*({UNIT})(?!\w)")
PACK_RE = re.compile(
    rf"(?<![\w\d.,-])({NUMBER})\s*(?:канистр\w*|уп\.?|упаков\w*|бутыл\w*)"
    rf"\s+по\s+({NUMBER})\s*({UNIT})(?!\w)"
)
BATCH_RE = re.compile(r"\bB-[A-Z]+-\d{3}-\d+\b", re.I)
DOC_RE = re.compile(r"\b(?:НК|NK|УПД)[ -]\d+\b", re.I)
OPERATIONS = {
    "receipt": r"\b(?:приход|поступлен)\w*",
    "consume": r"\b(?:расход|израсход|потреблен)\w*",
    "writeoff": r"\bсписан\w*",
    "return": r"\bвозврат\w*",
    "correction": r"\bкорректиров\w*",
}


def extract_date(text: str) -> str | None:
    """Parse DD/MM/YY as day-first, YY as 20YY; invalid/conflicting dates abstain."""
    values: set[str] = set()
    for match in DATE_RE.finditer(canonical_text(text)):
        a, b, c, d, e, f, g, h, i = match.groups()
        if a:
            year, month, day = int(a), int(b), int(c)
        elif d:
            year, month, day = int(f), int(e), int(d)
            year += CENTURY if len(f) == 2 else 0
        else:
            year, month, day = int(i), MONTHS[h], int(g)
        try:
            values.add(date(year, month, day).isoformat())
        except ValueError:
            return None
    return next(iter(values)) if len(values) == 1 else None


def unit_info(unit: str) -> tuple[str, Decimal]:
    """Return base dimension and scale; mass/volume are never interconverted."""
    if unit.startswith(("мл", "миллилитр")):
        return "л", Decimal("0.001")
    if unit.startswith(("л", "литр")):
        return "л", Decimal(1)
    if unit.startswith(("кг", "килограмм")):
        return "кг", Decimal(1)
    if unit.startswith(("г", "грамм")):
        return "кг", Decimal("0.001")
    return ("пар" if unit.startswith("пар") else "шт"), Decimal(1)


def extract_quantity(text: str, base_unit: str | None) -> float | None:
    """Extract one unambiguous quantity, respecting explicit pack contents."""
    if base_unit is None:
        return None
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    packs = list(PACK_RE.finditer(text))
    values: list[Decimal] = []
    for pack in packs:
        count, amount, unit = pack.groups()
        count_value = Decimal(count.replace(" ", "").replace(",", "."))
        amount_value = Decimal(amount.replace(" ", "").replace(",", "."))
        if count_value < 0 or amount_value <= 0 or count_value % 1:
            return None
        dimension, factor = unit_info(unit)
        if dimension != base_unit:
            return None
        values.append(count_value * amount_value * factor)
    for match in QUANTITY_RE.finditer(text):
        if any(pack.start() <= match.start() < pack.end() for pack in packs):
            continue
        amount, unit = match.groups()
        # Never accept the tail of a malformed grouped number.
        if re.search(r"\d[ ]+$", text[: match.start()]):
            return None
        dimension, factor = unit_info(unit)
        if dimension != base_unit:
            return None
        values.append(Decimal(amount.replace(" ", "").replace(",", ".")) * factor)
    return float(values[0]) if len(values) == 1 else None


def normalize_movement(text: str) -> dict[str, Any]:
    """Normalize a record; quantities are unsigned except for corrections.

    Return direction is intentionally not guessed. This parser does not post
    transactions to stock or replay the unrelated March examples into September.
    """
    if not isinstance(text, str) or len(text) > MAX_TEXT_LENGTH:
        raise ValueError("Ожидается строка допустимой длины")
    value = canonical_text(text)
    batches = set(BATCH_RE.findall(text.upper()))
    documents = set(DOC_RE.findall(text.upper()))
    # Batch identifiers are not independent SKU assertions.
    candidates = sku_candidates(BATCH_RE.sub("", value))
    sku = candidates[0] if len(candidates) == 1 else None
    product = get_catalog().get(sku)
    if product is None:
        sku = None
    sites = locations(value)
    operations = [op for op, pattern in OPERATIONS.items() if re.search(pattern, value)]
    operation = operations[0] if len(operations) == 1 else None
    unit = product.unit if product else None
    # The year suffix "2026 г." is a date, not grams.
    quantity_text = DATE_RE.sub("", value)
    for pattern in (BATCH_RE, DOC_RE, SKU_RE):
        quantity_text = pattern.sub(" ", quantity_text)
    quantity_text = re.sub(r"\bг\.(?=\s|:|$)", "", quantity_text)
    qty = extract_quantity(quantity_text, unit)
    if qty is not None and qty < 0 and operation != "correction":
        qty = None
    return {
        "date": extract_date(value),
        "sku": sku,
        "location": sites[0] if len(sites) == 1 else None,
        "operation": operation,
        "qty": qty,
        "unit": unit,
        "batch": next(iter(batches)) if len(batches) == 1 else None,
        "doc_no": next(iter(documents)) if len(documents) == 1 else None,
    }
