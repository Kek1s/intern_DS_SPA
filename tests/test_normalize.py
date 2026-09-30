"""Independent field expectations and ambiguous/invalid movement cases."""

import pytest

from spa_assistant import normalize_movement
from spa_assistant.normalize import extract_date

EXPECTED = [
    ("2026-03-05", "OIL-001", "MS-01", "receipt", 10, "л", None, "НК-345"),
    ("2026-03-01", "OIL-001", "MS-01", "consume", 0.45, "л", "B-OIL-001-012", None),
    ("2026-06-03", "SCRB-020", "Сочи", "writeoff", 1.2, "кг", None, None),
    ("2026-03-07", "WRAP-030", "MS-02", "consume", 3.5, "кг", "B-WRAP-030-004", None),
    ("2026-03-12", "OIL-002", None, "return", 2, "л", None, None),
    ("2026-03-08", "CONS-051", "MS-01", "consume", 48, "пар", None, None),
    ("2026-03-15", "CONS-052", "MS-02", "correction", -120, "шт", None, None),
    (None, "CONS-051", "Красная Поляна", "receipt", 200, "пар", None, None),
]
FIELDS = ("date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no")


@pytest.mark.parametrize("index", range(8))
def test_supplied_movements(dataset: dict, index: int) -> None:
    """All 64 field outcomes, including absent values, must match the gold table."""
    assert normalize_movement(dataset["movements"][index]["text"]) == dict(
        zip(FIELDS, EXPECTED[index], strict=True)
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("05.03.2026", "2026-03-05"),
        ("1 марта 2026 г.", "2026-03-01"),
        ("03/06/26", "2026-06-03"),
        ("2026-03-08", "2026-03-08"),
        ("29.02.2024", "2024-02-29"),
        ("29.02.2026", None),
        ("31.04.2026", None),
        ("1 января 2026 и 2 января 2026", None),
        ("без даты", None),
    ],
)
def test_dates(value: str, expected: str | None) -> None:
    """Validate calendar dates, locale and ambiguity."""
    assert extract_date(value) == expected


@pytest.mark.parametrize(
    ("text", "qty"),
    [
        ("расход oil 001 450мл", 0.45),
        ("SCRB-020 расход 3500 г", 3.5),
        ("OIL-001 приход 2 канистры по 5 л", 10),
        ("CONS-051 приход 4 уп. по 50 пар", 200),
        ("OIL-001 расход 3 кг", None),
        ("OIL-001 расход 2 л и 3 л", None),
        ("CONS-052 корректировка −120 шт", -120),
        ("CONS-052 расход -120 шт", None),
        ("расход масла 2 л", None),
    ],
)
def test_units(text: str, qty: float | None) -> None:
    """Respect dimensions, signed corrections, pack multiplication and conflicts."""
    assert normalize_movement(text)["qty"] == qty


def test_conflicts_and_unknown_identifiers() -> None:
    """Do not guess identities or an operation when multiple are present."""
    result = normalize_movement("приход расход OIL-001 OIL-002 MS-01 MS-02 2 л")
    assert result["sku"] is result["location"] is result["operation"] is None
    assert normalize_movement("приход FAKE-999 2 л")["sku"] is None
    assert all(value is None for value in normalize_movement("").values())


def test_document_and_batch_do_not_create_quantity() -> None:
    """Numbers inside identifiers are not stock movements."""
    result = normalize_movement("OIL-001 B-OIL-001-012 НК-345")
    assert result["sku"] == "OIL-001"
    assert result["qty"] is None


def test_oversized_input() -> None:
    """Bound work on untrusted text."""
    with pytest.raises(ValueError):
        normalize_movement("a" * 10_001)
