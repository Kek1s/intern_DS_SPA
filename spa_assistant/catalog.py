"""Validated, immutable catalog loaded once; no repeated disk I/O per request."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def decimal_number(value: Any, name: str, *, negative: bool = False) -> Decimal:
    """Reject booleans, NaN, infinities and negative quantities by default."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name}: требуется конечное число")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name}: требуется число") from error
    if not number.is_finite() or (number < 0 and not negative):
        raise ValueError(f"{name}: недопустимое значение")
    return number


@dataclass(frozen=True, slots=True)
class Product:
    """Supplier quantities and prices refer to the SKU's base unit."""

    sku: str
    name: str
    unit: str
    pack_size: Decimal
    min_order_qty: Decimal
    safety_stock_days: int
    lead_time_days: int
    price: Decimal

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Product":
        """Validate a catalog row at its input boundary."""
        required = {
            "sku",
            "name",
            "unit",
            "pack_size",
            "min_order_qty",
            "safety_stock_days",
            "lead_time_days",
            "price",
        }
        if not isinstance(row, Mapping) or not required <= row.keys():
            raise ValueError("Неполная строка справочника")
        if not all(isinstance(row[key], str) for key in ("sku", "name", "unit")):
            raise ValueError("SKU, название и единица должны быть строками")
        for field in ("safety_stock_days", "lead_time_days"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field}: требуется неотрицательное целое")
        if row["unit"] not in {"л", "кг", "шт", "пар"}:
            raise ValueError("Неизвестная базовая единица")
        if not re.fullmatch(r"[A-Z]+-\d{3}", row["sku"]):
            raise ValueError("Неверный SKU")
        values = {
            key: decimal_number(row[key], key)
            for key in ("pack_size", "min_order_qty", "price")
        }
        if values["pack_size"] <= 0:
            raise ValueError("pack_size должен быть положительным")
        return cls(
            sku=row["sku"],
            name=row["name"],
            unit=row["unit"],
            safety_stock_days=row["safety_stock_days"],
            lead_time_days=row["lead_time_days"],
            **values,
        )


@lru_cache(maxsize=1)
def get_catalog() -> Mapping[str, Product]:
    """Read the assignment catalog relative to the module, independent of cwd."""
    path = ROOT / "catalog.json"
    if not path.exists():
        path = Path(__file__).with_name("catalog.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    products = {row["sku"]: Product.from_dict(row) for row in rows}
    if len(products) != len(rows):
        raise ValueError("Дублирующиеся SKU")
    return MappingProxyType(products)


def canonical_text(text: str) -> str:
    """Normalize Russian case and typographic minus without losing semantics."""
    return text.casefold().replace("ё", "е").replace("−", "-")


SKU_RE = re.compile(r"(?<![\w-])([a-z]+)[ -]?(\d{3})(?![\w-])", re.I)
ALIASES = {
    "OIL-001": (r"миндал", r"базов\w*\s+масл", r"масл\w*\s+базов"),
    "OIL-002": (r"ароматическ", r"лаванд"),
    "SCRB-020": (r"скраб",),
    "WRAP-030": (r"альгинат", r"обертыван"),
    "CONS-051": (r"тапоч",),
    "CONS-052": (r"шапоч",),
}


def sku_candidates(text: str) -> list[str]:
    """Return all explicit/name matches; never choose between two oils silently."""
    value = canonical_text(text)
    explicit = {f"{m[1].upper()}-{m[2]}" for m in SKU_RE.finditer(value)}
    if explicit:
        return sorted(explicit)
    matches = {
        sku
        for sku, patterns in ALIASES.items()
        if any(re.search(pattern, value) for pattern in patterns)
    }
    if re.search(r"\bмасл", value) and not matches.intersection({"OIL-001", "OIL-002"}):
        matches.update(("OIL-001", "OIL-002"))
    return sorted(matches)


def locations(text: str) -> list[str]:
    """Do not invent a mapping between named sites and MS identifiers."""
    value = canonical_text(text)
    found = {f"MS-{m[1]}" for m in re.finditer(r"\bms[ -]?(\d{2})\b", value)}
    if "сочи" in value:
        found.add("Сочи")
    if re.search(r"красн\w*\s+полян", value):
        found.add("Красная Поляна")
    return sorted(found)
