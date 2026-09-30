"""Generate human-readable tables and full machine-readable audit results."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluation import backtest
from .forecast import forecast_demand, purchase_plan
from .normalize import normalize_movement
from .questions import answer_question

EXTRA_QUESTIONS = [
    "Сколько OIL-001 закупить на 30 дней?",
    "Посчитай закупку скраба на квартал",
    "Сколько альгинатной маски закупить на 90 дней?",
    "Какой бюджет закупок на год?",
    "Что нужно заказать в ближайшие 30 дней?",
]


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Escape cell content so generated Markdown remains a valid table."""

    def cell(value: Any) -> str:
        """Format a value and escape it for a Markdown table cell."""
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def run_pipeline(dataset: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every supplied example and five additional explicit questions."""
    history = dataset["history"]
    movements = [
        {"id": row["id"], **normalize_movement(row["text"])}
        for row in dataset["movements"]
    ]
    forecasts = [
        forecast_demand(history, sku, horizon, {})
        for sku in sorted(history["weekly_consumption"])
        for horizon in (30, 90)
    ]
    questions = []
    for index, question in enumerate(dataset["questions"] + EXTRA_QUESTIONS, start=1):
        parsed, confidence, answer = answer_question(
            question, {"history": history, "llm_enabled": False}
        )
        questions.append(
            {
                "id": f"Q{index}",
                "question": question,
                "parsed": parsed,
                "confidence": confidence,
                "answer": answer,
            }
        )
    return {
        "as_of": history["as_of"],
        "movements": movements,
        "forecasts": forecasts,
        "questions": questions,
        "backtest": backtest(history),
        "quarter_plan": purchase_plan(history, 90),
    }


def write_report(
    results: dict[str, Any],
    root: Path,
    performance: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> None:
    """Write reproducible results; mark unexecuted checks rather than inventing them."""
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for name in ("dataset.json", "catalog.json"):
        results.setdefault("source_sha256", {})[name] = hashlib.sha256(
            (root / name).read_bytes(),
        ).hexdigest()
    (artifacts / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    parts = [
        "# Результаты тестового задания\n",
        "Отчёт создан командой `python main.py`; полные ответы и параметры — "
        "в [artifacts/results.json](artifacts/results.json). Дата данных: "
        f"**{results['as_of']}**. Дата компьютера на расчёты не влияет.\n",
        "## Часть 1. Нормализация M1–M8\n",
    ]
    fields = ["date", "sku", "location", "operation", "qty", "unit", "batch", "doc_no"]
    parts.append(
        table(
            ["ID", *fields, "Не указано в записи"],
            [
                [
                    row["id"],
                    *(row[field] for field in fields),
                    ", ".join(field for field in fields if row[field] is None),
                ]
                for row in results["movements"]
            ],
        )
    )
    parts.append(
        "Пустые значения — отсутствие исходных сведений. M3: `03/06/26` трактуется "
        "как 3 июня 2026 в русской локали; альтернативу 6 марта без уточнения "
        "исключить нельзя. M8 сопоставляется по однозначному названию тапочек. "
        "MS-01/MS-02 не отождествляются с городами. Эталон всех полей задан в тестах.\n"
    )
    parts.append("## Часть 2. Прогноз на 30 и 90 дней\n")
    forecast_fields = [
        "sku",
        "horizon_days",
        "avg_daily_consumption",
        "forecast_demand",
        "current_stock",
        "incoming_qty",
        "safety_stock",
        "reorder_point",
        "recommended_qty",
        "estimated_cost",
        "stockout_date",
        "confidence",
    ]
    parts.append(
        table(
            forecast_fields,
            [[row[field] for field in forecast_fields] for row in results["forecasts"]],
        )
    )
    parts.append(
        "Количество — в базовой единице SKU, стоимость — в рублях. "
        "Дата дефицита условная: без недатированных поставок и без новой закупки. "
        "Положительный объём в пути условно вычитается из закупки на горизонт; "
        "консервативный режим `credit_undated_incoming=False` доступен отдельно.\n"
    )
    parts.append("### Сквозной сценарий\n")
    parts.append(
        "Скраб имеет низкий запас и риск дефицита до стандартной поставки. "
        "Ниже — полный след расчёта; низкая уверенность требует подтверждения "
        "нового уровня спроса, а не удаления предупреждения.\n\n"
        + next(
            row["explanation"]
            for row in results["forecasts"]
            if row["sku"] == "SCRB-020" and row["horizon_days"] == 90
        )
        + "\n"
    )
    parts.append("### Проверка прогноза во времени\n")
    parts.append(
        table(
            [
                "SKU",
                "Метод",
                "Фолдов",
                "MAE, ед./нед.",
                "Фолдов 4 нед.",
                "MAE суммы за 4 нед.",
                "Пар пересчётов",
                "Среднее изменение заказа",
            ],
            [
                [
                    row["sku"],
                    row["method"],
                    row["folds"],
                    row["mae_weekly"],
                    row["horizon_folds"],
                    row["mae_horizon_total"],
                    row["order_revision_pairs"],
                    row["mean_absolute_order_revision"],
                ]
                for row in results["backtest"]
            ],
        )
    )
    parts.append(
        "Expanding-origin: обучаемся только на предыдущих неделях, прогнозируем "
        "следующую, отсутствующие целевые значения пропускаем. Методы оценены на "
        "одинаковых фолдах. Выбор окна не оптимизировался по этим ошибкам. "
        "Одна последняя неделя может выигрывать на растущем ряду; усреднение "
        "выбрано ради устойчивости закупок к одиночным скачкам. Такой короткий "
        "backtest не подтверждает точность прогноза на квартал или год.\n"
    )
    parts.append("## Часть 3. Все 15 исходных вопросов\n")
    questions = results["questions"]
    parts.append(
        table(
            [
                "ID",
                "Вопрос",
                "Интент",
                "SKU",
                "Объект",
                "Период",
                "Лимит",
                "Статус",
                "Уверенность",
            ],
            [
                [
                    row["id"],
                    row["question"],
                    row["parsed"]["intent"],
                    row["parsed"]["sku"],
                    row["parsed"]["location"],
                    row["parsed"]["period"],
                    row["parsed"]["budget_limit"],
                    row["parsed"]["status"],
                    row["confidence"],
                ]
                for row in questions[:15]
            ],
        )
    )
    unknown = sum(row["parsed"]["intent"] == "unknown" for row in questions[:15])
    unavailable = sum(
        row["parsed"]["status"] == "data_unavailable" for row in questions[:15]
    )
    parts.append(
        f"Доля уточнений `unknown`: {unknown}/15 = {unknown / 15:.1%}; "
        f"распознанных запросов без нужных данных: {unavailable}/15. "
        "Это измерение на намеренно неоднозначных вопросах, не оценка качества "
        "на реальном потоке. Нельзя считать отказ ошибкой, если числовой ответ "
        "потребовал бы выдумать отсутствующие данные.\n"
    )
    parts.append("### Ещё пять контрольных сценариев\n")
    parts.append(
        table(
            ["ID", "Вопрос", "Интент", "Стоимость", "Статус"],
            [
                [
                    row["id"],
                    row["question"],
                    row["parsed"]["intent"],
                    row["parsed"].get("estimated_cost"),
                    row["parsed"]["status"],
                ]
                for row in questions[15:]
            ],
        )
    )
    parts.append("### Полные ответы\n")
    for row in questions:
        parts.append(
            f"<details><summary>{row['id']}: {row['question']}</summary>\n\n"
            + row["answer"]
            + "\n\n</details>\n"
        )
    parts.append((root / "docs" / "METHODS.md").read_text(encoding="utf-8"))
    parts.append("## Фактически выполненные проверки\n")
    if verification:
        parts.append(
            table(
                ["Проверка", "Результат"],
                [[name, value] for name, value in verification.items()],
            )
        )
    else:
        parts.append(
            "В этом запуске проверки не выполнялись. "
            "Запустите `python main.py --verify`.\n"
        )
    parts.append("## Производительность\n")
    if performance:
        parts.append(
            f"Python {performance['python']}; {performance['platform']}; "
            f"повторов на функцию: {performance['repeats']}.\n"
        )
        parts.append(
            table(
                ["Функция", "Медиана, мс", "p95, мс", "Пик Python-аллокаций, KiB"],
                [
                    [name, data["median_ms"], data["p95_ms"], data["python_peak_kib"]]
                    for name, data in performance["cases"].items()
                ],
            )
        )
        parts.append(
            "Измерения локальные, после прогрева; без сети/LLM. Задержки измерены "
            "без tracemalloc, память — отдельным прогоном. Пик аллокаций не равен "
            "RSS процесса; память интерпретатора и весов LLM сюда не включена. "
            "Результаты не являются обещанием для другого оборудования.\n"
        )
    else:
        parts.append("В этом запуске не измерялась: используйте `--benchmark`.\n")
    parts.append("## Воспроизводимость исходных данных\n")
    parts.append(
        table(["Файл", "SHA-256"], list(map(list, results["source_sha256"].items())))
    )
    (root / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
