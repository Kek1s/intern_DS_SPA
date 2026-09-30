"""Reproducible time-series backtesting and local performance measurements."""

import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from .catalog import get_catalog
from .forecast import estimate_level, forecast_demand, round_order
from .normalize import normalize_movement
from .questions import answer_question

BENCHMARK_REPEATS = 300
WARMUP_REPEATS = 10
BACKTEST_INITIAL_WEEKS = 4
BACKTEST_HORIZON_WEEKS = 4


def backtest(history: dict[str, Any]) -> list[dict[str, Any]]:
    """One-step expanding-origin MAE; each prediction sees strictly earlier weeks.

    Missing targets are skipped, not scored as zero. All baseline methods are
    evaluated at identical origins. These few folds cannot validate 90-day error.
    """
    results = []
    for sku, series in history["weekly_consumption"].items():
        errors: dict[str, list[float]] = {
            "expanding_mean": [],
            "last_observation": [],
            "robust_recent_mean": [],
        }
        cumulative_errors = {method: [] for method in errors}
        order_revisions = {method: [] for method in errors}
        previous_orders: dict[str, tuple[int, float]] = {}
        product = get_catalog().get(sku)
        for origin in range(BACKTEST_INITIAL_WEEKS, len(series)):
            if series[origin] is None:
                continue
            train = series[:origin]
            observed = [value for value in train if value is not None]
            if not observed:
                continue
            try:
                recent = float(estimate_level(train)["weekly_level"])
            except ValueError:
                continue
            predictions = {
                "expanding_mean": statistics.fmean(observed),
                "last_observation": observed[-1],
                "robust_recent_mean": recent,
            }
            for method, predicted in predictions.items():
                errors[method].append(abs(series[origin] - predicted))
                targets = series[origin : origin + BACKTEST_HORIZON_WEEKS]
                if len(targets) == BACKTEST_HORIZON_WEEKS and all(
                    value is not None for value in targets
                ):
                    cumulative_errors[method].append(
                        abs(sum(targets) - predicted * BACKTEST_HORIZON_WEEKS)
                    )
                if product is not None:
                    # Fixed zero-stock scenario isolates forecast-driven revisions.
                    order = float(
                        round_order(
                            Decimal(str(predicted)) * BACKTEST_HORIZON_WEEKS,
                            product.pack_size,
                            product.min_order_qty,
                        )
                    )
                    previous = previous_orders.get(method)
                    if previous is not None and previous[0] == origin - 1:
                        order_revisions[method].append(abs(order - previous[1]))
                    previous_orders[method] = (origin, order)
        results.extend(
            {
                "sku": sku,
                "method": method,
                "folds": len(values),
                "mae_weekly": statistics.fmean(values) if values else None,
                "horizon_weeks": BACKTEST_HORIZON_WEEKS,
                "horizon_folds": len(cumulative_errors[method]),
                "mae_horizon_total": (
                    statistics.fmean(cumulative_errors[method])
                    if cumulative_errors[method]
                    else None
                ),
                "order_revision_pairs": len(order_revisions[method]),
                "mean_absolute_order_revision": (
                    statistics.fmean(order_revisions[method])
                    if order_revisions[method]
                    else None
                ),
            }
            for method, values in errors.items()
        )
    return results


def benchmark(dataset: dict[str, Any]) -> dict[str, Any]:
    """Measure warmed call latency separately from traced Python allocations.

    No network or LLM. Peak is tracemalloc's Python allocation peak, not RSS.
    All measured outputs are discarded so memory does not grow with iterations.
    """
    history = dataset["history"]
    cases: dict[str, Callable[[], Any]] = {
        "normalize_movement": lambda: normalize_movement(
            dataset["movements"][0]["text"]
        ),
        "forecast_demand": lambda: forecast_demand(history, "OIL-001", 90, {}),
        "answer_question": lambda: answer_question(
            "Какой бюджет закупок на квартал?",
            {"history": history, "llm_enabled": False},
        ),
    }
    output: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "repeats": BENCHMARK_REPEATS,
        "mode": "warm, offline, single process; tracemalloc peak is not RSS",
        "cases": {},
    }
    for name, call in cases.items():
        for _ in range(WARMUP_REPEATS):
            call()
        samples = []
        for _ in range(BENCHMARK_REPEATS):
            start = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        tracemalloc.start()
        tracemalloc.reset_peak()
        for _ in range(BENCHMARK_REPEATS):
            call()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        output["cases"][name] = {
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[int(len(samples) * 0.95) - 1],
            "python_peak_kib": peak / 1024,
        }
    return output
