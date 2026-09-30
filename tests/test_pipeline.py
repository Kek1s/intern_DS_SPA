"""End-to-end pipeline, report provenance and real local HTTP integration."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from spa_assistant.catalog import ROOT, Product, decimal_number, get_catalog
from spa_assistant.evaluation import backtest, benchmark
from spa_assistant.report import run_pipeline, write_report
from spa_assistant.web import make_handler


def test_full_pipeline_and_report(dataset: dict, tmp_path: Path) -> None:
    """Run all examples and persist a report without mutating source input."""
    snapshot = json.dumps(dataset, sort_keys=True)
    results = run_pipeline(dataset)
    assert len(results["movements"]) == 8
    assert len(results["forecasts"]) == 6
    assert len(results["questions"]) == 20
    assert json.dumps(dataset, sort_keys=True) == snapshot
    (tmp_path / "docs").mkdir()
    for name in ("dataset.json", "catalog.json", "docs/METHODS.md"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    write_report(results, tmp_path)
    saved = json.loads(
        (tmp_path / "artifacts/results.json").read_text(encoding="utf-8")
    )
    assert len(saved["source_sha256"]["dataset.json"]) == 64
    report = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "Q20" in report and "M8" in report
    assert "В этом запуске проверки не выполнялись" in report
    measurements = benchmark(dataset)
    assert measurements["cases"]["forecast_demand"]["python_peak_kib"] > 0
    write_report(results, tmp_path, measurements, {"pytest": "PASS"})
    assert "PASS" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")


def test_backtest_no_future_leakage() -> None:
    """The first forecast cannot know about the future demand jump."""
    history = {"weekly_consumption": {"X": [7.0] * 4 + [70.0]}}
    for row in backtest(history):
        assert row["folds"] == 1
        assert row["mae_weekly"] == 63


def test_packaged_catalog_stays_identical() -> None:
    """Package installation and source checkout must use identical catalog data."""
    assert (ROOT / "catalog.json").read_bytes() == (
        ROOT / "spa_assistant/catalog.json"
    ).read_bytes()
    assert len(get_catalog()) == 6


def test_catalog_validation() -> None:
    """Invalid supplier constraints cannot enter the calculation layer."""
    row = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))[0]
    for field, bad in (
        ("pack_size", 0),
        ("lead_time_days", True),
        ("unit", "unknown"),
        ("sku", "unknown"),
    ):
        with pytest.raises(ValueError):
            Product.from_dict({**row, field: bad})
    with pytest.raises(ValueError):
        decimal_number(None, "qty")


def test_real_http_demo(dataset: dict) -> None:
    """Both demo actions call the production core; invalid requests are rejected."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dataset))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base, timeout=5) as response:
            assert "От остатка к плану закупок" in response.read().decode()
        for path, payload in (
            ("/api/plan", {"horizon_days": 90}),
            ("/api/question", {"question": "Какой бюджет закупок на квартал?"}),
        ):
            request = Request(
                base + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read())
            if path == "/api/plan":
                assert result["estimated_cost"] == 414189
            else:
                assert result["parameters"]["estimated_cost"] == 414189
        with pytest.raises(HTTPError) as missing:
            urlopen(base + "/../dataset.json", timeout=5)
        assert missing.value.code == 404
        with pytest.raises(HTTPError) as invalid:
            urlopen(Request(base + "/api/plan", data=b'{"horizon_days":-1}'), timeout=5)
        assert invalid.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
