"""Run the complete assignment pipeline, validation, benchmarks or local demo."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from spa_assistant import answer_question, forecast_demand, normalize_movement
from spa_assistant.evaluation import benchmark
from spa_assistant.report import run_pipeline, write_report

ROOT = Path(__file__).resolve().parent
__all__ = ["normalize_movement", "forecast_demand", "answer_question"]


def verify(root: Path) -> dict[str, Any]:
    """Execute real tools; failed checks prevent a success report."""
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    commands = {
        "pytest": [
            "pytest",
            "-q",
            "--cov=spa_assistant",
            "--cov-report=term-missing",
            "--cov-report=json:artifacts/coverage.json",
            "--junitxml=artifacts/pytest.xml",
        ],
        "ruff_check": ["ruff", "check", "."],
        "ruff_format": ["ruff", "format", "--check", "."],
    }
    summary: dict[str, Any] = {}
    for name, arguments in commands.items():
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        output = result.stdout + result.stderr
        (artifacts / f"{name}.txt").write_text(output, encoding="utf-8")
        print(output)
        if result.returncode:
            raise SystemExit(result.returncode)
        summary[name] = "PASS"
    suite = ET.parse(artifacts / "pytest.xml").getroot().find("testsuite")
    if suite is not None:
        summary["tests_passed"] = int(suite.attrib["tests"]) - int(
            suite.attrib["skipped"]
        )
    coverage = json.loads((artifacts / "coverage.json").read_text(encoding="utf-8"))
    summary["line_coverage_percent"] = round(coverage["totals"]["percent_covered"], 2)
    (artifacts / "verification.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Provide a dependency-free default and opt-in development checks."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SPA inventory DS assignment")
    parser.add_argument(
        "--verify", action="store_true", help="Run pytest, coverage and ruff"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Measure latency and allocations"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Open a loopback-only web service"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--question", help="Answer one Russian question")
    parser.add_argument(
        "--check-gigachat",
        action="store_true",
        help="Check GigaChat API without exposing credentials",
    )
    parser.add_argument(
        "--ollama-model", help="Opt in to local Ollama routing for one question"
    )
    args = parser.parse_args()
    if args.check_gigachat:
        from spa_assistant.llm import gigachat_scores_with_status

        _, status = gigachat_scores_with_status("Какой бюджет закупок на квартал?")
        print(f"GigaChat: {status}")
        if status != "ok":
            raise SystemExit(1)
        return
    dataset = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    if args.serve:
        from spa_assistant.web import serve

        serve(dataset, args.port)
        return
    if args.question:
        context = {"history": dataset["history"], "ollama_model": args.ollama_model}
        parsed, confidence, answer = answer_question(args.question, context)
        print(
            json.dumps(
                {"parameters": parsed, "confidence": confidence, "answer": answer},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return
    verification = verify(ROOT) if args.verify else None
    performance = benchmark(dataset) if args.benchmark else None
    results = run_pipeline(dataset)
    write_report(results, ROOT, performance, verification)
    if performance:
        (ROOT / "artifacts" / "benchmark.json").write_text(
            json.dumps(performance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("Готово: RESULTS.md и artifacts/results.json")


if __name__ == "__main__":
    main()
