"""Shared immutable source fixtures; tests copy data before changing it."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary tests never inherit live credentials from the host environment."""
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_SCOPE", raising=False)


@pytest.fixture
def dataset() -> dict:
    """Load the supplied dataset for each test."""
    return json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))


@pytest.fixture
def history(dataset: dict) -> dict:
    """Provide independent history to mutation-based edge tests."""
    return dataset["history"]
