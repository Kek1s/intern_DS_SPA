"""Minimal single-user loopback demo over the exact same calculation functions."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .forecast import purchase_plan
from .questions import answer_question

MAX_REQUEST_BYTES = 32_768


def make_handler(dataset: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    """Build a handler over read-only demo data; no stock mutation endpoints exist."""

    class Handler(BaseHTTPRequestHandler):
        """Serve an allowlisted page and two JSON calculation endpoints."""

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            """Send an HTTP response with standard headers and the supplied body."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            """Serve the single page; arbitrary files are not exposed."""
            if self.path != "/":
                self._reply(404, b"Not found", "text/plain")
                return
            self._reply(
                200,
                Path(__file__).with_name("web.html").read_bytes(),
                "text/html; charset=utf-8",
            )

        def do_POST(self) -> None:
            """Validate bounded JSON and return calculations or an explicit error."""
            if self.path not in {"/api/plan", "/api/question"}:
                self._reply(404, b"Not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Недопустимый размер запроса")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/plan":
                    result = purchase_plan(
                        dataset["history"],
                        request["horizon_days"],
                        {
                            "budget_limit": request.get("budget_limit"),
                            "credit_undated_incoming": request.get(
                                "credit_undated_incoming", True
                            ),
                        },
                    )
                else:
                    parsed, confidence, answer = answer_question(
                        request["question"],
                        {
                            "history": dataset["history"],
                            "previous_params": request.get("previous_params"),
                        },
                    )
                    result = {
                        "parameters": parsed,
                        "confidence": confidence,
                        "answer": answer,
                    }
                body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
                self._reply(200, body, "application/json; charset=utf-8")
            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                OverflowError,
            ) as error:
                body = json.dumps({"error": str(error)}, ensure_ascii=False).encode()
                self._reply(400, body, "application/json; charset=utf-8")

        def log_message(self, format: str, *args: Any) -> None:
            """Keep the local console quiet; no questions or source data are logged."""

    return Handler


def serve(dataset: dict[str, Any], port: int = 8000) -> None:
    """Run locally until Ctrl+C; not a production authentication boundary."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(dataset))
    print(f"Демонстрация: http://127.0.0.1:{port} — Ctrl+C для остановки")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
