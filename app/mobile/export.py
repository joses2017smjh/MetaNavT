"""Capture real fixture HTTP responses for the mobile app's offline demo.

``python -m app.mobile.export --output app/mobile/demo-response.json``
Captured timing values are one local run, not mobile/device performance claims.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.mobile.demo import STARTER_QUERIES, create_app


def capture() -> dict:
    with TestClient(create_app()) as client:
        examples = []
        for index, query in enumerate(STARTER_QUERIES, start=1):
            response = client.post("/api/retrieve/", json={"query": query, "k": 3})
            response.raise_for_status()
            examples.append({"id": f"demo-{index}", "query": query, "response": response.json()})
        health = client.get("/health")
        health.raise_for_status()
    return {
        "schema_version": 1,
        "mode": "demo",
        "synthetic": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "timing_notice": "Server-stage times from this fixture capture, not a phone or production benchmark.",
        "provenance": json.loads(Path(__file__).with_name("fixture-provenance.json").read_text()),
        "health": health.json(),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(capture(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
