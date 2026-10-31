"""Combine benchmark run summaries into a compact CSV comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "method",
    "model",
    "request_count",
    "successful_requests",
    "failed_requests",
    "wall_time_seconds",
    "requests_per_second",
    "completion_tokens_per_second",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "requests_with_spec_metrics",
    "accepted_draft_tokens",
    "draft_tokens",
    "spec_steps",
    "accepted_draft_tokens_per_step",
    "acceptance_length_including_bonus",
    "draft_acceptance_rate",
]


def load_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    run_path = path.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    latency = summary.get("latency_ms", {})
    acceptance = summary.get("speculative_decoding", {})
    values = {
        "method": run.get("method", path.parent.name),
        "model": run.get("model"),
        "request_count": summary.get("request_count"),
        "successful_requests": summary.get("successful_requests"),
        "failed_requests": summary.get("failed_requests"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "requests_per_second": summary.get("requests_per_second"),
        "completion_tokens_per_second": summary.get("completion_tokens_per_second"),
        "latency_p50_ms": latency.get("p50"),
        "latency_p90_ms": latency.get("p90"),
        "latency_p99_ms": latency.get("p99"),
        "requests_with_spec_metrics": acceptance.get("requests_with_metrics"),
        "accepted_draft_tokens": acceptance.get("num_accepted_draft_tokens"),
        "draft_tokens": acceptance.get("num_draft_tokens"),
        "spec_steps": acceptance.get("num_spec_steps"),
        "accepted_draft_tokens_per_step": acceptance.get(
            "mean_accepted_draft_tokens_per_step"
        ),
        "acceptance_length_including_bonus": acceptance.get(
            "mean_acceptance_length_including_bonus"
        ),
        "draft_acceptance_rate": acceptance.get("draft_acceptance_rate"),
    }
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="summary.json files")
    parser.add_argument("--output", type=Path, default=Path("comparison.csv"))
    args = parser.parse_args()
    rows = [load_row(path) for path in args.summaries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
