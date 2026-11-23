"""Estimate shared accelerator queueing from recorded proposal batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import percentile


def simulate(logs: list[list[dict]], service_scale: float = 1) -> list[dict]:
    arrivals = []
    for source, records in enumerate(logs):
        valid = [
            (index, batch) for index, batch in enumerate(records)
            if batch.get("proposals") and not batch.get("misses")
        ]
        if not valid:
            continue
        origin = valid[0][1]["arrival_ms"]
        for index, batch in valid:
            delay = batch["delay"]
            sent = batch["arrival_ms"] - origin
            outbound = delay["round_trip_ms"] / 2 + delay["upload_ms"]
            inbound = delay["round_trip_ms"] / 2 + delay["download_ms"]
            service = service_scale * (delay["first_token_ms"] + delay["tbt_ms"])
            arrivals.append((sent + outbound, source, index, sent, service, inbound))
    available = 0.0
    results = []
    for ready, source, index, sent, service, inbound in sorted(arrivals):
        start = max(ready, available)
        available = start + service
        results.append({
            "source": source,
            "batch_index": index,
            "sent_ms": sent,
            "accelerator_start_ms": start,
            "queue_ms": start - ready,
            "service_ms": service,
            "total_ms": available + inbound - sent,
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--service-scale", type=float, default=1)
    args = parser.parse_args()
    if not 0 <= args.service_scale < float("inf"):
        parser.error("--service-scale must be finite and nonnegative")
    batches = simulate([
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in args.logs
    ], args.service_scale)
    waits = [row["queue_ms"] for row in batches]
    print(json.dumps({
        "p50_queue_ms": percentile(waits, 0.50),
        "p99_queue_ms": percentile(waits, 0.99),
        "batches": batches,
    }, indent=2))
