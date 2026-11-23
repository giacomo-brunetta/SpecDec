"""Inspect a verified decode step or export its replay fixture."""

import argparse
import json
from pathlib import Path

from .experiment import expected_steps
from .replay import load_trace


def inspect(trace_path: Path, baseline_path: Path, request_id: str, step: int, k: int):
    with trace_path.open(encoding="utf-8") as rows:
        trace = load_trace(rows)
    records = json.loads(baseline_path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    source = next((record for record in records if record["request_id"] == request_id), None)
    if source is None:
        raise ValueError(f"Unknown request ID {request_id!r}")
    prompt, tokens = source["prompt_token_ids"], source["token_ids"]
    steps = expected_steps(prompt, tokens, trace, k)
    if not 0 <= step < len(steps):
        raise ValueError(f"Expected a zero-indexed step below {len(steps)}")
    current = steps[step]
    fixture = []
    for offset in (current["output_offset"], current["next_prefix_length"] - len(prompt)):
        prefix = prompt + tokens[:offset]
        entry = trace.get(tuple(prefix))
        if entry is not None:
            fixture.append({
                "prefix_token_ids": prefix,
                "prompt_length": entry.prompt_length,
                "draft_token_ids": list(entry.draft[:k]),
            })
    return current, fixture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    current, fixture = inspect(args.trace, args.baseline, args.request_id, args.step, args.k)
    print(json.dumps(current, indent=2))
    if args.fixture:
        with args.fixture.open("x", encoding="utf-8") as output:
            for row in fixture:
                output.write(json.dumps(row) + "\n")
