"""Check greedy parity and acceptance using the real vLLM verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .driver import REPLAY_ENV, running_scenario
from .serve import read_scenario


PROMPT = [{"role": "user", "content": "List the positive integers in order, starting at one."}]


def smoke(
    plan: Path, scenario: str, output: Path, baseline_result: Path | None = None
) -> None:
    output.mkdir(exist_ok=False, parents=True)
    server, baseline = read_scenario(plan, scenario)
    baseline = {**baseline, "speculative_config": None,
                "per_request_spec_decode_metrics": "none"}
    server = {**server, "extra_args": [*server.get("extra_args", []), "--no-async-scheduling"]}
    options = {"ignore_eos": True}
    if baseline_result is None:
        with running_scenario(server, baseline, output / "baseline.log", env=REPLAY_ENV) as driver:
            expected = driver.chat(PROMPT, 13, **options)
    else:
        expected = json.loads(baseline_result.read_text(encoding="utf-8"))
    (output / "baseline.json").write_text(json.dumps(expected, indent=2) + "\n")
    prompt = expected["prompt_token_ids"]
    tokens = expected["token_ids"]
    if not prompt or len(tokens or []) != 13:
        raise AssertionError("Baseline did not return prompt IDs and exactly 13 output IDs")
    with (output / "trace.jsonl").open("w") as trace:
        for offset in range(1, len(tokens) + 1):
            proposal = tokens[offset:offset + 4]
            if offset == 1:
                proposal[1] = (proposal[1] + 1) % 100
                if proposal[1] == tokens[2]:
                    proposal[1] = (proposal[1] + 1) % 100
            trace.write(json.dumps({
                "prefix_token_ids": prompt + tokens[:offset],
                "prompt_length": len(prompt),
                "draft_token_ids": proposal,
            }) + "\n")
    config = output / "replay.json"
    config.write_text(json.dumps({
        "trace": "trace.jsonl", "log": "steps.jsonl",
        "latency": {"round_trip_ms": 1, "first_token_ms": 1, "tbt_ms": 0.25},
    }) + "\n")
    replay = {
        **baseline,
        "speculative_config": {
            "method": "custom_class",
            "model": "specdec.replay.ReplayProposer",
            "num_speculative_tokens": 4,
        },
        "per_request_spec_decode_metrics": "detailed",
    }
    with running_scenario(
        server, replay, output / "server.log",
        env={**REPLAY_ENV, "SPECDEC_REPLAY_CONFIG": str(config.resolve())},
    ) as driver:
        result = driver.chat(PROMPT, 13, **options)
    (output / "replay-result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["token_ids"] == tokens, "Greedy token IDs diverged"
    metrics = result["speculative_decoding"]
    assert metrics is not None, "vLLM did not return speculative decoding metrics"
    expected_counts = {
        "num_spec_steps": 3,
        "num_draft_tokens": 12,
        "num_accepted_draft_tokens": 9,
    }
    for key, value in expected_counts.items():
        assert metrics[key] == value, f"{key}: expected {value}, got {metrics[key]}"
    assert metrics["per_step_accepted"] == [1, 4, 4], metrics
    assert metrics["per_step_drafted"] == [4, 4, 4], metrics
    steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert not any(step["misses"] for step in steps)
    assert any(step["delay"]["total_ms"] == 2.75 for step in steps)
    print(json.dumps(expected_counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, help="Reuse the baseline from the same plan")
    args = parser.parse_args()
    smoke(args.plan, args.scenario, args.output, args.baseline_result)
