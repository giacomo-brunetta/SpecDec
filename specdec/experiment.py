"""Capture real draft proposals and compare paired vLLM replay scenarios."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from .benchmark import read_jsonl, summarize
from .driver import REPLAY_ENV, Driver, running_scenario
from .latency import LatencyProfile
from .replay import load_trace
from .serve import read_scenario


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_requests(driver: Driver, workload: list[dict], settings: dict) -> tuple[list[dict], dict]:
    def run(item: tuple[int, dict]) -> dict:
        index, row = item
        messages = row.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": row["prompt"]}]
        return {
            **driver.chat(messages, settings["max_tokens"], ignore_eos=True, **stream_options),
            "request_id": row.get("id", f"request-{index:06d}"),
        }

    stream_options = {"stream": True, "stream_options": {"include_usage": True}} if settings.get(
        "stream", False
    ) else {}
    items = list(enumerate(workload))
    for index in range(settings.get("warmup_requests", 0)):
        run(items[index % len(items)])
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=settings["concurrency"]) as pool:
        records = list(pool.map(run, items))
    return records, summarize(records, time.perf_counter() - started)


def capture(driver: Driver, targets: list[dict], max_k: int, output: Path) -> dict:
    seen: dict[tuple[int, ...], int] = {}
    with output.open("w", encoding="utf-8") as stream:
        for target in targets:
            prompt, sequence = target["prompt_token_ids"], target["token_ids"]
            if not prompt or not sequence:
                raise ValueError("Target did not return prompt and output IDs")
            for offset in range(1, len(sequence) + 1):
                prefix = prompt + sequence[:offset]
                key = tuple(prefix)
                if key in seen:
                    if seen[key] != len(prompt):
                        raise ValueError("The same prefix appears with different prompt lengths")
                    continue
                seen[key] = len(prompt)
                length = min(max_k, len(sequence) - offset - 1)
                proposal = driver.complete_tokens(prefix, length) if length > 0 else []
                if length > 0 and len(proposal) != length:
                    raise ValueError(f"Drafter returned {len(proposal)} of {length} requested IDs")
                stream.write(json.dumps({
                    "prefix_token_ids": prefix,
                    "prompt_length": len(prompt),
                    "draft_token_ids": proposal,
                }) + "\n")
    with output.open(encoding="utf-8") as stream:
        return load_trace(stream)


def expected_steps(prompt: list[int], output: list[int], trace: dict, k: int) -> list[dict]:
    offset = 1
    steps = []
    while offset < len(output):
        prefix = prompt + output[:offset]
        draft = trace[tuple(prefix)].draft[:k]
        accepted = 0
        for predicted, actual in zip(draft, output[offset:]):
            if predicted != actual:
                break
            accepted += 1
        if draft:
            steps.append({
                "output_offset": offset,
                "prefix_length": len(prefix),
                "draft_token_ids": list(draft),
                "accepted_token_ids": list(draft[:accepted]),
                "target_token_id": output[offset + accepted],
                "next_prefix_length": len(prefix) + accepted + 1,
            })
        offset += accepted + 1
    return steps


def verify(targets: list[dict], replay: list[dict], trace: dict, k: int) -> list[dict]:
    if len(targets) != len(replay):
        raise AssertionError("Baseline and replay returned different request counts")
    comparisons = []
    for target, actual in zip(targets, replay):
        prompt = target["prompt_token_ids"]
        tokens = target["token_ids"]
        steps = expected_steps(prompt, tokens, trace, k)
        name = target["request_id"]
        if prompt != actual["prompt_token_ids"]:
            raise AssertionError(f"Prompt token IDs diverged for {name}")
        actual_tokens = actual["token_ids"] or []
        if tokens != actual_tokens:
            offset = next(
                (i for i, (left, right) in enumerate(zip(tokens, actual_tokens)) if left != right),
                min(len(tokens), len(actual_tokens)),
            )
            prior = next((step for step in reversed(steps)
                          if step["output_offset"] <= offset), None)
            raise AssertionError(f"Token IDs diverged for {name} at offset {offset}, replay step {prior}")
        metrics = actual["speculative_decoding"] or {}
        per_step = {
            "per_step_accepted": [len(step["accepted_token_ids"]) for step in steps],
            "per_step_drafted": [len(step["draft_token_ids"]) for step in steps],
        }
        for key, values in per_step.items():
            observed = metrics.get(key) or []
            if observed != values:
                mismatch = next(
                    (i for i, (left, right) in enumerate(zip(values, observed)) if left != right),
                    min(len(values), len(observed)),
                )
                step = steps[mismatch] if mismatch < len(steps) else None
                raise AssertionError(
                    f"{name} {key} diverged at step {mismatch}: {step}, observed {observed}"
                )
        counts = {
            "num_spec_steps": len(steps),
            "num_draft_tokens": sum(len(step["draft_token_ids"]) for step in steps),
            "num_accepted_draft_tokens": sum(len(step["accepted_token_ids"]) for step in steps),
        }
        for key, value in counts.items():
            if metrics.get(key, 0) != value:
                raise AssertionError(f"{name} {key}: expected {value}, got {metrics}")
        comparisons.append({"request_id": name, "steps": steps, **counts})
    return comparisons


def run(settings_path: Path, output: Path) -> None:
    root = settings_path.resolve().parent
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    dataset = root / settings["dataset"]
    workload = read_jsonl(dataset)[:settings.get("max_requests")]
    concurrencies = settings["concurrency"]
    lengths = settings["draft_lengths"]
    profiles = settings["profiles"]
    if (not workload or not concurrencies or not lengths or not profiles
            or any(type(n) is not int or n < 1 for n in concurrencies + lengths)
            or settings["max_tokens"] < 3):
        raise ValueError("Require a workload, profiles, positive concurrency and k, and max_tokens ≥ 3")
    for profile in profiles.values():
        LatencyProfile(**profile)
    if any(not name.isidentifier() for name in profiles):
        raise ValueError("Profile names must be valid identifiers")

    plan = root / settings["plan"]
    target, baseline = read_scenario(plan, settings.get("baseline", "baseline"))
    drafter, draft_scenario = read_scenario(plan, settings["drafter_scenario"])
    if baseline.get("speculative_config") is not None:
        raise ValueError("The baseline scenario must have speculative decoding disabled")
    target = {**target, "extra_args": [*target.get("extra_args", []), "--no-async-scheduling"]}
    draft_scenario = {**draft_scenario, "speculative_config": None,
                      "per_request_spec_decode_metrics": "none"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        **settings,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "target": target,
        "baseline_config": baseline,
        "drafter": drafter,
        "drafter_config": draft_scenario,
    })
    baseline_runs = {}
    summaries = {}
    with running_scenario(target, baseline, output / "baseline.log", env=REPLAY_ENV) as driver:
        for concurrency in concurrencies:
            options = {**settings, "concurrency": concurrency}
            records, summary = run_requests(driver, workload, options)
            baseline_runs[concurrency] = records
            write_json(output / f"baseline-c{concurrency}.json", records)
            summaries[f"baseline-c{concurrency}"] = summary

    trace_file = output / "trace.jsonl"
    baseline_records = [record for records in baseline_runs.values() for record in records]
    with running_scenario(drafter, draft_scenario, output / "drafter.log") as driver:
        trace = capture(driver, baseline_records, max(lengths), trace_file)

    for k in lengths:
        replay_scenario = {
            **baseline,
            "per_request_spec_decode_metrics": "detailed",
            "speculative_config": {
                "method": "custom_class",
                "model": "specdec.replay.ReplayProposer",
                "num_speculative_tokens": k,
            },
        }
        for profile, latency in profiles.items():
            name = f"k{k}-{profile}"
            case = output / name
            case.mkdir()
            config = case / "replay.json"
            write_json(config, {
                "trace": str(trace_file.resolve()),
                "latency": latency,
                "log": "steps.jsonl",
            })
            env = {**REPLAY_ENV, "SPECDEC_REPLAY_CONFIG": str(config.resolve())}
            with running_scenario(target, replay_scenario, case / "server.log", env=env) as driver:
                for concurrency in concurrencies:
                    options = {**settings, "concurrency": concurrency}
                    records, summary = run_requests(driver, workload, options)
                    write_json(case / f"requests-c{concurrency}.json", records)
                    audit = verify(baseline_runs[concurrency], records, trace, k)
                    write_json(case / f"verification-c{concurrency}.json", audit)
                    rate = summary["completion_tokens_per_second"]
                    baseline_rate = summaries[f"baseline-c{concurrency}"]["completion_tokens_per_second"]
                    summary["speedup_vs_baseline"] = (
                        rate / baseline_rate if rate is not None and baseline_rate else None
                    )
                    summaries[f"{name}-c{concurrency}"] = summary
            write_json(output / "summaries.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.settings, args.output)
