"""Capture drafter proposals and run all 52 speculative replay cells."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from .benchmark import read_jsonl
from .campaign import K, MAX_TOKENS, baseline, measure, write_checkpoint
from .driver import REPLAY_ENV, running_scenario
from .experiment import verify
from .latency import LatencyProfile
from .replay import load_trace
from .serve import read_scenario

ACTIVE_PARAMS_B = {"E2B": 2.3, "E4B": 4.5, "12B": 11.95}
RATES = {"ideal": None, "fpga": 1, "asic": 10, "cs3": 25}
LINKS = {
    "ideal": {},
    "nvlink": {"round_trip_ms": 0.008, "up": 250e9, "down": 250e9},
    "pcie": {"round_trip_ms": 0.020, "up": 23e9, "down": 23e9},
    "hdr100": {"round_trip_ms": 0.025, "up": 12e9, "down": 12e9},
}
ORDER_SEED = 555
CAPTURE_BATCH = 8


def latency_profile(model: str, link_name: str, compute: str) -> dict:
    rate = RATES[compute]
    # 1 GFLOP / (1 TFLOP/s) = 1 ms.
    token_ms = 2 * ACTIVE_PARAMS_B[model] / rate if rate else 0
    link = LINKS[link_name]
    return {
        "round_trip_ms": link.get("round_trip_ms", 0),
        "first_token_ms": token_ms,
        "tbt_ms": token_ms,
        "upload_bytes_per_second": link.get("up"),
        "download_bytes_per_second": link.get("down"),
        "upload_prompt_tokens": True,
    }


def cells() -> list[tuple[str, str, str]]:
    grid = list(itertools.product(ACTIVE_PARAMS_B, LINKS, RATES))
    grid += [("oracle", link, "ideal") for link in LINKS]
    random.Random(ORDER_SEED).shuffle(grid)
    preflight = [("oracle", "ideal", "ideal"), ("E2B", "ideal", "ideal")]
    return [*preflight, *(cell for cell in grid if cell not in preflight)]


def jsonl(rows):
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def drafter_trace(
    output: Path, target: dict, model: str, targets: list[dict],
    *, before_capture: Callable[[], None] | None = None,
) -> Path:
    root = output / "traces" / model
    trace_path = root / "trace.jsonl"
    if trace_path.exists() and (root / "capture.json").exists():
        return trace_path
    root.mkdir(parents=True, exist_ok=True)
    server = {
        **target,
        "model": str(Path(target["model"]).with_name(f"gemma-4-{model}-it")),
        "served_model_name": f"gemma-4-{model}-it",
        "pipeline_parallel_size": 1, "gpu_memory_utilization": 0.92,
        "extra_args": [
            "--enforce-eager", "--language-model-only",
            "--max-num-seqs", str(CAPTURE_BATCH), "--max-num-batched-tokens", "4096",
            "--enable-prefix-caching", "--disable-uvicorn-access-log",
        ],
    }
    checkpoints = [root / f"{index:03d}-{record['request_id']}.jsonl"
                   for index, record in enumerate(targets)]
    if not all(path.exists() for path in checkpoints):
        print(f"Capturing {model}", flush=True)
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        pinned_gpu = (before or "0").split(",")[0]
        scope = (nullcontext(None) if model == "oracle" else running_scenario(
            server, {"speculative_config": None}, root / "server.log",
            env={"CUDA_VISIBLE_DEVICES": pinned_gpu},
        ))
        with scope as driver:
            for index, (record, path) in enumerate(zip(targets, checkpoints), 1):
                if path.exists():
                    continue
                if before_capture is not None:
                    before_capture()
                rows = capture_prompt(driver, record)
                temp = path.with_suffix(".tmp")
                temp.write_text(jsonl(rows), encoding="utf-8")
                temp.replace(path)
                if index % 10 == 0:
                    print(f"Trace {model}: {index}/{len(targets)} prompts", flush=True)
    temp = trace_path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        for path in checkpoints:
            stream.write(path.read_bytes())
    with temp.open(encoding="utf-8") as stream:
        entries = len(load_trace(stream))
    write_checkpoint(root / "capture.json", {"model": model, "prefixes": entries,
                     "source": "target_greedy" if model == "oracle" else server,
                     "batch_size": None if model == "oracle" else CAPTURE_BATCH})
    temp.replace(trace_path)
    return trace_path


def capture_prompt(driver, record: dict) -> list[dict]:
    prompt, output = record["prompt_token_ids"], record["token_ids"]
    rows = []
    requests = defaultdict(list)
    for offset in range(1, len(output) + 1):
        length = min(K, len(output) - offset - 1)
        row = {"prefix_token_ids": prompt + output[:offset],
               "prompt_length": len(prompt), "draft_token_ids": []}
        if length > 0:
            if driver is None:
                row["draft_token_ids"] = output[offset:offset + length]
            else:
                requests[length].append(row)
        rows.append(row)
    for length, batch in requests.items():
        for start in range(0, len(batch), CAPTURE_BATCH):
            part = batch[start:start + CAPTURE_BATCH]
            proposals = driver.complete_token_batch(
                [row["prefix_token_ids"] for row in part], length,
            )
            for row, tokens in zip(part, proposals):
                row["draft_token_ids"] = tokens
    return rows


def audit_replay(targets, requests, trace, log: Path) -> list[dict]:
    if len(targets) != len(requests):
        raise AssertionError("Baseline and replay request counts differ")
    audits = []
    with log.open("rb") as stream:
        for gold, actual in zip(targets, requests):
            if (gold["request_id"] != actual["request_id"]
                    or gold["prompt_token_ids"] != actual["prompt_token_ids"]):
                raise AssertionError(f"Prompt changed for {gold['request_id']}")
            stream.seek(actual["step_log_start"])
            steps = [json.loads(line) for line in stream.read(
                actual["step_log_end"] - actual["step_log_start"]
            ).splitlines()]
            if not steps or sum(len(step["committed_token_ids"]) for batch in steps
                                for step in batch["steps"]) != MAX_TOKENS:
                raise AssertionError(f"Incomplete step log for {gold['request_id']}")
            tokens = actual["token_ids"]
            gold_ids = gold["token_ids"]
            mismatch = next((i for i, (a, b) in enumerate(zip(gold_ids, tokens)) if a != b), None)
            misses = sum(step["misses"] for step in steps)
            parity_error = None
            if mismatch is None:
                try:
                    verify([gold], [actual], trace, K)
                except AssertionError as exc:
                    parity_error = str(exc)
            audit = {
                "request_id": gold["request_id"],
                "group": gold["metadata"]["group"],
                "identical_output": mismatch is None,
                "first_divergent_token_zero_based": mismatch,
                "spec_metric_parity_error": parity_error,
                "trace_misses": misses,
                "logged_batches": len(steps),
                "proposals": sum(step["proposals"] for step in steps),
                "upload_bytes": sum(step["delay"]["upload_bytes"] for step in steps),
                "download_bytes": sum(step["delay"]["download_bytes"] for step in steps),
                "modeled_link_ms": sum(sum(step["delay"][k] for k in (
                    "round_trip_ms", "upload_ms", "download_ms")) for step in steps),
                "modeled_draft_ms": sum(step["delay"]["first_token_ms"] + step["delay"]["tbt_ms"]
                                        for step in steps),
                "observed_wait_ms": sum(step["wait_ms"] for step in steps),
            }
            audits.append(audit)
    return audits


def run_cell(output, target, greedy, workload, targets, trace_path, cell):
    model, link, compute = cell
    name = f"{model}-{link}-{compute}"
    root = output / "cells" / name
    if (root / "summary.json").exists():
        return
    if root.exists():
        root.rename(root.with_name(f"{root.name}.incomplete-{time.time_ns()}"))
    root.mkdir(parents=True)
    profile = latency_profile(*cell)
    LatencyProfile(**profile)
    config = root / "replay.json"
    write_checkpoint(config, {
        "trace": str(trace_path.resolve()), "latency": profile,
        "log": "steps.jsonl", "deadline_wait": True, "strict": False,
    })
    scenario = {
        **greedy,
        "per_request_spec_decode_metrics": "detailed",
        "speculative_config": {
            "method": "custom_class", "model": "specdec.replay.ReplayProposer",
            "num_speculative_tokens": K,
        },
    }
    print(f"Starting replay {name}", flush=True)
    env = {**REPLAY_ENV, "SPECDEC_REPLAY_CONFIG": str(config.resolve())}
    log = root / "steps.jsonl"
    with running_scenario(target, scenario, root / "server.log", env=env) as driver:
        requests, summary = measure(driver, workload, root / "requests.jsonl", step_log=log)
    with trace_path.open(encoding="utf-8") as stream:
        trace = load_trace(stream)
    audit = audit_replay(targets, requests, trace, log)
    write_checkpoint(root / "audit.json", audit)
    write_checkpoint(root / "summary.json", summary)
    print(f"Finished {name}: {summary['completion_tokens_per_second']:.2f} tok/s, "
          f"{sum(row['identical_output'] for row in audit)}/{len(audit)} identical", flush=True)


def sweep(plan: Path, dataset: Path, output: Path) -> None:
    baseline(plan, dataset, output)
    workload = read_jsonl(dataset)
    targets = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    target, greedy = read_scenario(plan, "baseline")
    order = cells()
    write_checkpoint(output / "grid.json", {
        "active_params_billions": ACTIVE_PARAMS_B, "effective_tflops": RATES,
        "links": LINKS, "seed": ORDER_SEED, "order": order,
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    })
    traces = {}
    for cell in order:
        model = cell[0]
        if model not in traces:
            traces[model] = drafter_trace(output, target, model, targets)
        run_cell(output, target, greedy, workload, targets, traces[model], cell)
    print("All 52 replay cells complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sweep(args.plan, args.dataset, args.output)


if __name__ == "__main__":
    main()
