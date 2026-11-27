"""Run the paired 100-prompt Gemma accelerator replay campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .benchmark import read_jsonl, summarize
from .driver import REPLAY_ENV, running_scenario
from .experiment import write_json
from .serve import read_scenario

MAX_TOKENS = 128
K = 4


def write_checkpoint(path: Path, data) -> None:
    temp = path.with_suffix(".tmp")
    write_json(temp, data)
    temp.replace(path)


def measure(driver, workload: list[dict], output: Path, *,
            step_log: Path | None = None) -> tuple[list[dict], dict]:
    def request(row):
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        return driver.chat(
            messages, MAX_TOKENS, ignore_eos=True, stream=True,
            stream_options={"include_usage": True},
        )

    request(workload[0])
    records = []
    log_start = step_log.stat().st_size if step_log is not None else None
    started = time.perf_counter()
    with output.open("w", encoding="utf-8", buffering=1) as journal:
        for index, row in enumerate(workload, 1):
            record = {**request(row), "request_id": row["id"], "metadata": row["metadata"]}
            if step_log is not None:
                log_end = step_log.stat().st_size
                record.update(step_log_start=log_start, step_log_end=log_end)
                log_start = log_end
            length = len(record["token_ids"] or [])
            if length != MAX_TOKENS:
                raise ValueError(f"{row['id']} returned {length} output tokens, expected {MAX_TOKENS}")
            journal.write(json.dumps(record) + "\n")
            records.append(record)
            if index % 10 == 0:
                print(f"{output.parent.name}/{output.stem}: {index}/{len(workload)} prompts", flush=True)
    return records, summarize(records, time.perf_counter() - started)


def baseline(plan: Path, dataset: Path, output: Path) -> None:
    target, scenario = read_scenario(plan, "baseline")
    workload = read_jsonl(dataset)
    if len(workload) != 100 or len({row["id"] for row in workload}) != 100:
        raise ValueError("Expected exactly 100 unique prompt IDs")
    manifest = {
        "target": target, "baseline": scenario, "max_tokens": MAX_TOKENS, "k": K,
        "concurrency": 1, "warmup_requests": 1, "ignore_eos": True,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        manifest["batch_invariant"] = True
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Workload or baseline differs from the existing campaign at {output}")
    write_checkpoint(manifest_path, manifest)
    if (output / "baseline.json").exists():
        print("Baseline checkpoint already exists", flush=True)
        return
    with running_scenario(target, scenario, output / "baseline.log", env=REPLAY_ENV) as driver:
        records, summary = measure(driver, workload, output / "baseline-progress.jsonl")
    write_checkpoint(output / "baseline-summary.json", summary)
    write_checkpoint(output / "baseline.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline(args.plan, args.dataset, args.output)


if __name__ == "__main__":
    main()
