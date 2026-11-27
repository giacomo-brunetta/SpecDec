"""Measure DSpark and EAGLE-3 over the four campaign link profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from .benchmark import read_jsonl
from .campaign import MAX_TOKENS, measure, write_checkpoint
from .campaign_sweep import LINKS, ORDER_SEED
from .driver import running_scenario
from .latency import LatencyProfile
from .native_link import ENV, account_round
from .serve import read_scenario

SPECULATORS = {
    "eagle3": {
        "k": 4,
        "revision": "28a1c8b4bb64dbaee883ba35341841138bdf1fe3",
        "activation_bytes_per_position": 3 * 5376 * 2,
    },
    "dspark": {
        # The checkpoint's documented deployment uses seven draft tokens.
        "k": 7,
        "revision": "63e2877348caa03a2c58ab3c47224f4164af08c4",
        "activation_bytes_per_position": 5 * 5376 * 2,
    },
}


def profile_args(link: str) -> dict:
    spec = LINKS[link]
    return {
        "round_trip_ms": spec.get("round_trip_ms", 0),
        "upload_bytes_per_second": spec.get("up"),
        "download_bytes_per_second": spec.get("down"),
    }


def rows_from_log(path: Path, records: list[dict], method: str, link: str) -> list[dict]:
    model = SPECULATORS[method]
    profile = LatencyProfile(**profile_args(link))
    audited = []
    with path.open("rb") as log:
        for record in records:
            start, end = record["step_log_start"], record["step_log_end"]
            log.seek(start)
            rounds = [json.loads(row) for row in log.read(end - start).splitlines()]
            if not rounds or sum(step["verified_tokens"] for step in rounds) <= 0:
                raise ValueError(f"No native speculation for {record['request_id']}")
            if rounds[0]["positions"] != record["prompt_tokens"]:
                raise ValueError(f"Expected one full-prompt activation transfer for {record['request_id']}")
            for step in rounds:
                if not 0 < step["draft_tokens"] <= model["k"]:
                    raise ValueError(f"Unexpected draft size for {record['request_id']}")
                expected = account_round(
                    profile, activation_bytes=step["positions"] * model["activation_bytes_per_position"],
                    positions=step["positions"], requests=1, draft_tokens=step["draft_tokens"],
                )
                if any(step[key] != value for key, value in expected.items()):
                    raise ValueError(f"Incorrect native payload or link cost for {record['request_id']}")
            audited.append({
                **record,
                "rounds": len(rounds),
                "round_shape": [[s[key] for key in (
                    "positions", "draft_tokens", "verified_tokens", "accepted_tokens"
                )] for s in rounds],
                "verified_draft_tokens": sum(step["verified_tokens"] for step in rounds),
                "accepted_draft_tokens": sum(step["accepted_tokens"] for step in rounds),
                "activation_bytes": sum(step["activation_bytes"] for step in rounds),
                "modeled_first_round_ms": rounds[0]["total_ms"],
                "modeled_link_ms": sum(step["total_ms"] for step in rounds),
                "observed_wait_ms": sum(step["wait_ms"] for step in rounds),
            })
    return audited


def run_cell(output: Path, server: dict, baseline: dict, workload: list[dict],
             cell: tuple[str, str], env: dict) -> None:
    method, link = cell
    root = output / f"{method}-{link}"
    result = root / "result.json"
    if result.exists():
        return
    root.mkdir(exist_ok=True)
    scenario = dict(baseline)
    cell_env = {**env, ENV: ""}
    log_path = None
    if method != "greedy":
        draft = SPECULATORS[method]
        log_path = root / "rounds.jsonl"
        log_path.write_text("", encoding="utf-8")
        scenario["speculative_config"] = {
            "model": f"RedHatAI/gemma-4-31B-it-speculator.{method}",
            "method": method,
            "revision": draft["revision"],
            "num_speculative_tokens": draft["k"],
            "enable_adaptive_verification": False,
        }
        cell_env[ENV] = json.dumps({"profile": profile_args(link), "log": str(log_path)})
    print(f"Running {root.name}", flush=True)
    with running_scenario(server, scenario, root / "server.log", env=cell_env) as driver:
        records, summary = measure(driver, workload, root / "requests.jsonl", step_log=log_path)
    if log_path is not None:
        records = rows_from_log(log_path, records, method, link)
    write_checkpoint(result, {"records": records, "summary": summary})


def report(output: Path, cells: list[tuple[str, str]]) -> None:
    def read_cell(method, link):
        return json.loads((output / f"{method}-{link}" / "result.json").read_text())

    greedy = read_cell("greedy", "ideal")
    greedy_rate = greedy["summary"]["completion_tokens_per_second"]
    rows = []
    for method, link in cells:
        result = read_cell(method, link)
        requests = result["records"]
        ideal = read_cell(method, "ideal") if method != "greedy" else greedy
        if [r["request_id"] for r in requests] != [r["request_id"] for r in greedy["records"]]:
            raise ValueError("Request order differs across cells")
        rate = result["summary"]["completion_tokens_per_second"]
        ideal_rate = ideal["summary"]["completion_tokens_per_second"]
        spec = method != "greedy"
        rounds = sum(r.get("rounds", 0) for r in requests)
        verified = sum(r.get("verified_draft_tokens", 0) for r in requests)
        accepted = sum(r.get("accepted_draft_tokens", 0) for r in requests)
        paired_latency_ratios = [
            r["latency_ms"] / c["latency_ms"] for r, c in zip(requests, ideal["records"])
            if r["token_ids"] == c["token_ids"] and r.get("round_shape") == c.get("round_shape")
        ]
        link_times = [r["modeled_link_ms"] for r in requests] if spec else []
        projected_seconds = ideal["summary"]["wall_time_seconds"] + sum(link_times) / 1000
        rows.append({
            "method": method, "link": link, "k": SPECULATORS.get(method, {}).get("k", 0),
            "requests": len(requests), "tokens_per_second": rate,
            "speedup_vs_greedy": rate / greedy_rate, "throughput_ratio_vs_ideal": rate / ideal_rate,
            "projected_tokens_per_second": (
                len(requests) * MAX_TOKENS / projected_seconds
                if len(paired_latency_ratios) == len(requests) else None
            ),
            "ttft_p50_ms": result["summary"]["ttft_ms"]["p50"],
            "tpot_p50_ms": statistics.median(
                (r["token_chunks"][-1]["elapsed_ms"] - r["ttft_ms"]) / (MAX_TOKENS - 1)
                for r in requests
            ),
            "exact_outputs_vs_greedy": sum(
                r["token_ids"] == c["token_ids"] for r, c in zip(requests, greedy["records"])
            ),
            "exact_outputs_and_round_shapes_vs_ideal": len(paired_latency_ratios),
            "median_paired_latency_ratio_vs_ideal": (
                statistics.median(paired_latency_ratios) if paired_latency_ratios else None
            ),
            "proposal_rounds": rounds, "verified_draft_tokens": verified, "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / verified if verified else None,
            "activation_bytes": sum(r.get("activation_bytes", 0) for r in requests),
            "modeled_first_round_p50_ms": statistics.median(
                r["modeled_first_round_ms"] for r in requests
            ) if spec else 0,
            "modeled_link_ms_per_request": statistics.mean(link_times) if spec else 0,
            "observed_wait_ms_per_request": statistics.mean(
                r["observed_wait_ms"] for r in requests
            ) if spec else 0,
        })
    with (output / "native-sweep.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Use the shortest and longest prompts")
    parser.add_argument("--no-batch-invariant", action="store_true")
    args = parser.parse_args()
    server, baseline = read_scenario(args.plan, "baseline")
    if baseline.get("speculative_config") or server.get("pipeline_parallel_size", 1) != 1:
        parser.error("Use a non-speculative baseline plan with tensor parallelism and PP=1")
    workload = read_jsonl(args.dataset)
    if len(workload) != 100 or len({row["id"] for row in workload}) != 100:
        parser.error("Expected the original 100 unique campaign prompts")
    if args.smoke:
        by_length = sorted(workload, key=lambda r: r["metadata"]["prompt_tokens"])
        workload = [by_length[0], by_length[-1]]
    links = ("ideal",) if args.smoke else LINKS
    cells = [(method, link) for method in SPECULATORS for link in links]
    physical = [cell for cell in cells if cell[1] != "ideal"]
    random.Random(ORDER_SEED).shuffle(physical)
    order = [(method, "ideal") for method in SPECULATORS] + physical
    env = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_BATCH_INVARIANT": "0" if args.no_batch_invariant else "1",
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "request_ids": [r["id"] for r in workload],
        "server": server, "baseline": baseline, "env": env,
        "links": LINKS, "speculators": SPECULATORS, "cell_order": order,
        "max_tokens": MAX_TOKENS, "vllm": "0.31.0", "warmup_requests": 1,
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        parser.error(f"Existing run at {output} has different settings")
    write_checkpoint(path, manifest)
    for cell in [("greedy", "ideal"), *order]:
        run_cell(output, server, baseline, workload, cell, env)
    report(output, [("greedy", "ideal"), *cells])


if __name__ == "__main__":
    main()
