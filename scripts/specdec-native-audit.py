"""Audit all real timing cells and summarize paired link effects."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from specdec.native_campaign import LINKS, profile_args, rows_from_log
from specdec.latency import LatencyProfile
from specdec.native_link import account_round

root, workload_path = map(Path, sys.argv[1:])
manifest = json.loads((root / "manifest.json").read_text())
SPECULATORS = manifest["speculators"]
assert hashlib.sha256(workload_path.read_bytes()).hexdigest() == manifest["dataset_sha256"]
workload = [json.loads(line) for line in workload_path.read_text().splitlines()]
assert len(workload) == 100
ids = [r["id"] for r in workload]
assert manifest["request_ids"] == ids and len(set(ids)) == 100
expected = {("greedy", "ideal")} | {(m, l) for m in SPECULATORS for l in LINKS}
header = (
    "| Method | Link | tok/s | Speedup vs greedy | TTFT p50 (ms) | TPOT p50 (ms) | "
    "Acceptance | Link ms/request | Wait ms/request | Exact vs greedy | Exact vs method ideal |\n"
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)
lines = [f"# Native speculator sweep for {manifest['server']['model']}", "",
         f"vLLM {manifest['vllm']}, {len(ids)} prompts, {manifest['max_tokens']} output tokens. "
         "Each speculative run includes native drafting and serialized transport.", "", header]

def load(method, link):
    return json.loads((root / f"{method}-{link}" / "result.json").read_text())

all_results = {cell: load(*cell) for cell in expected}
greedy_reqs = all_results["greedy", "ideal"]["records"]
greedy_rate = all_results["greedy", "ideal"]["summary"]["completion_tokens_per_second"]
summary = list(csv.DictReader((root / "native-sweep.csv").open()))
assert len(summary) == len(expected) and {(row["method"], row["link"]) for row in summary} == expected
strata = []
for row in summary:
    method, link = row["method"], row["link"]
    group = all_results[method, link]
    requests = group["records"]
    assert [r["request_id"] for r in requests] == ids
    assert all(r["status"] == "ok" and len(r["token_ids"]) == 128 for r in requests)
    for spec, baseline, data in zip(requests, greedy_reqs, workload):
        assert spec["prompt_token_ids"] == baseline["prompt_token_ids"]
        assert spec["prompt_tokens"] == data["metadata"]["prompt_tokens"]
    assert int(row["requests"]) == 100
    rate = group["summary"]["completion_tokens_per_second"]
    assert math.isclose(rate, 12800 / group["summary"]["wall_time_seconds"])
    assert math.isclose(float(row["speedup_vs_greedy"]), rate / greedy_rate)
    if method != "greedy":
        raw = rows_from_log(root / f"{method}-{link}" / "rounds.jsonl", requests, method, link, SPECULATORS)
        assert raw == requests
        for req in raw:
            delay = req["modeled_link_ms"]
            wait = req["observed_wait_ms"]
            assert wait >= delay * 0.99
        ideal = all_results[method, "ideal"]["records"]
        # Matched-trajectory projection uses ideal event sizes, avoiding
        # extrapolating a mismatched cell or attributing timing drift to wire delay.
        ideal_delays = []
        profile = LatencyProfile(**profile_args(link))
        width = SPECULATORS[method]["activation_bytes_per_position"]
        for req in ideal:
            modeled = [account_round(
                profile, activation_bytes=n * width, positions=n, requests=1, draft_tokens=k
            )["total_ms"] for n, k, _, _ in req["round_shape"]]
            ideal_delays.append((modeled[0], sum(modeled)))
        for bucket in ("short", "long"):
            pairs = [(a,b,c,d) for a,b,(c,d),item in zip(requests, ideal, ideal_delays, workload)
                     if item["metadata"]["group"] == bucket]
            assert len(pairs) == 50
            strata.append({
                "method": method, "link": link, "prompt_group": bucket, "requests": len(pairs),
                "matched_output_and_round_shape": sum(
                    a["token_ids"] == b["token_ids"] and a["round_shape"] == b["round_shape"]
                    for a,b,_,_ in pairs
                ),
                "measured_paired_ttft_delta_median_ms": statistics.median(
                    a["ttft_ms"] - b["ttft_ms"] for a,b,_,_ in pairs
                ),
                "measured_paired_latency_delta_median_ms": statistics.median(
                    a["latency_ms"] - b["latency_ms"] for a,b,_,_ in pairs
                ),
                "modeled_first_round_p50_ms": statistics.median(c for _,_,c,_ in pairs),
                "modeled_link_ms_per_request": statistics.mean(d for _,_,_,d in pairs),
                "modeled_penalty_pct_of_ideal_latency": 100 * sum(d for _,_,_,d in pairs)
                / sum(b["latency_ms"] for _,b,_,_ in pairs),
            })
    acceptance = f"{100*float(row['acceptance_rate']):.1f}%" if row["acceptance_rate"] else "—"
    lines.append(
        f"| {method} | {link} | {rate:.2f} | {rate/greedy_rate:.2f}× | "
        f"{float(row['ttft_p50_ms']):.1f} | {float(row['tpot_p50_ms']):.2f} | {acceptance} | "
        f"{float(row['modeled_link_ms_per_request']):.3f} | {float(row['observed_wait_ms_per_request']):.3f} | "
        f"{row['exact_outputs_vs_greedy']}/100 | {row['exact_outputs_and_round_shapes_vs_ideal']}/100 |"
    )
lines.extend(["", "## Modeled cost by prompt length", "",
              "| Method | Link | Short first round p50 (ms) | Long first round p50 (ms) | Short link ms/request | Long link ms/request |",
              "| --- | --- | ---: | ---: | ---: | ---: |"])
for short, long in zip(strata[::2], strata[1::2]):
    lines.append(f"| {short['method']} | {short['link']} | "
                 f"{short['modeled_first_round_p50_ms']:.3f} | {long['modeled_first_round_p50_ms']:.3f} | "
                 f"{short['modeled_link_ms_per_request']:.3f} | {long['modeled_link_ms_per_request']:.3f} |")
lines.extend(["", "The modeled link values come from actual draft-round tensor dimensions and the "
              "configured transport equation. This is a single pass per link. Compare the measured "
              "wait and paired projection to the observed throughput spread when assessing a "
              "sub-millisecond-per-round effect.", ""])
(root / "native-report.md").write_text("\n".join(lines))
with (root / "native-strata.csv").open("w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(strata[0]))
    writer.writeheader()
    writer.writerows(strata)
print(f"PASS: audited {len(expected)} cells on {len(ids)} prompts")
