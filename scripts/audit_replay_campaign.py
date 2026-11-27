"""Independently audit the real benchmark checkpoints and final summary."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import sys

from specdec.campaign_report import modeled_link_ms, read_requests
from specdec.campaign_sweep import audit_replay, cells, latency_profile
from specdec.replay import load_trace

root = Path(sys.argv[1])
require_all = "--complete" in sys.argv

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

gold = read_requests(root / "baseline.json")
workload_file = root / "workload.jsonl"
workload = [json.loads(line) for line in workload_file.read_text().splitlines()]
manifest = read(root / "manifest.json")
grid = read(root / "grid.json")
params = grid["active_params_billions"]
assert [row["id"] for row in workload] == [row["request_id"] for row in gold]
assert hashlib.sha256(workload_file.read_bytes()).hexdigest() == manifest["dataset_sha256"]
assert hashlib.sha256((root / "plan.json").read_bytes()).hexdigest() == grid["plan_sha256"]
assert (manifest["batch_invariant"], manifest["concurrency"], manifest["k"], manifest["max_tokens"]) == (True, 1, 4, 128)
assert [list(cell) for cell in cells(params)] == grid["order"]
assert collections.Counter(row["metadata"]["group"] for row in gold) == {"short": 50, "long": 50}
assert len({tuple(row["prompt_token_ids"]) for row in gold}) == 100
summary = read(root / "baseline-summary.json")
assert (summary["completion_tokens"], summary["successful_requests"], summary["failed_requests"]) == (12800, 100, 0)
greedy_rate = summary["completion_tokens_per_second"]
configured = {"-".join(cell): cell for cell in cells(params)}
finished = {path.parent.name for path in (root / "cells").glob("*/summary.json")}
assert finished <= configured.keys()
if require_all:
    assert finished == configured.keys()

traces = {}
for model in ("oracle", *params):
    path = root / "traces" / model / "trace.jsonl"
    with path.open(encoding="utf-8") as file:
        traces[model] = load_trace(file)
    assert len(traces[model]) == 12800

matched = misses = counted = 0
rates = {("greedy", "", ""): greedy_rate}
for name in sorted(finished):
    model, link, compute = configured[name]
    path = root / "cells" / name
    config = read(path / "replay.json")
    summary = read(path / "summary.json")
    assert config["latency"] == latency_profile(model, link, compute, params)
    assert Path(config["trace"]).resolve() == (root / "traces" / model / "trace.jsonl").resolve()
    assert (summary["completion_tokens"], summary["successful_requests"], summary["failed_requests"]) == (12800, 100, 0)
    rows = read_requests(path / "requests.jsonl", jsonl=True)
    assert all(sum(c["tokens"] for c in row["token_chunks"]) == 128 for row in rows)
    audit = audit_replay(gold, rows, traces[model], path / "steps.jsonl")
    assert audit == read(path / "audit.json")
    assert all(math.isclose(r["modeled_link_ms"], modeled_link_ms(r, link), rel_tol=1e-10,
                            abs_tol=1e-9) for r in audit)
    counted += len(audit)
    misses += sum(row["trace_misses"] for row in audit)
    matched += sum(row["identical_output"] and not row["trace_misses"]
                   and row["spec_metric_parity_error"] is None for row in audit)
    rates[model, link, compute] = summary["completion_tokens_per_second"]

print(f"Verified the workload hash, plan, grid, and all {len(traces)} 12,800-prefix traces. "
      f"Raw-log audits cover {len(finished)} completed replay cells ({counted} requests). "
      f"{matched} matched output and metrics. {misses} prefix misses.")
if require_all:
    with (root / "sweep.csv").open(encoding="utf-8", newline="") as file:
        report = list(csv.DictReader(file))
    assert len(report) == len(rates)
    index = {(r["model"], r["link"], r["compute"]): r for r in report}
    assert index.keys() == rates.keys()
    assert all(math.isclose(float(index[k]["tokens_per_s"]), value) and
               math.isclose(float(index[k]["speedup_all"]), value / greedy_rate)
               for k, value in rates.items())
    print("Verified the report CSV against all measured throughputs and its greedy baseline.")
