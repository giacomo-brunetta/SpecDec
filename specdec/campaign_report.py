"""Validate a finished SpecDec campaign and write its Markdown report and CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from .campaign import K, MAX_TOKENS
from .campaign_sweep import LINKS, RATES, cells, latency_profile
from .experiment import expected_steps
from .replay import load_trace


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_requests(path, *, jsonl=False):
    records = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
               if jsonl else read_json(path))
    if (len(records) != 100 or len({row["request_id"] for row in records}) != 100
            or any(row["status"] != "ok" or len(row["token_ids"]) != MAX_TOKENS
                   or row["completion_tokens"] != MAX_TOKENS for row in records)):
        raise ValueError(f"Expected 100 complete {MAX_TOKENS}-token requests in {path}")
    return records


def paired_speedup(baseline, replay, ids):
    if ids:
        return (sum(baseline[key]["latency_ms"] for key in ids)
                / sum(replay[key]["latency_ms"] for key in ids))
    return None


def timing(records):
    rows = list(records)
    return {
        "ttft_p50_ms": statistics.median(row["ttft_ms"] for row in rows),
        "tpot_p50_ms": statistics.median(
            sum(row["inter_chunk_ms"]) / (MAX_TOKENS - 1) for row in rows),
        "request_p50_ms": statistics.median(row["latency_ms"] for row in rows),
    }


def elapsed_at_token(row, output_length):
    count = 0
    for chunk in row["token_chunks"]:
        count += chunk["tokens"]
        if count >= output_length:
            return chunk["elapsed_ms"]
    raise ValueError(f"Incomplete stream timing for {row['request_id']}")


def prefix_speedup(baseline, replay, lengths, ids):
    if not ids:
        return None
    return (sum(elapsed_at_token(baseline[key], lengths[key]) for key in ids)
            / sum(elapsed_at_token(replay[key], lengths[key]) for key in ids))


def serialization_ms(steps, link):
    if link == "ideal":
        return 0
    config = LINKS[link]
    return 1000 * (steps["upload_bytes"] / config["up"]
                   + steps["download_bytes"] / config["down"])


def modeled_link_ms(steps, link):
    return (steps["proposals"] * LINKS[link].get("round_trip_ms", 0)
            + serialization_ms(steps, link))


def project_speedup(gold, ideal, ideal_audits, matched, cell, active_params):
    if not matched:
        return None
    model, link, compute = cell
    rate = RATES[compute]
    link_ms = sum(modeled_link_ms(row, link) for row in ideal_audits
                  if row["request_id"] in matched)
    compute_ms = 0
    if rate:
        drafted = sum(ideal[key]["speculative_decoding"]["num_draft_tokens"] for key in matched)
        compute_ms = 2 * active_params[model] * drafted / rate
    greedy_ms = sum(gold[key]["latency_ms"] for key in matched)
    ideal_ms = sum(ideal[key]["latency_ms"] for key in matched)
    return greedy_ms / (ideal_ms + link_ms + compute_ms)


def relative_span_pct(values):
    return 100 * (max(values) - min(values)) / statistics.median(values)


def report(output: Path) -> None:
    baseline = read_requests(output / "baseline.json")
    gold = {row["request_id"]: row for row in baseline}
    config_path = output / "target-generation-config.json"
    if config_path.exists():
        eos_ids = read_json(config_path).get("eos_token_id", [])
    else:
        eos_ids = []
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    natural_lengths = {
        key: next((i for i, token in enumerate(row["token_ids"], 1) if token in eos_ids),
                  MAX_TOKENS)
        for key, row in gold.items()
    }
    baseline_stats = read_json(output / "baseline-summary.json")
    if (baseline_stats["successful_requests"] != len(baseline)
            or baseline_stats["failed_requests"]
            or baseline_stats["completion_tokens"] != len(baseline) * MAX_TOKENS):
        raise ValueError("Greedy request records differ from the baseline summary")
    greedy_rate = baseline_stats["completion_tokens_per_second"]
    params = read_json(output / "grid.json")
    active_params = params["active_params_billions"]
    captures = {model: read_json(output / "traces" / model / "capture.json")
                for model in (*active_params, "oracle")}
    if any(capture["prefixes"] < 100 for capture in captures.values()):
        raise ValueError("A trace has too few unique prefixes for the 100-prompt workload")
    workload = read_json(output / "manifest.json")
    workload_path = output / "workload.jsonl"
    if workload_path.exists():
        workload_hash = hashlib.sha256(workload_path.read_bytes()).hexdigest()
        if workload_hash != workload["dataset_sha256"]:
            raise ValueError("Saved workload has changed since the baseline")
    if params["order"] != [list(cell) for cell in cells(active_params)]:
        raise ValueError("Saved grid differs from report configuration")
    if sum(row["metadata"]["group"] == "short" for row in baseline) != 50 or sum(
            row["metadata"]["group"] == "long" for row in baseline) != 50:
        raise ValueError("Expected two 50-prompt strata")

    sweep = {}
    common = {model: set(gold) for model in (*active_params, "oracle")}
    for cell in cells(active_params):
        model, link, compute = cell
        root = output / "cells" / "-".join(cell)
        stats = read_json(root / "summary.json")
        config = read_json(root / "replay.json")
        if config["latency"] != latency_profile(*cell, params=active_params):
            raise ValueError(f"Latency configuration differs for {root}")
        rows = read_requests(root / "requests.jsonl", jsonl=True)
        audits = read_json(root / "audit.json")
        if ([row["request_id"] for row in rows] != list(gold)
                or [row["request_id"] for row in audits] != list(gold)
                or stats["completion_tokens"] != 100 * MAX_TOKENS
                or stats["successful_requests"] != 100 or stats["failed_requests"]):
            raise ValueError(f"Request set or summary differs for {root}")
        same = set()
        for row, audit in zip(rows, audits):
            key = row["request_id"]
            exact = gold[key]["token_ids"] == row["token_ids"]
            if exact != audit["identical_output"]:
                raise ValueError(f"Output audit differs for {key} in {root}")
            if exact and audit["trace_misses"] == 0 and audit["spec_metric_parity_error"] is None:
                same.add(key)
        common[model] &= same
        sweep[cell] = (stats, {row["request_id"]: row for row in rows}, audits, same)

    metrics = [{
        "model": "greedy", "link": "", "compute": "", "effective_tflops": "",
        "tokens_per_s": greedy_rate, "speedup_all": 1.0, "matched_prompts": 100,
        "common_matched_prompts": 100, "speedup_common": 1.0,
        "projected_speedup_common": 1.0, "speedup_short_common": 1.0,
        "speedup_long_common": 1.0, "speedup_to_first_eos_common": 1.0 if eos_ids else None,
        **timing(baseline), "draft_acceptance_pct": None,
        "prefix_misses": 0, "prefix_lookups": 0, "mean_link_ms_per_request": 0,
        "mean_serialization_us_per_request": 0,
        "mean_draft_ms_per_request": 0, "mean_wait_ms_per_request": 0,
    }]
    for cell, (stats, requests, audits, matched) in sweep.items():
        model, link, compute = cell
        stable = sorted(common[model])
        _, ideal, ideal_audits, _ = sweep[model, "ideal", "ideal"]
        acceptance = stats["speculative_decoding"]["draft_acceptance_rate"]
        rate = stats["completion_tokens_per_second"]
        metrics.append({
            "model": model, "link": link, "compute": compute,
            "effective_tflops": RATES[compute], "tokens_per_s": rate, "speedup_all": rate / greedy_rate,
            "matched_prompts": len(matched), "common_matched_prompts": len(stable),
            "speedup_common": paired_speedup(gold, requests, stable),
            "projected_speedup_common": project_speedup(gold, ideal, ideal_audits, stable, cell, active_params),
            "speedup_short_common": paired_speedup(gold, requests, [
                key for key in stable if gold[key]["metadata"]["group"] == "short"]),
            "speedup_long_common": paired_speedup(gold, requests, [
                key for key in stable if gold[key]["metadata"]["group"] == "long"]),
            "speedup_to_first_eos_common": (
                prefix_speedup(gold, requests, natural_lengths, stable) if eos_ids else None
            ),
            **timing(requests.values()),
            "draft_acceptance_pct": 100 * acceptance if acceptance is not None else None,
            "prefix_misses": sum(row["trace_misses"] for row in audits),
            "prefix_lookups": sum(row["logged_batches"] for row in audits),
            "mean_link_ms_per_request": statistics.mean(row["modeled_link_ms"] for row in audits),
            "mean_serialization_us_per_request": 1000 * statistics.mean(
                serialization_ms(row, link) for row in audits
            ),
            "mean_draft_ms_per_request": statistics.mean(row["modeled_draft_ms"] for row in audits),
            "mean_wait_ms_per_request": statistics.mean(row["observed_wait_ms"] for row in audits),
        })
    index = {(row["model"], row["link"], row["compute"]): row for row in metrics}
    with (output / "sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    best = max((row for row in metrics if row["model"] in active_params),
               key=lambda row: row["tokens_per_s"])
    lines = [
        "# Speculative decoding interconnect and compute sweep", "",
        f"**{len(metrics)} configurations**, each measured on 100 prompts with {MAX_TOKENS} output tokens, "
        "at concurrency 1. The greedy baseline ran at "
        f"**{greedy_rate:.2f} tok/s** with median TTFT of "
        f"{metrics[0]['ttft_p50_ms']:.1f} ms and median TPOT of {metrics[0]['tpot_p50_ms']:.2f} ms.",
        "",
        f"The highest measured non-oracle throughput was **{best['tokens_per_s']:.2f} tok/s** "
        f"({best['speedup_all']:.3f}× greedy) with {best['model']}, {best['compute']} compute "
        f"and the {best['link']} link. This is the fastest observed cell in this run.",
        "",
        "The input is an English first-turn sample from LMSYS-Chat-1M with 50 short and 50 long "
        "prompts. Prompts are paired across cells. Each cell has one untimed warm-up request. "
        "EOS was ignored to equalize decode lengths. Target and drafters use BF16. "
        f"The target is {workload['target']['model']}. "
        "The GPU verifies the recorded proposals while the proposer injects the configured delays.",
        "",
        *(
            ["vLLM batch-invariant kernels were enabled in the target baseline and verifier "
             "to reduce changes in greedy decisions from numerical differences between "
             "single-token decode and a multi-token verification step.", ""]
            if workload.get("batch_invariant") else []
        ),
        f"Workload SHA-256 is `{workload['dataset_sha256']}`.",
        "",
        "## Observed throughput relative to greedy", "",
        "Speedups below use all 100 prompts. Identical output and acceptance-metric "
        "checks are tracked separately in the matched-prompt columns.",
        "",
        "| Drafter | Compute tier | Ideal link | NVLink | PCIe | HDR100 | Matched across cells |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in (*active_params, "oracle"):
        for compute in (RATES if model != "oracle" else ["ideal"]):
            rates = " | ".join(f"{index[model, link, compute]['speedup_all']:.3f}×" for link in LINKS)
            lines.append(f"| {model} | {compute} | {rates} | {len(common[model])}/100 |")

    if eos_ids:
        early_stops = sum(length < MAX_TOKENS for length in natural_lengths.values())
        lines += [
            "",
            f"The greedy output hit a configured stop token before 128 tokens on "
            f"**{early_stops}/100** prompts. The CSV also compares elapsed time through "
            "the first stop token, or the final token for prompts with no earlier stop, "
            "using matched prompts. This uses the original output chunks from the fixed-length "
            "runs and is a sensitivity check for natural stopping.",
        ]
    lines += [
        "", "## Modeled compute break-even", "",
        "For each model, the projection uses total request-time headroom from the "
        "zero-compute, zero-link run on prompts that matched across every cell for that model. "
        "It subtracts the link time estimated from that run's proposal count and byte totals, "
        "then divides the total GFLOP work of the drafted tokens by remaining milliseconds. "
        "The answer is in effective TFLOP/s. The boundary depends on this target and workload.",
        "The CSV also includes projected speedups for the same matched prompts. These add the "
        "modeled link and compute costs to the measured ideal replay time for each cell.",
        "",
        "| Drafter | Common prompts | Ideal link | NVLink | PCIe | HDR100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in active_params:
        _, requests, audit, _ = sweep[model, "ideal", "ideal"]
        stable = sorted(common[model])
        audit = [row for row in audit if row["request_id"] in common[model]]
        headroom = sum(gold[key]["latency_ms"] - requests[key]["latency_ms"] for key in stable)
        drafted = sum(requests[key]["speculative_decoding"]["num_draft_tokens"] for key in stable)
        entries = []
        for link in LINKS:
            remaining = headroom - sum(modeled_link_ms(row, link) for row in audit)
            entries.append(f"{2 * active_params[model] * drafted / remaining:.4f}"
                           if remaining > 0 else "—")
        lines.append(f"| {model} | {len(stable)} | " + " | ".join(entries) + " |")

    lines += [
        "", "## Draft quality on the fixed greedy trajectories", "",
        "These counts use all 100 baseline outputs and each saved proposal trace.",
        "",
        "| Drafter | Proposed tokens | Accepted tokens | Acceptance | Committed tokens per verification including bonus |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in (*active_params, "oracle"):
        trace_path = output / "traces" / model / "trace.jsonl"
        with trace_path.open(encoding="utf-8") as stream:
            trace = load_trace(stream)
        rounds = [step for row in baseline for step in expected_steps(
            row["prompt_token_ids"], row["token_ids"], trace, K)]
        drafted_count = sum(len(step["draft_token_ids"]) for step in rounds)
        accepted_count = sum(len(step["accepted_token_ids"]) for step in rounds)
        lines.append(f"| {model} | {drafted_count} | {accepted_count} | "
                     f"{100 * accepted_count / drafted_count:.1f}% | "
                     f"{1 + accepted_count / len(rounds):.2f} |")
    lines += [
        "", "## Interconnect and replay fidelity", "",
        "| Drafter | HDR100 link ms/request at ideal compute | Of which serialization µs/request | "
        "Mean upload bytes/request | Mean download bytes/request | Prefix misses across cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in (*active_params, "oracle"):
        _, _, audits, _ = sweep[model, "hdr100", "ideal"]
        values = [statistics.mean(row[key] for row in audits)
                  for key in ("modeled_link_ms", "upload_bytes", "download_bytes")]
        misses = sum(row["prefix_misses"] for row in metrics if row["model"] == model)
        serialize = index[model, "hdr100", "ideal"]["mean_serialization_us_per_request"]
        lines.append(f"| {model} | {values[0]:.3f} | {serialize:.3f} | "
                     f"{values[1]:.0f} | {values[2]:.0f} | {misses} |")
    measured_spans = []
    projected_spans = []
    for model in (*active_params, "oracle"):
        if not common[model]:
            continue
        for compute in (RATES if model != "oracle" else ["ideal"]):
            rows = [index[model, link, compute] for link in LINKS]
            measured_spans.append(relative_span_pct([row["speedup_common"] for row in rows]))
            projected_spans.append(relative_span_pct([row["projected_speedup_common"] for row in rows]))
    if measured_spans:
        lines += [
            "",
            f"At fixed drafter and compute, the median observed spread across the four links "
            f"was **{statistics.median(measured_spans):.2f}%**. The median spread predicted "
            f"from the modeled link costs was **{statistics.median(projected_spans):.3f}%**. "
            "These spreads use matched prompts and divide the max–min speedup difference by "
            "the median speedup for the four links in each group.",
        ]
    lines += [
        "",
        "A matched prompt has identical token IDs, no trace misses, and the expected acceptance "
        "counts. If a verifier commits an off-trace token, the proposer falls back to target-only "
        "decoding until a trace hit. Per-request audits include the first differing offset and "
        "miss count. Matched-only speedups in the CSV use the common prompts for each model.",
        "",
        "The oracle proposes from the greedy target sequence with ideal compute. The model uses "
        "token IDs and a resident drafter KV cache. It charges a fixed round trip and byte "
        "serialization per nonempty proposal. The first-token compute charge equals steady TBT. "
        "Drafter prefill, extra catch-up compute, hidden-state traffic, and KV migration are "
        "outside this sweep. TPOT divides the first-to-last output-chunk interval by 127.",
        "",
        "Each configuration had one timing pass. The modeled interconnect costs are deterministic "
        "given proposals and payloads. Small differences in measured end-to-end throughput between "
        "physical link profiles also include run-to-run variation.",
        "",
        "The adjacent `sweep.csv` includes paired timings by input length, draft acceptance, "
        "trace coverage, modeled delay and observed wait. `grid.json`, `manifest.json`, "
        "`traces/`, and `cells/` record the inputs and auditable measurements.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'report.md'} and {output / 'sweep.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)
