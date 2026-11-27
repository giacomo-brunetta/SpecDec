"""Project tree verification time from measured causal-extension latency deltas."""

import json
import statistics
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache

from ..campaign_sweep import LINKS, RATES, latency_profile
from ..latency import LatencyProfile
from .acceptance import nodes, rounds
from .common import inputs, read_ranks, write_rows


def calibration_curve(samples):
    """Monotone least-squares fit of equally replicated size medians."""
    blocks = []
    for size, times in sorted(samples.items(), key=lambda pair: int(pair[0])):
        blocks.append(([int(size)], statistics.median(times)))
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            right, y = blocks.pop()
            left, x = blocks.pop()
            blocks.append(
                (
                    left + right,
                    (len(left) * x + len(right) * y) / (len(left) + len(right)),
                )
            )
    return {size: mean for sizes, mean in blocks for size in sizes}


def project(run, output, depths):
    _, models, records, fingerprint = inputs(run)
    params = {name: spec["active_params_b"] for name, spec in models.items()}
    ranks = {
        name: read_ranks(output, name, spec["model"], fingerprint, records)
        for name, spec in models.items()
    }
    oracle_path = run / "cells/oracle-ideal-ideal/requests.jsonl"
    oracle = [json.loads(line) for line in oracle_path.read_text().splitlines()]
    if [r["request_id"] for r in oracle] != [r["request_id"] for r in records]:
        raise ValueError("Oracle and greedy prompt order differ")
    if any(r["token_ids"] != o["token_ids"] for r, o in zip(records, oracle)):
        raise ValueError("Oracle differs from greedy trajectory")
    calibration = json.loads((output / "calibration.json").read_text())
    curves = {
        group: calibration_curve(calibration[prefix])
        for group, prefix in [("short", "128"), ("long", "640")]
    }
    max_nodes = min(max(curve) for curve in curves.values()) - 1

    @lru_cache(None)
    def delta(group, size):
        curve = curves[group]
        sizes = sorted(curve)
        if size in curve:
            return curve[size] - curve[5]
        hi = bisect_left(sizes, size)
        if not 0 < hi < len(sizes):
            raise ValueError(f"Size {size} outside calibration")
        a, b = sizes[hi - 1 : hi + 1]
        return curve[a] + (curve[b] - curve[a]) * (size - a) / (b - a) - curve[5]

    refs = []
    for rec in oracle:
        chunks = rec["token_chunks"]
        ref = statistics.mean(
            b["elapsed_ms"] - a["elapsed_ms"]
            for a, b in zip(chunks, chunks[1:])
            if b["tokens"] == 5
        )
        fixed = rec["ttft_ms"] + max(0, rec["latency_ms"] - chunks[-1]["elapsed_ms"])
        refs.append((ref, fixed))

    def totals(model, depth, switches):
        total = defaultdict(float)
        for i, rec in enumerate(records):
            tokens = rec["token_ids"]
            seq = [1] * len(tokens) if model == "oracle" else ranks[model][i]
            group = rec["metadata"]["group"]
            ref, fixed = refs[i]
            total["target"] += fixed
            last = pos = 1
            for offset, limit, accepted in rounds(seq, depth, switches):
                count = nodes(limit, switches)
                total["target"] += ref + delta(group, count + 1)
                total["rounds"] += 1
                total["accepted"] += accepted
                total["nodes"] += count
                total["upload"] += 4 * (
                    last + (len(rec["prompt_token_ids"]) if offset == 1 else 0)
                )
                total["download"] += 4 * count
                last = accepted + 1
                pos = offset + last
            if pos < len(tokens):
                total["target"] += ref + delta(group, 1)
        return total

    greedy = json.loads((run / "baseline-summary.json").read_text())[
        "completion_tokens_per_second"
    ]
    token_count = sum(len(r["token_ids"]) for r in records)
    rows = []
    for depth in depths:
        for model in ["oracle", *models]:
            choices = (
                [0]
                if model == "oracle"
                else sorted({0, 1, min(2, depth), min(3, depth), depth})
            )
            for s in choices:
                if nodes(depth, s) > max_nodes:
                    continue
                t = totals(model, depth, s)
                for link in LINKS:
                    for compute in ["ideal"] if model == "oracle" else RATES:
                        profile = LatencyProfile(
                            **latency_profile(model, link, compute, params=params)
                        )
                        transfer = profile.transfer(
                            t["upload"], t["download"], active=False
                        )
                        link_ms = (
                            transfer["total_ms"] + t["rounds"] * profile.round_trip_ms
                        )
                        draft_ms = t["nodes"] * profile.tbt_ms
                        rate = 1000 * token_count / (t["target"] + draft_ms + link_ms)
                        rows.append(
                            dict(
                                model=model,
                                depth=depth,
                                second_choices=s,
                                max_nodes=nodes(depth, s),
                                link=link,
                                compute=compute,
                                effective_tflops=RATES[compute],
                                rounds=int(t["rounds"]),
                                accepted_per_round=t["accepted"] / t["rounds"],
                                mean_nodes_per_round=t["nodes"] / t["rounds"],
                                target_ms_per_request=t["target"] / len(records),
                                draft_ms_per_request=draft_ms / len(records),
                                link_ms_per_request=link_ms / len(records),
                                tokens_per_second=rate,
                                speedup=rate / greedy,
                            )
                        )
    write_rows(output / "distance.csv", rows)
