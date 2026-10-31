"""Run a JSONL chat workload against a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            if not isinstance(row.get("messages"), list) and not isinstance(
                row.get("prompt"), str
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected a messages array or string prompt"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No requests found in {path}")
    return rows


def request_body(
    row: dict[str, Any], args: argparse.Namespace, model: str
) -> dict[str, Any]:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    else:
        messages = [{"role": "user", "content": row["prompt"]}]
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": 1,
    }
    if args.seed is not None:
        body["seed"] = args.seed
    if args.extra_body:
        body.update(json.loads(args.extra_body))
    return body


def post_chat(
    endpoint: str,
    body: dict[str, Any],
    timeout: float,
    api_key: str | None,
    index: int,
    dataset_row: dict[str, Any],
) -> dict[str, Any]:
    started_wall = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=encoded, headers=headers, method="POST")
    record: dict[str, Any] = {
        "index": index,
        "request_id": dataset_row.get("id", f"request-{index:06d}"),
        "metadata": dataset_row.get("metadata"),
        "started_at": started_wall,
        "status": "error",
        "latency_ms": None,
        "http_status": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "speculative_decoding": None,
        "error": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            record["http_status"] = response.status
        record["status"] = "ok"
        usage = payload.get("usage") or {}
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")
        record["total_tokens"] = usage.get("total_tokens")
        metrics = payload.get("metrics") or {}
        record["speculative_decoding"] = metrics.get("speculative_decoding")
        choices = payload.get("choices") or []
        record["finish_reason"] = choices[0].get("finish_reason") if choices else None
        record["response_id"] = payload.get("id")
    except urllib.error.HTTPError as exc:
        record["http_status"] = exc.code
        try:
            record["error"] = exc.read().decode("utf-8", errors="replace")[:4000]
        except Exception:
            record["error"] = str(exc)
    except Exception as exc:  # preserve failures per request and finish the workload
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_ms"] = (time.perf_counter() - start) * 1000.0
    return record


def scrape_prometheus(url: str | None, timeout: float) -> dict[str, float] | None:
    if not url:
        return None
    try:
        from prometheus_client.parser import text_string_to_metric_families

        request = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
        values: dict[str, float] = {}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                if not sample.name.startswith("vllm:spec_decode_"):
                    continue
                labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
                key = f"{sample.name}{{{labels}}}"
                values[key] = float(sample.value)
        return values
    except Exception as exc:
        return {"__scrape_error__": str(exc)}


def metric_delta(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    if "__scrape_error__" in before or "__scrape_error__" in after:
        return {"error": before.get("__scrape_error__") or after.get("__scrape_error__")}
    keys = sorted(set(before) | set(after))
    deltas: dict[str, float] = {}
    for key in keys:
        start = before.get(key, 0.0)
        end = after.get(key, 0.0)
        # Counters can reset if the server restarted during the run.
        deltas[key] = end - start if end >= start else end
    return deltas


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


def summarize(records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successful = [r for r in records if r["status"] == "ok"]
    latencies = [float(r["latency_ms"]) for r in successful if r["latency_ms"] is not None]
    prompt_tokens = [r["prompt_tokens"] for r in successful if isinstance(r["prompt_tokens"], int)]
    completion_tokens = [
        r["completion_tokens"]
        for r in successful
        if isinstance(r["completion_tokens"], int)
    ]
    spec = [
        r["speculative_decoding"]
        for r in successful
        if isinstance(r.get("speculative_decoding"), dict)
    ]
    accepted = sum(int(x.get("num_accepted_draft_tokens", 0)) for x in spec)
    drafted = sum(int(x.get("num_draft_tokens", 0)) for x in spec)
    steps = sum(int(x.get("num_spec_steps", 0)) for x in spec)
    accepted_per_step = accepted / steps if steps else None
    acceptance: dict[str, Any] = {
        "requests_with_metrics": len(spec),
        "requests_without_metrics": len(successful) - len(spec),
        "num_accepted_draft_tokens": accepted,
        "num_draft_tokens": drafted,
        "num_spec_steps": steps,
        "draft_acceptance_rate": accepted / drafted if drafted else None,
        "mean_accepted_draft_tokens_per_step": accepted_per_step,
        "mean_acceptance_length_including_bonus": 1 + accepted_per_step
        if accepted_per_step is not None
        else None,
    }
    if spec:
        max_hist_len = max(len(x.get("acceptance_histogram", [])) for x in spec)
        hist = [0] * max_hist_len
        for item in spec:
            for i, count in enumerate(item.get("acceptance_histogram", [])):
                hist[i] += int(count)
        acceptance["acceptance_histogram"] = hist
    return {
        "request_count": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "wall_time_seconds": wall_seconds,
        "requests_per_second": len(successful) / wall_seconds if wall_seconds > 0 else None,
        "prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "completion_tokens_per_second": sum(completion_tokens) / wall_seconds
        if completion_tokens and wall_seconds > 0
        else None,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
        },
        "speculative_decoding": acceptance,
    }


def make_run_dir(root: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    path = root / f"{stamp}_{safe_label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def infer_metrics_url(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    prefix = parsed.path.removesuffix("/v1/chat/completions").rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, f"{prefix}/metrics", "", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL prompt workload")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True, help="Served model name in API requests")
    parser.add_argument("--method", required=True, help="Run label, e.g. baseline, eagle3, mtp")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full dataset this many times")
    parser.add_argument("--max-requests", type=int, help="Optional cap after repeating the dataset")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-key-env", default="SPECDEC_API_KEY")
    parser.add_argument("--extra-body", help="JSON object merged into each API request")
    parser.add_argument("--metrics-url", help="Defaults to endpoint's /metrics URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.repeat < 1 or args.warmup_requests < 0:
        raise SystemExit("concurrency/repeat must be positive and warmup-requests nonnegative")
    if args.max_tokens < 1:
        raise SystemExit("max-tokens must be positive")
    if args.extra_body:
        try:
            value = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--extra-body must be a JSON object: {exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit("--extra-body must be a JSON object")

    dataset = read_jsonl(args.dataset)
    work = (dataset * args.repeat)[: args.max_requests]
    api_key = os.environ.get(args.api_key_env)
    url = args.metrics_url or infer_metrics_url(args.endpoint)

    # Warmup is intentionally excluded from request and Prometheus measurements.
    warmups = [dataset[i % len(dataset)] for i in range(args.warmup_requests)]
    for i, row in enumerate(warmups):
        result = post_chat(
            args.endpoint,
            request_body(row, args, args.model),
            args.timeout,
            api_key,
            -(i + 1),
            row,
        )
        if result["status"] != "ok":
            raise SystemExit(f"Warmup request failed: {result['error']}")

    metrics_before = scrape_prometheus(url, args.timeout)
    wall_start = time.perf_counter()
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                post_chat,
                args.endpoint,
                request_body(row, args, args.model),
                args.timeout,
                api_key,
                index,
                row,
            ): index
            for index, row in enumerate(work)
        }
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    wall_seconds = time.perf_counter() - wall_start
    records.sort(key=lambda r: r["index"])
    metrics_after = scrape_prometheus(url, args.timeout)

    run_dir = make_run_dir(args.output_root, args.method)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "model": args.model,
        "endpoint": args.endpoint,
        "dataset": str(args.dataset.resolve()),
        "dataset_rows": len(dataset),
        "request_count": len(work),
        "warmup_requests": args.warmup_requests,
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "extra_body": json.loads(args.extra_body) if args.extra_body else {},
        "prometheus_metrics_url": url,
    }
    summary = summarize(records, wall_seconds)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "before": metrics_before,
                "after": metrics_after,
                "delta": metric_delta(metrics_before, metrics_after),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 1 if summary["failed_requests"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
