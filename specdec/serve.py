"""Launch a vLLM server for one scenario in a JSON plan."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read plan {args.plan}: {exc}") from exc
    server: dict[str, Any] = plan.get("server", {})
    scenario: dict[str, Any] | None = plan.get("scenarios", {}).get(args.scenario)
    if scenario is None:
        raise SystemExit(f"Scenario {args.scenario!r} is not in {args.plan}")
    if not scenario.get("enabled", False):
        raise SystemExit(f"Scenario {args.scenario!r} is disabled in {args.plan}")
    executable = shutil.which("vllm")
    if executable is None:
        raise SystemExit("vllm executable not found; activate the intended environment first")
    model = server.get("model")
    if not model:
        raise SystemExit("server.model is required")

    command = [executable, "serve", str(model)]
    options = (
        ("served_model_name", "--served-model-name"),
        ("dtype", "--dtype"),
        ("tensor_parallel_size", "--tensor-parallel-size"),
        ("pipeline_parallel_size", "--pipeline-parallel-size"),
        ("max_model_len", "--max-model-len"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
        ("host", "--host"),
        ("port", "--port"),
    )
    for key, flag in options:
        if server.get(key) is not None:
            command.extend((flag, str(server[key])))

    metrics_level = scenario.get("per_request_spec_decode_metrics", "none")
    if metrics_level not in {"none", "summary", "detailed"}:
        raise SystemExit("per_request_spec_decode_metrics must be none, summary, or detailed")
    if metrics_level != "none":
        command.extend(("--per-request-spec-decode-metrics", metrics_level))

    speculative_config = scenario.get("speculative_config")
    if speculative_config is not None:
        command.extend(("--speculative-config", json.dumps(speculative_config, separators=(",", ":"))))
    command.extend(str(item) for item in server.get("extra_args", []))
    command.extend(str(item) for item in scenario.get("extra_args", []))
    print("Starting:", " ".join(command), flush=True)
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
