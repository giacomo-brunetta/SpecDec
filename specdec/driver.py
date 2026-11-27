"""Start an isolated vLLM scenario and exercise its HTTP interface."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .benchmark import post_chat
from .serve import build_command


REPLAY_ENV = {"VLLM_USE_V2_MODEL_RUNNER": "0", "SPECDEC_SERIAL_PP": "1"}


class Driver:
    def __init__(self, endpoint: str, model: str, timeout: float = 180):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int, **options) -> dict:
        result = post_chat(
            self.endpoint,
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "return_token_ids": True,
                **options,
            },
            self.timeout,
            None,
            0,
            {},
        )
        if result["status"] != "ok":
            raise RuntimeError(f"Request failed: {result['error']}")
        return result

    def complete_tokens(self, prefix: list[int], max_tokens: int) -> list[int]:
        return self.complete_token_batch([prefix], max_tokens)[0]

    def complete_token_batch(self, prefixes: list[list[int]], max_tokens: int) -> list[list[int]]:
        body = {
            "model": self.model, "prompt": prefixes, "max_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0, "return_token_ids": True,
        }
        request = urllib.request.Request(
            self.endpoint.replace("/chat/completions", "/completions"),
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            choices = json.load(response)["choices"]
        choices.sort(key=lambda choice: choice["index"])
        if [choice["index"] for choice in choices] != list(range(len(prefixes))):
            raise RuntimeError("Drafter returned unexpected completion indices")
        proposals = [choice.get("token_ids") for choice in choices]
        if any(not isinstance(tokens, list) or len(tokens) != max_tokens for tokens in proposals):
            raise RuntimeError(f"Drafter did not return {max_tokens} token IDs per prefix")
        return proposals


@contextlib.contextmanager
def running_scenario(
    server: dict,
    scenario: dict,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    startup_timeout: float = 1200,
):
    host = "127.0.0.1"
    with socket.socket() as address:
        address.bind((host, 0))
        port = address.getsockname()[1]

    command = build_command({**server, "host": host, "port": port}, scenario)
    url = f"http://{host}:{port}"
    model = server.get("served_model_name") or server["model"]
    runtime_env = {**os.environ, **(env or {})}
    package_root = str(Path(__file__).resolve().parent.parent)
    runtime_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (package_root, runtime_env.get("PYTHONPATH")))
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, env=runtime_env, start_new_session=True,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {proc.returncode}. See {log_path}")
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise TimeoutError(f"vLLM startup timed out. See {log_path}")
            yield Driver(f"{url}/v1/chat/completions", model)
        finally:
            # A new session makes this group exclusive to the server we launched.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
