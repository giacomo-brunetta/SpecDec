"""Trace-backed proposer for the pinned vLLM V1 custom_class hook."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .latency import LatencyProfile, Proposal


def wait_until_deadline(seconds: float) -> None:
    # Sleep through the bulk of longer waits, then spin to avoid scheduler
    # granularity dominating the tens-of-microseconds link profiles.
    deadline = time.perf_counter() + seconds
    spin_window = 0.0002
    remaining = deadline - time.perf_counter()
    if remaining > spin_window:
        time.sleep(remaining - spin_window)
    while time.perf_counter() < deadline:
        pass


@dataclass(frozen=True)
class TraceEntry:
    draft: tuple[int, ...]
    prompt_length: int


def load_trace(rows: Iterable[str]) -> dict[tuple[int, ...], TraceEntry]:
    trace: dict[tuple[int, ...], TraceEntry] = {}
    for number, line in enumerate(rows, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prefix = tuple(row["prefix_token_ids"])
        entry = TraceEntry(tuple(row["draft_token_ids"]), row["prompt_length"])
        if type(entry.prompt_length) is not int or not 0 < entry.prompt_length < len(prefix) or any(
            type(token) is not int or token < 0 for token in (*prefix, *entry.draft)
        ):
            raise ValueError(f"Invalid replay row {number}")
        if trace.setdefault(prefix, entry) != entry:
            raise ValueError(f"Conflicting proposals for the same prefix at row {number}")
    if not trace:
        raise ValueError("Replay trace is empty")
    return trace


class Replay:
    def __init__(
        self,
        trace: dict[tuple[int, ...], TraceEntry],
        k: int,
        profile: LatencyProfile,
        *,
        wait: Callable[[float], None] = time.sleep,
        log: Callable[[dict], None] | None = None,
        strict: bool = True,
    ):
        self.trace = trace
        self.k = k
        self.profile = profile
        self.wait = wait
        self.log = log
        self.strict = strict

    def propose(self, sampled, lengths, token_rows) -> list[list[int]]:
        started = time.perf_counter()
        drafts = []
        transfers = []
        steps = []
        missing_lengths = []
        for committed, length, tokens in zip(sampled, lengths, token_rows):
            if not committed:
                drafts.append([])
                continue
            # V1 runs this hook after bookkeeping has committed the sampled IDs.
            prefix = tuple(int(token) for token in tokens[:length])
            entry = self.trace.get(prefix)
            draft = list(entry.draft[:self.k]) if entry else []
            drafts.append(draft)
            if entry is None:
                missing_lengths.append(len(prefix))
            elif draft:
                initial = len(prefix) == entry.prompt_length + 1
                transfers.append(Proposal(len(committed), len(draft), len(prefix), initial))
            if self.log is not None:
                steps.append({
                    "prefix_length": len(prefix),
                    "committed_token_ids": list(committed),
                    "draft_token_ids": draft,
                    "miss": entry is None,
                })
        delay = self.profile.estimate(transfers)
        misses = len(missing_lengths)
        if misses and self.strict:
            if self.log is not None:
                self.log({"steps": steps, "misses": misses, "delay": delay, "wait_ms": 0})
            raise ValueError(f"Replay prefixes missing at lengths {missing_lengths}")
        wait_start = time.perf_counter()
        if delay["total_ms"]:
            self.wait(delay["total_ms"] / 1000)
        if self.log is not None and steps:
            self.log({
                "steps": steps,
                "arrival_ms": started * 1000,
                "proposals": len(transfers),
                "misses": misses,
                "delay": delay,
                "lookup_and_model_ms": (wait_start - started) * 1000,
                "wait_ms": (time.perf_counter() - wait_start) * 1000 if delay["total_ms"] else 0,
            })
        return drafts


class ReplayProposer:
    def __init__(self, vllm_config):
        config_path = Path(os.environ["SPECDEC_REPLAY_CONFIG"]).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.log_path = config_path.parent / config["log"] if config.get("log") else None
        with (config_path.parent / config["trace"]).open(encoding="utf-8") as rows:
            self.replay = Replay(
                load_trace(rows),
                vllm_config.speculative_config.num_speculative_tokens,
                LatencyProfile(**config.get("latency", {})),
                wait=wait_until_deadline if config.get("deadline_wait") else time.sleep,
                log=self.write_log if self.log_path else None,
                strict=config.get("strict", True),
            )

    def write_log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *, slot_mappings=None):
        return self.replay.propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
