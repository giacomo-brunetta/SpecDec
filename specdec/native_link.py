"""Charge a serial link for the actual tensors in a native drafting round."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from .latency import LatencyProfile
from .replay import wait_until_deadline

ENV = "SPECDEC_NATIVE_LINK"


@lru_cache(maxsize=1)
def load_link(settings: str) -> tuple[LatencyProfile, Path]:
    config = json.loads(settings)
    return LatencyProfile(**config["profile"]), Path(config["log"])


def account_round(
    profile: LatencyProfile,
    *,
    activation_bytes: int,
    positions: int,
    requests: int,
    draft_tokens: int,
) -> dict:
    # Two int32 fields per position (ID and position), and two per
    # request (anchor ID and accepted boundary). Reply contains token IDs.
    upload = activation_bytes + 4 * (2 * positions + 2 * requests)
    return {
        "positions": positions,
        "activation_bytes": activation_bytes,
        "draft_tokens": draft_tokens,
        **profile.transfer(upload, 4 * draft_tokens),
    }


def finish_round(
    settings: str,
    *,
    activation_bytes: int,
    positions: int,
    requests: int,
    draft_tokens: int,
    verified_tokens: int,
    sampled_count: Callable[[], int],
    synchronize: Callable[[], None],
    log_rank: bool,
) -> None:
    if requests != 1:
        raise ValueError("The native link experiment requires one request per round")
    profile, path = load_link(settings)
    round_stats = account_round(
        profile, activation_bytes=activation_bytes, positions=positions,
        requests=requests, draft_tokens=draft_tokens,
    )
    # Wait on every TP rank after device work, so wire delay is fully serial.
    # The zero-delay control executes the same synchronization.
    synchronize()
    waited_ms = 0.0
    if round_stats["total_ms"]:
        start = time.perf_counter()
        wait_until_deadline(round_stats["total_ms"] / 1000)
        waited_ms = 1000 * (time.perf_counter() - start)
    if log_rank:
        accepted = sampled_count() - 1 if verified_tokens else 0
        if not 0 <= accepted <= verified_tokens:
            raise ValueError("Native sampler accepted an invalid number of draft tokens")
        with path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({
                **round_stats, "wait_ms": waited_ms,
                "verified_tokens": verified_tokens, "accepted_tokens": accepted,
            }) + "\n")
