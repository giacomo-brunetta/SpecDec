"""Blocking accelerator timing for one batched proposal."""


from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence


@dataclass(frozen=True)
class Proposal:
    committed_tokens: int
    draft_tokens: int
    prefix_tokens: int
    initial: bool = False


@dataclass(frozen=True)
class LatencyProfile:
    round_trip_ms: float = 0
    first_token_ms: float = 0
    tbt_ms: float = 0
    upload_bytes_per_second: float | None = None
    download_bytes_per_second: float | None = None
    token_bytes: int = 4
    upload_prompt_tokens: bool = False
    hidden_bytes_per_committed_token: int = 0
    kv_bytes_per_position: int = 0
    kv_transfer: Literal["none", "initial", "per_round"] = "none"
    kv_scope: Literal["prefix", "delta"] = "prefix"
    kv_direction: Literal["upload", "download"] = "upload"

    def __post_init__(self):
        for name in ("round_trip_ms", "first_token_ms", "tbt_ms"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("upload_bytes_per_second", "download_bytes_per_second"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("token_bytes", "hidden_bytes_per_committed_token", "kv_bytes_per_position"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, choices in (
            ("kv_transfer", {"none", "initial", "per_round"}),
            ("kv_scope", {"prefix", "delta"}),
            ("kv_direction", {"upload", "download"}),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"Invalid {name}: expected one of {sorted(choices)}")

    def estimate(self, batch: Sequence[Proposal]) -> dict[str, float | int]:
        committed = sum(p.committed_tokens for p in batch)
        upload = committed * (self.token_bytes + self.hidden_bytes_per_committed_token)
        if self.upload_prompt_tokens:
            upload += self.token_bytes * sum(
                p.prefix_tokens - p.committed_tokens for p in batch if p.initial
            )
        download = sum(p.draft_tokens * self.token_bytes for p in batch)
        kv_bytes = sum(
            self.kv_bytes_per_position * (
                p.prefix_tokens if self.kv_scope == "prefix" else p.committed_tokens
            )
            for p in batch
            if self.kv_transfer == "per_round"
            or (self.kv_transfer == "initial" and p.initial)
        )
        if self.kv_direction == "upload":
            upload += kv_bytes
        else:
            download += kv_bytes
        steps = max((p.draft_tokens for p in batch), default=0)
        upload_ms = (
            1000 * upload / self.upload_bytes_per_second if self.upload_bytes_per_second else 0
        )
        download_ms = (
            1000 * download / self.download_bytes_per_second if self.download_bytes_per_second else 0
        )
        round_trip = self.round_trip_ms if batch else 0
        compute = self.first_token_ms + (steps - 1) * self.tbt_ms if steps else 0
        return {
            "upload_bytes": upload,
            "download_bytes": download,
            "kv_bytes": kv_bytes,
            "round_trip_ms": round_trip,
            "upload_ms": upload_ms,
            "download_ms": download_ms,
            "first_token_ms": self.first_token_ms if steps else 0,
            "tbt_ms": (steps - 1) * self.tbt_ms if steps else 0,
            "total_ms": round_trip + upload_ms + download_ms + compute,
        }
