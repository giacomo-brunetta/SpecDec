import csv
import hashlib
import json
from pathlib import Path

from ..campaign_sweep import drafter_models
from ..serve import read_scenario


def inputs(run):
    plan = run / "plan.json"
    target, _ = read_scenario(plan, "baseline")
    models = drafter_models(plan, target)
    raw = (run / "baseline.json").read_bytes()
    records = json.loads(raw)
    return target, models, records, hashlib.sha256(raw).hexdigest()


def read_ranks(output, model, checkpoint, fingerprint, records):
    data = json.loads((output / f"{model}-ranks.json").read_text())
    if data["baseline_sha256"] != fingerprint or data["checkpoint"] != checkpoint:
        raise ValueError(
            f"Rank capture belongs to another baseline or drafter: {model}"
        )
    seqs = data["ranks"]
    if len(seqs) != len(records) or any(
        len(seq) != len(row["token_ids"]) or any(r not in (1, 2, 3) for r in seq)
        for seq, row in zip(seqs, records)
    ):
        raise ValueError(f"Invalid rank capture for {model}")
    return seqs


def write_rows(path: Path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
