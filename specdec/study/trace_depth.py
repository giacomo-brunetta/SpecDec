"""Cross-check chain acceptance against explicit proposal traces."""

import csv
import json

from .common import inputs


def run(src, dest, depths):
    _, models, baseline, _ = inputs(src)
    with (dest / "trace-depth.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "k",
                "prompts",
                "rounds",
                "accepted",
                "proposed",
                "accepted_per_round",
                "acceptance_pct",
            ]
        )
        for model in models:
            totals = {k: [0, 0, 0] for k in depths}
            for i, r in enumerate(baseline):
                t = r["token_ids"]
                p = r["prompt_token_ids"]
                path = src / "traces" / model / f"{i:03d}-{r['request_id']}.jsonl"
                proposals = {}
                for line in path.open():
                    row = json.loads(line)
                    offset = len(row["prefix_token_ids"]) - len(p)
                    assert row["prefix_token_ids"] == p + t[:offset]
                    proposals[offset] = row["draft_token_ids"]
                for k, counts in totals.items():
                    offset = 1
                    while offset < len(t) - 1:
                        length = min(k, len(t) - offset - 1)
                        a = 0
                        while a < length and proposals[offset + a][0] == t[offset + a]:
                            a += 1
                        if k == 4:
                            explicit = 0
                            for left, right in zip(proposals[offset], t[offset:]):
                                if left != right:
                                    break
                                explicit += 1
                            assert explicit == a, (model, i, offset, explicit, a)
                        counts[0] += 1
                        counts[1] += a
                        counts[2] += length
                        offset += a + 1
            for k, (rounds, accepted, proposed) in totals.items():
                w.writerow(
                    [
                        model,
                        k,
                        len(baseline),
                        rounds,
                        accepted,
                        proposed,
                        accepted / rounds,
                        100 * accepted / proposed,
                    ]
                )
