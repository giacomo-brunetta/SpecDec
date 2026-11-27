import csv
import json

from specdec.campaign import measure
from specdec.driver import REPLAY_ENV, running_scenario
from specdec.serve import read_scenario


def run(base, root):
    gold = json.loads((base / "baseline.json").read_text())
    work = {
        r["id"]: r
        for r in map(json.loads, (base / "workload.jsonl").read_text().splitlines())
    }
    cases = []
    for group in ("short", "long"):
        records = [r for r in gold if r["metadata"]["group"] == group]
        cases.extend(zip([6, 8, 10, 12, 16, 24, 32], records[:7]))
    with (root / "oracle-trace.jsonl").open("w") as f:
        for k, r in cases:
            p, t = r["prompt_token_ids"], r["token_ids"]
            for i in range(1, len(t) + 1):
                f.write(
                    json.dumps(
                        dict(
                            prefix_token_ids=p + t[:i],
                            prompt_length=len(p),
                            draft_token_ids=t[i : min(i + k, len(t) - 1)],
                        )
                    )
                    + "\n"
                )
    config = root / "oracle-config.json"
    config.write_text(
        json.dumps(
            dict(trace="oracle-trace.jsonl", strict=False, log="oracle-steps.jsonl")
        )
    )
    target, scenario = read_scenario(base / "plan.json", "baseline")
    scenario.update(
        per_request_spec_decode_metrics="detailed",
        speculative_config=dict(
            method="custom_class",
            model="specdec.replay.ReplayProposer",
            num_speculative_tokens=32,
        ),
    )
    env = {**REPLAY_ENV, "SPECDEC_REPLAY_CONFIG": str(config.resolve())}
    with running_scenario(
        target, scenario, root / "oracle-server.log", env=env
    ) as driver:
        obs, _ = measure(
            driver,
            [work[r["request_id"]] for _, r in cases],
            root / "oracle-requests.jsonl",
            step_log=root / "oracle-steps.jsonl",
        )
    with (root / "oracle-check.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "depth",
                "group",
                "elapsed_ms",
                "greedy_ms",
                "output_matches",
                "drafted",
                "accepted",
            ]
        )
        for (k, r), o in zip(cases, obs):
            m = o["speculative_decoding"]
            w.writerow(
                [
                    k,
                    r["metadata"]["group"],
                    o["latency_ms"],
                    r["latency_ms"],
                    o["token_ids"] == r["token_ids"],
                    m["num_draft_tokens"],
                    m["num_accepted_draft_tokens"],
                ]
            )
