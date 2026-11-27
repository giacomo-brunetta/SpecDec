import json
import random
import time

from specdec.driver import REPLAY_ENV, running_scenario
from specdec.serve import read_scenario


def run(base, root):
    sizes = [1, 5, 7, 9, 11, 13, 17, 25, 33, 45, 67, 91, 153, 325, 561]
    target, _ = read_scenario(base / "plan.json", "baseline")
    from transformers import AutoTokenizer

    vocab_size = len(AutoTokenizer.from_pretrained(target["model"]))
    low, high = (1024, 20000) if vocab_size >= 20000 else (0, vocab_size)
    target["max_model_len"] = 1536
    target["extra_args"] = [
        "--enforce-eager",
        "--language-model-only",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "1536",
        "--enable-prefix-caching",
        "--disable-uvicorn-access-log",
        "--no-async-scheduling",
    ]
    rng = random.Random(555)
    data = {}
    with running_scenario(target, {}, root / "server.log", env=REPLAY_ENV) as driver:
        for prefix_len in (128, 640):
            prefix = [rng.randrange(low, high) for _ in range(prefix_len)]
            driver.complete_tokens(prefix, 1)
            for rep in range(6):
                order = sizes.copy()
                rng.shuffle(order)
                for n in order:
                    suffix = [rng.randrange(low, high) for _ in range(n)]
                    start = time.perf_counter()
                    driver.complete_tokens(prefix + suffix, 1)
                    if rep:
                        data.setdefault(str(prefix_len), {}).setdefault(
                            str(n), []
                        ).append(1000 * (time.perf_counter() - start))
            print("completed prefix", prefix_len, flush=True)
    (root / "calibration.json").write_text(json.dumps(data))
