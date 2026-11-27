"""Select the paired SpecDec workload from cached LMSYS-Chat-1M Arrow shards."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import Dataset, concatenate_datasets
    from transformers import AutoTokenizer

    files = sorted(args.dataset_cache.glob("lmsys-chat-1m-train-*.arrow"))
    if not files:
        raise ValueError(f"No training shards at {args.dataset_cache}")
    dataset = concatenate_datasets([Dataset.from_file(str(file)) for file in files])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    seed = 555
    rng = random.Random(seed)
    # Sample without replacement and retain the random ordering across both strata.
    candidates = rng.sample(range(len(dataset)), min(len(dataset), 20000))
    quotas = {"short": 50, "long": 50}
    seen = set()
    rows = []
    for index in candidates:
        source = dataset[index]
        turns = source["conversation"]
        if (source["language"] != "English" or source.get("redacted")
                or not turns or turns[0]["role"] != "user"
                or any(m.get("flagged", False) for m in source.get("openai_moderation") or [])):
            continue
        prompt = turns[0]["content"].strip()
        if not prompt or prompt in seen or len(prompt) > 12000:
            continue
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
        length = len(ids)
        group = "short" if 32 <= length <= 128 else "long" if 384 <= length <= 768 else None
        if group is None or quotas[group] == 0:
            continue
        quotas[group] -= 1
        seen.add(prompt)
        rows.append({
            "id": f"lmsys-{index:07d}", "prompt": prompt,
            "metadata": {"group": group, "prompt_tokens": length,
                         "source_row": index, "source_id": source["conversation_id"]},
        })
        if not any(quotas.values()):
            break
    if any(quotas.values()):
        raise ValueError(f"Could not fill prompt quotas: {quotas}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} prompts using seed {seed} from {len(dataset)} dataset rows")


if __name__ == "__main__":
    main()
