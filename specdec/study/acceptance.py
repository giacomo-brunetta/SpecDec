"""Score fixed-budget layouts and full-depth binary trees on target-prefix ranks."""

import math
from collections import Counter
from functools import lru_cache

from .common import inputs, read_ranks, write_rows


@lru_cache(None)
def nodes(depth, second_choices):
    return sum(
        math.comb(level, j)
        for level in range(1, depth + 1)
        for j in range(min(second_choices, level) + 1)
    )


def rounds(seq, depth, second_choices, topology=None):
    """Yield offset, available tree depth and accepted length of nonempty proposals."""
    offset = 1
    while offset < len(seq) - 1:
        limit = min(depth, len(seq) - offset - 1)
        used = accepted = 0
        for j, rank in enumerate(seq[offset : offset + limit]):
            used += rank == 2
            if rank > 2 or used > second_choices:
                break
            if (
                topology is not None
                and tuple(seq[offset : offset + j + 1]) not in topology
            ):
                break
            accepted += 1
        yield offset, limit, accepted
        offset += 1 + accepted


def layout(shape, budget):
    exits = {
        "chain": 0,
        "root leaf": 1,
        "early exits": budget // 4,
        "two chains": budget // 2,
    }[shape]
    result = {(1,) * i for i in range(1, budget - exits + 1)}
    for i in range(exits):
        result.add((2,) + (1,) * i if shape == "two chains" else (1,) * i + (2,))
    return result


def capture(run, output, name):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    target, models, records, fingerprint = inputs(run)
    spec = models[name]
    target_tok = AutoTokenizer.from_pretrained(target["model"])
    draft_tok = AutoTokenizer.from_pretrained(spec["model"])
    if target_tok.get_vocab() != draft_tok.get_vocab():
        raise ValueError("Rank replay requires a shared token-ID vocabulary")
    engine = LLM(
        model=spec["model"],
        dtype=target.get("dtype", "auto"),
        enforce_eager=True,
        language_model_only=True,
        max_model_len=target["max_model_len"],
        max_num_seqs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.92,
    )
    ranks = []
    for row in records:
        prompt, tokens = row["prompt_token_ids"], row["token_ids"]
        out = engine.generate(
            [{"prompt_token_ids": prompt + tokens}],
            SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=2),
            use_tqdm=False,
        )[0]
        probs = out.prompt_logprobs[len(prompt) :]
        if len(probs) != len(tokens):
            raise ValueError(
                "Drafter returned an incomplete set of prompt log probabilities"
            )
        ranks.append([min(p[t].rank, 3) for p, t in zip(probs, tokens)])
    from ..campaign import write_checkpoint

    write_checkpoint(
        output / f"{name}-ranks.json",
        {
            "baseline_sha256": fingerprint,
            "checkpoint": spec["model"],
            "ranks": ranks,
        },
    )


def analyze(run, output, depths):
    _, models, records, fingerprint = inputs(run)
    summary, histogram, fixed = [], [], []
    for name, spec in models.items():
        seqs = read_ranks(output, name, spec["model"], fingerprint, records)
        for depth in depths:
            for s in sorted({0, 1, min(2, depth), min(3, depth), depth}):
                counts = Counter(n for seq in seqs for _, _, n in rounds(seq, depth, s))
                total = sum(counts.values())
                row = dict(
                    model=name, depth=depth, max_second_choices=s, nodes=nodes(depth, s)
                )
                summary.append(
                    dict(
                        row,
                        rounds=total,
                        accepted_per_round=sum(n * c for n, c in counts.items())
                        / total,
                        pct_at_least_6=100
                        * sum(c for n, c in counts.items() if n >= 6)
                        / total,
                        pct_at_least_8=100
                        * sum(c for n, c in counts.items() if n >= 8)
                        / total,
                    )
                )
                for n in range(depth + 1):
                    histogram.append(
                        dict(
                            row,
                            accepted_length=n,
                            rounds_exact=counts[n],
                            total_rounds=total,
                            pct_exact=100 * counts[n] / total,
                            pct_at_least=100
                            * sum(c for k, c in counts.items() if k >= n)
                            / total,
                        )
                    )
        for budget in (4, 8, 16):
            for shape in ("chain", "root leaf", "early exits", "two chains"):
                tree = layout(shape, budget)
                counts = Counter(
                    n for seq in seqs for _, _, n in rounds(seq, budget, budget, tree)
                )
                total = sum(counts.values())
                accepted = sum(n * c for n, c in counts.items())
                if shape == "chain":
                    reference = total
                fixed.append(
                    dict(
                        model=name,
                        budget=budget,
                        shape=shape,
                        prompts=len(records),
                        proposal_rounds=total,
                        accepted_tokens=accepted,
                        accepted_per_round=accepted / total,
                        round_reduction_pct_vs_chain=100 * (1 - total / reference),
                    )
                )
    write_rows(output / "tree-long.csv", summary)
    write_rows(output / "acceptance-length-distribution.csv", histogram)
    write_rows(output / "tree-results.csv", fixed)
