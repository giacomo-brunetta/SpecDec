# Gemma 4 interconnect and compute campaign

The campaign crosses four modeled interconnects (ideal, direct NVLink 3,
host-staged PCIe 4.0, and host-staged HDR100) with ideal, 1, 10, and 25 effective
TFLOP/s for each of Gemma 4 E2B, E4B, and 12B. A Gemma 4 31B oracle with ideal
compute runs across the four links. The same Gemma 4 31B also supplies the
greedy baseline. The total is 53 configurations at concurrency 1, k=4, and
128 generated tokens per prompt. This sweep uses token IDs with resident
drafter KV.

The workload contains 100 unique English first turns sampled with seed 555
from the LMSYS-Chat-1M training cache. The short and long strata each contain
50 prompts. The selection bounds, including the target chat template, are
32–128 and 384–768 tokens. Record the cache revision, prompt IDs, workload
checksum, and resolved plan with each run. The checkpoint files retain the
request order and token IDs. [LMSYS-Chat-1M paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html)

## Run

Use the [private vLLM overlay and environment](replay.md) with an eager BF16
Gemma 4 31B target on two PCIe A100s (pipeline parallelism 2). Set the plan to
one sequence, 1,024 context tokens, synchronous scheduling, and disabled prefix
caching so each target request performs its own prefill. Use local
Gemma 4 E2B, E4B, and 12B checkpoint directories beside the target directory.
The Arrow workload selector also needs `datasets` in its Python environment.
The inference phases use the existing vLLM environment.

```bash
python -m specdec.campaign_workload \
  --dataset-cache "$LMSYS_ARROW_DIR" --tokenizer "$TARGET" --output "$WORKLOAD"
python -m specdec.campaign_sweep \
  --plan "$PLAN" --dataset "$WORKLOAD" --output "$RUN_DIR"
python -m specdec.campaign_report --output "$RUN_DIR"
```

The sweep obtains greedy target trajectories, captures draft proposals in
batches over prefixes, and verifies the replay on the target GPUs.
The prefill and catch-up cost of a deployed remote drafter would require
additional measurements. The compute tiers use `2 × active parameters`
FLOPs per draft token. See the [interconnect model](interconnect-experiment.md)
for the physical profiles.

| Drafter | Effective parameters | 1 TFLOP/s | 10 TFLOP/s | 25 TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| E2B | 2.3B | 4.600 ms/token | 0.460 ms/token | 0.184 ms/token |
| E4B | 4.5B | 9.000 ms/token | 0.900 ms/token | 0.360 ms/token |
| 12B | 11.95B | 23.900 ms/token | 2.390 ms/token | 0.956 ms/token |

The 1 TFLOP/s tier is an illustrative slow setting. The 10 TFLOP/s tier is
anchored to the reported 0.30 ms/token Qwen3-1.7B Corsair result. The
25 TFLOP/s tier is anchored to the 0.63 ms Llama 3.1 8B single-stream CS-3
inter-token latency in Fig. 8 of the project’s IPDPS 2026 paper. All three
rates are equivalent single-stream decode rates used to project per-token
latency for Gemma. [Gemma 4 model card](https://huggingface.co/google/gemma-4-E2B-it)
[d-Matrix experiment](https://www.d-matrix.ai/corsair-speculative-decoding/)
[IPDPS paper](https://doi.org/10.1109/IPDPS65963.2026.00083)

The runner checkpoints the baseline, each prompt's proposal trace, and each
finished timing configuration. Re-run with the same setup and output directory
to continue an interrupted campaign. If the verifier commits a token outside
the trace, the replay continues target-only for prefixes absent from the trace.
The results include match, miss, and acceptance audits and separate paired
speedups for the prompts that matched throughout each drafter's sweep.

The report compares modeled delays with observed throughput and calculates
compute break-even on matched output. Timing differences at the scale of
the physical link RTTs should be interpreted from the modeled delay fields.
