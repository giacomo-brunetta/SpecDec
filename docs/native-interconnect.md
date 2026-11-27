# DSpark and EAGLE-3 interconnect sweep

Run eight speculative cells and a matched greedy control on the existing
100 LMSYS prompts with 128 output tokens per request, temperature zero and
concurrency one. Use the same target checkpoint and four link profiles as the
independent-drafter campaign. Measure a new greedy control with the same
vLLM version and tensor parallel placement as the two native speculators.
All cells use Model Runner V2, which provides the DSpark proposer.

EAGLE-3 uses four draft tokens, matching the independent-drafter campaign.
DSpark uses seven, following the deployment configuration for its block
checkpoint. Adaptive verification is disabled to keep the budget fixed.
[EAGLE-3 checkpoint](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3)
[DSpark checkpoint](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark)

## Payload and timing

The hook counts the elements and dtype of the concatenated verifier
activation tensor passed to the drafter. It includes the full first prefill
and subsequent decode positions, including positions that the padded native
path passes after rejection. The modeled wire layout adds 4-byte IDs and
positions, one anchor ID and one accepted-boundary index per request, and a
4-byte ID for each draft token in the reply. The drafter retains its KV cache.

EAGLE-3 uses three BF16 hidden-state vectors at width 5,376, or 32,256 bytes
per position. DSpark uses five, or 53,760 bytes per position. The simulator
charges one RTT plus serialization in each direction. It inserts a serial
wait after existing native draft GPU work to represent the blocking transfer
on the critical path. The ideal control performs the same synchronization
and logging. These cells include native drafter compute and model a single
logical activation transfer. The model excludes target TP traffic, network
contention, and application RPC. [EAGLE-3 tensor configuration](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3/blob/28a1c8b4bb64dbaee883ba35341841138bdf1fe3/config.json)
[DSpark tensor configuration](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark/blob/63e2877348caa03a2c58ab3c47224f4164af08c4/config.json)

The following are analytical link-only estimates.

| Link | EAGLE-3 full decode round | DSpark full decode round | EAGLE-3 768-token prefill | DSpark 768-token prefill |
| --- | ---: | ---: | ---: | ---: |
| Ideal | 0 µs | 0 µs | 0 ms | 0 ms |
| NVLink | 8.65 µs | 9.72 µs | 0.107 ms | 0.173 ms |
| PCIe, host-staged | 27.02 µs | 38.70 µs | 1.097 ms | 1.815 ms |
| HDR100, host-staged | 38.45 µs | 60.85 µs | 2.090 ms | 3.466 ms |

The full decode round sends five positions for EAGLE-3 and eight for DSpark.
With one prefill round, up to 127 full decode rounds, and a 768-token prompt,
the HDR100 upper bounds for a 128-token completion are 6.97 ms and 11.19 ms,
respectively. Compare link penalties to each speculator's measured ideal
cell. The small predicted absolute differences make the modeled latency and
the measured wait more informative than raw throughput rankings.

## Run

Use a separate vLLM 0.31.0 CUDA environment on Sami with the wheel compatible
with its driver. Apply the patch to the matching vLLM source or wheel. From
the SpecDec checkout, run:

```bash
SPECDEC_REPO="$(pwd)"
VLLM_SITE="$(python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')"
(cd "$VLLM_SITE" && patch -p1 < "$SPECDEC_REPO/patches/vllm-0.31-native-link.patch")
cp configs/native-interconnect.example.json /tmp/specdec-native-plan.json
```

Set the plan's model to the original Gemma 4 31B snapshot on Sami. Use the
original campaign's workload JSONL file for `WORKLOAD`. Run with `--smoke`
first in a separate output directory to validate both GPU paths on the
shortest and longest prompts. Run without that option for the full sweep.

```bash
python -m specdec.native_campaign --plan /tmp/specdec-native-plan.json \
  --dataset "$WORKLOAD" --output "$RUN_DIR"
```

The runner checkpoints each finished cell. The aggregate CSV reports
observed throughput, TTFT, TPOT, wait, exact token agreement, per-round
acceptance parity and the link penalty. A projected throughput column adds
the transport costs to the ideal measurement when outputs and round shapes
match across all paired prompts. The raw request records and round logs are
stored with the run.
