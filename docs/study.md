# Reproduce the speculative decoding study

The study has three independently runnable parts. The replay campaign measures a target verifying an independent drafter or an oracle. Native EAGLE-3 and DSpark measure target-specific speculator checkpoints. The tree study scores alternate continuations on saved greedy trajectories and projects latency under four hardware rates and four links.

## Choose a target and drafters

Copy [the replay study plan](../configs/study.example.json). Set `server.model` to a target checkpoint and set its parallelism, dtype, maximum sequence length and memory allocation for your GPUs. Each entry in `drafters` has a short name, checkpoint path or Hugging Face ID, and **active billions of parameters per generated token**. For a mixture-of-experts model this is the active count. Use checkpoints with the same token-ID vocabulary and compatible special tokens. Independent drafter capture runs one drafter on one GPU at a time. Use local snapshot paths to pin weight and tokenizer revisions.

The original workload uses 50 short and 50 long prompts, concurrency one, 128 output tokens per request, greedy decoding and a depth-four live replay control. Keep those settings to compare against the reference study. The context limit must cover the tokenized prompt, output and rank-capture lookahead. Regenerate the workload with the new target tokenizer to preserve the two length strata. Fixed prompt IDs across targets answer a different question and may change the length distribution.

## Measured replay

Use [the pinned environment](reproduce.md) and [the private V1 replay overlay](replay.md). Use the Arrow cache of LMSYS-Chat-1M under the dataset's access terms. Run the following from the checkout root, after activating the vLLM environment. The selector also needs the Hugging Face datasets package. Use a fresh directory for each model family or checkpoint revision.

```bash
export PYTHONPATH="$PWD:$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
PLAN=configs/study.example.json   # or your edited copy
RUN=runs/gemma4/replay
OUT=runs/gemma4/tree
mkdir -p "$RUN"
TARGET=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["server"]["model"])' "$PLAN")
python -m specdec.campaign_workload --dataset-cache "$LMSYS_ARROW_DIR" --tokenizer "$TARGET" --output "$RUN/workload.jsonl"
python -m specdec.campaign_sweep --plan "$PLAN" --dataset "$RUN/workload.jsonl" --output "$RUN"
python -m specdec.campaign_report --output "$RUN"
```

The sweep saves the resolved inputs, greedy outputs, per-prompt proposal traces, timings and acceptance audits. The target generation config can be placed at `$RUN/target-generation-config.json` for the optional natural-stop analysis. Aggregate speedup uses a fixed output length. See [campaign design](campaign.md) for the link and compute profiles.

## Depth, acceptance and trees

```bash
python -m specdec.study trace-depth --run "$RUN" --output "$OUT"
for model in $(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["drafters"])' "$PLAN"); do
  python -m specdec.study capture --model "$model" --run "$RUN" --output "$OUT"
done
python -m specdec.study acceptance --run "$RUN" --output "$OUT"
python -m specdec.study calibrate --run "$RUN" --output "$OUT"
python -m specdec.study project --run "$RUN" --output "$OUT"
python -m specdec.study check-oracle --run "$RUN" --output "$OUT"
python -m specdec.study plot --output "$OUT" --model 12B --link hdr100
```

Replace `12B` with a drafter key from your plan. Plotting requires Matplotlib 3.6 or later. Capture, calibration and the live oracle check use GPUs. The acceptance and projection stages run on CPU. Pass `--depths 4 8 16` to the depth, acceptance or projection command to select a subset.

| Artifact | Interpretation |
|---|---|
| `trace-depth.csv` | Chain acceptance from the explicit greedy proposal traces, with a depth-four consistency check |
| `tree-results.csv` | Chain, root leaf, early exits and two chains at equal node budgets of 4, 8 and 16 |
| `tree-long.csv`, `acceptance-length-distribution.csv` | Mean and exact distribution for trees that retain full depth and allow a bounded number of second-ranked choices along each path |
| `calibration.json` | Five timed samples per causal extension size at each of two cached prefix lengths on the configured target |
| `distance.csv` | Projected throughput, acceptance, candidate count and target, draft and link time components |
| `oracle-check.csv` | Live oracle replay with one short and one long prompt at each of seven depths |
| `*-speedup.pdf`, `*-speedup.png` | Projected speedup against draft depth, one curve per effective compute tier |

A proposal round ends at the first unsupported token and advances one additional verifier token. Reported accepted length excludes that bonus. Only rounds with a nonempty proposal enter the acceptance distribution. The captured rank at each target prefix is sufficient to score a fixed top-two tree under greedy verification. Each topology advances its own offsets through the trajectory.

The time model charges all candidate nodes, including off-path work. The verifier estimate uses the measured depth-four oracle timing for each prompt plus a size-dependent delta from cached causal extensions. The calibration covers 1–561 positions at prefix lengths 128 and 640. Configure a target supporting a 1,536-token calibration context. A live tree-mask kernel, per-node vocabulary projections and KV compaction can change the true cost. Independent drafter compute assumes twice the active parameter count in FLOPs per node, no overlap and resident KV. The oracle has ideal drafting and returns the target continuation. Ratios from `distance.csv` are projections; native and replay timing summaries come from live vLLM runs.

## Native methods

Use [the native 0.31 runtime and hook](native-interconnect.md) and a **separate output directory**. Edit [the native plan](../configs/native-interconnect.example.json) to specify the matching target, supported speculator checkpoint, revision, draft budget and activation bytes per position for each method. Activation bytes depend on hidden width, number of tapped layers and dtype. Keep only methods for which the new target has compatible trained weights. The native campaign measures its own greedy baseline.

```bash
python -m specdec.native_campaign --plan "$NATIVE_PLAN" --dataset "$RUN/workload.jsonl" --output "$NATIVE_OUT"
python scripts/specdec-native-audit.py "$NATIVE_OUT" "$RUN/workload.jsonl"
```

## Reference and verification

[Reference CSVs](../reference/gemma4) contain aggregate Gemma results. The original distance sweep and its short-depth extension are separate files. The packaged projector emits both in one table together with additional short-depth controls. Raw prompts, generated text, ranks, per-request traces and weights stay in local run directories. Historical Gemma comparison plots can be regenerated with the scripts under `scripts/figures/`.

```bash
python -m unittest discover -s tests -v
```
