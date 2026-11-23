# Accelerator replay on Sami

Replay uses the installed vLLM V1 custom proposer. It feeds exact draft token
IDs into normal GPU verification using a target-only set of GPU weights. The integration uses the V1 model runner, synchronous scheduling, and
greedy decoding. The Python driver launches one server at a time on loopback
and terminates only the process group it created.

## Initial checks

Use the environment and CUDA library settings from [the setup guide](reproduce.md).
Make a local copy of the example plan with paths to the local checkpoints.
Use GPUs allocated to the run. The BF16 target plan uses both A100s. Prepare
a private overlay for the pinned vLLM build. It guards drafter-only metadata
on earlier pipeline ranks and lets the driver use one in-flight batch so vLLM
commits output tokens before scheduling the next speculative step.

```bash
scripts/prepare_vllm_replay.sh "$OVERLAY"
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
```

```bash
python -m unittest discover -s tests -v
python -m specdec.smoke --plan "$PLAN" --output "$RUN_DIR"
```

The smoke test creates a synthetic trace with a forced rejection followed by
fully accepted rounds. It checks exact greedy output parity, the per-step
acceptance counts, and the wait configured for each replay round.

## Paired experiment

Copy [the example settings](../configs/replay-experiment.example.json) and
resolve the plan and dataset paths relative to the settings file. Run it with
`python -m specdec.experiment --settings "$SETTINGS" --output "$RUN_DIR"`.
The driver serves the target to collect greedy token IDs, serves the independent
drafter to capture proposals at each possible committed prefix, then starts the
target again for each replay scenario. Use compatible target and drafter
tokenizers and checkpoints. The worked configuration assumes the Gemma 4 family.
It keeps the same workload, generation settings, and single in-flight batch
schedule across paired target runs. Individual batches can contain multiple
requests.

Each run records a resolved manifest with the workload hash, server logs,
target token sequences, proposals, request records, offline acceptance
comparisons, and aggregate summaries. Per-batch step logs distinguish the
configured delays from measured wait. Set `stream` to record time to first
token and inter-chunk gaps. Each streamed response chunk records how many
tokens arrived together.

Use `python -m specdec.inspect_replay --help` to display a zero-indexed
verification step or save a two-step JSONL fixture. The experiment checks
drafted and accepted counts against the detailed verifier metrics returned by
vLLM and identifies the first mismatching step.

## Latency profile

For a nonempty batch of proposals, the worker waits once for the configured
round trip, serialized transfers in both directions, time to the first draft
token, and one TBT interval for each remaining draft token in the longest row.
Times are in milliseconds and bandwidths in bytes per second. An omitted
bandwidth models zero serialization time for that direction. The model assumes
the accelerator drafts batch rows in parallel, with a single first-token cost
and TBT profile for the batch.

The profile accounts for committed token IDs uploaded and draft token IDs
downloaded. `upload_prompt_tokens` adds the prompt on the first round.
`hidden_bytes_per_committed_token` adds an activation payload. Optional KV
transfer can be `none`, `initial`, or `per_round`, sized by
`kv_bytes_per_position` and charged in `kv_direction`. By default it sizes
the full committed prefix. Set `kv_scope` to `delta` to size by newly
committed tokens. The KV profile represents link traffic for the configured cache placement.

A replay row specifies `prefix_token_ids`, `prompt_length`, and
`draft_token_ids`. An empty draft is a valid hit, useful near output limits.
Missing prefixes fail by default. This keeps timing meaningful when a
baseline and replay workload drift. For standalone replay set
`SPECDEC_REPLAY_CONFIG` to a JSON file with a `trace` path, optional `latency`
object, and optional `log` path. Relative paths resolve from that file.

## Shared accelerator queue

Run `python -m specdec.replay_queue path/to/steps.jsonl` to estimate queue
wait for the recorded batches. Pass multiple logs to model separate clients
starting together. Each batch joins the queue after upload and half of the
fixed RTT. The single accelerator serves batches in arrival order, and link
return time can overlap another batch's service. Use `--service-scale` to
sweep accelerator compute time. The analysis holds the recorded arrival
times fixed, so use a new serving run to measure feedback from queue delays
into future proposal arrivals.
