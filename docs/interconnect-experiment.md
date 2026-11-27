# Drafter interconnect experiment

The experiment asks when separating the speculative drafter from the target
costs enough to erase its decode speedup. Compare the same proposal trace,
requests, draft length, verifier, and drafter compute profile across three
attachment options. Each option connects the target to the drafter. The
target's own pipeline-parallel placement stays fixed.

## Reference profiles

These are modeled reference systems with warm, uncontended links and
preallocated buffers. Bandwidth is effective payload bandwidth per direction
in decimal GB/s. Fixed RTT covers a small request and reply at the transport
layer, including modeled staging and completion costs. Application RPC,
scheduler queueing, and drafter compute are outside this fixed RTT.

| Attachment | Data path | Fixed RTT | Effective bandwidth |
| --- | --- | ---: | ---: |
| NVLink 3 direct | Target → drafter over an A100-class NVLink bridge | 8 µs | 250 GB/s |
| PCIe 4.0 x16, host-staged | Target → pinned host DRAM → drafter on the same host | 20 µs | 23 GB/s |
| HDR100 InfiniBand, host-staged | Target → pinned host DRAM → 100 Gb/s fabric → remote pinned host DRAM → drafter | 25 µs | 12 GB/s |

- **NVLink.** NVIDIA specifies up to 600 GB/s aggregate across both directions
  for the A100 PCIe bridge. Published DGX A100 P2P results report 281 GB/s
  one-way and 3.65 µs transfer latency. A separate DGX A100 benchmark reports
  roughly 250 GB/s one-way and 3 µs. These are NVSwitch measurements used as
  anchors for a full-width direct bridge. The model uses 250 GB/s and rounds
  two small transfers to 8 µs. [A100 PCIe product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/A100-PCIE-Prduct-Brief.pdf),
  [DGX A100 study, Table 4](https://www.mdpi.com/1996-1073/14/2/376#tab4-energies-14-00376),
  [Microway DGX A100 P2P benchmark](https://www.microway.com/hardware/dgx-a100-review-throughput-and-hardware-summary/).
- **PCIe.** NVIDIA reports about 23 GB/s for pinned A100 host transfers.
  A100 measurements on Perlmutter and Polaris put host/device copy startup
  around 4–6 µs per leg. Model four legs per request/reply at 5 µs each.
  The 23 GB/s setting assumes chunked copies that pipeline across the two
  PCIe links and enough host memory bandwidth to sustain both legs.
  [NVIDIA HGX A100 Software User Guide](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html),
  [DOE interconnect microbenchmarks, Table 6](https://carlpearson.net/pdf/20231113_siefert_pmbs.pdf).
- **Network.** The Toubkal HDR100 measurements report 12.34 GB/s and about
  1 µs one-way between host buffers. The modeled RTT includes four 5 µs
  host/device copy startups, two 1 µs fabric traversals, and a 3 µs allowance
  for handoffs. The fabric is the slowest stage when copies and NIC transfers
  are pipelined. This is a host-staged HPC path using pinned buffers and a
  low-overhead transport. GPUDirect RDMA and TCP RPC have different costs.
  [Toubkal study, Table 16](https://d-nb.info/1383883793/34),
  [DOE interconnect microbenchmarks, Table 6](https://carlpearson.net/pdf/20231113_siefert_pmbs.pdf).

## Timing model

For each nonempty proposal batch:

```text
T_link = RTT_min + upload_bytes / B_upload + download_bytes / B_download
T_round = T_link + first_draft_token_time + (k_actual - 1) × drafter_TBT
```

Here k_actual is the longest draft in the batch. Use consistent time units. The simulator accepts milliseconds and bytes per
second. B is the slowest stage in the corresponding direction when the stages
pipeline. Request and response serialize separately. If staging copies run
sequentially, transfer times add. With the reference component rates that
gives 11.5 GB/s for the two PCIe stages and about 5.87 GB/s for PCIe + HDR100 +
PCIe. Use those rates as a staging-overlap sensitivity check. Keep the same
fixed RTT for this bandwidth sensitivity.

| Payload across both directions per proposal batch | NVLink | PCIe | HDR100 |
| --- | ---: | ---: | ---: |
| 64 B | 8.00 µs | 20.00 µs | 25.01 µs |
| 1 MiB | 12.19 µs | 65.59 µs | 112.38 µs |

## Offline pilot

The [saved E2B smoke trace](../results/20261123-191454_e2b-accelerator-replay/)
contained 19 nonempty proposal batches across its concurrency 1 and 2 runs,
with at most 136 bytes per batch. Applying the reference profiles to those
recorded payloads gives median link costs of 8.00, 20.00, and 25.00 µs.
For this token-only trace, the fixed RTT accounts for almost all modeled
link time. On Sami, 100 calls requesting an 8 µs Python sleep averaged
about 60 µs wall time in a standalone spot check. Use the model's delay
fields for comparisons at this resolution.

## Replay

Copy [the replay settings](../configs/replay-interconnect.example.json) and
point them at the local target/drafter plan and dataset. Follow
[the replay setup](replay.md) for the private vLLM overlay, then run:

```bash
python -m specdec.experiment --settings "$SETTINGS" --output "$RUN_DIR"
```

The example uses two smoke prompts, four draft tokens, and a zero-delay
speculative control. It models a token exchange with a resident drafter
cache and zero drafter compute to isolate link costs. For the report, run
each drafter trace across all three attachment options. Give the three
profiles identical first-token and TBT values, and cross the topologies with
the payload policies that the drafter architecture actually supports. An
independent drafter uses its own KV representation. Model KV transfer only
where the sender and receiver agree on the cache layout, and include the
initial transfer in TTFT. A hidden-state or KV trace should carry an explicit
byte count for the applicable tensor layout and dtype.

Report exact output and acceptance parity, modeled link cost per proposal
batch, TTFT, inter-chunk latency, throughput, and speedup over the greedy
baseline. Use the modeled breakdown for differences on the order of tens of
microseconds. Python sleep and scheduler jitter affect the measured replay
wait at that scale. Use larger payloads and repeated paired runs to measure
when the link changes the end-to-end outcome.
