"""Create combined figures from the two audited Gemma 4 31B campaigns."""

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--replay", required=True, type=Path)
parser.add_argument("--native", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
REPLAY, NATIVE, OUTPUT = args.replay, args.native, args.output
LINKS = ("ideal", "nvlink", "pcie", "hdr100")
COMPUTE = ("fpga", "asic", "cs3", "ideal")
LINK_LABELS = ("Ideal", "NVLink", "PCIe", "HDR100")
MODELS = {
    "E2B": ("Gemma E2B", "#3972B2", "o"),
    "E4B": ("Gemma E4B", "#157C82", "s"),
    "12B": ("Gemma 12B", "#7955A4", "D"),
    "eagle3": ("EAGLE-3", "#A3620E", "^"),
    "dspark": ("DSpark", "#B83F58", "v"),
    "oracle": ("Oracle", "#515B69", "P"),
}
REPLAY_METHODS = ("E2B", "E4B", "12B")
BASELINE = "#6B7584"

def load_csv(path, keys):
    with path.open() as file:
        rows = list(csv.DictReader(file))
    cells = {tuple(r[k] for k in keys): r for r in rows}
    assert len(cells) == len(rows)
    return cells

replay = load_csv(REPLAY / "sweep.csv", ("model", "link", "compute"))
native = load_csv(NATIVE / "native-sweep.csv", ("method", "link"))
orig_manifest = json.loads((REPLAY / "manifest.json").read_text())
native_manifest = json.loads((NATIVE / "manifest.json").read_text())
assert orig_manifest["dataset_sha256"] == native_manifest["dataset_sha256"]
assert orig_manifest["max_tokens"] == native_manifest["max_tokens"] == 128
assert len(replay) == 53 and len(native) == 9
assert all(int(r["matched_prompts"]) == 100 for r in replay.values())
assert all(int(r["requests"]) == int(r["exact_outputs_vs_greedy"]) == 100 for r in native.values())
assert all(int(r["exact_outputs_and_round_shapes_vs_ideal"]) == 100 for r in native.values())

plt.rcParams.update({
    "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": "#1A2633", "axes.labelcolor": "#1A2633",
    "xtick.color": "#364455", "ytick.color": "#364455",
    "axes.edgecolor": "#C3CBD4", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

def figure(title, subtitle, legend=True):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    fig.subplots_adjust(left=.085, right=.975, top=.69 if legend else .74, bottom=.24, wspace=.32)
    fig.text(.085, .93, title, weight="bold", size=21)
    fig.text(.085, .866, subtitle, size=10, color="#364455")
    if legend:
        handles = [
            Line2D([], [], color=color, marker=marker, linewidth=1.8, label=label)
            for label, color, marker in MODELS.values()
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.075, .821),
                   ncols=6, frameon=False, columnspacing=1.45, handlelength=1.65)
    return fig, axes

def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E0E6EC", linewidth=.75)

def save(fig, stem, note):
    fig.text(.085, .061, note, size=9, color="#364455", linespacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{stem}.{ext}", dpi=220)
    plt.close(fig)

def performance():
    fig, (compute_ax, link_ax) = figure(
        "Speculative decoding throughput by method",
        "Gemma 4 31B  |  100 paired prompts  |  128 generated tokens per request  |  concurrency 1",
    )
    for j, model in enumerate(REPLAY_METHODS):
        _, color, marker = MODELS[model]
        values = [[float(replay[model, link, tier]["speedup_all"]) for link in LINKS]
                  for tier in COMPUTE]
        centers = [mean(v) for v in values]
        spread = [[c - min(v) for c, v in zip(centers, values)],
                  [max(v) - c for c, v in zip(centers, values)]]
        compute_ax.errorbar([i + (j - 1) * .055 for i in range(4)], centers,
                            yerr=spread, color=color, marker=marker, capsize=3.5,
                            linewidth=1.6, markersize=5)
    for model, (_, color, marker) in MODELS.items():
        if model in REPLAY_METHODS or model == "oracle":
            tier = "cs3" if model in REPLAY_METHODS else "ideal"
            y = [float(replay[model, link, tier]["speedup_all"]) for link in LINKS]
        else:
            y = [float(native[model, link]["speedup_vs_greedy"]) for link in LINKS]
        link_ax.plot(range(4), y, color=color, marker=marker, linewidth=1.65,
                     linestyle="--" if model == "oracle" else "-", markersize=5)
    for ax in (compute_ax, link_ax):
        ax.axhline(1, color=BASELINE, linewidth=1, linestyle=":")
        ax.set_ylim(.78, 4.65)
        ax.set_yticks([1, 2, 3, 4], ["1×", "2×", "3×", "4×"])
        grid(ax)
    compute_ax.set_title("Replay at different draft compute rates", loc="left", pad=10)
    compute_ax.set_xticks(range(4), ["1", "10", "25", "Ideal"])
    compute_ax.set_xlabel("Effective drafter compute (TFLOP/s)")
    compute_ax.set_ylabel("Throughput relative to matched greedy")
    link_ax.set_title("Measured speedup across links", loc="left", pad=10)
    link_ax.set_xticks(range(4), LINK_LABELS)
    link_ax.set_xlabel("Simulated link")
    link_ax.set_ylabel("Throughput relative to matched greedy")
    save(fig, "method-throughput",
         "Left shows the mean and range across links. Right uses 25 effective TFLOP/s for Gemma, zero compute for the oracle, and native compute for EAGLE-3 and DSpark.\n"
         "Each campaign uses its matched greedy baseline (7.58 tok/s with PP=2 for replay, 7.44 tok/s with TP=2 for native). DSpark uses k=7. Other methods use k=4.\n"
         "Each cell had one timing pass. Observed differences across links include timing variation.")

def transport():
    fig, (link_ax, breakdown_ax) = figure(
        "Transport costs by drafting method",
        "Mean per request  |  Fixed round trip plus upload and reply serialization",
    )
    totals, serialization = {}, {}
    for model, (_, color, marker) in MODELS.items():
        if model in REPLAY_METHODS or model == "oracle":
            rows = [replay[model, link, "ideal"] for link in LINKS]
            delay = [float(row["mean_link_ms_per_request"]) for row in rows]
            serial = float(rows[-1]["mean_serialization_us_per_request"]) / 1000
        else:
            rows = [native[model, link] for link in LINKS]
            delay = [float(row["modeled_link_ms_per_request"]) for row in rows]
            rtt_per_request = float(native_manifest["links"]["hdr100"]["round_trip_ms"]) * (
                int(rows[-1]["proposal_rounds"]) / int(rows[-1]["requests"])
            )
            serial = delay[-1] - rtt_per_request
        assert 0 <= serial <= delay[-1] and delay == sorted(delay)
        totals[model], serialization[model] = delay[-1], serial
        link_ax.plot(range(4), delay, color=color, marker=marker, linewidth=1.65,
                     linestyle="--" if model == "oracle" else "-", markersize=5)
    link_ax.set_title("Modeled link time", loc="left", pad=10)
    link_ax.set_xlabel("Simulated link")
    link_ax.set_xticks(range(4), LINK_LABELS)
    link_ax.set_ylabel("Mean link cost per request (ms)")
    link_ax.set_ylim(-.05, 3.65)
    grid(link_ax)

    ordered = sorted(MODELS, key=totals.get)
    delays = [totals[m] for m in ordered]
    serial = [serialization[m] for m in ordered]
    rtt = [t - s for t, s in zip(delays, serial)]
    breakdown_ax.barh(range(6), rtt, color="#6B829E", height=.55, label="Fixed round trips")
    breakdown_ax.barh(range(6), serial, left=rtt, color="#D38F43", height=.55,
                      label="Payload serialization")
    for i, total in enumerate(delays):
        breakdown_ax.text(total + .046, i, f"{total:.2f}", va="center", size=9)
    breakdown_ax.set_yticks(range(6), [MODELS[m][0] for m in ordered])
    breakdown_ax.set_ylim(-.65, 6.2)
    breakdown_ax.set_xlim(0, 3.9)
    breakdown_ax.set_title("What contributes to HDR100 cost", loc="left", pad=10)
    breakdown_ax.set_xlabel("Mean link cost per request (ms)")
    breakdown_ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=8.5,
                        handlelength=1.3, columnspacing=1.2)
    grid(breakdown_ax, "x")
    save(fig, "method-transport",
         "Gemma drafters and the oracle exchange token IDs. EAGLE-3 and DSpark also receive BF16 verifier activations.\n"
         "The Gemma drafters use zero added compute here to isolate transport. HDR100 assumes a 25 µs RTT and 12 GB/s in each direction.\n"
         "The modeled costs use the recorded proposal rounds and payload sizes across the same 100 prompts.")

OUTPUT.mkdir(exist_ok=True)
performance()
transport()
