"""Make a project-report figure from the fully audited SpecDec sweep CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {"E2B": "#0072B2", "E4B": "#D55E00", "12B": "#009E73", "oracle": "#71787D"}
LINKS = ("ideal", "nvlink", "pcie", "hdr100")
COMPUTE = ("fpga", "asic", "cs3", "ideal")


def load(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = {(r["model"], r["link"], r["compute"]): r for r in rows}
    expected = {("greedy", "", "")}
    expected.update((m, l, c) for m in MODELS if m != "oracle" for l in LINKS for c in COMPUTE)
    expected.update(("oracle", l, "ideal") for l in LINKS)
    if set(data) != expected or len(rows) != len(data):
        raise ValueError(f"Expected a complete 53-configuration sweep in {path}")
    return data


def figure(data, output):
    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.right": False,
                         "axes.spines.top": False, "font.size": 10})
    fig, (perf, net) = plt.subplots(1, 2, figsize=(11.5, 4.9))
    x = range(len(COMPUTE))
    for j, (model, color) in enumerate(MODELS.items()):
        if model == "oracle":
            continue
        times = [[float(data[model, link, tier]["speedup_all"]) for link in LINKS]
                 for tier in COMPUTE]
        y = [mean(values) for values in times]
        yerr = [[average - min(values) for average, values in zip(y, times)],
                [max(values) - average for average, values in zip(y, times)]]
        perf.errorbar([i + (j - 1)*.055 for i in x], y, yerr=yerr, capsize=3, linewidth=1.75,
                      marker="o", markersize=5, color=color, label=model)
    upper = float(data["oracle", "ideal", "ideal"]["speedup_all"])
    perf.axhline(upper, linestyle="--", color=MODELS["oracle"], label="Ideal oracle", linewidth=1)
    perf.axhline(1, linestyle=":", color="#545A60", linewidth=.9)
    perf.set_xticks(list(x), ["1", "10", "25", "Ideal"])
    perf.set_ylim(bottom=0)
    perf.set_xlabel("Effective drafter compute (TFLOP/s)")
    perf.set_ylabel("Throughput relative to greedy")
    perf.set_title("A   Measured GPU replay", loc="left", fontweight="bold")
    perf.legend(loc="best", frameon=False, ncol=2)
    perf.grid(axis="y", alpha=.18)
    perf.set_axisbelow(True)

    for j, (model, color) in enumerate(MODELS.items()):
        link_data = [float(data[model, link, "ideal"]["mean_link_ms_per_request"]) for link in LINKS]
        net.bar([i + (j - 1.5)*.19 for i in x], link_data, .17, color=color,
                label="Oracle" if model == "oracle" else model)
    net.set_xticks(list(x), ["Ideal", "NVLink", "PCIe", "HDR100"])
    net.set_ylim(bottom=0)
    net.set_ylabel("Mean modeled link time per request (ms)")
    net.set_title("B   Token-only interconnect cost", loc="left", fontweight="bold")
    net.legend(frameon=False, ncol=2)
    net.grid(axis="y", alpha=.18)
    net.set_axisbelow(True)

    fig.subplots_adjust(left=.083, right=.975, top=.88, bottom=.22, wspace=.28)
    fig.text(.083, .059,
             "100 paired prompts × 128 generated tokens, concurrency 1, k=4. "
             "Panel A shows the mean and min–max across four link cells per compute tier.\n"
             "Panel B holds draft compute at zero and shows the modeled RTT plus serialization.")
    for ext in ["pdf", "png"]:
        fig.savefig(output.with_suffix("." + ext), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    figure(load(args.csv), args.output)
