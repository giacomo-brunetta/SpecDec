"""Plot the native speculative decoding speedup and modeled link costs."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(sys.argv[1])
with (root / "native-sweep.csv").open() as file:
    cells = {(row["method"], row["link"]): row for row in csv.DictReader(file)}
with (root / "native-strata.csv").open() as file:
    strata = {(row["method"], row["link"], row["prompt_group"]): row
              for row in csv.DictReader(file)}

plt.rcParams.update({
    "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "font.size": 10,
})
fig, (speed, link) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
methods = [("eagle3", "EAGLE-3", "#0072B2", "o"), ("dspark", "DSpark", "#D55E00", "s")]
links = ("ideal", "nvlink", "pcie", "hdr100")
labels = ("Ideal", "NVLink", "PCIe", "HDR100")
greedy = float(cells["greedy", "ideal"]["tokens_per_second"])
for method, label, color, marker in methods:
    observed = [float(cells[method, name]["speedup_vs_greedy"]) for name in links]
    projected = [
        float(cells[method, name]["projected_tokens_per_second"]) / greedy
        if cells[method, name]["projected_tokens_per_second"] else float("nan")
        for name in links
    ]
    speed.plot(range(4), projected, linestyle="--", color=color, linewidth=1.2)
    speed.scatter(range(4), observed, marker=marker, color=color, s=30)
    for group, style in (("short", ":"), ("long", "-")):
        y = [float(strata[method, name, group]["modeled_link_ms_per_request"])
             for name in links]
        link.plot(range(4), y, label=f"{label}, {group}", linestyle=style,
                  marker=marker, color=color, linewidth=1.4, markersize=4)

speed.set_title("End-to-end speedup", loc="left", fontweight="bold")
speed.set_ylabel("Throughput relative to greedy")
speed.axhline(1, color="#787878", linewidth=0.7)
speed.legend(handles=[
    *(Line2D([], [], color=color, marker=marker, linestyle="", label=label)
      for _, label, color, marker in methods),
    Line2D([], [], color="#666666", linestyle="--", label="Projection from ideal"),
], frameon=False, loc="center left")
link.set_title("Modeled transport cost by prompt length", loc="left", fontweight="bold")
link.set_ylabel("Mean link cost per request (ms)")
link.legend(frameon=False, loc="upper left")
for axis in (speed, link):
    axis.set_xticks(range(4), labels)
    axis.grid(axis="y", alpha=.17)
    axis.set_axisbelow(True)
fig.savefig(root / "native-sweep-figure.pdf", bbox_inches="tight")
fig.savefig(root / "native-sweep-figure.png", dpi=200, bbox_inches="tight")
