"""Plot the fixed-node-budget comparison and exact length histograms for one drafter."""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, required=True, help="Study data and figure directory"
)
parser.add_argument("--model", required=True)
parser.add_argument("--depths", nargs="+", type=int, default=[16, 32])
args = parser.parse_args()
plt.style.use("seaborn-v0_8-whitegrid")
with (args.output / "tree-results.csv").open() as file:
    rows = [r for r in csv.DictReader(file) if r["model"] == args.model]
if not rows:
    parser.error("Drafter absent from tree-results.csv")
fig, ax = plt.subplots(layout="constrained", figsize=(6, 3.5))
for shape, marker in zip(("chain", "root leaf", "early exits", "two chains"), "osD^"):
    series = sorted(
        (int(r["budget"]), float(r["accepted_per_round"]))
        for r in rows
        if r["shape"] == shape
    )
    ax.plot(*zip(*series), marker=marker, label=shape.capitalize())
ax.set(
    xlabel="Draft nodes",
    ylabel="Accepted tokens per round",
    title=args.model,
    xticks=[4, 8, 16],
)
ax.legend(frameon=False)
fig.savefig(args.output / f"{args.model}-fixed-budget.pdf")
with (args.output / "acceptance-length-distribution.csv").open() as file:
    rows = [
        r
        for r in csv.DictReader(file)
        if r["model"] == args.model and r["max_second_choices"] == "1"
    ]
fig, axes = plt.subplots(
    len(args.depths),
    1,
    sharex=True,
    sharey=True,
    squeeze=False,
    layout="constrained",
    figsize=(7, 2.5 * len(args.depths)),
)
for (ax,), depth in zip(axes, args.depths):
    data = sorted(
        (int(r["accepted_length"]), float(r["pct_exact"]))
        for r in rows
        if int(r["depth"]) == depth
    )
    if not data:
        parser.error(f"No distribution for depth {depth}")
    ax.bar(*zip(*data), width=0.8, color="#188065")
    ax.set(title=f"{args.model} · Depth {depth}", ylabel="Proposal rounds (%)")
axes[-1, 0].set_xlabel("Accepted draft tokens")
fig.savefig(args.output / f"{args.model}-length-distribution.pdf")
