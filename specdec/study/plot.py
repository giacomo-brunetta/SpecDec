import csv

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot(output, model, link):
    with (output / "distance.csv").open() as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["model"] == model
            and r["link"] == link
            and int(r["second_choices"]) == 1
        ]
    if not rows:
        raise ValueError(f"No one-second-choice tree rows for {model} on {link}")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    for tier, label, color, marker in [
        ("ideal", "Ideal compute", "#824a85", "D"),
        ("cs3", "25 TFLOP/s", "#00796b", "o"),
        ("asic", "10 TFLOP/s", "#286aa6", "s"),
        ("fpga", "1 TFLOP/s", "#bd6517", "^"),
    ]:
        points = sorted(
            (int(r["depth"]), float(r["speedup"])) for r in rows if r["compute"] == tier
        )
        ax.plot(*zip(*points), label=label, color=color, marker=marker, ms=5, lw=2)
    ax.axhline(1, color="#505050", lw=1, ls=":")
    ax.set(
        xlabel="Draft depth",
        ylabel="Projected speedup over greedy",
        title=f"{model} tree on {link}",
        xticks=sorted({int(r["depth"]) for r in rows}),
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
    ax.grid(axis="x", visible=False)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Drafter compute",
    )
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{model}-{link}-speedup.{ext}", dpi=230)
    plt.close(fig)
