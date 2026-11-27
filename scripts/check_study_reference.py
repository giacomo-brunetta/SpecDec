"""Check the historical Gemma rows against aggregate CSVs from a reproduction."""

import argparse
import csv
import math
from pathlib import Path


def compare(reference, actual, keys):
    def key(row):
        return tuple(row[k] for k in keys)

    with actual.open() as stream:
        rows = list(csv.DictReader(stream))
    by_key = {key(r): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"Duplicate rows in {actual}")
    count = 0
    with reference.open() as stream:
        for row in csv.DictReader(stream):
            got = by_key[key(row)]
            for name, expected in row.items():
                if got[name] == expected:
                    continue
                try:
                    matches = math.isclose(
                        float(got[name]), float(expected), rel_tol=1e-9, abs_tol=1e-9
                    )
                except ValueError:
                    matches = False
                if not matches:
                    raise ValueError(
                        f"{reference.name} {key(row)} {name}: {got[name]} vs {expected}"
                    )
            count += 1
    print(f"{reference.name}: reproduced {count} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ref = Path(__file__).resolve().parents[1] / "reference/gemma4"
    timing_keys = ("model", "depth", "second_choices", "link", "compute")
    for name, actual, keys in [
        ("trace-depth", "trace-depth", ("model", "k")),
        ("tree-results", "tree-results", ("model", "budget", "shape")),
        ("tree-long", "tree-long", ("model", "depth", "max_second_choices")),
        (
            "acceptance-length-distribution",
            "acceptance-length-distribution",
            ("model", "depth", "max_second_choices", "accepted_length"),
        ),
        ("distance", "distance", timing_keys),
        ("short-depth", "distance", timing_keys),
    ]:
        compare(ref / f"{name}.csv", args.output / f"{actual}.csv", keys)
