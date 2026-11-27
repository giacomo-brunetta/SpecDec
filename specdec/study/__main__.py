"""Run the acceptance study, timing calibration, oracle validation and plots."""

import argparse
from pathlib import Path

from . import acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "capture",
        "trace-depth",
        "acceptance",
        "calibrate",
        "project",
        "check-oracle",
        "plot",
    ):
        cmd = sub.add_parser(name)
        cmd.add_argument(
            "--output", type=Path, required=True, help="Study output directory"
        )
        if name != "plot":
            cmd.add_argument(
                "--run", type=Path, required=True, help="Completed campaign directory"
            )
        if name in ("capture", "plot"):
            cmd.add_argument(
                "--model", required=True, help="Drafter name from the plan"
            )
        if name in ("acceptance", "trace-depth", "project"):
            cmd.add_argument(
                "--depths",
                nargs="+",
                type=int,
                default=[*range(1, 17), 20, 24, 32]
                if name != "project"
                else [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32],
            )
        if name == "plot":
            cmd.add_argument("--link", default="hdr100")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "depths") and (
        min(args.depths) < 1 or len(set(args.depths)) != len(args.depths)
    ):
        parser.error("Depths must be distinct positive integers")
    if args.command == "capture":
        acceptance.capture(args.run, args.output, args.model)
    elif args.command == "trace-depth":
        from .trace_depth import run

        run(args.run, args.output, args.depths)
    elif args.command == "acceptance":
        acceptance.analyze(args.run, args.output, args.depths)
    elif args.command == "calibrate":
        from .calibrate import run

        run(args.run, args.output)
    elif args.command == "project":
        from .distance import project

        project(args.run, args.output, args.depths)
    elif args.command == "check-oracle":
        from .check_oracle import run

        run(args.run, args.output)
    else:
        from .plot import plot

        plot(args.output, args.model, args.link)


if __name__ == "__main__":
    main()
