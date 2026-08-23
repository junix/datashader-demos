"""Command-line entry point for rendering and validating demos."""

from __future__ import annotations

import argparse
from pathlib import Path

from .render import DEMOS, render_demo, validate_png


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="datashader-demos")
    subcommands = result.add_subparsers(dest="command", required=True)
    render = subcommands.add_parser("render", help="render transparent PNG artifacts")
    render.add_argument("demos", nargs="*", metavar="DEMO")
    render.add_argument("--output-dir", type=Path, default=Path("out"))
    render.add_argument("--rows", type=int, help="override row count for every selected demo")
    validate = subcommands.add_parser("validate", help="validate transparent PNG artifacts")
    validate.add_argument("--output-dir", type=Path, default=Path("out"))
    subcommands.add_parser("list", help="list available demo names")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "list":
        for demo in DEMOS.values():
            print(f"{demo.name:20} {demo.default_rows:>10,} rows  {demo.title}")
        return
    if args.command == "render":
        names = args.demos or list(DEMOS)
        unknown = sorted(set(names) - set(DEMOS))
        if unknown:
            raise SystemExit(f"unknown demo(s): {', '.join(unknown)}")
        for name in names:
            path = render_demo(name, args.output_dir, rows=args.rows)
            stats = validate_png(path)
            print(
                f"rendered {path} "
                f"({stats.width}x{stats.height}, {stats.transparent_pixels:,} transparent pixels)"
            )
        return
    for name in DEMOS:
        path = args.output_dir / f"{name}-transparent.png"
        stats = validate_png(path)
        print(f"validated {path} ({stats.visible_pixels:,} visible pixels)")


if __name__ == "__main__":
    main()
