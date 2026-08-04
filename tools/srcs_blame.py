#!/usr/bin/env python3
# ruff: noqa: T201, ANN201
import argparse
import sys
from pathlib import Path

from simple_rcs.simple_rcs import SimpleRCS


def resolve_rcs_path(target_path: Path, explicit_rcs_path: str | None = None, srcs_dir_name: str = ".srcs") -> Path:
    """Resolves the path to the .srcs file."""
    if explicit_rcs_path:
        return Path(explicit_rcs_path)

    # Allow passing the .srcs file directly as the positional argument
    if target_path.suffix == ".srcs":
        return target_path

    rcs_filename = target_path.name + ".srcs"
    current_dir_rcs = Path(rcs_filename)

    if current_dir_rcs.exists():
        return current_dir_rcs

    srcs_dir = Path(srcs_dir_name)
    return srcs_dir / rcs_filename


def main():
    parser = argparse.ArgumentParser(description="Annotate each line of HEAD with the revision that last modified it.")
    parser.add_argument("content_file", help="Path to the tracked file (or the .srcs file itself)")
    parser.add_argument("rcs_file", nargs='?', help="Optional explicit path to the .srcs file")
    parser.add_argument("--srcs-dir", default=".srcs", help="Directory holding .srcs files (default: .srcs)")
    parser.add_argument("--depth", type=int,
        help="Limit backward traversal to this many versions; older lines are blamed on the oldest reached version")

    args = parser.parse_args()

    target_path = Path(args.content_file)
    rcs_path = resolve_rcs_path(target_path, args.rcs_file, args.srcs_dir)

    if not rcs_path.exists():
        print(f"Error: RCS file '{rcs_path}' not found.")
        sys.exit(1)

    rcs = SimpleRCS(str(rcs_path))

    annotations = rcs.blame(depth=args.depth)

    if not annotations:
        if not rcs.head_info:
            print("No history found.")
        elif rcs.head_info.get("is_binary"):
            print(f"Error: '{rcs_path}' HEAD is binary; blame is not supported.")
        else:
            print("No lines to annotate (empty content).")
        sys.exit(1)

    ver_w = max(len(e.get("ver") or "?") for e in annotations)
    author_w = max(len(e.get("author") or "?") for e in annotations)
    num_w = len(str(len(annotations)))

    for lineno, entry in enumerate(annotations, start=1):
        ver = entry.get("ver") or "?"
        author = entry.get("author") or "?"
        # Dates are ISO strings; trim sub-second precision for display
        date = (entry.get("date") or "")[:19]
        print(f"{ver:<{ver_w}} ({author:<{author_w}} {date}) {lineno:>{num_w}}: {entry['line']}")


if __name__ == "__main__":
    main()
