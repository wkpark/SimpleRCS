#!/usr/bin/env python3
# ruff: noqa: T201, ANN201
import argparse
import os
import sys
from pathlib import Path

from simple_rcs.simple_rcs import SimpleRCS
from simple_rcs.simple_rcs_gpg import gpg_sign_callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit content to SimpleRCS with GPG signing.")

    # Changed: content_file is now the primary argument
    parser.add_argument("content_file", help="Path to the file containing the content to commit")

    # Changed: rcs_file is optional. If not provided, it's auto-resolved.
    parser.add_argument("rcs_file", nargs='?', help="Optional explicit path to the .srcs file")

    parser.add_argument("-m", "--message", required=True, help="Commit log message")
    parser.add_argument("-a", "--author", default=os.environ.get("USER", "unknown"), help="Author name")
    parser.add_argument("--no-sign", action="store_true", help="Disable GPG signing")
    parser.add_argument("--binary", action="store_true", help="Force committing content as binary")
    parser.add_argument(
        "--encoding",
        default="base64",
        choices=["raw", "base64", "base85"],
        help=(
            "How binary payloads are encoded. raw is RCS-style: keep the bytes, let the block's '@' "
            "escaping carry them -- ~0.4%% overhead against base64's 33%%, and faster to decode. "
            "base85 uses git's alphabet. base64 streams are byte-identical to earlier releases; "
            "raw and base85 blocks cannot be read by simple-rcs 0.2.0."
        ),
    )

    parser.add_argument("--allow-empty", action="store_true",
        help="Commit even when the content is identical to HEAD. Without it an "
             "unchanged file is refused with exit 1, as 'git commit' does -- a "
             "repeat commit stores an empty delta, which every later checkout "
             "of an older version then has to walk back through.")

    # New: Configure storage directory
    parser.add_argument("--srcs-dir", default=".srcs",
        help="Directory to store .srcs files if rcs_file is not provided (default: .srcs)")

    args = parser.parse_args()

    target_path = Path(args.content_file)
    if not target_path.exists():
        print(f"Error: Content file '{args.content_file}' not found.")
        sys.exit(1)

    # Resolve RCS File Path
    if args.rcs_file:
        # User explicitly provided the RCS file path
        rcs_path = Path(args.rcs_file)
    else:
        # Auto-resolution logic
        rcs_filename = target_path.name + ".srcs"

        # 1. Check if .srcs file exists in the current directory (legacy/override behavior)
        current_dir_rcs = Path(rcs_filename)

        if current_dir_rcs.exists():
            rcs_path = current_dir_rcs
        else:
            # 2. Use the hidden directory (default: .srcs)
            srcs_dir = Path(args.srcs_dir)
            if not srcs_dir.exists():
                os.makedirs(srcs_dir, exist_ok=True)
                print(f"Created version directory: {srcs_dir}")

            rcs_path = srcs_dir / rcs_filename

    # Read as bytes first to support both text and binary files.
    try:
        content = target_path.read_bytes()
    except Exception as e:
        print(f"Error reading file '{target_path}': {e}")
        sys.exit(1)

    # commit() stores bytes as binary (BSDIFF) blocks, which disables line
    # deltas and blame — decodable content must be committed as str.
    # NUL check first: control bytes are valid UTF-8, so decode alone would
    # misclassify e.g. b"\x00\x01\x02" as text (same heuristic as git).
    if not args.binary and b"\x00" not in content[:8192]:
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            pass  # genuinely binary; keep bytes

    # Initialize SimpleRCS
    # SimpleRCS expects str path
    rcs = SimpleRCS(str(rcs_path))

    # Before the signing setup: there is no point taking a GPG signature for a
    # commit that is about to be refused.
    if not args.allow_empty and rcs.matches_head(content):
        head_ver = rcs.head_info["ver"]
        print(f"nothing to commit, '{target_path.name}' matches HEAD (v{head_ver})")
        sys.exit(1)

    callbacks = []
    if not args.no_sign:
        callbacks = [gpg_sign_callback]
        # Only print if we are actually signing
        # print("Signing enabled (GPG).")

    try:
        new_ver = rcs.commit(
            content=content,
            author=args.author,
            log=args.message,
            signer_callbacks=callbacks,
            encoding=args.encoding,
        )
        print(f"Successfully committed '{target_path.name}' -> '{rcs_path}' (Version {new_ver})")

    except Exception as e:
        print(f"Commit failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
