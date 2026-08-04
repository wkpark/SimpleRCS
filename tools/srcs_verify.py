#!/usr/bin/env python3
# ruff: noqa: T201, ANN201
import argparse
import os
import sys

from simple_rcs.simple_rcs import SimpleRCS
from simple_rcs.simple_rcs_gpg import gpg_verify_callback


def main():
    parser = argparse.ArgumentParser(description="Verify SimpleRCS file integrity and GPG signatures.")
    parser.add_argument("file_path", help="Path to the .srcs file")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    rcs = SimpleRCS(args.file_path)

    print(f"Verifying '{args.file_path}'...")
    is_valid = rcs.verify(verifier_callbacks=[gpg_verify_callback])

    if is_valid:
        print("✅ VERIFICATION SUCCESS: All hashes and signatures are valid.")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED: Integrity compromised or signature invalid.")
        sys.exit(1)

if __name__ == "__main__":
    main()
