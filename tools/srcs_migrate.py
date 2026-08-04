#!/usr/bin/env python3
# ruff: noqa: T201, ANN201
import argparse
import sys

from simple_rcs.simple_rcs_gpg import gpg_sign_callback
from simple_rcs.simple_rcs_migrate import migrate


def main():
    parser = argparse.ArgumentParser(description="Migrate SimpleRCS v1 file to v2 with GPG signing.")
    parser.add_argument("src_path", help="Source v1 file path")
    parser.add_argument("dst_path", help="Destination v2 file path")
    parser.add_argument("--no-sign", action="store_true", help="Disable GPG signing during migration")

    args = parser.parse_args()

    callbacks = []
    if not args.no_sign:
        callbacks = [gpg_sign_callback]
        print("Signing enabled (GPG) for all migrated revisions.")

    success = migrate(args.src_path, args.dst_path, signer_callbacks=callbacks)

    if success:
        print(f"✅ Successfully migrated '{args.src_path}' to '{args.dst_path}'")
        sys.exit(0)
    else:
        print("❌ Migration FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
