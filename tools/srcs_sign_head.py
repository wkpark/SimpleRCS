#!/usr/bin/env python3
# ruff: noqa: T201, ANN201
import argparse
import functools
import os
import sys

from simple_rcs.simple_rcs import SimpleRCS
from simple_rcs.simple_rcs_gpg import gpg_sign_callback


def main():
    parser = argparse.ArgumentParser(description="Add GPG signatures to the HEAD of a SimpleRCS file.")
    parser.add_argument("file_path", help="Path to the .srcs file")
    parser.add_argument("-s", "--signer", action="append", type=str,
        help="Specify signer ID. Currently only GPG is supported. Can be specified multiple times.")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    rcs = SimpleRCS(args.file_path)

    if rcs._version < 2:
        print(f"Error: File '{args.file_path}' is v1 format. Signing is only supported for v2.")
        print("Please migrate the file to v2 first using 'srcs_migrate.py'.")
        sys.exit(1)

    signers = []
    if args.signer:
        signers = args.signer
    else:
        env_signer = os.environ.get("SRCS_SIGNING_KEY")
        if env_signer:
            signers = [env_signer]

    if not signers:
        print("No signers specified (via -s or SRCS_SIGNING_KEY). Exiting.")
        sys.exit(0)

    callbacks = []
    for s in signers:
        # Use partial to bind the signer_id to the callback
        callbacks.append(functools.partial(gpg_sign_callback, signer_id=s))

    print(f"Attempting to sign HEAD of '{args.file_path}' with {len(signers)} GPG key(s).")

    success = rcs.sign_head(signer_callbacks=callbacks)

    if success:
        print(f"✅ Successfully signed HEAD of '{args.file_path}'.")
        sys.exit(0)
    else:
        print("❌ Failed to sign HEAD.")
        sys.exit(1)

if __name__ == "__main__":
    main()
