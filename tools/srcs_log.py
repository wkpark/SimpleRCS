#!/usr/bin/env python3
# ruff: noqa: T201, ANN201
import argparse
import os
import sys


# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime

from src.app.common.simple_rcs import SimpleRCS
from src.app.common.simple_rcs_gpg import get_gpg_uid, gpg_verify_callback


def print_log_entry(entry: dict, rcs: SimpleRCS, show_signature: bool = False) -> None:
    ver = entry.get('ver', 'unknown')
    print(f"commit {ver}")

    if entry.get('hash'):
        full_hash = entry['hash']
        short_hash = full_hash[:8] # Shorten to first 8 characters
        print(f"Hash: {short_hash} ({full_hash[:16]}...{full_hash[-16:]})")

    print(f"Author: {entry.get('author')}")
    print(f"Date:   {entry.get('date')}")

    signatures = entry.get('signatures')
    if signatures:
        print("Signatures:")
        for sig_entry in signatures:
            # sig format: signer_id|timestamp|sig_val
            parts = sig_entry.split('|')
            if len(parts) >= 3:
                signer_id = parts[0]
                ts_str = parts[1]

                # Format timestamp
                try:
                    dt_object = datetime.fromisoformat(ts_str)
                    formatted_ts = dt_object.astimezone().strftime('%a %d %b %Y %I:%M:%S %p %Z')
                except ValueError:
                    formatted_ts = ts_str

                # Get UID and Trust Level
                signer_uid, trust_level = get_gpg_uid(signer_id)

                print(f"gpg: Signature made {formatted_ts}")
                print(f"gpg:                using RSA key {signer_id}")

                if show_signature and rcs:
                    # Perform verification
                    is_valid, valid_signer_id = rcs.verify_block_signature(entry, gpg_verify_callback)
                    if is_valid and valid_signer_id == signer_id:
                        # Display actual trust level if available
                        trust_display = f" [{trust_level}]" if trust_level != "unknown" else ""
                        print(f"gpg: Good signature from \"{signer_uid}\"{trust_display}")
                    else:
                        print(f"gpg: BAD signature from \"{signer_uid}\"")
                else:
                    # Neutral display
                    print(f"  gpg: Signer: \"{signer_uid}\" (ID: {signer_id})") # Keep ID visible for neutral display
            else:
                print(f"  gpg: (Malformed signature entry: {sig_entry})")

    print("") # Empty line before message

    log_msg = entry.get('log', '')
    # Indent log message
    for line in log_msg.splitlines():
        print(f"    {line}")

    print("") # Separator

def main():
    parser = argparse.ArgumentParser(description="Show commit logs of a SimpleRCS file.")
    parser.add_argument("file_path", help="Path to the .srcs file")
    parser.add_argument("-n", "--limit", type=int, help="Limit number of commits to show")
    parser.add_argument("-r", "--reverse", action="store_true", help="Show oldest commits first")
    parser.add_argument("--show-signature", action="store_true", help="Validate GPG signatures")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    rcs = SimpleRCS(args.file_path)

    logs = rcs.log(limit=args.limit, reverse=args.reverse)

    if not logs:
        print("No history found.")
        return

    for entry in logs:
        print_log_entry(entry, rcs=rcs, show_signature=args.show_signature)

if __name__ == "__main__":
    main()
