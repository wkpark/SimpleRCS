import argparse
import difflib
import io
import os
import sys
import time
from pathlib import Path


# ruff: noqa: T201, ANN201
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app.common.myersdiff import MyersSequenceMatcher
from src.app.common.pydifflib import StreamSequenceMatcher
from src.app.common.simple_rcs import SimpleRCS


def resolve_rcs_path(target_path: Path, explicit_rcs_path: str = None, srcs_dir_name: str = ".srcs") -> Path:
    """Resolves the path to the .srcs file."""
    if explicit_rcs_path:
        return Path(explicit_rcs_path)

    rcs_filename = target_path.name + ".srcs"
    current_dir_rcs = Path(rcs_filename)

    if current_dir_rcs.exists():
        return current_dir_rcs

    srcs_dir = Path(srcs_dir_name)
    return srcs_dir / rcs_filename

def print_matcher_unified_diff(matcher, lines_a: list[str], lines_b: list[str], fromfile: str, tofile: str, context: int = 3):
    """
    Generates and prints a unified diff using a given matcher with proper context.
    Works with any matcher providing get_opcodes().
    """
    fromdate = time.ctime(os.stat(fromfile).st_mtime) if os.path.exists(fromfile) else time.ctime()
    todate = time.ctime()

    print(f"--- {fromfile}\t{fromdate}")
    print(f"+++ {tofile}\t{todate}")

    opcodes = list(matcher.get_opcodes())

    # Check if there are any changes
    if all(tag == 'equal' for tag, _, _, _, _ in opcodes):
        return

    # Logic adapted from difflib.SequenceMatcher.get_grouped_opcodes
    grouped_opcodes = []
    group = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal' and i2 - i1 > 2 * context:
            # Large equal block: split it

            # End current group with context
            # Only apply if there is a current group (meaning preceding changes)
            if group:
                group.append((tag, i1, min(i2, i1 + context), j1, min(j2, j1 + context)))
                grouped_opcodes.append(group)
                group = []

            # Start next group with context
            i1 = max(i1, i2 - context)
            j1 = max(j1, j2 - context)

        group.append((tag, i1, i2, j1, j2))

    # Trim the trailing equal block of the last group if necessary
    if group:
        last_tag, li1, li2, lj1, lj2 = group[-1]
        if last_tag == 'equal':
            # Restrict to context lines
            new_li2 = min(li2, li1 + context)
            new_lj2 = min(lj2, lj1 + context)
            group[-1] = (last_tag, li1, new_li2, lj1, new_lj2)

    if group and not (len(group) == 1 and group[0][0] == 'equal'):
        grouped_opcodes.append(group)

    # Print hunks
    for group in grouped_opcodes:
        # Determine hunk range
        first, last = group[0], group[-1]
        i1, i2 = first[1], last[2]
        j1, j2 = first[3], last[4]

        # Adjust for printing (1-based, handle empty ranges)
        range_a = f"{i1+1},{i2-i1}" if i2-i1 != 1 else f"{i1+1}"
        range_b = f"{j1+1},{j2-j1}" if j2-j1 != 1 else f"{j1+1}"

        print(f"@@ -{range_a} +{range_b} @@")

        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                for line in lines_a[i1:i2]:
                    print(" " + line, end='')
            elif tag == 'delete':
                for line in lines_a[i1:i2]:
                    print("-" + line, end='')
            elif tag == 'insert':
                for line in lines_b[j1:j2]:
                    print("+" + line, end='')
            elif tag == 'replace':
                for line in lines_a[i1:i2]:
                    print("-" + line, end='')
                for line in lines_b[j1:j2]:
                    print("+" + line, end='')

            # Handle missing newlines at end of file if necessary (difflib usually adds \n\ No newline...)
            pass # print() usually adds \n, but we used end='' for lines with \n.

def main() -> None:
    parser = argparse.ArgumentParser(description="Show diffs between revisions or working file.")
    parser.add_argument("content_file", help="Path to the file to diff")
    parser.add_argument("rcs_file", nargs='?', help="Optional explicit path to the .srcs file")

    parser.add_argument("-r", "--revision", help="Revision(s) to compare. Format: '1.1' (vs HEAD) or '1.1:1.2'")
    parser.add_argument("--srcs-dir", default=".srcs", help="Directory to store .srcs files (default: .srcs)")
    parser.add_argument("--engine", default="difflib", choices=["difflib", "pydifflib", "myers"], help="Diff engine to use")

    args = parser.parse_args()

    target_path = Path(args.content_file)
    rcs_path = resolve_rcs_path(target_path, args.rcs_file, args.srcs_dir)

    if not rcs_path.exists():
        print(f"Error: RCS file '{rcs_path}' not found.")
        sys.exit(1)

    rcs = SimpleRCS(str(rcs_path))

    # ... (Revision logic same as before) ...
    ver_a_num = None
    ver_b_num = None

    if args.revision:
        if ':' in args.revision:
            ver_a_num, ver_b_num = args.revision.split(':', 1)
        else:
            ver_a_num = args.revision

    # 1. Fetch Content A (Base)
    try:
        if ver_a_num:
            content_a = rcs.checkout(ver_a_num)
            label_a = f"a/{target_path} (v{ver_a_num})"
        else:
            content_a = rcs.checkout() # HEAD
            head_ver = rcs.head_info['ver'] if rcs.head_info else "NEW"
            label_a = f"a/{target_path} (HEAD v{head_ver})"
    except ValueError as e:
        print(f"Error fetching version A: {e}")
        sys.exit(1)

    # 2. Fetch Content B (Target)
    try:
        if ver_b_num:
            content_b = rcs.checkout(ver_b_num)
            label_b = f"b/{target_path} (v{ver_b_num})"
        elif ver_a_num:
            if not target_path.exists():
                 print(f"Error: Working file '{target_path}' not found.")
                 sys.exit(1)
            content_b = target_path.read_bytes()
            label_b = f"b/{target_path} (Working Copy)"
        else:
            if not target_path.exists():
                 print(f"Error: Working file '{target_path}' not found.")
                 sys.exit(1)
            content_b = target_path.read_bytes()
            label_b = f"b/{target_path} (Working Copy)"
    except ValueError as e:
        print(f"Error fetching version B: {e}")
        sys.exit(1)

    # 3. Check for Binary
    if isinstance(content_a, bytes) or isinstance(content_b, bytes):
        def ensure_str(data):
            if isinstance(data, str):
                return data, False
            try:
                return data.decode('utf-8'), False
            except UnicodeDecodeError:
                return data, True

        content_a_str, is_bin_a = ensure_str(content_a)
        content_b_str, is_bin_b = ensure_str(content_b)

        if is_bin_a or is_bin_b:
            print(f"Binary files {label_a} and {label_b} differ")
            sys.exit(0 if content_a == content_b else 1)
        else:
            content_a = content_a_str
            content_b = content_b_str

    # 4. Generate Diff
    print(f"Diffing using engine: {args.engine}", file=sys.stderr)
    start_time = time.perf_counter()

    if args.engine == "difflib":
        diff_lines = difflib.unified_diff(
            content_a.splitlines(keepends=True),
            content_b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
        )
        sys.stdout.writelines(diff_lines)

    elif args.engine == "pydifflib":
        # pydifflib uses streams
        stream_a = io.BytesIO(content_a.encode('utf-8'))
        stream_b = io.BytesIO(content_b.encode('utf-8'))
        matcher = StreamSequenceMatcher(stream_a, stream_b, chunk_size=None)
        lines_a = content_a.splitlines(keepends=True)
        lines_b = content_b.splitlines(keepends=True)
        opcodes = list(matcher.get_opcodes())
        print_matcher_unified_diff(matcher, lines_a, lines_b, label_a, label_b)

    elif args.engine == "myers":
        # Myers works on lists of lines
        lines_a = content_a.splitlines(keepends=True)
        lines_b = content_b.splitlines(keepends=True)
        matcher = MyersSequenceMatcher(None, lines_a, lines_b)
        opcodes = list(matcher.get_opcodes())
        print_matcher_unified_diff(matcher, lines_a, lines_b, label_a, label_b)
    end_time = time.perf_counter()
    print(f"\nTime taken: {end_time - start_time:.4f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
